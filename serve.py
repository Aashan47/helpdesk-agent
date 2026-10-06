"""The support console: a zero-dependency web app around the agent.

    GEMINI_API_KEY=... python serve.py           # then open http://localhost:8000

Pure Python standard library, no web framework. The UI (helpdesk_agent/ui/index.html) is a
ticket inbox: the agent handles tickets while each step streams live, and every ticket ends
as a cited reply draft, an escalation with a hand-off note, or an explicit model-unavailable.
Escalated tickets form the human queue; the help centre the agent replies from can be edited
in place and is re-indexed on every change.

Endpoints
    GET  /                      the console
    GET  /meta                  company, model, threshold (and where it came from), calibration
    GET  /articles              the help centre (every article the agent can reply from)
    POST /articles              {"title","category","text"[,"category_name"]} -> add an article
    PUT  /articles/<id>         {"title"|"category"|"text"} -> edit (POST /articles/<id> also works)
    DELETE /articles/<id>       remove an article (POST /articles/<id>/delete also works)
    POST /articles/reset        back to the shipped help centre
    GET  /articles/export       the current help centre as a JSON download
    GET  /inbox                 the ticket queue (?status=escalated to filter)
    GET  /inbox/<id>            one ticket with its full result (trace, hand-off note)
    POST /inbox                 {"message","customer","subject"} -> new ticket
    GET  /inbox/<id>/stream     handle a ticket; server-sent events, one per step, then the result
    POST /inbox/<id>/send       mark an auto-resolved reply as sent
    POST /inbox/<id>/escalate   send a ticket to the human queue regardless of the agent
    POST /inbox/<id>/close      a human answered it: out of the queue
    POST /inbox/<id>/reopen     back to new (re-run, e.g. after fixing the help centre)
    GET  /queue                 the human queue: escalated tickets with their hand-off notes
    GET  /session               counts by status for this session
    POST /ticket                {"message"} -> one-shot result (non-streaming API)
    POST /compare               {"message"} -> what a generic chatbot (same model, no help centre) replies
    GET  /health

Guardrails (see helpdesk_agent/guards.py and kb.py): messages and articles are normalised and
size-capped; each client IP is rate-limited; at most HDA_CONCURRENCY agent runs are in flight;
a reply is never sent without a citation, or when its key fact is not in the cited article; a
ticket cannot be handled twice at once; a model outage is reported as such, never as an
escalation; every run works on a snapshot of the help centre, so an edit mid-run is safe.
"""

from __future__ import annotations

import json
import os
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

from helpdesk_agent import agent as agent_mod
from helpdesk_agent import corpus, guards, kb, llm, trace
from helpdesk_agent.inbox import Inbox, TransitionError

HERE = os.path.dirname(os.path.abspath(__file__))
UI = os.path.join(HERE, "helpdesk_agent", "ui", "index.html")
RESULTS = os.path.join(HERE, "results", "helpdesk", "results.json")
KB_DIR = os.path.join(HERE, "knowledge_base")
REPO = "https://github.com/Aashan47/helpdesk-agent"
MAX_BODY = 64 * 1024

print("Loading help centre + agent...")
_C = corpus.load_corpus()
_CUSTOM = _C["is_custom"]
_cats: dict[str, str] = {}
for _a in _C["articles"]:
    _cats.setdefault(_a["category"], _a["category_name"])
_WRITE = bool(os.environ.get("HDA_KB_WRITE")) and not _CUSTOM
_KB = kb.KnowledgeBase([{**a, "text": _C["contexts"][i]} for i, a in enumerate(_C["articles"])],
                       _C["name"], _C["company"], _cats, agent_factory=agent_mod.Agent,
                       write_dir=KB_DIR if _WRITE else None)
_TAU = corpus.abstention_threshold(_CUSTOM)
_INBOX = Inbox(seed_path=None if _CUSTOM else Inbox.__init__.__defaults__[0])
_LIMIT = guards.RateLimiter(rate_per_min=float(os.environ.get("HDA_RATE_PER_MIN", 30)), burst=15)
_GATE = guards.Gate(int(os.environ.get("HDA_CONCURRENCY", 3)))
_s0 = _KB.snapshot()
print(f"Ready: {len(_s0.articles)} articles from {_C['name']}; escalation threshold = {_TAU:.2f}; "
      f"retrieval = {_s0.agent.retrieval_mode}; samples = {_s0.agent.n}"
      f"{'; help-centre edits written to knowledge_base/' if _WRITE else ''}")


# ---- meta -------------------------------------------------------------------------------
def _threshold_source() -> str:
    if os.environ.get("HDA_THRESHOLD"):
        return "set by HDA_THRESHOLD (deployment override)"
    if _CUSTOM:
        return "default for an uncalibrated corpus (at least 3 of 5 samples agree)"
    return "calibrated on 68 labelled Northwind tickets (split-conformal selective prediction)"


def _calibration() -> dict | None:
    if _CUSTOM or not os.path.exists(RESULTS):
        return None
    try:
        with open(RESULTS) as f:
            r = json.load(f)
        c = r["calibrated_abstention"]
        return {"alpha": r["alpha_target_error"], "ece": r["ece"],
                "selective_error": r["test_selective_error_at_threshold"],
                "guarantee_held": r["guarantee_held"], "dataset": r["dataset"],
                "n": r["n_questions"], "n_test": r.get("n_test"),
                "n_unanswerable": r["n_unanswerable"],
                "calibrated_threshold": r["calibrated_threshold"],
                "coverage": c["coverage"], "selective_accuracy": c["selective_accuracy"],
                "hallucination": c["hallucination_rate_unanswerable"],
                "task_accuracy": c["task_accuracy"],
                "trust_hallucination": r["uncalibrated_trust_model"]["hallucination_rate_unanswerable"]}
    except Exception:  # noqa: BLE001
        return None


def _kb_info(snap: kb.Snapshot) -> dict:
    return {"version": snap.version, "edits": snap.edits, "n": len(snap.articles),
            "write_through": _WRITE, "max_articles": _KB.max_articles,
            "limits": {"title": kb.TITLE_MAX, "body": kb.BODY_MAX, "body_min": kb.BODY_MIN}}


def _meta() -> dict:
    snap = _KB.snapshot()
    return {"name": "helpdesk-agent", "repo": REPO, "company": _KB.company,
            "kb_name": _KB.name, "n_articles": len(snap.articles), "categories": _KB.categories(),
            "is_custom": _CUSTOM, "kb": _kb_info(snap),
            "threshold": round(_TAU, 3), "threshold_source": _threshold_source(),
            "model": os.environ.get("HDA_MODEL", "gemini-2.5-flash"),
            "model_key": bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GEMENI_API_KEY")),
            "retriever": snap.agent.retrieval_mode, "n_samples": snap.agent.n,
            "max_steps": snap.agent.max_steps, "calibration": _calibration(),
            "commit": os.environ.get("RENDER_GIT_COMMIT", "")[:7] or None,
            "limits": {"max_message_chars": guards.MAX_MESSAGE_CHARS}}


def _articles_payload() -> dict:
    snap = _KB.snapshot()
    return {"n": len(snap.articles), "name": _KB.name, "categories": _KB.categories(),
            "kb": _kb_info(snap), "articles": [dict(a) for a in snap.articles]}


# ---- ticket handling ------------------------------------------------------------------
_TITLES = {"ok": "Resolved from the help centre",
           "not_in_documents": "Not covered by the help centre",
           "no_supported_answer": "No article supports a reply",
           "low_confidence": "Confidence below the calibrated threshold",
           "no_citation": "Reply had no source article",
           "not_grounded": "Reply not grounded in the cited article",
           "model_unavailable": "Model unavailable"}


def _explain(snap: kb.Snapshot, pred: dict, d: dict) -> tuple[str, str, str]:
    """(status, reason_code, reason). Resolved, escalated for a stated reason, or model
    unavailable — never blurred. Two output guards run after the calibrated decision: a reply
    without a citation, or whose key fact is not in the cited article, is escalated."""
    n, votes = snap.agent.n, pred.get("votes", 0)
    searches = sum(1 for s in pred.get("steps", []) if s.get("action") == "search")
    if pred.get("error"):
        return "error", "model_unavailable", pred["error"]
    if d["answered"]:
        cite = d.get("citation")
        if not cite:
            return ("escalated", "no_citation",
                    "the samples agreed on an answer but did not point to an article, so it "
                    "cannot be sent.")
        if not guards.grounded(d.get("answer", ""), cite["text"]):
            return ("escalated", "not_grounded",
                    "the answer's key fact does not appear in the article it cites, so it "
                    "cannot be sent.")
        return ("resolved", "ok",
                f"{votes} of {n} independent samples found the answer in the articles and "
                f"{pred['confidence']:.2f} of them agreed on it, above the {_TAU:.2f} threshold.")
    if any(s.get("action") == "abstain" for s in pred.get("steps", [])):
        return ("escalated", "not_in_documents",
                f"after {searches} search{'es' if searches != 1 else ''} the agent judged the "
                f"help centre does not cover this, and stopped before drafting a reply.")
    if not pred.get("retrieved"):
        return ("escalated", "not_in_documents",
                "no article in the help centre matched the ticket at all.")
    if not pred.get("answerable"):
        return ("escalated", "no_supported_answer",
                f"articles were retrieved, but only {votes} of {n} samples found a reply "
                f"supported by them (a majority is required).")
    return ("escalated", "low_confidence",
            f"{votes} of {n} samples drafted a reply, but only {pred['confidence']:.2f} agreed on "
            f"the same answer, below the calibrated threshold of {_TAU:.2f}.")


def _handoff(snap: kb.Snapshot, question: str, pred: dict, code: str, reason: str) -> dict:
    """Deterministic hand-off note for the human — from the trace, no model call, so it is
    always available and never invents."""
    searches = [s["query"] for s in pred.get("steps", []) if s.get("action") == "search"]
    closest = [{"id": snap.article(i)["id"], "title": snap.article(i)["title"],
                "category": snap.article(i)["category_name"]}
               for i in pred.get("retrieved", [])[:3]]
    if code == "not_in_documents":
        action = ("Answer manually. If this comes up again, add an article: the agent will then "
                  "resolve it automatically.")
    elif code == "no_supported_answer":
        action = ("Check the closest articles; the answer may be implied but not stated. If so, "
                  "make the article explicit.")
    elif code in ("no_citation", "not_grounded"):
        action = ("The agent drafted a reply it could not tie to an article. Treat the draft as "
                  "unverified; answer from the articles yourself.")
    else:
        action = ("The articles partly cover this but the samples disagreed on the exact answer. "
                  "Read the closest articles and reply; consider clarifying the article.")
    return {"customer_message": question, "searched": searches, "closest": closest,
            "reason_title": _TITLES.get(code, code), "reason": reason, "suggested_action": action,
            "holding_reply": ("Thanks for getting in touch. I've passed this to a teammate who can "
                              "answer it properly, and you'll hear back from us shortly.")}


def _handle(question: str, on_event=None) -> dict:
    """Run one ticket against a snapshot of the help centre. Corpus positions are translated
    to stable article ids in everything that leaves this function (trace steps, retrieved
    list, citation), so the console and the hand-off note survive later edits."""
    t0 = time.time()
    snap = _KB.snapshot()

    def emit(step: dict) -> None:
        if "ids" in step:
            step["ids"] = snap.ids(step["ids"])     # same dict the agent keeps in `steps`
        if on_event:
            on_event(step)

    pred = snap.agent.predict(question, on_event=emit)
    d = agent_mod.decide(pred, _TAU)
    status, code, reason = _explain(snap, pred, d)
    sent_ok = status == "resolved"
    latency = round((time.time() - t0) * 1000)
    trace.log({"q": question[:200], "status": status, "code": code,
               "confidence": round(d["confidence"], 2), "llm_calls": pred.get("llm_calls", 0),
               "searches": sum(1 for s in pred.get("steps", []) if s.get("action") == "search"),
               "latency_ms": latency, "kb_version": snap.version})
    cite = d.get("citation") if sent_ok else None
    out = {"status": status, "question": question,
           "reply": (d.get("reply") or d.get("answer", "")) if sent_ok else "",
           "answer": d.get("answer", "") if sent_ok else "",
           "confidence": round(d["confidence"], 3), "threshold": round(_TAU, 3),
           "votes": pred.get("votes", 0),
           "votes_total": snap.agent.n if pred.get("retrieved") else 0,
           "reason_code": code, "reason_title": _TITLES.get(code, code), "reason": reason,
           "citation": None, "retrieved": snap.ids(pred.get("retrieved", [])),
           "steps": list(pred.get("steps", [])),
           "llm_calls": pred.get("llm_calls", 0), "latency_ms": latency,
           "kb_version": snap.version, "handoff": None}
    if cite:
        a = snap.article(cite["corpus_id"])
        out["citation"] = {"id": a["id"], "title": a["title"], "category": a["category_name"],
                           "text": a["text"]}
    if status == "escalated":
        out["handoff"] = _handoff(snap, question, pred, code, reason)
    return out


def _compare(question: str) -> dict:
    """The thing people compare against: the same model, told to be a helpful support bot for
    this company, with no help centre. Shown side by side so the difference is concrete."""
    t0 = time.time()
    try:
        out = llm.complete(
            f"You are the customer-support assistant for {_KB.company or 'our company'}, a "
            "team-collaboration SaaS product. Reply to the customer's message helpfully in one or "
            f"two sentences.\n\nCUSTOMER: {question}\nREPLY:", max_tokens=160)
        reply = out.strip()
        if reply.upper().startswith("REPLY:"):
            reply = reply[6:].strip()
        return {"reply": reply, "latency_ms": round((time.time() - t0) * 1000)}
    except llm.LLMError as exc:
        return {"error": str(exc), "latency_ms": round((time.time() - t0) * 1000)}


def _queue() -> dict:
    """The human queue: escalated tickets, newest first, each with its hand-off note."""
    return {"tickets": _INBOX.list("escalated"), "closed": _INBOX.counts()["closed"],
            "session": _INBOX.counts()}


# ---- http -----------------------------------------------------------------------------
class HttpError(Exception):
    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code, self.message = code, message


class Handler(BaseHTTPRequestHandler):
    server_version = "helpdesk-agent"
    sys_version = ""

    # -- plumbing ----------------------------------------------------------------------
    def _headers(self, code: int, ctype: str, length: int | None = None,
                 extra: dict | None = None) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        if length is not None:
            self.send_header("Content-Length", str(length))
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy",
                         "default-src 'self'; script-src 'self' 'unsafe-inline'; "
                         "style-src 'self' 'unsafe-inline'; connect-src 'self'; img-src 'self' data:; "
                         "frame-ancestors 'none'")
        self.end_headers()

    def _send(self, code: int, body, ctype: str = "application/json",
              extra: dict | None = None) -> None:
        data = (body if isinstance(body, (bytes, str)) else json.dumps(body))
        data = data.encode() if isinstance(data, str) else data
        self._headers(code, ctype, len(data), extra)
        if self.command != "HEAD":            # HEAD: same status and headers, no body
            self.wfile.write(data)

    def _client(self) -> str:
        fwd = self.headers.get("X-Forwarded-For", "")
        return (fwd.split(",")[0].strip() if fwd else self.client_address[0]) or "?"

    def _body(self) -> dict:
        try:
            n = int(self.headers.get("Content-Length", 0) or 0)
        except ValueError:
            raise HttpError(400, "bad Content-Length") from None
        if n > MAX_BODY:
            raise HttpError(413, "request body too large")
        raw = self.rfile.read(n) if n else b"{}"
        try:
            obj = json.loads(raw or b"{}")
        except (ValueError, UnicodeDecodeError):
            raise HttpError(400, "body must be JSON") from None
        if not isinstance(obj, dict):
            raise HttpError(400, "body must be a JSON object")
        return obj

    @staticmethod
    def _id(part: str, what: str) -> int:
        if not part.isdigit() or len(part) > 9:
            raise HttpError(404, f"no such {what}")
        return int(part)

    def _limited(self) -> None:
        if not _LIMIT.allow(self._client()):
            raise HttpError(429, "too many requests from this client; try again in a minute")

    def log_message(self, *a):  # quiet
        pass

    def _route(self, method: str) -> None:
        try:
            url = urlsplit(self.path)
            parts = url.path.strip("/").split("/")
            query = parse_qs(url.query)
            handler = {"GET": self._get, "POST": self._post, "PUT": self._put,
                       "DELETE": self._delete}[method]
            handler(url.path, parts, query)
        except HttpError as e:
            self._send(e.code, {"error": e.message})
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as exc:  # noqa: BLE001
            self._send(500, {"error": f"internal error: {exc.__class__.__name__}"})

    def do_GET(self):  # noqa: N802
        self._route("GET")

    def do_HEAD(self):  # noqa: N802
        """Uptime monitors (UptimeRobot's free HTTP check) probe with HEAD. Answer it like
        GET without the body, except the agent stream: a HEAD must never start an agent
        run or claim a ticket, so that path is refused."""
        path = urlsplit(self.path).path.strip("/").split("/")
        if path[0] == "inbox" and len(path) == 3 and path[2] == "stream":
            self._send(405, {"error": "use GET to stream a ticket"}, extra={"Allow": "GET"})
            return
        self._route("GET")

    def do_POST(self):  # noqa: N802
        self._route("POST")

    def do_PUT(self):  # noqa: N802
        self._route("PUT")

    def do_DELETE(self):  # noqa: N802
        self._route("DELETE")

    # -- streaming --------------------------------------------------------------------
    def _stream(self, tid: int) -> None:
        """Server-sent events: one event per agent step as it happens, then the result."""
        self._limited()
        try:
            t = _INBOX.get(tid)
        except KeyError:
            raise HttpError(404, "no such ticket") from None
        try:
            _INBOX.set_status(tid, "handling")
        except TransitionError as exc:
            raise HttpError(409, str(exc)) from None
        if not _GATE.acquire():
            _INBOX.set_status(tid, "new")
            raise HttpError(429, "the agent is busy with other tickets; try again shortly")

        self._headers(200, "text/event-stream; charset=utf-8")

        def send(kind: str, payload: dict) -> None:
            try:
                self.wfile.write(f"event: {kind}\ndata: {json.dumps(payload)}\n\n".encode())
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass            # client left; keep going so the ticket still gets its result

        try:
            send("ticket", Inbox._public(_INBOX.get(tid)))
            try:
                result = _handle(t["message"], on_event=lambda step: send("step", step))
            except Exception as exc:  # noqa: BLE001
                _INBOX.set_status(tid, "error")
                send("done", {"error": f"internal error: {exc.__class__.__name__}"})
                return
            status = {"resolved": "auto-resolved", "escalated": "escalated"}.get(result["status"], "error")
            pub = _INBOX.set_status(tid, status, result=result)
            send("done", {"result": result, "ticket": pub, "session": _INBOX.counts()})
        finally:
            _GATE.release()

    # -- GET --------------------------------------------------------------------------
    def _get(self, path: str, parts: list[str], query: dict) -> None:
        if parts[0] == "inbox" and len(parts) == 3 and parts[2] == "stream":
            self._stream(self._id(parts[1], "ticket"))
        elif parts[0] == "inbox" and len(parts) == 2:
            try:
                t = _INBOX.get(self._id(parts[1], "ticket"))
            except KeyError:
                raise HttpError(404, "no such ticket") from None
            self._send(200, {"ticket": Inbox._public(t), "result": t.get("result")})
        elif path in ("/", "/index.html"):
            with open(UI, encoding="utf-8") as f:
                self._send(200, f.read(), "text/html; charset=utf-8")
        elif path == "/meta":
            self._send(200, _meta())
        elif path == "/articles":
            self._send(200, _articles_payload())
        elif path == "/articles/export":
            name = kb.slugify(_KB.name) + ".json"
            self._send(200, json.dumps(_KB.export(), indent=2),
                       extra={"Content-Disposition": f'attachment; filename="{name}"'})
        elif path == "/inbox":
            status = (query.get("status") or [None])[0]
            self._send(200, {"tickets": _INBOX.list(status), "session": _INBOX.counts()})
        elif path == "/queue":
            self._send(200, _queue())
        elif path == "/session":
            self._send(200, _INBOX.counts())
        elif path == "/health":
            self._send(200, {"ok": True, "articles": len(_KB.snapshot().articles),
                             "commit": os.environ.get("RENDER_GIT_COMMIT", "")[:7] or None})
        else:
            raise HttpError(404, "not found")

    # -- articles ---------------------------------------------------------------------
    def _article_result(self, article: dict | None, **extra) -> None:
        self._send(200, {"article": article, **_articles_payload(), **extra})

    def _put(self, path: str, parts: list[str], query: dict) -> None:
        if parts[0] == "articles" and len(parts) == 2:
            self._edit_article(self._id(parts[1], "article"))
        else:
            raise HttpError(404, "not found")

    def _delete(self, path: str, parts: list[str], query: dict) -> None:
        if parts[0] == "articles" and len(parts) == 2:
            self._delete_article(self._id(parts[1], "article"))
        else:
            raise HttpError(404, "not found")

    def _edit_article(self, aid: int) -> None:
        self._limited()
        b = self._body()
        try:
            a = _KB.update(aid, title=b.get("title"), category=b.get("category"),
                           text=b.get("text"), category_name=b.get("category_name"))
        except KeyError:
            raise HttpError(404, "no such article") from None
        except kb.KBError as exc:
            raise HttpError(400, str(exc)) from None
        self._article_result(a)

    def _delete_article(self, aid: int) -> None:
        self._limited()
        try:
            a = _KB.delete(aid)
        except KeyError:
            raise HttpError(404, "no such article") from None
        except kb.KBError as exc:
            raise HttpError(409, str(exc)) from None
        self._article_result(a)

    # -- POST -------------------------------------------------------------------------
    def _post(self, path: str, parts: list[str], query: dict) -> None:
        if parts[0] == "inbox" and len(parts) == 3:
            tid, act = self._id(parts[1], "ticket"), parts[2]
            target = {"send": "sent", "escalate": "escalated", "reopen": "new",
                      "close": "closed"}.get(act)
            if not target:
                raise HttpError(404, "not found")
            try:
                t = _INBOX.set_status(tid, target)
            except KeyError:
                raise HttpError(404, "no such ticket") from None
            except TransitionError as exc:
                raise HttpError(409, str(exc)) from None
            self._send(200, {"ticket": t, "session": _INBOX.counts()})
            return
        if parts[0] == "articles":
            if len(parts) == 1:
                self._limited()
                b = self._body()
                if len(_KB.list()) >= _KB.max_articles:
                    raise HttpError(409, f"the help centre is full ({_KB.max_articles} articles)")
                try:
                    a = _KB.add(b.get("title", ""), b.get("category", ""), b.get("text", ""),
                                category_name=b.get("category_name"))
                except kb.KBError as exc:
                    raise HttpError(400, str(exc)) from None
                self._article_result(a)
            elif parts[1] == "reset" and len(parts) == 2:
                self._limited()
                self._article_result(None, discarded=_KB.reset())
            elif len(parts) == 2:
                self._edit_article(self._id(parts[1], "article"))
            elif len(parts) == 3 and parts[2] == "delete":
                self._delete_article(self._id(parts[1], "article"))
            else:
                raise HttpError(404, "not found")
            return
        b = self._body()
        try:
            q = guards.clean_message(b.get("message", b.get("question", "")))
        except guards.MessageError as exc:
            raise HttpError(400, str(exc)) from None
        if path == "/inbox":
            self._limited()
            try:
                t = _INBOX.add(q, customer=str(b.get("customer", ""))[:60],
                               subject=str(b.get("subject", ""))[:120])
            except TransitionError as exc:
                raise HttpError(409, str(exc)) from None
            self._send(200, {"ticket": t, "session": _INBOX.counts()})
        elif path in ("/ticket", "/ask", "/compare"):
            self._limited()
            if not _GATE.acquire():
                raise HttpError(429, "the agent is busy; try again shortly")
            try:
                self._send(200, _compare(q) if path == "/compare" else _handle(q))
            finally:
                _GATE.release()
        else:
            raise HttpError(404, "not found")


def main():
    port = int(os.environ.get("PORT", "8000"))
    print(f"Serving on http://localhost:{port}  (Ctrl+C to stop)")
    srv = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    srv.daemon_threads = True
    srv.serve_forever()


if __name__ == "__main__":
    main()
