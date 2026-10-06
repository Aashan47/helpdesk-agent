"""HEAD requests (uptime monitors probe with HEAD) — offline, no model calls."""

import os
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

os.environ.setdefault("HDA_RETRIEVER", "tfidf")
import serve  # noqa: E402


class Head(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv = ThreadingHTTPServer(("127.0.0.1", 0), serve.Handler)
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()
        cls.base = f"http://127.0.0.1:{cls.srv.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()

    def req(self, method, path):
        r = urllib.request.Request(self.base + path, method=method)
        try:
            with urllib.request.urlopen(r, timeout=10) as resp:
                return resp.status, dict(resp.headers), resp.read()
        except urllib.error.HTTPError as e:
            return e.code, dict(e.headers), e.read()

    def test_head_matches_get_without_a_body(self):
        for path in ("/health", "/", "/meta", "/inbox"):
            g = self.req("GET", path)
            h = self.req("HEAD", path)
            self.assertEqual((h[0], h[2]), (200, b""), path)
            self.assertEqual(h[1]["Content-Type"], g[1]["Content-Type"], path)
            self.assertEqual(h[1]["Content-Length"], str(len(g[2])), path)

    def test_head_on_unknown_path_is_404_not_501(self):
        self.assertEqual(self.req("HEAD", "/nope")[0], 404)

    def test_head_never_starts_an_agent_run(self):
        tid = serve._INBOX.list()[0]["id"]
        before = serve._INBOX.get(tid)["status"]
        status, headers, body = self.req("HEAD", f"/inbox/{tid}/stream")
        self.assertEqual((status, headers.get("Allow"), body), (405, "GET", b""))
        self.assertEqual(serve._INBOX.get(tid)["status"], before)


if __name__ == "__main__":
    unittest.main()
