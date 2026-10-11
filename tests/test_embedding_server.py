"""HTTP contract test for proxy/embedding_server.py with a stub encoder —
no model download. Run: python3 -m unittest tests.test_embedding_server -v
"""
import json
import threading
import unittest
import urllib.error
import urllib.request

from proxy.embedding_server import make_server


class FakeEncoder:
    def encode(self, texts):
        return [[1.0, 0.0] if "alpha" in t else [0.0, 1.0] for t in texts]


class EmbeddingServer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = make_server("127.0.0.1", 0, FakeEncoder(),
                                 "fake-model")
        cls.url = f"http://127.0.0.1:{cls.server.server_port}"
        cls.thread = threading.Thread(target=cls.server.serve_forever,
                                      daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def post(self, body):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        req = urllib.request.Request(
            self.url + "/v1/embeddings", data=data,
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req) as resp:
                return resp.status, json.loads(resp.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    def test_health(self):
        with urllib.request.urlopen(self.url + "/health") as resp:
            self.assertEqual(resp.status, 200)
            self.assertEqual(json.loads(resp.read()), {"status": "ok"})

    def test_string_input(self):
        status, body = self.post({"model": "anything", "input": "alpha"})
        self.assertEqual(status, 200)
        self.assertEqual(body["object"], "list")
        self.assertEqual(body["model"], "fake-model")
        self.assertEqual(body["data"],
                         [{"object": "embedding", "index": 0,
                           "embedding": [1.0, 0.0]}])
        self.assertIn("prompt_tokens", body["usage"])

    def test_list_input_indexes_in_order(self):
        status, body = self.post({"model": "x",
                                  "input": ["beta", "alpha", "gamma"]})
        self.assertEqual(status, 200)
        self.assertEqual([d["index"] for d in body["data"]], [0, 1, 2])
        self.assertEqual([d["embedding"] for d in body["data"]],
                         [[0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
        self.assertEqual(body["usage"]["total_tokens"], 3)

    def test_malformed_json_is_400(self):
        status, body = self.post(b"{not json")
        self.assertEqual(status, 400)
        self.assertIn("error", body)

    def test_bad_input_type_is_400(self):
        for bad in ({"input": 42}, {"input": ["ok", 7]},
                    {"input": []}, {}):
            status, body = self.post(bad)
            self.assertEqual(status, 400, bad)
            self.assertIn("error", body)

    def test_unknown_path_is_404(self):
        status, _ = self.post({"input": "x"})
        self.assertEqual(status, 200)  # sanity: right path works
        req = urllib.request.Request(self.url + "/v1/other", data=b"{}")
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            urllib.request.urlopen(req)
        self.assertEqual(ctx.exception.code, 404)


if __name__ == "__main__":
    unittest.main()
