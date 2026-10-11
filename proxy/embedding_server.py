#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""proxy/embedding_server.py — OpenAI-compatible /v1/embeddings on CPU.

Wraps a sentence-transformers model behind the exact endpoint shape the
score proxy's Router POSTs to (see docs/score-proxy.md). The requested
model name is ignored: whatever was loaded at startup is served. Only
dependency is sentence-transformers; everything else is stdlib.
"""
import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class Encoder:
    """Narrows SentenceTransformer to the handler's one-call interface:
    .encode(list[str]) -> list[list[float]], L2-normalized."""

    def __init__(self, model_name):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name, device="cpu")

    def encode(self, texts):
        vecs = self.model.encode(texts, normalize_embeddings=True)
        return [v.tolist() for v in vecs]


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self):
        if self.path == "/health":
            self._send_json(200, {"status": "ok"})
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/v1/embeddings":
            return self._send_json(404, {"error": "not found"})
        length = int(self.headers.get("Content-Length") or 0)
        try:
            req = json.loads(self.rfile.read(length))
            inp = req["input"]
            texts = [inp] if isinstance(inp, str) else list(inp)
            if not texts or not all(isinstance(t, str) for t in texts):
                raise ValueError("input must be a non-empty string or "
                                 "list of strings")
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
            return self._send_json(400, {"error": str(e)})
        try:
            vecs = self.server.encoder.encode(texts)
        except Exception as e:
            return self._send_json(500, {"error": f"encode failed: {e}"})
        tokens = sum(len(t.split()) for t in texts)  # rough, word-based
        self._send_json(200, {
            "object": "list",
            "data": [{"object": "embedding", "index": i, "embedding": v}
                     for i, v in enumerate(vecs)],
            "model": self.server.model_name,
            "usage": {"prompt_tokens": tokens, "total_tokens": tokens},
        })

    def _send_json(self, status, obj):
        data = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def make_server(host, port, encoder, model_name):
    srv = ThreadingHTTPServer((host, port), Handler)
    srv.encoder = encoder
    srv.model_name = model_name
    return srv


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="google/embeddinggemma-2")
    ap.add_argument("--port", type=int, default=8890)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()
    encoder = Encoder(args.model)  # raises loudly if the model can't load
    srv = make_server(args.host, args.port, encoder, args.model)
    print(f"embedding server on http://{args.host}:{srv.server_port} "
          f"(model {args.model})")
    srv.serve_forever()


if __name__ == "__main__":
    main()
