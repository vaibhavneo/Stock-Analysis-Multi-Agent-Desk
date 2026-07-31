"""HTTP server for the Quantum Professor web UI."""
from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .library import TOPICS, BOOKS, learning_path, topics_at_level, books_at_level
from .physics import solve, SOLVERS
from .problems import problems_for_topic
from .serialize import topic_to_dict, book_to_dict, problem_to_dict

WEB_ROOT = Path(__file__).parent / "web"

CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".json": "application/json; charset=utf-8",
}


class QuantumHandler(BaseHTTPRequestHandler):
    server_version = "QuantumProfUI/0.1"

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self._send_cors_headers()
        self.end_headers()

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        query = parsed.query

        if path == "/api/topics":
            self._handle_topics(query)
        elif path == "/api/topic":
            self._handle_topic(query)
        elif path == "/api/books":
            self._handle_books(query)
        elif path == "/api/solve":
            self._handle_solve(query)
        elif path == "/api/path":
            self._handle_path(query)
        elif path == "/api/problems":
            self._handle_problems(query)
        else:
            self._serve_static(path)

    def log_message(self, fmt: str, *args: object) -> None:
        return

    def _handle_topics(self, query: str) -> None:
        params = parse_qs(query)
        level = params.get("level", [None])[0]
        if level:
            ts = topics_at_level(level)
        else:
            ts = list(TOPICS.values())
        self._send_json({"topics": [topic_to_dict(t) for t in ts]})

    def _handle_topic(self, query: str) -> None:
        params = parse_qs(query)
        topic_id = params.get("id", [""])[0].strip()
        if not topic_id or topic_id not in TOPICS:
            self._send_json({"error": f"Unknown topic: {topic_id!r}"}, status=404)
            return
        t = TOPICS[topic_id]
        d = topic_to_dict(t)
        d["problems"] = [problem_to_dict(p) for p in problems_for_topic(topic_id)]
        self._send_json(d)

    def _handle_books(self, query: str) -> None:
        params = parse_qs(query)
        level = params.get("level", [None])[0]
        if level:
            bs = books_at_level(level)
        else:
            bs = list(BOOKS.values())
        self._send_json({"books": [book_to_dict(b) for b in bs]})

    def _handle_solve(self, query: str) -> None:
        params = parse_qs(query)
        topic_id = params.get("topic", [""])[0].strip()
        if not topic_id:
            self._send_json({"error": "topic parameter required"}, status=400)
            return
        kwargs: dict = {}
        for k, vs in params.items():
            if k == "topic":
                continue
            v = vs[0]
            try:
                kwargs[k] = int(v)
            except ValueError:
                try:
                    kwargs[k] = float(v)
                except ValueError:
                    kwargs[k] = v
        try:
            result = solve(topic_id, **kwargs)
        except (ValueError, KeyError) as exc:
            self._send_json({"error": str(exc), "available_solvers": sorted(SOLVERS)}, status=400)
            return
        self._send_json(result)

    def _handle_path(self, query: str) -> None:
        params = parse_qs(query)
        topic_id = params.get("id", [""])[0].strip()
        if not topic_id or topic_id not in TOPICS:
            self._send_json({"error": f"Unknown topic: {topic_id!r}"}, status=404)
            return
        path_ids = learning_path(topic_id)
        self._send_json({
            "target": topic_id,
            "path": [topic_to_dict(TOPICS[t]) for t in path_ids],
        })

    def _handle_problems(self, query: str) -> None:
        params = parse_qs(query)
        topic_id = params.get("topic", [""])[0].strip()
        problems = problems_for_topic(topic_id) if topic_id else []
        self._send_json({"problems": [problem_to_dict(p) for p in problems]})

    def _serve_static(self, path: str) -> None:
        if path == "/":
            target = WEB_ROOT / "index.html"
        else:
            target = (WEB_ROOT / path.lstrip("/")).resolve()

        web_root_resolved = WEB_ROOT.resolve()
        if not str(target).startswith(str(web_root_resolved)) or not target.exists():
            self.send_error(404)
            return

        body = target.read_bytes()
        content_type = CONTENT_TYPES.get(target.suffix, "application/octet-stream")
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self._send_cors_headers()
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self._send_cors_headers()
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_cors_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve the Quantum Professor web UI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5052)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    server = ThreadingHTTPServer((args.host, args.port), QuantumHandler)
    print(f"Quantum Professor UI at http://{args.host}:{args.port}")
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
