"""Command-line interface for the quantum mechanics professor package."""
from __future__ import annotations

import argparse
import glob as _glob
import json
import sys


def _topic_cmd(args: argparse.Namespace) -> int:
    from .library import topic as get_topic, TOPICS
    from .serialize import topic_to_dict
    from .problems import problems_for_topic

    try:
        t = get_topic(args.id)
    except KeyError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        print(f"Available topics: {', '.join(sorted(TOPICS))}", file=sys.stderr)
        return 1

    d = topic_to_dict(t)
    print(f"\n{'='*60}")
    print(f"  {d['title']}  [{d['level']}]")
    print(f"{'='*60}")
    print(f"\nPrerequisites: {', '.join(d['prerequisites']) or 'none'}")
    print(f"\nKey Concepts:")
    for c in d["key_concepts"]:
        print(f"  • {c}")
    print(f"\nKey Equations:")
    for eq in d["key_equations"]:
        print(f"  {eq}")
    print(f"\nIntuition:\n  {d['intuition']}")
    print(f"\nRecommended Books: {', '.join(d['book_refs'])}")

    problems = problems_for_topic(args.id)
    if problems:
        print(f"\nPractice Problems ({len(problems)}):")
        for p in problems:
            print(f"  [{p.difficulty}] {p.stem[:80]}...")
    return 0


def _path_cmd(args: argparse.Namespace) -> int:
    from .library import learning_path, TOPICS

    try:
        path = learning_path(args.id)
    except KeyError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"\nLearning path to '{args.id}':")
    for i, t_id in enumerate(path, 1):
        t = TOPICS[t_id]
        print(f"  {i:2d}. [{t.level:20s}] {t.title}")
    return 0


def _books_cmd(args: argparse.Namespace) -> int:
    from .library import books_at_level, BOOKS
    from .serialize import book_to_dict

    if args.level:
        books = books_at_level(args.level)
    else:
        books = list(BOOKS.values())
        books.sort(key=lambda b: (b.level, b.id))

    for b in books:
        d = book_to_dict(b)
        print(f"\n[{d['level']}] {d['title']}")
        print(f"  Authors: {', '.join(d['authors'])}")
        print(f"  Why read: {d['why_read']}")
    return 0


def _solve_cmd(args: argparse.Namespace) -> int:
    from .physics import solve, SOLVERS

    topic_id = args.topic_id
    kwargs: dict = {}
    for kv in args.params or []:
        if "=" not in kv:
            print(f"Error: param must be key=value, got {kv!r}", file=sys.stderr)
            return 1
        k, v = kv.split("=", 1)
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
        print(f"Error: {exc}", file=sys.stderr)
        print(f"Available solvers: {', '.join(sorted(SOLVERS))}", file=sys.stderr)
        return 1

    print(json.dumps(result, indent=2))
    return 0


def _search_cmd(args: argparse.Namespace) -> int:
    from .ingest import ingest_files, search
    from .serialize import search_result_to_dict

    patterns = args.files or ["*.txt", "*.md"]
    paths: list = []
    for pat in patterns:
        paths.extend(_glob.glob(pat, recursive=True))

    if not paths:
        print("No files matched. Pass --files pattern to specify files to search.", file=sys.stderr)
        return 1

    corpus = ingest_files(paths)
    results = search(corpus, args.query, top_k=args.top_k)

    if not results:
        print("No results found.")
        return 0

    for r in results:
        print(f"\n[score={r.score:.4f}] {r.file}")
        print(f"  {r.excerpt[:160]}")
    return 0


def _serve_cmd(args: argparse.Namespace) -> int:
    from .web_server import main as web_main

    sys.argv = ["quantum_prof.web_server", "--host", args.host, "--port", str(args.port)]
    return web_main()


def _ask_cmd(args: argparse.Namespace) -> int:
    try:
        from .agent import QuantumProfessor
    except ImportError:
        print("Error: 'anthropic' package required. Install with: pip install anthropic", file=sys.stderr)
        return 1

    prof = QuantumProfessor()
    answer = prof.ask(" ".join(args.question))
    print(answer)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m quantum_prof.cli",
        description="Feynman-style quantum mechanics professor",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_topic = sub.add_parser("topic", help="Show details for a topic")
    p_topic.add_argument("id", help="Topic ID (e.g. particle-in-a-box)")

    p_path = sub.add_parser("path", help="Show learning path to a topic")
    p_path.add_argument("id", help="Target topic ID")

    p_books = sub.add_parser("books", help="List recommended books")
    p_books.add_argument("--level", choices=["basics", "intermediate", "advanced", "advanced_topics"])

    p_solve = sub.add_parser("solve", help="Run a physics calculation")
    p_solve.add_argument("topic_id", help="Topic/solver ID (e.g. particle-in-a-box)")
    p_solve.add_argument("--params", nargs="*", metavar="key=value", help="Solver parameters")

    p_search = sub.add_parser("search", help="TF-IDF search over local files")
    p_search.add_argument("query", help="Search query")
    p_search.add_argument("--files", nargs="*", metavar="glob", help="File glob patterns")
    p_search.add_argument("--top-k", type=int, default=5, dest="top_k")

    p_serve = sub.add_parser("serve", help="Start the web UI server")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=5052)

    p_ask = sub.add_parser("ask", help="Ask the AI professor a question")
    p_ask.add_argument("question", nargs="+", help="Your question")

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    dispatch = {
        "topic": _topic_cmd,
        "path": _path_cmd,
        "books": _books_cmd,
        "solve": _solve_cmd,
        "search": _search_cmd,
        "serve": _serve_cmd,
        "ask": _ask_cmd,
    }
    return dispatch[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
