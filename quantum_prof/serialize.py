"""Dict-serialization helpers for quantum_prof data types."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .ingest import SearchResult
    from .library import Book, Topic
    from .problems import Problem


def topic_to_dict(topic: "Topic") -> dict:
    return {
        "id": topic.id,
        "title": topic.title,
        "level": topic.level,
        "prerequisites": list(topic.prerequisites),
        "key_concepts": list(topic.key_concepts),
        "key_equations": list(topic.key_equations),
        "intuition": topic.intuition,
        "book_refs": list(topic.book_refs),
        "problem_ids": list(topic.problem_ids),
    }


def book_to_dict(book: "Book") -> dict:
    return {
        "id": book.id,
        "title": book.title,
        "authors": list(book.authors),
        "level": book.level,
        "topics": list(book.topics),
        "note": book.note,
        "why_read": book.why_read,
    }


def problem_to_dict(problem: "Problem") -> dict:
    return {
        "id": problem.id,
        "topic_id": problem.topic_id,
        "difficulty": problem.difficulty,
        "stem": problem.stem,
        "hint": problem.hint,
        "answer_latex": problem.answer_latex,
    }


def search_result_to_dict(result: "SearchResult") -> dict:
    return {
        "file": result.file,
        "score": result.score,
        "excerpt": result.excerpt,
    }
