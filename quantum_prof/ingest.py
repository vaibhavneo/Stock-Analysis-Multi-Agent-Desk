"""TF-IDF document search — stdlib only, no numpy/scipy."""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class SearchResult:
    file: str
    score: float
    excerpt: str


@dataclass
class Corpus:
    docs: List[Dict]  # [{path, tokens, text}]
    idf: Dict[str, float]


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _tf(tokens: List[str]) -> Dict[str, float]:
    counts: Dict[str, int] = {}
    for t in tokens:
        counts[t] = counts.get(t, 0) + 1
    total = max(len(tokens), 1)
    return {t: c / total for t, c in counts.items()}


def ingest_files(paths: List[str]) -> Corpus:
    """Build a TF-IDF corpus from a list of .txt / .md file paths."""
    docs = []
    for p in paths:
        try:
            text = Path(p).read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        tokens = _tokenize(text)
        docs.append({"path": p, "tokens": tokens, "tf": _tf(tokens), "text": text})

    n = len(docs)
    if n == 0:
        return Corpus(docs=[], idf={})

    df: Dict[str, int] = {}
    for doc in docs:
        for term in set(doc["tokens"]):
            df[term] = df.get(term, 0) + 1

    idf = {term: math.log((n + 1) / (count + 1)) + 1 for term, count in df.items()}
    return Corpus(docs=docs, idf=idf)


def _excerpt(text: str, query_tokens: List[str], max_chars: int = 200) -> str:
    lower = text.lower()
    best_pos = 0
    best_hits = -1
    for i in range(0, max(1, len(text) - max_chars), max(1, max_chars // 4)):
        window = lower[i : i + max_chars]
        hits = sum(1 for t in query_tokens if t in window)
        if hits > best_hits:
            best_hits, best_pos = hits, i
    snippet = text[best_pos : best_pos + max_chars].strip()
    return snippet.replace("\n", " ")


def search(corpus: Corpus, query: str, top_k: int = 5) -> List[SearchResult]:
    """Return top_k documents ranked by TF-IDF cosine similarity."""
    if not corpus.docs:
        return []
    q_tokens = _tokenize(query)
    if not q_tokens:
        return []

    q_tf = _tf(q_tokens)
    q_vec = {t: q_tf[t] * corpus.idf.get(t, 0.0) for t in q_tf}
    q_norm = math.sqrt(sum(v ** 2 for v in q_vec.values())) or 1.0

    results = []
    for doc in corpus.docs:
        doc_vec = {t: doc["tf"].get(t, 0.0) * corpus.idf.get(t, 0.0) for t in q_vec}
        dot = sum(q_vec[t] * doc_vec.get(t, 0.0) for t in q_vec)
        doc_norm = math.sqrt(sum(v ** 2 for v in doc_vec.values())) or 1.0
        score = dot / (q_norm * doc_norm)
        if score > 0:
            excerpt = _excerpt(doc["text"], q_tokens)
            results.append(SearchResult(file=doc["path"], score=round(score, 6), excerpt=excerpt))

    results.sort(key=lambda r: r.score, reverse=True)
    return results[:top_k]
