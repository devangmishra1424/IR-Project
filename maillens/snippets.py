"""Result snippets and query-term highlighting for the search UI.

E-mail bodies are untrusted text (some contain HTML), so everything returned here is
HTML-escaped first and only our own ``<mark>`` tags are added afterwards.
"""

from __future__ import annotations

import html
import re
from collections.abc import Collection

from maillens.text.analyzer import Analyzer

_WORD = re.compile(r"[A-Za-z0-9]+")


def query_terms(query: str, analyzer: Analyzer) -> set[str]:
    """The analysed (stemmed, stop-word-free) terms of a query."""
    return set(analyzer(query))


def _is_hit(word: str, terms: Collection[str], analyzer: Analyzer) -> bool:
    return analyzer.stem(word.lower()) in terms


def highlight(text: str, terms: Collection[str], analyzer: Analyzer) -> str:
    """HTML-escaped ``text`` with every word whose stem is in ``terms`` wrapped in <mark>."""
    parts: list[str] = []
    last = 0
    for match in _WORD.finditer(text):
        if _is_hit(match.group(0), terms, analyzer):
            parts.append(html.escape(text[last : match.start()]))
            parts.append(f"<mark>{html.escape(match.group(0))}</mark>")
            last = match.end()
    parts.append(html.escape(text[last:]))
    return "".join(parts)


def best_window(text: str, terms: Collection[str], analyzer: Analyzer, *, width: int = 300) -> str:
    """The ``width``-character stretch of ``text`` that contains the most query terms.

    Falls back to the start of the text when nothing matches (typical for dense-only hits).
    """
    if width < 20:
        raise ValueError("width must be at least 20")
    flat = " ".join(text.split())
    if len(flat) <= width:
        return flat
    hits = [m.start() for m in _WORD.finditer(flat) if _is_hit(m.group(0), terms, analyzer)]
    if not hits:
        return flat[:width].rsplit(" ", 1)[0] + " …"
    best_start, best_count = hits[0], 0
    for pos in hits:
        count = sum(1 for other in hits if pos <= other < pos + width)
        if count > best_count:
            best_start, best_count = pos, count
    start = max(0, best_start - 40)
    if start > 0:
        start = flat.find(" ", start) + 1 or start  # begin on a word boundary
    end = min(len(flat), start + width)
    snippet = flat[start:end]
    if end < len(flat):
        snippet = snippet.rsplit(" ", 1)[0] + " …"
    return ("… " if start > 0 else "") + snippet
