"""Presentation helpers for the app, free of Streamlit so they can be unit-tested.

Everything that comes from an e-mail (subject, sender, body) is untrusted text, so it is
HTML-escaped here before it reaches ``unsafe_allow_html`` markup; only our own tags are added.
"""

from __future__ import annotations

import html
from collections.abc import Callable, Mapping, Sequence
from itertools import combinations

import pandas as pd

from maillens import theme
from maillens.rankers.base import Hit

CSS = f"""
<style>
html, body, [class*="css"] {{
  font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
}}
.block-container {{ padding-top: 2.2rem; max-width: 1480px; }}
.ml-brand {{ font-size: 1.7rem; font-weight: 700; letter-spacing: -0.02em; color: {theme.INK}; }}
.ml-tagline {{ color: {theme.INK_SECONDARY}; margin: 0.1rem 0 0.9rem 0; }}
.ml-colhead {{ border-top: 3px solid var(--c); padding-top: 0.55rem; margin-bottom: 0.7rem; }}
.ml-colhead .name {{ font-weight: 650; color: {theme.INK}; }}
.ml-colhead .desc {{ color: {theme.INK_MUTED}; font-size: 0.78rem; line-height: 1.3; }}
.ml-card {{
  border: 1px solid {theme.GRID}; border-radius: 10px; padding: 0.7rem 0.8rem;
  margin-bottom: 0.3rem; background: {theme.PAPER};
}}
.ml-card .top {{ display: flex; gap: 0.6rem; align-items: baseline; }}
.ml-rank {{
  font-variant-numeric: tabular-nums; color: {theme.INK_MUTED}; font-size: 0.8rem; min-width: 1.1rem;
}}
.ml-subject {{ font-weight: 600; line-height: 1.25; color: {theme.INK}; }}
.ml-meta {{ color: {theme.INK_SECONDARY}; font-size: 0.76rem; margin: 0.15rem 0 0.4rem 1.7rem; }}
.ml-snippet {{ color: #2b2a28; font-size: 0.83rem; line-height: 1.42; margin-left: 1.7rem; }}
.ml-snippet mark {{ background: #fbe9b0; color: inherit; padding: 0 0.1em; border-radius: 3px; }}
.ml-ranks {{ margin: 0.5rem 0 0 1.7rem; display: flex; flex-wrap: wrap; gap: 0.3rem; }}
.ml-chip {{
  font-size: 0.68rem; border: 1px solid {theme.GRID}; border-radius: 999px;
  padding: 0.04rem 0.45rem; color: {theme.INK_SECONDARY}; display: inline-flex;
  align-items: center; gap: 0.3rem; font-variant-numeric: tabular-nums;
}}
.ml-chip .dot {{ width: 0.5rem; height: 0.5rem; border-radius: 50%; background: var(--c); }}
.ml-chip.here {{ border-color: {theme.BASELINE}; color: {theme.INK}; font-weight: 600; }}
.ml-more {{ color: {theme.INK_MUTED}; font-size: 0.74rem; margin: -0.1rem 0 0.5rem 0.2rem; }}
.ml-agree {{ color: {theme.INK_SECONDARY}; font-size: 0.8rem; margin: 0.2rem 0 0.9rem 0; }}
.ml-agree b {{ color: {theme.INK}; font-variant-numeric: tabular-nums; }}
.ml-body {{
  white-space: pre-wrap; font-size: 0.86rem; line-height: 1.5; color: #2b2a28;
  border: 1px solid {theme.GRID}; border-radius: 8px; padding: 0.8rem 0.9rem; background: {theme.PAPER};
  max-height: 340px; overflow-y: auto;
}}
.ml-body mark {{ background: #fbe9b0; color: inherit; padding: 0 0.1em; border-radius: 3px; }}
.ml-relevant {{ color: {theme.INK}; font-weight: 700; }}
</style>
"""

SYSTEM_DESCRIPTIONS = {
    "tfidf": "Classic vector space: cosine of sub-linear TF-IDF vectors.",
    "bm25": "Probabilistic: saturating term frequency and length normalisation.",
    "dense": "Meaning, not words: nearest e-mail chunk by embedding similarity.",
    "hybrid": "Reciprocal rank fusion of TF-IDF, BM25 and dense rank lists.",
}


def format_date(value: pd.Timestamp) -> str:
    return f"{value.day} {value.strftime('%b %Y')}"


def column_header_html(system: str) -> str:
    color = theme.SYSTEM_COLORS[system]
    return (
        f'<div class="ml-colhead" style="--c:{color}">'
        f'<div class="name">{html.escape(theme.SYSTEM_LABELS[system])}</div>'
        f'<div class="desc">{html.escape(SYSTEM_DESCRIPTIONS[system])}</div></div>'
    )


def rank_chips_html(ranks: Mapping[str, int | None], current: str) -> str:
    """Small chips showing this e-mail's rank in every ranker ("-" where it was not retrieved)."""
    chips = []
    for system in theme.SYSTEM_ORDER:
        rank = ranks.get(system)
        label = f"#{rank}" if rank is not None else "–"
        css = "ml-chip here" if system == current else "ml-chip"
        chips.append(
            f'<span class="{css}" style="--c:{theme.SYSTEM_COLORS[system]}">'
            f'<span class="dot"></span>{html.escape(theme.SYSTEM_SHORT[system])} {label}</span>'
        )
    return f'<div class="ml-ranks">{"".join(chips)}</div>'


def card_html(
    *,
    rank: int,
    subject: str,
    sender: str,
    when: pd.Timestamp,
    thread_size: int,
    snippet_html: str,
    ranks: Mapping[str, int | None],
    current: str,
) -> str:
    """One result card. ``snippet_html`` must already be escaped (see ``snippets.highlight``)."""
    title = html.escape(subject.strip() or "(no subject)")
    thread = f" · thread of {thread_size}" if thread_size > 1 else ""
    meta = f"{html.escape(sender)} · {format_date(when)}{thread}"
    return (
        '<div class="ml-card">'
        f'<div class="top"><span class="ml-rank">{rank}</span><span class="ml-subject">{title}</span></div>'
        f'<div class="ml-meta">{meta}</div>'
        f'<div class="ml-snippet">{snippet_html}</div>'
        f"{rank_chips_html(ranks, current)}"
        "</div>"
    )


def collapse_threads(hits: Sequence[Hit], thread_of: Callable[[int], int]) -> list[tuple[Hit, int]]:
    """Keep each thread's best-ranked hit; also report how many lower hits were folded into it."""
    kept: dict[int, list[Hit]] = {}
    for hit in hits:
        kept.setdefault(thread_of(hit.doc_id), []).append(hit)
    return [(group[0], len(group) - 1) for group in kept.values()]


def top_overlap(per_ranker: Mapping[str, Sequence[int]], n: int) -> list[tuple[str, str, int]]:
    """For each pair of rankers, how many documents their top ``n`` lists share."""
    tops = {name: set(list(docs)[:n]) for name, docs in per_ranker.items()}
    pairs = []
    for a, b in combinations(theme.SYSTEM_ORDER, 2):
        if a in tops and b in tops:
            pairs.append((a, b, len(tops[a] & tops[b])))
    return pairs


def agreement_html(per_ranker: Mapping[str, Sequence[int]], n: int) -> str:
    """One line summarising how much the rankers' top lists agree."""
    parts = [
        f"{html.escape(theme.SYSTEM_SHORT[a])} ∩ {html.escape(theme.SYSTEM_SHORT[b])} "
        f"<b>{shared}/{n}</b>"
        for a, b, shared in top_overlap(per_ranker, n)
    ]
    return f'<div class="ml-agree">Top-{n} agreement: {" · ".join(parts)}</div>'
