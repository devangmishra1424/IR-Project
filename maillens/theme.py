"""Colours and labels shared by the figures and the search app.

The four ranker colours are the first four slots of a categorical palette validated for
colour-blind separation (adjacent CVD delta-E 9.1, normal-vision 22.9; the aqua and yellow marks sit
below 3:1 contrast on white, so every chart also carries a legend, distinct marker shapes and a
table). A colour belongs to a ranker for good: filtering or re-ordering never repaints it.
Text is always set in the ink colours, never in a series colour.
"""

from __future__ import annotations

# ----------------------------------------------------------------------------- ink and chrome
PAPER = "#ffffff"
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

# ----------------------------------------------------------------------------- rankers
# Legend and slot order: the best-performing ranker first, as it is the one the story is about.
SYSTEM_ORDER = ("hybrid", "bm25", "dense", "tfidf")
SYSTEM_COLORS = {
    "hybrid": "#2a78d6",  # categorical slot 1 (blue)
    "bm25": "#eb6834",  # slot 2 (orange)
    "dense": "#1baf7a",  # slot 3 (aqua)
    "tfidf": "#eda100",  # slot 4 (yellow)
}
SYSTEM_MARKERS = {"hybrid": "o", "bm25": "s", "dense": "^", "tfidf": "D"}
SYSTEM_LABELS = {
    "tfidf": "TF-IDF",
    "bm25": "BM25",
    "dense": "Sentence-Transformer",
    "hybrid": "Hybrid RRF",
}
SYSTEM_SHORT = {"tfidf": "TF-IDF", "bm25": "BM25", "dense": "Dense", "hybrid": "Hybrid RRF"}
# The order used in tables, following the assignment sheet.
TABLE_ORDER = ("tfidf", "bm25", "dense", "hybrid")

KIND_LABELS = {
    "question": "Natural-language questions",
    "subject": "Thread subject lines",
    "keyword": "Thread keyword queries",
}
KIND_SHORT = {"question": "Questions", "subject": "Subject lines", "keyword": "Keywords"}
