import re

import numpy as np
import pandas as pd
import pytest

from maillens import theme
from maillens.evaluation import figures

SYSTEMS = list(theme.SYSTEM_ORDER)
KINDS = list(theme.KIND_LABELS)


def assert_png(path, min_bytes=4000):
    data = path.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    assert len(data) > min_bytes


def test_theme_defines_every_ranker_once_with_a_distinct_colour_and_marker():
    assert set(theme.SYSTEM_COLORS) == set(theme.SYSTEM_ORDER) == set(theme.SYSTEM_LABELS)
    assert set(theme.SYSTEM_MARKERS) == set(theme.SYSTEM_ORDER)
    assert len(set(theme.SYSTEM_COLORS.values())) == 4
    assert len(set(theme.SYSTEM_MARKERS.values())) == 4
    assert all(re.fullmatch(r"#[0-9a-f]{6}", c) for c in theme.SYSTEM_COLORS.values())
    assert set(theme.TABLE_ORDER) == set(theme.SYSTEM_ORDER)


@pytest.fixture
def curves():
    rows = []
    for si, system in enumerate(SYSTEMS):
        for kind in KINDS:
            for level in np.linspace(0, 1, 11):
                rows.append(
                    {
                        "system": system,
                        "role": "test",
                        "kind": kind,
                        "recall": level,
                        "precision": max(0.0, 0.9 - 0.5 * level - 0.05 * si),
                    }
                )
    return pd.DataFrame(rows)


def test_precision_recall_figure_is_written(tmp_path, curves):
    path = figures.precision_recall_figure(
        curves, {"question": 1515, "subject": 300, "keyword": 300}, tmp_path / "f" / "pr.png"
    )
    assert_png(path)


@pytest.fixture
def ci():
    rows = []
    for si, system in enumerate(SYSTEMS):
        for kind in KINDS:
            base = 0.8 - 0.1 * si
            rows.append(
                {
                    "kind": kind,
                    "system": system,
                    "MAP": base,
                    "MAP_low": base - 0.03,
                    "MAP_high": base + 0.03,
                    "nDCG@10": base + 0.05,
                    "nDCG@10_low": base + 0.02,
                    "nDCG@10_high": base + 0.08,
                }
            )
    return pd.DataFrame(rows)


def test_metrics_bars_figure_is_written(tmp_path, ci):
    assert_png(figures.metrics_bars_figure(ci, tmp_path / "bars.png"))


def test_rrf_k_figure_is_written(tmp_path):
    ks = [1, 5, 10, 20, 40, 60, 100, 200, 500]
    sweep = pd.DataFrame(
        [
            {"kind": kind, "system": f"hybrid_k{k}", "MAP": 0.7 + 0.01 * np.log10(k)}
            for kind in KINDS
            for k in ks
        ]
    )
    refs = pd.DataFrame(
        [
            {"kind": kind, "system": s, "MAP": m}
            for kind in KINDS
            for s, m in (("bm25", 0.7), ("dense", 0.5))
        ]
    )
    assert_png(figures.rrf_k_figure(sweep, refs, tmp_path / "k.png"))


def test_success_at_k_figure_is_written(tmp_path):
    rng = np.random.default_rng(0)
    rows = [
        {"role": "test", "kind": "question", "system": system, "first_relevant_rank": int(r)}
        for system in SYSTEMS
        for r in rng.integers(0, 80, size=60)
    ]
    assert_png(figures.success_at_k_figure(pd.DataFrame(rows), tmp_path / "s.png"))


def test_corpus_figure_is_written(tmp_path):
    rng = np.random.default_rng(1)
    n = 300
    emails = pd.DataFrame(
        {
            "date": pd.Timestamp("1997-01-01", tz="UTC")
            + pd.to_timedelta(rng.integers(0, 6 * 365, n), unit="D"),
            "thread_id": rng.integers(0, 120, n),
            "fresh": ["word " * int(w) for w in rng.integers(0, 600, n)],
            "quoted": [""] * n,
        }
    )
    assert_png(figures.corpus_figure(emails, tmp_path / "c.png"))
