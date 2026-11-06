"""Figures for the report and the app, drawn from the CSV files the evaluation writes.

Design rules (see ``maillens/theme.py``): one colour per ranker for good, 2px lines with
distinct markers, hairline solid grid, text in ink colours, a legend whenever there are two or
more series, and every exact number also available in a table.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from itertools import pairwise
from pathlib import Path
from typing import Literal

import matplotlib

matplotlib.use("Agg")  # no display needed; figures are written to files

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from maillens import theme

DPI = 200
LINE_WIDTH = 1.6  # ~2px
MARKER_SIZE = 6  # ~8px
plt.rcParams.update(
    {
        "font.family": ["Segoe UI", "DejaVu Sans"],
        "font.size": 8.5,
        "axes.edgecolor": theme.BASELINE,
        "axes.labelcolor": theme.INK_SECONDARY,
        "axes.titlecolor": theme.INK,
        "xtick.color": theme.INK_MUTED,
        "ytick.color": theme.INK_MUTED,
        "text.color": theme.INK,
        "figure.facecolor": theme.PAPER,
        "axes.facecolor": theme.PAPER,
        "savefig.facecolor": theme.PAPER,
    }
)


def _style(ax: Axes, *, grid_axis: Literal["both", "x", "y"] = "y") -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(theme.BASELINE)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(length=0, labelsize=8)
    ax.grid(axis=grid_axis, color=theme.GRID, linewidth=0.6, linestyle="-")
    ax.set_axisbelow(True)


def _legend(fig: Figure, systems: Sequence[str]) -> None:
    handles = [
        Line2D(
            [0],
            [0],
            color=theme.SYSTEM_COLORS[s],
            marker=theme.SYSTEM_MARKERS[s],
            linewidth=LINE_WIDTH,
            markersize=MARKER_SIZE,
            markeredgecolor=theme.PAPER,
            markeredgewidth=1.0,
            label=theme.SYSTEM_LABELS[s],
        )
        for s in systems
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=len(handles),
        frameon=False,
        fontsize=8.5,
        labelcolor=theme.INK_SECONDARY,
        handlelength=2.4,
        columnspacing=1.8,
    )


def _save(fig: Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return path


def _systems_present(frame: pd.DataFrame, wanted: Iterable[str] = theme.SYSTEM_ORDER) -> list[str]:
    have = set(frame["system"])
    return [s for s in wanted if s in have]


# ----------------------------------------------------------------------------- results
def precision_recall_figure(
    curves: pd.DataFrame, queries_per_kind: dict[str, int], out: Path
) -> Path:
    """Mean 11-point interpolated precision-recall curves, one panel per query kind."""
    kinds = [k for k in theme.KIND_LABELS if k in set(curves["kind"])]
    systems = _systems_present(curves)
    fig, axes = plt.subplots(1, len(kinds), figsize=(7.2, 2.9), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, kind in zip(axes, kinds, strict=True):
        _style(ax)
        for system in systems:
            data = curves[(curves["kind"] == kind) & (curves["system"] == system)].sort_values(
                "recall"
            )
            ax.plot(
                data["recall"],
                data["precision"],
                color=theme.SYSTEM_COLORS[system],
                marker=theme.SYSTEM_MARKERS[system],
                markevery=2,
                linewidth=LINE_WIDTH,
                markersize=MARKER_SIZE,
                markeredgecolor=theme.PAPER,
                markeredgewidth=1.0,
                solid_capstyle="round",
                solid_joinstyle="round",
            )
        n = queries_per_kind.get(kind)
        ax.set_title(
            f"{theme.KIND_SHORT[kind]}" + (f"  (n = {n:,})" if n else ""),
            fontsize=9,
            loc="left",
            pad=8,
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_xlabel("Recall")
    axes[0].set_ylabel("Interpolated precision")
    fig.subplots_adjust(bottom=0.27, wspace=0.12)
    _legend(fig, systems)
    return _save(fig, out)


def metrics_bars_figure(
    ci: pd.DataFrame, out: Path, metrics: Sequence[str] = ("MAP", "nDCG@10")
) -> Path:
    """Grouped columns of the headline metrics with 95% bootstrap intervals."""
    kinds = [k for k in theme.KIND_LABELS if k in set(ci["kind"])]
    systems = _systems_present(ci)
    fig, axes = plt.subplots(1, len(metrics), figsize=(7.2, 3.0), sharey=True)
    axes = np.atleast_1d(axes)
    width = 0.17
    offsets = (np.arange(len(systems)) - (len(systems) - 1) / 2) * (width + 0.02)
    for ax, metric in zip(axes, metrics, strict=True):
        _style(ax)
        for j, system in enumerate(systems):
            sub = ci[ci["system"] == system].set_index("kind").reindex(kinds)
            x = np.arange(len(kinds)) + offsets[j]
            mean = sub[metric].to_numpy()
            err = np.vstack(
                [mean - sub[f"{metric}_low"].to_numpy(), sub[f"{metric}_high"].to_numpy() - mean]
            )
            ax.bar(x, mean, width=width, color=theme.SYSTEM_COLORS[system], linewidth=0, zorder=2)
            ax.errorbar(
                x,
                mean,
                yerr=err,
                fmt="none",
                ecolor=theme.INK_MUTED,
                elinewidth=0.8,
                capsize=0,
                zorder=3,
            )
            for xi, value, high in zip(x, mean, sub[f"{metric}_high"].to_numpy(), strict=True):
                ax.text(
                    xi,
                    high + 0.015,
                    f"{value:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=6.5,
                    color=theme.INK_SECONDARY,
                )
        ax.set_xticks(np.arange(len(kinds)))
        ax.set_xticklabels([theme.KIND_SHORT[k] for k in kinds])
        ax.set_title(metric, fontsize=9, loc="left", pad=8)
        ax.set_ylim(0, 1.0)
    axes[0].set_ylabel("Mean over test queries (bars: 95% CI)")
    fig.subplots_adjust(bottom=0.27, wspace=0.08)
    _legend(fig, systems)
    return _save(fig, out)


def rrf_k_figure(
    sweep: pd.DataFrame, references: pd.DataFrame, out: Path, metric: str = "MAP"
) -> Path:
    """How the hybrid's score changes with RRF's k, against the single rankers as references."""
    sweep = sweep.assign(k=sweep["system"].str.removeprefix("hybrid_k").astype(int))
    kinds = [k for k in theme.KIND_LABELS if k in set(sweep["kind"])]
    fig, axes = plt.subplots(1, len(kinds), figsize=(7.2, 2.9))
    axes = np.atleast_1d(axes)
    for ax, kind in zip(axes, kinds, strict=True):
        _style(ax)
        data = sweep[sweep["kind"] == kind].sort_values("k")
        ax.plot(
            data["k"],
            data[metric],
            color=theme.SYSTEM_COLORS["hybrid"],
            marker=theme.SYSTEM_MARKERS["hybrid"],
            linewidth=LINE_WIDTH,
            markersize=MARKER_SIZE,
            markeredgecolor=theme.PAPER,
            markeredgewidth=1.0,
            zorder=3,
        )
        ref = references[references["kind"] == kind].set_index("system")
        for system in ("bm25", "dense"):
            if system in ref.index:
                y = float(ref.loc[system, metric])
                ax.axhline(y, color=theme.SYSTEM_COLORS[system], linewidth=1.2, zorder=2)
                ax.text(
                    data["k"].max(),
                    y,
                    f" {theme.SYSTEM_SHORT[system]}",
                    va="center",
                    ha="left",
                    fontsize=7,
                    color=theme.INK_SECONDARY,
                )
        ax.axvline(60, color=theme.BASELINE, linewidth=0.8, zorder=1)
        ax.set_xscale("log")
        ax.set_xticks([1, 10, 60, 500])
        ax.set_xticklabels(["1", "10", "60", "500"])
        ax.minorticks_off()
        ax.set_xlabel("RRF constant k (log scale)")
        ax.set_title(theme.KIND_SHORT[kind], fontsize=9, loc="left", pad=8)
        lo = min(data[metric].min(), ref[metric].min() if len(ref) else 1.0)
        hi = max(data[metric].max(), ref[metric].max() if len(ref) else 0.0)
        pad = max(0.01, (hi - lo) * 0.15)
        ax.set_ylim(lo - pad, hi + pad)
    axes[0].set_ylabel(metric)
    fig.subplots_adjust(bottom=0.2, wspace=0.55)
    return _save(fig, out)


def success_at_k_figure(per_query: pd.DataFrame, out: Path, max_k: int = 50) -> Path:
    """Share of test questions whose answer is found within the top k (known-item search)."""
    data = per_query[(per_query["role"] == "test") & (per_query["kind"] == "question")]
    systems = _systems_present(data)
    ks = np.arange(1, max_k + 1)
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    _style(ax)
    for system in systems:
        ranks = data.loc[data["system"] == system, "first_relevant_rank"].to_numpy()
        found = np.array([(ranks > 0) & (ranks <= k) for k in ks]).mean(axis=1)
        ax.plot(
            ks,
            found,
            color=theme.SYSTEM_COLORS[system],
            linewidth=LINE_WIDTH,
            marker=theme.SYSTEM_MARKERS[system],
            markevery=(4, 10),
            markersize=MARKER_SIZE,
            markeredgecolor=theme.PAPER,
            markeredgewidth=1.0,
        )
    ax.set_xlim(1, max_k)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("Rank cut-off k")
    ax.set_ylabel("Questions answered in the top k")
    fig.subplots_adjust(bottom=0.32)
    _legend(fig, systems)
    return _save(fig, out)


# ----------------------------------------------------------------------------- the data set
def _single_series_bars(ax: Axes, labels: Sequence[str], values: Sequence[float]) -> None:
    ax.bar(
        range(len(values)),
        values,
        width=0.62,
        color=theme.SYSTEM_COLORS["hybrid"],
        linewidth=0,
        zorder=2,
    )
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)


def corpus_figure(emails: pd.DataFrame, out: Path) -> Path:
    """Three views of the evaluation corpus: e-mails per year, thread sizes, length in words."""
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5))
    for ax in axes:
        _style(ax)

    years = emails["date"].dt.year.value_counts().sort_index()
    _single_series_bars(axes[0], [str(y) for y in years.index], years.to_numpy())
    axes[0].set_title("E-mails per year", fontsize=9, loc="left", pad=8)
    axes[0].set_ylabel("E-mails")
    axes[0].tick_params(axis="x", labelsize=7, rotation=45)

    sizes = emails.groupby("thread_id").size()
    edges = [1, 2, 3, 4, 6, 11, 21]
    names = ["1", "2", "3", "4-5", "6-10", "11-20", ">20"]
    counts = [
        int(((sizes >= lo) & (sizes < hi)).sum())
        for lo, hi in zip(edges, [*edges[1:], 10**9], strict=True)
    ]
    _single_series_bars(axes[1], names, counts)
    axes[1].set_yscale("log")
    axes[1].set_title("Messages per thread", fontsize=9, loc="left", pad=8)
    axes[1].set_ylabel("Threads (log scale)")
    axes[1].tick_params(axis="x", labelsize=7)

    words = (emails["fresh"] + " " + emails["quoted"]).str.split().str.len()
    bins = [0, 1, 25, 50, 100, 200, 400, 800, 10**9]
    labels = ["0", "1-24", "25-49", "50-99", "100-199", "200-399", "400-799", "800+"]
    counts = [int(((words >= lo) & (words < hi)).sum()) for lo, hi in pairwise(bins)]
    _single_series_bars(axes[2], labels, counts)
    axes[2].set_title("Body length (words)", fontsize=9, loc="left", pad=8)
    axes[2].set_ylabel("E-mails")
    axes[2].tick_params(axis="x", labelsize=6.5, rotation=45)

    fig.subplots_adjust(wspace=0.45, bottom=0.25)
    return _save(fig, out)


def make_all_figures(results_dir: Path, emails: pd.DataFrame) -> list[Path]:
    """Draw every figure from the CSV files the evaluation wrote (and the corpus itself)."""
    out = results_dir / "figures"
    info = json.loads((results_dir / "run_info.json").read_text(encoding="utf-8"))
    test_counts = {
        key.split("/", 1)[1]: int(n)
        for key, n in info["queries"].items()
        if key.startswith("test/")
    }
    curves = pd.read_csv(results_dir / "pr_curves.csv")
    main = pd.read_csv(results_dir / "metrics_main.csv")
    return [
        precision_recall_figure(
            curves[curves["role"] == "test"], test_counts, out / "precision_recall.png"
        ),
        metrics_bars_figure(pd.read_csv(results_dir / "metrics_ci.csv"), out / "metrics_bars.png"),
        rrf_k_figure(pd.read_csv(results_dir / "rrf_k_sweep.csv"), main, out / "rrf_k_sweep.png"),
        success_at_k_figure(
            pd.read_csv(results_dir / "per_query.csv.gz"), out / "success_at_k.png"
        ),
        corpus_figure(emails, out / "corpus.png"),
    ]


# ----------------------------------------------------------------------------- the pipeline
def _box(
    ax: Axes, x: float, y: float, w: float, h: float, text: str, accent: str | None = None
) -> None:
    from matplotlib.patches import FancyBboxPatch

    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0,rounding_size=0.09",
            linewidth=0.9,
            edgecolor=theme.BASELINE,
            facecolor=theme.PAPER,
            zorder=2,
        )
    )
    if accent:
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                0.07,
                h,
                boxstyle="round,pad=0,rounding_size=0.035",
                linewidth=0,
                facecolor=accent,
                zorder=3,
            )
        )
    ax.text(
        x + w / 2 + (0.03 if accent else 0),
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=7.6,
        color=theme.INK,
        linespacing=1.25,
        zorder=4,
    )


def _arrow(ax: Axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops={
            "arrowstyle": "-|>",
            "color": theme.INK_MUTED,
            "linewidth": 1.0,
            "shrinkA": 0,
            "shrinkB": 0,
            "mutation_scale": 8,
        },
        zorder=1,
    )


def pipeline_figure(counts: dict[str, int], out: Path) -> Path:
    """The architecture: what happens offline when the indexes are built and online per query."""
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    ax.set_xlim(0, 10)
    ax.set_ylim(0.15, 5)
    ax.axis("off")
    colors = theme.SYSTEM_COLORS

    ax.text(
        0,
        4.88,
        "Offline: build the indexes",
        fontsize=8.2,
        color=theme.INK_SECONDARY,
        fontweight="bold",
        va="center",
    )
    raw = f"{counts['raw_rows']:,}" if "raw_rows" in counts else "raw dump"
    kept = f"{counts['sample_messages']:,}" if "sample_messages" in counts else "sample"
    _box(ax, 0.0, 3.55, 1.75, 0.9, f"Enron dump\n{raw} files")
    _box(ax, 2.15, 3.55, 2.05, 0.9, f"De-duplicate · clean\nthread · sample\n{kept} e-mails")
    _box(ax, 4.6, 4.05, 2.0, 0.55, "Analyzer: tokens,\nstop-words, stems")
    _box(ax, 7.0, 4.05, 1.55, 0.55, "Inverted\nindex")
    _box(ax, 4.6, 3.35, 2.0, 0.55, "Split into\n≤4 chunks / e-mail")
    _box(ax, 7.0, 3.35, 2.6, 0.55, "Sentence-Transformer\nchunk embeddings")
    _arrow(ax, (1.75, 4.0), (2.15, 4.0))
    _arrow(ax, (4.2, 4.1), (4.6, 4.33))
    _arrow(ax, (4.2, 3.9), (4.6, 3.62))
    _arrow(ax, (6.6, 4.33), (7.0, 4.33))
    _arrow(ax, (6.6, 3.62), (7.0, 3.62))

    ax.plot([0, 10], [2.95, 2.95], color=theme.GRID, linewidth=0.8)
    ax.text(
        0,
        2.7,
        "Online: answer one query",
        fontsize=8.2,
        color=theme.INK_SECONDARY,
        fontweight="bold",
        va="center",
    )
    _box(ax, 0.0, 1.35, 1.2, 0.9, "Query")
    _box(ax, 1.6, 1.8, 1.9, 0.55, "Analyzer", None)
    _box(ax, 1.6, 0.95, 1.9, 0.55, "Encoder", None)
    _box(ax, 4.05, 2.0, 1.7, 0.5, "TF-IDF cosine", colors["tfidf"])
    _box(ax, 4.05, 1.4, 1.7, 0.5, "BM25", colors["bm25"])
    _box(ax, 4.05, 0.8, 1.7, 0.5, "Dense, max chunk", colors["dense"])
    _box(ax, 6.35, 1.4, 1.75, 0.9, "Reciprocal rank\nfusion", colors["hybrid"])
    _box(ax, 8.55, 1.4, 1.45, 0.9, "Ranked\ne-mails")
    _arrow(ax, (1.2, 1.9), (1.6, 2.07))
    _arrow(ax, (1.2, 1.7), (1.6, 1.22))
    _arrow(ax, (3.5, 2.07), (4.05, 2.25))
    _arrow(ax, (3.5, 2.07), (4.05, 1.65))
    _arrow(ax, (3.5, 1.22), (4.05, 1.05))
    for y in (2.25, 1.65, 1.05):
        _arrow(ax, (5.75, y), (6.35, 1.85))
    _arrow(ax, (8.1, 1.85), (8.55, 1.85))
    ax.text(
        0,
        0.38,
        "TF-IDF and BM25 read the inverted index; the dense ranker reads the chunk embeddings.",
        fontsize=7.4,
        color=theme.INK_MUTED,
        va="center",
    )
    return _save(fig, out)
