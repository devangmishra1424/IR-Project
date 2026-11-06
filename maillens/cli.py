"""Command-line entry point: ``python -m maillens <command>``.

The stages run in this order, and each one is safe to re-run:

    download   fetch the two public datasets from the Hugging Face Hub
    prepare    de-duplicate, thread, sample and clean the e-mails
    index      build the sparse (TF-IDF / BM25) inverted indexes
    embed      embed every chunk with the sentence-transformer (slow, resumable)
    queries    build the question and thread query sets with their relevance judgements
    evaluate   tune on dev queries, report on test queries, write results/
    search     answer one query with all four rankers
    app        launch the Streamlit search interface
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from maillens import config
from maillens.index.inverted import InvertedIndex

log = logging.getLogger("maillens")


def _doc_freq(index: InvertedIndex) -> Callable[[str], int]:
    def doc_freq(stem: str) -> int:
        term_id = index.term_id(stem)
        return 0 if term_id is None else int(index.df[term_id])

    return doc_freq


def cmd_download(args: argparse.Namespace) -> int:
    from maillens.prep.download import download_all

    download_all()
    return 0


def cmd_prepare(args: argparse.Namespace) -> int:
    from maillens.prep.corpus import build_corpus

    stats = build_corpus(target=args.target, seed=args.seed)
    for key, value in stats.items():
        log.info("%-28s %s", key, f"{value:,}")
    return 0


def cmd_index(args: argparse.Namespace) -> int:
    from maillens.pipeline import SPARSE_DIR, build_sparse_index
    from maillens.prep.corpus import load_emails, load_raw_bodies
    from maillens.text.analyzer import Analyzer

    emails = load_emails()
    analyzer = Analyzer()
    base = config.ARTIFACTS_DIR / SPARSE_DIR
    build_sparse_index(emails, analyzer, base / "main")
    # Same pipeline over the uncleaned bodies, for the cleaning ablation.
    raw = load_raw_bodies().reindex(emails["doc_id"]).fillna("").to_numpy()
    build_sparse_index(emails.assign(text=raw), analyzer, base / "raw")
    return 0


def cmd_embed(args: argparse.Namespace) -> int:
    from maillens.pipeline import DENSE_DIR, build_dense_index
    from maillens.prep.corpus import load_emails
    from maillens.rankers.dense import SentenceTransformerEncoder

    emails = load_emails()
    build_dense_index(emails, SentenceTransformerEncoder(), config.ARTIFACTS_DIR / DENSE_DIR)
    return 0


def cmd_queries(args: argparse.Namespace) -> int:
    from maillens.evaluation.queries import build_all_queries
    from maillens.pipeline import SPARSE_DIR
    from maillens.prep.corpus import load_emails
    from maillens.text.analyzer import Analyzer

    index = InvertedIndex.load(config.ARTIFACTS_DIR / SPARSE_DIR / "main")
    queries = build_all_queries(
        load_emails(),
        Analyzer(),
        _doc_freq(index),
        dev_threads=args.dev_threads,
        test_threads=args.test_threads,
    )
    counts: dict[str, int] = {}
    for query in queries:
        key = f"{query.role}/{query.kind}"
        counts[key] = counts.get(key, 0) + 1
    log.info("queries written: %s", dict(sorted(counts.items())))
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    from maillens.engine import SearchEngine
    from maillens.evaluation.experiments import run_evaluation
    from maillens.evaluation.queries import QUERIES_FILE, load_queries
    from maillens.pipeline import SPARSE_DIR

    engine = SearchEngine.load()
    raw_index = InvertedIndex.load(config.ARTIFACTS_DIR / SPARSE_DIR / "raw")
    queries = load_queries(config.PROCESSED_DIR / QUERIES_FILE)
    run_evaluation(engine, raw_index, queries)
    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    import pandas as pd

    from maillens.evaluation.analysis import length_analysis
    from maillens.evaluation.queries import QUERIES_FILE, load_queries
    from maillens.prep.corpus import load_emails

    per_query = pd.read_csv(config.RESULTS_DIR / "per_query.csv.gz")
    queries = load_queries(config.PROCESSED_DIR / QUERIES_FILE)
    table = length_analysis(per_query, queries, load_emails())
    path = config.RESULTS_DIR / "length_analysis.csv"
    table.to_csv(path, index=False, float_format="%.4f")
    log.info("wrote %s", path)
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    from maillens.engine import SearchEngine

    engine = SearchEngine.load()
    comparison = engine.compare(args.query, args.k)
    for system, hits in comparison.per_ranker.items():
        print(f"\n== {system} ==")
        for hit in hits:
            row = engine.email(hit.doc_id)
            day = row["date"].strftime("%Y-%m-%d")
            subject = (row["subject"] or "(no subject)")[:60]
            print(f"{hit.rank:>3}. {hit.score:8.4f}  {day}  {subject}  <{row['sender']}>")
    return 0


def cmd_figures(args: argparse.Namespace) -> int:
    from maillens.evaluation.figures import make_all_figures
    from maillens.prep.corpus import load_emails

    for path in make_all_figures(config.RESULTS_DIR, load_emails()):
        log.info("wrote %s (%.0f KB)", path, path.stat().st_size / 1024)
    return 0


def cmd_app(args: argparse.Namespace) -> int:
    app = Path(__file__).parent / "app" / "streamlit_app.py"
    # Run from the repository root so .streamlit/config.toml (the theme) is picked up.
    return subprocess.call(
        [sys.executable, "-m", "streamlit", "run", str(app), *args.streamlit_args],
        cwd=config.ROOT,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m maillens",
        description="MailLens: compare TF-IDF, BM25, dense and hybrid RRF search on Enron e-mail.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("download", help="fetch the datasets").set_defaults(run=cmd_download)

    prepare = sub.add_parser("prepare", help="de-duplicate, thread, sample and clean the e-mails")
    prepare.add_argument("--target", type=int, default=config.SAMPLE_TARGET, help="corpus size")
    prepare.add_argument("--seed", type=int, default=config.SEED)
    prepare.set_defaults(run=cmd_prepare)

    sub.add_parser("index", help="build the TF-IDF / BM25 indexes").set_defaults(run=cmd_index)
    sub.add_parser("embed", help="build the dense index (slow, resumable)").set_defaults(
        run=cmd_embed
    )

    queries = sub.add_parser("queries", help="build the query sets and relevance judgements")
    queries.add_argument("--dev-threads", type=int, default=100)
    queries.add_argument("--test-threads", type=int, default=300)
    queries.set_defaults(run=cmd_queries)

    sub.add_parser("evaluate", help="run the full evaluation").set_defaults(run=cmd_evaluate)
    sub.add_parser("analyze", help="e-mail length analysis of the results").set_defaults(
        run=cmd_analyze
    )
    sub.add_parser("figures", help="draw the figures from the results").set_defaults(
        run=cmd_figures
    )

    search = sub.add_parser("search", help="search from the terminal")
    search.add_argument("query")
    search.add_argument("-k", type=int, default=5, help="results per ranker")
    search.set_defaults(run=cmd_search)

    app = sub.add_parser("app", help="launch the Streamlit interface")
    app.add_argument("streamlit_args", nargs="*", help="extra arguments passed to streamlit")
    app.set_defaults(run=cmd_app)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    return int(args.run(args))
