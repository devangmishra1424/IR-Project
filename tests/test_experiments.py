"""End-to-end run of the whole evaluation protocol on a small synthetic corpus."""

import json

import numpy as np
import pandas as pd
import pytest

from maillens.engine import SearchEngine
from maillens.evaluation.experiments import (
    BM25_GRID_B,
    BM25_GRID_K1,
    COMPARISONS,
    MAIN_SYSTEMS,
    RRF_K_SWEEP,
    run_evaluation,
)
from maillens.evaluation.figures import make_all_figures
from maillens.evaluation.queries import Query
from maillens.index.inverted import InvertedIndex
from maillens.pipeline import build_dense_chunks, index_text
from maillens.rankers.bm25 import Bm25Ranker
from maillens.rankers.dense import DenseRanker
from maillens.rankers.tfidf import TfidfRanker
from maillens.text.analyzer import Analyzer
from tests.helpers import HashingEncoder

N_THREADS, PER_THREAD = 14, 5


def build_world():
    rng = np.random.default_rng(0)
    rows = []
    for t in range(N_THREADS):
        topic = [f"topic{t}w{j}" for j in range(4)]
        for m in range(PER_THREAD):
            noise = " ".join(f"noise{int(x)}" for x in rng.integers(0, 40, 6))
            rows.append(
                {
                    "doc_id": len(rows),
                    "message_id": f"<{len(rows)}@t>",
                    "subject": f"project {topic[0]} update",
                    "subject_norm": f"project {topic[0]} update",
                    "sender": f"person{t}@enron.com",
                    "recipients": ["x@enron.com"],
                    "date": pd.Timestamp("2000-01-01", tz="UTC") + pd.Timedelta(days=30 * t + m),
                    "mailbox": f"box{t % 3}",
                    "folder": "inbox",
                    "thread_id": t,
                    "thread_size": PER_THREAD,
                    "fresh": f"{' '.join(topic)} {noise} message{m}"
                    + (f" unique{t}" if m == 0 else ""),
                    "quoted": "",
                }
            )
    emails = pd.DataFrame(rows)
    emails["text"] = emails["fresh"]
    analyzer = Analyzer()
    docs = [
        analyzer(index_text(s, t)) for s, t in zip(emails["subject"], emails["text"], strict=True)
    ]
    index, raw_index = InvertedIndex.build(docs), InvertedIndex.build(docs)
    encoder = HashingEncoder()
    texts, chunk_doc = build_dense_chunks(emails)
    dense = DenseRanker(encoder.encode(texts), chunk_doc, len(emails), encoder)
    engine = SearchEngine(
        emails, analyzer, TfidfRanker(index, analyzer), Bm25Ranker(index, analyzer), dense
    )

    queries = []
    for t in range(N_THREADS):
        role = "dev" if t < 4 else "test"
        members = frozenset(range(t * PER_THREAD, (t + 1) * PER_THREAD))
        topic = [f"topic{t}w{j}" for j in range(4)]
        queries.append(
            Query(f"q-{t}", f"{topic[1]} unique{t}", "question", role, frozenset({t * PER_THREAD}))
        )
        queries.append(Query(f"s-{t}", f"project {topic[0]} update", "subject", role, members))
        queries.append(Query(f"k-{t}", f"{topic[2]} {topic[3]}", "keyword", role, members))
    return engine, raw_index, emails, queries


@pytest.fixture(scope="module")
def evaluated(tmp_path_factory):
    root = tmp_path_factory.mktemp("eval")
    (root / "artifacts").mkdir()
    (root / "processed").mkdir()
    (root / "processed" / "corpus_stats.json").write_text(json.dumps({"raw_rows": 1}))
    engine, raw_index, emails, queries = build_world()
    info = run_evaluation(
        engine,
        raw_index,
        queries,
        results_dir=root / "results",
        artifacts_dir=root / "artifacts",
        processed_dir=root / "processed",
        depth=50,
    )
    return root, info, emails, queries


def test_every_output_file_is_written(evaluated):
    root, _, _, _ = evaluated
    names = {p.name for p in (root / "results").iterdir()}
    assert {
        "bm25_tuning.csv",
        "metrics_main.csv",
        "metrics_dev.csv",
        "metrics_all_systems.csv",
        "metrics_ci.csv",
        "pr_curves.csv",
        "significance.csv",
        "rrf_k_sweep.csv",
        "ablation.csv",
        "per_query.csv.gz",
        "queries.csv",
        "run_info.json",
    } <= names


def test_tuning_stays_on_the_grid_and_is_saved_for_the_app(evaluated):
    root, info, _, _ = evaluated
    saved = json.loads((root / "artifacts" / "params.json").read_text())["bm25"]
    assert saved == info["bm25_tuned"]
    assert saved["k1"] in BM25_GRID_K1 and saved["b"] in BM25_GRID_B
    grid = pd.read_csv(root / "results" / "bm25_tuning.csv")
    assert len(grid) == len(BM25_GRID_K1) * len(BM25_GRID_B)


def test_main_table_covers_every_system_and_kind_on_test_queries_only(evaluated):
    root, _, _, queries = evaluated
    main = pd.read_csv(root / "results" / "metrics_main.csv")
    assert set(main["role"]) == {"test"}
    assert set(main["system"]) == set(MAIN_SYSTEMS)
    assert set(main["kind"]) == {"question", "subject", "keyword"}
    assert len(main) == 12
    n_test_per_kind = sum(1 for q in queries if q.role == "test" and q.kind == "question")
    assert (main["queries"] == n_test_per_kind).all()
    for metric in ("P@10", "R@100", "MAP", "nDCG@10", "MRR"):
        assert main[metric].between(0, 1).all()


def test_easy_synthetic_queries_are_solved_by_every_ranker(evaluated):
    root, _, _, _ = evaluated
    main = pd.read_csv(root / "results" / "metrics_main.csv")
    # Each query has distinctive words: the lexical rankers must be near-perfect, and the
    # hashing stand-in for the neural encoder (which has word-hash collisions) must still be good.
    lexical = main[main["system"].isin(["tfidf", "bm25"])]
    assert (lexical["MAP"] > 0.95).all()
    assert (main["MAP"] > 0.7).all()


def test_significance_covers_every_comparison_kind_and_metric(evaluated):
    root, _, _, _ = evaluated
    sig = pd.read_csv(root / "results" / "significance.csv")
    assert len(sig) == 3 * 2 * len(COMPARISONS)
    assert sig["p_value"].between(0, 1).all() and sig["p_bonferroni"].between(0, 1).all()
    assert (sig["ci_low"] <= sig["ci_high"]).all()


def test_rrf_sweep_has_every_k_for_every_kind(evaluated):
    root, _, _, _ = evaluated
    sweep = pd.read_csv(root / "results" / "rrf_k_sweep.csv")
    assert set(sweep["system"]) == {f"hybrid_k{k}" for k in RRF_K_SWEEP}
    assert len(sweep) == len(RRF_K_SWEEP) * 3


def test_ablation_contains_the_raw_and_default_variants(evaluated):
    root, _, _, _ = evaluated
    ablation = pd.read_csv(root / "results" / "ablation.csv")
    assert {"bm25_raw", "tfidf_raw", "bm25_default", "hybrid_bm25_dense"} <= set(ablation["system"])


def test_per_query_file_supports_the_explorer(evaluated):
    root, _, _, queries = evaluated
    per_query = pd.read_csv(root / "results" / "per_query.csv.gz")
    assert set(per_query["role"]) == {"dev", "test"}
    assert (per_query["first_relevant_rank"] >= 0).all()
    catalogue = pd.read_csv(root / "results" / "queries.csv")
    assert len(catalogue) == len(queries)
    assert set(catalogue.columns) >= {"qid", "text", "kind", "role", "n_relevant"}


def test_run_info_records_provenance(evaluated):
    _, info, _, _ = evaluated
    assert info["queries"]["test/question"] == N_THREADS - 4
    assert info["rrf_k"] == 60 and info["index"]["documents"] == N_THREADS * PER_THREAD
    assert {"numpy", "pandas", "torch"} <= set(info["versions"])


def test_all_figures_can_be_drawn_from_the_result_files(evaluated):
    root, _, emails, _ = evaluated
    paths = make_all_figures(root / "results", emails)
    assert len(paths) == 5
    for path in paths:
        assert path.exists() and path.read_bytes()[:4] == b"\x89PNG"


def test_evaluation_needs_both_dev_and_test_queries(evaluated, tmp_path):
    _, _, _, queries = evaluated
    engine, raw_index, _, _ = build_world()
    (tmp_path / "a").mkdir()
    with pytest.raises(ValueError, match="dev and test"):
        run_evaluation(
            engine,
            raw_index,
            [q for q in queries if q.role == "test"],
            results_dir=tmp_path / "r",
            artifacts_dir=tmp_path / "a",
            processed_dir=tmp_path,
        )
