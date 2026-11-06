"""MailLens: four ways to search the Enron e-mails, side by side.

Run with ``python -m maillens app`` (or ``streamlit run maillens/app/streamlit_app.py``).
"""

from __future__ import annotations

import html
import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # run from any working directory

import pandas as pd
import streamlit as st
from pandas.io.formats.style import Styler

from maillens import config, theme
from maillens.app import ui
from maillens.engine import SYSTEMS, Comparison, SearchEngine
from maillens.evaluation.queries import QUERIES_FILE, Query, load_queries
from maillens.snippets import best_window, highlight, query_terms

FIGURES = config.RESULTS_DIR / "figures"
EXAMPLES = [
    "Who should Katja contact about copyright for the credit article?",
    "When is the 2000 ENA Offsite scheduled?",
    "What did Joe say about who should be the lead on the EPMI tariff filing?",
    "california power market crisis",
    "ISDA master agreement",
    "socal needles window allocation",
]
SNIPPET_CHARS = 260

st.set_page_config(page_title="MailLens · Enron e-mail search", page_icon="✉️", layout="wide")
st.markdown(ui.CSS, unsafe_allow_html=True)


# ----------------------------------------------------------------------------- loading
@st.cache_resource(show_spinner="Loading the indexes and the embedding model…")
def load_engine() -> SearchEngine:
    return SearchEngine.load()


def get_engine() -> SearchEngine:
    try:
        return load_engine()
    except FileNotFoundError as err:
        st.error(
            f"A required file is missing: `{err.filename}`. Build the data first with "
            "`python -m maillens download`, `prepare`, `index` and `embed` (see the README)."
        )
        st.stop()


@st.cache_data(show_spinner=False)
def read_results(name: str) -> pd.DataFrame | None:
    path = config.RESULTS_DIR / name
    return pd.read_csv(path) if path.exists() else None


@st.cache_data(show_spinner=False)
def read_json(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def read_queries() -> dict[str, Query]:
    path = config.PROCESSED_DIR / QUERIES_FILE
    return {q.qid: q for q in load_queries(path)} if path.exists() else {}


# ----------------------------------------------------------------------------- search state
def _pick_example() -> None:
    choice = st.session_state.get("example_pick")
    if choice:
        st.session_state["query_box"] = choice
        st.session_state["active_query"] = choice


@st.cache_data(show_spinner=False, max_entries=64)
def run_comparison(
    _engine: SearchEngine,
    query: str,
    k: int,
    sender: str,
    mailbox: str,
    date_from: date | None,
    date_to: date | None,
) -> Comparison:
    allowed = _engine.mask(
        sender_contains=sender, mailbox=mailbox, date_from=date_from, date_to=date_to
    )
    return _engine.compare(query, k, allowed=allowed)


def sidebar(engine: SearchEngine) -> dict[str, object]:
    summary = engine.summary()
    st.sidebar.markdown(f"### {summary.emails:,} e-mails")
    st.sidebar.caption(
        f"{summary.threads:,} threads · {summary.mailboxes} mailboxes · "
        f"{summary.first:%Y}–{summary.last:%Y}"
    )
    st.sidebar.markdown("#### Filters")
    sender = st.sidebar.text_input("Sender contains", placeholder="e.g. skilling")
    mailboxes = sorted(engine.emails["mailbox"].unique())
    mailbox = st.sidebar.selectbox("Mailbox", ["Any", *mailboxes])
    first, last = summary.first.date(), summary.last.date()
    span = st.sidebar.date_input(
        "Sent between", value=(first, last), min_value=first, max_value=last
    )
    date_from = date_to = None
    if isinstance(span, tuple) and len(span) == 2 and span != (first, last):
        date_from, date_to = span
    st.sidebar.markdown("#### Display")
    k = st.sidebar.slider("Results per ranker", 3, 10, 5)
    fold = st.sidebar.toggle("Fold same-thread results", help="Show one result per conversation.")
    st.sidebar.caption(f"RRF constant k = {engine.rrf_k}")
    return {
        "sender": sender,
        "mailbox": "" if mailbox == "Any" else mailbox,
        "date_from": date_from,
        "date_to": date_to,
        "k": k,
        "fold": fold,
    }


# ----------------------------------------------------------------------------- result cards
def snippet_for(engine: SearchEngine, system: str, query: str, doc_id: int, terms: set[str]) -> str:
    text = str(engine.emails["text"].iloc[doc_id])
    if system == "dense":
        text, _ = engine.best_passage(query, doc_id)
    window = best_window(text, terms, engine.analyzer, width=SNIPPET_CHARS)
    return highlight(window, terms, engine.analyzer)


def explain(
    engine: SearchEngine, comparison: Comparison, system: str, query: str, doc_id: int
) -> None:
    if system in ("tfidf", "bm25"):
        parts = engine.explain_terms(query, system, doc_id)
        if not parts:
            st.caption("No query term occurs in this e-mail.")
            return
        frame = pd.DataFrame(
            {
                "term (stem)": [p.term for p in parts],
                "times in e-mail": [p.tf for p in parts],
                "e-mails containing it": [p.df for p in parts],
                "score share": [p.contribution for p in parts],
            }
        )
        st.dataframe(
            frame.head(8),
            hide_index=True,
            column_config={"score share": st.column_config.NumberColumn(format="%.3f")},
        )
        st.caption(
            "Rare terms count for more (idf); repeating a term helps less and less."
            if system == "bm25"
            else "Cosine similarity: each term's weight in the query times its weight in the e-mail."
        )
    elif system == "dense":
        passage, similarity = engine.best_passage(query, doc_id)
        st.caption(f"Best-matching passage · cosine similarity {similarity:.3f}")
        st.markdown(
            f'<div class="ml-snippet" style="margin-left:0">{html.escape(passage[:600])}</div>',
            unsafe_allow_html=True,
        )
    else:
        fused = next((h for h in comparison.fused if h.doc_id == doc_id), None)
        rows = []
        for name in ("tfidf", "bm25", "dense"):
            rank = comparison.ranks[name].get(doc_id)
            rows.append(
                {
                    "ranker": theme.SYSTEM_SHORT[name],
                    "its rank": f"#{rank}" if rank else f"not in top {engine.depth}",
                    f"1 / ({engine.rrf_k} + rank)": (1 / (engine.rrf_k + rank)) if rank else 0.0,
                }
            )
        st.dataframe(
            pd.DataFrame(rows),
            hide_index=True,
            column_config={
                f"1 / ({engine.rrf_k} + rank)": st.column_config.NumberColumn(format="%.4f")
            },
        )
        if fused:
            st.caption(f"Fused score {fused.score:.4f} = the three shares added together.")


def render_column(
    engine: SearchEngine,
    comparison: Comparison,
    system: str,
    query: str,
    terms: set[str],
    fold: bool,
) -> None:
    st.markdown(ui.column_header_html(system), unsafe_allow_html=True)
    hits = comparison.per_ranker[system]
    thread_ids = engine.emails["thread_id"].to_numpy()
    shown = (
        ui.collapse_threads(hits, lambda d: int(thread_ids[d])) if fold else [(h, 0) for h in hits]
    )
    if not shown:
        st.caption("No e-mail contains any of the query terms.")
        return
    for position, (hit, hidden) in enumerate(shown, start=1):
        row = engine.email(hit.doc_id)
        ranks = {s: comparison.ranks[s].get(hit.doc_id) for s in SYSTEMS}
        st.markdown(
            ui.card_html(
                rank=position if fold else hit.rank,
                subject=str(row["subject"]),
                sender=str(row["sender"]),
                when=row["date"],
                thread_size=int(row["thread_size"]),
                snippet_html=snippet_for(engine, system, query, hit.doc_id, terms),
                ranks=ranks,
                current=system,
            ),
            unsafe_allow_html=True,
        )
        if hidden:
            st.markdown(
                f'<div class="ml-more">+ {hidden} more from this thread</div>',
                unsafe_allow_html=True,
            )
        with st.expander("Why this rank?"):
            explain(engine, comparison, system, query, hit.doc_id)
        if st.button("Open e-mail and thread", key=f"open-{system}-{hit.doc_id}", width="stretch"):
            show_email(hit.doc_id, query)


@st.dialog("E-mail", width="large")
def show_email(doc_id: int, query: str) -> None:
    engine = get_engine()
    row = engine.email(doc_id)
    terms = query_terms(query, engine.analyzer)
    recipients = ", ".join(row["recipients"][:8]) + (" …" if len(row["recipients"]) > 8 else "")
    st.markdown(
        f'<div class="ml-subject" style="font-size:1.15rem">{html.escape(str(row["subject"]) or "(no subject)")}</div>'
        f'<div class="ml-meta" style="margin-left:0">From {html.escape(str(row["sender"]))} · '
        f"{row['date']:%d %b %Y %H:%M} UTC · mailbox {html.escape(str(row['mailbox']))}"
        f" / {html.escape(str(row['folder']))}<br>To {html.escape(recipients) or '—'}</div>",
        unsafe_allow_html=True,
    )
    body = highlight(str(row["text"]) or "(empty message)", terms, engine.analyzer)
    st.markdown(f'<div class="ml-body">{body}</div>', unsafe_allow_html=True)
    thread = engine.thread(int(row["thread_id"]))
    if len(thread) > 1:
        st.markdown(f"**Conversation · {len(thread)} messages**")
        view = pd.DataFrame(
            {
                "": ["▶" if d == doc_id else "" for d in thread["doc_id"]],
                "sent": thread["date"].dt.strftime("%d %b %Y %H:%M"),
                "from": thread["sender"],
                "subject": thread["subject"],
            }
        )
        st.dataframe(view, hide_index=True, height=min(35 * len(view) + 40, 320))


# ----------------------------------------------------------------------------- tabs
def render_search(engine: SearchEngine, options: dict[str, object]) -> None:
    st.markdown('<div class="ml-brand">MailLens</div>', unsafe_allow_html=True)
    summary = engine.summary()
    st.markdown(
        f'<div class="ml-tagline">Four ways to search {summary.emails:,} Enron e-mails, '
        "side by side. Ask a question or type keywords.</div>",
        unsafe_allow_html=True,
    )
    st.pills(
        "Try an example",
        EXAMPLES,
        key="example_pick",
        on_change=_pick_example,
        label_visibility="collapsed",
    )
    with st.form("search_form", border=False):
        box, button = st.columns([7, 1], vertical_alignment="bottom")
        box.text_input(
            "Search",
            key="query_box",
            placeholder="e.g. Who should Katja contact about copyright?",
            label_visibility="collapsed",
        )
        submitted = button.form_submit_button("Search", type="primary", width="stretch")
    if submitted:
        st.session_state["active_query"] = st.session_state.get("query_box", "").strip()

    query = st.session_state.get("active_query", "")
    if not query:
        st.info("Pick an example above or type a query to compare the four rankers.")
        return
    comparison = run_comparison(
        engine,
        query,
        int(options["k"]),  # type: ignore[call-overload]
        str(options["sender"]),
        str(options["mailbox"]),
        options["date_from"],  # type: ignore[arg-type]
        options["date_to"],  # type: ignore[arg-type]
    )
    top_k = int(options["k"])  # type: ignore[call-overload]
    st.markdown(
        ui.agreement_html(
            {s: [h.doc_id for h in comparison.per_ranker[s]] for s in SYSTEMS}, top_k
        ),
        unsafe_allow_html=True,
    )
    terms = query_terms(query, engine.analyzer)
    for column, system in zip(st.columns(4, gap="medium"), theme.TABLE_ORDER, strict=True):
        with column:
            render_column(engine, comparison, system, query, terms, bool(options["fold"]))


def _best_in_column(frame: pd.DataFrame) -> Styler:
    return frame.style.highlight_max(
        subset=[c for c in frame.columns if c not in ("Method", "queries")],
        props="font-weight:700;background-color:#eaf2fc",
    )


def render_results_table(kind: str) -> None:
    main = read_results("metrics_main.csv")
    if main is None:
        return
    data = main[main["kind"] == kind].set_index("system").reindex(list(theme.TABLE_ORDER))
    frame = pd.DataFrame(
        {
            "Method": [theme.SYSTEM_LABELS[s] for s in data.index],
            "P@10": data["P@10"].to_numpy(),
            "R@100": data["R@100"].to_numpy(),
            "MAP": data["MAP"].to_numpy(),
            "nDCG@10": data["nDCG@10"].to_numpy(),
            "MRR": data["MRR"].to_numpy(),
        }
    )
    st.dataframe(
        _best_in_column(frame).format({c: "{:.3f}" for c in frame.columns if c != "Method"}),
        hide_index=True,
    )
    st.caption(f"{int(data['queries'].iloc[0]):,} test queries. Best value per column in bold.")


def render_findings() -> None:
    main, sig = read_results("metrics_main.csv"), read_results("significance.csv")
    if main is None or sig is None:
        return
    lines = []
    for kind in theme.KIND_LABELS:
        sub = main[main["kind"] == kind].set_index("system")
        best = sub["MAP"].idxmax()
        row = sig[
            (sig["kind"] == kind)
            & (sig["metric"] == "MAP")
            & (sig["system_a"] == "hybrid")
            & (sig["system_b"] == "bm25")
        ]
        note = ""
        if not row.empty:
            r = row.iloc[0]
            verdict = "significant" if r["p_bonferroni"] < 0.05 else "not significant"
            note = (
                f" Hybrid RRF − BM25 on MAP: {r['diff']:+.3f} "
                f"(95% CI {r['ci_low']:+.3f} to {r['ci_high']:+.3f}, {verdict})."
            )
        lines.append(
            f"- **{theme.KIND_LABELS[kind]}:** best MAP is {theme.SYSTEM_LABELS[best]} "
            f"({sub.loc[best, 'MAP']:.3f}).{note}"
        )
    st.markdown("\n".join(lines))


def show_figure(name: str, caption: str) -> None:
    path = FIGURES / name
    if path.exists():
        st.image(str(path), caption=caption, width="stretch")
    else:
        st.caption(f"{name} has not been generated yet (`python -m maillens figures`).")


def render_explorer(engine: SearchEngine) -> None:
    st.subheader("Query explorer")
    st.caption("Pick a test query and see where each ranker puts the e-mail(s) it should find.")
    catalogue = read_results("queries.csv")
    queries = read_queries()
    if catalogue is None or not queries:
        st.info("Run `python -m maillens queries` and `evaluate` to enable the explorer.")
        return
    test = catalogue[catalogue["role"] == "test"]
    left, right = st.columns([1, 3])
    kind = left.radio("Query type", list(theme.KIND_LABELS), format_func=theme.KIND_SHORT.get)
    pool = test[test["kind"] == kind]
    needle = right.text_input("Filter the list", placeholder="type to narrow the queries")
    if needle:
        pool = pool[pool["text"].str.contains(needle, case=False, regex=False)]
    if pool.empty:
        st.caption("No query matches that filter.")
        return
    qid = right.selectbox(
        f"{len(pool):,} queries",
        pool["qid"].tolist(),
        format_func=lambda q: str(pool.set_index("qid").loc[q, "text"]),
    )
    query = queries[qid]
    result = engine.compare(query.text, 10)
    st.markdown(f"**{html.escape(query.text)}** · {len(query.relevant)} relevant e-mail(s)")
    for column, system in zip(st.columns(4), theme.TABLE_ORDER, strict=True):
        with column:
            st.markdown(ui.column_header_html(system), unsafe_allow_html=True)
            ranks = [result.ranks[system][d] for d in query.relevant if d in result.ranks[system]]
            st.markdown(
                f"First relevant e-mail: **{'#' + str(min(ranks)) if ranks else 'not in the top ' + str(engine.depth)}**"
            )
            for hit in result.per_ranker[system]:
                row = engine.email(hit.doc_id)
                mark = "✓" if hit.doc_id in query.relevant else "·"
                css = "ml-relevant" if mark == "✓" else ""
                st.markdown(
                    f'<div class="ml-meta {css}" style="margin:0.1rem 0">{mark} {hit.rank}. '
                    f"{html.escape(str(row['subject'])[:48] or '(no subject)')}</div>",
                    unsafe_allow_html=True,
                )


def render_evaluation(engine: SearchEngine) -> None:
    info_path = config.RESULTS_DIR / "run_info.json"
    if read_results("metrics_main.csv") is None or not info_path.exists():
        st.info("The evaluation has not been run yet: `python -m maillens evaluate`.")
        return
    info = read_json(str(info_path))
    counts = info["queries"]
    st.markdown(
        f"Evaluated on **{counts.get('test/question', 0):,} questions**, "
        f"**{counts.get('test/subject', 0):,} subject lines** and "
        f"**{counts.get('test/keyword', 0):,} keyword queries** that were never used for tuning "
        f"(BM25 k1 = {info['bm25_tuned']['k1']}, b = {info['bm25_tuned']['b']} chosen on separate "
        f"dev queries)."
    )
    render_findings()
    tabs = st.tabs([theme.KIND_SHORT[k] for k in theme.KIND_LABELS])
    for tab, kind in zip(tabs, theme.KIND_LABELS, strict=True):
        with tab:
            render_results_table(kind)
    left, right = st.columns(2)
    with left:
        show_figure("precision_recall.png", "Mean interpolated precision at each recall level")
        show_figure("success_at_k.png", "Share of questions answered within the top k")
    with right:
        show_figure("metrics_bars.png", "MAP and nDCG@10 with 95% confidence intervals")
        show_figure("rrf_k_sweep.png", "Effect of the RRF constant k")
    with st.expander("Paired significance tests"):
        sig = read_results("significance.csv")
        if sig is not None:
            st.dataframe(sig, hide_index=True)
    with st.expander("Ablations: cleaning, BM25 tuning, which rankers to fuse"):
        ablation = read_results("ablation.csv")
        if ablation is not None:
            st.dataframe(ablation, hide_index=True)
    render_explorer(engine)


PIPELINE_DOT = f"""
digraph G {{
  rankdir=LR; bgcolor="transparent"; nodesep=0.35; ranksep=0.5;
  node [shape=box, style="rounded,filled", fillcolor="#ffffff", color="{theme.BASELINE}",
        fontname="Helvetica", fontsize=11, fontcolor="{theme.INK}", margin="0.18,0.1"];
  edge [color="{theme.INK_MUTED}", arrowsize=0.7];
  raw [label="Enron dump\\n517,401 files"];
  prep [label="De-duplicate · clean\\nthread · sample"];
  query [label="Query", fillcolor="#eef4fc"];
  an [label="Analyzer\\ntokens · stop-words · stems"];
  inv [label="Inverted index"];
  tfidf [label="TF-IDF", color="{theme.SYSTEM_COLORS["tfidf"]}", penwidth=2];
  bm25 [label="BM25", color="{theme.SYSTEM_COLORS["bm25"]}", penwidth=2];
  emb [label="Sentence-Transformer\\nchunk embeddings"];
  dense [label="Dense (MaxP)", color="{theme.SYSTEM_COLORS["dense"]}", penwidth=2];
  rrf [label="Reciprocal rank fusion", color="{theme.SYSTEM_COLORS["hybrid"]}", penwidth=2];
  out [label="Ranked e-mails", fillcolor="#eef4fc"];
  raw -> prep -> an -> inv; prep -> emb;
  query -> tfidf; query -> bm25; query -> dense;
  inv -> tfidf; inv -> bm25; emb -> dense;
  tfidf -> rrf; bm25 -> rrf; dense -> rrf; rrf -> out;
}}
"""


def render_about(engine: SearchEngine) -> None:
    st.markdown("## How it works")
    st.graphviz_chart(PIPELINE_DOT, width="stretch")
    left, right = st.columns(2)
    with left:
        st.markdown("**TF-IDF** scores an e-mail by the cosine between sub-linear TF-IDF vectors.")
        st.latex(
            r"w_{t,d} = (1+\ln \mathrm{tf}_{t,d}) \cdot \left(\ln\tfrac{1+N}{1+\mathrm{df}_t}+1\right)"
        )
        st.markdown("**BM25** adds term-frequency saturation and document-length normalisation.")
        st.latex(
            r"\sum_{t\in q} \mathrm{idf}_t \cdot "
            r"\frac{\mathrm{tf}\,(k_1+1)}{\mathrm{tf}+k_1\,(1-b+b\,|d|/\overline{|d|})}"
        )
    with right:
        st.markdown(
            "**Dense retrieval** embeds overlapping chunks and scores an e-mail by its best chunk."
        )
        st.latex(r"\mathrm{score}(d)=\max_{c\in d}\ \cos\big(E(q),E(c)\big)")
        st.markdown(
            "**Reciprocal rank fusion** needs ranks only, so the scores never need aligning."
        )
        st.latex(rf"\mathrm{{RRF}}(d)=\sum_m \frac{{1}}{{{engine.rrf_k}+\mathrm{{rank}}_m(d)}}")
    stats_path = config.PROCESSED_DIR / "corpus_stats.json"
    if stats_path.exists():
        stats = read_json(str(stats_path))
        st.markdown("#### The corpus")
        st.markdown(
            f"The raw dump holds **{stats['raw_rows']:,}** files, but **{stats['duplicate_content']:,}** "
            f"of them are copies of another e-mail filed in a different folder (they carry different "
            f"Message-IDs, so only the content gives them away). After removing copies and "
            f"{stats['dropped_implausible_date']:,} messages with impossible dates, "
            f"**{stats['unique_messages']:,}** distinct e-mails remain; this app searches a "
            f"sample of **{stats['sample_messages']:,}** of them in **{stats['sample_threads']:,}** "
            "conversations."
        )
    figure = FIGURES / "corpus.png"
    if figure.exists():
        st.image(str(figure), width="stretch")
    st.markdown(
        "#### Limits to keep in mind\n"
        "- Enron has no official relevance labels. The evaluation uses labels derived from "
        "a published question set and from conversation threads, which favour known-item lookups.\n"
        "- Thread reconstruction is a heuristic (subject, people, time) because the dump has no "
        "reply headers.\n"
        "- Dense retrieval embeds the first ~500 words of each e-mail."
    )


def main() -> None:
    engine = get_engine()
    options = sidebar(engine)
    search_tab, eval_tab, about_tab = st.tabs(["Search", "Evaluation", "How it works"])
    with search_tab:
        render_search(engine, options)
    with eval_tab:
        render_evaluation(engine)
    with about_tab:
        render_about(engine)


main()
