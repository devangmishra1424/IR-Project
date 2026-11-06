# MailLens walkthrough

A guide to the project for the presentation and the viva: what every part does, why it was built
that way, and small numeric examples you can check by hand. Every number below that comes from
the real data is reproducible with the commands in the README.

## 1. The project in thirty seconds

An e-mail search engine over 50,002 real Enron e-mails. The same query is answered by four
retrieval models side by side, and the project measures which one finds the right e-mail:

| Ranker | Idea | One-line summary |
|---|---|---|
| TF-IDF | vector space | match words, weight rare words more, compare by cosine |
| BM25 | probabilistic | like TF-IDF, but repeating a word helps less and long e-mails are not favoured |
| Sentence-Transformer | meaning | turn text into a vector of 384 numbers; near vectors mean similar meaning |
| Hybrid RRF | fusion | merge the three rank lists using only ranks |

## 2. The pipeline, step by step

```
download -> prepare -> index -> embed -> queries -> evaluate -> figures -> app
```

| Step | What happens | Code |
|---|---|---|
| download | two public Hugging Face datasets: 517,401 e-mails and a set of questions with their answer e-mails | `maillens/prep/download.py` |
| prepare | remove duplicates, clean bodies, rebuild threads, sample 50,002 e-mails | `maillens/prep/` |
| index | analyse the text and build the inverted index | `maillens/index/inverted.py`, `maillens/pipeline.py` |
| embed | split each e-mail into at most four chunks and embed every chunk (about 35 minutes on a laptop CPU) | `maillens/rankers/dense.py` |
| queries | build the test questions and their relevant e-mails | `maillens/evaluation/queries.py` |
| evaluate | tune on dev queries, measure on test queries, run significance tests and ablations | `maillens/evaluation/experiments.py` |
| app | the Streamlit interface | `maillens/app/` |

## 3. The data decisions, and why

**The dump contains the same e-mail many times.** 517,401 files, but 263,553 of them are copies of
another e-mail: the same message filed in several folders (`all_documents`,
`discussion_threads`, `sent`, ...) or delivered to several mailboxes.
The copies have *different* Message-IDs, so only the content (sender, minute, subject, a hash of
the body) reveals them. Left in, they would fill result lists with the same e-mail and make
retrieving an identical copy of the answer count as a miss. After removing them, and 605 e-mails
whose date is impossible (1980, 2043, ...), 253,243 distinct e-mails remain.

**Cleaning separates what the sender wrote from what they quoted.** A body is split into `fresh`
text and `quoted` text (replies and forwards, including the Lotus Notes layouts this corpus is full
of), and legal footers, URLs, HTML, and mail-routing noise such as `Kay Mann/Corp/Enron@ENRON` are
removed. The quoted part is *kept*, not thrown away, because a forwarded article is often the real
content of a message.

**Threads are rebuilt, because the dump has no reply headers.** Two e-mails join a thread when they
share a normalised subject (no `Re:`/`Fw:`), were sent within 30 days of the thread's latest message,
and share a participant. Generic subjects ("hi", empty) never thread, and a thread stops growing at
500 messages.

**The sample is built from whole threads.** Sampling single e-mails would cut every conversation
into pieces; sampling threads keeps them intact and keeps the same mix of single messages and long
conversations. Every e-mail that a test question is about is always included.

**Why 50,002 and not all 253,243.** The sentence-transformer runs at about 40 chunks per second on
the 6-core laptop CPU used here. The plan for 100,000 e-mails would have taken over an hour; 50,002
e-mails with at most four chunks each takes about half an hour and still contains every question's
answer. Absolute scores on a bigger corpus would be lower; the comparison between rankers is what
this project measures.

## 4. The four rankers with numbers you can check

### The inverted index

For each term, a list of the e-mails that contain it and how many times. The lists of all terms
are stored back to back in two flat arrays (`doc_ids`, `tfs`) plus an offsets array that says
where each term's list starts, so a lookup is two array reads.

### BM25

```
score(q, d) = sum over query terms t of  idf(t) * tf * (k1 + 1) / (tf + k1 * (1 - b + b * |d| / avgdl))
idf(t)      = ln(1 + (N - df + 0.5) / (df + 0.5))
```

Worked example (this exact case is a unit test). Three tiny documents `a a b`, `b c`,
`c c c d`; query `a`; k1 = 1.2, b = 0.75.

- N = 3, df(a) = 1, so idf = ln(1 + 2.5 / 1.5) = 0.9808
- average length = (3 + 2 + 4) / 3 = 3
- document 0 has tf = 2 and length 3: length part = 1.2 * (0.25 + 0.75 * 3/3) = 1.2
- term part = 2 * 2.2 / (2 + 1.2) = 1.375
- score = 0.9808 * 1.375 = **1.3486**

`k1` controls how fast repeating a word stops helping; `b` controls how much a long document is
penalised. They were tuned on dev queries only (see the results section).

### TF-IDF

```
weight(t, d) = (1 + ln tf) * (ln((1 + N) / (1 + df)) + 1)      document and query vectors are length-normalised
score(q, d)  = cosine(query vector, document vector)
```

On the same toy corpus: idf(a) = ln(4/2) + 1 = 1.6931; document 0 contains `a` twice and `b` once
(df(b) = 2, idf = 1.2877). Raw weights: a = (1 + ln 2) * 1.6931 = 2.8667, b = 1.2877. Length = 3.1426,
so document 0's weight for `a` is 0.9122. The query `a` normalises to 1, hence the score is **0.9122**.
This implementation is tested against scikit-learn's `TfidfVectorizer` on a small corpus.

### Dense retrieval

A long e-mail does not fit the model's input, so each e-mail is cut into overlapping windows of
140 words (at most four per e-mail), each window gets the subject line as a prefix, and every
window is embedded. An e-mail's score is its best window: `max over chunks of cosine(query, chunk)`.
Search is one matrix-vector product over 85,852 chunk vectors.

### Reciprocal rank fusion

```
RRF(d) = sum over rankers m of 1 / (k + rank_m(d))        k = 60
```

Worked example (also a unit test). List A = [d1, d2, d3] and list B = [d3, d1, d4]:
d1 = 1/61 + 1/62 = 0.03252, d3 = 1/63 + 1/61 = 0.03227, d2 = 1/62 = 0.01613, d4 = 1/63 = 0.01587,
so the fused order is d1, d3, d2, d4. Only ranks are used, so the raw scores of TF-IDF, BM25 and
cosine never need to be put on a common scale.

## 5. How the evaluation works

**No official relevance labels exist for Enron.** Two label sources are used instead:

1. **Questions.** A published set of natural-language questions, each with the e-mail that answers
   it. 1,515 of them are the test queries; 300 more, drawn from a different part of the set, are the
   dev queries used only for tuning.
2. **Threads.** For 300 test threads (and 100 dev threads), a *subject-line query* and a *keyword
   query* (four characteristic words that are not in the subject). Every e-mail of the thread is
   relevant. Both queries are built for the same threads, so they are directly comparable.

**Dev and test are separate.** BM25's `k1` and `b` were chosen on the dev queries; all reported
numbers come from the test queries, which were never used for any choice.

**Metrics**, with the example ranking [1, 2, 3, 4, 5] where documents 2 and 5 are relevant:

| Metric | Meaning | Example |
|---|---|---|
| P@10 | share of the top 10 that is relevant | 2 of 5 returned: 0.4 at depth 5 |
| R@100 | share of all relevant e-mails found in the top 100 | both found: 1.0 |
| AP / MAP | mean of the precision at each relevant hit (MAP: averaged over queries) | (1/2 + 2/5) / 2 = 0.45 |
| nDCG@10 | like AP but with a log discount on rank | 0.624 |
| MRR | 1 / rank of the first relevant e-mail | 1/2 = 0.5 |

**Significance.** Differences are tested on the per-query scores with a paired bootstrap (95%
confidence interval) and a sign-flip randomisation test, with a Bonferroni correction for the six
pairs compared.

**A ceiling for P@10.** With one relevant e-mail, P@10 cannot exceed 0.1, so a low P@10 on questions
is not a failure; `results/ceilings.csv` gives the ceiling for each query kind.

## 6. What was measured

Test queries only: 1,515 questions, 300 subject-line queries and 300 keyword queries. MAP:

| | TF-IDF | BM25 | Sentence-Transformer | Hybrid RRF |
|---|---:|---:|---:|---:|
| Questions | 0.512 | **0.787** | 0.603 | 0.759 |
| Subject lines | 0.602 | **0.891** | 0.775 | 0.863 |
| Keyword queries | 0.509 | **0.755** | 0.138 | 0.536 |

The other metrics, the confidence intervals and the significance tests are in `results/` and the
README. Of the 18 pairwise MAP comparisons, 17 are significant after the Bonferroni correction;
the exception is the hybrid against TF-IDF on keyword queries (+0.028, 95% interval -0.001 to
+0.056). On nDCG@10 the gap between the hybrid and BM25 on subject lines is also not significant
(-0.014, p = 0.15).

## 7. Reading the results

What the data shows, and where an explanation is a hypothesis rather than a measurement:

1. **BM25 is the best single ranker on every query kind.** Measured. It is also the only ranker that
   is never far from the best.
2. **TF-IDF is the weakest lexical ranker, and e-mail length explains much of it.** Measured: on
   questions its MRR falls from 0.651 when the answer e-mail has at most 50 words to 0.379 when it has
   over 400, while BM25 stays between 0.73 and 0.82 (`results/length_analysis.csv`, written by
   `python -m maillens analyze`). Cosine normalisation
   divides by the length of the whole e-mail, so a long forward is penalised for being long.
3. **The dense ranker is respectable on natural language and weak on keyword lists.** Measured: MAP
   0.603 on questions, 0.775 on subject lines, 0.138 on keyword queries. A list of four loose words is
   not a sentence the encoder was trained on; names and rare terms are what BM25 handles best.
   (The second half is the usual explanation, not something this project isolates.)
4. **Equal-weight fusion does not beat BM25 here.** Measured: MAP is lower on all three kinds. Fusion
   assumes the lists are about equally good; here two of the three are much weaker, and RRF gives
   each the same vote. What fusion does buy is recall: Recall@100 is the highest for the hybrid on
   questions (0.984 against 0.982) and subject lines (0.983 against 0.977).
5. **RRF's constant is not a free choice.** Measured: on keyword queries MAP is 0.636 at k = 5, 0.536
   at k = 60 and 0.394 at k = 500; on questions 0.783, 0.759 and 0.749. A smaller k lets each list's
   top ranks dominate, which favours the strongest list.
6. **Counting BM25 three times closes most of the gap.** Measured, exploratory (the weights were not
   tuned): 0.791 on questions (BM25 0.787) and 0.890 on subject lines (0.891), but 0.654 on keywords.
7. **Cleaning shrinks the index and is neutral for questions.** Measured: 111,205 terms against
   142,413 on raw text. BM25 MAP is 0.787 against 0.783 (questions) and 0.755 against 0.757
   (keywords), but 0.891 against 0.919 on subject lines. A probable reason, not tested here: the
   quoted headers the cleaner removes repeat the subject line, which helps find replies.
8. **Tuning helps some query kinds.** Measured: BM25 with `k1 = 0.3` against the textbook 1.2 scores
   0.891 against 0.862 on subject lines and 0.755 against 0.736 on keywords, but 0.787 against 0.794
   on questions. `k1` was chosen on dev queries, so this is an honest, slightly mixed outcome.

## 8. Likely questions

**Why does BM25 beat TF-IDF?** Both reward rare words, but TF-IDF's cosine divides by the whole
document length and counts every repetition, while BM25 saturates repeated terms and normalises
length only partly (`b`). E-mails quote each other, so repetition and length are noisy signals.

**What does the inverted index buy you?** A query only touches the postings of its own terms
instead of scanning 50,002 e-mails: two array slices per term, and a score accumulator per e-mail.

**Why `k = 60` in RRF, and does it matter?** 60 is the value suggested in the original paper. Here it
matters on keyword queries (0.536 against 0.636 at k = 5), so the sweep is reported.

**Why does the hybrid not beat BM25?** See 7.4 and 7.6: the weaker lists get an equal vote. The
experiment that weights BM25 more shows the hybrid can match it, not exceed it, on these queries.

**How did you get relevance judgements without labels?** Two automatic sources: a published set of
questions with their answer e-mails, and conversation threads (every e-mail of a thread is relevant
to its subject-line and keyword queries). Both favour known-item lookups, which is a stated limit.

**How do you know you did not tune on the test set?** BM25's parameters were chosen on 500 separate dev
queries; the test queries were only used for the reported numbers. Nothing else was tuned.

**What are the metrics?** P@10 (precision in the top 10), R@100, MAP (mean of the precision at each
relevant hit), nDCG@10 (graded by rank), MRR (1 / rank of the first relevant e-mail). See section 5.

**Why is P@10 so low for questions?** A question usually has one relevant e-mail, so P@10 cannot
exceed 0.1; averaged over questions the ceiling is 0.13 and BM25 reaches 0.115.

**Why chunk the e-mails for the dense ranker?** The encoder reads about 256 word pieces. Long e-mails
are cut into overlapping 140-word windows (at most four) and an e-mail scores as its best window.

**Why remove duplicates by content?** The dump stores a message once per folder or mailbox with a new
ID each time; half the files were copies. They would fill result lists and make an identical copy of
the answer count as a miss.

**How is TF-IDF checked for correctness?** A unit test compares its scores with scikit-learn's
`TfidfVectorizer` on a small corpus, and another with a hand-computed value (section 4).

**What would you do next?** Weight the fusion by query type (keywords go to BM25), re-rank the top
results with a cross-encoder, embed the whole e-mail rather than its first four chunks, and score
fields (subject, own text, quoted text) separately as in BM25F.

## 9. A demo script (about four minutes)

1. Open the app (`python -m maillens app`). Point at the sidebar: 50,002 e-mails, filters.
2. Click the example *Who should Katja contact about copyright for the credit article?* The four
   columns answer the same query; the chips on each result say where the *other* rankers put it, and
   the line above the columns shows how much their top lists agree.
3. Open *Why this rank?* on a BM25 result (term shares), on the dense result (the matched passage and
   its cosine) and on the hybrid result (the three RRF shares).
4. Try *What did the Governor do regarding our scheduled meeting?*: the dense ranker finds the answer
   at rank 1 where BM25 has it at rank 10. Then *What is the fee TD is charging us for the new option
   structure?*: BM25 has it at rank 1, the dense ranker at rank 34. Neither ranker wins everywhere.
5. Open the *Evaluation* tab: the results table, the precision-recall curves, and the query explorer
   (pick a test question and see where each ranker puts the right e-mail).
6. Open *How it works*: the pipeline and the formulas.
7. Close with the finding: BM25 is the strongest single ranker here, fusion helps recall but not
   precision at rank 1, and the limits (automatic labels, a 50,002-e-mail sample).
