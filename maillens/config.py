"""Paths and constants shared across the pipeline.

Everything large or reproducible lives under ``data/`` and ``artifacts/`` (both git-ignored).
Set ``MAILLENS_HOME`` to relocate the whole tree (the tests do this).
"""

from __future__ import annotations

import os
from pathlib import Path

# huggingface_hub reads these when first imported; make the network timeouts explicit.
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "30")

ROOT = Path(os.environ.get("MAILLENS_HOME", Path(__file__).resolve().parents[1]))
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = ROOT / "artifacts"
RESULTS_DIR = ROOT / "results"

# Public Hugging Face datasets (downloaded anonymously, only the parquet files below).
EMAILS_REPO = "corbt/enron-emails"
QUESTIONS_REPO = "corbt/enron_emails_sample_questions"
EMAIL_SHARDS = tuple(f"data/train-0000{i}-of-00003.parquet" for i in range(3))
QUESTION_FILES = ("data/train-00000-of-00001.parquet", "data/test-00000-of-00001.parquet")

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

SEED = 42
SAMPLE_TARGET = 100_000  # e-mails in the evaluation corpus (see README for why)

# Retrieval defaults.
BM25_K1 = 1.2
BM25_B = 0.75
RRF_K = 60  # the constant suggested by Cormack et al. (2009)
EVAL_DEPTH = 1000  # ranked-list depth used for MAP / recall
UI_DEPTH = 200  # candidates fused per ranker in the search app

# Dense retrieval: long e-mails are split into overlapping word windows (MaxP).
CHUNK_WORDS = 140
CHUNK_STRIDE = 120
MAX_CHUNKS_PER_EMAIL = 8

# Bounds on pathological input (some Enron "e-mails" are megabytes of spreadsheet dump).
MAX_BODY_CHARS = 20_000
