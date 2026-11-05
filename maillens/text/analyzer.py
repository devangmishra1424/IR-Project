"""The analysis chain applied to documents and queries alike.

lower-case -> split on non-alphanumerics -> drop stop-words -> Snowball (Porter2) stem.

Documents and queries MUST go through the same chain, otherwise a query term can never match
an indexed term. The chain is deliberately small so it can be explained in one breath.
"""

from __future__ import annotations

import re
from functools import lru_cache

from nltk.stem.snowball import SnowballStemmer

from maillens.text.stopwords import ENGLISH_STOPWORDS

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STEM_CACHE_SIZE = 500_000  # bounded: the vocabulary of 100k e-mails is well below this


class Analyzer:
    """Turn raw text into index terms."""

    def __init__(
        self,
        *,
        stem: bool = True,
        remove_stopwords: bool = True,
        max_token_len: int = 30,
    ) -> None:
        if max_token_len < 1:
            raise ValueError("max_token_len must be >= 1")
        self._stemmer = SnowballStemmer("english") if stem else None
        self._stopwords = ENGLISH_STOPWORDS if remove_stopwords else frozenset()
        self._max_len = max_token_len
        # Per-instance cache: stemming a handful of unique words once beats stemming 15M tokens.
        self._stem_cached = lru_cache(maxsize=_STEM_CACHE_SIZE)(self._stem_uncached)

    def _stem_uncached(self, token: str) -> str:
        return self._stemmer.stem(token) if self._stemmer is not None else token

    def stem(self, token: str) -> str:
        """Stem a single lower-case token (identity when stemming is off)."""
        return self._stem_cached(token)

    def raw_tokens(self, text: str) -> list[str]:
        """Lower-cased alphanumeric tokens, before stop-word removal and stemming."""
        return [t for t in _TOKEN_RE.findall(text.lower()) if len(t) <= self._max_len]

    def __call__(self, text: str) -> list[str]:
        return [self.stem(t) for t in self.raw_tokens(text) if t not in self._stopwords]
