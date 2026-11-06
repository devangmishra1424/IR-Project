"""Metadata filters for the search UI, expressed as a boolean mask over document ids."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd


def build_mask(
    docs: pd.DataFrame,
    *,
    sender_contains: str = "",
    mailbox: str = "",
    date_from: date | None = None,
    date_to: date | None = None,
) -> np.ndarray | None:
    """Boolean mask of documents that pass every given filter; None when no filter is active.

    ``docs`` must have the columns ``sender`` (str), ``mailbox`` (str) and ``date``
    (timezone-aware UTC timestamps). Both date bounds are inclusive.
    """
    mask = np.ones(len(docs), dtype=bool)
    active = False
    if sender_contains.strip():
        needle = sender_contains.strip()
        mask &= docs["sender"].str.contains(needle, case=False, regex=False, na=False).to_numpy()
        active = True
    if mailbox.strip():
        mask &= (docs["mailbox"] == mailbox.strip()).to_numpy()
        active = True
    if date_from is not None:
        mask &= (docs["date"] >= pd.Timestamp(date_from, tz="UTC")).to_numpy()
        active = True
    if date_to is not None:
        end = pd.Timestamp(date_to, tz="UTC") + pd.Timedelta(days=1)
        mask &= (docs["date"] < end).to_numpy()
        active = True
    return mask if active else None
