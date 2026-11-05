"""Download the two public datasets (parquet files only) from the Hugging Face Hub.

Anonymous (no token is sent), idempotent (finished files are skipped, interrupted downloads
resume) and with bounded retries on *transient* failures only: a 4xx means a wrong name or a
permissions problem, and retrying it would just hide the bug.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from huggingface_hub import hf_hub_download
from huggingface_hub.errors import HfHubHTTPError, LocalEntryNotFoundError

from maillens import config

log = logging.getLogger(__name__)

MAX_ATTEMPTS = 3
BACKOFF_SECONDS = 5.0
ETAG_TIMEOUT_SECONDS = 30.0
_TRANSIENT_STATUS = frozenset({408, 429, 500, 502, 503, 504})


def _http_status(err: HfHubHTTPError) -> int | None:
    response = getattr(err, "response", None)
    return None if response is None else int(response.status_code)


def fetch(repo_id: str, filename: str, dest: Path) -> Path:
    """Download one file of a dataset repo into ``dest`` and return its local path."""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            return Path(
                hf_hub_download(
                    repo_id=repo_id,
                    repo_type="dataset",
                    filename=filename,
                    local_dir=dest,
                    token=False,
                    etag_timeout=ETAG_TIMEOUT_SECONDS,
                )
            )
        except HfHubHTTPError as err:
            status = _http_status(err)
            if status not in _TRANSIENT_STATUS or attempt == MAX_ATTEMPTS:
                raise
            reason = f"HTTP {status}"
        except LocalEntryNotFoundError as err:
            # Raised when the Hub cannot be reached and no local copy exists.
            if attempt == MAX_ATTEMPTS:
                raise
            reason = f"network error: {err}"
        delay = BACKOFF_SECONDS * 2 ** (attempt - 1)
        log.warning(
            "download of %s/%s failed (%s); attempt %d of %d, retrying in %.0fs",
            repo_id,
            filename,
            reason,
            attempt,
            MAX_ATTEMPTS,
            delay,
        )
        time.sleep(delay)
    raise AssertionError("unreachable: the loop either returns or raises")


def download_all(raw_dir: Path = config.RAW_DIR) -> list[Path]:
    """Fetch the three e-mail shards and the two question files; return their local paths."""
    plan = [(config.EMAILS_REPO, "enron-emails", f) for f in config.EMAIL_SHARDS] + [
        (config.QUESTIONS_REPO, "questions", f) for f in config.QUESTION_FILES
    ]
    paths: list[Path] = []
    for repo_id, folder, filename in plan:
        log.info("fetching %s : %s", repo_id, filename)
        path = fetch(repo_id, filename, raw_dir / folder)
        size = path.stat().st_size
        if size == 0:
            raise OSError(f"{path} was downloaded but is empty")
        log.info("ok: %s (%.1f MB)", path.name, size / 1e6)
        paths.append(path)
    return paths
