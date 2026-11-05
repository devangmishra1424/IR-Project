"""E-mail cleaning: split a body into fresh and quoted text, and strip boilerplate.

Enron bodies carry a lot that is not the author's own words: quoted replies, forwarded
messages with their headers, legal footers, Lotus Notes routing (``Kay Mann/Corp/Enron@ENRON``),
URLs and the occasional HTML. The cleaner

* separates the text the sender wrote (``fresh``) from earlier or forwarded material
  (``quoted``) - the quoted part is kept, because forwarded articles are often the real
  content of a message, but it is a separate field so it can be down-weighted;
* removes noise that would only add junk terms to the index.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass

from maillens.config import MAX_BODY_CHARS

# ----------------------------------------------------------------------------- subjects
_SUBJECT_PREFIX = re.compile(r"^\s*(?:(?:re|fw|fwd)\s*:\s*)+", re.IGNORECASE)
_WHITESPACE = re.compile(r"\s+")


def normalize_subject(subject: str | None) -> str:
    """Subject without Re:/Fw: prefixes, case-folded, whitespace collapsed (thread key)."""
    stripped = _SUBJECT_PREFIX.sub("", subject or "")
    return _WHITESPACE.sub(" ", stripped).strip().casefold()


# ----------------------------------------------------------------------------- body: split
# Divider lines announcing earlier or forwarded material. They carry no content, so they are
# dropped from the text.
_DIVIDERS = (
    re.compile(r"^\s*-{2,}\s*Original Message\s*-{2,}\s*$", re.IGNORECASE),
    re.compile(r"^\s*-{2,}\s*Forwarded by\b.*$", re.IGNORECASE),  # Lotus Notes
    re.compile(r"^\s*-{2,}\s*Forwarded Message\s*-{2,}\s*$", re.IGNORECASE),
    re.compile(r"^\s*On\s.{4,160}\bwrote:\s*$", re.IGNORECASE),
    # The Notes "Forwarded by" divider is hard-wrapped; this is its second half.
    re.compile(r"^\s*\d{1,2}:\d{2}\s*[AP]M\s*-{3,}\s*$", re.IGNORECASE),
)
# Notes headers put the sender's name on one line and the date alone on the next.
_DATE_ONLY_LINE = re.compile(
    r"^\s*\d{1,2}/\d{1,2}/\d{2,4}\s+\d{1,2}:\d{2}(?::\d{2})?\s*(?:[AP]M)?\s*$", re.IGNORECASE
)
# Headers of a quoted message pasted without a divider: "From: ..." or the Notes style
# "slafontaine@globalp.com on 03/05/2001 11:31:02 AM", followed by To:/cc:/Subject: lines.
_FROM_LINE = re.compile(r"^\s*From:\s+\S", re.IGNORECASE)
_NAME_ON_DATE = re.compile(
    r"^\s*\S.{0,120}?\s+on\s+\d{1,2}/\d{1,2}/\d{2,4}\s+\d{1,2}:\d{2}(?::\d{2})?\s*(?:[AP]M)?\s*$",
    re.IGNORECASE,
)
_HEADER_LINE = re.compile(
    r"^\s*(?:from|to|cc|bcc|sent|date|subject|importance|attachments?|reply-to)\s*:",
    re.IGNORECASE,
)
_QUOTE_LINE = re.compile(r"^\s*>")
_LOOKAHEAD = 4  # a header-looking line only opens a quoted block if another header follows


def _is_divider(line: str) -> bool:
    return any(p.match(line) for p in _DIVIDERS)


def _header_follows(lines: list[str], i: int) -> bool:
    return any(_HEADER_LINE.match(other) for other in lines[i + 1 : i + 1 + _LOOKAHEAD])


def _opens_bare_header_block(lines: list[str], i: int) -> bool:
    return bool(_FROM_LINE.match(lines[i]) or _NAME_ON_DATE.match(lines[i])) and _header_follows(
        lines, i
    )


def split_body(body: str) -> tuple[str, str]:
    """(fresh, quoted): everything before the first quoted-block marker, and everything after.

    Divider lines themselves are dropped (they are not content), and lines starting with ``>``
    are quoted wherever they appear.
    """
    lines = body.splitlines()
    fresh: list[str] = []
    quoted: list[str] = []
    in_quoted = False
    for i, line in enumerate(lines):
        if _is_divider(line):
            in_quoted = True
            continue
        if not in_quoted:
            if _opens_bare_header_block(lines, i):
                in_quoted = True
            elif _DATE_ONLY_LINE.match(line) and _header_follows(lines, i):
                in_quoted = True
                if fresh and fresh[-1].strip():  # the sender's name sits on the line above
                    quoted.append(fresh.pop())
        if in_quoted or _QUOTE_LINE.match(line):
            quoted.append(line)
        else:
            fresh.append(line)
    return "\n".join(fresh), "\n".join(quoted)


# ----------------------------------------------------------------------------- body: noise
_ENRON_FOOTER = re.compile(
    r"\*{10,}\s*This e-?mail is the property of Enron.{0,2500}?(?:\*{10,}|\Z)",
    re.IGNORECASE | re.DOTALL,
)
_URL = re.compile(r"(?:https?|ftp)://\S+|www\.\S+", re.IGNORECASE)
_NOTES_ROUTING = re.compile(r"(?:/[A-Za-z0-9]{2,8}){1,3}@[A-Za-z0-9]{2,10}")
_EMAIL = re.compile(r"\b([A-Za-z0-9][A-Za-z0-9._%+-]*)@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
# Only real HTML tag names: a generic "<anything>" pattern would also delete "<name@domain>".
_HTML_TAG = re.compile(
    r"</?(?:html|head|body|div|span|p|br|b|i|u|a|font|table|tbody|tr|td|th|ul|ol|li|img|style|"
    r"script|meta|title|center|h[1-6]|strong|em|hr|pre|blockquote|o:p|!doctype)\b[^<>]{0,200}>",
    re.IGNORECASE,
)
_LONG_NUMBER = re.compile(r"\b\d{12,}\b")
_SEPARATOR_LINE = re.compile(r"^[\W_]{4,}$")
_LOCAL_PART_JOINERS = re.compile(r"[._%+-]+")
_DISCLAIMER_CUES = (
    "confidential",
    "intended recipient",
    "privileged",
    "unauthorized",
    "prohibited",
    "delete this",
    "disclaimer",
)
_BLANK_RUNS = re.compile(r"\n\s*\n\s*(?:\n\s*)+")


def _is_disclaimer(paragraph: str) -> bool:
    lowered = paragraph.lower()
    return sum(cue in lowered for cue in _DISCLAIMER_CUES) >= 3


def _strip_noise(text: str, *, drop_headers: bool) -> str:
    text = _HTML_TAG.sub(" ", text)
    text = _URL.sub(" ", text)
    text = _NOTES_ROUTING.sub("", text)
    text = _EMAIL.sub(lambda m: _LOCAL_PART_JOINERS.sub(" ", m.group(1)), text)
    text = _LONG_NUMBER.sub(" ", text)
    kept: list[str] = []
    for line in text.splitlines():
        if _QUOTE_LINE.match(line):
            line = _QUOTE_LINE.sub("", line, count=1)
        line = _WHITESPACE.sub(" ", line).strip()
        if _SEPARATOR_LINE.match(line):
            continue
        if drop_headers and _HEADER_LINE.match(line):
            continue
        kept.append(line)
    paragraphs = _BLANK_RUNS.sub("\n\n", "\n".join(kept)).split("\n\n")
    return "\n\n".join(p for p in paragraphs if not _is_disclaimer(p)).strip()


@dataclass(frozen=True, slots=True)
class CleanedBody:
    fresh: str
    quoted: str

    @property
    def text(self) -> str:
        """Everything the index sees for the body: fresh text first, then quoted material."""
        return f"{self.fresh}\n{self.quoted}".strip()


def clean_body(body: str | None, *, max_chars: int = MAX_BODY_CHARS) -> CleanedBody:
    """Split ``body`` into fresh/quoted text and remove boilerplate; total length is bounded."""
    if not body:
        return CleanedBody("", "")
    body = _ENRON_FOOTER.sub("", html.unescape(body))
    fresh_raw, quoted_raw = split_body(body)
    fresh = _strip_noise(fresh_raw, drop_headers=False)[:max_chars]
    quoted = _strip_noise(quoted_raw, drop_headers=True)[: max(0, max_chars - len(fresh))]
    return CleanedBody(fresh, quoted)
