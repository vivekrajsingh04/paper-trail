"""Deterministic scan for figures that *could* be facts.

This module never decides what a number means -- that is the language model's
job.  It answers a narrower question with certainty: "where on this page is
there a quantity at all, and what characters does it occupy?"

Three jobs come out of that:

1. **Pre-filter.**  Pages with no quantities and no state-change language are
   never sent to the model, which is most of the cost saved on a 500-page
   corpus.

2. **Verification target.**  Every fact the model emits must quote text that
   actually exists on the page.  The scanner supplies the ground truth.

3. **Coverage.**  Comparing detected literals against facts produced gives an
   honest recall signal per page: "47 quantities present, 12 became facts,
   here are the 35 that did not."  That number is the system's own account of
   what it missed, and it is reported rather than hidden.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# A quantity: optional currency, digits with Indian or international grouping,
# optional decimal, optional trailing magnitude word or percent sign.
QUANTITY_RE = re.compile(
    r"""
    (?P<pre>[₹$€£]|\bRs\.?|\bUS\$|\bINR\b|\bUSD\b)?      # leading currency
    \s*
    (?P<open>\()?                                          # accounting negative
    (?P<num>
        \d{1,3}(?:[, ]\d{2,3})+(?:\.\d+)?
      | \d+\.\d+
      | \d{1,15}
    )
    (?P<close>\))?
    \s*
    (?P<post>
        %|per\s?cent|percent|bps?
      | \b(?:cr|crore|crores|lakh|lakhs|mn|million|bn|billion|tn|trillion|
            k|thousand|th)\b
      | \b(?:tonnes?|tons?|kg|mt)\b
      | \b(?:days?|times|x)\b
    )?
    """,
    re.VERBOSE | re.IGNORECASE,
)

# Language that signals a semantic (non-numeric) fact worth extracting: status
# changes, appointments, classifications, definitions.
STATE_RE = re.compile(
    r"\b(appointed|resigned|retired|ceased|inducted|re-?designated|"
    r"stepped\s+down|elevated|promoted|demised|vacated|"
    r"incorporated|renamed|merged|acquired|divested|"
    r"is\s+(?:a|an|the)\s+\w+\s+director|independent\s+director|"
    r"managing\s+director|chief\s+\w+\s+officer|"
    r"registered\s+office|principal\s+place\s+of\s+business|"
    r"listed\s+on|delisted|classified\s+as|designated\s+as)\b",
    re.IGNORECASE,
)

# Purely structural numbers that are almost never facts on their own.
_NOISE_CONTEXT = re.compile(
    r"\b(note|notes|page|clause|section|schedule|para|paragraph|annexure|"
    r"chapter|table|figure|chart|regulation|rule|item|sr\.?\s*no|s\.?\s*no)\b"
    r"[\s.:]*$",
    re.IGNORECASE,
)


@dataclass
class Candidate:
    start: int
    end: int
    text: str
    number: str
    currency: str | None
    magnitude: str | None
    is_percent: bool
    line: str
    line_start: int
    salience: str = "high"  # "high" | "low"
    reason: str | None = None  # why it was demoted, if it was


# Tokens that make a number a *measurement* rather than an incidental digit.
def classify_salience(text_before: str, cand_text: str, number: str,
                      currency: str | None, magnitude: str | None,
                      is_percent: bool, line: str) -> tuple[str, str | None]:
    """Separate measurements from structural digits.

    Footnote markers, fiscal-year suffixes and page numbers are *detected* --
    never silently dropped -- but excluded from the coverage denominator, so
    "what did the extractor miss?" stays a meaningful question.
    """
    tail = text_before[-3:].lower()

    # "FY24", "Q4", "H1", "CY2024": the digits belong to a period token.
    if re.search(r"(?:fy|cy|q|h)$", tail) and not currency and not magnitude:
        return "low", "period_token"

    # Bare footnote reference: a small parenthesised integer with no unit.
    if cand_text.startswith("(") and cand_text.endswith(")"):
        inner = number.replace(",", "")
        if "." not in inner and len(inner) <= 2 and not currency and not magnitude:
            return "low", "footnote_marker"

    # A line that is nothing but the number: almost always a page number.
    if line.strip() == cand_text.strip() and len(number) <= 4 and not currency \
            and not magnitude and not is_percent:
        return "low", "standalone_number"

    if currency or magnitude or is_percent:
        return "high", None
    if "," in number or "." in number:
        return "high", None
    # A bare integer with no unit anywhere on the line carries little meaning.
    if len(number) <= 2:
        return "low", "bare_small_integer"
    return "high", None


def _line_bounds(text: str, pos: int) -> tuple[int, int]:
    start = text.rfind("\n", 0, pos) + 1
    end = text.find("\n", pos)
    if end == -1:
        end = len(text)
    return start, end


def scan_quantities(text: str, min_len: int = 1) -> list[Candidate]:
    """Find every quantity-shaped token on a page."""
    out: list[Candidate] = []
    for m in QUANTITY_RE.finditer(text):
        num = m.group("num")
        if num is None or len(num.replace(",", "")) < min_len:
            continue

        start, end = m.start(), m.end()
        # Trim whitespace the regex may have absorbed around the unit.
        while end > start and text[end - 1].isspace():
            end -= 1
        while start < end and text[start].isspace():
            start += 1

        ls, le = _line_bounds(text, start)
        line = text[ls:le]

        prefix = text[ls:start]
        if _NOISE_CONTEXT.search(prefix):
            continue

        post = (m.group("post") or "").strip().lower()
        currency = (m.group("pre") or "").strip() or None
        magnitude = post if post and not post.startswith(("%", "per", "bps", "bp")) else None
        is_percent = post.startswith(("%", "per", "bps", "bp"))
        cand_text = text[start:end]
        salience, reason = classify_salience(
            text[ls:start], cand_text, num, currency, magnitude, is_percent, line
        )
        out.append(
            Candidate(
                start=start,
                end=end,
                text=cand_text,
                number=num,
                currency=currency,
                magnitude=magnitude,
                is_percent=is_percent,
                line=line,
                line_start=ls,
                salience=salience,
                reason=reason,
            )
        )
    return out


def salient(text: str) -> list[Candidate]:
    """Only the quantities that look like measurements."""
    return [c for c in scan_quantities(text) if c.salience == "high"]


def has_state_language(text: str) -> bool:
    return bool(STATE_RE.search(text))


def page_is_interesting(text: str, min_quantities: int = 2) -> bool:
    """Cheap gate deciding whether a page is worth an extraction call."""
    if len(text.strip()) < 120:
        return False
    if has_state_language(text):
        return True
    return len(salient(text)) >= min_quantities


def coverage(text: str, extracted_quotes: list[str]) -> dict:
    """How many detected quantities ended up inside an extracted fact.

    Returns the ratio plus the literals that were left on the table, so a
    reviewer can see exactly what the extractor walked past.
    """
    cands = salient(text)
    if not cands:
        return {"detected": 0, "covered": 0, "ratio": 1.0, "missed": []}

    spans: list[tuple[int, int]] = []
    for q in extracted_quotes:
        idx = text.find(q)
        while idx != -1:
            spans.append((idx, idx + len(q)))
            idx = text.find(q, idx + 1)

    missed = []
    covered = 0
    for c in cands:
        if any(s <= c.start and c.end <= e for s, e in spans):
            covered += 1
        else:
            missed.append({"text": c.text, "line": c.line.strip()[:120], "start": c.start})

    return {
        "detected": len(cands),
        "covered": covered,
        "ratio": covered / len(cands),
        "missed": missed[:40],
    }
