"""Normalise the many ways documents write a time period.

The starter corpus alone contains "FY24", "FY2023-24", "2024-25",
"FY2024/25", "Q4 FY24", "H1 FY25" and "year ended March 31, 2024".  Three of
those denote *the same window* while looking nothing alike, and two that look
almost identical ("FY24" vs "FY2024-25") denote different years.  String
comparison is hopeless here, so every period is resolved to a date interval.

Fiscal-year convention
----------------------
The April-March convention is configurable and recorded on each Period as
`calendar`.  It is a default, not a hard-coded rule: `FISCAL_START_MONTH` can be
changed for corpora on other conventions, and documents that state their own
year-end let the extractor override it.  This is the one piece of domain
convention the system assumes, and it is declared rather than buried.
"""

from __future__ import annotations

import re
from datetime import date

from .models import Period

FISCAL_START_MONTH = 4  # April; fiscal year FY_N ends in month 3 of calendar N

_MONTHS = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
    "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
    "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9, "oct": 10,
    "october": 10, "nov": 11, "november": 11, "dec": 12, "december": 12,
}


def _eom(year: int, month: int) -> date:
    if month == 12:
        return date(year, 12, 31)
    nxt = date(year + (month // 12), (month % 12) + 1, 1)
    return date.fromordinal(nxt.toordinal() - 1)


def _fy_window(end_year: int) -> tuple[date, date]:
    """Window for the fiscal year *ending* in calendar `end_year`."""
    start = date(end_year - 1, FISCAL_START_MONTH, 1)
    end = _eom(end_year, FISCAL_START_MONTH - 1 if FISCAL_START_MONTH > 1 else 12)
    return start, end


def _expand_year(y: int) -> int:
    """Map a 2-digit year onto the current century."""
    if y >= 100:
        return y
    return 2000 + y if y < 90 else 1900 + y


def parse_period(text: str | None) -> Period | None:
    """Resolve a written period into a normalised interval.

    Returns None when nothing period-like is present, so callers can leave the
    dimension unspecified rather than guessing.
    """
    if not text:
        return None
    raw = text.strip()
    s = raw.lower().replace("–", "-").replace("—", "-").replace("–", "-")
    s = re.sub(r"\s+", " ", s)

    # --- Quarter: "Q4 FY24", "Q4FY2024", "fourth quarter of 2024-25" ---------
    m = re.search(r"\bq([1-4])\s*(?:of\s*)?f?y?\s*'?(\d{2,4})(?:\s*-\s*(\d{2,4}))?", s)
    if m:
        q = int(m.group(1))
        end_year = _resolve_fy_end(m.group(2), m.group(3))
        start_month = FISCAL_START_MONTH + 3 * (q - 1)
        year_offset, start_month = divmod(start_month - 1, 12)
        start_month += 1
        sy = end_year - 1 + year_offset
        start = date(sy, start_month, 1)
        em = start_month + 2
        ey = sy + (em - 1) // 12
        em = ((em - 1) % 12) + 1
        return Period(label=raw, start=start.isoformat(), end=_eom(ey, em).isoformat(),
                      granularity="quarter", calendar="IN_FY")

    # --- Half: "H1 FY25", "first half of FY25" ------------------------------
    m = re.search(r"\b(h[12]|first half|second half)\s*(?:of\s*)?f?y?\s*'?(\d{2,4})(?:\s*-\s*(\d{2,4}))?", s)
    if m:
        h = 1 if m.group(1) in {"h1", "first half"} else 2
        end_year = _resolve_fy_end(m.group(2), m.group(3))
        start_month = FISCAL_START_MONTH + 6 * (h - 1)
        year_offset, start_month = divmod(start_month - 1, 12)
        start_month += 1
        sy = end_year - 1 + year_offset
        start = date(sy, start_month, 1)
        em = start_month + 5
        ey = sy + (em - 1) // 12
        em = ((em - 1) % 12) + 1
        return Period(label=raw, start=start.isoformat(), end=_eom(ey, em).isoformat(),
                      granularity="half", calendar="IN_FY")

    # --- Explicit year-end date: "year ended March 31, 2024" ----------------
    m = re.search(r"(?:ended|ending|as at|as on|as of)?\s*(\d{1,2})?\s*"
                  r"(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?,?\s*"
                  r"(\d{1,2})?,?\s*(\d{4})", s)
    if m and ("ended" in s or "ending" in s or "as at" in s or "as on" in s or "as of" in s):
        month = _MONTHS[m.group(2)]
        year = int(m.group(4))
        if month == FISCAL_START_MONTH - 1 or (FISCAL_START_MONTH == 1 and month == 12):
            start, end = _fy_window(year)
            return Period(label=raw, start=start.isoformat(), end=end.isoformat(),
                          granularity="year", calendar="IN_FY")
        return Period(label=raw, start=date(year, month, 1).isoformat(),
                      end=_eom(year, month).isoformat(), granularity="point",
                      calendar="point")

    # --- Fiscal year spans: "FY2023-24", "2024-25", "FY2024/25" -------------
    m = re.search(r"\bf?y?\s*'?(\d{4})\s*[-/]\s*'?(\d{2,4})\b", s)
    if m:
        end_year = _resolve_fy_end(m.group(1), m.group(2))
        start, end = _fy_window(end_year)
        return Period(label=raw, start=start.isoformat(), end=end.isoformat(),
                      granularity="year", calendar="IN_FY")

    # --- Single fiscal year: "FY24", "FY2024", "fiscal 2024" ---------------
    m = re.search(r"\b(?:fy|fiscal(?:\s+year)?)\s*'?(\d{2,4})\b", s)
    if m:
        end_year = _expand_year(int(m.group(1)))
        start, end = _fy_window(end_year)
        return Period(label=raw, start=start.isoformat(), end=end.isoformat(),
                      granularity="year", calendar="IN_FY")

    # --- Bare calendar year: "in 2024", "CY2024" ---------------------------
    m = re.search(r"\b(?:cy\s*)?(\d{4})\b", s)
    if m:
        y = int(m.group(1))
        if 1900 <= y <= 2100:
            return Period(label=raw, start=date(y, 1, 1).isoformat(),
                          end=date(y, 12, 31).isoformat(), granularity="year",
                          calendar="CY")

    return None


def _resolve_fy_end(first: str, second: str | None) -> int:
    """Given "2023"+"24" or "24"+None, return the calendar year the FY ends in."""
    a = _expand_year(int(first))
    if second is None:
        return a
    b = int(second)
    if b < 100:
        # "2023-24" -> ends 2024;  "2024-25" -> ends 2025
        b = (a // 100) * 100 + b
        if b < a:
            b += 100
    return b


def periods_equal(a: Period | None, b: Period | None) -> bool:
    if a is None or b is None:
        return a is None and b is None
    if a.start and a.end and b.start and b.end:
        return a.start == b.start and a.end == b.end
    return a.label.strip().lower() == b.label.strip().lower()


def period_overlap(a: Period, b: Period) -> bool:
    if not (a.start and a.end and b.start and b.end):
        return False
    return not (a.end < b.start or b.end < a.start)


def contains(outer: Period, inner: Period) -> bool:
    """True when `inner` falls entirely inside `outer` (e.g. Q4 FY24 in FY24)."""
    if not (outer.start and outer.end and inner.start and inner.end):
        return False
    return outer.start <= inner.start and inner.end <= outer.end
