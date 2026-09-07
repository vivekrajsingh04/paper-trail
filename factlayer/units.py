"""Number parsing, unit normalisation, and implied-precision intervals.

The central idea
----------------
A figure printed in a document is a *rounded* statement about a true value.
How it is rounded is visible in how it is written:

    "8,142"      rounded to the nearest 1        -> true value in [8141.5, 8142.5]
    "81,415.38"  rounded to the nearest 0.01     -> [81415.375, 81415.385]
    "6.4"        rounded to the nearest 0.1      -> [6.35, 6.45]

Once both sides of a comparison are converted to a common unit, "do these two
figures agree?" becomes "do their implied intervals intersect?" -- a question
with a defensible answer, rather than a hand-tuned percentage tolerance.

This matters because the tolerances involved differ by six orders of magnitude
across a single corpus.  Comparing "₹8,142 Cr" to "81,415.38 Mn" needs a
tolerance of ~0.006%; distinguishing "6.4 per cent" from "6.5 per cent" needs
one tighter than 1.6%.  No single fixed epsilon does both.  Reading the
precision off the digits does.

Unit tables below encode general knowledge (SI prefixes, Indian numbering,
currency codes), not anything about the starter documents.  Nothing here keys
off a filename, a company, or a specific metric.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Literal

BoundKind = Literal["exact", "at_least", "at_most", "approximate"]


# --------------------------------------------------------------------------
# Magnitude and unit vocabulary
# --------------------------------------------------------------------------

# Multiplier applied to the written number.  Indian and international scales.
MAGNITUDES: dict[str, float] = {
    "": 1.0,
    "hundred": 1e2,
    "thousand": 1e3,
    "k": 1e3,
    "lakh": 1e5,
    "lac": 1e5,
    "lakhs": 1e5,
    "million": 1e6,
    "mn": 1e6,
    "mln": 1e6,
    "m": 1e6,
    "crore": 1e7,
    "cr": 1e7,
    "crores": 1e7,
    "billion": 1e9,
    "bn": 1e9,
    "b": 1e9,
    "trillion": 1e12,
    "tn": 1e12,
    "trn": 1e12,
}

CURRENCIES: dict[str, str] = {
    "₹": "INR", "rs": "INR", "rs.": "INR", "inr": "INR", "rupee": "INR",
    "rupees": "INR", "$": "USD", "us$": "USD", "usd": "USD", "€": "EUR",
    "eur": "EUR", "£": "GBP", "gbp": "GBP",
}

# Dimension families.  Members of the same family are convertible; members of
# different families are never compared.
MASS: dict[str, float] = {
    "tonne": 1.0, "tonnes": 1.0, "ton": 1.0, "tons": 1.0, "mt": 1.0,
    "kg": 1e-3, "kilogram": 1e-3, "kilograms": 1e-3,
}

PERCENTISH = {"%", "per cent", "percent", "pc", "percentage", "ppt", "pp",
              "percentage point", "percentage points", "basis point",
              "basis points", "bps", "bp"}

# Units that are ratios of two other units and should never be scale-converted.
RATIO_HINTS = {"days", "day", "times", "x", "ratio", "index", "per share"}


@dataclass
class ParsedNumber:
    value: float          # the bare number as written, before magnitude scaling
    ulp: float            # implied rounding step of the written digits
    negative: bool
    bound: BoundKind
    sign_convention: str | None


# --------------------------------------------------------------------------
# Number parsing
# --------------------------------------------------------------------------

_APPROX_TOKENS = ("~", "≈", "about", "approx", "approximately", "around", "circa", "c.")
_ATLEAST_TOKENS = (">", "≥", "over", "more than", "at least", "in excess of", "north of")
_ATMOST_TOKENS = ("<", "≤", "under", "less than", "below", "at most", "up to")

_NUM_RE = re.compile(
    r"""
    (?P<sign>[-−–+]?)
    (?P<digits>
        \d{1,3}(?:[, \s]\d{2,3})+(?:\.\d+)?   # 1,23,456.78 or 1,234,567.89
      | \d+(?:\.\d+)?                                # 1234.56
      | \.\d+                                        # .56
    )
    """,
    re.VERBOSE,
)


def parse_number(raw: str) -> ParsedNumber | None:
    """Parse a numeric literal, recovering the precision implied by its digits.

    `ulp` is the unit in the last place: the rounding step the author used.
    For "12.30" that is 0.01 even though the trailing zero adds no magnitude --
    writing the zero is itself a precision claim.
    """
    if raw is None:
        return None
    s = raw.strip()
    low = s.lower()

    bound: BoundKind = "exact"
    if any(t in low for t in _APPROX_TOKENS):
        bound = "approximate"
    if low.rstrip().endswith("+") or any(t in low for t in _ATLEAST_TOKENS):
        bound = "at_least"
    elif any(t in low for t in _ATMOST_TOKENS):
        bound = "at_most"

    # Parenthesised negatives are the accounting convention: (452) == -452.
    sign_convention = None
    negative = False
    if re.search(r"\(\s*[\d.,]", s) and ")" in s:
        negative = True
        sign_convention = "parenthesised_negative"

    m = _NUM_RE.search(s)
    if not m:
        return None

    digits = m.group("digits")
    clean = re.sub(r"[, \s]", "", digits)
    try:
        value = float(clean)
    except ValueError:
        return None

    if m.group("sign") in {"-", "−", "–"}:
        negative = True

    # Implied precision: half a step in the last written decimal place.
    if "." in clean:
        decimals = len(clean.split(".")[1])
        ulp = 10.0 ** (-decimals)
    else:
        ulp = 1.0

    return ParsedNumber(
        value=value, ulp=ulp, negative=negative, bound=bound,
        sign_convention=sign_convention,
    )


# --------------------------------------------------------------------------
# Unit normalisation
# --------------------------------------------------------------------------

@dataclass
class NormalisedUnit:
    canonical: str        # e.g. "INR_million", "percent", "tonne", "count"
    family: str           # e.g. "currency:INR", "percent", "mass", "count"
    factor: float         # multiply the written value by this to reach canonical


def normalise_unit(unit_raw: str | None, magnitude_raw: str | None = None) -> NormalisedUnit:
    """Resolve a written unit + magnitude word into a canonical unit and factor.

    Canonical choices are arbitrary but must be stable:
      currency -> millions of the currency
      mass     -> tonnes
      percent  -> percent
      count    -> ones
    """
    u = (unit_raw or "").strip().lower().rstrip(".")
    mag = (magnitude_raw or "").strip().lower().rstrip(".")

    # A magnitude word may arrive embedded in the unit string ("Rs. crore").
    mag_factor = 1.0
    if mag and mag in MAGNITUDES:
        mag_factor = MAGNITUDES[mag]
    else:
        for token, mult in sorted(MAGNITUDES.items(), key=lambda kv: -len(kv[0])):
            if token and re.search(rf"\b{re.escape(token)}\b", u):
                mag_factor = mult
                u = re.sub(rf"\b{re.escape(token)}\b", " ", u).strip()
                break

    if u in PERCENTISH or any(p in u for p in ("per cent", "percent", "%")):
        # Percentages are never magnitude-scaled.
        return NormalisedUnit("percent", "percent", 1.0)

    if any(h in u for h in RATIO_HINTS):
        return NormalisedUnit(u or "ratio", f"ratio:{u or 'ratio'}", 1.0)

    for sym, code in CURRENCIES.items():
        if sym in u:
            # canonical = millions of that currency
            return NormalisedUnit(f"{code}_million", f"currency:{code}", mag_factor / 1e6)

    for token, mult in MASS.items():
        if re.search(rf"\b{re.escape(token)}\b", u):
            return NormalisedUnit("tonne", "mass", mag_factor * mult)

    if u in {"", "count", "units", "no", "nos", "number"}:
        return NormalisedUnit("count", "count", mag_factor)

    # Unknown unit: keep it, but scale by any magnitude word we recognised.
    return NormalisedUnit(u, f"other:{u}", mag_factor)


# --------------------------------------------------------------------------
# Interval construction
# --------------------------------------------------------------------------

# Widening applied to figures hedged with "about"/"~".  A hedged figure claims
# less precision than its digits suggest; we grant it an order of magnitude.
APPROX_WIDENING = 10.0


def build_interval(
    parsed: ParsedNumber,
    unit: NormalisedUnit,
    rel_slack: float = 0.0,
) -> tuple[float, float, float]:
    """Return (canonical_value, lo, hi) for a parsed number in canonical units.

    `rel_slack` adds an optional relative tolerance on top of digit precision.
    It defaults to zero: on the starter corpus, digit precision alone resolves
    every genuine pair, and a nonzero default would quietly paper over real
    disagreements.  It is exposed for corpora that restate figures loosely.
    """
    sign = -1.0 if parsed.negative else 1.0
    canonical_value = sign * parsed.value * unit.factor

    half = 0.5 * parsed.ulp * unit.factor
    if parsed.bound == "approximate":
        half *= APPROX_WIDENING

    lo = canonical_value - half
    hi = canonical_value + half

    if rel_slack:
        pad = abs(canonical_value) * rel_slack
        lo -= pad
        hi += pad

    if parsed.bound == "at_least":
        hi = math.inf
    elif parsed.bound == "at_most":
        lo = -math.inf

    return canonical_value, lo, hi


def intervals_overlap(
    a: tuple[float, float], b: tuple[float, float], atol: float = 1e-9
) -> bool:
    """True when two closed intervals intersect (touching counts as disjoint).

    Touching is treated as disjoint on purpose: "6.4" gives [6.35, 6.45] and
    "6.5" gives [6.45, 6.55].  They meet at exactly 6.45.  Calling that
    agreement would merge every adjacent pair of one-decimal figures in the
    corpus, so we require genuine overlap.
    """
    lo = max(a[0], b[0])
    hi = min(a[1], b[1])
    return (hi - lo) > atol


def relative_gap(a: float, b: float) -> float:
    """Symmetric relative difference, safe at zero."""
    denom = max(abs(a), abs(b))
    if denom == 0:
        return 0.0
    return abs(a - b) / denom
