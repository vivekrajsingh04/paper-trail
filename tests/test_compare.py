"""End-to-end checks of the comparison engine on figures taken from the corpus.

These use hand-built facts rather than live extraction so the reasoning layer
can be tested without a model or an API key.  The numbers, units and wording are
copied verbatim from the starter documents.
"""

from __future__ import annotations

import pytest

from factlayer.canon import MetricResolver, normalise_metric
from factlayer.compare import ComparisonEngine, compare_values, dimensions
from factlayer.models import Evidence, Fact, Quantity, Verdict
from factlayer.periods import parse_period
from factlayer.units import build_interval, normalise_unit, parse_number


def mk(
    metric: str,
    value_raw: str,
    unit_raw: str | None,
    period: str | None,
    doc: str,
    page: int = 0,
    subject: str = "Delhivery Limited",
    **qualifiers: str,
) -> Fact:
    parsed = parse_number(value_raw)
    unit = normalise_unit(unit_raw)
    canon, lo, hi = build_interval(parsed, unit)
    q = Quantity(
        raw=value_raw, value=parsed.value, unit=unit.canonical, unit_raw=unit_raw,
        canonical_value=canon, canonical_unit=unit.canonical, lo=lo, hi=hi, ulp=parsed.ulp,
    )
    f = Fact(
        metric=metric,
        metric_key=normalise_metric(metric),
        subject=subject,
        quantity=q,
        period=parse_period(period),
        qualifiers={k: v for k, v in qualifiers.items()},
        evidence=Evidence(
            doc_id=doc, doc_title=doc, page=page, char_start=0,
            char_end=len(value_raw), snippet=value_raw, quote=value_raw,
        ),
        confidence=0.9,
    )
    return f


def engine() -> ComparisonEngine:
    return ComparisonEngine(MetricResolver(use_llm=False))


# ---------------------------------------------------------------- case 1
def test_corroboration_across_units_and_documents():
    """Deck says 8,142 Cr; annual report says 81,415.38 Mn. Same fact."""
    a = mk("revenue from services", "8,142", "Rs. crore", "FY24", "deck", scope="consolidated")
    b = mk("revenue from services", "81,415.38", "Rs. million", "FY24", "annual_report",
           scope="consolidated")
    rel = engine()._judge(engine().evaluate(a, b))
    assert rel.verdict is Verdict.CORROBORATES
    assert rel.cross_document
    assert rel.reasoning["values"]["relative_gap"] < 0.0001
    assert "overlap" in rel.explanation


def test_corroboration_survives_magnitude_difference():
    """1.4 Mn tonnes vs 1,429K tonnes: a 2% gap that is still agreement."""
    a = mk("PTL freight tonnage", "1.4", "million tonnes", "FY24", "deck")
    b = mk("PTL freight tonnage", "1,429", "thousand tonnes", "FY24", "annual_report")
    ev = engine().evaluate(a, b)
    assert ev["values"]["agree"] is True
    assert ev["values"]["relative_gap"] > 0.02  # 2%+ apart, yet corroborating


# ---------------------------------------------------------------- case 2
def test_genuine_contradiction_between_institutions():
    """RBI projects 6.5% for FY26; the IMF projects 6.6%. Nothing explains it."""
    a = mk("real GDP growth", "6.5", "per cent", "FY26", "rbi", subject="India",
           basis="projection")
    b = mk("real GDP growth", "6.6", "per cent", "FY2025/26", "imf", subject="India",
           basis="projection")
    e = engine()
    rel = e._judge(e.evaluate(a, b))
    assert rel.verdict is Verdict.CONTRADICTS
    assert rel.cross_document
    assert rel.differing_dims == []


def test_narrow_gap_is_not_swallowed_by_tolerance():
    """6.4 vs 6.5 must not be called agreement: intervals touch but do not overlap."""
    a = mk("real GDP growth", "6.4", "per cent", "FY25", "es", subject="India")
    b = mk("real GDP growth", "6.5", "per cent", "FY25", "rbi", subject="India")
    assert compare_values(a, b)["agree"] is False


# ---------------------------------------------------------------- case 3
def test_reconciled_by_reporting_scope():
    """Standalone vs consolidated revenue for the same year is not a conflict."""
    a = mk("revenue from operations", "74,540.82", "Rs. million", "FY24", "annual_report",
           scope="standalone")
    b = mk("revenue from operations", "81,415.38", "Rs. million", "FY24", "annual_report",
           scope="consolidated")
    e = engine()
    rel = e._judge(e.evaluate(a, b))
    assert rel.verdict is Verdict.RECONCILED
    assert rel.explained_by == "scope"


def test_reconciled_by_estimate_vintage():
    """6.4% (first advance estimate) vs 6.5% (provisional) for the same year."""
    a = mk("real GDP growth", "6.4", "per cent", "FY25", "es", subject="India",
           basis="first_advance_estimate")
    b = mk("real GDP growth", "6.5", "per cent", "FY25", "rbi", subject="India",
           basis="provisional_estimate")
    e = engine()
    rel = e._judge(e.evaluate(a, b))
    assert rel.verdict is Verdict.RECONCILED
    assert rel.explained_by == "basis"
    assert "vintage" in rel.explanation or "basis" in rel.explanation


def test_reconciled_by_period():
    """Full-year revenue vs one quarter of it."""
    a = mk("revenue from services", "8,142", "Rs. crore", "FY24", "deck")
    b = mk("revenue from services", "2,076", "Rs. crore", "Q4 FY24", "deck")
    e = engine()
    rel = e._judge(e.evaluate(a, b))
    assert rel.verdict is Verdict.RECONCILED
    assert rel.explained_by == "period"


# ---------------------------------------------------------------- guards
def test_different_metrics_are_never_compared():
    """Revenue from operations and revenue from services differ by traded goods."""
    a = mk("revenue from operations", "72,253.01", "Rs. million", "FY23", "ar")
    b = mk("revenue from services", "72,236.47", "Rs. million", "FY23", "ar")
    assert engine().evaluate(a, b) is None


def test_incompatible_units_are_not_compared():
    a = mk("EBITDA", "1,266.41", "Rs. million", "FY24", "ar")
    b = mk("EBITDA", "1.6", "per cent", "FY24", "ar")
    assert compare_values(a, b) is None


def test_period_notation_differences_do_not_create_dimensions():
    """FY25, 2024-25 and FY2024/25 are one period, so `period` must not conflict."""
    a = mk("real GDP growth", "6.5", "per cent", "2024-25", "rbi", subject="India")
    b = mk("real GDP growth", "6.5", "per cent", "FY2024/25", "imf", subject="India")
    dims = engine().evaluate(a, b)["dimensions"]
    assert "period" in dims["agree"]
    assert "period" not in dims["conflict"]


def test_explanatory_power_is_learned_not_declared():
    """A dimension that always accompanies differing values scores high."""
    facts = []
    for i, yr in enumerate(["FY20", "FY21", "FY22", "FY23", "FY24"]):
        facts.append(mk("segment revenue", f"{100 + i * 37}", "Rs. million", yr, "ar",
                        scope="consolidated"))
    e = engine()
    e.build(facts)
    assert e.stats.dim_stats["period"].explanatory_power > 0.7
