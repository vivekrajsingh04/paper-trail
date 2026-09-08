"""End-to-end checks of the comparison engine on figures taken from the corpus.

These use hand-built facts rather than live extraction so the reasoning layer
can be tested without a model or an API key.  The numbers, units and wording are
copied verbatim from the starter documents.
"""

from __future__ import annotations


from factlayer.canon import MetricResolver, normalise_metric
from factlayer.compare import ComparisonEngine, compare_values
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


# ---------------------------------------------------------------- regressions
def test_quarter_with_colon_is_not_a_whole_year():
    """The RBI writes "Q1:2024-25"; parsing it as the year merged quarters into it."""
    q = parse_period("Q1:2024-25")
    y = parse_period("2024-25")
    assert q.granularity == "quarter"
    assert (q.start, q.end) == ("2024-04-01", "2024-06-30")
    assert (q.start, q.end) != (y.start, y.end)


def test_subject_that_restates_the_metric_is_not_a_dimension():
    """A subject echoing the metric must not become an explanation for a gap."""
    from factlayer.compare import dimensions

    echo = mk("real GDP growth", "6.5", "per cent", "FY25", "rbi", subject="real GDP")
    real = mk("real GDP growth", "7.8", "per cent", "FY25", "imf", subject="India")
    assert "subject" not in dimensions(echo)
    assert dimensions(real)["subject"] == "india"

    # With the echoed subject discarded, nothing spurious explains the gap.
    e = engine()
    rel = e._judge(e.evaluate(echo, real))
    assert rel.explained_by != "subject"


def test_equal_values_across_different_periods_are_not_corroboration():
    """Q1 growth and full-year growth both reading 6.5% is coincidence."""
    a = mk("real GDP growth", "6.5", "per cent", "Q1:2024-25", "rbi", subject="India")
    b = mk("real GDP growth", "6.5", "per cent", "FY2024/25", "imf", subject="India")
    e = engine()
    rel = e._judge(e.evaluate(a, b))
    assert rel.verdict is Verdict.RELATED
    assert "period" in rel.differing_dims
    assert "coincide" in rel.explanation


def test_equal_values_on_the_same_subject_still_corroborate():
    a = mk("real GDP growth", "6.5", "per cent", "2024-25", "rbi", subject="India")
    b = mk("real GDP growth", "6.5", "per cent", "FY2024/25", "imf", subject="India")
    e = engine()
    rel = e._judge(e.evaluate(a, b))
    assert rel.verdict is Verdict.CORROBORATES


# ------------------------------------------------- abstention (needs_review)
def _engine_with_learned_period():
    """An engine that has seen enough pairs to trust `period` as explanatory."""
    e = engine()
    facts = [
        mk("segment revenue", f"{100 + i * 40}", "Rs. million", yr, "ar", scope="consolidated")
        for i, yr in enumerate(["FY20", "FY21", "FY22", "FY23", "FY24", "FY25"])
    ]
    e.build(facts)
    return e


def test_missing_period_is_abstained_not_called_a_contradiction():
    """A false contradiction costs more than an admitted gap."""
    e = _engine_with_learned_period()
    assert e._dim_power("period") >= 0.6

    a = mk("real GDP growth", "6.0", "per cent", "FY25", "es", subject="India")
    b = mk("real GDP growth", "7.8", "per cent", None, "imf", subject="India")
    rel = e._judge(e.evaluate(a, b))
    assert rel.verdict is Verdict.NEEDS_REVIEW
    assert rel.review_reason == "period_unknown_on_one_side"
    assert "held for review" in rel.explanation


def test_a_fully_specified_disagreement_is_still_a_contradiction():
    """Abstention must not swallow genuine conflicts."""
    e = _engine_with_learned_period()
    a = mk("real GDP growth", "6.5", "per cent", "FY26", "rbi", subject="India",
           basis="projection")
    b = mk("real GDP growth", "6.6", "per cent", "FY2025/26", "imf", subject="India",
           basis="projection")
    rel = e._judge(e.evaluate(a, b))
    assert rel.verdict is Verdict.CONTRADICTS


def test_a_weak_missing_dimension_does_not_trigger_abstention():
    """Only dimensions that measurably move values disqualify a comparison."""
    e = _engine_with_learned_period()
    a = mk("real GDP growth", "6.5", "per cent", "FY26", "rbi", subject="India",
           basis="projection", footnote_marker="3")
    b = mk("real GDP growth", "6.6", "per cent", "FY2025/26", "imf", subject="India",
           basis="projection")
    rel = e._judge(e.evaluate(a, b))
    # `footnote_marker` has never been seen, so it sits at the 0.5 prior — below
    # the abstain threshold, and must not be used to duck the verdict.
    assert rel.verdict is Verdict.CONTRADICTS
