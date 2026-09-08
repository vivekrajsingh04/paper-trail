"""The relationship engine: deciding how any two facts stand to each other.

Every verdict is computed, never generated.  The language model's only role in
this module is answering "are these two metric names the same measurement?" --
and even that answer is cached and shown to the reviewer with its provenance.
Whether two figures agree, and what explains it when they do not, is arithmetic.

The judgement in two steps
--------------------------

**Do the values agree?**  Not "are they within 1%" -- do the intervals implied
by how each figure was *written* intersect?  See `units.py` for why a fixed
tolerance cannot work: on this corpus a 2.03% difference is agreement (1.4 Mn
tonnes vs 1,429K tonnes) while a 1.54% difference is a real conflict (6.4% vs
6.5% GDP growth).  Precision, not proximity.

**If they disagree, is something different about what they measure?**  Facts
carry an open set of dimensions -- period, subject, and whatever qualifier keys
the extractor found worth recording.  When two facts disagree on value *and* on
exactly one dimension, that dimension is the explanation.  When they disagree on
value and agree on every dimension, there is nothing left to explain it, and
that is a contradiction.

Which dimensions can explain a difference is *learned*, not declared
----------------------------------------------------------------------
Hard-coding "period and scope are explanatory" would be a document-specific
rule, and would break on the next corpus.  Instead the engine measures, across
all comparable pairs, how often each dimension co-occurs with a value
difference.  A dimension that almost always accompanies differing values
(reporting scope, estimate vintage) earns a high explanatory weight; one that
usually does not is weak evidence, and a reconciliation resting on it is
reported with correspondingly low confidence.  New qualifier keys are scored the
same way the moment they appear, which is what lets the schema grow.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field

from rapidfuzz import fuzz

from .canon import MetricResolver, normalise_metric, subject_key
from .models import Fact, Relation, Verdict
from .units import intervals_overlap, relative_gap

# Dimensions that describe *where a fact came from* rather than what it measures.
# They are recorded but never treated as explaining a value difference.
PROVENANCE_DIMS = {"source", "source_table", "table", "note", "page", "document"}

MAX_BLOCK = 400  # guard against a pathological metric group

# A dimension missing on one side is only disqualifying if that dimension is the
# kind of thing that moves values. Whether it is gets measured, not assumed --
# `period` scores 0.97 on this corpus and `scale` 0.33, so not knowing a period
# is fatal to a comparison while not knowing a scale is not.
ABSTAIN_POWER = 0.60


@dataclass
class DimStat:
    """How reliably a dimension accompanies a difference in value."""

    dim: str
    n_pairs: int = 0
    n_value_differs: int = 0

    @property
    def explanatory_power(self) -> float:
        """P(values differ | this is the only dimension in conflict).

        Smoothed toward 0.5 so a dimension seen twice does not earn a
        confident weight.
        """
        prior_weight = 4.0
        return (self.n_value_differs + 0.5 * prior_weight) / (self.n_pairs + prior_weight)


@dataclass
class CompareStats:
    pairs_considered: int = 0
    pairs_compared: int = 0
    skipped_metric_mismatch: int = 0
    skipped_unit_family: int = 0
    skipped_incomparable: int = 0
    dim_stats: dict[str, DimStat] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "pairs_considered": self.pairs_considered,
            "pairs_compared": self.pairs_compared,
            "skipped_metric_mismatch": self.skipped_metric_mismatch,
            "skipped_unit_family": self.skipped_unit_family,
            "skipped_incomparable": self.skipped_incomparable,
            "dimensions": {
                d: {
                    "pairs": s.n_pairs,
                    "value_differs": s.n_value_differs,
                    "explanatory_power": round(s.explanatory_power, 3),
                }
                for d, s in sorted(
                    self.dim_stats.items(), key=lambda kv: -kv[1].explanatory_power
                )
            },
        }


# --------------------------------------------------------------------------
# Dimension extraction
# --------------------------------------------------------------------------

def _subject_is_informative(f: Fact) -> bool:
    """False when the subject merely restates the metric.

    Extractors sometimes fill `subject` with a fragment of the metric name
    ("real GDP" for the metric "real GDP growth") instead of the entity the
    figure is about ("India"). Treating that as a dimension is worse than
    having no subject at all: two facts about the same thing then appear to
    differ on subject, and the engine will happily cite that non-difference as
    the explanation for a real gap in value.
    """
    sk = subject_key(f.subject)
    if not sk:
        return False
    mk = normalise_metric(f.metric)
    if not mk:
        return True
    if sk in mk or mk.startswith(sk):
        return False
    return fuzz.token_set_ratio(sk, mk) < 85


def dimensions(f: Fact) -> dict[str, str]:
    """The full dimension vector of a fact: period, subject, and its qualifiers."""
    dims: dict[str, str] = {}
    if f.period is not None:
        dims["period"] = f.period.key
    sk = subject_key(f.subject)
    if sk and _subject_is_informative(f):
        dims["subject"] = sk
    for k, v in f.qualifiers.items():
        key = str(k).strip().lower()
        val = str(v).strip().lower()
        if key and val and key not in dims:
            dims[key] = val
    return dims


def compare_dimensions(a: Fact, b: Fact) -> dict:
    """Split the two dimension vectors into agreeing, conflicting and one-sided."""
    da, db = dimensions(a), dimensions(b)
    agree, conflict, one_sided = [], [], []
    for k in sorted(set(da) | set(db)):
        va, vb = da.get(k), db.get(k)
        if va is None or vb is None:
            one_sided.append(k)
        elif va == vb:
            agree.append(k)
        else:
            conflict.append(k)
    return {
        "left": da,
        "right": db,
        "agree": agree,
        "conflict": conflict,
        "unspecified": one_sided,
    }


# --------------------------------------------------------------------------
# Value comparison
# --------------------------------------------------------------------------

def _unit_family(f: Fact) -> str | None:
    if f.quantity is None or f.quantity.canonical_unit is None:
        return None
    u = f.quantity.canonical_unit
    if u.endswith("_million"):
        return f"currency:{u.split('_')[0]}"
    return u


def compare_values(a: Fact, b: Fact) -> dict | None:
    """Compare two facts' values. Returns None when they are not comparable."""
    if a.kind == "semantic" or b.kind == "semantic":
        if a.kind != b.kind:
            return None
        sa = (a.state or "").strip().lower()
        sb = (b.state or "").strip().lower()
        if not sa or not sb:
            return None
        return {
            "mode": "state",
            "agree": sa == sb,
            "left": a.state,
            "right": b.state,
        }

    qa, qb = a.quantity, b.quantity
    if qa is None or qb is None:
        return None
    fa, fb = _unit_family(a), _unit_family(b)
    if fa is None or fb is None or fa != fb:
        return None
    if not (qa.has_interval and qb.has_interval):
        return None

    overlap = intervals_overlap((qa.lo, qa.hi), (qb.lo, qb.hi))
    gap = relative_gap(qa.canonical_value or 0.0, qb.canonical_value or 0.0)

    # How far apart are they, measured in the coarser figure's rounding step?
    coarser = max(qa.hi - qa.lo, qb.hi - qb.lo)
    separation = None
    if coarser > 0 and math.isfinite(coarser):
        raw_gap = abs((qa.canonical_value or 0) - (qb.canonical_value or 0))
        separation = raw_gap / coarser

    return {
        "mode": "interval",
        "agree": overlap,
        "left_value": qa.canonical_value,
        "right_value": qb.canonical_value,
        "left_interval": [qa.lo, qa.hi],
        "right_interval": [qb.lo, qb.hi],
        "unit": qa.canonical_unit,
        "relative_gap": gap,
        "separation_in_ulp": separation,
        "left_raw": f"{qa.raw} {qa.unit_raw or ''}".strip(),
        "right_raw": f"{qb.raw} {qb.unit_raw or ''}".strip(),
    }


# --------------------------------------------------------------------------
# Explanation rendering
# --------------------------------------------------------------------------

def _fmt(x: float | None, decimals: int | None = None) -> str:
    if x is None:
        return "?"
    if not math.isfinite(x):
        return "unbounded"
    if decimals is not None:
        return f"{x:,.{decimals}f}"
    ax = abs(x)
    if ax >= 1000:
        return f"{x:,.0f}"
    if ax >= 1:
        return f"{x:,.2f}".rstrip("0").rstrip(".")
    return f"{x:,.4g}"


def _fmt_interval(lo: float | None, hi: float | None) -> str:
    """Show an interval with enough decimals that its two ends stay distinct.

    Rounding "81,415.375 to 81,415.385" down to "81,415 to 81,415" would hide
    exactly the precision the verdict turns on.
    """
    if lo is None or hi is None:
        return "?"
    if not (math.isfinite(lo) and math.isfinite(hi)):
        return f"{_fmt(lo)}–{_fmt(hi)}"
    width = abs(hi - lo)
    if width == 0:
        return _fmt(lo)
    decimals = max(0, math.ceil(-math.log10(width)) + 1)
    decimals = min(decimals, 6)
    return f"{_fmt(lo, decimals)}–{_fmt(hi, decimals)}"


def _fmt_separation(sep: float | None) -> str:
    """Describe how far apart two figures are in units of their rounding step."""
    if sep is None:
        return ""
    if sep > 100:
        return " (far beyond any rounding difference)"
    return f" ({sep:.1f}x the coarser figure's rounding step)"


def render_explanation(verdict: Verdict, r: dict) -> str:
    """Turn the reasoning record into prose. Every clause traces to a field."""
    vals = r.get("values") or {}
    dims = r.get("dimensions") or {}
    metric = r.get("metric_match", {})
    parts: list[str] = []

    same_note = ""
    if metric.get("source") == "llm":
        same_note = " (matched by model adjudication)"
    elif metric.get("source") == "exact":
        same_note = ""
    parts.append(f"Both facts report {r.get('metric_label', 'the same metric')}{same_note}.")

    if dims.get("agree"):
        parts.append("They agree on " + ", ".join(dims["agree"]) + ".")

    if vals.get("mode") == "interval":
        lo1, hi1 = vals["left_interval"]
        lo2, hi2 = vals["right_interval"]
        unit = vals.get("unit", "")
        parts.append(
            f"A states {vals['left_raw']} = {_fmt(vals['left_value'])} {unit} "
            f"(implied range {_fmt_interval(lo1, hi1)}); "
            f"B states {vals['right_raw']} = {_fmt(vals['right_value'])} {unit} "
            f"(implied range {_fmt_interval(lo2, hi2)})."
        )
        if vals["agree"]:
            parts.append(
                f"The ranges overlap, so the {vals['relative_gap'] * 100:.3g}% difference is "
                "fully accounted for by how precisely each figure was written."
            )
        else:
            sep_txt = _fmt_separation(vals.get("separation_in_ulp"))
            parts.append(
                f"The ranges do not overlap{sep_txt}, so the "
                f"{vals['relative_gap'] * 100:.3g}% difference is a real difference in "
                "what is being claimed, not a rounding artefact."
            )
    elif vals.get("mode") == "state":
        if vals["agree"]:
            parts.append(f"Both assert the same state: {vals['left']!r}.")
        else:
            parts.append(f"A asserts {vals['left']!r} while B asserts {vals['right']!r}.")

    if verdict is Verdict.RECONCILED:
        dim = r.get("explained_by")
        da, db = dims.get("left", {}), dims.get("right", {})
        power = r.get("explanatory_power")
        parts.append(
            f"They differ on {dim}: A is {da.get(dim)!r}, B is {db.get(dim)!r}. "
            f"That difference explains the gap, so this is not a contradiction."
        )
        n_obs = r.get("explanatory_observations") or 0
        if power is not None and n_obs >= 5:
            parts.append(
                f"Across this corpus, {dim} accompanies a change in value in "
                f"{power * 100:.0f}% of the {n_obs} pairs where it is the only "
                "dimension in conflict."
            )
        elif power is not None:
            parts.append(
                f"This corpus has too few comparable pairs ({n_obs}) to say how "
                f"reliably {dim} explains a value difference, so the verdict is "
                "held at low confidence."
            )
        others = [d for d in dims.get("conflict", []) if d != dim]
        if others:
            parts.append(
                "Note that " + ", ".join(others) + " also differ, so the attribution "
                "is not clean-cut."
            )
    elif verdict is Verdict.CONTRADICTS:
        if dims.get("unspecified"):
            parts.append(
                "No recorded dimension distinguishes them"
                f" ({', '.join(dims['unspecified'])} is specified on only one side,"
                " which may be an extraction gap rather than genuine agreement)."
            )
        else:
            parts.append(
                "Every recorded dimension matches, so nothing in the documents "
                "explains the difference."
            )
    elif verdict is Verdict.NEEDS_REVIEW:
        risky = r.get("unspecified_risky") or []
        dim = risky[0] if risky else "a scoping dimension"
        da, db = dims.get("left", {}), dims.get("right", {})
        side = "A" if dim in db and dim not in da else "B"
        parts.append(
            f"Nothing recorded distinguishes these two facts, but {dim} is missing "
            f"from {side}, and on this corpus {dim} accompanies a change in value "
            f"in {r.get('review_power', 0) * 100:.0f}% of the pairs where it is the "
            "only thing that differs. Whether these are even the same claim cannot "
            "be settled from what the documents say, so this is held for review "
            "rather than reported as a contradiction."
        )
    elif verdict is Verdict.RELATED:
        parts.append(
            "The values coincide, but " + ", ".join(dims.get("conflict", []))
            + " differ — so these are two different claims that happen to share a "
            "number, not two sources agreeing."
        )

    return " ".join(parts)


# --------------------------------------------------------------------------
# The engine
# --------------------------------------------------------------------------

class ComparisonEngine:
    def __init__(self, resolver: MetricResolver | None = None):
        self.resolver = resolver or MetricResolver()
        self.stats = CompareStats()

    # -- blocking ---------------------------------------------------------
    def _blocks(self, facts: list[Fact]) -> dict[str, list[Fact]]:
        blocks: dict[str, list[Fact]] = defaultdict(list)
        for f in facts:
            blocks[f.metric_key or normalise_metric(f.metric)].append(f)
        return blocks

    def candidate_pairs(self, facts: list[Fact]) -> list[tuple[Fact, Fact]]:
        """Pairs worth comparing: same metric block, or linked blocks.

        Comparing all N^2 pairs is both wasteful and meaningless -- two facts
        about unrelated metrics have no relationship to report.  Blocking by
        canonical metric, then bridging blocks that are lexically or semantically
        close, keeps the work proportional to how much the corpus actually
        overlaps.
        """
        blocks = self._blocks(facts)
        keys = list(blocks)

        # Within-block pairs first: the metric names are already identical, so
        # these are free -- no adjudication, no model call.
        pairs: list[tuple[Fact, Fact]] = []
        for k in keys:
            group = blocks[k][:MAX_BLOCK]
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    pairs.append((group[i], group[j]))

        # Cross-block links, each scored by how strongly it was proposed.
        linked: dict[tuple[str, str], float] = {}
        for a, b, score in self.resolver.candidate_pairs(keys):
            linked[(a, b)] = max(linked.get((a, b), 0.0), score / 100.0)
        for (a, b), score in self.resolver._sem.items():
            if a in blocks and b in blocks:
                linked[(a, b)] = max(linked.get((a, b), 0.0), score)

        # Strongest first. Adjudication is budgeted, and spending that budget in
        # whatever order pairs happened to be generated meant the cap fell on
        # arbitrary pairs -- on this corpus it cut off the annual report's
        # revenue variants at semantic ranks 1, 8 and 15, which is exactly the
        # cross-document corroboration the layer exists to find.
        for (a, b), _ in sorted(linked.items(), key=lambda kv: -kv[1]):
            ga, gb = blocks.get(a, [])[:MAX_BLOCK], blocks.get(b, [])[:MAX_BLOCK]
            for fa in ga:
                for fb in gb:
                    pairs.append((fa, fb))
        return pairs

    def _dim_power(self, dim: str) -> float:
        st = self.stats.dim_stats.get(dim)
        return st.explanatory_power if st else 0.5

    # -- single pair ------------------------------------------------------
    def evaluate(self, a: Fact, b: Fact) -> dict | None:
        """Everything computable about a pair, before verdict assignment."""
        self.stats.pairs_considered += 1
        if a.id == b.id:
            return None

        match = self.resolver.same_metric(
            a.metric, a.subject, a.quantity.unit_raw if a.quantity else None,
            a.evidence.quote,
            b.metric, b.subject, b.quantity.unit_raw if b.quantity else None,
            b.evidence.quote,
        )
        if not match.get("same"):
            self.stats.skipped_metric_mismatch += 1
            return None

        values = compare_values(a, b)
        if values is None:
            self.stats.skipped_unit_family += 1
            return None

        dims = compare_dimensions(a, b)
        self.stats.pairs_compared += 1

        conflicts = [d for d in dims["conflict"] if d not in PROVENANCE_DIMS]
        return {
            "a": a,
            "b": b,
            "metric_match": match,
            "metric_label": f"{a.metric!r}" if a.metric == b.metric
                            else f"{a.metric!r} / {b.metric!r}",
            "values": values,
            "dimensions": dims,
            "conflicts": conflicts,
        }

    # -- corpus pass ------------------------------------------------------
    def build(self, facts: list[Fact]) -> list[Relation]:
        """Two passes: gather evidence about dimensions, then judge with it."""
        self.stats = CompareStats()
        evaluated: list[dict] = []

        for a, b in self.candidate_pairs(facts):
            ev = self.evaluate(a, b)
            if ev is not None:
                evaluated.append(ev)

        # Pass 1: how often does each dimension accompany a value difference,
        # when it is the *only* thing in conflict?
        for ev in evaluated:
            if len(ev["conflicts"]) != 1:
                continue
            dim = ev["conflicts"][0]
            st = self.stats.dim_stats.setdefault(dim, DimStat(dim=dim))
            st.n_pairs += 1
            if not ev["values"]["agree"]:
                st.n_value_differs += 1

        # Pass 2: assign verdicts using what pass 1 learned.
        relations: list[Relation] = []
        for ev in evaluated:
            relations.append(self._judge(ev))
        return relations

    def _judge(self, ev: dict) -> Relation:
        a, b = ev["a"], ev["b"]
        values, dims, conflicts = ev["values"], ev["dimensions"], ev["conflicts"]
        agree = bool(values["agree"])

        explained_by: str | None = None
        power: float | None = None
        n_obs: int = 0
        review_reason: str | None = None

        # Dimensions one fact records and the other does not. We do not know
        # whether they match; treating silence as agreement is a guess.
        unspecified = [
            d for d in dims["unspecified"] if d not in PROVENANCE_DIMS
        ]
        risky = sorted(
            (d for d in unspecified if self._dim_power(d) >= ABSTAIN_POWER),
            key=lambda d: -self._dim_power(d),
        )

        if agree and not conflicts:
            verdict = Verdict.CORROBORATES
        elif agree:
            # Values coinciding across a dimension that differs is not
            # corroboration -- it is a coincidence. Q1 growth of 6.5% and
            # full-year growth of 6.5% are two different claims that happen to
            # share a number, and presenting that as two sources agreeing would
            # be exactly the false confidence this layer exists to prevent.
            verdict = Verdict.RELATED
        elif not conflicts and risky:
            # Values differ, nothing recorded distinguishes the facts -- but a
            # dimension that usually does is simply absent from one of them.
            # Calling that a contradiction would be asserting something the
            # documents do not support. In due-diligence work a false
            # contradiction costs more than an admitted gap, so we abstain.
            verdict = Verdict.NEEDS_REVIEW
            review_reason = f"{risky[0]}_unknown_on_one_side"
        elif not conflicts:
            verdict = Verdict.CONTRADICTS
        else:
            # Credit the conflicting dimension that most reliably moves values.
            ranked = sorted(
                conflicts,
                key=lambda d: -(
                    self.stats.dim_stats[d].explanatory_power
                    if d in self.stats.dim_stats
                    else 0.5
                ),
            )
            explained_by = ranked[0]
            st = self.stats.dim_stats.get(explained_by)
            power = st.explanatory_power if st else 0.5
            n_obs = st.n_pairs if st else 0
            verdict = Verdict.RECONCILED

        reasoning = {
            "review_reason": review_reason,
            "unspecified_risky": risky,
            "review_power": round(self._dim_power(risky[0]), 3) if risky else None,
            "explanatory_observations": n_obs,
            "metric_match": ev["metric_match"],
            "metric_label": ev["metric_label"],
            "values": values,
            "dimensions": dims,
            "explained_by": explained_by,
            "explanatory_power": round(power, 3) if power is not None else None,
            "left": _fact_brief(a),
            "right": _fact_brief(b),
        }

        confidence = self._confidence(a, b, ev, verdict, power)
        rel = Relation(
            left_id=a.id,
            right_id=b.id,
            verdict=verdict,
            reasoning=reasoning,
            explanation=render_explanation(verdict, reasoning),
            differing_dims=conflicts,
            explained_by=explained_by,
            review_reason=review_reason,
            confidence=confidence,
            cross_document=a.evidence.doc_id != b.evidence.doc_id,
            surface_distance=values.get("relative_gap"),
        )
        return rel

    @staticmethod
    def _confidence(a: Fact, b: Fact, ev: dict, verdict: Verdict, power: float | None) -> float:
        """How much to trust this verdict.

        Three independent things can undermine a relation: either fact may have
        been extracted badly, the metric match may be a model guess rather than
        a string identity, and a reconciliation may rest on a dimension that
        rarely explains anything.  They compose multiplicatively.
        """
        c = min(a.confidence, b.confidence)

        if ev["metric_match"].get("source") == "llm":
            c *= 0.9
        elif ev["metric_match"].get("source") in {"blocked", "unresolved"}:
            c *= 0.6

        # An unspecified dimension on one side means we may simply not know.
        n_unspec = len([d for d in ev["dimensions"]["unspecified"] if d not in PROVENANCE_DIMS])
        if verdict is Verdict.CONTRADICTS and n_unspec:
            c *= max(0.45, 1.0 - 0.18 * n_unspec)

        if verdict is Verdict.RECONCILED and power is not None:
            c *= 0.5 + 0.5 * power
            if len(ev["conflicts"]) > 1:
                c *= 0.85

        if verdict is Verdict.NEEDS_REVIEW:
            # An abstention is a confident statement about our own ignorance.
            c = min(0.9, c * 0.95)

        if verdict is Verdict.CONTRADICTS:
            sep = ev["values"].get("separation_in_ulp")
            if sep is not None and sep < 1.5:
                # Barely separated: plausibly a restatement rather than a conflict.
                c *= 0.8

        return round(max(0.05, min(0.99, c)), 3)


def _fact_brief(f: Fact) -> dict:
    return {
        "id": f.id,
        "metric": f.metric,
        "subject": f.subject,
        "value": f.quantity.canonical_value if f.quantity else None,
        "raw": f"{f.quantity.raw} {f.quantity.unit_raw or ''}".strip() if f.quantity else f.state,
        "unit": f.quantity.canonical_unit if f.quantity else None,
        "period": f.period.label if f.period else None,
        "period_key": f.period.key if f.period else None,
        "qualifiers": f.qualifiers,
        "doc_id": f.evidence.doc_id,
        "doc_title": f.evidence.doc_title,
        "page": f.evidence.page,
        "page_label": f.evidence.page_label,
        "quote": f.evidence.quote,
    }


def salience(rel: Relation) -> float:
    """Ranking score for review: what should a human look at first?

    Weighted toward conflicts, cross-document links and confident verdicts,
    because those are the ones that change a reader's mind.
    """
    base = {
        Verdict.CONTRADICTS: 1.0,
        Verdict.NEEDS_REVIEW: 0.85,   # a human has to look at these
        Verdict.RECONCILED: 0.72,
        Verdict.CORROBORATES: 0.55,
        Verdict.RELATED: 0.3,
    }[rel.verdict]
    s = base * (0.45 + 0.55 * rel.confidence)
    if rel.cross_document:
        s *= 1.5
    if rel.verdict is Verdict.RECONCILED and len(rel.differing_dims) > 1:
        s *= 0.8
    return round(s, 4)
