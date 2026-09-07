#!/usr/bin/env python
"""Score the fact knowledge layer against a hand-written expectation set.

    python evals/run_evals.py              # offline suites only
    python evals/run_evals.py --with-db    # also check the populated layer
    python evals/run_evals.py --json out.json

Why this exists
---------------
The thresholds in this system are claims about behaviour: that digit-implied
precision separates rounding from disagreement, that three notations for one
fiscal year collapse to one key, that a difference with an explanation is not a
contradiction.  Claims like that decay silently when the code changes.  This
turns each of them into a test that fails loudly.

Three of the four suites need no API key and no database, so they gate every
commit.  The grounding suite runs against whatever has been ingested and is the
one that catches extraction regressions -- including the invariant that matters
most: every stored fact's quote must still be findable at the offsets it claims.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from factlayer.canon import MetricResolver, normalise_metric  # noqa: E402
from factlayer.compare import ComparisonEngine  # noqa: E402
from factlayer.models import Evidence, Fact, Quantity  # noqa: E402
from factlayer.periods import parse_period  # noqa: E402
from factlayer.units import (  # noqa: E402
    build_interval,
    intervals_overlap,
    normalise_unit,
    parse_number,
)

CASES = Path(__file__).parent / "cases.yaml"

GREEN, RED, YELLOW, DIM, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m"


@dataclass
class Suite:
    name: str
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    failures: list[dict] = field(default_factory=list)

    @property
    def total(self) -> int:
        return self.passed + self.failed

    @property
    def rate(self) -> float:
        return self.passed / self.total if self.total else 1.0

    def ok(self, _id: str) -> None:
        self.passed += 1

    def bad(self, _id: str, expected, got, note: str = "") -> None:
        self.failed += 1
        self.failures.append({"id": _id, "expected": expected, "got": got, "note": note})


def _quantity(spec: dict) -> tuple[float, float, float]:
    parsed = parse_number(str(spec["value"]))
    unit = normalise_unit(spec.get("unit"), spec.get("magnitude"))
    return build_interval(parsed, unit)


# -------------------------------------------------------------------- suites

def suite_value_agreement(cases: list[dict]) -> Suite:
    """Does interval overlap classify each pair the way a careful reader would?"""
    s = Suite("value agreement (implied precision)")
    for c in cases:
        _, lo1, hi1 = _quantity(c["left"])
        _, lo2, hi2 = _quantity(c["right"])
        got = "agree" if intervals_overlap((lo1, hi1), (lo2, hi2)) else "disagree"
        if got == c["expect"]:
            s.ok(c["id"])
        else:
            s.bad(c["id"], c["expect"], got,
                  f"[{lo1:.4g},{hi1:.4g}] vs [{lo2:.4g},{hi2:.4g}]")
    return s


def suite_period_resolution(cases: list[dict]) -> Suite:
    s = Suite("period resolution")
    for c in cases:
        p = parse_period(c["text"])
        got = (p.start, p.end) if p else (None, None)
        if got == (c["start"], c["end"]):
            s.ok(c["text"])
        else:
            s.bad(c["text"], f"{c['start']}..{c['end']}", f"{got[0]}..{got[1]}")
    return s


def suite_period_equivalence(cases: list[dict]) -> Suite:
    """Different notations for one window must produce one key."""
    s = Suite("period equivalence across documents")
    for c in cases:
        keys = {parse_period(t).key if parse_period(t) else None for t in c["group"]}
        if len(keys) == 1 and None not in keys:
            s.ok(" == ".join(c["group"]))
        else:
            s.bad(" == ".join(c["group"]), "one shared key", sorted(map(str, keys)))
    return s


def _mk_fact(spec: dict) -> Fact:
    q = None
    if "value" in spec:
        parsed = parse_number(str(spec["value"]))
        unit = normalise_unit(spec.get("unit"))
        canon, lo, hi = build_interval(parsed, unit)
        q = Quantity(
            raw=str(spec["value"]), value=parsed.value, unit=unit.canonical,
            unit_raw=spec.get("unit"), canonical_value=canon,
            canonical_unit=unit.canonical, lo=lo, hi=hi, ulp=parsed.ulp,
        )
    return Fact(
        metric=spec["metric"],
        metric_key=normalise_metric(spec["metric"]),
        subject=spec.get("subject", "Delhivery Limited"),
        quantity=q,
        period=parse_period(spec.get("period")),
        qualifiers={str(k): str(v) for k, v in (spec.get("qualifiers") or {}).items()},
        evidence=Evidence(
            doc_id=spec.get("doc", "doc"), doc_title=spec.get("doc", "doc"), page=0,
            char_start=0, char_end=1, snippet="", quote=str(spec.get("value", "")),
        ),
        confidence=0.9,
    )


def suite_relation_verdicts(cases: list[dict]) -> Suite:
    """The reasoning engine, on facts built by hand so extraction is not in scope."""
    s = Suite("relation verdicts")
    engine = ComparisonEngine(MetricResolver(use_llm=False))
    for c in cases:
        a, b = _mk_fact(c["left"]), _mk_fact(c["right"])
        ev = engine.evaluate(a, b)
        if ev is None:
            got = "no_relation"
            explained = None
        else:
            rel = engine._judge(ev)
            got = rel.verdict.value
            explained = rel.explained_by
        if got != c["expect"]:
            s.bad(c["id"], c["expect"], got)
            continue
        if c.get("explained_by") and explained != c["explained_by"]:
            s.bad(c["id"], f"{c['expect']} by {c['explained_by']}", f"{got} by {explained}")
            continue
        s.ok(c["id"])
    return s


def suite_grounding(db: str) -> Suite:
    """Every stored fact must still be findable at the offsets it claims.

    This is the invariant that makes the evidence trustworthy.  If it ever
    fails, a citation in the UI is pointing at the wrong ink.
    """
    from factlayer.store import Store

    s = Suite("grounding invariants (live database)")
    store = Store(db)
    facts = store.all_facts()
    if not facts:
        s.skipped = 1
        return s

    pages: dict[tuple[str, int], str] = {}
    for f in facts:
        key = (f.evidence.doc_id, f.evidence.page)
        if key not in pages:
            rec = store.page_text(*key)
            pages[key] = (rec or {}).get("text", "")
        text = pages[key]
        e = f.evidence

        if not text:
            s.bad(f.id, "page text stored", "missing")
            continue
        if text[e.char_start:e.char_end] != e.quote:
            s.bad(f.id, f"quote at [{e.char_start}:{e.char_end}]",
                  repr(text[e.char_start:e.char_end][:60]), note=repr(e.quote[:60]))
            continue
        if f.quantity is not None:
            digits = "".join(ch for ch in f.quantity.raw if ch.isdigit())
            if digits and digits not in "".join(ch for ch in e.quote if ch.isdigit()):
                s.bad(f.id, f"value {f.quantity.raw} inside quote", repr(e.quote[:60]))
                continue
        if f.quantity is not None and not f.quantity.has_interval:
            s.bad(f.id, "precision interval present", "missing")
            continue
        s.ok(f.id)
    return s


def suite_calibration(db: str) -> dict | None:
    """Are stated confidences meaningful, or decorative?

    Without per-relation ground truth we cannot measure accuracy directly, so
    this reports the distribution and the share of each verdict that survives a
    confidence filter -- enough to see whether the score separates anything.
    A labelled sample is the obvious next step and is noted in the README.
    """
    from factlayer.store import Store

    store = Store(db)
    rels = store.list_relations(limit=100000)
    if not rels:
        return None
    buckets: dict[str, dict[str, int]] = {}
    for r in rels:
        b = f"{int(r['confidence'] * 10) / 10:.1f}"
        d = buckets.setdefault(b, {})
        d[r["verdict"]] = d.get(r["verdict"], 0) + 1
        d["n"] = d.get("n", 0) + 1
    return {
        "relations": len(rels),
        "mean_confidence": round(sum(r["confidence"] for r in rels) / len(rels), 3),
        "by_bucket": dict(sorted(buckets.items())),
    }


# -------------------------------------------------------------------- runner

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--with-db", action="store_true", help="also run live-database suites")
    ap.add_argument("--db", default="data/factlayer.db")
    ap.add_argument("--json", default=None, help="write a machine-readable report here")
    ap.add_argument("--strict", action="store_true", help="exit non-zero on any failure")
    args = ap.parse_args()

    spec = yaml.safe_load(CASES.read_text())
    suites = [
        suite_value_agreement(spec["value_agreement"]),
        suite_period_resolution(spec["period_resolution"]),
        suite_period_equivalence(spec["period_equivalence"]),
        suite_relation_verdicts(spec["relation_verdicts"]),
    ]
    calibration = None
    if args.with_db:
        suites.append(suite_grounding(args.db))
        calibration = suite_calibration(args.db)

    print()
    width = max(len(s.name) for s in suites) + 2
    for s in suites:
        if s.skipped and not s.total:
            print(f"  {YELLOW}skip{RESET}  {s.name:<{width}} nothing ingested yet")
            continue
        colour = GREEN if not s.failed else RED
        mark = "pass" if not s.failed else "FAIL"
        print(f"  {colour}{mark}{RESET}  {s.name:<{width}} "
              f"{s.passed}/{s.total}  ({s.rate:.0%})")
        for f in s.failures[:6]:
            print(f"        {DIM}{f['id']}{RESET}: expected {f['expected']}, got {f['got']}"
                  + (f"  {DIM}{f['note']}{RESET}" if f.get("note") else ""))
        if len(s.failures) > 6:
            print(f"        {DIM}... and {len(s.failures) - 6} more{RESET}")

    total_p = sum(s.passed for s in suites)
    total_f = sum(s.failed for s in suites)
    print(f"\n  {total_p}/{total_p + total_f} checks passed")

    if calibration:
        print(f"\n  {DIM}confidence distribution over {calibration['relations']} relations "
              f"(mean {calibration['mean_confidence']}){RESET}")
        for bucket, counts in calibration["by_bucket"].items():
            bits = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()) if k != "n")
            print(f"    {bucket}  n={counts['n']:<5} {DIM}{bits}{RESET}")

    if args.json:
        Path(args.json).write_text(json.dumps(
            {
                "suites": [
                    {"name": s.name, "passed": s.passed, "failed": s.failed,
                     "rate": s.rate, "failures": s.failures}
                    for s in suites
                ],
                "calibration": calibration,
            },
            indent=2,
        ))
        print(f"\n  report written to {args.json}")

    return 1 if (total_f and args.strict) else 0


if __name__ == "__main__":
    raise SystemExit(main())
