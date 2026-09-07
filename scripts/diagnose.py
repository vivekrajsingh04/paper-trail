#!/usr/bin/env python
"""Find the system's own failures and record them for the UI.

    python scripts/diagnose.py            # report
    python scripts/diagnose.py --write    # also store the headline failure

The brief asks for an extraction or reasoning failure that was actually found.
Hand-picking one from memory would be weaker than letting the layer indict
itself, so this looks for the signatures of a bad extraction:

* **Intra-document contradictions.**  When a single document appears to state
  two different values for the identical metric, period and scope, the document
  is rarely inconsistent -- far more often a table was read down the wrong
  column.  These are ranked first because each one is either a genuine finding
  or a genuine bug, and both are worth a reviewer's attention.

* **Dense pages with thin coverage.**  Pages carrying many salient quantities
  where few became facts: the recall problem, localised.

* **Relocated attributions.**  Facts whose quote was not on the page the model
  named, recovered by searching the rest of the batch.  Each one is a near-miss
  that the verifier caught.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from factlayer.store import Store  # noqa: E402


def intra_document_conflicts(store: Store, limit: int = 15) -> list[dict]:
    """Contradictions where both facts come from the same document."""
    rows = store.list_relations(verdict="contradicts", cross_document=False, limit=400)
    out = []
    for r in rows:
        rr = r.get("reasoning", {})
        left, right = rr.get("left", {}), rr.get("right", {})
        vals = rr.get("values", {})
        if vals.get("mode") != "interval":
            continue
        out.append({
            "relation_id": r.get("id"),
            "doc": left.get("doc_title") or left.get("doc_id"),
            "metric": left.get("metric"),
            "period": left.get("period"),
            "left": {"raw": left.get("raw"), "page": left.get("page"),
                     "quote": left.get("quote"), "qualifiers": left.get("qualifiers")},
            "right": {"raw": right.get("raw"), "page": right.get("page"),
                      "quote": right.get("quote"), "qualifiers": right.get("qualifiers")},
            "relative_gap": vals.get("relative_gap"),
            "confidence": r.get("confidence"),
            "salience": r.get("salience"),
        })
    out.sort(key=lambda d: -(d.get("salience") or 0))
    return out[:limit]


def relocated_facts(store: Store, limit: int = 20) -> list[dict]:
    """Facts the verifier had to move to a different page than the model claimed."""
    out = []
    for f in store.all_facts():
        if f.verification.get("page_relocated_from") is not None:
            out.append({
                "fact_id": f.id, "metric": f.metric,
                "claimed_page": f.verification["page_relocated_from"],
                "actual_page": f.evidence.page,
                "quote": f.evidence.quote[:100],
            })
            if len(out) >= limit:
                break
    return out


def thin_coverage(store: Store, limit: int = 12) -> list[dict]:
    out = []
    for d in store.list_documents():
        for p in (d.get("stats") or {}).get("lowest_coverage_pages", []):
            out.append({"doc": d["title"], **p})
    out.sort(key=lambda w: (w.get("ratio", 1), -w.get("detected", 0)))
    return out[:limit]


def duplicate_value_different_metric(store: Store, limit: int = 10) -> list[dict]:
    """The same figure extracted under two metric names on the same page.

    A signal that the extractor split one row into two facts, or that a metric
    label was attached to the wrong number.
    """
    seen: dict[tuple, list] = defaultdict(list)
    for f in store.all_facts():
        if f.quantity is None or f.quantity.canonical_value is None:
            continue
        key = (f.evidence.doc_id, f.evidence.page, round(f.quantity.canonical_value, 6))
        seen[key].append(f)
    out = []
    for key, facts in seen.items():
        metrics = {f.metric for f in facts}
        if len(facts) > 1 and len(metrics) > 1:
            out.append({
                "doc_id": key[0], "page": key[1], "value": key[2],
                "metrics": sorted(metrics)[:5],
                "quotes": sorted({f.evidence.quote[:70] for f in facts})[:4],
            })
    return out[:limit]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default="data/factlayer.db")
    ap.add_argument("--write", action="store_true",
                    help="store the headline failure so the UI can show it")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    store = Store(args.db)
    report = {
        "intra_document_conflicts": intra_document_conflicts(store),
        "relocated_facts": relocated_facts(store),
        "thin_coverage_pages": thin_coverage(store),
        "same_value_two_metrics": duplicate_value_different_metric(store),
    }

    print("\n=== intra-document contradictions (likely mis-read tables) ===")
    for c in report["intra_document_conflicts"][:6]:
        print(f"\n  {c['metric']}  [{c['period']}]  in {c['doc']}")
        print(f"    A p{c['left']['page']}: {c['left']['raw']:>16}  {json.dumps(c['left']['qualifiers'])[:70]}")
        print(f"       {c['left']['quote'][:90]!r}")
        print(f"    B p{c['right']['page']}: {c['right']['raw']:>16}  {json.dumps(c['right']['qualifiers'])[:70]}")
        print(f"       {c['right']['quote'][:90]!r}")
        print(f"    gap={c['relative_gap']:.2%}  confidence={c['confidence']}")

    print(f"\n=== facts relocated by the verifier: {len(report['relocated_facts'])} ===")
    for r in report["relocated_facts"][:5]:
        print(f"  {r['metric'][:44]:46s} claimed p{r['claimed_page']} -> actually p{r['actual_page']}")

    print(f"\n=== same value, two metric names on one page: {len(report['same_value_two_metrics'])} ===")
    for d in report["same_value_two_metrics"][:5]:
        print(f"  p{d['page']} value={d['value']:,.2f}  metrics={d['metrics']}")

    print("\n=== thinnest coverage pages ===")
    for w in report["thin_coverage_pages"][:6]:
        print(f"  {w['doc'][:34]:36s} p{w['page']:<4d} {w['covered']}/{w['detected']} "
              f"({w['ratio']:.0%})")

    if args.write and report["intra_document_conflicts"]:
        c = report["intra_document_conflicts"][0]
        store.set_meta("diagnostics", {
            "headline_failure": {
                "title": "Wide financial tables: a value can be bound to the wrong column",
                "description": (
                    f"In {c['doc']}, the layer holds two conflicting values for "
                    f"{c['metric']!r} ({c['period']}): {c['left']['raw']} on page "
                    f"{c['left']['page']} and {c['right']['raw']} on page {c['right']['page']}, "
                    f"a {c['relative_gap']:.1%} gap with no dimension distinguishing them. "
                    "In a statutory filing that is far more likely to be a misread column "
                    "than an inconsistent document: these tables put four figures on one "
                    "line (standalone and consolidated, current and prior year) with the "
                    "headers on a separate line, and PDF text extraction flattens that "
                    "layout away."
                ),
                "evidence": f"A: {c['left']['quote'][:150]}   |   B: {c['right']['quote'][:150]}",
                "handling": (
                    "The system does not hide this. It surfaces the pair as a contradiction "
                    "with reduced confidence, and because both facts keep their exact page "
                    "spans a reviewer can open each one on its source page and settle it in "
                    "seconds. Intra-document contradictions are treated as a review queue "
                    "rather than a conclusion. The real fix, which is not built, is to "
                    "reconstruct the table grid from the word geometry already captured at "
                    "ingest and hand the model a cell with its full header path instead of "
                    "a flattened line."
                ),
                "relation_id": c["relation_id"],
            }
        })
        print("\nwrote headline failure to meta.diagnostics")

    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2))
        print(f"report written to {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
