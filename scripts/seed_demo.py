#!/usr/bin/env python
"""Seed a knowledge layer from hand-built facts, for UI work without a model.

    python scripts/seed_demo.py --db data/demo.db

Every fact here is a real figure from the starter documents, and each one is
grounded the same way the extractor grounds facts: the quote is located in the
actual page text, and the bounding boxes come from the actual PDF geometry.  So
the evidence viewer exercises the real path -- nothing about the rendering or
the reasoning is faked, only the model call that would have proposed the fact.

This exists because iterating on the interface should not consume an API quota,
and because it gives the test suite a realistic fixture.  It is not part of the
normal pipeline.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from factlayer.canon import MetricResolver, normalise_metric  # noqa: E402
from factlayer.compare import ComparisonEngine  # noqa: E402
from factlayer.extract import _assemble_fact, _find_quote  # noqa: E402
from factlayer.extract import DocProfile  # noqa: E402
from factlayer.ingest import ingest_pdf  # noqa: E402
from factlayer.pipeline import UPLOAD_DIR  # noqa: E402
from factlayer.store import Store  # noqa: E402

ROOT = Path("starter-datasets/starter-datasets")

# (relative path, corpus, profile) for each document we seed from.
DOCS = {
    "deck": (ROOT / "delhivery/03-delhivery-q4-fy24-earnings-presentation.pdf", "delhivery",
             DocProfile(title="Delhivery Q4 & FY24 Earnings Presentation",
                        publisher="Delhivery Limited", doc_type="earnings presentation",
                        primary_entity="Delhivery Limited", published="2024-05-17")),
    "ar": (ROOT / "delhivery/02-delhivery-annual-report-fy24-excerpt.pdf", "delhivery",
           DocProfile(title="Delhivery Annual Report FY24", publisher="Delhivery Limited",
                      doc_type="annual report", primary_entity="Delhivery Limited",
                      published="2024-08-01")),
    "survey": (ROOT / "india-macroeconomy/01-india-economic-survey-2024-25-excerpt.pdf",
               "india-macroeconomy",
               DocProfile(title="Economic Survey 2024-25", publisher="Government of India",
                          doc_type="economic survey", primary_entity="India",
                          published="2025-01-31")),
    "rbi": (ROOT / "india-macroeconomy/02-rbi-annual-report-2024-25-excerpt.pdf",
            "india-macroeconomy",
            DocProfile(title="RBI Annual Report 2024-25", publisher="Reserve Bank of India",
                       doc_type="annual report", primary_entity="India",
                       published="2025-05-29")),
    "imf": (ROOT / "india-macroeconomy/03-imf-india-2025-article-iv-excerpt.pdf",
            "india-macroeconomy",
            DocProfile(title="IMF India 2025 Article IV Consultation",
                       publisher="International Monetary Fund", doc_type="staff report",
                       primary_entity="India", published="2025-11-25")),
}

# Facts to seed. `find` is a distinctive string searched for on the page, which
# becomes the grounded quote. Nothing is invented: if the string is not on the
# page, the fact is skipped and reported.
SEED = [
    # -- Case 1: corroboration across documents, different units --------------
    ("deck", 5, "₹8,142 Cr", "revenue from services", "8,142", "₹ Cr", "FY24",
     {"scope": "consolidated", "basis": "reported"}),
    ("ar", 3, "₹81,415Mn", "revenue from services", "81,415", "₹ Mn", "FY24",
     {"scope": "consolidated", "basis": "reported"}),

    ("deck", 5, "₹127Cr", "EBITDA", "127", "₹ Cr", "FY24",
     {"scope": "consolidated", "basis": "reported"}),
    ("ar", 3, "₹1,266Mn", "EBITDA", "1,266", "₹ Mn", "FY24",
     {"scope": "consolidated", "basis": "reported"}),

    # -- Case 3b: reconciled by reporting scope -------------------------------
    ("ar", 21, "74,540.82", "revenue from operations", "74,540.82", "₹ Mn", "FY24",
     {"scope": "standalone", "basis": "reported"}),
    ("ar", 21, "81,415.38", "revenue from operations", "81,415.38", "₹ Mn", "FY24",
     {"scope": "consolidated", "basis": "reported"}),

    # -- Case 3c: reconciled by period ----------------------------------------
    ("deck", 6, "₹2,076 Cr", "revenue from services", "2,076", "₹ Cr", "Q4 FY24",
     {"scope": "consolidated", "basis": "reported"}),

    # -- Case 3a: reconciled by estimate vintage ------------------------------
    ("survey", 13, "6.4 per cent", "real GDP growth", "6.4", "per cent", "FY25",
     {"basis": "first_advance_estimate"}),
    ("rbi", 7, "6.5 per cent", "real GDP growth", "6.5", "per cent", "2024-25",
     {"basis": "provisional_estimate"}),

    # -- Case 2: genuine contradiction between institutions -------------------
    ("rbi", 16, "6.5 per cent", "real GDP growth", "6.5", "per cent", "2025-26",
     {"basis": "projection"}),
    ("imf", 12, "6.6 percent", "real GDP growth", "6.6", "percent", "FY2025/26",
     {"basis": "projection"}),

    # -- Corroboration of the FY25 outturn across two institutions ------------
    ("imf", 9, "6.5 percent", "real GDP growth", "6.5", "percent", "FY2024/25",
     {"basis": "provisional_estimate"}),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default="data/demo.db")
    args = ap.parse_args()

    db = Path(args.db)
    db.unlink(missing_ok=True)
    store = Store(db)

    loaded: dict[str, tuple] = {}
    for key, (path, corpus, profile) in DOCS.items():
        if not path.exists():
            print(f"  missing {path}")
            continue
        doc, pages = ingest_pdf(path, corpus=corpus)
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        stored = UPLOAD_DIR / f"{doc.doc_id}.pdf"
        if not stored.exists():
            shutil.copyfile(path, stored)
        store.save_document(doc, str(stored), profile.to_dict(),
                            {"note": "seeded fixture; facts hand-specified, grounding real"})
        store.save_pages(doc.doc_id, pages)
        loaded[key] = (doc, pages, profile)
        print(f"  loaded {key}: {doc.n_pages} pages")

    facts = []
    skipped = []
    for key, page_no, find, metric, value, unit, period, quals in SEED:
        if key not in loaded:
            continue
        doc, pages, profile = loaded[key]
        page = pages[page_no]
        span = _find_quote(page.text, find)
        if span is None:
            skipped.append((key, page_no, find))
            continue
        rf = {"kind": "numeric", "metric": metric, "subject": profile.primary_entity,
              "value_raw": value, "unit_raw": unit, "period_raw": period,
              "qualifiers": quals, "confidence": 0.92}
        fact = _assemble_fact(doc, profile, page, rf, span, "seed")
        if not isinstance(fact, str) and fact is not None:
            fact.metric_key = normalise_metric(metric)
            fact.extractor = "seed"
            facts.append(fact)
        else:
            skipped.append((key, page_no, find))

    store.save_facts(facts)
    print(f"\n  {len(facts)} facts grounded, {len(skipped)} skipped")
    for s in skipped:
        print(f"    skipped: {s[0]} p{s[1]} {s[2]!r} not found on that page")

    engine = ComparisonEngine(MetricResolver(use_llm=False))
    rels = engine.build(facts)
    store.save_relations(rels)
    store.set_meta("dim_stats", {
        d: {"pairs": s.n_pairs, "value_differs": s.n_value_differs}
        for d, s in engine.stats.dim_stats.items()
    })

    from collections import Counter
    print(f"  {len(rels)} relations: {dict(Counter(r.verdict.value for r in rels))}")
    print("\n  run:  uvicorn factlayer.api:app --reload")
    print(f"  with: FACTLAYER_DB={db}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
