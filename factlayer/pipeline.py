"""Orchestration: one PDF in, facts and relations out.

Deliberately a thin seam.  Every interesting decision lives in a module this
one calls; the value here is that the order of operations is visible in one
place, and that the honest accounting -- what was extracted, what was rejected,
what was never looked at -- is assembled alongside the results rather than
discarded.
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path

from .canon import normalise_metric
from .candidates import coverage, page_is_interesting, salient
from .compare import ComparisonEngine
from .extract import extract_document, profile_document
from .ingest import ingest_pdf
from .models import Fact
from .store import Store

UPLOAD_DIR = Path("data/uploads")


def _coverage_report(pages, facts: list[Fact]) -> dict:
    """What fraction of detectable quantities became facts, and what did not.

    This is the system's own recall estimate.  It is deliberately reported
    rather than smoothed over: a page where 40 quantities were detected and 6
    became facts is a page the extractor mostly walked past, and a reviewer
    should be able to see that.
    """
    by_page: dict[int, list[str]] = {}
    for f in facts:
        by_page.setdefault(f.evidence.page, []).append(f.evidence.quote)

    detected = covered = 0
    worst: list[dict] = []
    for p in pages:
        if not page_is_interesting(p.text):
            continue
        c = coverage(p.text, by_page.get(p.page, []))
        detected += c["detected"]
        covered += c["covered"]
        if c["detected"] >= 8 and c["ratio"] < 0.5:
            worst.append(
                {
                    "page": p.page,
                    "page_label": p.page_label,
                    "detected": c["detected"],
                    "covered": c["covered"],
                    "ratio": round(c["ratio"], 3),
                    "examples": [m["line"] for m in c["missed"][:4]],
                }
            )
    worst.sort(key=lambda w: (w["ratio"], -w["detected"]))
    return {
        "quantities_detected": detected,
        "quantities_in_facts": covered,
        "coverage_ratio": round(covered / detected, 4) if detected else None,
        "lowest_coverage_pages": worst[:12],
    }


def ingest_path(
    store: Store,
    path: str | Path,
    corpus: str | None = None,
    max_workers: int = 8,
    progress=None,
    force: bool = False,
    max_pages: int | None = None,
) -> dict:
    """Ingest one PDF into the knowledge layer, linking it to what is already there."""
    path = Path(path)
    t0 = time.time()

    doc, pages = ingest_pdf(path, corpus=corpus)

    existing = store.document_by_sha(doc.sha256)
    if existing and not force:
        return {
            "status": "already_ingested",
            "doc_id": existing["doc_id"],
            "title": existing["title"],
            "message": "This document is already in the knowledge layer.",
        }

    # Keep a copy so the UI can render pages and draw evidence boxes later.
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    stored_path = UPLOAD_DIR / f"{doc.doc_id}.pdf"
    if not stored_path.exists():
        shutil.copyfile(path, stored_path)

    if progress:
        progress({"stage": "profiling", "doc_id": doc.doc_id})
    profile = profile_document(doc, pages)

    if progress:
        progress({"stage": "extracting", "doc_id": doc.doc_id})

    def _page_progress(done: int, total: int) -> None:
        if progress:
            progress({"stage": "extracting", "done": done, "total": total, "doc_id": doc.doc_id})

    facts, profile, stats = extract_document(
        doc, pages, profile=profile, max_workers=max_workers,
        progress=_page_progress, max_pages=max_pages,
    )
    for f in facts:
        f.metric_key = normalise_metric(f.metric)

    cov = _coverage_report(pages, facts)
    extraction_stats = {**stats.as_dict(), **cov, "elapsed_s": round(time.time() - t0, 1)}

    store.save_document(doc, str(stored_path), profile.to_dict(), extraction_stats)
    store.save_pages(doc.doc_id, pages)
    store.save_facts(facts)

    if progress:
        progress({"stage": "linking", "doc_id": doc.doc_id, "facts": len(facts)})
    relations, compare_stats = store.link_new_facts(facts)
    store.save_relations(relations)

    from collections import Counter

    verdicts = Counter(r.verdict.value for r in relations)
    return {
        "status": "ok",
        "doc_id": doc.doc_id,
        "title": profile.title or doc.title,
        "publisher": profile.publisher,
        "published": profile.published,
        "pages": doc.n_pages,
        "facts": len(facts),
        "relations": len(relations),
        "new_relations_by_verdict": dict(verdicts),
        "extraction": extraction_stats,
        "comparison": compare_stats,
        "elapsed_s": round(time.time() - t0, 1),
    }


def rebuild_relations(store: Store) -> dict:
    """Recompute every relation from stored facts.

    Not used during normal operation -- ingestion is incremental -- but useful
    after changing the comparison logic, and it is how the two paths are checked
    against each other.
    """
    facts = store.all_facts()
    engine = ComparisonEngine()
    engine.resolver.index_metrics([f.metric for f in facts])
    relations = engine.build(facts)
    with store._lock:
        store._conn.execute("DELETE FROM relations")
        store._conn.commit()
    store.save_relations(relations)
    store.set_meta(
        "dim_stats",
        {
            d: {"pairs": s.n_pairs, "value_differs": s.n_value_differs}
            for d, s in engine.stats.dim_stats.items()
        },
    )
    engine.resolver.save()
    return {"facts": len(facts), "relations": len(relations), **engine.stats.as_dict()}
