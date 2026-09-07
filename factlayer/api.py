"""HTTP interface to the knowledge layer.

Three things this exposes that a fact API usually does not, and which are the
point of the project:

* `/api/relations/{id}` returns the full reasoning record, not just a verdict --
  the intervals, the dimension comparison, how the metric names were matched and
  by what means.  A reviewer can disagree with a conclusion and see exactly which
  step produced it.
* `/api/evidence/{fact_id}` returns the rendered source page with the rectangles
  needed to draw a box around the figure the fact came from.
* `/api/diagnostics` returns what the system got wrong or never looked at:
  rejected extractions, per-page coverage, and the learned dimension statistics.
"""

from __future__ import annotations

import shutil
import tempfile
import threading
import uuid
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, Response

from .config import describe
from .ingest import render_page_png
from .models import Verdict
from .pipeline import ingest_path, rebuild_relations
from .store import Store

app = FastAPI(
    title="Fact Knowledge Layer",
    description="Extracts evidence-backed facts from PDFs and computes how they relate.",
    version="1.0.0",
)

_store: Store | None = None
_store_lock = threading.Lock()

# In-flight ingestion jobs.  Kept in memory on purpose: they are progress
# reporting for a single-process prototype, not durable state.
JOBS: dict[str, dict] = {}
WEB_DIR = Path(__file__).parent / "web"


def store() -> Store:
    global _store
    with _store_lock:
        if _store is None:
            _store = Store()
    return _store


# --------------------------------------------------------------------------
# Meta
# --------------------------------------------------------------------------

@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", **describe(), "facts": store().fact_count()}


@app.get("/api/summary")
def summary() -> dict:
    return store().summary()


@app.get("/api/documents")
def documents() -> list[dict]:
    return store().list_documents()


# --------------------------------------------------------------------------
# Ingestion
# --------------------------------------------------------------------------

def _run_ingest(job_id: str, tmp_path: str, filename: str, corpus: str | None) -> None:
    job = JOBS[job_id]
    try:
        def progress(ev: dict) -> None:
            job.update({k: v for k, v in ev.items() if k != "doc_id"})

        result = ingest_path(store(), tmp_path, corpus=corpus, progress=progress)
        job.update({"stage": "done", "status": "ok", "result": result})
    except Exception as exc:  # noqa: BLE001
        job.update({"stage": "error", "status": "error", "error": str(exc)[:500]})
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@app.post("/api/documents")
async def upload(
    background: BackgroundTasks,
    file: UploadFile = File(...),
    corpus: str | None = Query(None, description="label grouping related documents"),
) -> dict:
    """Accept a PDF and ingest it in the background.

    Returns a job id immediately; poll `/api/jobs/{id}` for progress.  Extraction
    on a large document takes minutes, so blocking the request would be a poor
    interface and a fragile one.
    """
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(400, "only PDF uploads are supported")

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    shutil.copyfileobj(file.file, tmp)
    tmp.close()

    job_id = uuid.uuid4().hex[:12]
    JOBS[job_id] = {"id": job_id, "filename": file.filename, "stage": "queued", "status": "running"}
    background.add_task(_run_ingest, job_id, tmp.name, file.filename or "upload.pdf", corpus)
    return {"job_id": job_id, "status": "running"}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str) -> dict:
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "unknown job")
    return job


@app.post("/api/rebuild")
def rebuild() -> dict:
    """Recompute all relations from stored facts (after a logic change)."""
    return rebuild_relations(store())


# --------------------------------------------------------------------------
# Facts
# --------------------------------------------------------------------------

@app.get("/api/facts")
def facts(
    q: str | None = None,
    doc_id: str | None = None,
    metric_key: str | None = None,
    period_key: str | None = None,
    limit: int = Query(60, le=500),
    offset: int = 0,
) -> dict:
    found = store().search_facts(
        q=q, doc_id=doc_id, metric_key=metric_key, period_key=period_key,
        limit=limit, offset=offset,
    )
    return {"count": len(found), "facts": [f.model_dump() for f in found]}


@app.get("/api/facts/{fact_id}")
def fact_detail(fact_id: str) -> dict:
    f = store().get_fact(fact_id)
    if f is None:
        raise HTTPException(404, "unknown fact")
    return {
        "fact": f.model_dump(),
        "relations": store().list_relations(fact_id=fact_id, limit=60),
    }


# --------------------------------------------------------------------------
# Relations
# --------------------------------------------------------------------------

@app.get("/api/relations")
def relations(
    verdict: str | None = Query(None, description="corroborates|contradicts|reconciled"),
    cross_document: bool | None = None,
    min_confidence: float = 0.0,
    limit: int = Query(50, le=300),
    offset: int = 0,
) -> dict:
    if verdict and verdict not in {v.value for v in Verdict}:
        raise HTTPException(400, f"verdict must be one of {[v.value for v in Verdict]}")
    rows = store().list_relations(
        verdict=verdict, cross_document=cross_document,
        min_confidence=min_confidence, limit=limit, offset=offset,
    )
    return {"count": len(rows), "counts": store().relation_counts(), "relations": rows}


@app.get("/api/relations/{relation_id}")
def relation_detail(relation_id: str) -> dict:
    rows = store()._q("SELECT payload, salience FROM relations WHERE id=?", (relation_id,))
    if not rows:
        raise HTTPException(404, "unknown relation")
    import json

    rel = json.loads(rows[0]["payload"])
    rel["salience"] = rows[0]["salience"]
    left = store().get_fact(rel["left_id"])
    right = store().get_fact(rel["right_id"])
    rel["left"] = left.model_dump() if left else None
    rel["right"] = right.model_dump() if right else None
    return rel


# --------------------------------------------------------------------------
# Evidence
# --------------------------------------------------------------------------

@app.get("/api/evidence/{fact_id}")
def evidence(fact_id: str) -> dict:
    """Everything needed to show a fact in its source: page, span, rectangles."""
    f = store().get_fact(fact_id)
    if f is None:
        raise HTTPException(404, "unknown fact")
    page = store().page_text(f.evidence.doc_id, f.evidence.page)
    return {
        "fact_id": f.id,
        "doc_id": f.evidence.doc_id,
        "doc_title": f.evidence.doc_title,
        "page": f.evidence.page,
        "page_label": f.evidence.page_label,
        "quote": f.evidence.quote,
        "snippet": f.evidence.snippet,
        "char_start": f.evidence.char_start,
        "char_end": f.evidence.char_end,
        "bboxes": [b.model_dump() for b in f.evidence.bboxes],
        "page_text": (page or {}).get("text", ""),
        "image_url": f"/api/page-image/{f.evidence.doc_id}/{f.evidence.page}",
    }


@app.get("/api/page-image/{doc_id}/{page}")
def page_image(doc_id: str, page: int, dpi: int = Query(130, ge=50, le=300)) -> Response:
    path = store().document_path(doc_id)
    if not path or not Path(path).exists():
        raise HTTPException(404, "source PDF not available")
    try:
        png, width, height = render_page_png(path, page, dpi=dpi)
    except (IndexError, ValueError) as exc:
        raise HTTPException(404, f"cannot render page: {exc}") from exc
    return Response(
        content=png,
        media_type="image/png",
        headers={
            "X-Page-Width": str(width),
            "X-Page-Height": str(height),
            "Access-Control-Expose-Headers": "X-Page-Width, X-Page-Height",
            "Cache-Control": "public, max-age=3600",
        },
    )


# --------------------------------------------------------------------------
# The four required cases, selected automatically
# --------------------------------------------------------------------------

@app.get("/api/cases")
def cases() -> dict:
    """Pick the strongest live example of each case the brief asks for.

    Selected by query against whatever is currently stored, not curated by
    hand -- so the demo reflects a real run, and re-running on new documents
    surfaces that corpus's own examples.
    """
    s = store()

    def top(verdict: str, cross: bool | None, **kw) -> dict | None:
        rows = s.list_relations(verdict=verdict, cross_document=cross, limit=1, **kw)
        return rows[0] if rows else None

    corroborated = (top("corroborates", True, min_confidence=0.5)
                    or top("corroborates", None, min_confidence=0.5))
    contradiction = (top("contradicts", True, min_confidence=0.4)
                     or top("contradicts", None, min_confidence=0.4))

    # For the reconciled case prefer one explained by a dimension other than
    # period: a period difference is the least surprising kind of explanation.
    reconciled = None
    for r in s.list_relations(verdict="reconciled", limit=120, min_confidence=0.3):
        if r.get("explained_by") and r["explained_by"] != "period":
            reconciled = r
            break
    reconciled = reconciled or top("reconciled", None)

    diagnostics = s.get_meta("diagnostics", {}) or {}

    return {
        "corroboration": corroborated,
        "contradiction": contradiction,
        "reconciled": reconciled,
        "failure": diagnostics.get("headline_failure"),
        "note": (
            "Selected by query from the current knowledge layer, not hard-coded. "
            "Re-running on a different corpus yields that corpus's own examples."
        ),
    }


@app.get("/api/diagnostics")
def diagnostics() -> dict:
    """What the system rejected, missed, or learned about its own dimensions."""
    s = store()
    docs = s.list_documents()
    per_doc = []
    for d in docs:
        st = d.get("stats") or {}
        per_doc.append(
            {
                "doc_id": d["doc_id"],
                "title": d["title"],
                "pages": d["n_pages"],
                "facts": d["n_facts"],
                "pages_sent": st.get("pages_sent"),
                "facts_proposed": st.get("facts_proposed"),
                "facts_kept": st.get("facts_kept"),
                "verification_pass_rate": st.get("verification_pass_rate"),
                "rejected_quote_not_found": st.get("rejected_quote_not_found"),
                "rejected_value_mismatch": st.get("rejected_value_mismatch"),
                "coverage_ratio": st.get("coverage_ratio"),
                "quantities_detected": st.get("quantities_detected"),
                "quantities_in_facts": st.get("quantities_in_facts"),
                "lowest_coverage_pages": st.get("lowest_coverage_pages", [])[:5],
                "rejections": st.get("rejections", [])[:10],
            }
        )
    return {
        "documents": per_doc,
        "dimensions": s.get_meta("dim_stats", {}),
        "notes": s.get_meta("diagnostics", {}),
    }


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    path = WEB_DIR / "index.html"
    if not path.exists():
        return HTMLResponse("<h1>Fact Knowledge Layer</h1><p>UI not built.</p>")
    return HTMLResponse(path.read_text())


@app.get("/app.js")
def app_js() -> FileResponse:
    return FileResponse(WEB_DIR / "app.js", media_type="application/javascript")


@app.get("/style.css")
def app_css() -> FileResponse:
    return FileResponse(WEB_DIR / "style.css", media_type="text/css")
