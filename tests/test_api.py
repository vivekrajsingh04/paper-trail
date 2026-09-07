"""API surface tests against a temporary store seeded with hand-built facts.

No model calls and no PDFs: these check that the HTTP layer exposes the
reasoning record faithfully and that filters behave, which is what a reviewer
poking at the API will actually rely on.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from factlayer import api as api_mod
from factlayer.canon import MetricResolver
from factlayer.compare import ComparisonEngine
from factlayer.store import Store
from tests.test_compare import mk


@pytest.fixture()
def client(tmp_path, monkeypatch):
    store = Store(tmp_path / "test.db")

    facts = [
        mk("revenue from services", "8,142", "Rs. crore", "FY24", "deck",
           scope="consolidated"),
        mk("revenue from services", "81,415.38", "Rs. million", "FY24", "annual-report",
           scope="consolidated"),
        mk("real GDP growth", "6.5", "per cent", "FY26", "rbi", subject="India",
           basis="projection"),
        mk("real GDP growth", "6.6", "per cent", "FY2025/26", "imf", subject="India",
           basis="projection"),
        mk("revenue from operations", "74,540.82", "Rs. million", "FY24", "annual-report",
           scope="standalone"),
        mk("revenue from operations", "81,415.38", "Rs. million", "FY24", "annual-report",
           scope="consolidated"),
    ]
    store.save_facts(facts)

    engine = ComparisonEngine(MetricResolver(use_llm=False))
    store.save_relations(engine.build(facts))

    monkeypatch.setattr(api_mod, "_store", store)
    return TestClient(api_mod.app)


def test_health_and_summary(client):
    h = client.get("/api/health").json()
    assert h["status"] == "ok"
    assert h["facts"] == 6

    s = client.get("/api/summary").json()
    assert s["facts"] == 6
    assert s["relations"]["total"] > 0


def test_relations_filter_by_verdict(client):
    for verdict in ("corroborates", "contradicts", "reconciled"):
        d = client.get(f"/api/relations?verdict={verdict}").json()
        assert all(r["verdict"] == verdict for r in d["relations"]), verdict

    bad = client.get("/api/relations?verdict=nonsense")
    assert bad.status_code == 400


def test_relations_filter_cross_document(client):
    d = client.get("/api/relations?cross_document=true").json()
    assert d["relations"], "expected at least one cross-document relation"
    assert all(r["cross_document"] for r in d["relations"])


def test_relation_carries_full_reasoning(client):
    """The audit trail is the product; it must survive the round trip."""
    d = client.get("/api/relations?verdict=corroborates").json()
    rel = d["relations"][0]
    r = rel["reasoning"]

    assert r["values"]["mode"] == "interval"
    assert len(r["values"]["left_interval"]) == 2
    assert "agree" in r["dimensions"]
    assert r["metric_match"]["source"] in {"exact", "llm", "lexical_high", "cache"}
    assert rel["explanation"]

    detail = client.get(f"/api/relations/{rel['id']}").json()
    assert detail["left"]["metric"]
    assert detail["right"]["metric"]


def test_cases_endpoint_returns_the_required_three(client):
    c = client.get("/api/cases").json()
    assert c["corroboration"]["verdict"] == "corroborates"
    assert c["contradiction"]["verdict"] == "contradicts"
    assert c["reconciled"]["verdict"] == "reconciled"
    # Case 3 should prefer an explanation that is not merely a period difference.
    assert c["reconciled"]["explained_by"] == "scope"


def test_facts_search_and_detail(client):
    d = client.get("/api/facts?q=revenue").json()
    assert d["count"] >= 4

    fid = d["facts"][0]["id"]
    detail = client.get(f"/api/facts/{fid}").json()
    assert detail["fact"]["id"] == fid
    assert "relations" in detail

    assert client.get("/api/facts/nope").status_code == 404


def test_evidence_endpoint_reports_missing_source_gracefully(client):
    d = client.get("/api/facts?q=revenue").json()
    fid = d["facts"][0]["id"]
    ev = client.get(f"/api/evidence/{fid}").json()
    assert ev["fact_id"] == fid
    assert "image_url" in ev
    # These fixtures have no backing PDF, so rendering must 404 rather than crash.
    assert client.get(ev["image_url"]).status_code == 404


def test_upload_rejects_non_pdf(client):
    r = client.post("/api/documents", files={"file": ("x.txt", b"hello", "text/plain")})
    assert r.status_code == 400
