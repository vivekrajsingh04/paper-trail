"""SQLite persistence for the knowledge layer.

Chosen for the same reason a prototype should choose anything: it is the least
machinery that meets the requirement.  The layer needs durable facts, indexed
lookup by metric and period, and the ability to add a document without
recomputing the world.  A single file gives all three, ships inside the repo,
and needs no service running for a reviewer to try it.

Incremental ingestion
---------------------
Adding a document compares its new facts against the facts already stored, plus
against each other -- never the full corpus against itself.  Ingesting the Nth
document costs O(new x existing) rather than O(total^2), so the knowledge layer
grows by addition instead of rebuild.  Corpus-level dimension statistics are the
one genuinely global quantity, and they are refreshed incrementally from stored
counts rather than by re-judging every pair.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any, Iterable

from .compare import ComparisonEngine, salience
from .models import Document, Fact, Relation

DB_PATH = Path("data/factlayer.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id      TEXT PRIMARY KEY,
    title       TEXT,
    filename    TEXT,
    path        TEXT,
    n_pages     INTEGER,
    sha256      TEXT UNIQUE,
    corpus      TEXT,
    publisher   TEXT,
    published   TEXT,
    profile     TEXT,
    stats       TEXT,
    ingested_at TEXT
);

CREATE TABLE IF NOT EXISTS pages (
    doc_id     TEXT,
    page       INTEGER,
    page_label TEXT,
    text       TEXT,
    PRIMARY KEY (doc_id, page)
);

CREATE TABLE IF NOT EXISTS facts (
    id              TEXT PRIMARY KEY,
    doc_id          TEXT,
    kind            TEXT,
    metric          TEXT,
    metric_key      TEXT,
    subject_key     TEXT,
    period_key      TEXT,
    canonical_unit  TEXT,
    canonical_value REAL,
    page            INTEGER,
    confidence      REAL,
    payload         TEXT
);
CREATE INDEX IF NOT EXISTS idx_facts_metric  ON facts(metric_key);
CREATE INDEX IF NOT EXISTS idx_facts_doc     ON facts(doc_id);
CREATE INDEX IF NOT EXISTS idx_facts_period  ON facts(period_key);

CREATE TABLE IF NOT EXISTS relations (
    id             TEXT PRIMARY KEY,
    left_id        TEXT,
    right_id       TEXT,
    verdict        TEXT,
    confidence     REAL,
    salience       REAL,
    cross_document INTEGER,
    explained_by   TEXT,
    payload        TEXT
);
CREATE INDEX IF NOT EXISTS idx_rel_verdict ON relations(verdict);
CREATE INDEX IF NOT EXISTS idx_rel_left    ON relations(left_id);
CREATE INDEX IF NOT EXISTS idx_rel_right   ON relations(right_id);
CREATE INDEX IF NOT EXISTS idx_rel_sal     ON relations(salience DESC);

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);
"""


class Store:
    def __init__(self, path: str | Path = DB_PATH):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    # -- helpers ----------------------------------------------------------
    def _q(self, sql: str, args: Iterable = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(sql, tuple(args)).fetchall()

    def close(self) -> None:
        self._conn.close()

    # -- documents --------------------------------------------------------
    def document_by_sha(self, sha: str) -> dict | None:
        rows = self._q("SELECT * FROM documents WHERE sha256=?", (sha,))
        return dict(rows[0]) if rows else None

    def save_document(self, doc: Document, path: str, profile: dict, stats: dict) -> None:
        with self._lock:
            self._conn.execute(
                """INSERT OR REPLACE INTO documents
                   (doc_id,title,filename,path,n_pages,sha256,corpus,publisher,published,
                    profile,stats,ingested_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    doc.doc_id, profile.get("title") or doc.title, doc.filename, str(path),
                    doc.n_pages, doc.sha256, doc.corpus, profile.get("publisher"),
                    profile.get("published"), json.dumps(profile), json.dumps(stats),
                    doc.ingested_at,
                ),
            )
            self._conn.commit()

    def save_pages(self, doc_id: str, pages) -> None:
        with self._lock:
            self._conn.executemany(
                "INSERT OR REPLACE INTO pages (doc_id,page,page_label,text) VALUES (?,?,?,?)",
                [(doc_id, p.page, p.page_label, p.text) for p in pages],
            )
            self._conn.commit()

    def list_documents(self) -> list[dict]:
        rows = self._q("SELECT * FROM documents ORDER BY corpus, filename")
        out = []
        for r in rows:
            d = dict(r)
            d["profile"] = json.loads(d.get("profile") or "{}")
            d["stats"] = json.loads(d.get("stats") or "{}")
            d["n_facts"] = self._q(
                "SELECT COUNT(*) c FROM facts WHERE doc_id=?", (d["doc_id"],)
            )[0]["c"]
            out.append(d)
        return out

    def page_text(self, doc_id: str, page: int) -> dict | None:
        rows = self._q("SELECT * FROM pages WHERE doc_id=? AND page=?", (doc_id, page))
        return dict(rows[0]) if rows else None

    def document_path(self, doc_id: str) -> str | None:
        rows = self._q("SELECT path FROM documents WHERE doc_id=?", (doc_id,))
        return rows[0]["path"] if rows else None

    # -- facts ------------------------------------------------------------
    def save_facts(self, facts: list[Fact]) -> None:
        from .canon import subject_key

        rows = []
        for f in facts:
            rows.append(
                (
                    f.id, f.evidence.doc_id, f.kind, f.metric, f.metric_key,
                    subject_key(f.subject), f.period.key if f.period else None,
                    f.quantity.canonical_unit if f.quantity else None,
                    f.quantity.canonical_value if f.quantity else None,
                    f.evidence.page, f.confidence, f.model_dump_json(),
                )
            )
        with self._lock:
            self._conn.executemany(
                """INSERT OR REPLACE INTO facts
                   (id,doc_id,kind,metric,metric_key,subject_key,period_key,
                    canonical_unit,canonical_value,page,confidence,payload)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                rows,
            )
            self._conn.commit()

    def all_facts(self, doc_id: str | None = None, limit: int | None = None) -> list[Fact]:
        sql = "SELECT payload FROM facts"
        args: list = []
        if doc_id:
            sql += " WHERE doc_id=?"
            args.append(doc_id)
        if limit:
            sql += f" LIMIT {int(limit)}"
        return [Fact.model_validate_json(r["payload"]) for r in self._q(sql, args)]

    def get_fact(self, fact_id: str) -> Fact | None:
        rows = self._q("SELECT payload FROM facts WHERE id=?", (fact_id,))
        return Fact.model_validate_json(rows[0]["payload"]) if rows else None

    def search_facts(
        self,
        q: str | None = None,
        doc_id: str | None = None,
        metric_key: str | None = None,
        period_key: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Fact]:
        sql = "SELECT payload FROM facts WHERE 1=1"
        args: list = []
        if q:
            sql += " AND (metric LIKE ? OR payload LIKE ?)"
            args += [f"%{q}%", f"%{q}%"]
        if doc_id:
            sql += " AND doc_id=?"
            args.append(doc_id)
        if metric_key:
            sql += " AND metric_key=?"
            args.append(metric_key)
        if period_key:
            sql += " AND period_key=?"
            args.append(period_key)
        sql += " ORDER BY confidence DESC LIMIT ? OFFSET ?"
        args += [limit, offset]
        return [Fact.model_validate_json(r["payload"]) for r in self._q(sql, args)]

    def fact_count(self) -> int:
        return self._q("SELECT COUNT(*) c FROM facts")[0]["c"]

    # -- relations --------------------------------------------------------
    def save_relations(self, relations: list[Relation]) -> None:
        rows = [
            (
                r.id, r.left_id, r.right_id, r.verdict.value, r.confidence,
                salience(r), int(r.cross_document), r.explained_by, r.model_dump_json(),
            )
            for r in relations
        ]
        with self._lock:
            self._conn.executemany(
                """INSERT OR REPLACE INTO relations
                   (id,left_id,right_id,verdict,confidence,salience,cross_document,
                    explained_by,payload)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                rows,
            )
            self._conn.commit()

    def list_relations(
        self,
        verdict: str | None = None,
        cross_document: bool | None = None,
        fact_id: str | None = None,
        min_confidence: float = 0.0,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict]:
        sql = "SELECT payload, salience FROM relations WHERE confidence>=?"
        args: list = [min_confidence]
        if verdict:
            sql += " AND verdict=?"
            args.append(verdict)
        if cross_document is not None:
            sql += " AND cross_document=?"
            args.append(int(cross_document))
        if fact_id:
            sql += " AND (left_id=? OR right_id=?)"
            args += [fact_id, fact_id]
        sql += " ORDER BY salience DESC LIMIT ? OFFSET ?"
        args += [limit, offset]
        out = []
        for r in self._q(sql, args):
            d = json.loads(r["payload"])
            d["salience"] = r["salience"]
            out.append(d)
        return out

    def relation_counts(self) -> dict:
        rows = self._q(
            "SELECT verdict, cross_document, COUNT(*) c FROM relations "
            "GROUP BY verdict, cross_document"
        )
        out: dict[str, Any] = {"total": 0, "by_verdict": {}, "cross_document": 0}
        for r in rows:
            out["total"] += r["c"]
            out["by_verdict"][r["verdict"]] = out["by_verdict"].get(r["verdict"], 0) + r["c"]
            if r["cross_document"]:
                out["cross_document"] += r["c"]
        return out

    # -- meta -------------------------------------------------------------
    def set_meta(self, key: str, value: Any) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO meta (key,value) VALUES (?,?)",
                (key, json.dumps(value)),
            )
            self._conn.commit()

    def get_meta(self, key: str, default: Any = None) -> Any:
        rows = self._q("SELECT value FROM meta WHERE key=?", (key,))
        return json.loads(rows[0]["value"]) if rows else default

    # -- incremental linking ---------------------------------------------
    def link_new_facts(
        self, new_facts: list[Fact], engine: ComparisonEngine | None = None
    ) -> tuple[list[Relation], dict]:
        """Relate a batch of new facts to each other and to everything stored.

        This is what makes ingestion incremental.  The engine still needs
        corpus-wide dimension statistics to judge reconciliations, so those are
        carried in `meta` and updated as pairs accumulate, rather than being
        recomputed from scratch on every upload.
        """
        engine = engine or ComparisonEngine()
        existing = self.all_facts()
        new_ids = {f.id for f in new_facts}
        prior = [f for f in existing if f.id not in new_ids]

        # Seed the engine's dimension statistics from what previous runs learned.
        from .compare import DimStat

        saved = self.get_meta("dim_stats", {}) or {}
        for dim, rec in saved.items():
            engine.stats.dim_stats[dim] = DimStat(
                dim=dim, n_pairs=rec.get("pairs", 0), n_value_differs=rec.get("value_differs", 0)
            )

        universe = prior + new_facts
        engine.resolver.index_metrics([f.metric for f in universe])

        # Only pairs that involve at least one new fact.
        pairs = [
            (a, b)
            for a, b in engine.candidate_pairs(universe)
            if a.id in new_ids or b.id in new_ids
        ]

        evaluated = []
        for a, b in pairs:
            ev = engine.evaluate(a, b)
            if ev is not None:
                evaluated.append(ev)

        for ev in evaluated:
            if len(ev["conflicts"]) != 1:
                continue
            dim = ev["conflicts"][0]
            st = engine.stats.dim_stats.setdefault(dim, DimStat(dim=dim))
            st.n_pairs += 1
            if not ev["values"]["agree"]:
                st.n_value_differs += 1

        relations = [engine._judge(ev) for ev in evaluated]

        self.set_meta(
            "dim_stats",
            {
                d: {"pairs": s.n_pairs, "value_differs": s.n_value_differs}
                for d, s in engine.stats.dim_stats.items()
            },
        )
        engine.resolver.save()
        return relations, engine.stats.as_dict()

    # -- summary ----------------------------------------------------------
    def summary(self) -> dict:
        docs = self.list_documents()
        return {
            "documents": len(docs),
            "pages": sum(d["n_pages"] for d in docs),
            "facts": self.fact_count(),
            "relations": self.relation_counts(),
            "dimensions": self.get_meta("dim_stats", {}),
            "corpora": sorted({d["corpus"] for d in docs if d["corpus"]}),
        }
