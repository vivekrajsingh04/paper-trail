"""Turn pages into verified, evidence-backed facts.

The pipeline is deliberately shaped so that the language model has the narrowest
possible licence:

    page text ──► model proposes {quote, metric, scope, ...}
                        │
                        ▼
              verifier: does `quote` occur verbatim on this page?
                        │  no ──► discarded, counted, reported
                        ▼ yes
              verifier: is `value_raw` inside that quote?
                        │  no ──► discarded, counted, reported
                        ▼ yes
              deterministic: parse number, normalise unit, build interval,
                             resolve period, map span to bounding boxes
                        │
                        ▼
                     Fact

The model contributes *interpretation* -- what a figure measures and how it is
scoped.  Every number, every unit conversion, every date and every coordinate is
computed from the page.  This is why a hallucinated figure cannot survive: it
has no quote to match.
"""

from __future__ import annotations

import concurrent.futures as cf
import json
import os
import re
import threading
from dataclasses import dataclass, field

from . import candidates as cand
from .ingest import IngestedPage
from .llm import LLMUnavailable, complete_json
from .models import Document, Evidence, Fact, Quantity
from .periods import parse_period
from .prompts import BATCH_EXTRACT_PROMPT, DOC_PROFILE_PROMPT, EXTRACT_PROMPT
from .units import build_interval, normalise_unit, parse_number

MAX_FACTS_PER_PAGE = 25
MAX_FACTS_PER_PAGE_BATCHED = 14
SNIPPET_PAD = 220

# Pages are sent several at a time.  This cuts call count roughly five-fold,
# which keeps a 500-page corpus inside free-tier request quotas and makes large
# PDFs practical; it also gives the model the neighbouring pages it needs when a
# table's column headers sit on the page before its rows.
BATCH_PAGES = int(os.environ.get("FACTLAYER_BATCH_PAGES", "8"))
BATCH_CHARS = int(os.environ.get("FACTLAYER_BATCH_CHARS", "45000"))


@dataclass
class DocProfile:
    title: str = ""
    publisher: str = ""
    doc_type: str = ""
    primary_entity: str = ""
    published: str | None = None
    as_of: str | None = None
    default_qualifiers: dict[str, str] = field(default_factory=dict)
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "title": self.title, "publisher": self.publisher, "doc_type": self.doc_type,
            "primary_entity": self.primary_entity, "published": self.published,
            "as_of": self.as_of, "default_qualifiers": self.default_qualifiers,
            "notes": self.notes,
        }


@dataclass
class ExtractionStats:
    pages_seen: int = 0
    pages_sent: int = 0
    pages_failed: int = 0
    facts_proposed: int = 0
    facts_kept: int = 0
    rejected_quote_not_found: int = 0
    rejected_value_mismatch: int = 0
    rejected_unparseable: int = 0
    rejections: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        d = self.__dict__.copy()
        d["rejections"] = self.rejections[:60]
        d["verification_pass_rate"] = (
            self.facts_kept / self.facts_proposed if self.facts_proposed else 1.0
        )
        return d


# --------------------------------------------------------------------------
# Document profiling
# --------------------------------------------------------------------------

def profile_document(doc: Document, pages: list[IngestedPage], n_pages: int = 4) -> DocProfile:
    """Infer publisher, vintage and document-wide qualifiers from the front matter.

    Publication date matters more than it looks: two reports can state
    different values for the same quantity simply because one was cut earlier.
    Without a vintage the system cannot tell that apart from a real conflict.
    """
    excerpt_parts = []
    budget = 9000
    for p in pages[:n_pages]:
        chunk = p.text[:3000]
        if budget - len(chunk) < 0:
            break
        excerpt_parts.append(f"[page {p.page}]\n{chunk}")
        budget -= len(chunk)
    excerpt = "\n\n".join(excerpt_parts)

    prompt = DOC_PROFILE_PROMPT.format(filename=doc.filename, excerpt=excerpt)
    try:
        res = complete_json(prompt, tag=f"profile:{doc.doc_id}")
        d = res.data if isinstance(res.data, dict) else {}
    except Exception:  # noqa: BLE001
        # Profiling is best-effort context, not a hard dependency: extraction
        # still works without it, just with fewer document-wide qualifiers.
        return DocProfile(title=doc.title, primary_entity="", publisher="")

    dq = d.get("default_qualifiers") or {}
    if not isinstance(dq, dict):
        dq = {}
    return DocProfile(
        title=(d.get("title") or doc.title or "").strip(),
        publisher=(d.get("publisher") or "").strip(),
        doc_type=(d.get("doc_type") or "").strip(),
        primary_entity=(d.get("primary_entity") or "").strip(),
        published=d.get("published"),
        as_of=d.get("as_of"),
        default_qualifiers={str(k): str(v) for k, v in dq.items() if v is not None},
        notes=(d.get("notes") or "").strip(),
    )


# --------------------------------------------------------------------------
# Quote grounding
# --------------------------------------------------------------------------

def _find_quote(text: str, quote: str) -> tuple[int, int] | None:
    """Locate a model-supplied quote in the page, tolerating whitespace drift.

    Exact match first.  If that fails, retry with whitespace collapsed, since
    models routinely normalise the line breaks that PDF text is full of.
    Anything looser than that is refused: a quote we cannot pin down exactly is
    not evidence.
    """
    if not quote:
        return None
    i = text.find(quote)
    if i != -1:
        return i, i + len(quote)

    # Whitespace-insensitive search: build a regex from the quote's tokens.
    tokens = [re.escape(t) for t in quote.split()]
    if not tokens:
        return None
    pattern = r"\s+".join(tokens)
    m = re.search(pattern, text)
    if m:
        return m.start(), m.end()

    # Last resort: the quote with all whitespace removed, mapped back to offsets.
    stripped = re.sub(r"\s+", "", quote)
    if len(stripped) < 4:
        return None
    compact_idx: list[int] = []
    compact_chars: list[str] = []
    for idx, ch in enumerate(text):
        if not ch.isspace():
            compact_chars.append(ch)
            compact_idx.append(idx)
    j = "".join(compact_chars).find(stripped)
    if j == -1:
        return None
    return compact_idx[j], compact_idx[j + len(stripped) - 1] + 1


# --------------------------------------------------------------------------
# Page extraction
# --------------------------------------------------------------------------

def extract_page(
    doc: Document,
    profile: DocProfile,
    page: IngestedPage,
    stats: ExtractionStats,
    lock: threading.Lock,
) -> list[Fact]:
    label = f" (printed page {page.page_label})" if page.page_label else ""
    prompt = EXTRACT_PROMPT.format(
        title=profile.title or doc.title,
        publisher=profile.publisher or "unknown",
        doc_type=profile.doc_type or "unknown",
        primary_entity=profile.primary_entity or "unknown",
        published=profile.published or "unknown",
        default_qualifiers=json.dumps(profile.default_qualifiers, ensure_ascii=False),
        page_no=page.page,
        page_label=label,
        page_text=page.text[:14000],
        max_facts=MAX_FACTS_PER_PAGE,
    )

    try:
        res = complete_json(prompt, tag=f"extract:{doc.doc_id}:p{page.page}")
    except (LLMUnavailable, ValueError, Exception) as exc:  # noqa: BLE001
        with lock:
            stats.pages_failed += 1
            stats.rejections.append(
                {"page": page.page, "why": "call_failed", "detail": str(exc)[:200]}
            )
        return []

    payload = res.data
    raw_facts = payload.get("facts", []) if isinstance(payload, dict) else payload
    if not isinstance(raw_facts, list):
        raw_facts = []

    kept: list[Fact] = []
    local_rejects: list[dict] = []
    n_proposed = 0
    n_quote_fail = n_value_fail = n_unparseable = 0

    for rf in raw_facts:
        if not isinstance(rf, dict):
            continue
        n_proposed += 1

        quote = (rf.get("quote") or "").strip()
        span = _find_quote(page.text, quote)
        if span is None:
            n_quote_fail += 1
            local_rejects.append(
                {"page": page.page, "why": "quote_not_on_page", "quote": quote[:120],
                 "metric": str(rf.get("metric"))[:80]}
            )
            continue

        cs, ce = span
        kind = "semantic" if rf.get("kind") == "semantic" else "numeric"
        metric = (rf.get("metric") or "").strip()
        if not metric:
            n_unparseable += 1
            continue

        quantity = None
        if kind == "numeric":
            value_raw = str(rf.get("value_raw") or "").strip()
            if not value_raw:
                n_unparseable += 1
                local_rejects.append({"page": page.page, "why": "no_value", "metric": metric[:80]})
                continue

            # The stated value must itself appear inside the quoted evidence.
            digits = re.sub(r"[^\d]", "", value_raw)
            quoted_digits = re.sub(r"[^\d]", "", page.text[cs:ce])
            if digits and digits not in quoted_digits:
                n_value_fail += 1
                local_rejects.append(
                    {"page": page.page, "why": "value_not_in_quote", "value": value_raw[:40],
                     "quote": page.text[cs:ce][:120], "metric": metric[:80]}
                )
                continue

            parsed = parse_number(value_raw)
            if parsed is None:
                n_unparseable += 1
                local_rejects.append(
                    {"page": page.page, "why": "unparseable_number", "value": value_raw[:40]}
                )
                continue

            unit_raw = rf.get("unit_raw")
            unit = normalise_unit(unit_raw)
            canon, lo, hi = build_interval(parsed, unit)
            quantity = Quantity(
                raw=value_raw,
                value=-parsed.value if parsed.negative else parsed.value,
                unit=unit.canonical,
                unit_raw=unit_raw,
                canonical_value=canon,
                canonical_unit=unit.canonical,
                lo=lo,
                hi=hi,
                ulp=parsed.ulp,
                sign_convention=parsed.sign_convention,
            )

        qualifiers = rf.get("qualifiers") or {}
        if not isinstance(qualifiers, dict):
            qualifiers = {}
        merged = {str(k): str(v) for k, v in profile.default_qualifiers.items()}
        merged.update({str(k): str(v) for k, v in qualifiers.items() if v is not None})

        snip_start = max(0, cs - SNIPPET_PAD)
        snip_end = min(len(page.text), ce + SNIPPET_PAD)

        fact = Fact(
            kind=kind,
            metric=metric,
            subject=(rf.get("subject") or profile.primary_entity or None),
            quantity=quantity,
            state=(rf.get("state") or None) if kind == "semantic" else None,
            period=parse_period(rf.get("period_raw")),
            qualifiers=merged,
            evidence=Evidence(
                doc_id=doc.doc_id,
                doc_title=profile.title or doc.title,
                page=page.page,
                page_label=page.page_label,
                char_start=cs,
                char_end=ce,
                snippet=page.text[snip_start:snip_end],
                quote=page.text[cs:ce],
                bboxes=page.bboxes_for_span(cs, ce),
            ),
            extractor=f"llm:{res.model}",
            confidence=float(rf.get("confidence") or 0.6),
        )
        kept.append(fact)

    with lock:
        stats.pages_sent += 1
        stats.facts_proposed += n_proposed
        stats.facts_kept += len(kept)
        stats.rejected_quote_not_found += n_quote_fail
        stats.rejected_value_mismatch += n_value_fail
        stats.rejected_unparseable += n_unparseable
        stats.rejections.extend(local_rejects)

    return kept


def _build_batches(pages: list[IngestedPage]) -> list[list[IngestedPage]]:
    """Group consecutive pages into calls, bounded by count and character budget."""
    batches: list[list[IngestedPage]] = []
    current: list[IngestedPage] = []
    size = 0
    for p in pages:
        n = len(p.text)
        if current and (len(current) >= BATCH_PAGES or size + n > BATCH_CHARS):
            batches.append(current)
            current, size = [], 0
        current.append(p)
        size += n
    if current:
        batches.append(current)
    return batches


def extract_batch(
    doc: Document,
    profile: DocProfile,
    pages: list[IngestedPage],
    stats: ExtractionStats,
    lock: threading.Lock,
) -> list[Fact]:
    """Extract facts from several consecutive pages in one call."""
    by_id = {p.page: p for p in pages}
    blocks = []
    for p in pages:
        label = f" (printed page {p.page_label})" if p.page_label else ""
        blocks.append(f"=== PAGE {p.page}{label} ===\n{p.text[:BATCH_CHARS]}")
    prompt = BATCH_EXTRACT_PROMPT.format(
        title=profile.title or doc.title,
        publisher=profile.publisher or "unknown",
        doc_type=profile.doc_type or "unknown",
        primary_entity=profile.primary_entity or "unknown",
        published=profile.published or "unknown",
        default_qualifiers=json.dumps(profile.default_qualifiers, ensure_ascii=False),
        pages_block="\n\n".join(blocks),
        max_facts=MAX_FACTS_PER_PAGE_BATCHED,
    )

    try:
        res = complete_json(prompt, tag=f"extract:{doc.doc_id}:b{pages[0].page}")
    except Exception as exc:  # noqa: BLE001
        with lock:
            stats.pages_failed += len(pages)
            stats.rejections.append(
                {"page": pages[0].page, "why": "call_failed",
                 "detail": f"batch {pages[0].page}-{pages[-1].page}: {str(exc)[:180]}"}
            )
        return []

    payload = res.data
    raw_facts = payload.get("facts", []) if isinstance(payload, dict) else payload
    if not isinstance(raw_facts, list):
        raw_facts = []

    kept: list[Fact] = []
    rejects: list[dict] = []
    n_proposed = n_quote_fail = n_value_fail = n_unparseable = 0

    for rf in raw_facts:
        if not isinstance(rf, dict):
            continue
        n_proposed += 1
        quote = (rf.get("quote") or "").strip()

        # Locate the quote on the page the model named; if it mis-attributed,
        # fall back to the other pages in this batch rather than dropping a
        # good fact, and record that the page had to be corrected.
        page = by_id.get(rf.get("page") if isinstance(rf.get("page"), int) else -1)
        span = _find_quote(page.text, quote) if page else None
        relocated = False
        if span is None:
            for cand_page in pages:
                if cand_page is page:
                    continue
                alt = _find_quote(cand_page.text, quote)
                if alt is not None:
                    page, span, relocated = cand_page, alt, True
                    break

        if page is None or span is None:
            n_quote_fail += 1
            rejects.append({"page": rf.get("page"), "why": "quote_not_on_page",
                            "quote": quote[:120], "metric": str(rf.get("metric"))[:80]})
            continue

        fact = _assemble_fact(doc, profile, page, rf, span, res.model)
        if fact is None:
            n_unparseable += 1
            rejects.append({"page": page.page, "why": "unparseable",
                            "metric": str(rf.get("metric"))[:80],
                            "value": str(rf.get("value_raw"))[:40]})
            continue
        if fact == "value_mismatch":
            n_value_fail += 1
            rejects.append({"page": page.page, "why": "value_not_in_quote",
                            "value": str(rf.get("value_raw"))[:40],
                            "quote": page.text[span[0]:span[1]][:120]})
            continue
        if relocated:
            fact.verification["page_relocated_from"] = rf.get("page")
            fact.confidence *= 0.9
        kept.append(fact)

    with lock:
        stats.pages_sent += len(pages)
        stats.facts_proposed += n_proposed
        stats.facts_kept += len(kept)
        stats.rejected_quote_not_found += n_quote_fail
        stats.rejected_value_mismatch += n_value_fail
        stats.rejected_unparseable += n_unparseable
        stats.rejections.extend(rejects)
    return kept


def _assemble_fact(doc, profile, page, rf, span, model):
    """Build a verified Fact, or a marker explaining why it could not be built."""
    cs, ce = span
    kind = "semantic" if rf.get("kind") == "semantic" else "numeric"
    metric = (rf.get("metric") or "").strip()
    if not metric:
        return None

    quantity = None
    if kind == "numeric":
        value_raw = str(rf.get("value_raw") or "").strip()
        if not value_raw:
            return None
        digits = re.sub(r"[^\d]", "", value_raw)
        quoted_digits = re.sub(r"[^\d]", "", page.text[cs:ce])
        if digits and digits not in quoted_digits:
            return "value_mismatch"
        parsed = parse_number(value_raw)
        if parsed is None:
            return None
        unit_raw = rf.get("unit_raw")
        unit = normalise_unit(unit_raw)
        canon, lo, hi = build_interval(parsed, unit)
        quantity = Quantity(
            raw=value_raw, value=-parsed.value if parsed.negative else parsed.value,
            unit=unit.canonical, unit_raw=unit_raw, canonical_value=canon,
            canonical_unit=unit.canonical, lo=lo, hi=hi, ulp=parsed.ulp,
            sign_convention=parsed.sign_convention,
        )

    qualifiers = rf.get("qualifiers") or {}
    if not isinstance(qualifiers, dict):
        qualifiers = {}
    merged = {str(k): str(v) for k, v in profile.default_qualifiers.items()}
    merged.update({str(k): str(v) for k, v in qualifiers.items() if v is not None})

    return Fact(
        kind=kind,
        metric=metric,
        subject=(rf.get("subject") or profile.primary_entity or None),
        quantity=quantity,
        state=(rf.get("state") or None) if kind == "semantic" else None,
        period=parse_period(rf.get("period_raw")),
        qualifiers=merged,
        evidence=Evidence(
            doc_id=doc.doc_id, doc_title=profile.title or doc.title, page=page.page,
            page_label=page.page_label, char_start=cs, char_end=ce,
            snippet=page.text[max(0, cs - SNIPPET_PAD): min(len(page.text), ce + SNIPPET_PAD)],
            quote=page.text[cs:ce], bboxes=page.bboxes_for_span(cs, ce),
        ),
        extractor=f"llm:{model}",
        confidence=float(rf.get("confidence") or 0.6),
    )


def extract_document(
    doc: Document,
    pages: list[IngestedPage],
    profile: DocProfile | None = None,
    max_workers: int = 8,
    progress=None,
) -> tuple[list[Fact], DocProfile, ExtractionStats]:
    """Extract every interesting page of a document, in parallel."""
    profile = profile or profile_document(doc, pages)
    stats = ExtractionStats(pages_seen=len(pages))
    lock = threading.Lock()

    targets = [p for p in pages if cand.page_is_interesting(p.text)]
    batches = _build_batches(targets)
    facts: list[Fact] = []

    with cf.ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(extract_batch, doc, profile, b, stats, lock) for b in batches]
        done = 0
        for fut in cf.as_completed(futures):
            facts.extend(fut.result())
            done += 1
            if progress:
                progress(done, len(batches))

    # Deduplicate: the same figure quoted twice on a page is one fact.
    seen: dict[str, Fact] = {}
    for f in facts:
        if f.id not in seen or f.confidence > seen[f.id].confidence:
            seen[f.id] = f
    return list(seen.values()), profile, stats
