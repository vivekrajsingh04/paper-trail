"""Core data model for the fact knowledge layer.

Design notes
------------
Three ideas drive this schema:

1. **Facts carry their own uncertainty.**  A figure written as "8,142" is not
   the real number 8142.0 -- it is a claim that the true value lies somewhere in
   [8141.5, 8142.5].  We keep that interval on every quantity, because almost
   every comparison downstream reduces to "do these two intervals overlap?".

2. **Qualifiers are an open dict, not fixed columns.**  What distinguishes two
   otherwise-identical facts is document-specific: entity scope in a company
   filing, estimate vintage in a macro report, geography in a segment table.
   We refuse to enumerate those in advance -- the extractor proposes qualifier
   keys, and the comparison engine treats any key it has never seen before as a
   first-class dimension.  This is what lets the schema grow with the corpus.

3. **Evidence is a span, not a sentence.**  Every fact points at exact character
   offsets in a specific page of a specific document, plus the on-page bounding
   boxes needed to draw a box around it.  A fact that cannot be traced back to
   ink on a page is not a fact.
"""

from __future__ import annotations

import hashlib
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------
# Evidence
# --------------------------------------------------------------------------

class BBox(BaseModel):
    """A rectangle on a page, in PDF points, origin top-left."""

    x0: float
    y0: float
    x1: float
    y1: float


class Evidence(BaseModel):
    """Where a fact physically lives in a source document."""

    doc_id: str
    doc_title: str
    page: int  # 0-indexed physical page in the PDF
    page_label: str | None = None  # the page number printed on the page, if any
    char_start: int
    char_end: int
    snippet: str  # the surrounding text, for display
    quote: str  # the exact substring the value was read from
    bboxes: list[BBox] = Field(default_factory=list)

    @property
    def locator(self) -> str:
        return f"{self.doc_id}#p{self.page}:{self.char_start}-{self.char_end}"


# --------------------------------------------------------------------------
# Quantities and their precision
# --------------------------------------------------------------------------

class Quantity(BaseModel):
    """A number as written, plus what that writing implies about precision.

    `lo`/`hi` bound the true value given how the figure was rounded, expressed
    in canonical units.  "8,142 Cr" implies the author rounded to the nearest
    crore, so the true value is in [8141.5, 8142.5] Cr = [81415, 81425] million.
    """

    raw: str  # exactly as it appeared in the document
    value: float  # parsed, in `unit`
    unit: str | None = None  # canonical unit id, e.g. "INR_crore", "percent"
    unit_raw: str | None = None  # the unit as written, e.g. "Cr", "per cent"

    canonical_value: float | None = None  # value converted to `canonical_unit`
    canonical_unit: str | None = None  # e.g. "INR_million"

    lo: float | None = None  # lower bound of true value, canonical units
    hi: float | None = None  # upper bound of true value, canonical units
    ulp: float | None = None  # implied rounding step, in `unit`
    sign_convention: str | None = None  # e.g. "parenthesised_negative"

    @property
    def has_interval(self) -> bool:
        return self.lo is not None and self.hi is not None


# --------------------------------------------------------------------------
# Periods
# --------------------------------------------------------------------------

class Period(BaseModel):
    """A normalised time window.

    Documents write the same window many ways: "FY24", "FY2023-24",
    "fiscal year ended March 31, 2024", "2023-24".  We resolve all of them to a
    half-open date interval so they compare cleanly.
    """

    label: str  # as written
    start: str | None = None  # ISO date, inclusive
    end: str | None = None  # ISO date, inclusive
    granularity: Literal["year", "half", "quarter", "month", "point", "range"] | None = None
    calendar: str | None = None  # e.g. "IN_FY" (Apr-Mar), "CY"

    @property
    def key(self) -> str:
        if self.start and self.end:
            return f"{self.start}..{self.end}"
        return self.label.strip().lower()


# --------------------------------------------------------------------------
# Facts
# --------------------------------------------------------------------------

class Fact(BaseModel):
    """One atomic, evidence-backed claim lifted out of a document."""

    id: str = ""
    kind: Literal["numeric", "semantic"] = "numeric"

    # What is being measured, as the document words it, plus a canonical key
    # used for blocking during comparison.
    metric: str
    metric_key: str = ""

    subject: str | None = None  # who/what the metric is about, e.g. "Delhivery Limited"
    subject_key: str = ""

    quantity: Quantity | None = None  # None for purely semantic facts
    state: str | None = None  # for semantic facts, e.g. "resigned", "active"

    period: Period | None = None

    # Open dimension set. Keys are proposed by the extractor and are NOT fixed.
    # Examples seen in practice: scope=consolidated|standalone, basis=provisional,
    # geography=India, segment=Express Parcel, adjustment=adjusted.
    qualifiers: dict[str, str] = Field(default_factory=dict)

    evidence: Evidence
    extractor: str = ""  # which pass produced this, for auditability
    confidence: float = 0.5
    notes: str | None = None

    # Populated post-hoc when the value could not be verified against the page.
    verification: dict[str, Any] = Field(default_factory=dict)

    def compute_id(self) -> str:
        h = hashlib.sha1(
            "|".join(
                [
                    self.evidence.locator,
                    self.metric_key or self.metric,
                    str(self.quantity.value if self.quantity else self.state),
                    self.period.key if self.period else "",
                    repr(sorted(self.qualifiers.items())),
                ]
            ).encode()
        ).hexdigest()[:16]
        return h

    def model_post_init(self, __context: Any) -> None:
        if not self.id:
            self.id = self.compute_id()


# --------------------------------------------------------------------------
# Relations
# --------------------------------------------------------------------------

class Verdict(str, Enum):
    CORROBORATES = "corroborates"
    CONTRADICTS = "contradicts"
    RECONCILED = "reconciled"  # apparent contradiction explained by a qualifier
    RELATED = "related"  # values coincide across a dimension that differs
    NEEDS_REVIEW = "needs_review"  # not enough context to judge; see `review_reason`


class Relation(BaseModel):
    """A computed judgement about two facts.

    Every field here is derived deterministically from the two facts.  The
    `explanation` is rendered from `reasoning`, never invented -- if a reviewer
    disagrees with a verdict they can point at the exact step that produced it.
    """

    id: str = ""
    left_id: str
    right_id: str
    verdict: Verdict

    # The audit trail.  This is the "show your working" payload.
    reasoning: dict[str, Any] = Field(default_factory=dict)
    explanation: str = ""

    # How differently the two facts state the same thing (0 = identical wording).
    surface_distance: float | None = None
    # Which open qualifier keys disagree between the two facts.
    differing_dims: list[str] = Field(default_factory=list)
    # For RECONCILED: the dimension credited with explaining the gap.
    explained_by: str | None = None
    # For NEEDS_REVIEW: why the comparison was refused rather than decided.
    review_reason: str | None = None

    confidence: float = 0.5
    cross_document: bool = False

    def model_post_init(self, __context: Any) -> None:
        if not self.id:
            pair = "|".join(sorted([self.left_id, self.right_id]))
            self.id = hashlib.sha1(pair.encode()).hexdigest()[:16]


# --------------------------------------------------------------------------
# Documents
# --------------------------------------------------------------------------

class PageText(BaseModel):
    page: int
    page_label: str | None = None
    text: str
    # word -> bbox map, used to locate a character span on the page
    word_spans: list[tuple[int, int, float, float, float, float]] = Field(default_factory=list)


class Document(BaseModel):
    doc_id: str
    title: str
    filename: str
    n_pages: int
    sha256: str
    corpus: str | None = None
    published: str | None = None  # inferred publication date, ISO
    publisher: str | None = None
    ingested_at: str | None = None
