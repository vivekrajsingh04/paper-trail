"""PDF -> page text, with a character-offset to bounding-box map.

Why not just `page.get_text()`
------------------------------
Grounding a fact means being able to point at it on the page, not merely name
the page it was on.  So instead of taking the flat text dump, we read the
word list (each word carries its rectangle), reconstruct the page text from
those words ourselves, and record the character range each word occupies in the
text we built.

That gives an exact, invertible mapping: any character span the extractor cites
can be turned back into a set of rectangles to draw on the rendered page.  The
text the language model sees and the text we can highlight are the same string,
so a citation can never drift out of alignment with the ink.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pymupdf

from .models import BBox, Document, PageText


@dataclass
class WordBox:
    start: int
    end: int
    x0: float
    y0: float
    x1: float
    y1: float


@dataclass
class IngestedPage:
    page: int
    text: str
    words: list[WordBox] = field(default_factory=list)
    page_label: str | None = None

    def bboxes_for_span(self, char_start: int, char_end: int, merge_lines: bool = True) -> list[BBox]:
        """Rectangles covering a character range, merged per text line."""
        hits = [w for w in self.words if w.start < char_end and w.end > char_start]
        if not hits:
            return []
        if not merge_lines:
            return [BBox(x0=w.x0, y0=w.y0, x1=w.x1, y1=w.y1) for w in hits]

        # Group words whose vertical extents substantially overlap: one box per line.
        lines: list[list[WordBox]] = []
        for w in sorted(hits, key=lambda w: (round(w.y0, 1), w.x0)):
            placed = False
            for line in lines:
                ref = line[0]
                if min(ref.y1, w.y1) - max(ref.y0, w.y0) > 0.5 * min(ref.y1 - ref.y0, w.y1 - w.y0):
                    line.append(w)
                    placed = True
                    break
            if not placed:
                lines.append([w])

        out = []
        for line in lines:
            out.append(
                BBox(
                    x0=min(w.x0 for w in line),
                    y0=min(w.y0 for w in line),
                    x1=max(w.x1 for w in line),
                    y1=max(w.y1 for w in line),
                )
            )
        return out


# Page numbers printed on the page: a short numeric or roman token sitting alone
# in the top or bottom margin.
_LABEL_RE = re.compile(r"^\s*(?:page\s*)?([ivxlcdm]{1,7}|\d{1,4})\s*$", re.I)


def _guess_page_label(page: pymupdf.Page, text: str) -> str | None:
    height = page.rect.height
    try:
        blocks = page.get_text("blocks")
    except Exception:
        return None
    candidates: list[tuple[float, str]] = []
    for b in blocks:
        if len(b) < 5:
            continue
        y0, y1, content = b[1], b[3], str(b[4])
        stripped = content.strip()
        if not stripped or len(stripped) > 12:
            continue
        m = _LABEL_RE.match(stripped)
        if not m:
            continue
        # Prefer tokens near the very top or very bottom of the page.
        margin = min(y0, height - y1)
        if margin < 0.12 * height:
            candidates.append((margin, m.group(1)))
    if not candidates:
        return None
    candidates.sort()
    return candidates[0][1]


def ingest_pdf(path: str | Path, doc_id: str | None = None, corpus: str | None = None
               ) -> tuple[Document, list[IngestedPage]]:
    """Read a PDF into page text plus per-word geometry."""
    path = Path(path)
    raw = path.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    doc_id = doc_id or f"{path.stem[:48]}-{sha[:8]}"

    pdf = pymupdf.open(stream=raw, filetype="pdf")
    pages: list[IngestedPage] = []

    for pno in range(pdf.page_count):
        page = pdf[pno]
        # (x0, y0, x1, y1, word, block_no, line_no, word_no)
        words = page.get_text("words")
        words.sort(key=lambda w: (w[5], w[6], w[7]))

        buf: list[str] = []
        boxes: list[WordBox] = []
        cursor = 0
        prev_key: tuple[int, int] | None = None

        for x0, y0, x1, y1, token, bno, lno, _wno in words:
            key = (bno, lno)
            if prev_key is None:
                sep = ""
            elif key != prev_key:
                sep = "\n"
            else:
                sep = " "
            if sep:
                buf.append(sep)
                cursor += len(sep)
            start = cursor
            buf.append(token)
            cursor += len(token)
            boxes.append(WordBox(start=start, end=cursor, x0=x0, y0=y0, x1=x1, y1=y1))
            prev_key = key

        text = "".join(buf)
        pages.append(
            IngestedPage(
                page=pno,
                text=text,
                words=boxes,
                page_label=_guess_page_label(page, text),
            )
        )

    meta = pdf.metadata or {}
    title = (meta.get("title") or "").strip() or path.stem.replace("-", " ").replace("_", " ")

    document = Document(
        doc_id=doc_id,
        title=title,
        filename=path.name,
        n_pages=pdf.page_count,
        sha256=sha,
        corpus=corpus,
        ingested_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
    pdf.close()
    return document, pages


def render_page_png(path: str | Path, page_no: int, dpi: int = 130) -> tuple[bytes, float, float]:
    """Render one page to PNG, returning the image and its point dimensions.

    The UI needs the point dimensions to scale stored bounding boxes onto the
    rasterised image.
    """
    pdf = pymupdf.open(str(path))
    page = pdf[page_no]
    rect = page.rect
    pix = page.get_pixmap(dpi=dpi)
    data = pix.tobytes("png")
    pdf.close()
    return data, rect.width, rect.height


def to_page_models(pages: list[IngestedPage]) -> list[PageText]:
    return [
        PageText(
            page=p.page,
            page_label=p.page_label,
            text=p.text,
            word_spans=[(w.start, w.end, w.x0, w.y0, w.x1, w.y1) for w in p.words],
        )
        for p in pages
    ]
