"""Prompts for document profiling and fact extraction.

A deliberate constraint runs through both prompts: the model is asked to
*label text that exists*, never to produce values.  It must quote the document
verbatim, and every quote is checked against the page before the fact is
accepted.  A model that invents a number cannot get it past the verifier,
because the number will not be on the page.

The second constraint is that no metric vocabulary is supplied.  The model is
told to use the document's own wording and to invent qualifier keys as the
material demands.  Nothing here names a company, a report, or a metric, so the
same prompts work on documents the system has never seen.
"""

DOC_PROFILE_PROMPT = """\
You are cataloguing a document so that facts extracted from it can later be
compared against facts from other documents.

Read the excerpt below (the opening pages of a PDF) and return JSON:

{{
  "title": "the document's own title",
  "publisher": "the organisation that published it",
  "doc_type": "e.g. annual report, prospectus, earnings presentation, staff report, survey",
  "primary_entity": "the main organisation or economy the document is about",
  "published": "YYYY-MM-DD if stated or clearly inferable, else null",
  "as_of": "the latest period the document reports on, in the document's own words, else null",
  "default_qualifiers": {{}},
  "notes": "anything that affects how its figures should be read"
}}

`default_qualifiers` carries context that applies to figures throughout the
document unless a figure overrides it. Use short lowercase keys and values.
Include a key only when the document genuinely establishes it. Judgement, not a
checklist: a consolidated financial statement establishes a reporting scope; a
forecast published at a known vintage establishes when its estimates were cut.

FILENAME: {filename}

EXCERPT:
---
{excerpt}
---

Return only the JSON object.
"""


EXTRACT_PROMPT = """\
You extract structured, evidence-backed facts from one page of a document.
Downstream, these facts are compared across documents to find agreement and
conflict, so what matters most is that each fact is precisely *scoped*: two
figures that look contradictory are usually measuring subtly different things.

DOCUMENT CONTEXT
  title:      {title}
  publisher:  {publisher}
  type:       {doc_type}
  about:      {primary_entity}
  published:  {published}
  applies to all figures unless overridden: {default_qualifiers}

PAGE {page_no}{page_label} TEXT
---
{page_text}
---

Return JSON: {{"facts": [ ... ]}}   Each fact:

{{
  "quote":      "the exact substring of the page containing the figure, copied character for character",
  "kind":       "numeric" | "semantic",
  "metric":     "what is measured, in the document's own words",
  "subject":    "the entity/thing it is about, or null",
  "value_raw":  "the number exactly as printed, e.g. \\"8,142\\" or \\"(452)\\"  (numeric only)",
  "unit_raw":   "the unit as printed, e.g. \\"Rs. crore\\", \\"%\\", \\"tonnes\\", or null",
  "state":      "for semantic facts, the status asserted, e.g. \\"resigned\\"  (semantic only)",
  "period_raw": "the period as printed, e.g. \\"FY24\\", \\"Q4 FY24\\", \\"2024-25\\", or null",
  "qualifiers": {{}},
  "confidence": 0.0-1.0
}}

RULES

1. `quote` must appear verbatim on the page above. Do not normalise spacing,
   expand abbreviations, or tidy punctuation. Facts whose quote cannot be found
   are discarded automatically, so copy carefully. Keep it short -- the figure
   and just enough words to identify it.

2. Never state a number that is not printed on the page. Do not convert units,
   compute totals, sum columns, or infer a figure from a chart. If a value is
   not written, there is no fact.

3. `qualifiers` is an open dictionary and it is the most important field.
   Record every dimension that would change what the figure means. You are not
   given a fixed list of keys: invent whatever the page requires, using short
   lowercase snake_case keys and values. Things that commonly matter include
   which entity boundary a figure covers, whether it is an estimate/projection
   or a settled actual and of what vintage, which segment or geography it is
   limited to, whether it has been adjusted or restated, and what it is measured
   relative to. Only include a key when the page or the document context
   actually establishes it. Do not guess.

4. Tables: read the row label AND every column header that applies. A single
   number in a financial table is usually scoped by at least two headers (for
   example a period and an entity boundary). Getting the column wrong silently
   produces a false fact, so if the layout is genuinely ambiguous, lower
   `confidence` and say why in the metric wording rather than guessing.

5. Prefer figures a reader would cite: headline metrics, table rows, stated
   totals and rates. Skip page furniture, footnote markers, cross-references and
   figures whose meaning depends on a chart you cannot read.

6. Semantic facts: extract status and relationship claims that could later
   change or be contradicted -- appointments, resignations, classifications,
   locations, listings. Set kind="semantic", fill `state`, omit value/unit.

7. Emit at most {max_facts} facts. If the page has more, keep the ones most
   likely to be cited or compared. An empty list is a valid answer.

Return only the JSON object.
"""


METRIC_MATCH_PROMPT = """\
Two facts were extracted from documents and may or may not describe the same
measurement. Decide whether they are directly comparable.

A: metric={a_metric!r}  subject={a_subject!r}  unit={a_unit!r}
B: metric={b_metric!r}  subject={b_subject!r}  unit={b_unit!r}

Example A quote: {a_quote!r}
Example B quote: {b_quote!r}

Return JSON:
{{
  "same_metric": true | false,
  "relationship": "identical" | "broader_narrower" | "component_of" | "different",
  "which_broader": "A" | "B" | null,
  "reason": "one sentence"
}}

Be strict. Two metrics are `identical` only if a difference in their values
would be a genuine inconsistency rather than an expected consequence of them
measuring different things. Revenue for a group and revenue for one of its
segments are not identical; they are broader_narrower. A total and one of its
components are component_of. If in doubt, answer false.
"""


BATCH_EXTRACT_PROMPT = """\
You extract structured, evidence-backed facts from several consecutive pages of
one document. Downstream, these facts are compared across documents to find
agreement and conflict, so what matters most is that each fact is precisely
*scoped*: two figures that look contradictory are usually measuring subtly
different things.

DOCUMENT CONTEXT
  title:      {title}
  publisher:  {publisher}
  type:       {doc_type}
  about:      {primary_entity}
  published:  {published}
  applies to all figures unless overridden: {default_qualifiers}

PAGES
Each page is delimited below. The number in the delimiter is the page id you
must report for facts drawn from that page.
---
{pages_block}
---

Return JSON: {{"facts": [ ... ]}}   Each fact:

{{
  "page":       <integer page id from the delimiter the quote came from>,
  "quote":      "the exact substring of that page containing the figure, copied character for character",
  "kind":       "numeric" | "semantic",
  "metric":     "what is measured, in the document's own words",
  "subject":    "the entity/thing it is about, or null",
  "value_raw":  "the number exactly as printed, e.g. \\"8,142\\" or \\"(452)\\"  (numeric only)",
  "unit_raw":   "the unit as printed, e.g. \\"Rs. crore\\", \\"%\\", \\"tonnes\\", or null",
  "state":      "for semantic facts, the status asserted, e.g. \\"resigned\\"  (semantic only)",
  "period_raw": "the period as printed, e.g. \\"FY24\\", \\"Q4 FY24\\", \\"2024-25\\", or null",
  "qualifiers": {{}},
  "confidence": 0.0-1.0
}}

RULES

1. `quote` must appear verbatim on the page you name in `page`. Do not normalise
   spacing, expand abbreviations, or tidy punctuation. Facts whose quote cannot
   be found are discarded automatically, so copy carefully. Keep it short -- the
   figure and just enough words to identify it.

2. Never state a number that is not printed on the page. Do not convert units,
   compute totals, sum columns, or infer a figure from a chart. If a value is
   not written, there is no fact.

3. `qualifiers` is an open dictionary and it is the most important field.
   Record every dimension that would change what the figure means. You are not
   given a fixed list of keys: invent whatever the pages require, using short
   lowercase snake_case keys and values. Things that commonly matter include
   which entity boundary a figure covers, whether it is an estimate/projection
   or a settled actual and of what vintage, which segment or geography it is
   limited to, whether it has been adjusted or restated, and what it is measured
   relative to. Only include a key when the page or the document context
   actually establishes it. Do not guess.

4. Tables: read the row label AND every column header that applies. A single
   number in a financial table is usually scoped by at least two headers (for
   example a period and an entity boundary). Getting the column wrong silently
   produces a false fact, so if the layout is genuinely ambiguous, lower
   `confidence` and say why in the metric wording rather than guessing.

5. A table's column headers often sit on an earlier page than its rows. Use the
   surrounding pages for context, but always attribute a fact to the page its
   quote is actually on.

6. Prefer figures a reader would cite: headline metrics, table rows, stated
   totals and rates. Skip page furniture, footnote markers, cross-references and
   figures whose meaning depends on a chart you cannot read.

7. Semantic facts: extract status and relationship claims that could later
   change or be contradicted -- appointments, resignations, classifications,
   locations, listings. Set kind="semantic", fill `state`, omit value/unit.

8. Emit at most {max_facts} facts per page. An empty list is a valid answer for
   a page with nothing citable on it.

Return only the JSON object.
"""
