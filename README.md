# Fact Knowledge Layer

Extracts evidence-backed facts from PDFs, links every fact to the exact ink it
came from, and works out whether facts across documents corroborate each other,
contradict each other, or only appear to disagree because they are measuring
subtly different things.

Built for the Superjoin engineering intern assignment.

---

## The idea in one table

Every relationship verdict in this system reduces to a single question: **when
is a difference between two numbers a real difference?** The usual answer is a
percentage tolerance. That answer cannot work, and this corpus proves it:

| Pair (from the starter documents) | Relative gap | Correct verdict |
|---|---|---|
| `1.4 Mn tonnes` vs `1,429K tonnes` | **2.03 %** | agree — same figure |
| `6.4 %` vs `6.5 %` GDP growth | **1.54 %** | disagree — real difference |

Any fixed tolerance loose enough to accept the first also wrongly accepts the
second. There is no epsilon that separates them.

So the system doesn't use one. A figure as written is a *rounded* claim, and
how it was rounded is visible in its digits: `1.4` claims precision to ±0.05 Mn,
while `6.4` claims ±0.05 percentage points. Convert both sides to a common unit,
carry those intervals through, and ask whether they **intersect**:

```
1.4 Mn t   -> [1,350,000 , 1,450,000]  ─┐ overlap  -> agree
1,429K t   -> [1,428,500 , 1,429,500]  ─┘

6.4 %      -> [6.35 , 6.45]            ─┐ disjoint -> real difference
6.5 %      -> [6.45 , 6.55]            ─┘
```

Both verdicts fall out of the same rule, and the rule is arithmetic a reviewer
can check by hand. That is the whole design in miniature: **compute the
judgement, don't generate it.**

---

## Setup and run instructions

Requires Python 3.11+. No services, no Docker, no build step.

```bash
git clone <this repo> && cd factlayer

# 1. environment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. credentials  (optional -- see "Running without a key" below)
cp .env.example .env
# put a Gemini key in it: https://aistudio.google.com/apikey
# GEMINI_API_KEY=...

# 3. build the knowledge layer from the starter documents
python scripts/ingest.py starter-datasets/starter-datasets/delhivery
python scripts/ingest.py starter-datasets/starter-datasets/india-macroeconomy

# 4. explore it
uvicorn factlayer.api:app --reload
# open http://127.0.0.1:8000
```

Upload further PDFs through the **Upload** tab in the UI, or:

```bash
curl -F "file=@your.pdf" "http://127.0.0.1:8000/api/documents?corpus=mine"
```

### Running without a key

Every model call is cached to `data/cache/`, keyed by a hash of the prompt, and
**those cache files are committed**. A clean checkout with no API key
reproduces the entire knowledge layer offline:

```bash
FACTLAYER_LLM=replay python scripts/ingest.py starter-datasets/starter-datasets/delhivery
```

In `replay` mode a cache miss is an error rather than a network call, so this is
also how the test suite stays hermetic. New PDFs still need a key.

### Tests and evaluations

```bash
pytest                                  # unit + reasoning tests, no key needed
python evals/run_evals.py               # offline evaluation suites
python evals/run_evals.py --with-db     # plus invariants over the built layer
```

---

## Video demo

**[link to be added]** — under 3 minutes, showing a PDF being processed and all
four required cases.

---

## Approach

### Architecture

```
PDF
 │
 ├─ ingest.py     text + per-word geometry; a char-offset → bounding-box map
 │                so any cited span can be drawn on the page
 │
 ├─ candidates.py deterministic scan for quantity-shaped tokens
 │                → page pre-filter, verification target, coverage metric
 │
 ├─ extract.py    model proposes {quote, metric, scope, period, qualifiers}
 │                → VERIFIER: quote must exist verbatim; value must be inside it
 │                → numbers, units, dates, coordinates all computed, never generated
 │
 ├─ canon.py      which metric names denote the same measurement
 │                (exact → lexical → semantic → model adjudication, cached)
 │
 ├─ compare.py    interval overlap + dimension diff → verdict + explanation
 │
 └─ store.py      SQLite; new documents link incrementally against existing facts
```

### The model is never allowed to produce a number

This is the central constraint. The extractor's output is checked before it
becomes a fact:

1. Its `quote` must appear **verbatim** on the page it names. Whitespace drift
   is tolerated (PDF text is full of spurious line breaks); anything looser is
   refused, because a quote we cannot pin to exact offsets is not evidence.
2. The `value_raw` it reports must appear **inside that quote**.
3. Only then are the number, its unit conversion, its precision interval, its
   period and its on-page rectangles computed — deterministically, from the page.

A hallucinated figure has no quote to match and cannot survive step 1. What the
model actually contributes is *interpretation*: what a figure measures, and how
it is scoped. That is the part that genuinely needs judgement.

The pass rate is reported per document in the **Diagnostics** tab, and rejected
proposals are listed rather than dropped silently.

### Facts carry an open set of dimensions

Two figures are only comparable if they measure the same thing under the same
conditions. Which conditions matter is document-specific — entity scope in a
filing, estimate vintage in a macro report, segment or geography in a table — so
the schema does not enumerate them. The extractor proposes qualifier keys
freely, and the comparison engine treats any key it has never seen as a
first-class dimension.

A fact is therefore: a metric, a subject, a quantity with a precision interval,
a normalised period, an **open dictionary of qualifiers**, and a span of
evidence.

### Which dimensions can explain a difference is learned, not declared

Hard-coding "period and scope are explanatory" would be exactly the kind of
document-specific rule the brief rules out. Instead the engine measures, across
every comparable pair in the corpus, how often each dimension accompanies a
change in value when it is the *only* thing in conflict:

```
P(values differ | this dimension is the sole conflict)
```

A dimension that almost always moves the value earns a high weight; a
reconciliation resting on a weak one is reported at low confidence. New
qualifier keys are scored the same way the moment they appear — which is what
lets the schema grow with the corpus rather than being fixed in advance. The
learned weights are visible in the **Diagnostics** tab.

### Verdict rules

With value agreement and a dimension diff in hand, the verdict is mechanical:

| values | conflicting dimensions | verdict |
|---|---|---|
| intervals overlap | — | **corroborates** |
| intervals disjoint | none | **contradicts** — nothing explains the gap |
| intervals disjoint | exactly one | **reconciled** by that dimension |
| intervals disjoint | several | **reconciled** by the strongest, lower confidence |

Confidence composes the three things that can undermine a verdict: either fact
being poorly extracted, the metric match being a model guess rather than a
string identity, and a reconciliation resting on a dimension that rarely
explains anything.

### Deciding when two metric names mean the same thing

This is where a fact layer quietly goes wrong. Merge too eagerly and you invent
contradictions: `revenue from operations` and `revenue from services` differ by
exactly the traded-goods line, so conflating them manufactures a conflict out of
correct arithmetic. Merge too timidly and cross-document corroboration is never
found, because one report writes `real GDP growth` and another writes
`real gross domestic product (GDP) growth`.

Four tiers, increasing in cost and decreasing in certainty:

1. identical normalised names → same, free, certain
2. high lexical similarity → adjudicate
3. high **semantic** similarity → adjudicate
4. neither → never compared

Tier 3 exists because string distance is blind to abbreviation:
`PTL freight tonnage` and `Part-truckload tonnage` are the same measurement and
score 63 on token-set ratio — below any threshold that isn't also flooded with
noise. Adjudications are cached in `data/metric_aliases.json`, so the alias map
is built from whatever the corpus actually contains.

### Engineering decisions and trade-offs

**SQLite over a graph database.** The brief notes that a graph database is not
the solution, and it is right: the interesting content is the *reasoning
attached to each edge*, not the topology. A graph store would have added
operational weight and answered no question the schema here cannot.

**Pages are batched into model calls.** Roughly five pages per call, bounded by
a character budget. This cuts call count about fivefold, keeps a 500-page corpus
inside free-tier request quotas, and — usefully — gives the model the
neighbouring pages it needs when a table's column headers sit on the page before
its rows. Facts are attributed per page; if the model misattributes one, the
verifier relocates it within the batch and records that it had to.

**Blocking before comparison.** Comparing all *N*² fact pairs is both wasteful
and meaningless. Facts are grouped by canonical metric, with bridges between
groups that are lexically or semantically close, so work stays proportional to
how much the corpus actually overlaps.

**Incremental ingestion.** Adding a document compares its facts against stored
facts and each other, never the whole corpus against itself. The Nth document
costs O(new × existing), not O(total²). Corpus-level dimension statistics are
carried in the store and updated as pairs accumulate.

**A rate limiter and key rotation.** The first full run silently lost 18 of 22
pages to HTTP 429s, which surfaced as "fewer facts" rather than as an error —
the most dangerous kind of failure. There is now a shared token bucket, retry
with server-supplied backoff, and optional rotation across several keys
(`GEMINI_API_KEYS=k1,k2,...`).

### AI tools used

- **Gemini 2.5 Flash** for the three model-facing jobs: document profiling,
  fact proposal, and adjudicating whether two metric names match. Provider is
  pluggable (`FACTLAYER_LLM=gemini|openai|anthropic|replay`); OpenAI and
  Anthropic paths are implemented.
- **`text-embedding-004`** for semantic candidate generation between metric
  names, cached to `data/embeddings.json`.
- **Claude Code (Opus)** as a pair-programmer throughout: designing the
  precision-interval approach, writing the modules and the evaluation set, and
  finding the rate-limit bug above.

---

## Limitations and next steps

**What does not work well yet**

- **Coverage is well below detection.** The deterministic scanner finds far more
  salient quantities than become facts. The extractor is instructed to prefer
  figures a reader would cite, so some of that gap is intentional — but not all
  of it, and the honest position is that recall on dense financial tables is the
  weakest part of the system. Per-page coverage is reported in **Diagnostics**
  precisely so this is visible rather than hidden.
- **Wide tables remain the hardest case.** See "A failure I found", below.
- **Confidence is principled but uncalibrated.** The score composes real signals
  and is not arbitrary, but no labelled sample has been used to check that 0.8
  means 80 %. `evals/run_evals.py --with-db` reports the distribution; turning
  that into a reliability curve needs a few hundred adjudicated pairs.
- **Semantic facts are shallow.** Status claims are extracted, but a director
  appearing in a 2022 prospectus and absent from a 2024 report is an *absence*,
  and the system reasons over what is present.
- **Scanned PDFs are out of scope.** All six starter documents carry text
  layers; there is no OCR fallback.
- **One declared convention.** Fiscal years are assumed April–March unless a
  document says otherwise. It is configurable and recorded on every period
  rather than buried.

**What I would build next, in order**

1. **A labelled pair set and a reliability curve.** The highest-value missing
   piece. It turns confidence from a plausible number into a measured one, and
   lets the contradiction detector's operating point be *chosen* rather than
   assumed — in due diligence a missed red flag costs far more than a false one
   an analyst dismisses in seconds, so the threshold should sit deliberately
   toward recall.
2. **Table-structure-aware extraction.** Reconstruct the row/column grid from
   word geometry (already captured) and hand the model a cell with its header
   path, rather than a flattened line. This addresses the single largest source
   of both missed and wrong facts.
3. **Derived tie-outs.** The word geometry is enough to check that stated
   components sum to stated totals, turning arithmetic consistency into another
   class of evidence.
4. **Absence as evidence**, so a director present in one document and missing
   from a later one becomes a reviewable observation.

---

## Additional notes

- **No credentials in the repository.** `.env` is gitignored; `.env.example`
  documents the variables. The committed cache means a reviewer can evaluate the
  whole system without an account.
- **Nothing is keyed to the starter documents.** No filename, company, metric
  vocabulary or fixed schema appears in the code. The unit tables encode general
  knowledge (SI prefixes, Indian numbering, currency codes), and the fiscal-year
  convention is a declared, configurable default.
- **The four required cases are selected by query**, not hard-coded — see
  `/api/cases`. Running on a different corpus surfaces that corpus's own
  examples.
- **`/api/relations/{id}` returns the full reasoning record**, not just a
  verdict: the intervals, the dimension comparison, and how the metric names
  were matched and by what means. A reviewer who disagrees with a conclusion can
  see exactly which step produced it.
