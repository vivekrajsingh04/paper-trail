"""Deciding when two differently-worded metrics are the same measurement.

This is the blocking step, and it is where a fact layer quietly goes wrong.
Merge too eagerly and the system invents contradictions between things that were
never comparable -- "revenue from operations" and "revenue from services" differ
by exactly the traded-goods line, so conflating them manufactures a conflict out
of correct arithmetic.  Merge too timidly and genuine corroboration across
documents is never found, because one report says "real GDP growth" and another
says "real gross domestic product (GDP) growth".

The resolution is three tiers of increasing cost and decreasing certainty:

    tier 1  identical normalised strings          -> same, free, certain
    tier 2  high lexical similarity               -> ask the model once, cache it
    tier 3  everything else                       -> never compared

No metric vocabulary is hard-coded.  The alias map is built from whatever
metrics the corpus actually contains, so it grows as documents arrive.
"""

from __future__ import annotations

import json
import re
import threading
from pathlib import Path

from rapidfuzz import fuzz, process

from .embed import semantic_pairs
from .llm import LLMUnavailable, complete_json
from .prompts import METRIC_MATCH_PROMPT

ALIAS_PATH = Path("data/metric_aliases.json")

# Words that carry no discriminating power in a metric name.
_STOP = {
    "the", "a", "an", "of", "for", "in", "on", "at", "to", "and", "as",
    "total", "value", "amount", "figure", "level", "rs", "inr", "usd",
    "million", "millions", "crore", "crores", "lakh", "billion", "mn", "cr", "bn",
    "per", "cent", "percent", "percentage", "pct",
}

# Modifiers that MUST survive normalisation: dropping them merges metrics that
# are genuinely different measurements.
_KEEP = {
    "adjusted", "unadjusted", "restated", "reported", "underlying", "normalised",
    "normalized", "pro", "forma", "gross", "net", "real", "nominal", "core",
    "headline", "average", "median", "per", "share", "diluted", "basic",
    "standalone", "consolidated", "service", "services",
}


def normalise_metric(metric: str) -> str:
    """Reduce a metric name to a comparable key without losing modifiers."""
    s = (metric or "").lower().strip()
    s = re.sub(r"\(.*?\)", " ", s)          # drop parentheticals: "(Rs. Mn)"
    s = re.sub(r"[^\w\s%]", " ", s)          # punctuation to space
    s = re.sub(r"\s+", " ", s).strip()

    tokens = []
    for t in s.split():
        if t in _KEEP:
            tokens.append(t)
            continue
        if t in _STOP:
            continue
        if len(t) > 3 and t.endswith("s") and not t.endswith("ss"):
            t = t[:-1]                       # crude singularisation
        tokens.append(t)
    return " ".join(tokens) or s


class MetricResolver:
    """Clusters metric names into comparable groups, caching model adjudications."""

    # Above this on token_sort_ratio, two names are trivial variants of each
    # other (punctuation, plural, word order) and no adjudication is needed.
    # token_sort_ratio is used rather than token_set_ratio because the latter
    # scores "EBITDA" against "Adjusted EBITDA" at 100 -- it ignores the extra
    # token, which is exactly the distinction that must not be lost.
    CERTAIN_LEXICAL = 95.0

    def __init__(
        self,
        fuzzy_threshold: float = 80.0,
        semantic_threshold: float = 0.80,
        use_llm: bool = True,
        adjudication_budget: int = 120,
    ):
        self.fuzzy_threshold = fuzzy_threshold
        self.semantic_threshold = semantic_threshold
        self.use_llm = use_llm
        self.adjudication_budget = adjudication_budget
        self.adjudications_used = 0
        self.budget_exhausted = 0
        self._verdicts: dict[str, dict] = {}
        self._sem: dict[tuple[str, str], float] = {}
        self._sem_ready = False
        self._lock = threading.Lock()
        self._load()

    def index_metrics(self, metrics: list[str]) -> dict:
        """Precompute semantic neighbours for the corpus's metric vocabulary.

        Called once before comparison.  Populates the semantic side of candidate
        generation; if embeddings are unavailable this is a no-op and the
        resolver runs lexical-only.
        """
        keys = [normalise_metric(m) for m in metrics if m]
        pairs = semantic_pairs(keys, threshold=self.semantic_threshold)
        with self._lock:
            for a, b, score in pairs:
                self._sem[(a, b) if a < b else (b, a)] = score
            self._sem_ready = bool(pairs)
        return {
            "unique_metrics": len(set(keys)),
            "semantic_pairs": len(pairs),
            "embeddings_available": self._sem_ready,
        }

    def _semantic_score(self, ka: str, kb: str) -> float:
        return self._sem.get((ka, kb) if ka < kb else (kb, ka), 0.0)

    # -- persistence ------------------------------------------------------
    def _load(self) -> None:
        if ALIAS_PATH.exists():
            try:
                self._verdicts = json.loads(ALIAS_PATH.read_text())
            except json.JSONDecodeError:
                self._verdicts = {}

    def stats(self) -> dict:
        return {
            "cached_verdicts": len(self._verdicts),
            "adjudications_used": self.adjudications_used,
            "adjudication_budget": self.adjudication_budget,
            "budget_exhausted_pairs": self.budget_exhausted,
            "semantic_index": len(self._sem),
        }

    def save(self) -> None:
        with self._lock:
            ALIAS_PATH.parent.mkdir(parents=True, exist_ok=True)
            ALIAS_PATH.write_text(json.dumps(self._verdicts, indent=1, sort_keys=True))

    @staticmethod
    def _pair_key(a: str, b: str) -> str:
        return " ".join(sorted([a, b]))

    # -- candidate generation --------------------------------------------
    def candidate_pairs(self, keys: list[str], limit_per_key: int = 12) -> list[tuple[str, str, float]]:
        """Non-identical metric keys close enough lexically to be worth adjudicating.

        The semantic half of candidate generation lives in `index_metrics`;
        this covers spelling-level variation only.
        """
        uniq = sorted(set(keys))
        if len(uniq) < 2:
            return []
        pairs: dict[tuple[str, str], float] = {}
        matches = process.cdist(uniq, uniq, scorer=fuzz.token_set_ratio, workers=-1)
        for i, a in enumerate(uniq):
            row = matches[i]
            ranked = sorted(range(len(uniq)), key=lambda j: -row[j])[: limit_per_key + 1]
            for j in ranked:
                if i == j:
                    continue
                score = float(row[j])
                if score < self.fuzzy_threshold:
                    continue
                key = (a, uniq[j]) if a < uniq[j] else (uniq[j], a)
                pairs[key] = max(pairs.get(key, 0.0), score)
        return [(a, b, s) for (a, b), s in sorted(pairs.items(), key=lambda kv: -kv[1])]

    # -- adjudication -----------------------------------------------------
    def same_metric(
        self,
        a_metric: str, a_subject: str | None, a_unit: str | None, a_quote: str,
        b_metric: str, b_subject: str | None, b_unit: str | None, b_quote: str,
    ) -> dict:
        """Are these two metrics directly comparable?

        Returns {same, relationship, reason, source}.  `source` records how the
        answer was reached so a reviewer can tell a free string match from a
        model judgement.
        """
        ka, kb = normalise_metric(a_metric), normalise_metric(b_metric)
        if ka == kb:
            return {"same": True, "relationship": "identical",
                    "reason": "identical normalised metric name", "source": "exact"}

        lex = float(fuzz.token_set_ratio(ka, kb))
        lex_sort = float(fuzz.token_sort_ratio(ka, kb))
        sem = self._semantic_score(ka, kb)

        # Unambiguously the same wording: decide for free.
        if lex_sort >= self.CERTAIN_LEXICAL:
            return {"same": True, "relationship": "identical",
                    "reason": f"near-identical wording (token-sort {lex_sort:.0f})",
                    "source": "lexical_high", "lexical": lex, "semantic": sem}

        # Unambiguously unrelated: decide for free.
        if lex < self.fuzzy_threshold and sem < self.semantic_threshold:
            return {"same": False, "relationship": "different",
                    "reason": f"metric names unrelated (lexical {lex:.0f}, semantic {sem:.2f})",
                    "source": "blocked", "lexical": lex, "semantic": sem}

        pk = self._pair_key(ka, kb)
        with self._lock:
            cached = self._verdicts.get(pk)
        if cached is not None:
            return {**cached, "source": cached.get("source", "cache")}

        if not self.use_llm:
            return {"same": False, "relationship": "unknown",
                    "reason": "lexically close but unadjudicated", "source": "lexical"}

        # Only the genuinely uncertain middle band costs a model call, and the
        # number of those is capped so one corpus cannot exhaust a daily quota.
        with self._lock:
            if self.adjudications_used >= self.adjudication_budget:
                self.budget_exhausted += 1
                return {"same": False, "relationship": "unknown",
                        "reason": "adjudication budget exhausted; not compared",
                        "source": "budget_exhausted", "lexical": lex, "semantic": sem}
            self.adjudications_used += 1

        prompt = METRIC_MATCH_PROMPT.format(
            a_metric=a_metric, a_subject=a_subject, a_unit=a_unit, a_quote=a_quote[:160],
            b_metric=b_metric, b_subject=b_subject, b_unit=b_unit, b_quote=b_quote[:160],
        )
        try:
            res = complete_json(prompt, tag="metric_match")
            d = res.data if isinstance(res.data, dict) else {}
            verdict = {
                "same": bool(d.get("same_metric")),
                "relationship": d.get("relationship") or "different",
                "which_broader": d.get("which_broader"),
                "reason": (d.get("reason") or "")[:240],
                "source": "llm",
                "lexical": lex,
                "semantic": sem,
            }
        except (LLMUnavailable, ValueError, Exception):  # noqa: BLE001
            verdict = {"same": False, "relationship": "unknown",
                       "reason": "adjudication unavailable", "source": "unresolved"}

        with self._lock:
            self._verdicts[pk] = verdict
        return verdict


def subject_key(subject: str | None) -> str:
    """Normalise an entity name enough to match across documents."""
    s = (subject or "").lower().strip()
    s = re.sub(r"\(.*?\)", " ", s)
    s = re.sub(r"\b(limited|ltd|pvt|private|inc|plc|corp|corporation|company|co)\b", " ", s)
    s = re.sub(r"[^\w\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()
