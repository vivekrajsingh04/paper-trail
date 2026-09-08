#!/usr/bin/env python
"""Print the corpus figures quoted in the README, straight from the layer.

    python scripts/report_stats.py

Numbers in a README rot the moment the code changes. This regenerates every
figure the README quotes from the live database, so the claim and the artefact
can be checked against each other in one command.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from factlayer.store import Store  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default="data/factlayer.db")
    args = ap.parse_args()

    s = Store(args.db)
    docs = s.list_documents()
    rels = s.relation_counts()
    bv = rels.get("by_verdict", {})

    print("## Corpus\n")
    print(f"{'document':46s} {'pp':>4s} {'sent':>5s} {'facts':>6s} {'verified':>9s} {'coverage':>9s}")
    tot_pp = tot_facts = 0
    for d in docs:
        st = d.get("stats") or {}
        vr = st.get("verification_pass_rate")
        cr = st.get("coverage_ratio")
        tot_pp += d["n_pages"]
        tot_facts += d["n_facts"]
        print(f"{(d['title'] or d['filename'])[:44]:46s} {d['n_pages']:4d} "
              f"{st.get('pages_sent', 0):5d} {d['n_facts']:6d} "
              f"{(f'{vr:.1%}' if vr is not None else '—'):>9s} "
              f"{(f'{cr:.1%}' if cr is not None else '—'):>9s}")
    print(f"{'TOTAL':46s} {tot_pp:4d} {'':5s} {tot_facts:6d}")

    print("\n## Relations\n")
    print(f"  total            {rels.get('total', 0)}")
    print(f"  corroborates     {bv.get('corroborates', 0)}")
    print(f"  reconciled       {bv.get('reconciled', 0)}")
    print(f"  contradicts      {bv.get('contradicts', 0)}")
    print(f"  cross-document   {rels.get('cross_document', 0)}")

    print("\n## Learned dimension weights\n")
    dims = s.get_meta("dim_stats", {}) or {}
    ranked = sorted(
        dims.items(),
        key=lambda kv: -((kv[1]["value_differs"] + 2) / (kv[1]["pairs"] + 4)),
    )
    for d, rec in ranked[:10]:
        power = (rec["value_differs"] + 2) / (rec["pairs"] + 4)
        print(f"  {d:22s} {power:5.3f}   ({rec['value_differs']}/{rec['pairs']} pairs)")

    import json
    from pathlib import Path as P

    aliases = P("data/metric_aliases.json")
    embeds = P("data/embeddings.json")
    print("\n## Caches (committed, so a reviewer needs no API key)\n")
    cache_files = list(P("data/cache").rglob("*.json"))
    print(f"  model responses   {len(cache_files)}")
    if aliases.exists():
        print(f"  metric verdicts   {len(json.loads(aliases.read_text()))}")
    if embeds.exists():
        print(f"  embeddings        {len(json.loads(embeds.read_text()))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
