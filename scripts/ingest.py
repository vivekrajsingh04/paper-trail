#!/usr/bin/env python
"""Ingest PDFs into the knowledge layer from the command line.

    python scripts/ingest.py starter-datasets/starter-datasets/delhivery/*.pdf
    python scripts/ingest.py --corpus delhivery --workers 12 path/to/*.pdf
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from factlayer.config import describe, has_llm_credentials  # noqa: E402
from factlayer.pipeline import ingest_path  # noqa: E402
from factlayer.store import Store  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("paths", nargs="+", help="PDF files to ingest")
    ap.add_argument("--corpus", default=None, help="label grouping related documents")
    ap.add_argument("--workers", type=int, default=8, help="concurrent extraction calls")
    ap.add_argument("--db", default="data/factlayer.db")
    ap.add_argument("--force", action="store_true", help="re-ingest even if already stored")
    args = ap.parse_args()

    cfg = describe()
    print(f"provider={cfg['provider']} model={cfg['model']} "
          f"credentials={'yes' if cfg['credentials_present'] else 'NO'} "
          f"cache_entries={cfg['cache']['entries']}")
    if not has_llm_credentials() and cfg["cache"]["entries"] == 0:
        print("\nNo API key and no cache. Set GEMINI_API_KEY in .env "
              "(https://aistudio.google.com/apikey) and re-run.", file=sys.stderr)
        return 2

    store = Store(args.db)
    files: list[Path] = []
    for p in args.paths:
        path = Path(p)
        files.extend(sorted(path.rglob("*.pdf")) if path.is_dir() else [path])

    t0 = time.time()
    for i, f in enumerate(files, 1):
        corpus = args.corpus or f.parent.name
        print(f"\n[{i}/{len(files)}] {f.name}  (corpus={corpus})")

        last = {"done": -1}

        def progress(ev: dict) -> None:
            if ev.get("stage") == "extracting" and "done" in ev:
                if ev["done"] != last["done"]:
                    last["done"] = ev["done"]
                    pct = 100 * ev["done"] / max(1, ev["total"])
                    print(f"\r    extracting {ev['done']}/{ev['total']} ({pct:.0f}%)",
                          end="", flush=True)
            else:
                print(f"\r    {ev.get('stage')}...", end="", flush=True)

        try:
            res = ingest_path(store, f, corpus=corpus, max_workers=args.workers,
                              progress=progress, force=args.force)
        except KeyboardInterrupt:
            print("\n  interrupted; progress so far is saved in the cache")
            return 130
        print()
        if res["status"] == "already_ingested":
            print(f"    already ingested as {res['doc_id']}")
            continue
        ex = res["extraction"]
        print(f"    {res['facts']} facts, {res['relations']} new relations "
              f"in {res['elapsed_s']}s")
        print(f"    verification pass rate {ex['verification_pass_rate']:.1%} "
              f"({ex['facts_kept']}/{ex['facts_proposed']} proposed facts kept)")
        if ex.get("coverage_ratio") is not None:
            print(f"    quantity coverage {ex['coverage_ratio']:.1%} "
                  f"({ex['quantities_in_facts']}/{ex['quantities_detected']})")
        if res["new_relations_by_verdict"]:
            print(f"    {res['new_relations_by_verdict']}")

    print(f"\n=== corpus summary ({time.time() - t0:.0f}s) ===")
    print(json.dumps(store.summary(), indent=2)[:2400])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
