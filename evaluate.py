"""Retrieval evaluation: Hit@1, Hit@3 and MRR per mode over labelled queries.

    python evaluate.py eval/queries.jsonl index

A query counts as answered when a retrieved chunk contains its `must_contain`
string. A query whose string does not occur anywhere in the corpus is reported
as broken rather than silently scored as a miss — an unanswerable query must
never look like a retrieval failure.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from search import BookSearch

MODES = ("dense", "sparse", "hybrid")


def load_queries(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def evaluate(index_dir: Path, queries_path: Path, k: int = 3) -> dict:
    queries = load_queries(queries_path)
    search = BookSearch(index_dir)
    corpus = " ".join(c["text"] for c in search.chunks).lower()

    rows = []
    for q in queries:
        needle = q["must_contain"].lower()
        row = {"query": q["query"], "must_contain": q["must_contain"],
               "in_corpus": needle in corpus}
        for mode in MODES:
            hits = search.search(q["query"], k=k, mode=mode)
            ranks = [i + 1 for i, h in enumerate(hits) if needle in h["text"].lower()]
            row[f"{mode}_rank"] = ranks[0] if ranks else None
            row[f"{mode}_hit1"] = ranks[:1] == [1]
            row[f"{mode}_hit3"] = bool(ranks)
            row[f"{mode}_rr"] = round(1.0 / ranks[0], 3) if ranks else 0.0
        rows.append(row)

    usable = [r for r in rows if r["in_corpus"]]
    summary = {"queries": len(rows), "usable": len(usable),
               "broken": [r["must_contain"] for r in rows if not r["in_corpus"]]}
    for mode in MODES:
        n = max(len(usable), 1)
        summary[mode] = {
            "hit@1": round(sum(r[f"{mode}_hit1"] for r in usable) / n, 3),
            "hit@3": round(sum(r[f"{mode}_hit3"] for r in usable) / n, 3),
            "mrr": round(sum(r[f"{mode}_rr"] for r in usable) / n, 3),
        }
    return {"summary": summary, "rows": rows}


def main() -> None:
    queries_path = Path(sys.argv[1] if len(sys.argv) > 1 else "eval/queries.jsonl")
    index_dir = Path(sys.argv[2] if len(sys.argv) > 2 else "index")
    result = evaluate(index_dir, queries_path)
    s = result["summary"]

    print(f"\nqueries={s['queries']}  usable={s['usable']}  broken={s['broken']}\n")
    header = f"{'query':52s}" + "".join(f"{m:>22s}" for m in MODES)
    print(header)
    print("-" * len(header))
    for r in result["rows"]:
        cells = "".join(
            f"{('hit@1' if r[f'{m}_hit1'] else ('hit@3' if r[f'{m}_hit3'] else 'miss')):>10s}"
            f"{('  rr=%.2f' % r[f'{m}_rr']):>12s}" for m in MODES)
        flag = "" if r["in_corpus"] else "  [not in corpus]"
        print(f"{r['query'][:50]:52s}{cells}{flag}")
    print()
    for m in MODES:
        print(f"{m:8s} hit@1={s[m]['hit@1']:.2f}  hit@3={s[m]['hit@3']:.2f}  mrr={s[m]['mrr']:.2f}")

    out = Path("eval/results.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
