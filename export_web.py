"""Export the corpus and index into one JSON bundle for the browser UI.

    python export_web.py corpus index web/data/corpus3d.json

The bundle is generated locally and is gitignored: it contains book text and
the dense chunk vectors, exactly like ``corpus.jsonl`` and ``index/dense.npy``.
It carries the same canonical attributes as the Streamlit app (both read
``corpus_attributes.py``), plus a 3-D PCA projection of the chunk vectors for
the Three.js view.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

from corpus_attributes import (WORD_RE, chapter_rows, character_matrix,
                               load_meta, load_records, project)
from search import BookSearch

PREVIEW_CHARS = 120
_CHAP_RE = re.compile(r"^(?:CHAPTER|Chapter)\s+([A-Za-z0-9-]+)\s*(?:[-\u2013]\s*(.+))?$")


def short_title(title: str) -> str:
    """'CHAPTER 7 - The Sorting Hat' -> 'The Sorting Hat'; 'CHAPTER ONE' -> 'One'."""
    m = _CHAP_RE.match(title.strip())
    if not m:
        return title.strip()
    return (m.group(2) or m.group(1)).strip().title()


def build_bundle(corpus_dir: Path, index_dir: Path,
                 preview_chars: int = PREVIEW_CHARS) -> dict:
    records = load_records(corpus_dir)
    meta = load_meta(corpus_dir)
    manifest = json.loads((index_dir / "manifest.json").read_text(encoding="utf-8"))
    chunks = [json.loads(l) for l in
              (index_dir / "chunks.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    dense = np.load(index_dir / "dense.npy").astype(np.float32)
    if len(chunks) != dense.shape[0]:
        raise SystemExit(f"chunks ({len(chunks)}) and dense rows ({dense.shape[0]}) "
                         f"disagree; rebuild the index")

    xyz = project(dense, n_components=3)
    rows = chapter_rows(records)

    points = [{
        "id": i,
        "ch": c["chapter_index"],
        "page": c["pages"][0] if c.get("pages") else None,
        "words": c["words"],
        "citation": BookSearch.citation(c),
        "preview": c["text"][:preview_chars],
        "text": c["text"],
        "x": round(float(xyz[i, 0]), 5),
        "y": round(float(xyz[i, 1]), 5),
        "z": round(float(xyz[i, 2]), 5),
    } for i, c in enumerate(chunks)]

    chapters = [{**r, "short": short_title(r["title"]),
                 "avg_sentence": round(r["words"] / max(r["sentences"], 1), 1)}
                for r in rows]

    return {
        "meta": {
            "source": meta.get("source"),
            "input_format": meta.get("input_format"),
            "pages": meta.get("pages"),
            "chapters": meta.get("chapters"),
            "words_total": meta.get("words_total"),
            "dialogue_share_pct": meta.get("dialogue_share_pct"),
            "unique_words": len({w.lower() for r in records for w in WORD_RE.findall(r["text"])}),
            "chunks": manifest["chunks"],
            "chunk_chars_mean": manifest["chunk_chars"]["mean"],
            "dense_model": manifest["dense_model"],
            "dense_dim": manifest["dense_dim"],
            "dense_backend": manifest["dense_backend"],
            "sparse_model": manifest["sparse_model"],
            "projection": "PCA-3 over the dense chunk vectors (sklearn, random_state=42)",
        },
        "chapters": chapters,
        "characters": character_matrix(records),
        "vector_dim": int(dense.shape[1]),
        "vectors": [round(float(v), 4) for v in dense.ravel()],
        "points": points,
    }


def main() -> None:
    corpus_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "corpus")
    index_dir = Path(sys.argv[2] if len(sys.argv) > 2 else "index")
    out_path = Path(sys.argv[3] if len(sys.argv) > 3 else "web/data/corpus3d.json")

    bundle = build_bundle(corpus_dir, index_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(bundle, ensure_ascii=False, separators=(",", ":")),
                        encoding="utf-8")
    print(json.dumps({
        "out": str(out_path),
        "bytes": out_path.stat().st_size,
        "chapters": len(bundle["chapters"]),
        "points": len(bundle["points"]),
        "vector_dim": bundle["vector_dim"],
        "characters": len(bundle["characters"]["names"]),
    }, indent=2))


if __name__ == "__main__":
    main()
