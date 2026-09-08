"""Corpus records -> retrieval index (chunk, embed, sparse, store).

Stage 1 chunk : sentence-aware, char-budgeted, sentence-overlap, chapter-tagged
Stage 2 dense : fastembed ONNX bge-small-en-v1.5 (384d), CPU, no torch
Stage 3 sparse: TF-IDF word 1-2 grams, sublinear
Stage 4 store : numpy dense matrix + scipy sparse matrix + JSONL chunk metadata

Usage:
    python build_index.py corpus index
"""
from __future__ import annotations

import json
import pickle
import re
import sys
import time
from pathlib import Path

import numpy as np

MODEL = "BAAI/bge-small-en-v1.5"
TARGET_CHARS = 1000
OVERLAP_SENTENCES = 1
MIN_CHARS = 200
SENT_SPLIT = re.compile(r'(?<=[.!?"\u201d])\s+')


def load_chapters(corpus_dir: Path) -> list[dict]:
    path = corpus_dir / "corpus.jsonl"
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def chunk_text(text: str) -> list[str]:
    """Sentence-aware packing to a char budget, with sentence overlap."""
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    sents: list[str] = []
    for p in paras:
        sents.extend(s.strip() for s in SENT_SPLIT.split(p) if s.strip())

    chunks, buf, size = [], [], 0
    for s in sents:
        if size + len(s) + 1 > TARGET_CHARS and size >= MIN_CHARS:
            chunks.append(" ".join(buf))
            buf = buf[-OVERLAP_SENTENCES:] if OVERLAP_SENTENCES else []
            size = sum(len(x) + 1 for x in buf)
        buf.append(s)
        size += len(s) + 1
    if buf and size >= MIN_CHARS:
        chunks.append(" ".join(buf))
    return chunks


def build_chunks(chapters: list[dict]) -> list[dict]:
    rows, cid = [], 0
    for c in chapters:
        for i, ch in enumerate(chunk_text(c["text"])):
            rows.append({
                "id": cid,
                "chapter_index": c["chapter_index"],
                "chapter": c["title"],
                "pages": c["pages"],
                "chunk_in_chapter": i,
                "chars": len(ch),
                "words": len(ch.split()),
                "text": ch,
            })
            cid += 1
    return rows


def default_embedder(texts: list[str], batch_size: int = 64) -> np.ndarray:
    """fastembed ONNX CPU embeddings, L2-normalized. One embedding implementation."""
    from fastembed import TextEmbedding
    model = TextEmbedding(model_name=MODEL)
    dense = np.array(list(model.embed(texts, batch_size=batch_size)), dtype=np.float32)
    return dense / (np.linalg.norm(dense, axis=1, keepdims=True) + 1e-12)


def build(corpus_dir: Path, out_dir: Path, embed_fn=None) -> dict:
    timing: dict = {}
    t0 = time.perf_counter()
    chapters = load_chapters(corpus_dir)
    chunks = build_chunks(chapters)
    if not chunks:
        raise SystemExit(f"no chunks produced from {corpus_dir}")
    timing["chunk_s"] = round(time.perf_counter() - t0, 2)
    texts = [c["text"] for c in chunks]

    t0 = time.perf_counter()
    dense = np.asarray((embed_fn or default_embedder)(texts), dtype=np.float32)
    dense /= np.linalg.norm(dense, axis=1, keepdims=True) + 1e-12
    timing["embed_s"] = round(time.perf_counter() - t0, 2)

    from sklearn.feature_extraction.text import TfidfVectorizer
    t0 = time.perf_counter()
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2),
                          sublinear_tf=True, min_df=1)
    sparse = vec.fit_transform(texts).astype(np.float32)
    timing["tfidf_s"] = round(time.perf_counter() - t0, 2)

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "dense.npy", dense)
    from scipy import sparse as sp
    sp.save_npz(out_dir / "tfidf.npz", sparse)
    with (out_dir / "tfidf_vectorizer.pkl").open("wb") as f:
        pickle.dump(vec, f)
    with (out_dir / "chunks.jsonl").open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    meta = json.loads((corpus_dir / "corpus_meta.json").read_text(encoding="utf-8")) \
        if (corpus_dir / "corpus_meta.json").exists() else {}
    manifest = {
        "corpus": {"chapters": len(chapters), "pages": meta.get("pages"),
                   "input_format": meta.get("input_format")},
        "chunks": len(chunks),
        "chunk_chars": {
            "target": TARGET_CHARS,
            "mean": round(float(np.mean([c["chars"] for c in chunks])), 1),
            "min": min(c["chars"] for c in chunks),
            "max": max(c["chars"] for c in chunks),
        },
        "dense_model": MODEL,
        "dense_dim": int(dense.shape[1]),
        "dense_backend": "fastembed onnxruntime, CPU",
        "sparse_model": "tfidf word 1-2gram sublinear, english stopwords",
        "sparse_nnz": int(sparse.nnz),
        "timing_s": timing,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    corpus_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "corpus")
    out_dir = Path(sys.argv[2] if len(sys.argv) > 2 else "index")
    print(json.dumps(build(corpus_dir, out_dir), indent=2))


if __name__ == "__main__":
    main()
