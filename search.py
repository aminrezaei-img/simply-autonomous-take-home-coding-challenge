"""Hybrid retrieval over the book index.

dense  : cosine similarity over fastembed vectors (numpy exact; the index is
         small enough that an ANN library would be unearned infrastructure)
sparse : TF-IDF cosine similarity
hybrid : Reciprocal Rank Fusion of the two ranked lists

Every hit carries chapter/page provenance, so an answer can always be traced
back to where it came from.

Usage:
    python search.py "how does the sorting hat decide" [index_dir]
"""
from __future__ import annotations

import json
import pickle
import re
import sys
from pathlib import Path

import numpy as np

RRF_K = 60
DEFAULT_INDEX = Path(__file__).resolve().parent / "index"


class HashingEncoder:
    """Deterministic char-hash bag-of-words encoder.

    No model download, no network. Used as an offline fallback when the ONNX
    model cannot load, and by the test suite so tests never depend on a
    downloaded model. Quality is far below the real embedder; it only exists
    so the tool degrades gracefully instead of failing.
    """

    def __init__(self, dim: int = 96):
        self.dim = dim

    def __call__(self, texts: list[str]) -> np.ndarray:
        import zlib
        m = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for w in re.findall(r"[a-z0-9']+", t.lower()):
                m[i, zlib.crc32(w.encode()) % self.dim] += 1.0
        return m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-12)


class BookSearch:
    def __init__(self, index_dir: Path | str = DEFAULT_INDEX, encoder=None):
        """encoder: optional callable(list[str]) -> (n, dim) float array.
        Defaults to fastembed, matching the document encoder used at build time."""
        self.dir = Path(index_dir)
        self.manifest = json.loads((self.dir / "manifest.json").read_text(encoding="utf-8"))
        self.chunks = [json.loads(l) for l in
                       (self.dir / "chunks.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
        self.dense = np.load(self.dir / "dense.npy")
        from scipy import sparse
        self.sparse = sparse.load_npz(self.dir / "tfidf.npz")
        with (self.dir / "tfidf_vectorizer.pkl").open("rb") as f:
            self.vec = pickle.load(f)
        self._encoder = encoder
        self._model = None

    # ------------------------------------------------------------ encoding
    def _embed_query(self, query: str) -> np.ndarray:
        if self._encoder is not None:
            v = np.asarray(self._encoder([query]), dtype=np.float32)
        else:
            if self._model is None:
                from fastembed import TextEmbedding
                self._model = TextEmbedding(model_name=self.manifest["dense_model"])
            v = np.array(list(self._model.query_embed([query])), dtype=np.float32)
        return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)

    # ------------------------------------------------------------ filtering
    def _allowed(self, chapter: int | None, page: int | None) -> list[int]:
        out = []
        for i, c in enumerate(self.chunks):
            if chapter is not None and c["chapter_index"] != chapter:
                continue
            if page is not None:
                if not c["pages"] or not (c["pages"][0] <= page <= c["pages"][1]):
                    continue
            out.append(i)
        return out

    @staticmethod
    def citation(chunk: dict) -> str:
        if chunk.get("pages"):
            return (f"{chunk['chapter']} (pp. {chunk['pages'][0]}-{chunk['pages'][1]}), "
                    f"chunk {chunk['chunk_in_chapter']}")
        return f"{chunk['chapter']}, chunk {chunk['chunk_in_chapter']}"

    # ------------------------------------------------------------ retrieval
    def search(self, query: str, k: int = 8, mode: str = "hybrid",
               chapter: int | None = None, page: int | None = None) -> list[dict]:
        allowed = set(self._allowed(chapter, page))
        if not allowed:
            return []
        depth = min(len(self.chunks), max(k * 10, 50))

        dense_rank: dict[int, int] = {}
        if mode in ("hybrid", "dense"):
            q = self._embed_query(query)[0]
            if q.shape[0] != self.dense.shape[1]:
                raise ValueError(
                    f"query embedding has {q.shape[0]} dims but the index has "
                    f"{self.dense.shape[1]}; rebuild the index with the same encoder "
                    f"({self.manifest.get('dense_model')})")
            scores = self.dense @ q
            order = np.argsort(-scores, kind="stable")[:depth]
            rank = 0
            for i in order:
                if int(i) in allowed:
                    rank += 1
                    dense_rank[int(i)] = rank

        sparse_rank: dict[int, int] = {}
        if mode in ("hybrid", "sparse"):
            qv = self.vec.transform([query])
            sims = (self.sparse @ qv.T).toarray().ravel()
            order = np.argsort(-sims, kind="stable")[:depth]
            rank = 0
            for i in order:
                if sims[i] <= 0:
                    break
                if int(i) in allowed:
                    rank += 1
                    sparse_rank[int(i)] = rank

        fused: dict[int, float] = {}
        for r in (dense_rank, sparse_rank):
            for i, rk in r.items():
                fused[i] = fused.get(i, 0.0) + 1.0 / (RRF_K + rk)

        # deterministic: RRF score desc, chunk id asc on ties
        top = sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
        hits = []
        for i, score in top:
            c = self.chunks[i]
            hits.append({
                "score": round(score, 6),
                "dense_rank": dense_rank.get(i),
                "sparse_rank": sparse_rank.get(i),
                "chapter_index": c["chapter_index"],
                "chapter": c["chapter"],
                "pages": c["pages"],
                "citation": self.citation(c),
                "text": c["text"],
            })
        return hits

    def highlight(self, text: str, query: str, width: int = 220) -> str:
        terms = [t for t in re.findall(r"\w+", query.lower()) if len(t) > 2]
        low = text.lower()
        pos = min((low.find(t) for t in terms if low.find(t) >= 0), default=0)
        start = max(0, pos - width // 3)
        snippet = text[start:start + width]
        return ("..." if start else "") + snippet + ("..." if start + width < len(text) else "")


def main() -> None:
    args = sys.argv[1:]
    idx = next((Path(a) for a in args if Path(a).is_dir()), DEFAULT_INDEX)
    query = " ".join(a for a in args if not Path(a).is_dir()) or "what does the sorting hat do"
    s = BookSearch(idx)
    print(f"query: {query!r}  (chunks={s.manifest['chunks']}, model={s.manifest['dense_model']})")
    for mode in ("hybrid", "dense", "sparse"):
        print(f"\n--- {mode} ---")
        for h in s.search(query, k=3, mode=mode):
            print(f"  {h['score']:.5f} d={h['dense_rank']} s={h['sparse_rank']} | {h['citation']}")
            print(f"    {h['text'][:150]}...")


if __name__ == "__main__":
    main()
