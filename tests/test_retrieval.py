"""Retrieval behaviour: provenance, filters, deterministic fusion, citations."""
from __future__ import annotations

from pathlib import Path

from search import BookSearch, HashingEncoder


def _search(index_dir) -> BookSearch:
    """Tests use the offline encoder: the tiny index is built with it, and it
    needs no model download."""
    return BookSearch(index_dir, encoder=HashingEncoder())


def test_every_chunk_carries_provenance(tiny_index: dict, tmp_path: Path):
    chunks = [__import__("json").loads(l) for l in
              (tmp_path / "index" / "chunks.jsonl").read_text(encoding="utf-8").splitlines()]
    assert chunks
    assert [c["id"] for c in chunks] == list(range(len(chunks)))
    for c in chunks:
        assert c["chapter_index"] in (1, 2, 3)
        assert c["chapter"].startswith("CHAPTER")
        assert c["chunk_in_chapter"] >= 0
        assert c["text"].strip()
    # chunk_in_chapter restarts inside each chapter
    assert {c["chapter_index"] for c in chunks} == {1, 2, 3}
    assert min(c["chunk_in_chapter"] for c in chunks) == 0


def test_metadata_filter_never_leaks_across_chapters(tiny_index: dict, tmp_path: Path):
    s = _search(tmp_path / "index")
    hits = s.search("storm hatches waves", k=5, mode="hybrid", chapter=3)
    assert hits, "chapter filter returned nothing"
    assert {h["chapter_index"] for h in hits} == {3}
    assert all("CHAPTER THREE" in h["citation"] for h in hits)


def test_rrf_fusion_is_deterministic(tiny_index: dict, tmp_path: Path):
    s = _search(tmp_path / "index")
    first = [h["citation"] for h in s.search("lamp burning gulls", k=5, mode="hybrid")]
    second = [h["citation"] for h in s.search("lamp burning gulls", k=5, mode="hybrid")]
    assert first == second


def test_hybrid_hit_carries_a_citation(tiny_index: dict, tmp_path: Path):
    s = _search(tmp_path / "index")
    hits = s.search("keep the lamp burning", k=3, mode="hybrid")
    assert hits
    top = hits[0]
    assert top["citation"].startswith("CHAPTER")
    assert "chunk" in top["citation"]
    assert top["chapter_index"] in (1, 2, 3)
