"""Ingestion behaviour: chapter boundaries, page furniture, continuity, and the
one canonical dialogue share."""
from __future__ import annotations

import json
from pathlib import Path

import ingest


def test_front_matter_toc_node_is_not_a_chapter(synthetic_pdf: Path):
    """Regression: a book-title TOC node must not become narrative chapter 1.

    Synthetic PDF = title page + 3 chapters; exactly 3 chapters must come out.
    """
    chapters, pages, front = ingest.chapters_from_pdf(synthetic_pdf)
    assert [c["title"] for c in chapters] == ["CHAPTER ONE", "CHAPTER TWO", "CHAPTER THREE"]
    assert len(chapters) == 3
    assert front == 1
    assert [c["chapter_index"] for c in chapters] == [1, 2, 3]


def test_page_furniture_and_hyphenation_are_cleaned(synthetic_pdf: Path):
    chapters, *_ = ingest.chapters_from_pdf(synthetic_pdf)
    text = chapters[0]["text"]
    assert "42" not in text                      # page number removed
    assert "light-\nhouse" not in text           # line-break hyphen repaired
    assert "lighthouse beam" in text


def test_paragraph_split_across_pages_is_stitched(synthetic_pdf: Path):
    chapters, *_ = ingest.chapters_from_pdf(synthetic_pdf)
    text = chapters[0]["text"]
    assert "climbed the stair and found the lamp unlit" in text
    assert '"We must keep the lamp burning," said Marlow.' in text


def test_txt_input_is_first_class(synthetic_txt: Path, synthetic_pdf: Path):
    """Plain text takes the same path and produces the same record shape."""
    chapters, pages, front = ingest.chapters_from_txt(synthetic_txt)
    assert len(chapters) == 3
    assert [c["title"] for c in chapters] == ["CHAPTER ONE", "CHAPTER TWO", "CHAPTER THREE"]
    assert pages is None
    assert front == 1                            # "The Lighthouse Keeper / by A. Author"

    pdf_chapters, *_ = ingest.chapters_from_pdf(synthetic_pdf)
    assert set(chapters[0]) == set(pdf_chapters[0])


def test_dialogue_share_is_one_canonical_calculation(tiny_corpus: dict, tmp_path: Path):
    """The meta value must equal a recomputation from the same corpus records,
    so the README, the UI and the tests can never disagree."""
    meta = json.loads((tmp_path / "corpus" / "corpus_meta.json").read_text(encoding="utf-8"))
    rows = [json.loads(l) for l in
            (tmp_path / "corpus" / "corpus.jsonl").read_text(encoding="utf-8").splitlines()]

    per_chapter = sum(ingest.dialogue_words(r["text"]) for r in rows)
    total = sum(len(r["text"].split()) for r in rows)

    assert meta["dialogue_words"] == per_chapter
    assert meta["words_total"] == total
    assert meta["dialogue_share_pct"] == round(100 * per_chapter / total, 2)
    assert meta["dialogue_share_pct"] == tiny_corpus["dialogue_share_pct"]
