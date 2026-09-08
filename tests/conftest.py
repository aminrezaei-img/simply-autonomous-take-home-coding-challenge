"""Shared fixtures: original synthetic corpus, no copyrighted text anywhere.

The synthetic novel is written for these tests only. It exercises the real
failure modes the pipeline must handle: a front-matter TOC node, page-number
furniture, a hyphenated line break, and a paragraph split across a page.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pymupdf
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import build_index  # noqa: E402
import ingest  # noqa: E402
from search import HashingEncoder  # noqa: E402

TITLE = "The Lighthouse Keeper"

CH1 = ("The keeper climbed the stair and found the lamp unlit. The light-\n"
       "house beam swept across the bay. \"We must keep the lamp burning,\" said Marlow. "
       "The gulls circled the tower all night long, and the wind carried salt across the "
       "narrow landing. Far below, the sea chewed at the black rocks with a patient, "
       "grinding sound that never stopped.")
CH2 = ("Marlow counted the ships. Seven sails crossed the strait before noon, and not one "
       "of them answered his lantern. \"The tide is turning,\" he said, and he wrote the "
       "hour in the log. The harbour bell rang twice for the fishing fleet, then fell "
       "silent again, leaving only the sound of ropes against the mast.")
CH3 = ("The storm broke at dawn. \"Batten the hatches,\" Marlow shouted. The waves "
       "hammered the rocks below the tower, and the lamp swung so hard that its beam cut "
       "wild arcs across the water. By noon the wind had torn the weathervane from the "
       "roof, and the keeper counted every crack in the glass with his own hands.")

TXT = (f"{TITLE}\nby A. Author\nA short novel of the northern coast, written for these tests only.\n\n"
       f"CHAPTER ONE\n\n{CH1}\n\n"
       f"CHAPTER TWO\n\n{CH2}\n\n"
       f"CHAPTER THREE\n\n{CH3}\n")


def _make_pdf(path: Path) -> None:
    doc = pymupdf.open()
    p1 = doc.new_page()                       # title page: front matter, not a chapter
    p1.insert_text((72, 72), TITLE)
    p2 = doc.new_page()
    p2.insert_text((300, 30), "42")           # page-number furniture
    p2.insert_text((72, 72), "CHAPTER ONE")
    p2.insert_text((72, 110), "The keeper climbed the stair and")
    p3 = doc.new_page()                       # chapter one continues across the page break
    p3.insert_text((72, 72), "found the lamp unlit. The light-\nhouse beam swept across "
                             "the bay. \"We must keep the lamp burning,\" said Marlow. "
                             "The gulls circled the tower all night long.")
    p4 = doc.new_page()
    p4.insert_text((72, 72), "CHAPTER TWO")
    p4.insert_text((72, 110), CH2)
    p5 = doc.new_page()
    p5.insert_text((72, 72), "CHAPTER THREE")
    p5.insert_text((72, 110), CH3)
    doc.set_toc([[1, TITLE, 1], [1, "CHAPTER ONE", 2], [1, "CHAPTER TWO", 4],
                 [1, "CHAPTER THREE", 5]])
    doc.save(path)
    doc.close()


@pytest.fixture
def synthetic_txt(tmp_path: Path) -> Path:
    path = tmp_path / "book.txt"
    path.write_text(TXT, encoding="utf-8")
    return path


@pytest.fixture
def synthetic_pdf(tmp_path: Path) -> Path:
    path = tmp_path / "book.pdf"
    _make_pdf(path)
    return path


@pytest.fixture
def tiny_corpus(tmp_path: Path, synthetic_txt: Path) -> dict:
    return ingest.build_corpus(synthetic_txt, tmp_path / "corpus")


@pytest.fixture
def tiny_index(tmp_path: Path, tiny_corpus: dict) -> dict:
    return build_index.build(tmp_path / "corpus", tmp_path / "index", embed_fn=HashingEncoder())
