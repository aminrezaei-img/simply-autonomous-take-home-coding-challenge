"""Corpus ingestion: plain text or PDF -> one normalized chapter representation.

    book.txt --\
               +--> chapters[] --> corpus/corpus.jsonl      (one record per chapter)
    book.pdf --/                --> corpus/corpus_meta.json (corpus-level attributes)

Both inputs converge on the same record shape, so every downstream stage
(chunking, indexing, retrieval, analytics) is input-format agnostic:

    {"chapter_index": 1, "title": "CHAPTER ONE",
     "pages": [10, 22] | None, "paragraphs": 42, "words": 3120, "text": "..."}

PDF path  : PyMuPDF block geometry + reading order, page-furniture removal,
            cross-page paragraph stitching, TOC-aware chapter boundaries.
TXT path  : typographic normalization, Gutenberg header/footer removal,
            CHAPTER-heading split (falls back to a single chapter if the file
            has no headings, rather than failing silently).

Usage:
    python ingest.py data/raw/book.txt corpus
    python ingest.py data/raw/book.pdf corpus
"""
from __future__ import annotations

import json
import re
import sys
import time
import unicodedata
from pathlib import Path

import pymupdf

# ---------------------------------------------------------------- patterns
PAGE_NUM = re.compile(r"^[\s]*(?:[ivxlcdm]+|\d{1,4})[\s]*$", re.I)
HYPHEN_BREAK = re.compile(r"(\w)-\s*\n\s*(\w)")
SENT_END = re.compile(r'[.!?"\u201d\u2019)\]]\s*$')
FRONT_MATTER_WORDS = 10  # a TOC node / preamble below this is not a chapter
HEADING = re.compile(r"^[ \t]*(?:CHAPTER|Chapter)\s+(?:[A-Za-z]+|\d{1,3}|[IVXLCDM]+)\b[^\n]*$")
QUOTED = re.compile(r'"[^"]+"')
GUTENBERG_START = re.compile(r"\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG.*?\*\*\*", re.I | re.S)
GUTENBERG_END = re.compile(r"\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG.*?\*\*\*", re.I | re.S)
PARA_SPLIT = re.compile(r"\n\s*\n")

PUNCT_MAP = {
    0x2018: "'", 0x2019: "'", 0x201C: '"', 0x201D: '"',
    0x2010: "-", 0x2011: "-", 0x2012: "-", 0x2013: "-",
    0x2014: "-", 0x2015: "-", 0x2026: "...", 0x00A0: " ",
}


def normalize(text: str) -> str:
    """Fold typographic punctuation to ASCII so one set of regexes works on
    both PDF-extracted and plain-text sources (curly quotes are the single
    biggest cause of silent dialogue-extraction failure)."""
    return unicodedata.normalize("NFKC", text).translate(PUNCT_MAP)


def dialogue_words(text: str) -> int:
    """Words inside straight double quotes. The one dialogue definition."""
    return sum(len(m.split()) for m in QUOTED.findall(text))


def clean_block(text: str) -> str:
    """Join wrapped lines inside one layout block into a single paragraph."""
    text = HYPHEN_BREAK.sub(r"\1\2", text)
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    return " ".join(lines).strip()


# ---------------------------------------------------------------- PDF path
def page_blocks(page: pymupdf.Page) -> list[tuple[float, float, str]]:
    """Text blocks in reading order, with page-number furniture removed."""
    height = page.rect.height
    out = []
    for b in page.get_text("blocks"):
        x0, y0, x1, y1, text, _no, btype = b
        if btype != 0:
            continue
        raw = text.strip()
        if not raw:
            continue
        if PAGE_NUM.match(raw) and (y1 < height * 0.10 or y0 > height * 0.90):
            continue
        para = clean_block(raw)
        if para:
            out.append((y0, x0, para))
    out.sort(key=lambda t: (round(t[0], 1), t[1]))
    return out


def stitch(paras: list[str]) -> list[str]:
    """Merge paragraphs split across a page break (mid-sentence)."""
    merged: list[str] = []
    for p in paras:
        if merged:
            prev = merged[-1]
            if not SENT_END.search(prev) and p[:1].islower():
                merged[-1] = prev + " " + p
                continue
        merged.append(p)
    return merged


def chapters_from_pdf(path: Path) -> tuple[list[dict], int, int]:
    doc = pymupdf.open(path)
    toc = doc.get_toc()

    starts = sorted({e[2] - 1 for e in toc if e[0] == 1})
    if not starts or starts[0] != 0:
        starts = sorted(set(starts) | {0})
    bounds = [(starts[i], starts[i + 1] if i + 1 < len(starts) else doc.page_count)
              for i in range(len(starts))]
    titles = {e[2] - 1: e[1] for e in toc if e[0] == 1}

    sections = []
    for a, b in bounds:
        paras: list[str] = []
        for pno in range(a, b):
            paras.extend(p for *_xy, p in page_blocks(doc[pno]))
        paras = stitch(paras)
        sections.append((titles.get(a) or (paras[0] if paras else "untitled"), a, b, paras))

    body = [s for s in sections if sum(len(p.split()) for p in s[3]) >= FRONT_MATTER_WORDS]
    chapters = []
    for i, (title, a, b, paras) in enumerate(body, 1):
        text = "\n\n".join(paras)
        chapters.append({
            "chapter_index": i,
            "title": title,
            "pages": [a + 1, b],
            "paragraphs": len(paras),
            "words": len(text.split()),
            "text": text,
        })
    return chapters, doc.page_count, len(sections) - len(body)


# ---------------------------------------------------------------- TXT path
def chapters_from_txt(path: Path) -> tuple[list[dict], None, int]:
    raw = normalize(path.read_text(encoding="utf-8", errors="replace"))
    m = GUTENBERG_START.search(raw)
    if m:
        raw = raw[m.end():]
    m = GUTENBERG_END.search(raw)
    if m:
        raw = raw[:m.start()]

    lines = raw.splitlines()
    marks = [i for i, ln in enumerate(lines)
             if HEADING.match(ln) and len(ln.strip()) <= 60]

    chapters = []
    if len(marks) >= 2:
        preamble = "\n".join(lines[:marks[0]]).strip()
        front = 1 if len(preamble.split()) >= FRONT_MATTER_WORDS else 0
        for n, i in enumerate(marks):
            j = marks[n + 1] if n + 1 < len(marks) else len(lines)
            body = "\n".join(lines[i + 1:j]).strip()
            paras = [clean_block(p) for p in PARA_SPLIT.split(body) if p.strip()]
            text = "\n\n".join(paras)
            chapters.append({
                "chapter_index": n + 1,
                "title": lines[i].strip(),
                "pages": None,
                "paragraphs": len(paras),
                "words": len(text.split()),
                "text": text,
            })
        return chapters, None, front

    # No usable headings: keep the whole file as one chapter instead of failing.
    paras = [clean_block(p) for p in PARA_SPLIT.split(raw) if p.strip()]
    text = "\n\n".join(paras)
    return [{
        "chapter_index": 1,
        "title": path.stem,
        "pages": None,
        "paragraphs": len(paras),
        "words": len(text.split()),
        "text": text,
    }], None, 0


# ---------------------------------------------------------------- driver
def build_corpus(src: Path, out_dir: Path) -> dict:
    t0 = time.perf_counter()
    if not src.exists():
        raise SystemExit(f"input not found: {src}")
    suffix = src.suffix.lower()
    if suffix == ".pdf":
        chapters, pages, front = chapters_from_pdf(src)
        fmt = "pdf"
    elif suffix in (".txt", ".text"):
        chapters, pages, front = chapters_from_txt(src)
        fmt = "txt"
    else:
        raise SystemExit(f"unsupported input {src.name!r}: expected .txt or .pdf")

    if not chapters:
        raise SystemExit(f"no chapters extracted from {src.name}")

    for c in chapters:
        c["dialogue_words"] = dialogue_words(c["text"])

    words_total = sum(c["words"] for c in chapters)
    dlg_total = sum(c["dialogue_words"] for c in chapters)
    meta = {
        "source": src.name,
        "input_format": fmt,
        "pages": pages,
        "chapters": len(chapters),
        "front_matter_skipped": front,
        "words_total": words_total,
        "dialogue_words": dlg_total,
        # single canonical dialogue share for the README, the UI and the tests
        "dialogue_share_pct": round(100 * dlg_total / max(words_total, 1), 2),
        "elapsed_s": round(time.perf_counter() - t0, 3),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "corpus.jsonl").open("w", encoding="utf-8") as f:
        for c in chapters:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    (out_dir / "corpus_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(json.dumps(meta, indent=2))
    for c in chapters:
        pages_lbl = f"{c['pages'][0]:>3}-{c['pages'][1]:<3}" if c["pages"] else "  n/a "
        print(f"  {c['chapter_index']:02d} {pages_lbl} {c['words']:>6} words  {c['title'][:52]}")
    return meta


def main() -> None:
    src = Path(sys.argv[1] if len(sys.argv) > 1 else "data/raw/book.txt")
    out = Path(sys.argv[2] if len(sys.argv) > 2 else "corpus")
    build_corpus(src, out)


if __name__ == "__main__":
    main()
