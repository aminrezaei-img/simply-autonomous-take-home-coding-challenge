"""Canonical corpus attributes, shared by the Streamlit app and the web export.

One definition per attribute. If the Streamlit app and the browser UI ever
disagree about a number, the bug is here, not in either front end.

Extraction methods
    dialogue share      words inside double quotes / total words, per chapter
    lexical stats       unique lowercased word forms, regex ``[A-Za-z']+``
    character presence  whole-word counts over a fixed 26-name lexicon
                        (a presence signal, NOT NER, NOT character discovery)
    sentiment           VADER compound, mean over a ~400-sentence spread sample
    projection          PCA over the dense chunk vectors (2-D or 3-D)
"""
from __future__ import annotations

import json
import re
from pathlib import Path

# Fixed presence lexicon. Whole-word mention counts, NOT named-entity
# recognition and NOT automatic character discovery: the list is hand-picked
# and both front ends say so. See README "character presence".
CHARACTER_PRESENCE = [
    "Harry", "Ron", "Hermione", "Dumbledore", "Hagrid", "Snape", "Voldemort",
    "Draco", "McGonagall", "Dudley", "Vernon", "Petunia", "Dobby", "Sirius",
    "Quirrell", "Neville", "Fred", "George", "Percy", "Wood",
    "Flitwick", "Trelawney", "Filch", "Norbert", "Firenze", "Griphook",
]

WORD_RE = re.compile(r"[A-Za-z']+")
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
_DIALOGUE_RE = re.compile(r'"[^"]+"')
_sia = None


def _analyzer():
    global _sia
    if _sia is None:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        _sia = SentimentIntensityAnalyzer()
    return _sia


def chapter_sentiment(text: str, max_sentences: int = 400) -> float:
    """Mean VADER compound over a spread sample of ~max_sentences sentences."""
    sents = _SENT_SPLIT.split(text)
    if not sents:
        return 0.0
    step = max(1, len(sents) // max_sentences)
    sample = sents[::step]
    return sum(_analyzer().polarity_scores(s)["compound"] for s in sample) / len(sample)


def chapter_rows(records: list[dict]) -> list[dict]:
    """Per-chapter attributes from ``corpus.jsonl`` records. Chapter text is not returned."""
    out = []
    for c in records:
        text = c["text"]
        words = len(text.split())
        dialogue_words = c.get("dialogue_words",
                               sum(len(m.split()) for m in _DIALOGUE_RE.findall(text)))
        out.append({
            "index": c["chapter_index"],
            "title": c["title"],
            "pages": f"{c['pages'][0]}-{c['pages'][1]}" if c.get("pages") else "n/a",
            "words": words,
            "chars": len(text),
            "paragraphs": len([p for p in text.split("\n\n") if p.strip()]),
            "sentences": len(re.findall(r"[.!?](?:\s|$)", text)),
            "dialogue_words": dialogue_words,
            "dialogue_pct": round(100 * dialogue_words / max(words, 1), 2),
            "unique_words": len({w.lower() for w in WORD_RE.findall(text)}),
            "sentiment": round(chapter_sentiment(text), 4),
        })
    return out


def character_matrix(records: list[dict], names=None, top: int = 14) -> dict:
    """Whole-word presence counts per chapter for the fixed lexicon.

    Returns ``{"names": [...], "counts": [[per chapter], ...]}`` for the ``top``
    most-mentioned names. Not NER: unseen names are invisible and pronouns and
    aliases are not resolved.
    """
    names = list(names or CHARACTER_PRESENCE)
    rows = [[len(re.findall(rf"\b{name}\b", c["text"])) for name in names] for c in records]
    totals = [sum(r[i] for r in rows) for i in range(len(names))]
    keep = sorted(range(len(names)), key=lambda i: totals[i], reverse=True)[:top]
    return {"names": [names[i] for i in keep], "counts": [[r[i] for i in keep] for r in rows]}


def project(vectors, n_components: int = 3, seed: int = 42):
    """PCA projection of the dense chunk vectors, shared by both front ends."""
    import numpy as np
    from sklearn.decomposition import PCA
    return PCA(n_components=n_components, random_state=seed).fit_transform(np.asarray(vectors))


def load_records(corpus_dir) -> list[dict]:
    path = Path(corpus_dir) / "corpus.jsonl"
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def load_meta(corpus_dir) -> dict:
    return json.loads((Path(corpus_dir) / "corpus_meta.json").read_text(encoding="utf-8"))
