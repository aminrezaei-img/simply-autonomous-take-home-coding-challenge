"""Export bundle tests: the browser UI reads one generated JSON artifact.

Runs on the synthetic fixture corpus, so no copyrighted text and no model
download are involved.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import export_web  # noqa: E402


def _bundle(tmp_path, tiny_corpus, tiny_index) -> dict:
    out = tmp_path / "web" / "corpus3d.json"
    bundle = export_web.build_bundle(tmp_path / "corpus", tmp_path / "index")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(bundle), encoding="utf-8")
    return json.loads(out.read_text(encoding="utf-8"))


def test_bundle_shape_and_counts(tmp_path, tiny_corpus, tiny_index):
    b = _bundle(tmp_path, tiny_corpus, tiny_index)
    assert len(b["points"]) == tiny_index["chunks"]
    assert len(b["chapters"]) == tiny_corpus["chapters"] == 3
    assert b["vector_dim"] == tiny_index["dense_dim"]
    assert len(b["vectors"]) == tiny_index["chunks"] * tiny_index["dense_dim"]


def test_every_point_has_provenance_and_3d_position(tmp_path, tiny_corpus, tiny_index):
    b = _bundle(tmp_path, tiny_corpus, tiny_index)
    ids = set()
    for p in b["points"]:
        assert {"id", "ch", "citation", "text", "x", "y", "z"} <= set(p)
        assert p["citation"].startswith("CHAPTER ")
        assert p["text"]
        ids.add(p["id"])
    assert ids == set(range(len(b["points"])))


def test_attributes_match_the_shared_module(tmp_path, tiny_corpus, tiny_index):
    """The web bundle and the Streamlit app must agree on every attribute."""
    import corpus_attributes as ca
    b = _bundle(tmp_path, tiny_corpus, tiny_index)
    records = ca.load_records(tmp_path / "corpus")
    rows = ca.chapter_rows(records)

    assert [c["words"] for c in b["chapters"]] == [r["words"] for r in rows]
    assert [c["dialogue_pct"] for c in b["chapters"]] == [r["dialogue_pct"] for r in rows]
    assert [c["sentiment"] for c in b["chapters"]] == [r["sentiment"] for r in rows]
    assert b["characters"] == ca.character_matrix(records)
    assert b["meta"]["unique_words"] == len(
        {w.lower() for r in records for w in ca.WORD_RE.findall(r["text"])})


def test_short_titles_are_human_readable(tmp_path, tiny_corpus, tiny_index):
    b = _bundle(tmp_path, tiny_corpus, tiny_index)
    assert [c["short"] for c in b["chapters"]] == ["One", "Two", "Three"]


def test_chunk_vector_row_count_guard(tmp_path, tiny_corpus, tiny_index):
    """A stale index must fail loudly instead of emitting a misaligned bundle."""
    import numpy as np
    index_dir = tmp_path / "index"
    np.save(index_dir / "dense.npy", np.zeros((2, tiny_index["dense_dim"]), dtype=np.float32))
    try:
        export_web.build_bundle(tmp_path / "corpus", index_dir)
    except SystemExit as err:
        assert "rebuild the index" in str(err)
    else:
        raise AssertionError("expected SystemExit for a stale index")
