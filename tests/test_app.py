"""The Streamlit tool renders and shows the canonical dialogue share."""
from __future__ import annotations

import json
from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app.py"


def test_app_renders_and_reports_canonical_dialogue_share(
        tiny_index: dict, tmp_path: Path, monkeypatch):
    monkeypatch.setenv("CORPUS_DIR", str(tmp_path / "corpus"))
    monkeypatch.setenv("CORPUS_INDEX_DIR", str(tmp_path / "index"))
    monkeypatch.setenv("CORPUS_ENCODER", "hashing")   # no model download in tests

    at = AppTest.from_file(str(APP), default_timeout=180).run()
    assert not at.exception, [e.value for e in at.exception]

    meta = json.loads((tmp_path / "corpus" / "corpus_meta.json").read_text(encoding="utf-8"))
    shown = {m.label: m.value for m in at.metric}
    assert shown["Dialogue share"] == f"{meta['dialogue_share_pct']:.1f}%"

    # regression: dialogue_pct must exist on the per-chapter table the Reader uses
    columns = {c for t in at.dataframe for c in t.value.columns}
    assert "dialogue_pct" in columns
