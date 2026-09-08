"""The extraction-flow diagram is generated from code, not drawn by hand.

The SVG is rendered inside an iframe component (Streamlit strips <svg> from
st.html), so these tests check the generator itself: well-formed markup, one
box per pipeline stage, and labels that name real components.
"""
from __future__ import annotations

import re

from flow_diagram import extraction_flow_svg


def test_svg_is_well_formed():
    svg = extraction_flow_svg()
    assert svg.startswith("<svg") and svg.endswith("</svg>")
    assert svg.count("<svg") == 1 and svg.count("</svg>") == 1
    # no external assets, no script: self-contained
    assert "http://www.w3.org/2000/svg" in svg
    assert "<script" not in svg and "<image" not in svg and "href=" not in svg


def test_every_pipeline_stage_has_a_box():
    svg = extraction_flow_svg()
    labels = re.findall(r"<text[^>]*>([^<]+)</text>", svg)
    expected = [
        "SOURCE TEXT",
        "NORMALIZE + SEGMENT",
        "CHAPTERS / SENTENCES / TOKENS",
        "Structural counts",
        "Character lexicon",
        "VADER",
        "sentence-aware chunks",
        "pacing / dialogue / vocabulary",
        "character presence",
        "sentiment arc",
        "TF-IDF sparse",
        "dense embeddings",
        "lexical retrieval",
        "semantic retrieval",
        "/ PCA",
        "Reciprocal Rank Fusion (RRF)",
        "cited interactive search",
    ]
    assert labels == expected
    assert svg.count("<rect") == 16           # one box per stage (PCA label wraps)
    assert svg.count("marker-end") == 16      # one arrow per transition


def test_no_claims_that_the_code_does_not_support():
    svg = extraction_flow_svg()
    lowered = svg.lower()
    assert not re.search(r"\bner\b", lowered)
    for forbidden in ("ai-powered", "gpt", "llm", "neural network"):
        assert forbidden not in lowered
