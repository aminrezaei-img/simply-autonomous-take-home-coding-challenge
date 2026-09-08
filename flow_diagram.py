"""Self-contained SVG for the extraction-flow diagram on the Questions tab.

Plain string building — no charting library, no JavaScript, no Streamlit import,
so the diagram can be unit-tested on its own. Every box label names a component
that exists in this repository (see README "Why these attributes?").
"""
from __future__ import annotations

# Fixed palette: the SVG is rendered inside an iframe, so it cannot inherit
# Streamlit's CSS variables. These two colours match the default light theme.
INK = "#31333f"
FILL = "#f0f2f6"
W, H = 1120, 620


def extraction_flow_svg() -> str:
    """Return the extraction flow as one inline SVG string."""
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="100%" '
        f'role="img" aria-label="Extraction flow: source text to cited interactive search" '
        f'font-family="Source Sans Pro, system-ui, sans-serif">',
        '<defs><marker id="qah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
        f'markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" '
        f'fill="{INK}" fill-opacity="0.55"/></marker></defs>',
    ]

    def box(cx: float, y: float, bw: float, bh: float, lines: list[str],
            strong: bool = False) -> None:
        parts.append(
            f'<rect x="{cx - bw / 2:.0f}" y="{y:.0f}" width="{bw:.0f}" height="{bh:.0f}" '
            f'rx="7" fill="{FILL}" stroke="{INK}" stroke-opacity="{0.55 if strong else 0.3}" '
            f'stroke-width="{1.6 if strong else 1.2}"/>'
        )
        for i, text in enumerate(lines):
            ty = y + bh / 2 + (i - (len(lines) - 1) / 2) * 17 + 5
            parts.append(
                f'<text x="{cx:.0f}" y="{ty:.0f}" text-anchor="middle" font-size="13" '
                f'font-weight="{600 if strong else 400}" fill="{INK}">{text}</text>'
            )

    def arrow(*points: tuple[float, float]) -> None:
        d = " ".join(f"{'M' if i == 0 else 'L'} {x:.0f} {y:.0f}"
                     for i, (x, y) in enumerate(points))
        parts.append(f'<path d="{d}" fill="none" stroke="{INK}" stroke-opacity="0.45" '
                     f'stroke-width="1.3" marker-end="url(#qah)"/>')

    box(560, 16, 300, 44, ["SOURCE TEXT"], strong=True)
    box(560, 86, 300, 44, ["NORMALIZE + SEGMENT"])
    box(560, 156, 380, 44, ["CHAPTERS / SENTENCES / TOKENS"])
    box(130, 236, 210, 44, ["Structural counts"])
    box(370, 236, 210, 44, ["Character lexicon"])
    box(610, 236, 210, 44, ["VADER"])
    box(860, 236, 210, 44, ["sentence-aware chunks"])
    box(130, 316, 210, 44, ["pacing / dialogue / vocabulary"])
    box(370, 316, 210, 44, ["character presence"])
    box(610, 316, 210, 44, ["sentiment arc"])
    box(810, 316, 170, 44, ["TF-IDF sparse"])
    box(1000, 316, 170, 44, ["dense embeddings"])
    box(810, 396, 170, 44, ["lexical retrieval"])
    box(1000, 396, 170, 44, ["semantic retrieval", "/ PCA"])
    box(905, 476, 360, 44, ["Reciprocal Rank Fusion (RRF)"], strong=True)
    box(905, 556, 360, 44, ["cited interactive search"], strong=True)

    arrow((560, 60), (560, 86))
    arrow((560, 130), (560, 156))
    for cx in (130, 370, 610, 860):
        arrow((560, 200), (560, 216), (cx, 216), (cx, 236))
    for cx in (130, 370, 610):
        arrow((cx, 280), (cx, 316))
    arrow((860, 280), (860, 298), (810, 298), (810, 316))
    arrow((860, 280), (860, 298), (1000, 298), (1000, 316))
    arrow((810, 360), (810, 396))
    arrow((1000, 360), (1000, 396))
    arrow((810, 440), (810, 458), (905, 458), (905, 476))
    arrow((1000, 440), (1000, 458), (905, 458), (905, 476))
    arrow((905, 520), (905, 556))
    parts.append("</svg>")
    return "".join(parts)
