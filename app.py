"""Interactive engagement tool for the corpus.

Run:
    streamlit run app.py

Tabs:
    Search            hybrid / dense / sparse retrieval with chapter-page citations
    Corpus analytics  structure, dialogue, character presence, sentiment
    Embedding space   2-D PCA projection of the chunk vectors
    Reader            chapter-by-chapter reading with provenance
    Questions & Methods  questions → measurable signals → extraction methods

Paths can be overridden: CORPUS_DIR, CORPUS_INDEX_DIR.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from flow_diagram import extraction_flow_svg
from search import BookSearch, HashingEncoder

ROOT = Path(__file__).resolve().parent
CORPUS_DIR = Path(os.environ.get("CORPUS_DIR", ROOT / "corpus"))
INDEX_DIR = Path(os.environ.get("CORPUS_INDEX_DIR", ROOT / "index"))

# Fixed presence lexicon. Whole-word mention counts, NOT named-entity
# recognition and NOT automatic character discovery: the list is hand-picked
# and the app says so. See README "character presence".
CHARACTER_PRESENCE = [
    "Harry", "Ron", "Hermione", "Dumbledore", "Hagrid", "Snape", "Voldemort",
    "Draco", "McGonagall", "Dudley", "Vernon", "Petunia", "Dobby", "Sirius",
    "Quirrell", "Neville", "Fred", "George", "Percy", "Wood",
    "Flitwick", "Trelawney", "Filch", "Norbert", "Firenze", "Griphook",
]

_SIA = SentimentIntensityAnalyzer()
WORD_RE = re.compile(r"[A-Za-z']+")


def chapter_sentiment(text: str) -> float:
    """Mean VADER compound over a spread sample of ~400 sentences."""
    sents = re.split(r"(?<=[.!?])\s+", text)
    if not sents:
        return 0.0
    step = max(1, len(sents) // 400)
    sample = sents[::step]
    return sum(_SIA.polarity_scores(s)["compound"] for s in sample) / len(sample)


@st.cache_data(show_spinner=False)
def load_corpus(corpus_dir: str) -> tuple[pd.DataFrame, dict]:
    corpus_dir = Path(corpus_dir)
    rows = [json.loads(l) for l in
            (corpus_dir / "corpus.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    meta = json.loads((corpus_dir / "corpus_meta.json").read_text(encoding="utf-8"))
    out = []
    for c in rows:
        text = c["text"]
        out.append({
            "index": c["chapter_index"],
            "title": c["title"],
            "pages": f"{c['pages'][0]}-{c['pages'][1]}" if c["pages"] else "n/a",
            "words": len(text.split()),
            "chars": len(text),
            "paragraphs": len([p for p in text.split("\n\n") if p.strip()]),
            "sentences": len(re.findall(r"[.!?](?:\s|$)", text)),
            "dialogue_words": c.get("dialogue_words",
                                    sum(len(m.split()) for m in re.findall(r'"[^"]+"', text))),
            "unique_words": len({w.lower() for w in WORD_RE.findall(text)}),
            "sentiment": round(chapter_sentiment(text), 4),
            "text": text,
        })
    df = pd.DataFrame(out)
    df["dialogue_pct"] = (100 * df["dialogue_words"] / df["words"].clip(lower=1)).round(2)
    return df, meta


@st.cache_data(show_spinner=False)
def load_chunks(index_dir: str) -> pd.DataFrame:
    rows = [json.loads(l) for l in
            (Path(index_dir) / "chunks.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def load_dense(index_dir: str) -> np.ndarray:
    return np.load(Path(index_dir) / "dense.npy")


@st.cache_resource(show_spinner="Loading embedding model...")
def get_search(index_dir: str) -> BookSearch:
    # CORPUS_ENCODER=hashing forces the offline fallback encoder (no model
    # download); the test suite uses it so tests stay fast and network-free.
    encoder = HashingEncoder() if os.environ.get("CORPUS_ENCODER") == "hashing" else None
    return BookSearch(index_dir, encoder=encoder)


def pca_map(dense: np.ndarray, chunks: pd.DataFrame) -> pd.DataFrame:
    from sklearn.decomposition import PCA
    xy = PCA(n_components=2, random_state=42).fit_transform(dense)
    return pd.DataFrame({
        "x": xy[:, 0], "y": xy[:, 1],
        "chapter": chunks["chapter"].str.replace(r"^(?:CHAPTER|Chapter)\s+\w+\s*[-–]\s*",
                                                 "", regex=True),
        "pages": chunks["pages"].astype(str),
        "preview": chunks["text"].str[:90],
    })


st.set_page_config(page_title="Harry Potter Corpus Explorer", layout="wide")
st.title("Harry Potter Corpus Explorer")

if not (CORPUS_DIR / "corpus.jsonl").exists() or not (INDEX_DIR / "manifest.json").exists():
    st.error("No corpus or index found. The book text is copyrighted, so it is not committed.")
    st.code(
        "# supply the book locally, then build the corpus and index\n"
        "python ingest.py data/raw/book.txt corpus\n"
        "python build_index.py corpus index\n"
        "streamlit run app.py",
        language="bash",
    )
    st.stop()

chapters, meta = load_corpus(str(CORPUS_DIR))
chunks = load_chunks(str(INDEX_DIR))
dense = load_dense(str(INDEX_DIR))
man = json.loads((INDEX_DIR / "manifest.json").read_text(encoding="utf-8"))

pages_lbl = f"{meta['pages']} pages · " if meta.get("pages") else ""
st.caption(
    f"{pages_lbl}{meta['chapters']} chapters · {meta['words_total']:,} words · "
    f"{man['chunks']} chunks · {man['dense_model']} ({man['dense_dim']}d, {man['dense_backend']}) · "
    "hybrid dense + TF-IDF retrieval"
)

tab_search, tab_stats, tab_map, tab_read, tab_questions = st.tabs(
    ["🔍 Search", "📊 Corpus analytics", "🗺️ Embedding space", "📖 Reader",
     "🧭 Questions & Methods"]
)

# ---------------------------------------------------------------- Search
with tab_search:
    c1, c2, c3 = st.columns([3, 1, 1])
    query = c1.text_input("Ask the corpus", "how does the sorting hat decide which house",
                          label_visibility="collapsed")
    mode = c2.selectbox("Retrieval", ["hybrid", "dense", "sparse"], label_visibility="collapsed")
    k = c3.slider("Results", 3, 20, 6, label_visibility="collapsed")

    f1, f2 = st.columns(2)
    chap_filter = f1.selectbox("Filter by chapter", ["all"] + list(chapters["title"]))
    has_pages = bool(meta.get("pages"))
    page_filter = f2.number_input("Filter by page (0 = off)", 0, int(meta.get("pages") or 0), 0,
                                  disabled=not has_pages,
                                  help=None if has_pages else "Page filters need a PDF source")

    if query.strip():
        chapter_id = None
        if chap_filter != "all":
            chapter_id = int(chapters.loc[chapters["title"] == chap_filter, "index"].iloc[0])
        hits = get_search(str(INDEX_DIR)).search(query, k=k, mode=mode,
                                                 chapter=chapter_id, page=int(page_filter) or None)
        if not hits:
            st.warning("No chunks match those filters.")
        for h in hits:
            with st.container(border=True):
                top = st.columns([4, 1, 1, 1])
                top[0].markdown(f"**{h['citation']}**")
                top[1].metric("RRF", f"{h['score']:.4f}")
                top[2].metric("dense #", h["dense_rank"] or "–")
                top[3].metric("tf-idf #", h["sparse_rank"] or "–")
                st.write(get_search(str(INDEX_DIR)).highlight(h["text"], query, width=600))
                with st.expander("full chunk"):
                    st.write(h["text"])

# ---------------------------------------------------------------- Analytics
with tab_stats:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Chapters", meta["chapters"])
    m2.metric("Words", f"{meta['words_total']:,}")
    m3.metric("Unique words",
              f"{len({w.lower() for t in chapters['text'] for w in WORD_RE.findall(t)}):,}")
    # canonical value computed once in ingest.py from the same corpus records
    m4.metric("Dialogue share", f"{meta['dialogue_share_pct']:.1f}%")

    st.subheader("Chapter length and pacing")
    df = chapters.copy()
    df["chapter"] = df["title"].str.replace(r"^(?:CHAPTER|Chapter)\s+\w+\s*[-–]\s*", "",
                                            regex=True)
    df["avg_sentence"] = df["words"] / df["sentences"].clip(lower=1)
    order = list(df["chapter"])
    bars = alt.Chart(df).mark_bar().encode(
        x=alt.X("chapter:N", sort=order, title=None),
        y=alt.Y("words:Q", title="words"),
        color=alt.Color("dialogue_pct:Q", scale=alt.Scale(scheme="viridis"),
                        title="dialogue %"),
        tooltip=["chapter", "words", "dialogue_pct", "pages"],
    ).properties(height=380)
    st.altair_chart(bars, width="stretch")
    st.caption("Bar height = chapter length; colour = share of words inside double quotes.")

    st.subheader("Emotional arc")
    line = alt.Chart(df).mark_line(point=True).encode(
        x=alt.X("chapter:N", sort=order, title=None),
        y=alt.Y("sentiment:Q", title="VADER compound (mean per chapter)"),
        tooltip=["chapter", "sentiment"],
    )
    zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(strokeDash=[4, 4]).encode(y="y")
    st.altair_chart((line + zero).properties(height=320), width="stretch")
    st.caption("Sentence-level VADER sentiment, sampled ~400 sentences per chapter.")

    st.subheader("Character presence")
    heat = []
    for _, r in chapters.iterrows():
        counts = {c: len(re.findall(rf"\b{c}\b", r["text"])) for c in CHARACTER_PRESENCE}
        heat.append({"chapter": r["title"].replace("CHAPTER ", "ch"), **counts})
    hdf = pd.DataFrame(heat).set_index("chapter")
    keep = hdf.sum().sort_values(ascending=False).head(14).index
    long = hdf[keep].T.reset_index().melt(id_vars="index", var_name="chapter",
                                          value_name="mentions")
    long = long.rename(columns={"index": "character"})
    hm = alt.Chart(long).mark_rect().encode(
        x=alt.X("chapter:N", sort=[c.replace("CHAPTER ", "ch") for c in hdf.index], title=None),
        y=alt.Y("character:N", title=None),
        color=alt.Color("mentions:Q", scale=alt.Scale(scheme="magma")),
        tooltip=["character", "chapter", "mentions"],
    ).properties(height=460)
    st.altair_chart(hm, width="stretch")
    st.caption(
        "Whole-word counts for a fixed 26-name presence lexicon. This is a "
        "presence signal, not NER and not automatic character discovery — "
        "names not on the list are invisible, and no pronouns or aliases are resolved."
    )

    st.subheader("Per-chapter table")
    st.dataframe(
        df[["index", "chapter", "pages", "words", "paragraphs", "sentences",
            "avg_sentence", "dialogue_pct", "unique_words", "sentiment"]].round(2),
        width="stretch", hide_index=True)

# ---------------------------------------------------------------- Map
with tab_map:
    st.markdown("2-D PCA projection of the chunk embeddings. Chunks that are "
                "semantically close sit close together — chapters should form their "
                "own neighbourhoods if the vector store is working.")
    m = pca_map(dense, chunks)
    scatter = alt.Chart(m).mark_circle(size=40, opacity=0.75).encode(
        x=alt.X("x:Q", title=None), y=alt.Y("y:Q", title=None),
        color=alt.Color("chapter:N", title="chapter"),
        tooltip=["chapter", "pages", "preview"],
    ).properties(height=620)
    st.altair_chart(scatter, width="stretch")
    st.caption(f"{len(m)} chunk vectors · {man['dense_dim']} dimensions → 2 via PCA")

# ---------------------------------------------------------------- Reader
with tab_read:
    pick = st.selectbox("Chapter", chapters["title"], label_visibility="collapsed")
    row = chapters.loc[chapters["title"] == pick].iloc[0]
    st.markdown(f"### {row['title']}")
    st.caption(f"pages {row['pages']} · {row['words']:,} words · "
               f"{row['paragraphs']} paragraphs · {row['dialogue_pct']:.1f}% dialogue")
    st.write(row["text"])


# ------------------------------------------------------- Questions & Methods
# Static reasoning panel: no computation, no new dependencies. It explains the
# attributes the other tabs already compute, using the same terms as the code.
with tab_questions:
    st.markdown("### Questions → signals → methods")
    st.markdown(
        "I chose attributes by starting from questions a reader might actually ask "
        "about a novel, then selecting signals that are measurable from the corpus, "
        "traceable to the source, and complementary rather than redundant."
    )

    principles = [
        ("Reader-relevant", "The metric should answer a question about the story or writing."),
        ("Extractable", "It must be computable from the corpus rather than inferred without evidence."),
        ("Traceable", "Results should retain chapter/page provenance where possible."),
        ("Complementary", "Structural, lexical, affective and semantic signals should reveal "
                          "different aspects of the same corpus."),
    ]
    for col, (name, text) in zip(st.columns(4), principles):
        with col:
            st.markdown(f"**{name}**")
            st.caption(text)

    cards = [
        {
            "question": "How does the pacing change through the book?",
            "why": "Chapter length, sentence length and dialogue share are simple but "
                   "interpretable signals of narrative tempo and writing density.",
            "attributes": [
                "chapter length",
                "average sentence length",
                "dialogue share (words inside quotation spans)",
                "unique vocabulary per chapter (lexical diversity)",
            ],
            "method": [
                "chapter segmentation",
                "token and sentence counts",
                "quotation-span counting",
                "vocabulary statistics",
            ],
            "inspect": "the Corpus analytics tab — chapter-length bars coloured by dialogue "
                       "share, and the per-chapter table (words, sentences, average sentence "
                       "length, dialogue %, unique words).",
            "limitation": "These describe form, not why a chapter feels fast or slow.",
        },
        {
            "question": "Which characters dominate different parts of the story?",
            "why": "Chapter-level presence makes changes in narrative focus visible across the book.",
            "attributes": ["character presence by chapter"],
            "method": [
                "curated character-presence lexicon",
                "exact word-boundary matching (not NER, not automatic character discovery)",
            ],
            "inspect": "the Corpus analytics tab — the character-presence heatmap (14 "
                       "most-mentioned names × chapters) and the per-chapter table.",
            "limitation": "Aliases, pronouns and ambiguous surnames are not fully resolved; "
                          "names outside the fixed list are invisible.",
        },
        {
            "question": "How does emotional tone change across the narrative?",
            "why": "It gives a compact view of shifts between more positive and negative "
                   "language over narrative time.",
            "attributes": ["chapter-level sentiment trajectory"],
            "method": [
                "sentence-level VADER compound scores",
                "aggregated by chapter (sampled sentences per chapter)",
            ],
            "inspect": "the Corpus analytics tab — the emotional arc line chart and the "
                       "sentiment column of the per-chapter table.",
            "limitation": "A lexicon-based proxy, not an emotion model and not ground truth.",
        },
        {
            "question": "How is meaning distributed through the book, and how can a reader retrieve it?",
            "why": "Dense and lexical representations expose different notions of similarity: "
                   "semantic paraphrase versus exact or rare terminology.",
            "attributes": [
                "sentence-aware chunks",
                "dense embeddings",
                "TF-IDF sparse representation",
                "Reciprocal Rank Fusion",
                "PCA projection of embeddings",
            ],
            "method": [
                "sentence-aware chunking with chapter/page provenance",
                "dense embeddings (fastembed ONNX, 384-d) and TF-IDF (1–2 grams)",
                "Reciprocal Rank Fusion of the two ranked lists",
                "PCA of the embedding matrix for the map",
            ],
            "inspect": "Search (three modes, chapter/page citations, per-hit dense and tf-idf "
                       "ranks), the Embedding space tab, and evaluate.py's measured Hit@1 / MRR.",
            "limitation": "PCA is a 2-D projection and should not be interpreted as a faithful "
                          "map of all high-dimensional relationships.",
        },
    ]

    st.divider()
    for pair in (cards[0:2], cards[2:4]):
        cols = st.columns(2)
        for col, card in zip(cols, pair):
            with col, st.container(border=True):
                st.markdown(f"**{card['question']}**")
                st.markdown(f"Why it is interesting. {card['why']}")
                st.markdown("**Measurable attribute**")
                st.markdown("\n".join(f"- {a}" for a in card["attributes"]))
                st.markdown("**Extraction method**")
                st.markdown("\n".join(f"- {m}" for m in card["method"]))
                st.markdown(f"**What you can inspect.** {card['inspect']}")
                st.caption(f"Limitation — {card['limitation']}")

    st.divider()
    st.markdown("**Extraction flow**")
    # components.html (shipped with Streamlit) because st.html sanitises <svg> away
    components.html(
        "<body style='margin:0;background:transparent'>" + extraction_flow_svg() + "</body>",
        height=640, scrolling=False,
    )
    st.caption("Every box names a component of this repository; the same terms appear in the README.")
