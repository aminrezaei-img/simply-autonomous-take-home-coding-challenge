// Pure data logic for the corpus explorer.
//
// No DOM, no Three.js, no fetch: every function here is importable from Node,
// which is how the keyword search and the in-browser cosine neighbours are
// verified against the real bundle. The Three.js rendering lives in app.js.

/** Lowercased word tokens. */
export function tokenize(text) {
  return text.toLowerCase().match(/[a-z']+/g) || [];
}

/**
 * Keyword search over chunk text: term frequency (log-damped) plus a phrase
 * bonus. Deliberately simple and explainable — the hybrid dense+TF-IDF+RRF
 * retrieval is in search.py; this is the browser-side keyword mode.
 */
export function keywordSearch(points, query, k = 8) {
  const terms = [...new Set(tokenize(query).filter((t) => t.length > 2))];
  if (!terms.length) return [];
  const phrase = query.trim().toLowerCase();
  const scored = [];
  for (const p of points) {
    const text = p.text.toLowerCase();
    let score = 0;
    const matched = [];
    for (const t of terms) {
      const hits = text.split(t).length - 1;
      if (hits > 0) {
        score += 1 + Math.log(hits);
        matched.push(t);
      }
    }
    if (score > 0) {
      if (phrase.length > 8 && text.includes(phrase)) score += 4;
      scored.push({ id: p.id, score: Math.round(score * 1000) / 1000, terms: matched });
    }
  }
  scored.sort((a, b) => b.score - a.score || a.id - b.id);
  return scored.slice(0, k);
}

/**
 * Cosine similarity between one chunk vector and every other, using the same
 * bge-small embeddings the index was built with. Exact, 501x384, ~2 ms.
 */
export function cosineTopK(vectors, dim, id, k = 8) {
  const n = Math.floor(vectors.length / dim);
  const q = vectors.slice(id * dim, (id + 1) * dim);
  const out = [];
  for (let i = 0; i < n; i++) {
    if (i === id) continue;
    let dot = 0;
    let na = 0;
    let nb = 0;
    const off = i * dim;
    for (let d = 0; d < dim; d++) {
      const v = vectors[off + d];
      dot += q[d] * v;
      na += q[d] * q[d];
      nb += v * v;
    }
    out.push({ id: i, score: dot / (Math.sqrt(na * nb) + 1e-12) });
  }
  out.sort((a, b) => b.score - a.score || a.id - b.id);
  return out.slice(0, k).map((r) => ({ ...r, score: Math.round(r.score * 10000) / 10000 }));
}

/** Golden-angle hue per chapter, so neighbouring chapters never share a colour. */
export function chapterHue(i) {
  return (i * 137.508) % 360;
}

/** Locale-stable thousands separator. */
export function formatInt(n) {
  return Number(n).toLocaleString("en-US");
}

/** Min/max of an array, used to scale the sidebar charts. */
export function extent(values) {
  let lo = Infinity;
  let hi = -Infinity;
  for (const v of values) {
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  return [lo === Infinity ? 0 : lo, hi === -Infinity ? 0 : hi];
}
