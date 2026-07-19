"""
Reasoning Layer: subgraph extraction + hybrid scoring (Eq. 1 in the
manuscript) for GACA/Graph-RAG-only, and flat dense retrieval (FAISS) for
the Flat-RAG baseline.

Embedding backend: TF-IDF (scikit-learn) indexed with FAISS
(IndexFlatIP over L2-normalized vectors = cosine similarity). This is a
REAL, LOCALLY REPRODUCIBLE embedding baseline that needs no API key and no
network access, disclosed as such in the manuscript (Section 5.1.1) rather
than claimed to be a pretrained dense encoder. Swap `Embedder` for an
OpenAI/sentence-transformers backend if reproducing with a stronger encoder.
"""

from __future__ import annotations
import numpy as np
import networkx as nx
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer


class Embedder:
    """Fit once on a corpus of node-text strings; encode() returns L2-normalized
    dense vectors usable directly as FAISS IndexFlatIP inputs (inner product on
    normalized vectors == cosine similarity)."""

    def __init__(self, max_features: int = 512):
        self.vectorizer = TfidfVectorizer(max_features=max_features, stop_words="english")
        self._fitted = False

    def fit(self, texts: list[str]) -> None:
        self.vectorizer.fit(texts if texts else [""])
        self._fitted = True

    def encode(self, texts: list[str]) -> np.ndarray:
        if not self._fitted:
            self.fit(texts)
        mat = self.vectorizer.transform(texts).toarray().astype("float32")
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return mat / norms


def node_text(g: nx.MultiDiGraph, n: str) -> str:
    d = g.nodes[n]
    parts = [d.get("kind", ""), d.get("name", ""), d.get("role", ""),
             d.get("transcript_excerpt", ""), str(d.get("category_id", "")),
             d.get("meta_class", ""), d.get("content_type", "")]
    return " ".join(p for p in parts if p)


def bounded_subgraph(g: nx.MultiDiGraph, anchor: str, hops: int = 2) -> dict[str, int]:
    """BFS over the undirected view, returns {node_id: hop_distance} for nodes
    within `hops` of the anchor (anchor excluded, distance starts at 1)."""
    und = g.to_undirected(as_view=True)
    dist = {anchor: 0}
    frontier = [anchor]
    for d in range(1, hops + 1):
        nxt = []
        for u in frontier:
            for v in und.neighbors(u):
                if v not in dist:
                    dist[v] = d
                    nxt.append(v)
        frontier = nxt
    dist.pop(anchor, None)
    return dist


def proximity(hop_distance: int) -> float:
    """1.0 at hop 1, 0.5 at hop 2 -- matches manuscript Eq. (1) description."""
    return 1.0 / hop_distance


def hybrid_score(
    g: nx.MultiDiGraph,
    anchor: str,
    query_text: str,
    embedder: Embedder,
    alpha: float,
    hops: int = 2,
) -> list[tuple[str, float]]:
    """Returns [(node_id, score)] sorted descending, per Eq. (1):
    score(n) = alpha * proximity(n) + (1 - alpha) * similarity(n)."""
    dist = bounded_subgraph(g, anchor, hops=hops)
    if not dist:
        return []
    node_ids = list(dist.keys())
    texts = [node_text(g, n) for n in node_ids]
    node_vecs = embedder.encode(texts)
    query_vec = embedder.encode([query_text])[0]
    sims = node_vecs @ query_vec  # cosine sim, vectors already normalized

    scored = []
    for i, n in enumerate(node_ids):
        prox = proximity(dist[n])
        sim = float(sims[i])
        score = alpha * prox + (1 - alpha) * sim
        scored.append((n, score))
    scored.sort(key=lambda x: -x[1])
    return scored


def flat_rag_topk(g: nx.MultiDiGraph, query_text: str, embedder: Embedder, k: int = 10) -> list[tuple[str, float]]:
    """Flat-RAG baseline (C2): FAISS flat cosine search over ALL node texts in
    the graph, ignoring graph structure entirely (per manuscript Table 2)."""
    node_ids = list(g.nodes())
    texts = [node_text(g, n) for n in node_ids]
    vecs = embedder.encode(texts)
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)
    qvec = embedder.encode([query_text])
    scores, idxs = index.search(qvec, min(k, len(node_ids)))
    return [(node_ids[i], float(scores[0, j])) for j, i in enumerate(idxs[0]) if i != -1]


if __name__ == "__main__":
    from kg import build_domain_graph, add_interaction

    g = build_domain_graph("healthcare")
    add_interaction(g, interaction_id="int::demo::1", profile_id="prof::dr_demo",
                     role="physician", timestamp=0,
                     transcript_excerpt="Patient presents with chronic lower back pain, anxious.",
                     gap_category_id=10, severity="high", meta_class="Communication")

    emb = Embedder()
    emb.fit([node_text(g, n) for n in g.nodes()])

    ranked = hybrid_score(g, "prof::dr_demo", "health literacy adaptation gap", emb, alpha=0.6)
    print("GACA hybrid top-5:", ranked[:5])

    flat = flat_rag_topk(g, "health literacy adaptation gap", emb, k=5)
    print("Flat-RAG top-5:", flat)
