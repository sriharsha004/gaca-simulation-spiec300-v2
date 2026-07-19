"""
Scalability analysis (manuscript Table 6 / Figure 3): measures REAL wall-clock
subgraph-extraction latency vs. graph size by synthetically growing a domain
graph with filler professionals/interactions (no LLM calls needed for the
graph-growth itself -- pure NetworkX + embedding cost). A small sample of
real end-to-end LLM calls (SAMPLE_PER_SIZE) is layered on top at each size
point to get a real P50 end-to-end latency including LLM inference, without
requiring thousands of calls just to characterize scaling.

This script needs an API key only for the small end-to-end LLM sample; the
extraction-latency curve itself is 100% real and runs offline (MOCK_MODE-safe).
"""

from __future__ import annotations
import argparse
import json
import random
import time
from pathlib import Path

from kg import build_domain_graph, add_professional_profile, add_interaction
from retrieval import Embedder, node_text, hybrid_score
from taxonomy import categories_for_domain
from prompts import GACA_SYSTEM, GACA_USER_TEMPLATE, taxonomy_listing_text
from taxonomy import TAXONOMY
from llm_client import chat_json
from run_conditions import relevant_node_ids, format_graph_context, format_rule_constraints

OUT_PATH = Path("results_scalability.jsonl")
GRAPH_SIZES = [500, 2500, 10000, 50000, 100000]
SAMPLE_PER_SIZE = 10
TAXONOMY_TEXT = taxonomy_listing_text(TAXONOMY)


def grow_graph_to_size(domain: str, target_nodes: int, seed: int = 7) -> "nx.MultiDiGraph":
    rng = random.Random(seed)
    g = build_domain_graph(domain)
    cats = categories_for_domain(domain)
    prof_idx = 0
    while g.number_of_nodes() < target_nodes:
        profile_id = f"prof::filler::{prof_idx}"
        add_professional_profile(g, profile_id, role="filler")
        n_interactions = rng.randint(1, 5)
        for t in range(n_interactions):
            if g.number_of_nodes() >= target_nodes:
                break
            cat = rng.choice(cats)
            add_interaction(
                g, interaction_id=f"int::filler::{prof_idx}::{t}", profile_id=profile_id,
                role="filler", timestamp=t,
                transcript_excerpt=f"synthetic filler interaction {prof_idx}-{t} for {cat.name}",
                gap_category_id=cat.id, severity="moderate", meta_class=cat.meta_class,
            )
        prof_idx += 1
    return g


def measure_extraction_latency(g, embedder: Embedder, anchor: str, query: str, alpha: float = 0.6,
                                repeats: int = 5) -> float:
    """Real wall-clock median latency (ms) of hybrid_score (subgraph extraction
    + scoring), NOT including any LLM call."""
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        hybrid_score(g, anchor, query, embedder, alpha=alpha)
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return times[len(times) // 2]


def sample_end_to_end(g, embedder: Embedder, domain: str, n: int, seed: int = 11) -> list[float]:
    """Real end-to-end latency (extraction + LLM call) for a small sample of
    synthetic queries at this graph size."""
    rng = random.Random(seed)
    cats = categories_for_domain(domain)
    latencies = []
    profile_ids = [nid for nid, d in g.nodes(data=True) if d.get("kind") == "professional_profile"]
    for i in range(n):
        anchor = rng.choice(profile_ids)
        cat = rng.choice(cats)
        query = f"possible {cat.name} in a {domain} interaction"
        t0 = time.perf_counter()
        ranked = hybrid_score(g, anchor, query, embedder, alpha=0.6)
        ctx = format_graph_context(g, ranked)
        rule_text = format_rule_constraints(domain, cat.id)
        prompt = GACA_USER_TEMPLATE.format(domain=domain, transcript=f"[scalability probe] {query}",
                                            graph_context=ctx, rule_constraints=rule_text,
                                            taxonomy_listing=TAXONOMY_TEXT)
        chat_json(GACA_SYSTEM, prompt)
        latencies.append((time.perf_counter() - t0) * 1000)
    return latencies


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="healthcare")
    ap.add_argument("--sizes", nargs="+", type=int, default=GRAPH_SIZES)
    ap.add_argument("--sample-per-size", type=int, default=SAMPLE_PER_SIZE)
    ap.add_argument("--out", default=str(OUT_PATH))
    args = ap.parse_args()

    out_path = Path(args.out)
    results = []
    for size in args.sizes:
        print(f"Growing graph to {size} nodes ...")
        g = grow_graph_to_size(args.domain, size)
        actual_size = g.number_of_nodes()

        embedder = Embedder()
        sample_texts = [node_text(g, n) for n in list(g.nodes())[:min(2000, actual_size)]]
        embedder.fit(sample_texts)

        profile_ids = [nid for nid, d in g.nodes(data=True) if d.get("kind") == "professional_profile"]
        anchor = random.choice(profile_ids)
        extraction_ms = measure_extraction_latency(g, embedder, anchor, "sample query text")

        e2e_latencies = sample_end_to_end(g, embedder, args.domain, args.sample_per_size)
        e2e_latencies.sort()
        p50 = e2e_latencies[len(e2e_latencies) // 2] if e2e_latencies else None

        row = {
            "target_size": size, "actual_size": actual_size,
            "extraction_ms": extraction_ms, "p50_e2e_ms": p50,
            "e2e_samples_ms": e2e_latencies,
        }
        results.append(row)
        print(row)
        with out_path.open("a") as f:
            f.write(json.dumps(row) + "\n")

    print(f"Done. Results in {out_path}")


if __name__ == "__main__":
    main()
