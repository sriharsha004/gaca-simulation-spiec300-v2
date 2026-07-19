"""
Runs a GACA-only degraded-input sensitivity analysis over corpus.jsonl.

This is intentionally separate from results_main.jsonl/results_ablation.jsonl:
it characterizes plausible deployment degradations requested by reviewers
without changing the main comparative experiment or its checkpoint files.

Configurations:
  noisy_incomplete_transcript -- deterministic transcript truncation plus
                                [inaudible] insertions
  missing_features            -- remove the synthetic 14-dim feature vector
  sparse_history              -- fresh graph per interaction; no accumulated
                                professional interaction history
  incomplete_contested_ontology -- remove prerequisite edges and a deterministic
                                  subset of ontology/resource nodes
"""

from __future__ import annotations
import argparse
import json
import random
import time
from pathlib import Path

import networkx as nx

from kg import build_domain_graph, add_interaction, add_professional_profile
from retrieval import Embedder, node_text, hybrid_score
from rules import rule_compliance_rate
from prompts import GACA_SYSTEM, GACA_USER_TEMPLATE, JUDGE_SYSTEM, JUDGE_USER_TEMPLATE
from llm_client import chat_json
from run_conditions import (
    load_corpus, assign_profiles, relevant_node_ids, mrr_recall_at_k,
    format_graph_context, format_rule_constraints, format_features,
    interaction_input, TAXONOMY_TEXT,
)


CORPUS_PATH = Path("corpus.jsonl")
OUT_PATH = Path("results_sensitivity.jsonl")
CONFIGS = [
    "noisy_incomplete_transcript",
    "missing_features",
    "sparse_history",
    "incomplete_contested_ontology",
]


def degrade_transcript(transcript: str, interaction_id: str) -> str:
    """Deterministically removes context and masks spans without changing labels."""
    rng = random.Random(interaction_id)
    lines = [line for line in transcript.splitlines() if line.strip()]
    if not lines:
        lines = [transcript]

    kept = []
    for i, line in enumerate(lines):
        if i > 0 and (i + 1) % 4 == 0:
            continue
        words = line.split()
        if len(words) > 12 and rng.random() < 0.45:
            start = rng.randint(3, max(3, len(words) - 6))
            stop = min(len(words), start + rng.randint(2, 5))
            words[start:stop] = ["[inaudible]"]
            line = " ".join(words)
        kept.append(line)

    degraded = "\n".join(kept)
    max_chars = max(500, int(len(degraded) * 0.65))
    if len(degraded) > max_chars:
        degraded = degraded[:max_chars].rsplit(" ", 1)[0] + "\n[transcript truncated]"
    return degraded


def incomplete_ontology_graph(g: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """Returns a graph with degraded ontology structure and fewer resources."""
    g2 = g.copy()
    for u, v, k, d in list(g2.edges(keys=True, data=True)):
        if d.get("kind") == "prerequisite-of":
            g2.remove_edge(u, v, k)

    concept_ids = {meta: list(ids) for meta, ids in g2.graph["concept_ids"].items()}
    resource_ids = {meta: list(ids) for meta, ids in g2.graph["resource_ids"].items()}

    for meta, ids in list(concept_ids.items()):
        if len(ids) > 1:
            remove_ids = ids[1::2]
            g2.remove_nodes_from([nid for nid in remove_ids if g2.has_node(nid)])
            concept_ids[meta] = [nid for nid in ids if nid not in remove_ids]
    for meta, ids in list(resource_ids.items()):
        if len(ids) > 1:
            remove_ids = ids[::2]
            g2.remove_nodes_from([nid for nid in remove_ids if g2.has_node(nid)])
            resource_ids[meta] = [nid for nid in ids if nid not in remove_ids]

    g2.graph["concept_ids"] = concept_ids
    g2.graph["resource_ids"] = resource_ids
    return g2


def degraded_record(rec: dict, config: str) -> dict:
    rec2 = dict(rec)
    if config == "noisy_incomplete_transcript":
        rec2["transcript"] = degrade_transcript(rec["transcript"], rec["interaction_id"])
    elif config == "missing_features":
        rec2.pop("feature_vector", None)
    return rec2


def run_one(config: str, base_graph, embedder: Embedder, rec: dict) -> dict:
    domain = rec["domain"]
    rec_for_prompt = degraded_record(rec, config)

    if config == "sparse_history":
        g = build_domain_graph(domain)
        add_professional_profile(g, rec["profile_id"], rec["role"])
    elif config == "incomplete_contested_ontology":
        g = incomplete_ontology_graph(base_graph)
    else:
        g = base_graph

    transcript = interaction_input(rec_for_prompt)
    query = rec_for_prompt["transcript"][:500]
    ranked = hybrid_score(g, rec["profile_id"], query, embedder, alpha=0.6)
    ranked_ids = [n for n, _ in ranked]
    ctx = format_graph_context(g, ranked)
    rule_text = format_rule_constraints(domain, rec["gap_category_id"])

    prompt = GACA_USER_TEMPLATE.format(
        domain=domain,
        transcript=transcript,
        graph_context=ctx,
        rule_constraints=rule_text,
        taxonomy_listing=TAXONOMY_TEXT,
    )
    out = chat_json(GACA_SYSTEM, prompt)

    relevant = relevant_node_ids(g, domain, rec["meta_class"])
    mrr, hit = mrr_recall_at_k(ranked_ids, relevant, k=10)
    recommendation = out.get("recommendation", "")
    compliance = rule_compliance_rate(domain, rec["gap_category_id"], recommendation)

    judge_prompt = JUDGE_USER_TEMPLATE.format(
        domain=domain,
        transcript=rec_for_prompt["transcript"],
        gap_id=rec["gap_category_id"],
        gap_name=rec["gap_category_name"],
        recommendation=recommendation,
    )
    judge_out = chat_json(JUDGE_SYSTEM, judge_prompt, temperature=0.0)

    return {
        "interaction_id": rec["interaction_id"],
        "domain": domain,
        "config": config,
        "true_gap_category_id": rec["gap_category_id"],
        "pred_gap_category_id": out.get("gap_category_id"),
        "recommendation": recommendation,
        "rule_compliance_rate": compliance,
        "mrr_at_10": mrr,
        "recall_at_10": hit,
        "latency_ms": out.get("_latency_ms"),
        "judge_rating": judge_out.get("rating"),
        "mock": out.get("_mock", False),
    }


def _load_done_keys(path: Path) -> set[tuple[str, str]]:
    if not path.exists():
        return set()
    done = set()
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            done.add((d["interaction_id"], d["config"]))
    return done


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=str(CORPUS_PATH))
    ap.add_argument("--out", default=str(OUT_PATH))
    ap.add_argument("--configs", nargs="+", default=CONFIGS)
    ap.add_argument("--limit", type=int, default=None, help="Optional record limit for smoke tests.")
    args = ap.parse_args()

    records = assign_profiles(load_corpus(Path(args.corpus)))
    records.sort(key=lambda r: (r["domain"], r["profile_id"], r["timestamp"]))
    if args.limit is not None:
        records = records[:args.limit]

    out_path = Path(args.out)
    done = _load_done_keys(out_path)
    domains = {r["domain"] for r in records}
    graphs = {d: build_domain_graph(d) for d in domains}
    embedder = Embedder()
    embedder.fit([r["transcript"][:500] for r in records])

    with out_path.open("a") as out_f:
        for rec in records:
            g = graphs[rec["domain"]]
            add_professional_profile(g, rec["profile_id"], rec["role"])
            for config in args.configs:
                key = (rec["interaction_id"], config)
                if key in done:
                    continue
                t0 = time.perf_counter()
                result = run_one(config, g, embedder, rec)
                result["wall_clock_ms"] = (time.perf_counter() - t0) * 1000
                out_f.write(json.dumps(result) + "\n")
                out_f.flush()
                done.add(key)
                print(f"{rec['interaction_id']:>20s} {config:30s} "
                      f"pred={result['pred_gap_category_id']} true={result['true_gap_category_id']} "
                      f"lat={result['latency_ms']:.0f}ms", flush=True)

            feats = format_features(rec.get("feature_vector"))
            excerpt = (f"[features: {feats}] {rec['transcript']}" if feats else rec["transcript"])
            add_interaction(
                g,
                interaction_id=rec["interaction_id"],
                profile_id=rec["profile_id"],
                role=rec["role"],
                timestamp=rec["timestamp"],
                transcript_excerpt=excerpt,
                gap_category_id=rec["gap_category_id"],
                severity=rec["severity"],
                meta_class=rec["meta_class"],
            )

    print(f"Done. Results in {out_path}")


if __name__ == "__main__":
    main()
