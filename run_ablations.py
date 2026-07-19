"""
Runs the 7 ablation configurations from manuscript Table 5 against the same
corpus.jsonl, writing results_ablation.jsonl (resumable). Reuses
run_conditions.py's corpus/profile assignment so results are directly
comparable to the "Full GACA" row (= the GACA condition in results_main.jsonl).

Configurations:
  full            -- alias for GACA condition (already in results_main.jsonl;
                     not re-run here, see analyze.py which pulls it in)
  no_temporal     -- fresh graph per call, no accumulated interaction history
  no_prereq       -- prerequisite-of edges stripped from the graph before retrieval
  no_rules        -- GACA prompt but rule_constraints section replaced with "(none)"
  alpha_1_0       -- alias for GR condition (pure graph proximity; not re-run,
                     reuse results_main.jsonl's GR rows)
  alpha_0_0       -- pure semantic similarity (hybrid_score with alpha=0.0)
  no_profile_anchor -- retrieval anchored on a generic domain-concept node
                     instead of the professional's own profile node
"""

from __future__ import annotations
import argparse
import json
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
OUT_PATH = Path("results_ablation.jsonl")
CONFIGS = ["no_temporal", "no_prereq", "no_rules", "alpha_0_0", "no_profile_anchor"]


def strip_prerequisite_edges(g: nx.MultiDiGraph) -> nx.MultiDiGraph:
    g2 = g.copy()
    to_remove = [(u, v, k) for u, v, k, d in g2.edges(keys=True, data=True) if d.get("kind") == "prerequisite-of"]
    g2.remove_edges_from(to_remove)
    return g2


def run_one(config: str, base_graph, embedder: Embedder, rec: dict, fresh_domain_graph_fn) -> dict:
    domain = rec["domain"]
    transcript = interaction_input(rec)      # LLM sees transcript + 14-dim features
    query = rec["transcript"][:500]          # retrieval queries on raw transcript only

    if config == "no_temporal":
        g = fresh_domain_graph_fn(domain)
        add_professional_profile(g, rec["profile_id"], rec["role"])
        anchor = rec["profile_id"]
    elif config == "no_prereq":
        g = strip_prerequisite_edges(base_graph)
        anchor = rec["profile_id"]
    elif config == "no_profile_anchor":
        g = base_graph
        anchor = g.graph["concept_ids"][rec["meta_class"]][0]  # generic foundational concept, not personalized
    else:
        g = base_graph
        anchor = rec["profile_id"]

    alpha = 0.0 if config == "alpha_0_0" else 0.6
    ranked = hybrid_score(g, anchor, query, embedder, alpha=alpha)
    ranked_ids = [n for n, _ in ranked]
    ctx = format_graph_context(g, ranked)

    if config == "no_rules":
        rule_text = "(none)"
    else:
        rule_text = format_rule_constraints(domain, rec["gap_category_id"])

    prompt = GACA_USER_TEMPLATE.format(domain=domain, transcript=transcript, graph_context=ctx,
                                        rule_constraints=rule_text, taxonomy_listing=TAXONOMY_TEXT)
    out = chat_json(GACA_SYSTEM, prompt)

    relevant = relevant_node_ids(g, domain, rec["meta_class"])
    mrr, hit = mrr_recall_at_k(ranked_ids, relevant, k=10)
    recommendation = out.get("recommendation", "")
    compliance = rule_compliance_rate(domain, rec["gap_category_id"], recommendation)

    judge_prompt = JUDGE_USER_TEMPLATE.format(domain=domain, transcript=rec["transcript"],
                                               gap_id=rec["gap_category_id"], gap_name=rec["gap_category_name"],
                                               recommendation=recommendation)
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
    return {(json.loads(l)["interaction_id"], json.loads(l)["config"]) for l in path.open() if l.strip()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=str(CORPUS_PATH))
    ap.add_argument("--out", default=str(OUT_PATH))
    ap.add_argument("--configs", nargs="+", default=CONFIGS)
    args = ap.parse_args()

    records = assign_profiles(load_corpus(Path(args.corpus)))
    records.sort(key=lambda r: (r["domain"], r["profile_id"], r["timestamp"]))
    done = _load_done_keys(Path(args.out))

    domains = {r["domain"] for r in records}
    graphs = {d: build_domain_graph(d) for d in domains}
    embedder = Embedder()
    embedder.fit([r["transcript"][:500] for r in records])

    out_path = Path(args.out)
    with out_path.open("a") as out_f:
        for rec in records:
            g = graphs[rec["domain"]]
            add_professional_profile(g, rec["profile_id"], rec["role"])
            for config in args.configs:
                key = (rec["interaction_id"], config)
                if key in done:
                    continue
                t0 = time.perf_counter()
                result = run_one(config, g, embedder, rec, lambda d: build_domain_graph(d))
                result["wall_clock_ms"] = (time.perf_counter() - t0) * 1000
                out_f.write(json.dumps(result) + "\n")
                out_f.flush()
                print(f"{rec['interaction_id']:>20s} {config:18s} "
                      f"pred={result['pred_gap_category_id']} true={result['true_gap_category_id']}")
            _feats = format_features(rec.get("feature_vector"))
            _excerpt = (f"[features: {_feats}] {rec['transcript']}" if _feats else rec["transcript"])
            add_interaction(g, interaction_id=rec["interaction_id"], profile_id=rec["profile_id"],
                             role=rec["role"], timestamp=rec["timestamp"], transcript_excerpt=_excerpt,
                             gap_category_id=rec["gap_category_id"], severity=rec["severity"],
                             meta_class=rec["meta_class"])

    print(f"Done. Results in {out_path}")


if __name__ == "__main__":
    main()
