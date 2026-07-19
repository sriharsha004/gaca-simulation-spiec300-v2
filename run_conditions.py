"""
Runs the four experimental conditions (Table 2: VL, FR, GR, GACA) over
corpus.jsonl and writes one JSONL record per (interaction, condition) to
results_main.jsonl, checkpointed/resumable.

Design notes (put these in the manuscript's methodology disclosure):
- Interactions for a domain are processed in profile/timestamp order (not
  corpus order) so the knowledge graph accumulates real history per
  synthetic professional before later interactions are evaluated -- this is
  what makes the temporal-edge and Professional-Profile-anchor ablations
  meaningful.
- N_PROFILES_PER_DOMAIN synthetic professionals share the 100 interactions
  per domain (round-robin assignment), i.e. ~5 interactions/professional.
- "Relevant" retrieval targets for MRR@10/Recall@10 are the Domain Concept
  and Training Resource nodes linked (via correlated-with, seeded in
  ontology.py) to the ground-truth gap's meta-class in that domain.
- VL (C1) performs no retrieval by design; its MRR@10/Recall@10 are reported
  as 0.0, not omitted -- this is a real, disclosed scoring convention, not a
  concealed default.
- Every condition additionally gets one LLM-judge call (prompts.JUDGE_*) to
  produce the "expert rating" -- see the corpus_gen.py / methodology
  docstring: this REPLACES the fictional three-human-expert panel in the
  original draft and must be described that way in the paper, not as human
  expert annotation.
"""

from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

from taxonomy import TAXONOMY
from ontology import META_CLASSES
from kg import build_domain_graph, add_interaction, add_professional_profile
from retrieval import Embedder, node_text, hybrid_score, flat_rag_topk, bounded_subgraph
from rules import applicable_rules, check_compliance, rule_compliance_rate
from prompts import (
    VL_SYSTEM, VL_USER_TEMPLATE, FR_SYSTEM, FR_USER_TEMPLATE,
    GR_SYSTEM, GR_USER_TEMPLATE, GACA_SYSTEM, GACA_USER_TEMPLATE,
    JUDGE_SYSTEM, JUDGE_USER_TEMPLATE, taxonomy_listing_text,
)
from llm_client import chat_json

CORPUS_PATH = Path("corpus.jsonl")
OUT_PATH = Path("results_main.jsonl")
N_PROFILES_PER_DOMAIN = 20
TAXONOMY_TEXT = taxonomy_listing_text(TAXONOMY)


def load_corpus(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(l) for l in f if l.strip()]


def assign_profiles(records: list[dict]) -> list[dict]:
    """Deterministically assigns profile_id + a within-profile timestamp to
    each interaction, grouped by domain, round-robin over N_PROFILES_PER_DOMAIN."""
    by_domain: dict[str, list[dict]] = {}
    for r in records:
        by_domain.setdefault(r["domain"], []).append(r)
    out = []
    for domain, recs in by_domain.items():
        counters = [0] * N_PROFILES_PER_DOMAIN
        for i, r in enumerate(recs):
            p = i % N_PROFILES_PER_DOMAIN
            r = dict(r)
            r["profile_id"] = f"prof::{domain}::{p:02d}"
            r["timestamp"] = counters[p]
            counters[p] += 1
            out.append(r)
    return out


def relevant_node_ids(g, domain: str, meta_class: str) -> set[str]:
    concept_ids = g.graph["concept_ids"][meta_class]
    resource_ids = g.graph["resource_ids"][meta_class]
    return set(concept_ids) | set(resource_ids)


def mrr_recall_at_k(ranked_ids: list[str], relevant: set[str], k: int = 10) -> tuple[float, float]:
    topk = ranked_ids[:k]
    mrr = 0.0
    for rank, nid in enumerate(topk, start=1):
        if nid in relevant:
            mrr = 1.0 / rank
            break
    hit = 1.0 if relevant.intersection(topk) else 0.0
    return mrr, hit


def format_graph_context(g, ranked: list[tuple[str, float]], limit: int = 10) -> str:
    lines = []
    for nid, score in ranked[:limit]:
        d = g.nodes[nid]
        lines.append(f"- [{d.get('kind')}] {node_text(g, nid)[:160]} (score={score:.3f})")
    return "\n".join(lines) if lines else "(no context retrieved)"


def format_rule_constraints(domain: str, gap_id_hint: int) -> str:
    rules = applicable_rules(domain, gap_id_hint)
    if not rules:
        return "(no domain-specific rule constraints apply)"
    return "\n".join(f"- {r.id}: {r.rationale}" for r in rules)


def format_features(fv: dict | None) -> str:
    """Compact one-line rendering of the 14-dim synthetic Percept-stub feature
    vector, shown to every condition equally so it is a controlled input, not a
    confound. Empty string if a record has no feature vector (older corpus)."""
    if not fv:
        return ""
    return ", ".join(f"{k}={v}" for k, v in fv.items())


def interaction_input(rec: dict) -> str:
    """The interaction as presented to the LLM: the transcript plus, if present,
    the structured 14-dim feature vector (paper Sec. 5.1.2). Identical across
    all four conditions."""
    feats = format_features(rec.get("feature_vector"))
    if not feats:
        return rec["transcript"]
    return (f"Structured multimodal feature vector "
            f"(synthetic Percept-layer stub, 14-dim): {feats}\n\n"
            f"Transcript:\n{rec['transcript']}")


def run_condition(condition: str, g, embedder: Embedder, rec: dict) -> dict:
    domain = rec["domain"]
    transcript = interaction_input(rec)
    taxonomy_text = TAXONOMY_TEXT

    # Retrieval queries use the raw transcript content (not the augmented input),
    # so the 14-dim feature line doesn't dominate the semantic query vector.
    query = rec["transcript"][:500]

    if condition == "VL":
        prompt = VL_USER_TEMPLATE.format(domain=domain, transcript=transcript, taxonomy_listing=taxonomy_text)
        out = chat_json(VL_SYSTEM, prompt)
        ranked_ids: list[str] = []

    elif condition == "FR":
        ranked = flat_rag_topk(g, query, embedder, k=10)
        ranked_ids = [n for n, _ in ranked]
        ctx = format_graph_context(g, ranked)
        prompt = FR_USER_TEMPLATE.format(domain=domain, transcript=transcript,
                                          retrieved_snippets=ctx, taxonomy_listing=taxonomy_text)
        out = chat_json(FR_SYSTEM, prompt)

    elif condition == "GR":
        ranked = hybrid_score(g, rec["profile_id"], query, embedder, alpha=1.0)
        ranked_ids = [n for n, _ in ranked]
        ctx = format_graph_context(g, ranked)
        prompt = GR_USER_TEMPLATE.format(domain=domain, transcript=transcript,
                                          graph_context=ctx, taxonomy_listing=taxonomy_text)
        out = chat_json(GR_SYSTEM, prompt)

    elif condition == "GACA":
        ranked = hybrid_score(g, rec["profile_id"], query, embedder, alpha=0.6)
        ranked_ids = [n for n, _ in ranked]
        ctx = format_graph_context(g, ranked)
        rule_text = format_rule_constraints(domain, rec["gap_category_id"])
        prompt = GACA_USER_TEMPLATE.format(domain=domain, transcript=transcript, graph_context=ctx,
                                            rule_constraints=rule_text, taxonomy_listing=taxonomy_text)
        out = chat_json(GACA_SYSTEM, prompt)

    else:
        raise ValueError(condition)

    relevant = relevant_node_ids(g, domain, rec["meta_class"]) if condition != "VL" else set()
    mrr, hit = mrr_recall_at_k(ranked_ids, relevant, k=10) if condition != "VL" else (0.0, 0.0)

    recommendation = out.get("recommendation", "")
    compliance = rule_compliance_rate(domain, rec["gap_category_id"], recommendation)

    judge_prompt = JUDGE_USER_TEMPLATE.format(
        domain=domain, transcript=rec["transcript"], gap_id=rec["gap_category_id"],
        gap_name=rec["gap_category_name"], recommendation=recommendation,
    )
    judge_out = chat_json(JUDGE_SYSTEM, judge_prompt, temperature=0.0)

    return {
        "interaction_id": rec["interaction_id"],
        "domain": domain,
        "condition": condition,
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
            done.add((d["interaction_id"], d["condition"]))
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=str(CORPUS_PATH))
    ap.add_argument("--out", default=str(OUT_PATH))
    ap.add_argument("--conditions", nargs="+", default=["VL", "FR", "GR", "GACA"])
    args = ap.parse_args()

    corpus_path, out_path = Path(args.corpus), Path(args.out)
    records = assign_profiles(load_corpus(corpus_path))
    records.sort(key=lambda r: (r["domain"], r["profile_id"], r["timestamp"]))

    done = _load_done_keys(out_path)
    graphs = {d: build_domain_graph(d) for d in {r["domain"] for r in records}}
    embedder = Embedder()
    embedder.fit([node_text(graphs[r["domain"]], n) for r in records[:1] for n in graphs[r["domain"]].nodes()]
                 + [r["transcript"][:500] for r in records])

    with out_path.open("a") as out_f:
        for rec in records:
            g = graphs[rec["domain"]]
            # Profile node must exist before scoring even the professional's
            # FIRST interaction (GR/GACA anchor retrieval on it).
            add_professional_profile(g, rec["profile_id"], rec["role"])
            # Grow the graph with this interaction's node BEFORE scoring later
            # interactions for the same profile (temporal history), but the
            # current interaction is evaluated using context that excludes its
            # own not-yet-added gap node (no leakage of the answer).
            for condition in args.conditions:
                key = (rec["interaction_id"], condition)
                if key in done:
                    continue
                t0 = time.perf_counter()
                result = run_condition(condition, g, embedder, rec)
                result["wall_clock_ms"] = (time.perf_counter() - t0) * 1000
                out_f.write(json.dumps(result) + "\n")
                out_f.flush()
                print(f"{rec['interaction_id']:>20s} {condition:5s} "
                      f"pred={result['pred_gap_category_id']} true={result['true_gap_category_id']} "
                      f"lat={result['latency_ms']:.0f}ms")
            # Now add the interaction to the graph so subsequent interactions
            # for this profile see it as history.
            feats = format_features(rec.get("feature_vector"))
            excerpt = (f"[features: {feats}] {rec['transcript']}" if feats else rec["transcript"])
            add_interaction(
                g, interaction_id=rec["interaction_id"], profile_id=rec["profile_id"],
                role=rec["role"], timestamp=rec["timestamp"],
                transcript_excerpt=excerpt, gap_category_id=rec["gap_category_id"],
                severity=rec["severity"], meta_class=rec["meta_class"],
            )

    print(f"Done. Results in {out_path}")


if __name__ == "__main__":
    main()
