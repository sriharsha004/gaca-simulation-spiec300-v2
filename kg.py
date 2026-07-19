"""
Belief Layer: knowledge graph construction implementing the GACA schema
(6 node types, 6 edge types; see manuscript Table 1 / Section 3.2).

Uses NetworkX (real dependency, matches the paper's stated implementation).
Node IDs are strings; node/edge type is stored as a `kind` attribute so the
same MultiDiGraph can hold all six node types and six edge types.

All six node types are instantiated: professional_profile, interaction_record,
performance_gap, domain_concept, training_resource, and assessment_item. All
six named edge types are instantiated: performed-by, evaluated-against,
prerequisite-of, correlated-with, addressed-by, improved-through.

Note on Assessment Item linkage: the paper's Table 1 declares Assessment Item
as a node type ("evaluations linked to specific domain concepts") but names no
edge type connecting it -- a gap in the original 6-edge schema. This
implementation attaches each Assessment Item to its Domain Concept with an
`assesses` edge, disclosed in the manuscript as an addition that fills that
omission rather than one of the six named edge types.
"""

from __future__ import annotations
import hashlib
import networkx as nx

from ontology import ONTOLOGY, TRAINING_RESOURCES, META_CLASSES
from taxonomy import TAXONOMY, categories_for_domain


def _det_unit(*parts: str) -> float:
    """Deterministic float in [0,1] from string parts (stable across runs and
    machines, unlike hash())."""
    h = hashlib.sha256("::".join(parts).encode()).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


def build_domain_graph(domain: str) -> nx.MultiDiGraph:
    """Builds the seed knowledge graph for one domain: Domain Concept nodes with
    prerequisite-of edges, Training Resource nodes, and correlated-with edges
    linking each gap category's meta-class to the matching concept thread and
    resources. Professional Profile / Interaction Record / Performance Gap
    nodes are added later, per-interaction, by `add_interaction`."""
    g = nx.MultiDiGraph()
    concept_ids: dict[str, list[str]] = {}  # meta_class -> [tier0_id, tier1_id, tier2_id]

    for meta, tiers in ONTOLOGY[domain].items():
        ids = []
        for tier_idx, name in enumerate(tiers):
            cid = f"concept::{domain}::{meta}::{tier_idx}"
            g.add_node(cid, kind="domain_concept", name=name, meta_class=meta,
                       difficulty=tier_idx, domain=domain)
            ids.append(cid)
            # Assessment Item node (6th node type): one evaluation per concept,
            # with difficulty + discrimination parameters (paper Sec. 3.2),
            # attached to its Domain Concept via an `assesses` edge.
            aid = f"assessment::{domain}::{meta}::{tier_idx}"
            g.add_node(aid, kind="assessment_item", name=f"{name} Assessment",
                       meta_class=meta, difficulty=tier_idx,
                       discrimination=round(0.3 + 0.6 * _det_unit(aid), 3),
                       domain=domain)
            g.add_edge(aid, cid, kind="assesses")
        for a, b in zip(ids, ids[1:]):
            g.add_edge(a, b, kind="prerequisite-of")
        concept_ids[meta] = ids

    resource_ids: dict[str, list[str]] = {}
    for meta, resources in TRAINING_RESOURCES[domain].items():
        ids = []
        for i, (name, content_type, effectiveness_prior) in enumerate(resources):
            rid = f"resource::{domain}::{meta}::{i}"
            g.add_node(rid, kind="training_resource", name=name, content_type=content_type,
                       effectiveness_prior=effectiveness_prior, meta_class=meta, domain=domain)
            ids.append(rid)
        resource_ids[meta] = ids

    # correlated-with: domain_concept (advanced tier) <-> training_resource in same meta-class
    for meta in META_CLASSES:
        advanced_concept = concept_ids[meta][-1]
        for rid in resource_ids[meta]:
            g.add_edge(advanced_concept, rid, kind="correlated-with")

    g.graph["domain"] = domain
    g.graph["concept_ids"] = concept_ids
    g.graph["resource_ids"] = resource_ids
    return g


def add_professional_profile(g: nx.MultiDiGraph, profile_id: str, role: str) -> None:
    if g.has_node(profile_id):
        return
    g.add_node(profile_id, kind="professional_profile", role=role, domain=g.graph["domain"])
    # improved-through (6th edge type): seed a small, deterministic synthetic
    # prior-training history so each profile carries at least one recorded
    # successful learning outcome, per the paper's "constructed incrementally
    # as professionals engage" framing. Disclosed as synthetic seeding, not a
    # measured outcome. One improved-through edge to a training resource,
    # chosen deterministically from the profile id.
    all_resource_ids = [rid for ids in g.graph["resource_ids"].values() for rid in ids]
    if all_resource_ids:
        pick = int(_det_unit(profile_id) * len(all_resource_ids)) % len(all_resource_ids)
        g.add_edge(profile_id, all_resource_ids[pick], kind="improved-through",
                   outcome_gain=round(0.1 + 0.4 * _det_unit(profile_id, "gain"), 3),
                   synthetic_prior=True)


def add_interaction(
    g: nx.MultiDiGraph,
    *,
    interaction_id: str,
    profile_id: str,
    role: str,
    timestamp: int,
    transcript_excerpt: str,
    gap_category_id: int,
    severity: str,
    meta_class: str,
) -> str:
    """Adds one Interaction Record + its Performance Gap node and wires the
    performed-by / evaluated-against / correlated-with edges. Returns the
    gap node id. This is what the corpus loader replays into a fresh or
    growing graph for each simulated professional."""
    add_professional_profile(g, profile_id, role)

    g.add_node(interaction_id, kind="interaction_record", profile_id=profile_id,
               timestamp=timestamp, transcript_excerpt=transcript_excerpt[:400],
               domain=g.graph["domain"])
    g.add_edge(interaction_id, profile_id, kind="performed-by")

    gap_id = f"gap::{interaction_id}"
    g.add_node(gap_id, kind="performance_gap", category_id=gap_category_id,
               severity=severity, timestamp=timestamp, domain=g.graph["domain"])
    g.add_edge(profile_id, gap_id, kind="evaluated-against")
    g.add_edge(interaction_id, gap_id, kind="evaluated-against")

    advanced_concept = g.graph["concept_ids"][meta_class][-1]
    g.add_edge(gap_id, advanced_concept, kind="correlated-with")
    for rid in g.graph["resource_ids"][meta_class]:
        g.add_edge(gap_id, rid, kind="correlated-with")
        g.add_edge(gap_id, rid, kind="addressed-by")

    return gap_id


if __name__ == "__main__":
    g = build_domain_graph("healthcare")
    add_interaction(
        g, interaction_id="int::demo::1", profile_id="prof::dr_demo", role="physician",
        timestamp=0, transcript_excerpt="Patient presents with chronic lower back pain...",
        gap_category_id=10, severity="high", meta_class="Communication",
    )
    node_kinds = {d["kind"] for _, d in g.nodes(data=True)}
    edge_kinds = {d["kind"] for _, _, d in g.edges(data=True)}
    print("nodes:", g.number_of_nodes(), "edges:", g.number_of_edges())
    print("node kinds:", sorted(node_kinds))
    print("edge kinds:", sorted(edge_kinds))
    # All 6 paper node types must be instantiated.
    expected_nodes = {"professional_profile", "interaction_record", "performance_gap",
                      "domain_concept", "training_resource", "assessment_item"}
    assert expected_nodes <= node_kinds, f"missing node types: {expected_nodes - node_kinds}"
    # All 6 paper edge types (+ the assessment linkage) must be instantiated.
    expected_edges = {"performed-by", "evaluated-against", "prerequisite-of",
                      "correlated-with", "addressed-by", "improved-through"}
    assert expected_edges <= edge_kinds, f"missing edge types: {expected_edges - edge_kinds}"
    print("OK: all 6 node types and all 6 named edge types instantiated.")
