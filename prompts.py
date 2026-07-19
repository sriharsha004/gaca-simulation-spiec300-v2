"""
All prompt templates used anywhere in the pipeline: corpus (transcript)
generation, and the four experimental conditions (VL, FR, GR, GACA).
Editor comment requires these be reported in full in Section 5.1.3 -- this
file IS that disclosure; paste verbatim into the manuscript appendix/table.
"""

CORPUS_GENERATION_SYSTEM = (
    "You are generating a SYNTHETIC training example for an AI research corpus. "
    "You are not describing a real person or a real event. Write a realistic but "
    "fictional interaction transcript for the stated professional domain that "
    "clearly exhibits the specified performance gap. The transcript is the "
    "corpus label ground truth by construction: it must unambiguously contain "
    "evidence of the stated gap category and no other major gap."
)

CORPUS_GENERATION_USER_TEMPLATE = """\
Domain: {domain}
Gap category to embed (ground truth): [{gap_id}] {gap_name} -- {gap_description}
Domain-specific example pattern to follow: {domain_example}
Target severity: {severity}
Target transcript length: approximately {target_words} words.

Write a realistic transcript of a single professional interaction
({role} with {counterpart}) that exhibits this specific gap. Include:
1. A brief scene-setting line (who, context, duration).
2. The transcript itself with speaker labels.
3. Do NOT explicitly name the gap category in the transcript -- show it
   through behavior/dialogue only, the way a real interaction would.

Return ONLY the transcript text, no commentary.
"""

# --- Condition prompts -------------------------------------------------

VL_SYSTEM = (
    "You are an AI assistant asked to evaluate a professional's performance from "
    "a transcript, with no additional context or retrieved knowledge. "
    "Identify the single most significant performance gap and propose one "
    "training recommendation."
)

VL_USER_TEMPLATE = """\
Domain: {domain}
Transcript:
{transcript}

Return a JSON object with exactly these keys:
{{"gap_category_id": <int 1-24 from the taxonomy below>,
  "gap_category_name": <string>,
  "confidence": <float 0-1>,
  "recommendation": <string, 2-4 sentences, a concrete training recommendation>}}

Taxonomy:
{taxonomy_listing}
"""

FR_SYSTEM = (
    "You are an AI assistant evaluating a professional's performance from a "
    "transcript, augmented with the following retrieved reference snippets "
    "(retrieved by flat dense similarity search, unordered by relation, no "
    "structural relationship between them is implied)."
)

FR_USER_TEMPLATE = """\
Domain: {domain}
Transcript:
{transcript}

Retrieved reference snippets (flat, unranked by structure):
{retrieved_snippets}

Return a JSON object with exactly these keys:
{{"gap_category_id": <int 1-24 from the taxonomy below>,
  "gap_category_name": <string>,
  "confidence": <float 0-1>,
  "recommendation": <string, 2-4 sentences, a concrete training recommendation>}}

Taxonomy:
{taxonomy_listing}
"""

GR_SYSTEM = (
    "You are an AI assistant evaluating a professional's performance from a "
    "transcript, augmented with a subgraph extracted from a structured "
    "knowledge graph (Professional Profile, prior Interaction Records, "
    "Performance Gaps, Domain Concepts, Training Resources), ranked by pure "
    "graph proximity to the professional's profile (no semantic similarity "
    "component)."
)

GR_USER_TEMPLATE = """\
Domain: {domain}
Transcript:
{transcript}

Graph-retrieved context (ranked by structural proximity only, alpha=1.0):
{graph_context}

Return a JSON object with exactly these keys:
{{"gap_category_id": <int 1-24 from the taxonomy below>,
  "gap_category_name": <string>,
  "confidence": <float 0-1>,
  "recommendation": <string, 2-4 sentences, a concrete training recommendation>}}

Taxonomy:
{taxonomy_listing}
"""

GACA_SYSTEM = (
    "You are the Reasoning Layer of GACA, a graph-augmented cognitive agent. "
    "You receive a subgraph ranked by hybrid score (structural proximity + "
    "semantic similarity, alpha=0.6) anchored on the professional's profile, "
    "plus applicable rule constraints from the Action Layer that your "
    "recommendation MUST satisfy."
)

GACA_USER_TEMPLATE = """\
Domain: {domain}
Transcript:
{transcript}

Hybrid-scored graph context (alpha=0.6, structural + semantic):
{graph_context}

Applicable Action Layer rule constraints (your recommendation MUST satisfy all of these):
{rule_constraints}

Return a JSON object with exactly these keys:
{{"gap_category_id": <int 1-24 from the taxonomy below>,
  "gap_category_name": <string>,
  "confidence": <float 0-1>,
  "recommendation": <string, 2-4 sentences, a concrete training recommendation
    that explicitly satisfies every listed rule constraint>,
  "graph_path_explanation": <string, cite the specific graph path (node ->
    edge -> node) that led to this recommendation>}}

Taxonomy:
{taxonomy_listing}
"""

# --- LLM-judge rating prompt (replaces the fictional 3-human-expert panel) ---

JUDGE_SYSTEM = (
    "You are an LLM-based evaluator rating a training recommendation's quality "
    "on a 1-5 Likert scale, given the source transcript and the ground-truth "
    "gap category. This is a DISCLOSED LLM-judge protocol, not a human expert "
    "panel -- rate strictly and consistently."
)

JUDGE_USER_TEMPLATE = """\
Domain: {domain}
Transcript:
{transcript}
Ground-truth gap category: [{gap_id}] {gap_name}
Recommendation to rate:
{recommendation}

Rate 1-5 (1=irrelevant/harmful, 3=generic but relevant, 5=specific, actionable,
and directly targets the ground-truth gap) and give a one-sentence justification.
Return JSON: {{"rating": <int 1-5>, "justification": <string>}}
"""


def taxonomy_listing_text(categories) -> str:
    return "\n".join(f"[{c.id}] {c.name}: {c.description}" for c in categories)
