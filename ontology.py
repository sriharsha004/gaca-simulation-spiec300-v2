"""
Author-curated domain ontologies for the three evaluated domains.

IMPORTANT (for the manuscript): these are original concept lists *inspired by*
publicly known competency frameworks (Common Core math practices, general
internal-medicine competency domains, standard contact-center QA dimensions)
but are NOT verbatim reproductions of those licensed/copyrighted frameworks,
and are much smaller than the inflated node counts (124/211/97) claimed in
the original draft. Report the real counts below in the revised paper, not
the old ones.

Each meta-class thread has 3 tiers (foundational -> intermediate -> advanced)
with prerequisite-of edges chaining tier N -> tier N+1 within the same thread.
"""

META_CLASSES = ("Communication", "Procedural/Regulatory", "Behavioral", "Relational")

# domain -> meta_class -> [foundational, intermediate, advanced] concept names
ONTOLOGY: dict[str, dict[str, list[str]]] = {
    "teaching": {
        "Communication": [
            "Clear Verbal Instruction",
            "Checking for Understanding",
            "Differentiated Explanation Strategies",
        ],
        "Procedural/Regulatory": [
            "Lesson Structure Basics",
            "Formative Assessment Design",
            "IEP/504 Accommodation Compliance",
        ],
        "Behavioral": [
            "Attention Cueing",
            "Proactive Classroom Management",
            "Restorative Practices",
        ],
        "Relational": [
            "Rapport Building",
            "Growth-Mindset Feedback",
            "Culturally Responsive Teaching",
        ],
    },
    "healthcare": {
        "Communication": [
            "Plain-Language Explanation",
            "Teach-Back Method",
            "Shared Decision-Making",
        ],
        "Procedural/Regulatory": [
            "Informed Consent Basics",
            "Red-Flag Symptom Screening",
            "Referral and Escalation Protocol",
        ],
        "Behavioral": [
            "Active Listening in Consultation",
            "Preventive Counseling",
            "Chronic Disease Follow-Up Planning",
        ],
        "Relational": [
            "Empathetic Presence",
            "Health Literacy Adaptation",
            "Anxiety-Aware Communication",
        ],
    },
    "customer_service": {
        "Communication": [
            "Plain-Language Troubleshooting",
            "Active Listening on Calls",
            "Expectation Setting",
        ],
        "Procedural/Regulatory": [
            "Identity Verification Protocol",
            "Case Documentation Standards",
            "Escalation Path Policy",
        ],
        "Behavioral": [
            "Call Structure Basics",
            "Proactive Status Updates",
            "Service Recovery Techniques",
        ],
        "Relational": [
            "Acknowledgment and Apology",
            "De-escalation Technique",
            "Personalized Retention Conversation",
        ],
    },
}

# 2 training resources per meta-class per domain (content_type, effectiveness_prior).
# effectiveness_prior is a DESIGN PARAMETER (an initialization value for the
# adaptive-learning module), not a measured outcome -- disclose as such.
TRAINING_RESOURCES: dict[str, dict[str, list[tuple]]] = {
    domain: {
        meta: [
            (f"{meta} Foundations Module ({domain})", "video", 0.65),
            (f"{meta} Applied Simulation ({domain})", "simulation", 0.75),
        ]
        for meta in META_CLASSES
    }
    for domain in ("teaching", "healthcare", "customer_service")
}


def concept_count(domain: str) -> int:
    return sum(len(v) for v in ONTOLOGY[domain].values())


def resource_count(domain: str) -> int:
    return sum(len(v) for v in TRAINING_RESOURCES[domain].values())


if __name__ == "__main__":
    for d in ONTOLOGY:
        print(f"{d}: {concept_count(d)} domain concepts, {resource_count(d)} training resources")
