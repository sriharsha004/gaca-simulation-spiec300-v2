"""
GACA gap taxonomy: 24 categories shared across all three evaluated domains
(teaching, healthcare, customer service). Each category is domain-instantiated
via `domain_examples` strings used by the corpus generator to keep transcripts
concrete rather than generic.

This taxonomy is author-defined methodology (a design artifact), not an
empirical claim -- it is legitimate to state and cite directly in the paper's
corpus-construction section.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class GapCategory:
    id: int
    name: str
    meta_class: str  # Communication | Procedural/Regulatory | Behavioral | Relational
    description: str
    domain_examples: dict = field(default_factory=dict)


TAXONOMY: list[GapCategory] = [
    GapCategory(1, "Instructional/Explanatory Clarity", "Communication",
        "Explanation of a concept, diagnosis, or process is delivered in a way the audience cannot follow.",
        {"teaching": "Introduces a new algebraic operation without checking prior step comprehension.",
         "healthcare": "Explains a diagnosis using unexplained clinical terminology.",
         "customer_service": "Describes a fix using internal jargon the customer does not recognize."}),
    GapCategory(2, "Active Listening Deficiency", "Communication",
        "Professional does not register or respond to what the other party explicitly said.",
        {"teaching": "Ignores a student's stated confusion and continues the lesson plan unchanged.",
         "healthcare": "Does not acknowledge patient's stated symptom timeline before proceeding.",
         "customer_service": "Repeats a scripted response after customer already explained the issue."}),
    GapCategory(3, "Empathy/Emotional Attunement Gap", "Relational",
        "Professional misses or dismisses an emotional cue (anxiety, frustration, fear).",
        {"teaching": "Does not respond to a visibly discouraged student after repeated failure.",
         "healthcare": "Proceeds clinically without acknowledging patient's stated anxiety.",
         "customer_service": "Does not acknowledge customer frustration before troubleshooting."}),
    GapCategory(4, "De-escalation Failure", "Relational",
        "An escalating emotional situation is not defused; tension increases through the interaction.",
        {"teaching": "Confrontation with a disruptive student escalates rather than resolves.",
         "healthcare": "Patient's anxiety increases rather than decreases over the consultation.",
         "customer_service": "Customer anger increases despite (or because of) the agent's response."}),
    GapCategory(5, "Behavioral/Conduct Management Gap", "Behavioral",
        "Disruptive or non-compliant behavior is observed but not effectively addressed.",
        {"teaching": "Off-task student behavior continues after a single ineffective redirection.",
         "healthcare": "N/A - primarily applicable to teaching/group settings.",
         "customer_service": "N/A - primarily applicable to teaching/group settings."}),
    GapCategory(6, "Proactive vs. Reactive Intervention Gap", "Behavioral",
        "Professional only reacts after a problem is already visible, with no preventive step taken earlier.",
        {"teaching": "No proactive engagement strategy used before disruption starts.",
         "healthcare": "No preventive counseling offered despite known risk factors.",
         "customer_service": "No proactive status update offered despite a known recurring service issue."}),
    GapCategory(7, "Prerequisite Skill Deficiency", "Procedural/Regulatory",
        "A foundational skill needed for the current task is observably missing.",
        {"teaching": "Attempts multi-step equations without confirming single-step mastery.",
         "healthcare": "Skips a foundational differential-diagnosis step before treatment discussion.",
         "customer_service": "Attempts advanced troubleshooting without confirming basic account verification."}),
    GapCategory(8, "Time Allocation Imbalance", "Procedural/Regulatory",
        "Disproportionate time spent on one phase of the interaction at the expense of another.",
        {"teaching": "Majority of lesson time on lecture, minimal time on guided practice.",
         "healthcare": "Majority of visit on examination, minimal time on explanation/discussion.",
         "customer_service": "Majority of call on technical steps, minimal time on relationship repair."}),
    GapCategory(9, "Jargon/Terminology Overuse", "Communication",
        "Domain-specific terminology used without translation for a non-expert audience.",
        {"teaching": "Uses undefined mathematical notation without explanation.",
         "healthcare": "Uses medical terminology (e.g., 'lumbar radiculopathy') without plain-language equivalent.",
         "customer_service": "Uses internal ticketing/product jargon the customer would not know."}),
    GapCategory(10, "Health Literacy / Audience Adaptation Gap", "Communication",
        "Communication is not adapted to the demonstrated comprehension level of the other party.",
        {"teaching": "Does not adjust explanation after a student demonstrates misunderstanding.",
         "healthcare": "Does not adjust explanation despite patient's confused follow-up questions.",
         "customer_service": "Does not simplify explanation despite customer's repeated clarifying questions."}),
    GapCategory(11, "Confirmation-of-Understanding Omission", "Procedural/Regulatory",
        "Professional does not verify that their explanation was understood (e.g., no teach-back).",
        {"teaching": "Moves to the next topic without checking for understanding.",
         "healthcare": "Does not use teach-back to confirm patient understood the treatment plan.",
         "customer_service": "Does not confirm customer understood the resolution before ending the call."}),
    GapCategory(12, "Documentation/Protocol Adherence Gap", "Procedural/Regulatory",
        "A required procedural or documentation step in the domain's standard protocol is skipped.",
        {"teaching": "Does not document a recurring behavioral incident per school policy.",
         "healthcare": "Does not document informed-consent discussion for a proposed treatment.",
         "customer_service": "Does not log the service disruption per company escalation policy."}),
    GapCategory(13, "Escalation Timing Error", "Procedural/Regulatory",
        "An escalation (to a specialist, supervisor, or authority) happens too early or too late.",
        {"teaching": "Escalates to administration before attempting available in-classroom strategies.",
         "healthcare": "Delays referral to a specialist despite a red-flag symptom.",
         "customer_service": "Escalates to a supervisor without attempting available service-recovery steps."}),
    GapCategory(14, "Safety/Compliance Deviation", "Procedural/Regulatory",
        "An action deviates from a safety-critical or regulatory requirement of the domain.",
        {"teaching": "Does not follow mandated safety procedure during a lab activity.",
         "healthcare": "Does not screen for a required red-flag symptom before ruling out serious pathology.",
         "customer_service": "Does not follow required identity-verification steps before making account changes."}),
    GapCategory(15, "Follow-Through Failure", "Behavioral",
        "An initial correct action is not sustained or completed.",
        {"teaching": "Starts a redirection strategy but does not follow through on the stated consequence.",
         "healthcare": "Recommends a follow-up test but does not confirm scheduling before the visit ends.",
         "customer_service": "Promises a callback but does not confirm a specific commitment before ending the call."}),
    GapCategory(16, "Personalization Gap", "Relational",
        "Response is generic and does not account for the individual's known history or context.",
        {"teaching": "Uses a one-size-fits-all explanation despite a student's documented learning profile.",
         "healthcare": "Ignores patient's documented history of prior similar visits.",
         "customer_service": "Does not reference the customer's known service history before responding."}),
    GapCategory(17, "Engagement Monitoring Gap", "Behavioral",
        "Observable disengagement (attention, participation) is not detected or addressed.",
        {"teaching": "Continues lecturing despite widespread visible disengagement.",
         "healthcare": "Does not notice patient's withdrawal/disengagement from the conversation.",
         "customer_service": "Does not notice customer's flattening tone indicating disengagement."}),
    GapCategory(18, "Feedback Delivery Gap", "Communication",
        "Feedback given is vague, untimely, or not actionable.",
        {"teaching": "Gives only a grade with no specific actionable feedback.",
         "healthcare": "Gives a vague prognosis with no concrete next steps.",
         "customer_service": "Gives a vague resolution timeline with no concrete commitment."}),
    GapCategory(19, "Conflict Resolution Deficiency", "Relational",
        "A disagreement or complaint is not resolved through an appropriate resolution process.",
        {"teaching": "Peer conflict between students is left unresolved.",
         "healthcare": "Disagreement about treatment plan is not addressed through shared decision-making.",
         "customer_service": "Billing dispute is not resolved through an appropriate service-recovery path."}),
    GapCategory(20, "Assessment/Diagnostic Accuracy Gap", "Procedural/Regulatory",
        "An evaluative judgment (grading, diagnosis, issue diagnosis) shows a specific, identifiable error.",
        {"teaching": "Formative assessment does not accurately capture demonstrated misunderstanding.",
         "healthcare": "Diagnostic reasoning omits a plausible differential given the presenting symptoms.",
         "customer_service": "Initial issue diagnosis misidentifies the root cause stated by the customer."}),
    GapCategory(21, "Resource/Referral Utilization Gap", "Procedural/Regulatory",
        "An available resource, tool, or referral pathway that fits the situation is not used.",
        {"teaching": "Does not use an available intervention resource for a documented recurring issue.",
         "healthcare": "Does not offer an available patient-education resource for a chronic condition.",
         "customer_service": "Does not offer an available loyalty/retention resource for an at-risk customer."}),
    GapCategory(22, "Consistency Across Interactions Gap", "Behavioral",
        "Current interaction is inconsistent with the professional's own prior established pattern or commitment.",
        {"teaching": "Enforces a classroom rule inconsistently with prior sessions.",
         "healthcare": "Advice given contradicts guidance given in a prior visit without acknowledging the change.",
         "customer_service": "Resolution offered contradicts a commitment made in a prior interaction."}),
    GapCategory(23, "Bias/Fairness Concern", "Relational",
        "Observable difference in treatment correlated with a demographic or individual characteristic, not performance.",
        {"teaching": "Calls on/redirects certain students disproportionately without a performance basis.",
         "healthcare": "Explanation depth appears to vary by patient characteristic rather than need.",
         "customer_service": "Tone/thoroughness appears to vary by customer characteristic rather than issue complexity."}),
    GapCategory(24, "Recordkeeping/Handoff Gap", "Procedural/Regulatory",
        "Information needed for continuity (next session, next provider, next agent) is not captured.",
        {"teaching": "No note left for the next lesson on the unresolved behavioral pattern.",
         "healthcare": "No handoff note for a covering physician on the unresolved concern.",
         "customer_service": "No case note left for the next agent handling a reopened ticket."}),
]

assert len(TAXONOMY) == 24, "Taxonomy must contain exactly 24 categories per manuscript claim."

DOMAINS = ("teaching", "healthcare", "customer_service")

# Category 5 ("Behavioral/Conduct Management Gap") is teaching-specific in this
# design; corpus generation excludes it for healthcare/customer_service draws.
DOMAIN_EXCLUSIONS = {
    "healthcare": {5},
    "customer_service": {5},
}


def categories_for_domain(domain: str) -> list[GapCategory]:
    excluded = DOMAIN_EXCLUSIONS.get(domain, set())
    return [c for c in TAXONOMY if c.id not in excluded]


if __name__ == "__main__":
    for d in DOMAINS:
        cats = categories_for_domain(d)
        print(f"{d}: {len(cats)} usable categories")
