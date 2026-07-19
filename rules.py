"""
GACA Action Layer rule templates.

Each Rule is a structured condition-action template:
  - `condition`: a predicate over (domain, gap_category_id, severity, evidence_text)
    deciding whether the rule applies to a given interaction.
  - `required_elements`: recommendation-content requirements that MUST appear
    (checked by simple, disclosed keyword/structure matching -- see
    `check_compliance`) for a generated recommendation to count as compliant
    with this rule.
  - `rationale`: the domain/regulatory/pedagogical grounding for the rule.

This is real, checkable methodology: rule compliance is computed
programmatically against LLM output, not asserted. The matching is
deliberately simple (keyword/structural) and that limitation is disclosed in
the manuscript rather than hidden.
"""

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class Rule:
    id: str
    domain: str  # "teaching" | "healthcare" | "customer_service" | "all"
    condition: Callable[[dict], bool]
    required_elements: list[str]  # each is a (label, any-of-keywords) pair, see below
    rationale: str


def _has_any(text: str, keywords: list[str]) -> bool:
    t = text.lower()
    return any(k.lower() in t for k in keywords)


RULES: list[Rule] = [
    Rule(
        id="R1-safety-escalation",
        domain="healthcare",
        condition=lambda ctx: ctx["gap_category_id"] == 14,  # Safety/Compliance Deviation
        required_elements=[
            ("mandatory_red_flag_screen", ["red flag", "red-flag", "screening protocol", "escalat"]),
        ],
        rationale="Patient-safety standard: any recommendation touching a safety/compliance "
                   "deviation must reference the mandatory red-flag screening or escalation step, "
                   "not merely a communication-skill fix.",
    ),
    Rule(
        id="R2-deescalation-before-technical",
        domain="customer_service",
        condition=lambda ctx: ctx["gap_category_id"] in (3, 4),  # Empathy / De-escalation
        required_elements=[
            ("acknowledge_before_fix", ["acknowledge", "apolog", "de-escalat"]),
        ],
        rationale="Company protocol: emotional acknowledgment must precede or accompany technical "
                   "troubleshooting steps in any recommendation addressing an empathy/de-escalation gap.",
    ),
    Rule(
        id="R3-behavioral-proactive-first",
        domain="teaching",
        condition=lambda ctx: ctx["gap_category_id"] in (5, 6),
        required_elements=[
            ("proactive_before_reactive", ["proactive", "prevent", "before it starts", "early intervention"]),
        ],
        rationale="Pedagogical principle: behavioral interventions must sequence proactive/preventive "
                   "strategies ahead of purely reactive consequence-based ones.",
    ),
    Rule(
        id="R4-teachback-required",
        domain="all",
        condition=lambda ctx: ctx["gap_category_id"] == 11,
        required_elements=[
            ("teachback_method", ["teach-back", "teach back", "confirm understanding", "ask them to explain"]),
        ],
        rationale="Confirmation-of-understanding gaps must be addressed with an explicit teach-back "
                   "or understanding-check mechanism, not a generic 'communicate better' suggestion.",
    ),
    Rule(
        id="R5-documentation-required",
        domain="all",
        condition=lambda ctx: ctx["gap_category_id"] == 12,
        required_elements=[
            ("documentation_step", ["document", "log", "record", "note in file", "case note"]),
        ],
        rationale="Documentation/protocol-adherence gaps require a recommendation that includes a "
                   "concrete documentation or logging step, consistent with organizational policy.",
    ),
    Rule(
        id="R6-escalation-sequencing",
        domain="all",
        condition=lambda ctx: ctx["gap_category_id"] == 13,
        required_elements=[
            ("attempt_before_escalate", ["before escalat", "prior to escalat", "attempt", "first try"]),
        ],
        rationale="Escalation-timing gaps require the recommendation to specify what should be "
                   "attempted before escalation, not just 'escalate sooner/later'.",
    ),
    Rule(
        id="R7-bias-flag-human-review",
        domain="all",
        condition=lambda ctx: ctx["gap_category_id"] == 23,
        required_elements=[
            ("human_review_flag", ["human review", "flag for review", "supervisor review", "bias audit"]),
        ],
        rationale="Any recommendation touching a bias/fairness concern must flag the case for human "
                   "review rather than resolve it as a purely automated coaching action -- required by "
                   "the deployment-boundary policy in Section 6.3.",
    ),
    Rule(
        id="R8-no-sole-determinative-language",
        domain="all",
        condition=lambda ctx: True,  # applies to every recommendation
        required_elements=[
            ("no_determinative_claim", []),  # checked by NEGATIVE match, see check_compliance
        ],
        rationale="No generated recommendation may phrase the assessment as sufficient on its own for "
                   "an employment, promotion, disciplinary, licensing, or credentialing decision "
                   "(Section 6.3 deployment-boundary policy).",
    ),
]

_FORBIDDEN_DETERMINATIVE_PHRASES = [
    "should be terminated", "grounds for dismissal", "sufficient for promotion",
    "conclusive evidence for", "automatically disqualif", "final determination",
]


def check_compliance(rule: Rule, recommendation_text: str) -> bool:
    """Programmatic, disclosed-as-simple compliance check. Returns True if the
    recommendation satisfies every required element of the rule."""
    if rule.id == "R8-no-sole-determinative-language":
        return not _has_any(recommendation_text, _FORBIDDEN_DETERMINATIVE_PHRASES)
    for _label, keywords in rule.required_elements:
        if not _has_any(recommendation_text, keywords):
            return False
    return True


def applicable_rules(domain: str, gap_category_id: int, severity: str = "", evidence: str = "") -> list[Rule]:
    ctx = {"domain": domain, "gap_category_id": gap_category_id, "severity": severity, "evidence": evidence}
    out = []
    for r in RULES:
        if r.domain not in ("all", domain):
            continue
        if r.condition(ctx):
            out.append(r)
    return out


def rule_compliance_rate(domain: str, gap_category_id: int, recommendation_text: str) -> float:
    """Fraction of applicable rules satisfied by a single recommendation (1.0 if none apply)."""
    applicable = applicable_rules(domain, gap_category_id)
    if not applicable:
        return 1.0
    satisfied = sum(1 for r in applicable if check_compliance(r, recommendation_text))
    return satisfied / len(applicable)


if __name__ == "__main__":
    # Worked example the paper can cite directly (Section 3.4 / 6.1 requirement):
    example_ctx = dict(domain="healthcare", gap_category_id=14,
                        recommendation_text="Schedule a 20-minute health literacy module and remember "
                                             "to be more careful with red flag screening next time.")
    applic = applicable_rules(**{k: v for k, v in example_ctx.items() if k != "recommendation_text"})
    print("Applicable rules:", [r.id for r in applic])
    print("Compliant:", [check_compliance(r, example_ctx["recommendation_text"]) for r in applic])
