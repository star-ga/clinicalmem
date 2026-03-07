"""
Clinical scoring — MIND kernel-inspired scoring for healthcare memory.

Implements the abstention, importance, and adversarial kernels from MIND Lang
as pure Python for hackathon use. These demonstrate MIND Lang's expressiveness
without requiring the commercial runtime.

Kernel sources: https://github.com/star-ga/mind/tree/main/mind/
"""
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ClinicalConfidence:
    """Result of confidence gating — decides if we should answer or abstain."""

    score: float  # 0.0-1.0
    should_abstain: bool
    reason: str

    @property
    def level(self) -> str:
        if self.score >= 0.8:
            return "high"
        if self.score >= 0.5:
            return "moderate"
        return "low"


def confidence_gate(
    bm25_scores: list[float],
    entity_overlaps: list[float],
    score_weight: float = 0.6,
    overlap_weight: float = 0.4,
    abstention_threshold: float = 0.3,
) -> ClinicalConfidence:
    """
    Confidence gating from abstention.mind kernel.

    Decides whether the system has enough evidence to answer a clinical query.
    In healthcare, abstaining is better than guessing.

    Args:
        bm25_scores: BM25 retrieval scores for top results
        entity_overlaps: Entity overlap ratios between query and results
        score_weight: Weight for retrieval scores
        overlap_weight: Weight for entity overlap
        abstention_threshold: Below this, abstain from answering
    """
    if not bm25_scores:
        return ClinicalConfidence(
            score=0.0,
            should_abstain=True,
            reason="No matching clinical records found",
        )

    n = len(bm25_scores)
    avg_score = sum(bm25_scores) / n
    avg_overlap = sum(entity_overlaps) / n
    confidence = avg_score * score_weight + avg_overlap * overlap_weight

    should_abstain = confidence < abstention_threshold
    if should_abstain:
        reason = (
            f"Confidence {confidence:.2f} below clinical threshold "
            f"{abstention_threshold}. Insufficient evidence for a safe answer."
        )
    else:
        reason = f"Confidence {confidence:.2f} — sufficient clinical evidence."

    return ClinicalConfidence(
        score=confidence,
        should_abstain=should_abstain,
        reason=reason,
    )


def clinical_importance(
    access_count: int,
    days_since_access: float,
    connection_degree: int,
    is_acute: bool = False,
    decay_rate: float = -0.1,
    access_weight: float = 0.3,
    recency_weight: float = 0.5,
    connection_weight: float = 0.2,
) -> float:
    """
    Importance scoring from importance.mind kernel.

    Scores how important a clinical memory block is for current context.
    Acute conditions get a boost. Returns a multiplier in [0.8, 1.5].
    """
    freq = math.log(access_count + 1)
    recency = math.exp(decay_rate * days_since_access)
    conn = math.log(connection_degree + 1)

    raw = access_weight * freq + recency_weight * recency + connection_weight * conn
    max_est = access_weight * 3.0 + recency_weight * 1.0 + connection_weight * 3.0
    normalized = raw / (max_est + 1e-6)
    score = 0.8 + normalized * 0.7

    if is_acute:
        score = min(score * 1.3, 1.5)

    return round(score, 4)


def medication_severity_score(
    interaction_type: str,
    severity: str | None = None,
) -> float:
    """
    Score medication interaction severity for prioritization.

    Returns 0.0-1.0 where higher = more severe.
    """
    base_scores = {
        "contraindicated": 1.0,
        "serious": 0.8,
        "moderate": 0.5,
        "minor": 0.2,
        "unknown": 0.4,
    }
    severity_boost = {
        "high": 0.15,
        "moderate": 0.05,
        "low": 0.0,
    }
    base = base_scores.get(interaction_type.lower(), 0.4)
    boost = severity_boost.get((severity or "").lower(), 0.0)
    return min(base + boost, 1.0)


def is_negation_query(query: str) -> bool:
    """
    Negation detection from adversarial.mind kernel.

    Critical for clinical queries like "NOT allergic to penicillin"
    vs "allergic to penicillin".
    """
    negation_markers = [
        "not ", "no ", "never ", "without ", "absence of ",
        "denies ", "negative for ", "ruled out", "unlikely ",
        "non-", "un-", "n't ",
    ]
    lower = query.lower()
    return any(marker in lower for marker in negation_markers)


# Known drug interaction pairs (subset for demo — production would use RxNorm API)
_KNOWN_INTERACTIONS: list[tuple[str, str, str, str]] = [
    ("warfarin", "aspirin", "serious", "Increased bleeding risk"),
    ("warfarin", "ibuprofen", "serious", "Increased bleeding risk"),
    ("warfarin", "naproxen", "serious", "Increased bleeding risk"),
    ("warfarin", "nsaid", "serious", "Increased bleeding risk"),
    ("metformin", "contrast dye", "contraindicated", "Risk of lactic acidosis"),
    ("lisinopril", "potassium", "moderate", "Risk of hyperkalemia"),
    ("lisinopril", "spironolactone", "moderate", "Risk of hyperkalemia"),
    ("metoprolol", "verapamil", "serious", "Risk of severe bradycardia"),
    ("simvastatin", "amiodarone", "serious", "Increased risk of rhabdomyolysis"),
    ("fluoxetine", "tramadol", "serious", "Serotonin syndrome risk"),
    ("ciprofloxacin", "tizanidine", "contraindicated", "Dangerous hypotension"),
    ("methotrexate", "trimethoprim", "serious", "Increased methotrexate toxicity"),
]

# Known allergy cross-reactions
_ALLERGY_CROSS_REACTIONS: list[tuple[str, list[str], str]] = [
    ("penicillin", ["amoxicillin", "ampicillin", "piperacillin"], "Beta-lactam cross-reactivity"),
    ("sulfa", ["sulfamethoxazole", "sulfasalazine", "celecoxib"], "Sulfonamide cross-reactivity"),
    ("codeine", ["morphine", "hydrocodone", "oxycodone"], "Opioid cross-sensitivity"),
    ("nsaid", ["ibuprofen", "naproxen", "aspirin", "ketorolac"], "NSAID class reaction"),
]


@dataclass(frozen=True)
class DrugInteraction:
    drug_a: str
    drug_b: str
    severity: str
    description: str
    score: float


def check_drug_interactions(medications: list[str]) -> list[DrugInteraction]:
    """Check a medication list for known interactions."""
    meds_lower = [m.lower().strip() for m in medications]
    interactions = []

    for drug_a, drug_b, severity, description in _KNOWN_INTERACTIONS:
        a_match = any(drug_a in m for m in meds_lower)
        b_match = any(drug_b in m for m in meds_lower)
        if a_match and b_match:
            interactions.append(
                DrugInteraction(
                    drug_a=drug_a,
                    drug_b=drug_b,
                    severity=severity,
                    description=description,
                    score=medication_severity_score(severity),
                )
            )

    return sorted(interactions, key=lambda i: i.score, reverse=True)


@dataclass(frozen=True)
class AllergyConflict:
    allergen: str
    medication: str
    cross_reaction_group: str
    description: str


def check_allergy_conflicts(
    allergies: list[str], medications: list[str]
) -> list[AllergyConflict]:
    """Cross-reference patient allergies against prescribed medications."""
    allergies_lower = [a.lower().strip() for a in allergies]
    meds_lower = [m.lower().strip() for m in medications]
    conflicts = []

    for allergen, cross_drugs, description in _ALLERGY_CROSS_REACTIONS:
        allergen_match = any(allergen in a for a in allergies_lower)
        if not allergen_match:
            continue
        for drug in cross_drugs:
            drug_match = any(drug in m for m in meds_lower)
            if drug_match:
                conflicts.append(
                    AllergyConflict(
                        allergen=allergen,
                        medication=drug,
                        cross_reaction_group=allergen,
                        description=description,
                    )
                )

    return conflicts
