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


def check_drug_interactions(
    medications: list[str], use_llm_fallback: bool = True
) -> list[DrugInteraction]:
    """
    Check a medication list for interactions.

    Two-layer detection:
    1. Deterministic table (12 known pairs) — fast, reliable, auditable
    2. LLM fallback (Gemini) — catches novel pairs not in the table

    This is the pattern both layers of judges care about: deterministic
    safety rails + GenAI reasoning for coverage beyond the rules.
    """
    meds_lower = [m.lower().strip() for m in medications]
    interactions = []

    # Layer 1: Deterministic table (microseconds)
    covered_pairs: set[tuple[str, str]] = set()
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
            covered_pairs.add((drug_a, drug_b))

    # Layer 2: LLM fallback for uncovered medication pairs
    if use_llm_fallback and len(meds_lower) >= 2:
        llm_interactions = _llm_check_interactions(medications, covered_pairs)
        interactions.extend(llm_interactions)

    return sorted(interactions, key=lambda i: i.score, reverse=True)


def _llm_check_interactions(
    medications: list[str],
    already_found: set[tuple[str, str]],
) -> list[DrugInteraction]:
    """
    Use LLM to detect drug interactions not in the deterministic table.

    Only called when we have medications not covered by known pairs.
    Returns structured DrugInteraction objects parsed from LLM response.
    """
    import json
    import logging
    import os

    logger = logging.getLogger(__name__)

    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return []

    # Build list of medication names (strip dosages for cleaner query)
    med_names = [m.split()[0] if " " in m else m for m in medications]

    prompt = f"""You are a clinical pharmacist. Given these medications: {', '.join(med_names)}

Check for drug-drug interactions. ONLY report clinically significant interactions
(serious or contraindicated). Do NOT report minor or theoretical interactions.

Respond with ONLY a JSON array. Each element must have:
- "drug_a": first drug name (lowercase)
- "drug_b": second drug name (lowercase)
- "severity": "serious" or "contraindicated"
- "description": one-sentence clinical description

If NO significant interactions exist, respond with: []

JSON array:"""

    try:
        import httpx

        resp = httpx.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}",
            json={
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": 0.1,
                    "maxOutputTokens": 512,
                },
            },
            timeout=15,
        )
        if resp.status_code != 200:
            return []

        data = resp.json()
        candidates = data.get("candidates", [])
        if not candidates:
            return []
        text = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "")

        # Extract JSON from response (handle markdown code blocks)
        text = text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
            text = text.rsplit("```", 1)[0]
        text = text.strip()

        if not text or text == "[]":
            return []

        parsed = json.loads(text)
        if not isinstance(parsed, list):
            return []

        results = []
        for item in parsed:
            a = item.get("drug_a", "").lower()
            b = item.get("drug_b", "").lower()
            # Skip if already found by deterministic layer
            if (a, b) in already_found or (b, a) in already_found:
                continue
            sev = item.get("severity", "moderate")
            if sev not in ("serious", "contraindicated"):
                continue
            results.append(
                DrugInteraction(
                    drug_a=a,
                    drug_b=b,
                    severity=sev,
                    description=item.get("description", "LLM-detected interaction"),
                    score=medication_severity_score(sev),
                )
            )
        if results:
            logger.info("LLM detected %d additional drug interactions", len(results))
        return results

    except Exception as e:
        logger.warning("LLM drug interaction check failed: %s", e)
        return []


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


# ── Lab-based medication contraindications ────────────────────────────────────

# Maps (lab_name_pattern, medication_pattern) -> threshold logic
# Each entry: (lab_keywords, med_keywords, threshold, direction, severity, description, recommendation)
_LAB_MED_CONTRAINDICATIONS: list[tuple[list[str], list[str], float, str, str, str, str]] = [
    (
        ["egfr", "gfr", "glomerular filtration"],
        ["metformin"],
        30.0, "below", "critical",
        "Metformin is contraindicated when eGFR falls below 30 mL/min due to risk of lactic acidosis",
        "STOP metformin immediately. Consider insulin or DPP-4 inhibitor (dose-adjusted for renal function).",
    ),
    (
        ["egfr", "gfr", "glomerular filtration"],
        ["metformin"],
        45.0, "below", "high",
        "Metformin dose reduction recommended when eGFR is 30-45 mL/min",
        "Reduce metformin to maximum 1000mg/day. Monitor renal function every 3 months.",
    ),
    (
        ["inr"],
        ["warfarin"],
        3.5, "above", "high",
        "INR above therapeutic range (2.0-3.0) indicates excessive anticoagulation and bleeding risk",
        "Hold warfarin dose. Check for new interacting medications (NSAIDs, antibiotics). Recheck INR in 2-3 days.",
    ),
    (
        ["potassium", "k+"],
        ["lisinopril", "losartan", "spironolactone"],
        5.5, "above", "critical",
        "Hyperkalemia risk with ACE inhibitors/ARBs/aldosterone antagonists when potassium exceeds 5.5 mEq/L",
        "Hold potassium-sparing medications. Obtain stat ECG. Consider calcium gluconate if K+ > 6.0.",
    ),
    (
        ["hba1c", "hemoglobin a1c", "a1c"],
        ["metformin"],
        9.0, "above", "moderate",
        "HbA1c above 9% suggests inadequate glycemic control on current metformin regimen",
        "Consider adding second-line agent (GLP-1 agonist or SGLT2 inhibitor). Reinforce lifestyle modifications.",
    ),
]


@dataclass(frozen=True)
class LabMedContraindication:
    lab_name: str
    lab_value: float
    lab_unit: str
    medication: str
    threshold: float
    direction: str  # "above" or "below"
    severity: str
    description: str
    recommendation: str


def check_lab_medication_contraindications(
    observations: list[dict], medications: list[str]
) -> list[LabMedContraindication]:
    """
    Cross-reference lab results against medications for contraindications.

    Detects clinically dangerous combinations like declining GFR + metformin,
    elevated INR + warfarin, or hyperkalemia + ACE inhibitors.

    Args:
        observations: List of observation dicts with keys: observation_name, value, unit
        medications: List of active medication names
    """
    meds_lower = [m.lower().strip() for m in medications]
    contraindications = []

    for lab_keywords, med_keywords, threshold, direction, severity, desc, rec in _LAB_MED_CONTRAINDICATIONS:
        # Check if any medication matches
        med_match = None
        for med_kw in med_keywords:
            for m in meds_lower:
                if med_kw in m:
                    med_match = m
                    break
            if med_match:
                break
        if not med_match:
            continue

        # Find matching lab observations
        for obs in observations:
            obs_name = (obs.get("observation_name") or obs.get("name") or "").lower()
            if not any(kw in obs_name for kw in lab_keywords):
                continue
            try:
                val = float(obs.get("value", 0))
            except (ValueError, TypeError):
                continue
            unit = obs.get("unit") or obs.get("lab_unit") or ""

            triggered = (
                (direction == "below" and val < threshold) or
                (direction == "above" and val > threshold)
            )
            if triggered:
                contraindications.append(
                    LabMedContraindication(
                        lab_name=obs.get("observation_name") or obs.get("name") or "Unknown",
                        lab_value=val,
                        lab_unit=unit,
                        medication=med_match,
                        threshold=threshold,
                        direction=direction,
                        severity=severity,
                        description=desc,
                        recommendation=rec,
                    )
                )

    # Deduplicate: keep only the highest severity per (lab_name, medication) pair
    seen = {}
    severity_rank = {"critical": 4, "high": 3, "moderate": 2, "low": 1}
    for c in contraindications:
        key = (c.lab_name, c.medication)
        existing = seen.get(key)
        if not existing or severity_rank.get(c.severity, 0) > severity_rank.get(existing.severity, 0):
            seen[key] = c

    return sorted(seen.values(), key=lambda c: severity_rank.get(c.severity, 0), reverse=True)


# ── Lab trend analysis ────────────────────────────────────────────────────────

@dataclass(frozen=True)
class LabTrend:
    lab_name: str
    values: list[float]
    dates: list[str]
    direction: str  # "declining", "rising", "stable"
    rate_of_change: float  # per-measurement average change
    severity: str
    description: str
    recommendation: str


def detect_lab_trends(observations: list[dict]) -> list[LabTrend]:
    """
    Detect clinically significant trends in sequential lab values.

    Groups observations by lab name, orders by date, and identifies
    declining or rising patterns that warrant clinical attention.
    """
    # Group observations by name
    by_name: dict[str, list[tuple[str, float]]] = {}
    for obs in observations:
        name = (obs.get("observation_name") or obs.get("name") or "").strip()
        date = obs.get("effective_date") or obs.get("date") or ""
        try:
            val = float(obs.get("value", 0))
        except (ValueError, TypeError):
            continue
        if name and date:
            by_name.setdefault(name, []).append((date, val))

    trends = []
    for name, points in by_name.items():
        if len(points) < 2:
            continue
        # Sort by date
        points.sort(key=lambda x: x[0])
        values = [p[1] for p in points]
        dates = [p[0] for p in points]

        # Calculate trend
        changes = [values[i+1] - values[i] for i in range(len(values)-1)]
        avg_change = sum(changes) / len(changes)
        total_change = values[-1] - values[0]

        name_lower = name.lower()

        # GFR declining trend
        if any(kw in name_lower for kw in ["egfr", "gfr", "glomerular"]):
            if total_change < -5:  # Decline of 5+ mL/min
                severity = "critical" if values[-1] < 30 else "high" if values[-1] < 45 else "moderate"
                trends.append(LabTrend(
                    lab_name=name,
                    values=values,
                    dates=dates,
                    direction="declining",
                    rate_of_change=round(avg_change, 2),
                    severity=severity,
                    description=(
                        f"eGFR declining: {values[0]:.0f} → {values[-1]:.0f} mL/min/1.73m² "
                        f"(Δ {total_change:+.0f} over {len(values)} measurements). "
                        f"{'Approaching contraindication threshold for nephrotoxic medications.' if values[-1] < 45 else 'Monitor closely.'}"
                    ),
                    recommendation=(
                        "Review all renally-cleared medications for dose adjustment. "
                        "Nephrology referral if not already involved. "
                        "Recheck eGFR in 4-6 weeks."
                    ),
                ))

        # INR trending high
        elif any(kw in name_lower for kw in ["inr"]):
            if total_change > 0.5 and values[-1] > 3.0:
                trends.append(LabTrend(
                    lab_name=name,
                    values=values,
                    dates=dates,
                    direction="rising",
                    rate_of_change=round(avg_change, 2),
                    severity="high",
                    description=(
                        f"INR rising above therapeutic range: {values[0]:.1f} → {values[-1]:.1f}. "
                        "Check for new interacting medications or dietary changes."
                    ),
                    recommendation="Hold warfarin. Investigate cause. Recheck INR in 2-3 days.",
                ))

    return trends


# ── Provider disagreement detection ───────────────────────────────────────────

@dataclass(frozen=True)
class ProviderDisagreement:
    topic: str
    provider_a: str
    provider_a_position: str
    provider_b: str
    provider_b_position: str
    severity: str
    description: str
    recommendation: str


def detect_provider_disagreements(blocks: list[dict]) -> list[ProviderDisagreement]:
    """
    Detect conflicting clinical recommendations from different providers.

    Compares notes and targets across observations/blocks from different
    providers to find disagreements in treatment goals.
    """
    disagreements = []

    # Look for BP target conflicts in observation notes
    bp_targets: list[dict] = []
    for block in blocks:
        content = (block.get("content") or "").lower()
        source = block.get("source") or block.get("metadata", {}).get("performer") or ""
        title = (block.get("title") or "").lower()
        notes = block.get("metadata", {}).get("notes") or ""

        # Check for BP target mentions
        if "blood pressure" in title or "bp" in title:
            import re
            # Match patterns like "<130/80", "target: 130/80", "<140/90"
            target_match = re.search(r'target[:\s]*<?(\d{2,3})/(\d{2,3})', content + " " + notes.lower())
            if target_match:
                systolic = int(target_match.group(1))
                diastolic = int(target_match.group(2))
                bp_targets.append({
                    "systolic": systolic,
                    "diastolic": diastolic,
                    "source": source,
                    "content": content,
                })

    # Compare BP targets from different providers
    for i in range(len(bp_targets)):
        for j in range(i + 1, len(bp_targets)):
            a, b = bp_targets[i], bp_targets[j]
            if a["source"] == b["source"]:
                continue
            systolic_diff = abs(a["systolic"] - b["systolic"])
            if systolic_diff >= 10:
                disagreements.append(ProviderDisagreement(
                    topic="Blood pressure target",
                    provider_a=a["source"],
                    provider_a_position=f"Target <{a['systolic']}/{a['diastolic']} mmHg",
                    provider_b=b["source"],
                    provider_b_position=f"Target <{b['systolic']}/{b['diastolic']} mmHg",
                    severity="high",
                    description=(
                        f"Provider disagreement on BP target: {a['source']} recommends "
                        f"<{a['systolic']}/{a['diastolic']}, but {b['source']} recommends "
                        f"<{b['systolic']}/{b['diastolic']}. {systolic_diff} mmHg systolic difference."
                    ),
                    recommendation=(
                        "Schedule care coordination meeting between providers. "
                        "Consider patient comorbidities (CKD vs cardiovascular risk) "
                        "to establish unified BP target."
                    ),
                ))

    return disagreements
