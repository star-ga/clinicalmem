"""
What-If Scenario Engine — Digital twin clinical simulation.

Simulates the effect of adding/removing medications or changing treatments
on a patient's clinical profile. Detects NEW interactions, allergy conflicts,
and contraindications that would arise from the proposed change.

Extends ClinicalMem's 6-layer safety pipeline into prospective analysis:
before a treatment change reaches the patient, run it through the same
deterministic interaction checks and evidence grounding used for real-time safety.
"""
import copy
import hashlib
import logging
from dataclasses import dataclass
from typing import Any


def _hash_patient_id(patient_id: str) -> str:
    """PHI-safe 16-char SHA-256 prefix of patient_id.

    iter-334 PHI extras-key migration: hash patient_id before logging
    via `extra={}`. Mirrors the iter-291 / iter-284 / iter-279 /
    iter-309 / iter-332 / iter-333 PHI discipline pattern.
    """
    return hashlib.sha256((patient_id or "").encode("utf-8")).hexdigest()[:16]

from engine.clinical_scoring import (
    DrugInteraction,
    AllergyConflict,
    LabMedContraindication,
    check_drug_interactions,
    check_allergy_conflicts,
    check_lab_medication_contraindications,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WhatIfResult:
    """Result of a what-if clinical scenario simulation."""

    scenario: str
    patient_id: str
    new_interactions: list[DrugInteraction]
    new_allergy_conflicts: list[AllergyConflict]
    new_lab_contraindications: list[LabMedContraindication]
    removed_risks: list[str]
    risk_delta: int  # positive = more risk, negative = safer
    recommendation: str
    safe_to_proceed: bool


def simulate_add_medication(
    patient_id: str,
    new_medication: str,
    current_medications: list[str],
    allergies: list[str],
    observations: list[dict],
) -> WhatIfResult:
    """
    Simulate: "What if we add [medication] to this patient's regimen?"

    Compares the safety profile BEFORE and AFTER adding the medication
    to identify NEW risks introduced by the change.
    """
    # PHI-safe entry log: arity counts only (no drug names, no allergy text).
    logger.debug(
        "what_if_add_entry",
        extra={
            "patient_id_hash_prefix": _hash_patient_id(patient_id),
            "scenario": "add",
            "current_med_count": len(current_medications),
            "allergy_count": len(allergies),
            "observation_count": len(observations),
        },
    )

    # Baseline: current interactions
    baseline_interactions = check_drug_interactions(
        current_medications, use_llm_fallback=False
    )
    baseline_allergy = check_allergy_conflicts(allergies, current_medications)
    baseline_lab = check_lab_medication_contraindications(
        observations, current_medications
    )

    # Scenario: add the new medication
    scenario_meds = current_medications + [new_medication]
    scenario_interactions = check_drug_interactions(
        scenario_meds, use_llm_fallback=False
    )
    scenario_allergy = check_allergy_conflicts(allergies, scenario_meds)
    scenario_lab = check_lab_medication_contraindications(
        observations, scenario_meds
    )

    # Diff: find NEW risks
    baseline_interaction_keys = {
        (i.drug_a, i.drug_b) for i in baseline_interactions
    }
    new_interactions = [
        i for i in scenario_interactions
        if (i.drug_a, i.drug_b) not in baseline_interaction_keys
    ]

    baseline_allergy_keys = {
        (c.allergen, c.medication) for c in baseline_allergy
    }
    new_allergy_conflicts = [
        c for c in scenario_allergy
        if (c.allergen, c.medication) not in baseline_allergy_keys
    ]

    baseline_lab_keys = {
        (c.lab_name, c.medication) for c in baseline_lab
    }
    new_lab_contra = [
        c for c in scenario_lab
        if (c.lab_name, c.medication) not in baseline_lab_keys
    ]

    # Risk delta
    new_risk_count = (
        len(new_interactions) + len(new_allergy_conflicts) + len(new_lab_contra)
    )
    risk_delta = new_risk_count

    # Safety assessment
    has_critical = any(
        i.severity == "contraindicated" for i in new_interactions
    ) or any(
        c.severity == "critical" for c in new_lab_contra
    ) or len(new_allergy_conflicts) > 0

    safe_to_proceed = not has_critical

    # PHI-safe outcome log: severity tier + counts only.
    # WARNING level when critical risk surfaces — operators must see this
    # at production default log levels. INFO otherwise.
    if has_critical:
        logger.warning(
            "what_if_add_critical_risk",
            extra={
                "patient_id_hash_prefix": _hash_patient_id(patient_id),
                "scenario": "add",
                "new_interaction_count": len(new_interactions),
                "new_allergy_count": len(new_allergy_conflicts),
                "new_lab_contra_count": len(new_lab_contra),
                "safe_to_proceed": False,
            },
        )
    else:
        logger.info(
            "what_if_add_outcome",
            extra={
                "patient_id_hash_prefix": _hash_patient_id(patient_id),
                "scenario": "add",
                "new_risk_count": new_risk_count,
                "safe_to_proceed": True,
            },
        )

    # Build recommendation
    if not new_risk_count:
        # DEBUG — recommendation-path tracking lets ops measure how
        # often each branch fires (no_risks / critical / monitored).
        # The 3-class distribution is a useful indicator of cohort
        # complexity drift across releases. PHI-safe: structural label
        # + scenario + patient_id (synthetic) only.
        logger.debug(
            "what_if_recommendation_path",
            extra={
                "patient_id_hash_prefix": _hash_patient_id(patient_id),
                "scenario": "add",
                "branch": "no_new_risks",
                "new_risk_count": 0,
            },
        )
        recommendation = (
            f"Adding {new_medication} introduces NO new interactions, "
            f"allergy conflicts, or lab contraindications. "
            f"Safe to proceed with standard monitoring."
        )
    elif has_critical:
        # DEBUG — recommendation-path tracking parity: the no_new_risks
        # branch already emits what_if_recommendation_path; the critical
        # branch had its own WARNING but no path log, leaving the
        # 3-class branch distribution incomplete for ops dashboards.
        # PHI-safe: structural label + scenario + patient_id only.
        logger.debug(
            "what_if_recommendation_path",
            extra={
                "patient_id_hash_prefix": _hash_patient_id(patient_id),
                "scenario": "add",
                "branch": "critical",
                "new_risk_count": new_risk_count,
            },
        )
        parts = []
        for i in new_interactions:
            if i.severity == "contraindicated":
                parts.append(f"CONTRAINDICATED: {i.drug_a} + {i.drug_b} — {i.description}")
        for c in new_allergy_conflicts:
            parts.append(f"ALLERGY CONFLICT: {c.allergen} allergy vs {c.medication}")
        for c in new_lab_contra:
            if c.severity == "critical":
                parts.append(f"LAB CONTRAINDICATION: {c.description}")
        recommendation = (
            f"DO NOT ADD {new_medication}. Critical risks detected: "
            + "; ".join(parts)
            + ". Consider alternative medications."
        )
    else:
        # DEBUG — completes the 3-class recommendation-path distribution
        # (no_new_risks / critical / monitored). The middle branch was
        # silent before iter-226 — ops couldn't measure how often
        # what-if simulations land on "monitored" vs the extremes.
        # PHI-safe: structural label + scenario + patient_id only.
        logger.debug(
            "what_if_recommendation_path",
            extra={
                "patient_id_hash_prefix": _hash_patient_id(patient_id),
                "scenario": "add",
                "branch": "monitored",
                "new_risk_count": new_risk_count,
            },
        )
        recommendation = (
            f"Adding {new_medication} introduces {new_risk_count} new risk(s) "
            f"but none are critical. Proceed with enhanced monitoring: "
            + "; ".join(
                f"{i.drug_a}+{i.drug_b} ({i.severity})" for i in new_interactions
            )
        )

    return WhatIfResult(
        scenario=f"Add {new_medication}",
        patient_id=patient_id,
        new_interactions=new_interactions,
        new_allergy_conflicts=new_allergy_conflicts,
        new_lab_contraindications=new_lab_contra,
        removed_risks=[],
        risk_delta=risk_delta,
        recommendation=recommendation,
        safe_to_proceed=safe_to_proceed,
    )


def simulate_remove_medication(
    patient_id: str,
    remove_medication: str,
    current_medications: list[str],
    allergies: list[str],
    observations: list[dict],
) -> WhatIfResult:
    """
    Simulate: "What if we remove [medication] from this patient's regimen?"

    Identifies risks that would be ELIMINATED by removing the medication.
    """
    # PHI-safe entry log.
    logger.debug(
        "what_if_remove_entry",
        extra={
            "patient_id_hash_prefix": _hash_patient_id(patient_id),
            "scenario": "remove",
            "current_med_count": len(current_medications),
        },
    )

    # Baseline
    baseline_interactions = check_drug_interactions(
        current_medications, use_llm_fallback=False
    )
    baseline_lab = check_lab_medication_contraindications(
        observations, current_medications
    )

    # Scenario: remove the medication
    scenario_meds = [
        m for m in current_medications
        if remove_medication.lower() not in m.lower()
    ]
    if len(scenario_meds) == len(current_medications):
        # WARNING — clinical-input-validation signal. The caller asked
        # to remove a medication that isn't in the current list, so the
        # simulation is a no-op. Could indicate a drug-name typo, an
        # outdated EHR pull, or a bug in the upstream caller.
        # PHI-safe: counts only, NOT the missing drug name (clinician
        # queries can carry narrative — same discipline as recall).
        logger.warning(
            "what_if_remove_drug_not_in_list",
            extra={
                "patient_id_hash_prefix": _hash_patient_id(patient_id),
                "scenario": "remove",
                "current_med_count": len(current_medications),
            },
        )
    scenario_interactions = check_drug_interactions(
        scenario_meds, use_llm_fallback=False
    )
    scenario_lab = check_lab_medication_contraindications(
        observations, scenario_meds
    )

    # Diff: find REMOVED risks
    scenario_keys = {(i.drug_a, i.drug_b) for i in scenario_interactions}
    removed_interactions = [
        i for i in baseline_interactions
        if (i.drug_a, i.drug_b) not in scenario_keys
    ]

    scenario_lab_keys = {(c.lab_name, c.medication) for c in scenario_lab}
    removed_lab = [
        c for c in baseline_lab
        if (c.lab_name, c.medication) not in scenario_lab_keys
    ]

    removed_risks = [
        f"Resolved: {i.drug_a} + {i.drug_b} ({i.severity}: {i.description})"
        for i in removed_interactions
    ] + [
        f"Resolved: {c.lab_name} contraindication with {c.medication}"
        for c in removed_lab
    ]

    risk_delta = -len(removed_risks)

    # PHI-safe outcome log: count of resolved risks only.
    logger.info(
        "what_if_remove_outcome",
        extra={
            "patient_id_hash_prefix": _hash_patient_id(patient_id),
            "scenario": "remove",
            "resolved_risk_count": len(removed_risks),
        },
    )

    if removed_risks:
        # DEBUG — recommendation-path observability for the resolved-risk
        # branch. Mirror-shape of iter-181 simulate_add path tracking.
        logger.debug(
            "what_if_remove_recommendation_path",
            extra={
                "patient_id_hash_prefix": _hash_patient_id(patient_id),
                "scenario": "remove",
                "branch": "resolved",
                "resolved_risk_count": len(removed_risks),
            },
        )
        recommendation = (
            f"Removing {remove_medication} eliminates {len(removed_risks)} risk(s): "
            + "; ".join(removed_risks)
            + ". Ensure the underlying condition is managed with an alternative."
        )
    else:
        # DEBUG — completes the 2-class remove distribution
        # (resolved / no_change). The no_change branch was silent before
        # iter-226 — important to differentiate from the
        # what_if_remove_drug_not_in_list WARNING (different semantic:
        # drug WAS in list but removing it doesn't change risk).
        logger.debug(
            "what_if_remove_recommendation_path",
            extra={
                "patient_id_hash_prefix": _hash_patient_id(patient_id),
                "scenario": "remove",
                "branch": "no_change",
                "resolved_risk_count": 0,
            },
        )
        recommendation = (
            f"Removing {remove_medication} does not change the risk profile. "
            f"Ensure clinical indication for discontinuation exists."
        )

    return WhatIfResult(
        scenario=f"Remove {remove_medication}",
        patient_id=patient_id,
        new_interactions=[],
        new_allergy_conflicts=[],
        new_lab_contraindications=[],
        removed_risks=removed_risks,
        risk_delta=risk_delta,
        recommendation=recommendation,
        safe_to_proceed=True,
    )


def simulate_swap_medication(
    patient_id: str,
    remove_medication: str,
    add_medication: str,
    current_medications: list[str],
    allergies: list[str],
    observations: list[dict],
) -> WhatIfResult:
    """
    Simulate: "What if we switch [drug A] to [drug B]?"

    Combines remove + add to show net risk change.
    """
    # PHI-safe entry log: arity counts only.
    logger.debug(
        "what_if_swap_entry",
        extra={
            "patient_id_hash_prefix": _hash_patient_id(patient_id),
            "scenario": "swap",
            "current_med_count": len(current_medications),
        },
    )

    # Step 1: Remove
    meds_after_remove = [
        m for m in current_medications
        if remove_medication.lower() not in m.lower()
    ]
    if len(meds_after_remove) == len(current_medications):
        # WARNING — caller asked to swap FROM a medication not in the
        # current list. Swap degrades to add-only; the upstream caller
        # almost certainly has a drug-name bug. PHI-safe: counts only.
        logger.warning(
            "what_if_swap_remove_target_not_found",
            extra={
                "patient_id_hash_prefix": _hash_patient_id(patient_id),
                "scenario": "swap",
                "current_med_count": len(current_medications),
            },
        )

    # Step 2: Add
    result = simulate_add_medication(
        patient_id=patient_id,
        new_medication=add_medication,
        current_medications=meds_after_remove,
        allergies=allergies,
        observations=observations,
    )

    # Also compute removed risks
    remove_result = simulate_remove_medication(
        patient_id=patient_id,
        remove_medication=remove_medication,
        current_medications=current_medications,
        allergies=allergies,
        observations=observations,
    )

    net_delta = result.risk_delta + remove_result.risk_delta

    # PHI-safe outcome log: categorical direction (SAFER / NEUTRAL / RISKIER)
    # plus net_delta integer. Drug names not logged (already covered by the
    # nested simulate_add / simulate_remove logs above).
    if net_delta < 0:
        swap_direction = "safer"
    elif net_delta == 0:
        swap_direction = "neutral"
    else:
        swap_direction = "riskier"
    log_fn = logger.warning if swap_direction == "riskier" else logger.info
    log_fn(
        "what_if_swap_outcome",
        extra={
            "patient_id_hash_prefix": _hash_patient_id(patient_id),
            "scenario": "swap",
            "net_delta": net_delta,
            "direction": swap_direction,
            "safe_to_proceed": result.safe_to_proceed,
        },
    )

    if net_delta < 0:
        rec = (
            f"Swapping {remove_medication} → {add_medication} is SAFER "
            f"(net {net_delta} risk change). "
        )
    elif net_delta == 0:
        rec = (
            f"Swapping {remove_medication} → {add_medication} has NEUTRAL "
            f"risk impact. "
        )
    else:
        rec = (
            f"Swapping {remove_medication} → {add_medication} INCREASES risk "
            f"(net +{net_delta}). "
        )

    if result.new_interactions:
        rec += "New interactions: " + "; ".join(
            f"{i.drug_a}+{i.drug_b}" for i in result.new_interactions
        ) + ". "
    if remove_result.removed_risks:
        rec += "Resolved: " + "; ".join(remove_result.removed_risks) + "."

    return WhatIfResult(
        scenario=f"Swap {remove_medication} → {add_medication}",
        patient_id=patient_id,
        new_interactions=result.new_interactions,
        new_allergy_conflicts=result.new_allergy_conflicts,
        new_lab_contraindications=result.new_lab_contraindications,
        removed_risks=remove_result.removed_risks,
        risk_delta=net_delta,
        recommendation=rec,
        safe_to_proceed=result.safe_to_proceed,
    )
