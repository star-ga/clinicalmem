"""
Clinical safety tools — medication safety and contradiction detection.

These are the high-value tools that differentiate ClinicalMem from
basic FHIR query agents. They perform multi-step clinical analysis
using MIND Lang scoring kernels.
"""
import logging
import os
import sys

from google.adk.tools import ToolContext

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from engine.clinical_memory import ClinicalMemEngine
from engine.fhir_client import FHIRClient, FHIRContext

logger = logging.getLogger(__name__)


def _demo_patient_id(tool_context: ToolContext) -> str:
    """Return patient_id from session state, falling back to demo patient if DEMO_MODE."""
    pid = tool_context.state.get("patient_id", "")
    if not pid and os.environ.get("DEMO_MODE", "").lower() in ("1", "true", "yes"):
        pid = "patient-sarah-mitchell"
    return pid


def _get_engine() -> ClinicalMemEngine:
    from a2a_agent.tools.memory_tools import _engine
    return _engine


def _auto_ingest(tool_context: ToolContext) -> None:
    fhir_url = tool_context.state.get("fhir_url", "")
    fhir_token = tool_context.state.get("fhir_token", "")
    patient_id = _demo_patient_id(tool_context)
    if not all([fhir_url, fhir_token, patient_id]):
        return
    engine = _get_engine()
    if patient_id in engine._patient_blocks and engine._patient_blocks[patient_id]:
        return
    try:
        ctx = FHIRContext(url=fhir_url, token=fhir_token, patient_id=patient_id)
        fhir = FHIRClient(ctx)
        engine.ingest_from_fhir(fhir)
    except Exception as e:
        logger.warning("Auto-ingest failed for %s: %s", patient_id, e)


def medication_safety_review(tool_context: ToolContext) -> dict:
    """
    Comprehensive medication safety assessment for the current patient.

    Performs drug-drug interaction detection, allergy cross-reference checking,
    and severity scoring using MIND Lang clinical scoring kernels.
    Returns a structured safety report with actionable findings.

    No arguments required — patient identity comes from the session FHIR context.
    """
    _auto_ingest(tool_context)
    patient_id = _demo_patient_id(tool_context)
    if not patient_id:
        return {"status": "error", "error_message": "No patient_id in session context."}

    engine = _get_engine()
    report = engine.medication_safety_check(patient_id)

    interactions = [
        {
            "drug_a": i.drug_a,
            "drug_b": i.drug_b,
            "severity": i.severity,
            "description": i.description,
            "severity_score": i.score,
        }
        for i in report.interactions
    ]

    allergy_conflicts = [
        {
            "allergen": c.allergen,
            "prescribed_medication": c.medication,
            "cross_reaction_group": c.cross_reaction_group,
            "description": c.description,
        }
        for c in report.allergy_conflicts
    ]

    critical_count = sum(
        1 for i in report.interactions if i.severity in ("contraindicated", "serious")
    ) + len(report.allergy_conflicts)

    return {
        "status": "success",
        "patient_id": report.patient_id,
        "medications_reviewed": report.medications,
        "medication_count": len(report.medications),
        "drug_interactions": interactions,
        "interaction_count": len(interactions),
        "allergy_conflicts": allergy_conflicts,
        "allergy_conflict_count": len(allergy_conflicts),
        "critical_findings": critical_count,
        "confidence": {
            "score": round(report.confidence.score, 3),
            "level": report.confidence.level,
        },
        "summary": report.summary,
        "audit_hash": report.audit_hash,
    }


def detect_record_contradictions(tool_context: ToolContext) -> dict:
    """
    Comprehensive clinical contradiction scanner for the current patient.

    Detects 5 types of contradictions:
    1. Allergy-medication conflicts (e.g., penicillin allergy + amoxicillin Rx)
    2. Drug-drug interactions (e.g., warfarin + ibuprofen bleeding risk)
    3. Lab-medication contraindications (e.g., declining GFR + metformin)
    4. Lab trend alerts (e.g., GFR declining trajectory)
    5. Provider disagreements (e.g., conflicting BP targets from different specialists)

    Each finding includes severity level and actionable clinical recommendation.
    No arguments required — patient identity comes from session FHIR context.
    """
    _auto_ingest(tool_context)
    patient_id = _demo_patient_id(tool_context)
    if not patient_id:
        return {"status": "error", "error_message": "No patient_id in session context."}

    engine = _get_engine()
    contradictions = engine.detect_contradictions(patient_id)

    critical = [c for c in contradictions if c["severity"] == "critical"]
    high = [c for c in contradictions if c["severity"] == "high"]

    escalation = None
    if critical:
        escalation = (
            f"IMMEDIATE CLINICAL REVIEW REQUIRED. "
            f"{len(critical)} critical finding(s): "
            + "; ".join(c["description"][:100] for c in critical)
        )
    elif high:
        escalation = (
            f"PRIORITY REVIEW RECOMMENDED. {len(high)} high-severity finding(s) detected."
        )

    # Verify audit chain integrity for trust proof
    chain_verified = engine.verify_audit_chain()
    trail = engine.get_audit_trail(limit=1)
    latest_audit_hash = trail[-1].get("entry_hash") or trail[-1].get("hash", "") if trail else ""

    return {
        "status": "success",
        "patient_id": patient_id,
        "contradictions": contradictions,
        "contradiction_count": len(contradictions),
        "critical_count": len(critical),
        "high_count": len(high),
        "types_found": list({c["type"] for c in contradictions}),
        "has_critical": bool(critical),
        "has_high": bool(high),
        "escalation": escalation,
        "audit_hash": latest_audit_hash,
        "chain_integrity": "verified" if chain_verified else "TAMPERED",
    }


def what_if_scenario(
    tool_context: ToolContext,
    action: str = "add",
    medication: str = "",
    swap_to: str = "",
) -> dict:
    """
    Simulate what happens if a medication is added, removed, or swapped.

    Runs the proposed change through the full safety pipeline BEFORE it reaches
    the patient. Shows new risks, resolved risks, and whether it's safe to proceed.

    Args:
        action: "add", "remove", or "swap"
        medication: The medication to add/remove (or remove in swap)
        swap_to: For swap only — the replacement medication
    """
    _auto_ingest(tool_context)
    patient_id = _demo_patient_id(tool_context)
    if not patient_id:
        return {"status": "error", "error_message": "No patient_id in session context."}

    engine = _get_engine()

    from engine.what_if import (
        simulate_add_medication,
        simulate_remove_medication,
        simulate_swap_medication,
    )

    blocks = engine._patient_blocks.get(patient_id, [])
    current_meds = [
        b.metadata.get("medication_name", "")
        for b in blocks if b.resource_type == "MedicationRequest"
    ]
    allergies = [
        b.metadata.get("allergen", "")
        for b in blocks if b.resource_type == "AllergyIntolerance"
    ]
    observations = [
        {"code": b.metadata.get("code", ""), "value": b.metadata.get("value"), "unit": b.metadata.get("unit", "")}
        for b in blocks if b.resource_type == "Observation"
    ]

    action_lower = action.strip().lower()
    if action_lower == "add":
        result = simulate_add_medication(patient_id, medication, current_meds, allergies, observations)
    elif action_lower == "remove":
        result = simulate_remove_medication(patient_id, medication, current_meds, allergies, observations)
    elif action_lower == "swap" and swap_to:
        result = simulate_swap_medication(patient_id, medication, swap_to, current_meds, allergies, observations)
    else:
        return {"status": "error", "error_message": f"Invalid action '{action}'. Use add/remove/swap."}

    return {
        "status": "success",
        "scenario": result.scenario,
        "safe_to_proceed": result.safe_to_proceed,
        "risk_delta": result.risk_delta,
        "new_interactions": [
            {"drug_a": i.drug_a, "drug_b": i.drug_b, "severity": i.severity, "description": i.description}
            for i in result.new_interactions
        ],
        "new_allergy_conflicts": [
            {"allergen": c.allergen, "medication": c.medication, "description": c.description}
            for c in result.new_allergy_conflicts
        ],
        "removed_risks": result.removed_risks,
        "recommendation": result.recommendation,
    }


def check_fda_alerts(tool_context: ToolContext) -> dict:
    """
    Query real FDA safety data for the patient's active medications.

    Checks openFDA for adverse event reports, FDA-approved label warnings
    (including black box warnings), and active drug recalls.
    This is real federal safety data from the FAERS database.

    No arguments required — patient identity comes from session FHIR context.
    """
    _auto_ingest(tool_context)
    patient_id = _demo_patient_id(tool_context)
    if not patient_id:
        return {"status": "error", "error_message": "No patient_id in session context."}

    engine = _get_engine()

    from engine.fda_client import get_safety_profile

    blocks = engine._patient_blocks.get(patient_id, [])
    medications = [
        b.metadata.get("medication_name", "")
        for b in blocks if b.resource_type == "MedicationRequest"
        and b.metadata.get("medication_name")
    ]

    if not medications:
        return {"status": "success", "medications": [], "alerts": [], "message": "No active medications found."}

    profile = get_safety_profile(medications)
    return {
        "status": "success",
        "medications": profile.medications,
        "alert_count": len(profile.alerts),
        "alerts": [
            {"drug": a.drug_name, "type": a.alert_type, "severity": a.severity, "description": a.description, "source": a.source}
            for a in profile.alerts[:20]
        ],
        "black_box_warnings": profile.black_box_warnings,
        "highest_severity": profile.highest_severity,
    }


def find_clinical_trials(tool_context: ToolContext) -> dict:
    """
    Match the patient's conditions to active recruiting clinical trials.

    Queries ClinicalTrials.gov API v2 for trials matching the patient's active
    conditions. Returns real NCT numbers, trial details, and enrollment info.

    No arguments required — patient identity comes from session FHIR context.
    """
    _auto_ingest(tool_context)
    patient_id = _demo_patient_id(tool_context)
    if not patient_id:
        return {"status": "error", "error_message": "No patient_id in session context."}

    engine = _get_engine()

    from engine.trials_client import match_patient_to_trials

    blocks = engine._patient_blocks.get(patient_id, [])
    conditions = [
        b.metadata.get("condition_name", "")
        for b in blocks if b.resource_type == "Condition"
        and b.metadata.get("condition_name")
    ]

    if not conditions:
        return {"status": "success", "conditions": [], "trials": [], "message": "No active conditions found."}

    result = match_patient_to_trials(conditions)
    return {
        "status": "success",
        "patient_conditions": list(result.patient_conditions),
        "total_trials_found": result.total_found,
        "trials": [
            {
                "nct_id": t.nct_id, "title": t.title, "status": t.status,
                "conditions": list(t.conditions), "phase": t.phase,
                "enrollment": t.enrollment, "url": t.url,
            }
            for t in result.matched_trials[:10]
        ],
    }


def consensus_verify(
    tool_context: ToolContext,
    finding: str = "",
) -> dict:
    """
    Multi-LLM consensus verification for a critical clinical safety finding.

    Sends the finding to up to 6 independent LLMs in parallel and requires
    >=2/3 agreement before confirming. Use for high-severity findings only.

    Args:
        finding: The clinical safety finding to verify (e.g., "Warfarin + Ibuprofen
            creates serious bleeding risk")
    """
    _auto_ingest(tool_context)
    patient_id = _demo_patient_id(tool_context)
    if not patient_id:
        return {"status": "error", "error_message": "No patient_id in session context."}
    if not finding:
        return {"status": "error", "error_message": "No finding provided to verify."}

    engine = _get_engine()

    from engine.consensus_engine import verify_finding_consensus_sync

    blocks = engine._patient_blocks.get(patient_id, [])
    evidence = [
        {"block_id": b.block_id, "title": b.title, "content": b.content}
        for b in blocks[:10]
    ]

    summary = engine.patient_summary(patient_id)
    patient_context = {
        "patient_id": patient_id,
        "medications": summary.get("medications", []),
        "conditions": summary.get("conditions", []),
        "allergies": summary.get("allergies", []),
    }

    result = verify_finding_consensus_sync(finding, evidence, patient_context)
    return {
        "status": "success",
        "finding": result.finding,
        "consensus_level": result.consensus_level,
        "agreement": f"{result.agreement_count}/{result.total_models}",
        "confidence_score": result.confidence_score,
        "should_report": result.should_report,
        "verdicts": [
            {"model": v.model, "agrees": v.agrees, "confidence": v.confidence, "reasoning": v.reasoning}
            for v in result.verdicts
        ],
    }


def explain_clinical_conflict(
    tool_context: ToolContext,
    conflict_index: int = 0,
) -> dict:
    """
    Generate a patient-specific LLM explanation for a detected clinical conflict.

    Uses deterministic detection + GenAI synthesis pattern:
    - Detection: rule-based safety rails (reliable, auditable)
    - Explanation: LLM-generated (expressive, context-aware, with evidence citations)
    - Abstention: hard gate when evidence is insufficient — refuses to guess

    Args:
        conflict_index: Which detected conflict to explain (0-based). Run
            detect_record_contradictions first to see available conflicts.
    """
    _auto_ingest(tool_context)
    patient_id = _demo_patient_id(tool_context)
    if not patient_id:
        return {"status": "error", "error_message": "No patient_id in session context."}

    engine = _get_engine()
    narrative = engine.explain_clinical_conflict(patient_id, conflict_index)

    chain_verified = engine.verify_audit_chain()
    trail = engine.get_audit_trail(limit=1)
    latest_audit_hash = trail[-1].get("entry_hash") or trail[-1].get("hash", "") if trail else ""

    return {
        "status": "success",
        "patient_id": patient_id,
        "narrative": narrative.narrative,
        "evidence_citations": narrative.evidence_citations,
        "confidence_score": round(narrative.confidence_score, 3),
        "abstained": narrative.abstained,
        "model_used": narrative.model_used,
        "audit_hash": latest_audit_hash,
        "chain_integrity": "verified" if chain_verified else "TAMPERED",
    }
