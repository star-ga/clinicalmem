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
