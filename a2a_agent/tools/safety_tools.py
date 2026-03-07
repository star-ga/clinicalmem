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


def _get_engine() -> ClinicalMemEngine:
    from a2a_agent.tools.memory_tools import _engine
    return _engine


def _auto_ingest(tool_context: ToolContext) -> None:
    fhir_url = tool_context.state.get("fhir_url", "")
    fhir_token = tool_context.state.get("fhir_token", "")
    patient_id = tool_context.state.get("patient_id", "")
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
    patient_id = tool_context.state.get("patient_id", "")
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
    Scan the patient's clinical record for contradictions and conflicts.

    Detects:
    - Allergy vs prescription conflicts (e.g., penicillin allergy + amoxicillin Rx)
    - Dangerous drug-drug interactions
    - Conflicting provider recommendations

    Uses MIND Lang adversarial kernel for negation-aware analysis.
    No arguments required — patient identity comes from session FHIR context.
    """
    _auto_ingest(tool_context)
    patient_id = tool_context.state.get("patient_id", "")
    if not patient_id:
        return {"status": "error", "error_message": "No patient_id in session context."}

    engine = _get_engine()
    contradictions = engine.detect_contradictions(patient_id)

    return {
        "status": "success",
        "patient_id": patient_id,
        "contradictions": contradictions,
        "contradiction_count": len(contradictions),
        "has_critical": any(c["severity"] == "critical" for c in contradictions),
        "has_high": any(c["severity"] == "high" for c in contradictions),
        "recommendation": (
            "IMMEDIATE REVIEW REQUIRED: Critical contradictions found in patient record."
            if any(c["severity"] == "critical" for c in contradictions)
            else "No critical contradictions detected."
        ),
    }
