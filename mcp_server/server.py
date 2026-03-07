"""
ClinicalMem MCP Server — Submission 1 for Agents Assemble Hackathon.

A SHARP-on-MCP compliant MCP server that gives any healthcare AI agent
persistent, intelligent clinical memory. Built on mind-mem engine with
MIND Lang scoring kernels.

STARGA Inc. | https://star.ga
"""
import json
import logging
import os
import sys

from fastmcp import FastMCP

# Ensure engine is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.clinical_memory import ClinicalMemEngine
from engine.fhir_client import FHIRClient, FHIRContext, FHIRClientError

logger = logging.getLogger(__name__)

# ── Global state ──────────────────────────────────────────────────────────────

_engine = ClinicalMemEngine()

# ── MCP Server ────────────────────────────────────────────────────────────────

mcp = FastMCP(
    "ClinicalMem",
    version="0.1.0",
    description=(
        "Persistent, auditable, contradiction-safe clinical memory. "
        "Provides intelligent medication safety analysis, clinical context recall, "
        "and tamper-proof audit trails for healthcare AI agents. "
        "Powered by mind-mem engine with MIND Lang scoring kernels."
    ),
)


def _get_fhir_context_from_headers(headers: dict[str, str] | None) -> FHIRContext | None:
    """Extract FHIR context from SHARP-on-MCP HTTP headers."""
    if not headers:
        return None
    url = headers.get("x-fhir-server-url", "") or headers.get("X-FHIR-Server-URL", "")
    token = headers.get("x-fhir-access-token", "") or headers.get("X-FHIR-Access-Token", "")
    patient = headers.get("x-patient-id", "") or headers.get("X-Patient-ID", "")
    if url and token and patient:
        return FHIRContext(url=url, token=token, patient_id=patient)
    return None


# ── Tool: Store Clinical Observation ──────────────────────────────────────────

@mcp.tool()
def store_clinical_observation(
    patient_id: str,
    observation_type: str,
    title: str,
    content: str,
    source: str = "manual",
    fhir_server_url: str = "",
    fhir_access_token: str = "",
) -> dict:
    """
    Store a clinical observation or note for a patient in persistent memory.

    Args:
        patient_id: The patient's FHIR ID
        observation_type: Type of observation (e.g., 'clinical_note', 'lab_result', 'medication_change')
        title: Brief title for the observation
        content: Full text content of the clinical observation
        source: Who/what created this observation
        fhir_server_url: FHIR server URL (from SHARP-on-MCP headers)
        fhir_access_token: FHIR access token (from SHARP-on-MCP headers)
    """
    from engine.clinical_memory import ClinicalBlock
    import time

    block = ClinicalBlock(
        block_id=f"obs-{int(time.time())}-{hash(content) % 10000}",
        patient_id=patient_id,
        resource_type=observation_type,
        title=title,
        content=content,
        metadata={"type": observation_type, "source": source},
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        source=source,
    )
    _engine._store_block(block)
    audit_hash = _engine._append_audit(
        "store_observation",
        {"patient_id": patient_id, "title": title, "type": observation_type},
    )
    return {
        "status": "success",
        "block_id": block.block_id,
        "audit_hash": audit_hash,
        "message": f"Stored observation '{title}' for patient {patient_id}",
    }


# ── Tool: Recall Patient Context ─────────────────────────────────────────────

@mcp.tool()
def recall_patient_context(
    patient_id: str,
    query: str,
    top_k: int = 10,
    fhir_server_url: str = "",
    fhir_access_token: str = "",
) -> dict:
    """
    Retrieve relevant patient history by query using hybrid search (BM25 + vector + RRF).

    Uses MIND Lang abstention kernel for confidence gating — will indicate when
    evidence is insufficient for a safe clinical answer.

    Args:
        patient_id: The patient's FHIR ID
        query: Natural language clinical query (e.g., "What medications is this patient on?")
        top_k: Maximum number of results to return
        fhir_server_url: FHIR server URL (from SHARP-on-MCP headers)
        fhir_access_token: FHIR access token (from SHARP-on-MCP headers)
    """
    # Auto-ingest from FHIR if we have context and no stored blocks
    if fhir_server_url and fhir_access_token:
        if patient_id not in _engine._patient_blocks or not _engine._patient_blocks[patient_id]:
            try:
                ctx = FHIRContext(url=fhir_server_url, token=fhir_access_token, patient_id=patient_id)
                fhir = FHIRClient(ctx)
                _engine.ingest_from_fhir(fhir)
            except Exception as e:
                logger.warning("Auto-ingest failed: %s", e)

    result = _engine.recall(patient_id, query, top_k=top_k)
    return {
        "status": "success",
        "patient_id": result.patient_id,
        "query": result.query,
        "confidence": {
            "score": round(result.confidence.score, 3),
            "level": result.confidence.level,
            "should_abstain": result.confidence.should_abstain,
            "reason": result.confidence.reason,
        },
        "results": result.blocks,
        "result_count": len(result.blocks),
        "audit_hash": result.audit_hash,
    }


# ── Tool: Check Medication Conflicts ─────────────────────────────────────────

@mcp.tool()
def check_medication_conflicts(
    patient_id: str,
    fhir_server_url: str = "",
    fhir_access_token: str = "",
) -> dict:
    """
    Detect drug-drug interaction contradictions for a patient's active medications.

    Uses MIND Lang adversarial kernel for negation-aware analysis and
    clinical scoring for severity ranking.

    Args:
        patient_id: The patient's FHIR ID
        fhir_server_url: FHIR server URL (from SHARP-on-MCP headers)
        fhir_access_token: FHIR access token (from SHARP-on-MCP headers)
    """
    _auto_ingest(patient_id, fhir_server_url, fhir_access_token)
    report = _engine.medication_safety_check(patient_id)

    # Clinical action recommendations based on severity
    recommendations = []
    for i in report.interactions:
        if i.severity == "contraindicated":
            recommendations.append(f"STOP: {i.drug_a} + {i.drug_b} is contraindicated. {i.description}.")
        elif i.severity == "serious":
            recommendations.append(f"REVIEW: {i.drug_a} + {i.drug_b} — {i.description}. Consider alternatives.")
    for c in report.allergy_conflicts:
        recommendations.append(
            f"STOP: {c.medication} prescribed despite {c.allergen} allergy ({c.description}). "
            f"Use alternative outside the {c.cross_reaction_group} class."
        )

    return {
        "status": "success",
        "patient_id": report.patient_id,
        "medications": report.medications,
        "interactions": [
            {
                "drug_a": i.drug_a,
                "drug_b": i.drug_b,
                "severity": i.severity,
                "description": i.description,
                "score": i.score,
            }
            for i in report.interactions
        ],
        "interaction_count": len(report.interactions),
        "allergy_conflicts": [
            {
                "allergen": c.allergen,
                "medication": c.medication,
                "cross_reaction_group": c.cross_reaction_group,
                "description": c.description,
            }
            for c in report.allergy_conflicts
        ],
        "allergy_conflict_count": len(report.allergy_conflicts),
        "recommendations": recommendations,
        "confidence": {
            "score": round(report.confidence.score, 3),
            "level": report.confidence.level,
        },
        "summary": report.summary,
        "audit_hash": report.audit_hash,
    }


# ── Tool: Check Allergy Conflicts ────────────────────────────────────────────

@mcp.tool()
def check_allergy_conflicts(
    patient_id: str,
    fhir_server_url: str = "",
    fhir_access_token: str = "",
) -> dict:
    """
    Cross-reference prescribed medications against known patient allergies.

    Detects cross-reactivity (e.g., Penicillin allergy vs Amoxicillin prescription).

    Args:
        patient_id: The patient's FHIR ID
        fhir_server_url: FHIR server URL (from SHARP-on-MCP headers)
        fhir_access_token: FHIR access token (from SHARP-on-MCP headers)
    """
    _auto_ingest(patient_id, fhir_server_url, fhir_access_token)
    report = _engine.medication_safety_check(patient_id)
    return {
        "status": "success",
        "patient_id": report.patient_id,
        "allergy_conflicts": [
            {
                "allergen": c.allergen,
                "medication": c.medication,
                "cross_reaction_group": c.cross_reaction_group,
                "description": c.description,
            }
            for c in report.allergy_conflicts
        ],
        "conflict_count": len(report.allergy_conflicts),
        "audit_hash": report.audit_hash,
    }


# ── Tool: Get Treatment Dependencies ─────────────────────────────────────────

@mcp.tool()
def get_treatment_dependencies(
    patient_id: str,
    fhir_server_url: str = "",
    fhir_access_token: str = "",
) -> dict:
    """
    Show causal chain of treatment decisions and their dependencies.

    Maps conditions to medications to observations, showing why each
    treatment was prescribed and what depends on it.

    Args:
        patient_id: The patient's FHIR ID
        fhir_server_url: FHIR server URL (from SHARP-on-MCP headers)
        fhir_access_token: FHIR access token (from SHARP-on-MCP headers)
    """
    _auto_ingest(patient_id, fhir_server_url, fhir_access_token)
    blocks = _engine._patient_blocks.get(patient_id, [])

    conditions = [b for b in blocks if b.resource_type == "Condition"]
    medications = [b for b in blocks if b.resource_type == "MedicationRequest"]

    # Build simple dependency map: condition -> medications
    dependencies = []
    for cond in conditions:
        cond_name = cond.metadata.get("condition_name", "").lower()
        related_meds = []
        for med in medications:
            med_name = med.metadata.get("medication_name", "").lower()
            # Simple heuristic: common condition-medication associations
            if _is_related(cond_name, med_name):
                related_meds.append(med.metadata.get("medication_name"))
        dependencies.append({
            "condition": cond.metadata.get("condition_name"),
            "severity": cond.metadata.get("severity"),
            "onset": cond.metadata.get("onset"),
            "related_medications": related_meds,
        })

    audit_hash = _engine._append_audit(
        "treatment_dependencies",
        {"patient_id": patient_id, "dependency_count": len(dependencies)},
    )
    return {
        "status": "success",
        "patient_id": patient_id,
        "dependencies": dependencies,
        "audit_hash": audit_hash,
    }


# ── Tool: Clinical Audit Trail ────────────────────────────────────────────────

@mcp.tool()
def get_clinical_audit_trail(limit: int = 50) -> dict:
    """
    Retrieve the tamper-proof hash-chain audit trail of all clinical decisions.

    Each entry is SHA-256 chained to the previous, providing HIPAA-grade
    tamper detection for clinical decision logging.

    Args:
        limit: Maximum number of audit entries to return (most recent first)
    """
    trail = _engine.get_audit_trail(limit)
    chain_valid = _engine.verify_audit_chain()
    return {
        "status": "success",
        "chain_integrity": "verified" if chain_valid else "TAMPERED",
        "entry_count": len(trail),
        "entries": trail[-limit:],
    }


# ── Tool: Summarize Patient History ───────────────────────────────────────────

@mcp.tool()
def summarize_patient_history(
    patient_id: str,
    fhir_server_url: str = "",
    fhir_access_token: str = "",
) -> dict:
    """
    Generate a condensed patient overview with importance-scored relevance.

    Uses MIND Lang importance kernel to prioritize acute conditions over
    historical ones, and recent observations over older ones.

    Args:
        patient_id: The patient's FHIR ID
        fhir_server_url: FHIR server URL (from SHARP-on-MCP headers)
        fhir_access_token: FHIR access token (from SHARP-on-MCP headers)
    """
    _auto_ingest(patient_id, fhir_server_url, fhir_access_token)
    summary = _engine.patient_summary(patient_id)
    return {"status": "success", **summary}


# ── Tool: Detect Belief Drift ─────────────────────────────────────────────────

@mcp.tool()
def detect_belief_drift(
    patient_id: str,
    fhir_server_url: str = "",
    fhir_access_token: str = "",
) -> dict:
    """
    Comprehensive clinical contradiction scanner.

    Detects 5 types of contradictions:
    1. Allergy-medication conflicts (e.g., Penicillin allergy + Amoxicillin prescription)
    2. Drug-drug interactions (e.g., Warfarin + NSAID bleeding risk)
    3. Lab-medication contraindications (e.g., declining GFR + Metformin)
    4. Lab trend alerts (e.g., GFR declining trajectory approaching danger threshold)
    5. Provider disagreements (e.g., conflicting BP targets from different specialists)

    Each finding includes severity level and actionable clinical recommendation.

    Args:
        patient_id: The patient's FHIR ID
        fhir_server_url: FHIR server URL (from SHARP-on-MCP headers)
        fhir_access_token: FHIR access token (from SHARP-on-MCP headers)
    """
    _auto_ingest(patient_id, fhir_server_url, fhir_access_token)
    contradictions = _engine.detect_contradictions(patient_id)

    critical = [c for c in contradictions if c["severity"] == "critical"]
    high = [c for c in contradictions if c["severity"] == "high"]

    escalation = None
    if critical:
        escalation = (
            "IMMEDIATE CLINICAL REVIEW REQUIRED. "
            f"{len(critical)} critical finding(s) detected that may pose imminent patient safety risk. "
            "Recommend human-in-the-loop verification before any clinical action."
        )
    elif high:
        escalation = (
            f"PRIORITY REVIEW RECOMMENDED. {len(high)} high-severity finding(s) detected. "
            "Schedule provider review within 24-48 hours."
        )

    return {
        "status": "success",
        "patient_id": patient_id,
        "contradictions": contradictions,
        "contradiction_count": len(contradictions),
        "critical_count": len(critical),
        "high_count": len(high),
        "types_found": list({c["type"] for c in contradictions}),
        "has_critical": bool(critical),
        "escalation": escalation,
    }


# ── Tool: Ingest FHIR Data ───────────────────────────────────────────────────

@mcp.tool()
def ingest_patient_data(
    patient_id: str,
    fhir_server_url: str,
    fhir_access_token: str,
) -> dict:
    """
    Pull and store all available FHIR data for a patient into clinical memory.

    Ingests medications, conditions, allergies, vital signs, and lab results.
    This creates the memory foundation that all other tools query against.

    Args:
        patient_id: The patient's FHIR ID
        fhir_server_url: The FHIR R4 server URL
        fhir_access_token: Bearer token for FHIR server auth
    """
    try:
        ctx = FHIRContext(url=fhir_server_url, token=fhir_access_token, patient_id=patient_id)
        fhir = FHIRClient(ctx)
        counts = _engine.ingest_from_fhir(fhir)
        total = sum(counts.values())
        return {
            "status": "success",
            "patient_id": patient_id,
            "ingested": counts,
            "total_blocks": total,
            "message": f"Ingested {total} clinical records for patient {patient_id}",
        }
    except FHIRClientError as e:
        return {"status": "error", "error_message": str(e)}
    except Exception as e:
        return {"status": "error", "error_message": f"Ingestion failed: {e}"}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _auto_ingest(patient_id: str, fhir_url: str, fhir_token: str) -> None:
    """Auto-ingest from FHIR if we have credentials and no stored data."""
    if not fhir_url or not fhir_token:
        return
    if patient_id in _engine._patient_blocks and _engine._patient_blocks[patient_id]:
        return
    try:
        ctx = FHIRContext(url=fhir_url, token=fhir_token, patient_id=patient_id)
        fhir = FHIRClient(ctx)
        _engine.ingest_from_fhir(fhir)
    except Exception as e:
        logger.warning("Auto-ingest failed for %s: %s", patient_id, e)


# Common condition-medication associations for dependency mapping
_CONDITION_MED_MAP: dict[str, list[str]] = {
    "diabetes": ["metformin", "insulin", "glipizide", "sitagliptin"],
    "hypertension": ["lisinopril", "amlodipine", "losartan", "metoprolol", "hydrochlorothiazide"],
    "atrial fibrillation": ["warfarin", "apixaban", "rivaroxaban", "metoprolol", "diltiazem"],
    "chronic kidney": ["lisinopril", "losartan", "furosemide"],
    "hyperlipidemia": ["atorvastatin", "simvastatin", "rosuvastatin"],
    "heart failure": ["metoprolol", "lisinopril", "furosemide", "spironolactone"],
    "depression": ["fluoxetine", "sertraline", "escitalopram"],
    "anxiety": ["sertraline", "buspirone", "escitalopram"],
    "pain": ["acetaminophen", "ibuprofen", "naproxen", "tramadol"],
    "infection": ["amoxicillin", "azithromycin", "ciprofloxacin", "doxycycline"],
}


def _is_related(condition: str, medication: str) -> bool:
    """Check if a condition and medication are commonly associated."""
    for cond_key, med_list in _CONDITION_MED_MAP.items():
        if cond_key in condition:
            if any(m in medication for m in med_list):
                return True
    return False


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    logger.info("Starting ClinicalMem MCP Server on port %d", port)
    mcp.run(transport="streamable-http", host="0.0.0.0", port=port)
