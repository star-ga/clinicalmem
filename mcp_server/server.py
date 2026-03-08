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
import time
from collections import defaultdict

from fastmcp import FastMCP

# Ensure engine is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.clinical_memory import ClinicalMemEngine
from engine.fhir_client import FHIRClient, FHIRContext, FHIRClientError

logger = logging.getLogger(__name__)

# ── Global state ──────────────────────────────────────────────────────────────

_engine = ClinicalMemEngine()

# ── Rate limiter (60 requests/minute per tool) ──────────────────────────────

_rate_limit_window: dict[str, list[float]] = defaultdict(list)
_RATE_LIMIT_MAX = 60
_RATE_LIMIT_WINDOW = 60.0


def _check_rate_limit(tool_name: str) -> None:
    """Raise ValueError if rate limit exceeded."""
    now = time.monotonic()
    cutoff = now - _RATE_LIMIT_WINDOW
    window = [t for t in _rate_limit_window[tool_name] if t > cutoff]
    if len(window) >= _RATE_LIMIT_MAX:
        raise ValueError(f"Rate limit exceeded for {tool_name} (max {_RATE_LIMIT_MAX}/min)")
    window.append(now)
    _rate_limit_window[tool_name] = window

# Demo mode: pre-load Sarah Mitchell fixture data
if os.environ.get("DEMO_MODE", "").lower() in ("1", "true", "yes"):
    try:
        _fixture_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "tests", "fixtures", "sarah_mitchell_bundle.json",
        )
        if os.path.exists(_fixture_path):
            with open(_fixture_path) as f:
                _bundle = json.load(f)
            _counts = _engine.ingest_from_bundle(_bundle, "patient-sarah-mitchell")
            logger.info("DEMO MODE: Pre-loaded Sarah Mitchell (%s)", _counts)
    except Exception as e:
        logger.warning("Demo auto-load failed: %s", e)

# ── MCP Server ────────────────────────────────────────────────────────────────

mcp = FastMCP(
    "ClinicalMem",
    instructions=(
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


# ── Tool: Explain Clinical Conflict (GenAI Synthesis) ────────────────────────

@mcp.tool()
def explain_clinical_conflict(
    patient_id: str,
    conflict_index: int = 0,
    fhir_server_url: str = "",
    fhir_access_token: str = "",
) -> dict:
    """
    Generate a patient-specific LLM explanation for a detected clinical conflict.

    Uses deterministic detection + GenAI synthesis pattern:
    - Detection: rule-based safety rails (reliable, auditable)
    - Explanation: LLM-generated (expressive, context-aware, with evidence citations)
    - Abstention: hard gate when evidence is insufficient — refuses to guess

    This demonstrates ClinicalMem's approach to trustworthy clinical AI:
    deterministic safety catches the problem, GenAI explains it in context.

    Args:
        patient_id: The patient's FHIR ID
        conflict_index: Which detected conflict to explain (0-based index)
        fhir_server_url: FHIR server URL (from SHARP-on-MCP headers)
        fhir_access_token: FHIR access token (from SHARP-on-MCP headers)
    """
    _auto_ingest(patient_id, fhir_server_url, fhir_access_token)
    narrative = _engine.explain_clinical_conflict(patient_id, conflict_index)
    return {
        "status": "success",
        "patient_id": patient_id,
        "narrative": narrative.narrative,
        "evidence_citations": narrative.evidence_citations,
        "confidence_score": round(narrative.confidence_score, 3),
        "abstained": narrative.abstained,
        "model_used": narrative.model_used,
    }


# ── Tool: Clinical Care Handoff (GenAI Synthesis) ───────────────────────────

@mcp.tool()
def clinical_care_handoff(
    patient_id: str,
    fhir_server_url: str = "",
    fhir_access_token: str = "",
) -> dict:
    """
    Generate a complete clinical care handoff note using GenAI synthesis.

    Combines all detected safety findings into a structured clinician-ready
    note with evidence citations. Uses LLM to synthesize findings into
    natural clinical language while maintaining traceability to source records.

    Demonstrates:
    - Evidence-grounded generation (every claim cites a source block)
    - Safe abstention (refuses when data is insufficient)
    - Deterministic detection + GenAI explanation pattern

    Args:
        patient_id: The patient's FHIR ID
        fhir_server_url: FHIR server URL (from SHARP-on-MCP headers)
        fhir_access_token: FHIR access token (from SHARP-on-MCP headers)
    """
    _auto_ingest(patient_id, fhir_server_url, fhir_access_token)
    narrative = _engine.clinical_handoff(patient_id)
    return {
        "status": "success",
        "patient_id": patient_id,
        "handoff_note": narrative.narrative,
        "evidence_citations": narrative.evidence_citations,
        "confidence_score": round(narrative.confidence_score, 3),
        "abstained": narrative.abstained,
        "model_used": narrative.model_used,
    }


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
