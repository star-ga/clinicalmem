"""
Clinical memory tools — persistent memory operations for the ClinicalMem agent.

These tools wrap the ClinicalMem engine to provide intelligent clinical
context recall with MIND-kernel confidence gating and importance scoring.
"""
import logging
import os
import sys
import time

from google.adk.tools import ToolContext

# Ensure engine is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from engine.clinical_memory import ClinicalMemEngine, ClinicalBlock
from engine.fhir_client import FHIRClient, FHIRContext

logger = logging.getLogger(__name__)

# Shared engine instance
_engine = ClinicalMemEngine()


def _get_engine() -> ClinicalMemEngine:
    return _engine


def _auto_ingest(tool_context: ToolContext) -> None:
    """Auto-ingest FHIR data if credentials are available and no data stored."""
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
        logger.info("Auto-ingested FHIR data for patient %s", patient_id)
    except Exception as e:
        logger.warning("Auto-ingest failed for %s: %s", patient_id, e)


def recall_clinical_context(query: str, tool_context: ToolContext, top_k: int = 5) -> dict:
    """
    Retrieve relevant patient clinical history using hybrid search.

    Uses BM25 + vector + RRF fusion with MIND Lang abstention kernel for
    confidence gating. Returns scored results with a confidence assessment
    indicating whether there is sufficient evidence for a clinical answer.

    Args:
        query: Natural language clinical question (e.g., "medication history",
               "recent lab results", "allergy information")
        top_k: Maximum number of results to return (default: 10)
    """
    _auto_ingest(tool_context)
    patient_id = tool_context.state.get("patient_id", "")
    if not patient_id:
        return {"status": "error", "error_message": "No patient_id in session context."}

    engine = _get_engine()
    result = engine.recall(patient_id, query, top_k=top_k or 10)

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


def store_clinical_note(
    title: str, content: str, observation_type: str, tool_context: ToolContext
) -> dict:
    """
    Store a clinical note or observation in persistent patient memory.

    All stored observations are hash-chain audited and available for
    future recall queries.

    Args:
        title: Brief title for the clinical note
        content: Full text of the clinical observation or note
        observation_type: Type of observation — 'clinical_note', 'lab_result',
                         'medication_change', 'assessment', 'plan'
    """
    patient_id = tool_context.state.get("patient_id", "")
    if not patient_id:
        return {"status": "error", "error_message": "No patient_id in session context."}

    engine = _get_engine()
    block = ClinicalBlock(
        block_id=f"note-{int(time.time())}-{hash(content) % 10000}",
        patient_id=patient_id,
        resource_type=observation_type or "clinical_note",
        title=title,
        content=content,
        metadata={"type": observation_type, "source": "agent"},
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        source="ClinicalMem Agent",
    )
    engine._store_block(block)
    audit_hash = engine._append_audit(
        "store_note",
        {"patient_id": patient_id, "title": title, "type": observation_type},
    )
    return {
        "status": "success",
        "block_id": block.block_id,
        "audit_hash": audit_hash,
        "message": f"Stored '{title}' for patient {patient_id}",
    }
