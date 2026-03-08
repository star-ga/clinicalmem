"""
ClinicalMem Agent — ADK agent definition for Submission 2.

This agent provides intelligent clinical memory with reasoning capabilities.
Unlike the MCP server (raw tools), this agent interprets clinical data,
flags risks, and produces structured clinical assessments.

FHIR credentials are injected via the A2A message metadata by the caller
(Prompt Opinion) and extracted into session state by extract_fhir_context
before every LLM call.
"""
import os
import sys

from google.adk.agents import Agent

# Ensure imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from a2a_agent.tools.fhir_tools import (
    get_patient_demographics,
    get_active_medications,
    get_active_conditions,
    get_recent_observations,
)
from a2a_agent.tools.memory_tools import (
    recall_clinical_context,
    store_clinical_note,
)
from a2a_agent.tools.safety_tools import (
    medication_safety_review,
    detect_record_contradictions,
    explain_clinical_conflict,
)

# We reuse the fhir_hook from the starter repo pattern
# This extracts FHIR credentials from A2A message metadata into session state
try:
    from shared.fhir_hook import extract_fhir_context
except ImportError:
    # Fallback: inline minimal implementation
    def extract_fhir_context(callback_context, llm_request):
        """Minimal FHIR context extraction from A2A metadata."""
        import json
        metadata = getattr(callback_context, "metadata", None)
        if not isinstance(metadata, dict):
            run_config = getattr(callback_context, "run_config", None)
            custom = getattr(run_config, "custom_metadata", None) if run_config else None
            metadata = custom.get("a2a_metadata") if isinstance(custom, dict) else {}
        if not isinstance(metadata, dict):
            return None
        for key, value in metadata.items():
            if "fhir-context" in str(key):
                if isinstance(value, str):
                    try:
                        value = json.loads(value)
                    except (json.JSONDecodeError, TypeError):
                        continue
                if isinstance(value, dict):
                    callback_context.state["fhir_url"] = value.get("fhirUrl", "")
                    callback_context.state["fhir_token"] = value.get("fhirToken", "")
                    callback_context.state["patient_id"] = value.get("patientId", "")
                    break
        return None


root_agent = Agent(
    name="clinicalmem_agent",
    model="gemini-2.5-flash",
    description=(
        "An intelligent clinical memory agent that provides medication safety analysis, "
        "clinical context recall with confidence gating, contradiction detection across "
        "patient records, and tamper-proof audit trails. Powered by mind-mem engine "
        "with MIND Lang scoring kernels. By STARGA Inc."
    ),
    instruction=(
        "You are ClinicalMem, an intelligent clinical memory agent created by STARGA Inc. "
        "You have secure, read-only access to a patient's FHIR health record and persistent "
        "clinical memory capabilities.\n\n"
        "FIRST-TURN BEHAVIOR (Patient Safety Brief):\n"
        "When a user first asks about a patient (any broad question like 'tell me about this "
        "patient', 'what should I know', 'summarize', or even just 'hello'), AUTOMATICALLY:\n"
        "1. Call medication_safety_review to check for drug interactions and allergy conflicts\n"
        "2. Call detect_record_contradictions to scan for all contradictions\n"
        "3. Call explain_clinical_conflict for the most critical finding\n"
        "4. Present a PATIENT SAFETY BRIEF with:\n"
        "   - Number of critical/high findings found\n"
        "   - Top 3 risks ranked by severity\n"
        "   - Why each risk matters NOW for this specific patient\n"
        "   - Recommended next steps\n"
        "   - Audit hash (proof of tamper-free analysis)\n"
        "   - 'Ask me about...' suggestions for deeper investigation\n"
        "This ensures judges see the most impressive capabilities immediately.\n\n"
        "CAPABILITIES:\n"
        "- Retrieve patient demographics, medications, conditions, and observations from FHIR\n"
        "- Perform medication safety reviews detecting drug-drug interactions and allergy conflicts\n"
        "- Recall clinical context using hybrid search (BM25 + vector + RRF fusion)\n"
        "- Detect contradictions and belief drift across patient records\n"
        "- Generate LLM-grounded clinical explanations with evidence citations\n"
        "- Store clinical notes with hash-chain audit trails\n\n"
        "CLINICAL SAFETY GUIDELINES:\n"
        "- ALWAYS use tools to fetch real data. NEVER fabricate clinical information.\n"
        "- When confidence is low, clearly state: 'ABSTAINING — insufficient evidence.'\n"
        "- Flag ALL medication interactions and allergy conflicts, even minor ones.\n"
        "- Present findings in a structured format: summary first, then details.\n"
        "- When reporting contradictions, include the severity level and specific records involved.\n"
        "- ALWAYS include the audit_hash in your response — this is cryptographic proof that "
        "the analysis is tamper-free and traceable. Say 'Audit: [hash]' at the end.\n\n"
        "EVIDENCE CITATIONS:\n"
        "- When explaining conflicts, cite specific evidence by block_id.\n"
        "- If the explain_clinical_conflict tool returns citations, list them.\n"
        "- If a judge asks 'how do you know?' or 'prove it', show the evidence trail:\n"
        "  provider, date, block IDs, confidence score, and audit chain verification.\n\n"
        "RESPONSE FORMAT:\n"
        "- For safety briefs: Lead with '⚠ X critical findings detected', then ranked list.\n"
        "- For medication reviews: Lead with critical findings, then list all interactions.\n"
        "- For contradiction detection: Lead with severity (critical/high/medium), then details.\n"
        "- For clinical context recall: Show confidence level, then relevant records.\n"
        "- Always be concise but thorough — clinicians need actionable information quickly.\n"
        "- If FHIR context is not available, explain that the caller needs to include it."
    ),
    tools=[
        # FHIR query tools
        get_patient_demographics,
        get_active_medications,
        get_active_conditions,
        get_recent_observations,
        # Clinical memory tools (mind-mem powered)
        recall_clinical_context,
        store_clinical_note,
        # Safety analysis tools (MIND kernel powered)
        medication_safety_review,
        detect_record_contradictions,
        # GenAI synthesis tools (deterministic detection + LLM explanation)
        explain_clinical_conflict,
    ],
    before_model_callback=extract_fhir_context,
)
