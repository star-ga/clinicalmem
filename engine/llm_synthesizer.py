"""
LLM Clinical Synthesizer — evidence-grounded clinical narrative generation.

Uses GenAI to explain detected conflicts in the patient's specific clinical
context, with explicit evidence citations and hard abstention when confidence
is low.  Deterministic safety rails (drug interactions, allergy checks) handle
detection; the LLM handles *synthesis and explanation*.

Falls back to structured template output when no LLM API key is available.
"""
import json
import logging
import os
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ClinicalNarrative:
    """LLM-generated clinical narrative with evidence citations."""

    narrative: str
    evidence_citations: list[dict[str, str]]
    confidence_score: float
    abstained: bool
    model_used: str
    audit_context: dict[str, Any]


# System prompt anchors the LLM to evidence and enforces abstention
_SYSTEM_PROMPT = """You are a clinical decision support assistant. You MUST:

1. ONLY use the evidence provided below. Never fabricate clinical facts.
2. CITE specific evidence blocks by their block_id for every claim you make.
3. If the evidence is insufficient to make a clinical statement, respond with
   exactly: "ABSTAIN: Insufficient evidence to provide clinical guidance."
4. Use professional clinical language appropriate for a care handoff note.
5. Include severity assessment and recommended actions.
6. Never provide a diagnosis — only flag safety concerns based on evidence.

Format citations as [block_id] inline."""


def _build_conflict_prompt(
    conflict: dict[str, Any],
    patient_context: dict[str, Any],
    evidence_blocks: list[dict[str, Any]],
) -> str:
    """Build a prompt for explaining a specific conflict in patient context."""
    evidence_text = "\n".join(
        f"- [{b.get('block_id', 'unknown')}] {b.get('title', '')}: {b.get('content', '')}"
        for b in evidence_blocks
    )
    patient_info = (
        f"Patient has {len(patient_context.get('conditions', []))} conditions, "
        f"{len(patient_context.get('medications', []))} medications, "
        f"{len(patient_context.get('allergies', []))} allergies."
    )
    conditions_text = ", ".join(
        c.get("name", "") for c in patient_context.get("conditions", [])
    )
    if conditions_text:
        patient_info += f"\nActive conditions: {conditions_text}"

    return f"""Explain this clinical safety concern in the context of this specific patient.

DETECTED CONFLICT:
- Type: {conflict.get('type', 'unknown')}
- Severity: {conflict.get('severity', 'unknown')}
- Description: {conflict.get('description', '')}
- Current recommendation: {conflict.get('recommendation', '')}

PATIENT CONTEXT:
{patient_info}

EVIDENCE BLOCKS:
{evidence_text}

Provide a 2-3 sentence clinical rationale explaining WHY this is dangerous for THIS specific patient, citing evidence block IDs. Then provide a concrete action recommendation."""


def _build_handoff_prompt(
    patient_context: dict[str, Any],
    contradictions: list[dict[str, Any]],
    safety_report: dict[str, Any],
    evidence_blocks: list[dict[str, Any]],
) -> str:
    """Build a prompt for generating a complete care handoff summary."""
    evidence_text = "\n".join(
        f"- [{b.get('block_id', 'unknown')}] {b.get('title', '')}: {b.get('content', '')}"
        for b in evidence_blocks[:20]  # Limit to top 20 blocks
    )
    conflicts_text = "\n".join(
        f"- [{c.get('severity', '?').upper()}] {c.get('description', '')}"
        for c in contradictions
    )
    meds_text = ", ".join(
        m.get("name", "") for m in patient_context.get("medications", [])
    )
    conditions_text = ", ".join(
        c.get("name", "") for c in patient_context.get("conditions", [])
    )

    return f"""Generate a clinical care handoff note for this patient based ONLY on the evidence below.

PATIENT OVERVIEW:
- Medications: {meds_text}
- Conditions: {conditions_text}
- Allergies: {', '.join(a.get('allergen', '') for a in patient_context.get('allergies', []))}

DETECTED SAFETY CONCERNS ({len(contradictions)} total):
{conflicts_text or 'None detected'}

MEDICATION SAFETY:
- Drug interactions: {safety_report.get('interaction_count', 0)}
- Allergy conflicts: {safety_report.get('allergy_conflict_count', 0)}

EVIDENCE BLOCKS:
{evidence_text}

Write a structured care handoff note with sections:
1. CRITICAL ALERTS (if any)
2. ACTIVE MEDICATIONS (with safety flags)
3. ACTIVE CONDITIONS
4. RECOMMENDED ACTIONS
Cite evidence blocks by [block_id] for every clinical claim."""


async def _call_medical_llm_async(prompt: str, system: str) -> tuple[str | None, str]:
    """Async medical LLM cascade: OpenAI GPT-4o → MedGemma → Gemini Flash."""
    openai_key = os.environ.get("OPENAI_API_KEY")
    google_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")

    if not openai_key and not google_key:
        return None, "none"

    try:
        import httpx
    except ImportError:
        return None, "none"

    async with httpx.AsyncClient(timeout=8) as client:
        if openai_key:
            try:
                resp = await client.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={"Authorization": f"Bearer {openai_key}"},
                    json={
                        "model": "gpt-5.4",
                        "messages": [
                            {"role": "system", "content": system},
                            {"role": "user", "content": prompt},
                        ],
                        "temperature": 0.2,
                        "max_tokens": 1024,
                    },
                )
                if resp.status_code == 200:
                    data = resp.json()
                    text = data["choices"][0]["message"]["content"]
                    if text:
                        return text, "OpenAI-GPT-4o"
                else:
                    logger.info("OpenAI returned %d, trying next", resp.status_code)
            except Exception as e:
                logger.info("OpenAI failed: %s, trying next", e)

        if google_key:
            for model_id, model_label in [
                ("medgemma-27b-text-v1", "MedGemma-27B"),
                ("gemini-3-flash", "gemini-3-flash"),
            ]:
                try:
                    resp = await client.post(
                        f"https://generativelanguage.googleapis.com/v1beta/models/{model_id}:generateContent?key={google_key}",
                        json={
                            "systemInstruction": {"parts": [{"text": system}]},
                            "contents": [{"parts": [{"text": prompt}]}],
                            "generationConfig": {
                                "temperature": 0.2,
                                "maxOutputTokens": 1024,
                            },
                        },
                    )
                    if resp.status_code != 200:
                        logger.info("%s returned %d, trying next", model_label, resp.status_code)
                        continue
                    data = resp.json()
                    candidates = data.get("candidates", [])
                    if candidates:
                        parts = candidates[0].get("content", {}).get("parts", [])
                        if parts and parts[0].get("text"):
                            return parts[0]["text"], model_label
                except Exception as e:
                    logger.info("%s failed: %s, trying next", model_label, e)
                    continue

    return None, "none"


def _call_medical_llm_sync(prompt: str, system: str) -> tuple[str | None, str]:
    """
    Call medical LLM with cascade: OpenAI GPT-4o → MedGemma → Gemini Flash.

    Uses whichever API keys are available. OpenAI has the strongest clinical
    validation (260 physicians, HIPAA BAA). MedGemma is purpose-built for
    medicine (87.7% MedQA). Gemini Flash is the general fallback.

    Returns (response_text, model_used).
    """
    openai_key = os.environ.get("OPENAI_API_KEY")
    google_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")

    if not openai_key and not google_key:
        return None, "none"

    try:
        import httpx
    except ImportError:
        return None, "none"

    # Attempt 1: OpenAI (if key available)
    if openai_key:
        try:
            resp = httpx.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {openai_key}"},
                json={
                    "model": "gpt-5.4",
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0.2,
                    "max_tokens": 1024,
                },
                timeout=8,
            )
            if resp.status_code == 200:
                data = resp.json()
                text = data["choices"][0]["message"]["content"]
                if text:
                    return text, "OpenAI-GPT-4o"
            else:
                logger.info("OpenAI returned %d, trying next model", resp.status_code)
        except Exception as e:
            logger.info("OpenAI failed: %s, trying next model", e)

    # Attempt 2-3: Google models (MedGemma → Gemini Flash)
    if google_key:
        google_models = [
            ("medgemma-27b-text-v1", "MedGemma-27B"),
            ("gemini-3-flash", "gemini-3-flash"),
        ]
        for model_id, model_label in google_models:
            try:
                resp = httpx.post(
                    f"https://generativelanguage.googleapis.com/v1beta/models/{model_id}:generateContent?key={google_key}",
                    json={
                        "systemInstruction": {"parts": [{"text": system}]},
                        "contents": [{"parts": [{"text": prompt}]}],
                        "generationConfig": {
                            "temperature": 0.2,
                            "maxOutputTokens": 1024,
                        },
                    },
                    timeout=8,
                )
                if resp.status_code != 200:
                    logger.info("%s returned %d, trying next", model_label, resp.status_code)
                    continue
                data = resp.json()
                candidates = data.get("candidates", [])
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    if parts and parts[0].get("text"):
                        return parts[0]["text"], model_label
            except Exception as e:
                logger.info("%s failed: %s, trying next", model_label, e)
                continue

    return None, "none"


def _template_conflict_explanation(
    conflict: dict[str, Any],
    patient_context: dict[str, Any],
) -> str:
    """Fallback template when LLM is unavailable."""
    conditions = ", ".join(
        c.get("name", "") for c in patient_context.get("conditions", [])
    )
    return (
        f"[{conflict.get('severity', 'unknown').upper()}] {conflict.get('description', '')}. "
        f"This patient has comorbidities ({conditions}) which may increase risk. "
        f"Recommendation: {conflict.get('recommendation', 'Review with prescriber.')}"
    )


def _template_handoff(
    patient_context: dict[str, Any],
    contradictions: list[dict[str, Any]],
) -> str:
    """Fallback handoff template when LLM is unavailable."""
    lines = ["CARE HANDOFF NOTE", "=" * 40]

    critical = [c for c in contradictions if c.get("severity") in ("critical", "high")]
    if critical:
        lines.append("\nCRITICAL ALERTS:")
        for c in critical:
            lines.append(f"  ! [{c['severity'].upper()}] {c['description']}")
            lines.append(f"    → {c.get('recommendation', 'Review required')}")

    lines.append("\nACTIVE MEDICATIONS:")
    for m in patient_context.get("medications", []):
        lines.append(f"  - {m.get('name', 'Unknown')} ({m.get('dosage', '')})")

    lines.append("\nACTIVE CONDITIONS:")
    for c in patient_context.get("conditions", []):
        lines.append(f"  - {c.get('name', 'Unknown')}")

    lines.append("\nALLERGIES:")
    for a in patient_context.get("allergies", []):
        lines.append(f"  - {a.get('allergen', 'Unknown')} ({a.get('criticality', '')})")

    return "\n".join(lines)


def explain_conflict(
    conflict: dict[str, Any],
    patient_context: dict[str, Any],
    evidence_blocks: list[dict[str, Any]],
    confidence_threshold: float = 0.3,
) -> ClinicalNarrative:
    """
    Generate a patient-specific clinical explanation for a detected conflict.

    Uses deterministic detection + LLM synthesis pattern:
    - Detection is rule-based (reliable, auditable)
    - Explanation is LLM-generated (expressive, context-aware)
    - Hard abstention when evidence is insufficient
    """
    # Check confidence — abstain if too low
    evidence_count = len(evidence_blocks)
    confidence = min(evidence_count / 3.0, 1.0)

    if confidence < confidence_threshold:
        return ClinicalNarrative(
            narrative="ABSTAIN: Insufficient evidence to provide clinical guidance. "
            f"Only {evidence_count} evidence block(s) available; "
            f"minimum threshold requires {int(confidence_threshold * 3)} blocks.",
            evidence_citations=[],
            confidence_score=confidence,
            abstained=True,
            model_used="abstention_gate",
            audit_context={
                "reason": "insufficient_evidence",
                "evidence_count": evidence_count,
                "threshold": confidence_threshold,
            },
        )

    # Build prompt
    prompt = _build_conflict_prompt(conflict, patient_context, evidence_blocks)

    # Try LLM synthesis (MedGemma → Gemini cascade)
    llm_response, model_used = _call_medical_llm_sync(prompt, _SYSTEM_PROMPT)

    if llm_response and "ABSTAIN" not in llm_response:
        # Extract citations from response
        import re

        citations = re.findall(r"\[([^\]]+)\]", llm_response)
        evidence_citations = [
            {"block_id": cid, "role": "supporting_evidence"}
            for cid in citations
            if any(b.get("block_id") == cid for b in evidence_blocks)
        ]
        return ClinicalNarrative(
            narrative=llm_response,
            evidence_citations=evidence_citations,
            confidence_score=confidence,
            abstained=False,
            model_used=model_used,
            audit_context={
                "conflict_type": conflict.get("type"),
                "evidence_count": evidence_count,
                "citations_found": len(evidence_citations),
            },
        )
    elif llm_response and "ABSTAIN" in llm_response:
        return ClinicalNarrative(
            narrative=llm_response,
            evidence_citations=[],
            confidence_score=confidence,
            abstained=True,
            model_used=model_used,
            audit_context={"reason": "llm_abstained", "evidence_count": evidence_count},
        )

    # Fallback to template
    narrative = _template_conflict_explanation(conflict, patient_context)
    return ClinicalNarrative(
        narrative=narrative,
        evidence_citations=[],
        confidence_score=confidence,
        abstained=False,
        model_used="template_fallback",
        audit_context={"reason": "llm_unavailable", "evidence_count": evidence_count},
    )


def generate_clinical_handoff(
    patient_context: dict[str, Any],
    contradictions: list[dict[str, Any]],
    safety_report: dict[str, Any],
    evidence_blocks: list[dict[str, Any]],
    confidence_threshold: float = 0.3,
) -> ClinicalNarrative:
    """
    Generate a complete clinical care handoff note.

    Synthesizes all detected findings into a structured clinician-ready note
    with evidence citations. Abstains if evidence is insufficient.
    """
    evidence_count = len(evidence_blocks)
    confidence = min(evidence_count / 5.0, 1.0)

    if confidence < confidence_threshold:
        return ClinicalNarrative(
            narrative="ABSTAIN: Insufficient clinical data for a safe handoff summary. "
            f"Only {evidence_count} evidence block(s) available.",
            evidence_citations=[],
            confidence_score=confidence,
            abstained=True,
            model_used="abstention_gate",
            audit_context={
                "reason": "insufficient_evidence",
                "evidence_count": evidence_count,
            },
        )

    prompt = _build_handoff_prompt(
        patient_context, contradictions, safety_report, evidence_blocks
    )

    llm_response, model_used = _call_medical_llm_sync(prompt, _SYSTEM_PROMPT)

    if llm_response and "ABSTAIN" not in llm_response:
        import re

        citations = re.findall(r"\[([^\]]+)\]", llm_response)
        evidence_citations = [
            {"block_id": cid, "role": "supporting_evidence"}
            for cid in citations
            if any(b.get("block_id") == cid for b in evidence_blocks)
        ]
        return ClinicalNarrative(
            narrative=llm_response,
            evidence_citations=evidence_citations,
            confidence_score=confidence,
            abstained=False,
            model_used=model_used,
            audit_context={
                "contradiction_count": len(contradictions),
                "evidence_count": evidence_count,
                "citations_found": len(evidence_citations),
            },
        )
    elif llm_response and "ABSTAIN" in llm_response:
        return ClinicalNarrative(
            narrative=llm_response,
            evidence_citations=[],
            confidence_score=confidence,
            abstained=True,
            model_used=model_used,
            audit_context={"reason": "llm_abstained"},
        )

    # Fallback to template
    narrative = _template_handoff(patient_context, contradictions)
    return ClinicalNarrative(
        narrative=narrative,
        evidence_citations=[],
        confidence_score=confidence,
        abstained=False,
        model_used="template_fallback",
        audit_context={"reason": "llm_unavailable", "evidence_count": evidence_count},
    )
