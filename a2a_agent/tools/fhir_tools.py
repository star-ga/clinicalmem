"""
FHIR query tools — direct FHIR R4 access for the ClinicalMem agent.

Reuses the same FHIR access pattern as the po-adk-python starter:
- FHIR credentials come from tool_context.state (injected by fhir_hook)
- httpx for HTTP calls
- Error handling returns structured dicts
"""
import logging

import httpx
from google.adk.tools import ToolContext

logger = logging.getLogger(__name__)

_FHIR_TIMEOUT = 15


def _get_fhir_context(tool_context: ToolContext):
    fhir_url = tool_context.state.get("fhir_url", "").rstrip("/")
    fhir_token = tool_context.state.get("fhir_token", "")
    patient_id = tool_context.state.get("patient_id", "")
    missing = [
        name for name, val in [
            ("fhir_url", fhir_url),
            ("fhir_token", fhir_token),
            ("patient_id", patient_id),
        ]
        if not val
    ]
    if missing:
        return {
            "status": "error",
            "error_message": f"FHIR context missing: {', '.join(missing)}. "
            "Ensure the caller includes 'fhir-context' in the A2A message metadata.",
        }
    return fhir_url, fhir_token, patient_id


def _fhir_get(fhir_url: str, token: str, path: str, params: dict | None = None) -> dict:
    resp = httpx.get(
        f"{fhir_url}/{path}",
        params=params,
        headers={"Authorization": f"Bearer {token}", "Accept": "application/fhir+json"},
        timeout=_FHIR_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()


def _coding_display(codings: list) -> str:
    for c in codings:
        if c.get("display"):
            return c["display"]
    return "Unknown"


def get_patient_demographics(tool_context: ToolContext) -> dict:
    """Fetch patient demographics (name, DOB, gender, contacts) from the FHIR server."""
    ctx = _get_fhir_context(tool_context)
    if isinstance(ctx, dict):
        return ctx
    fhir_url, fhir_token, patient_id = ctx
    try:
        patient = _fhir_get(fhir_url, fhir_token, f"Patient/{patient_id}")
    except Exception as e:
        return {"status": "error", "error_message": str(e)}

    names = patient.get("name", [])
    official = next((n for n in names if n.get("use") == "official"), names[0] if names else {})
    given = " ".join(official.get("given", []))
    family = official.get("family", "")
    return {
        "status": "success",
        "patient_id": patient_id,
        "name": f"{given} {family}".strip() or "Unknown",
        "birth_date": patient.get("birthDate"),
        "gender": patient.get("gender"),
    }


def get_active_medications(tool_context: ToolContext) -> dict:
    """Retrieve the patient's current active medication list from the FHIR server."""
    ctx = _get_fhir_context(tool_context)
    if isinstance(ctx, dict):
        return ctx
    fhir_url, fhir_token, patient_id = ctx
    try:
        bundle = _fhir_get(
            fhir_url, fhir_token, "MedicationRequest",
            {"patient": patient_id, "status": "active", "_count": "50"},
        )
    except Exception as e:
        return {"status": "error", "error_message": str(e)}

    meds = []
    for entry in bundle.get("entry", []):
        res = entry.get("resource", {})
        concept = res.get("medicationCodeableConcept", {})
        name = concept.get("text") or _coding_display(concept.get("coding", []))
        dosage_list = [d.get("text", "") for d in res.get("dosageInstruction", [])]
        meds.append({
            "medication": name,
            "dosage": dosage_list[0] if dosage_list else "Not specified",
            "authored_on": res.get("authoredOn"),
            "requester": (res.get("requester") or {}).get("display"),
        })
    return {"status": "success", "patient_id": patient_id, "count": len(meds), "medications": meds}


def get_active_conditions(tool_context: ToolContext) -> dict:
    """Retrieve the patient's active conditions and diagnoses from the FHIR server."""
    ctx = _get_fhir_context(tool_context)
    if isinstance(ctx, dict):
        return ctx
    fhir_url, fhir_token, patient_id = ctx
    try:
        bundle = _fhir_get(
            fhir_url, fhir_token, "Condition",
            {"patient": patient_id, "clinical-status": "active", "_count": "50"},
        )
    except Exception as e:
        return {"status": "error", "error_message": str(e)}

    conditions = []
    for entry in bundle.get("entry", []):
        res = entry.get("resource", {})
        code = res.get("code", {})
        conditions.append({
            "condition": code.get("text") or _coding_display(code.get("coding", [])),
            "severity": (res.get("severity") or {}).get("text"),
            "onset": res.get("onsetDateTime") or (res.get("onsetPeriod") or {}).get("start"),
        })
    return {"status": "success", "patient_id": patient_id, "count": len(conditions), "conditions": conditions}


def get_recent_observations(category: str, tool_context: ToolContext) -> dict:
    """
    Retrieve recent clinical observations (vitals, labs, social history).

    Args:
        category: FHIR observation category — 'vital-signs', 'laboratory', or 'social-history'
    """
    ctx = _get_fhir_context(tool_context)
    if isinstance(ctx, dict):
        return ctx
    fhir_url, fhir_token, patient_id = ctx
    category = (category or "vital-signs").strip().lower()
    try:
        bundle = _fhir_get(
            fhir_url, fhir_token, "Observation",
            {"patient": patient_id, "category": category, "_sort": "-date", "_count": "20"},
        )
    except Exception as e:
        return {"status": "error", "error_message": str(e)}

    observations = []
    for entry in bundle.get("entry", []):
        res = entry.get("resource", {})
        code = res.get("code", {})
        value, unit = None, None
        if "valueQuantity" in res:
            vq = res["valueQuantity"]
            value, unit = vq.get("value"), vq.get("unit")
        elif "valueString" in res:
            value = res["valueString"]
        observations.append({
            "observation": code.get("text") or _coding_display(code.get("coding", [])),
            "value": value,
            "unit": unit,
            "effective_date": res.get("effectiveDateTime"),
        })
    return {"status": "success", "patient_id": patient_id, "category": category, "observations": observations}
