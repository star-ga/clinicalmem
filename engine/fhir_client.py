"""
FHIR R4 client — query patient resources from a FHIR server.

Wraps httpx with Bearer token auth and FHIR JSON accept headers.
All methods return parsed dicts; errors raise FHIRClientError.
"""
import logging
from dataclasses import dataclass
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_FHIR_TIMEOUT = 15  # seconds


class FHIRClientError(Exception):
    """Raised when a FHIR request fails."""

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


@dataclass(frozen=True)
class FHIRContext:
    """Immutable FHIR connection context."""

    url: str
    token: str
    patient_id: str

    def validate(self) -> None:
        missing = []
        if not self.url:
            missing.append("url")
        if not self.token:
            missing.append("token")
        if not self.patient_id:
            missing.append("patient_id")
        if missing:
            raise FHIRClientError(f"FHIR context missing: {', '.join(missing)}")


class FHIRClient:
    """Stateless FHIR R4 REST client."""

    def __init__(self, ctx: FHIRContext):
        ctx.validate()
        self._url = ctx.url.rstrip("/")
        self._headers = {
            "Authorization": f"Bearer {ctx.token}",
            "Accept": "application/fhir+json",
        }
        self._patient_id = ctx.patient_id

    @property
    def patient_id(self) -> str:
        return self._patient_id

    def _get(self, path: str, params: dict[str, str] | None = None) -> dict:
        resp = httpx.get(
            f"{self._url}/{path}",
            params=params,
            headers=self._headers,
            timeout=_FHIR_TIMEOUT,
        )
        if resp.status_code >= 400:
            raise FHIRClientError(
                f"FHIR HTTP {resp.status_code}: {resp.text[:200]}",
                status_code=resp.status_code,
            )
        return resp.json()

    def get_patient(self) -> dict:
        return self._get(f"Patient/{self._patient_id}")

    def get_medications(self, status: str = "active") -> list[dict]:
        bundle = self._get(
            "MedicationRequest",
            {"patient": self._patient_id, "status": status, "_count": "100"},
        )
        return [e["resource"] for e in bundle.get("entry", []) if "resource" in e]

    def get_conditions(self, clinical_status: str = "active") -> list[dict]:
        bundle = self._get(
            "Condition",
            {"patient": self._patient_id, "clinical-status": clinical_status, "_count": "100"},
        )
        return [e["resource"] for e in bundle.get("entry", []) if "resource" in e]

    def get_allergies(self) -> list[dict]:
        bundle = self._get(
            "AllergyIntolerance",
            {"patient": self._patient_id, "_count": "100"},
        )
        return [e["resource"] for e in bundle.get("entry", []) if "resource" in e]

    def get_observations(
        self, category: str = "vital-signs", count: int = 20
    ) -> list[dict]:
        bundle = self._get(
            "Observation",
            {
                "patient": self._patient_id,
                "category": category,
                "_sort": "-date",
                "_count": str(count),
            },
        )
        return [e["resource"] for e in bundle.get("entry", []) if "resource" in e]

    def get_encounters(self, count: int = 20) -> list[dict]:
        bundle = self._get(
            "Encounter",
            {"patient": self._patient_id, "_sort": "-date", "_count": str(count)},
        )
        return [e["resource"] for e in bundle.get("entry", []) if "resource" in e]


def coding_display(codings: list[dict[str, Any]]) -> str:
    """Return the first human-readable display text from FHIR codings."""
    for c in codings:
        if c.get("display"):
            return c["display"]
    return "Unknown"


def extract_medication_name(resource: dict) -> str:
    """Extract human-readable medication name from a MedicationRequest."""
    concept = resource.get("medicationCodeableConcept", {})
    return (
        concept.get("text")
        or coding_display(concept.get("coding", []))
        or resource.get("medicationReference", {}).get("display", "Unknown")
    )


def extract_condition_name(resource: dict) -> str:
    """Extract human-readable condition name from a Condition resource."""
    code = resource.get("code", {})
    return code.get("text") or coding_display(code.get("coding", []))
