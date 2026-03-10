"""Tests for engine/fhir_client.py — SSRF protection, FHIR operations, helpers."""
import pytest
from unittest.mock import patch, MagicMock

from engine.fhir_client import (
    FHIRContext,
    FHIRClient,
    FHIRClientError,
    BundleFHIRClient,
    coding_display,
    extract_medication_name,
)


# ── FHIRContext validation ─────────────────────────────────────────────────────

class TestFHIRContextValidation:
    def test_missing_patient_id(self):
        """Line 41: patient_id missing raises on validate()."""
        ctx = FHIRContext(url="https://fhir.test/r4", token="tok", patient_id="")
        with pytest.raises(FHIRClientError, match="patient_id"):
            ctx.validate()

    def test_missing_url(self):
        ctx = FHIRContext(url="", token="tok", patient_id="p-1")
        with pytest.raises(FHIRClientError, match="url"):
            ctx.validate()

    def test_missing_token(self):
        ctx = FHIRContext(url="https://fhir.test/r4", token="", patient_id="p-1")
        with pytest.raises(FHIRClientError, match="token"):
            ctx.validate()

    def test_invalid_scheme(self):
        ctx = FHIRContext(url="ftp://fhir.test/r4", token="tok", patient_id="p-1")
        with pytest.raises(FHIRClientError, match="scheme"):
            ctx.validate()

    def test_localhost_blocked(self):
        ctx = FHIRContext(url="https://localhost/r4", token="tok", patient_id="p-1")
        with pytest.raises(FHIRClientError, match="localhost"):
            ctx.validate()

    def test_private_ip_blocked(self):
        ctx = FHIRContext(url="https://192.168.1.1/r4", token="tok", patient_id="p-1")
        with pytest.raises(FHIRClientError, match="private"):
            ctx.validate()

    def test_ipv6_private_blocked(self):
        """Lines 69-71: IPv6 in URL path candidate extraction."""
        ctx = FHIRContext(url="https://[::1]/r4", token="tok", patient_id="p-1")
        with pytest.raises(FHIRClientError):
            ctx.validate()

    def test_ipv6_in_path(self):
        """Lines 68-71: IPv6 address mishandled in URL path."""
        ctx = FHIRContext(url="https://fc00::1/r4", token="tok", patient_id="p-1")
        with pytest.raises(FHIRClientError):
            ctx.validate()

    def test_valid_context(self):
        ctx = FHIRContext(url="https://fhir.example.com/r4", token="tok", patient_id="p-1")
        ctx.validate()  # should not raise
        assert ctx.patient_id == "p-1"


# ── FHIRClient._get HTTP errors ───────────────────────────────────────────────

class TestFHIRClientGet:
    @patch("httpx.get")
    def test_http_400_raises(self, mock_get):
        """Line 106: status >= 400 raises FHIRClientError."""
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        mock_resp.text = "Not Found"
        mock_get.return_value = mock_resp

        ctx = FHIRContext(url="https://fhir.example.com/r4", token="tok", patient_id="p-1")
        client = FHIRClient(ctx)
        with pytest.raises(FHIRClientError, match="404"):
            client.get_patient()

    @patch("httpx.get")
    def test_get_medications(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "entry": [{"resource": {"resourceType": "MedicationRequest", "id": "m1"}}]
        }
        mock_get.return_value = mock_resp

        ctx = FHIRContext(url="https://fhir.example.com/r4", token="tok", patient_id="p-1")
        client = FHIRClient(ctx)
        meds = client.get_medications()
        assert len(meds) == 1

    @patch("httpx.get")
    def test_get_conditions(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"entry": []}
        mock_get.return_value = mock_resp

        ctx = FHIRContext(url="https://fhir.example.com/r4", token="tok", patient_id="p-1")
        client = FHIRClient(ctx)
        assert client.get_conditions() == []

    @patch("httpx.get")
    def test_get_encounters(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "entry": [{"resource": {"resourceType": "Encounter", "id": "e1"}}]
        }
        mock_get.return_value = mock_resp

        ctx = FHIRContext(url="https://fhir.example.com/r4", token="tok", patient_id="p-1")
        client = FHIRClient(ctx)
        assert len(client.get_encounters()) == 1


# ── BundleFHIRClient ─────────────────────────────────────────────────────────

class TestBundleFHIRClient:
    def test_get_patient_with_data(self):
        """Line 174: BundleFHIRClient.get_patient returns first patient."""
        client = BundleFHIRClient(
            {"Patient": [{"id": "p-1", "name": "Test"}]}, "p-1"
        )
        assert client.get_patient()["id"] == "p-1"

    def test_get_patient_empty(self):
        """Line 175: BundleFHIRClient.get_patient with no patients."""
        client = BundleFHIRClient({}, "p-1")
        assert client.get_patient() == {}

    def test_patient_id_property(self):
        client = BundleFHIRClient({}, "p-1")
        assert client.patient_id == "p-1"

    def test_get_encounters(self):
        """Line 198: BundleFHIRClient.get_encounters."""
        client = BundleFHIRClient(
            {"Encounter": [{"id": "e1"}, {"id": "e2"}]}, "p-1"
        )
        assert len(client.get_encounters(count=1)) == 1

    def test_get_observations_with_filter(self):
        client = BundleFHIRClient(
            {"Observation": [
                {"id": "o1", "category": [{"coding": [{"code": "vital-signs"}]}]},
                {"id": "o2", "category": [{"coding": [{"code": "laboratory"}]}]},
            ]},
            "p-1",
        )
        vitals = client.get_observations(category="vital-signs")
        assert len(vitals) == 1
        assert vitals[0]["id"] == "o1"


# ── coding_display ────────────────────────────────────────────────────────────

class TestCodingDisplay:
    def test_returns_display(self):
        """Lines 203-204: coding with display text."""
        assert coding_display([{"code": "x", "display": "Warfarin"}]) == "Warfarin"

    def test_returns_unknown_when_no_display(self):
        """Lines 205-206: no display → 'Unknown'."""
        assert coding_display([{"code": "x"}]) == "Unknown"

    def test_empty_list(self):
        assert coding_display([]) == "Unknown"


# ── extract_medication_name ───────────────────────────────────────────────────

class TestExtractMedicationName:
    def test_from_text(self):
        resource = {"medicationCodeableConcept": {"text": "Warfarin"}}
        assert extract_medication_name(resource) == "Warfarin"

    def test_from_coding(self):
        resource = {"medicationCodeableConcept": {
            "coding": [{"display": "Metformin"}]
        }}
        assert extract_medication_name(resource) == "Metformin"

    def test_unknown(self):
        assert extract_medication_name({}) == "Unknown"


# ── Lines 69-70: IPv6 in URL path candidate extraction ──────────────────────


class TestIPv6InPathExtraction:
    """Lines 68-70: URL with '::' in path triggers path-based IPv6 extraction."""

    def test_double_colon_in_path_empty_candidate(self):
        """Path starts with '/' so split('/')[0] is empty — covers lines 69-70."""
        ctx = FHIRContext(
            url="https://fhir.example.com/api::v2/r4",
            token="tok",
            patient_id="p-1",
        )
        ctx.validate()  # should not raise — '::' in path but candidate is empty
