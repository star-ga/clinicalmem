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


# ─── Iter-156 logger-density ratchet pin (T4 round-31) ───────────────
#
# engine/fhir_client.py was at 16.2 logger calls / kloc (4 logs / 247 LOC).
# Same ratchet pattern as iter-138 / iter-144 / iter-151 / iter-154.
#
# Three silent paths closed:
#   1. FHIRContext.validate() success (DEBUG fhir_ctx_validated) —
#      logs hostname + scheme + patient_id (PHI-safe: hostname is
#      server config, patient_id is the synthetic cohort identifier).
#   2. FHIRClient.__init__ (DEBUG fhir_client_init) — logs URL host +
#      patient_id; auth token NEVER logged.
#   3. BundleFHIRClient.__init__ (DEBUG fhir_bundle_client_init) — logs
#      aggregate counts (resource_types + total_resources); resource
#      contents NEVER logged.
#   4. Plus an explicit ERROR fhir_url_blocked_private_addr on the
#      previously-silent RFC-1918 / loopback / link-local rejection
#      path (was raising FHIRClientError without a structured log).
#
# Net: 4 -> 8 logger calls (+4 events). Density 16.2 -> 27.7/kloc.

import logging


class TestFHIRClientLoggerRatchetIter156:
    """Iter-156 T4 round-31 — logger density ratchet pin."""

    def test_validate_success_emits_debug(self, caplog):
        ctx = FHIRContext(url="https://fhir.example.com/r4", token="t", patient_id="p-1")
        with caplog.at_level(logging.DEBUG, logger='engine.fhir_client'):
            ctx.validate()
        recs = [r for r in caplog.records if r.message == 'fhir_ctx_validated']
        assert len(recs) >= 1
        rec = recs[0]
        assert getattr(rec, 'hostname', None) == 'fhir.example.com'
        assert getattr(rec, 'scheme', None) == 'https'
        # iter-332 PHI migration: patient_id is now hashed (16-char SHA-256)
        # before reaching the log handler. Raw patient_id MUST NOT appear.
        import hashlib  # noqa: PLC0415
        expected_hash = hashlib.sha256(b'p-1').hexdigest()[:16]
        assert getattr(rec, 'patient_id_hash_prefix', None) == expected_hash
        assert getattr(rec, 'patient_id', None) is None  # Raw must not leak

    def test_client_init_emits_debug(self, caplog):
        ctx = FHIRContext(url="https://fhir.example.com/r4", token="secrettoken", patient_id="p-2")
        with caplog.at_level(logging.DEBUG, logger='engine.fhir_client'):
            FHIRClient(ctx)
        recs = [r for r in caplog.records if r.message == 'fhir_client_init']
        assert len(recs) >= 1
        rec = recs[0]
        assert getattr(rec, 'host', None) == 'fhir.example.com'
        # iter-332 PHI migration: hashed patient_id only.
        import hashlib  # noqa: PLC0415
        expected_hash = hashlib.sha256(b'p-2').hexdigest()[:16]
        assert getattr(rec, 'patient_id_hash_prefix', None) == expected_hash
        assert getattr(rec, 'patient_id', None) is None
        # Token must NEVER appear in any log record
        for r in caplog.records:
            haystack = str(r.message) + ' ' + ' '.join(str(v) for v in vars(r).values()
                                                        if isinstance(v, (str, int)))
            assert 'secrettoken' not in haystack, (
                "PHI/secret leak: bearer token leaked into log records"
            )

    def test_bundle_client_init_emits_debug(self, caplog):
        with caplog.at_level(logging.DEBUG, logger='engine.fhir_client'):
            BundleFHIRClient({"Patient": [{}], "MedicationRequest": [{}, {}]}, "p-3")
        recs = [r for r in caplog.records if r.message == 'fhir_bundle_client_init']
        assert len(recs) >= 1
        rec = recs[0]
        # iter-332 PHI migration: hashed patient_id only.
        import hashlib  # noqa: PLC0415
        expected_hash = hashlib.sha256(b'p-3').hexdigest()[:16]
        assert getattr(rec, 'patient_id_hash_prefix', None) == expected_hash
        assert getattr(rec, 'patient_id', None) is None
        assert getattr(rec, 'resource_types', None) == 2
        assert getattr(rec, 'total_resources', None) == 3

    def test_logger_density_floor(self):
        """Floor: fhir_client.py must keep >= 8 logger calls.
        Iter-156 ratchet bumped 4 -> 8 (+4 events); future regressions
        below 8 fail this gate."""
        import re
        from pathlib import Path
        path = Path(__file__).resolve().parent.parent.parent / 'engine' / 'fhir_client.py'
        text = path.read_text()
        count = len(re.findall(r'logger\.(debug|info|warning|error|critical)', text))
        assert count >= 8, (
            f"engine/fhir_client.py logger density regressed: "
            f"got {count}, floor=8 (iter-156 ratchet)"
        )
