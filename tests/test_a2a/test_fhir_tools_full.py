"""Full coverage tests for a2a_agent/tools/fhir_tools.py — exception handlers and valueString branch."""
import os
import pytest
from unittest.mock import patch, MagicMock

a2a_types = pytest.importorskip("a2a.types")
google_adk = pytest.importorskip("google.adk")


def _make_ctx(state=None, patient_id="patient-sarah-mitchell"):
    ctx = MagicMock()
    ctx.state = state if state is not None else {
        "patient_id": patient_id,
        "fhir_url": "https://fhir.test/r4",
        "fhir_token": "tok",
    }
    ctx.metadata = None
    ctx.run_config = None
    return ctx


# ── Exception handlers (lines 66-67, 93-94, 122-123, 154-155) ───────────────


class TestGetPatientDemographicsException:
    """Lines 66-67: _fhir_get raises Exception in get_patient_demographics."""

    @patch(
        "a2a_agent.tools.fhir_tools._get_fhir_context",
        return_value=("https://fhir.test/r4", "tok", "p-1"),
    )
    @patch(
        "a2a_agent.tools.fhir_tools._fhir_get",
        side_effect=Exception("network error"),
    )
    def test_demographics_exception(self, mock_fhir_get, mock_ctx):
        from a2a_agent.tools.fhir_tools import get_patient_demographics

        ctx = _make_ctx()
        result = get_patient_demographics(ctx)
        assert result["status"] == "error"
        assert "network error" in result["error_message"]


class TestGetActiveMedicationsException:
    """Lines 93-94: _fhir_get raises Exception in get_active_medications."""

    @patch(
        "a2a_agent.tools.fhir_tools._get_fhir_context",
        return_value=("https://fhir.test/r4", "tok", "p-1"),
    )
    @patch(
        "a2a_agent.tools.fhir_tools._fhir_get",
        side_effect=Exception("timeout"),
    )
    def test_medications_exception(self, mock_fhir_get, mock_ctx):
        from a2a_agent.tools.fhir_tools import get_active_medications

        ctx = _make_ctx()
        result = get_active_medications(ctx)
        assert result["status"] == "error"
        assert "timeout" in result["error_message"]


class TestGetActiveConditionsException:
    """Lines 122-123: _fhir_get raises Exception in get_active_conditions."""

    @patch(
        "a2a_agent.tools.fhir_tools._get_fhir_context",
        return_value=("https://fhir.test/r4", "tok", "p-1"),
    )
    @patch(
        "a2a_agent.tools.fhir_tools._fhir_get",
        side_effect=Exception("connection refused"),
    )
    def test_conditions_exception(self, mock_fhir_get, mock_ctx):
        from a2a_agent.tools.fhir_tools import get_active_conditions

        ctx = _make_ctx()
        result = get_active_conditions(ctx)
        assert result["status"] == "error"
        assert "connection refused" in result["error_message"]


class TestGetRecentObservationsException:
    """Lines 154-155: _fhir_get raises Exception in get_recent_observations."""

    @patch(
        "a2a_agent.tools.fhir_tools._get_fhir_context",
        return_value=("https://fhir.test/r4", "tok", "p-1"),
    )
    @patch(
        "a2a_agent.tools.fhir_tools._fhir_get",
        side_effect=Exception("server unavailable"),
    )
    def test_observations_exception(self, mock_fhir_get, mock_ctx):
        from a2a_agent.tools.fhir_tools import get_recent_observations

        ctx = _make_ctx()
        result = get_recent_observations("vital-signs", ctx)
        assert result["status"] == "error"
        assert "server unavailable" in result["error_message"]


# ── valueString branch (line 165-166) ────────────────────────────────────────


class TestObservationValueStringBranch:
    """Line 165-166: Observation with valueString instead of valueQuantity."""

    @patch(
        "a2a_agent.tools.fhir_tools._get_fhir_context",
        return_value=("https://fhir.test/r4", "tok", "p-1"),
    )
    @patch("a2a_agent.tools.fhir_tools._fhir_get")
    def test_value_string_observation(self, mock_fhir_get, mock_ctx):
        from a2a_agent.tools.fhir_tools import get_recent_observations

        # Return a bundle with an Observation that uses valueString
        mock_fhir_get.return_value = {
            "resourceType": "Bundle",
            "type": "searchset",
            "total": 1,
            "entry": [
                {
                    "resource": {
                        "resourceType": "Observation",
                        "code": {
                            "text": "Urine Drug Screen",
                            "coding": [{"display": "Urine Drug Screen"}],
                        },
                        "valueString": "positive",
                        "effectiveDateTime": "2026-03-01T10:00:00Z",
                    }
                }
            ],
        }

        ctx = _make_ctx()
        result = get_recent_observations("laboratory", ctx)
        assert result["status"] == "success"
        assert len(result["observations"]) == 1
        obs = result["observations"][0]
        assert obs["value"] == "positive"
        assert obs["unit"] is None
        assert obs["observation"] == "Urine Drug Screen"

    @patch(
        "a2a_agent.tools.fhir_tools._get_fhir_context",
        return_value=("https://fhir.test/r4", "tok", "p-1"),
    )
    @patch("a2a_agent.tools.fhir_tools._fhir_get")
    def test_mixed_value_types(self, mock_fhir_get, mock_ctx):
        """Observations with both valueQuantity and valueString in same bundle."""
        from a2a_agent.tools.fhir_tools import get_recent_observations

        mock_fhir_get.return_value = {
            "resourceType": "Bundle",
            "type": "searchset",
            "total": 2,
            "entry": [
                {
                    "resource": {
                        "resourceType": "Observation",
                        "code": {"text": "Blood Pressure"},
                        "valueQuantity": {"value": 120, "unit": "mmHg"},
                        "effectiveDateTime": "2026-03-01T10:00:00Z",
                    }
                },
                {
                    "resource": {
                        "resourceType": "Observation",
                        "code": {"text": "Smoking Status"},
                        "valueString": "former smoker",
                        "effectiveDateTime": "2026-03-01T10:00:00Z",
                    }
                },
            ],
        }

        ctx = _make_ctx()
        result = get_recent_observations("social-history", ctx)
        assert result["status"] == "success"
        assert len(result["observations"]) == 2
        # First has valueQuantity
        assert result["observations"][0]["value"] == 120
        assert result["observations"][0]["unit"] == "mmHg"
        # Second has valueString
        assert result["observations"][1]["value"] == "former smoker"
        assert result["observations"][1]["unit"] is None
