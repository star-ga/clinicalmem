"""Tests for FDA client — openFDA API integration with mocked HTTP."""
import pytest
import respx
import httpx
from unittest.mock import patch

from engine.fda_client import (
    _get_httpx,
    get_adverse_events,
    get_label_warnings,
    get_drug_recalls,
    get_safety_profile,
    FDAAlert,
    FDASafetyProfile,
)


class TestGetAdverseEvents:
    @respx.mock
    def test_returns_alerts(self):
        respx.get("https://api.fda.gov/drug/event.json").mock(
            return_value=httpx.Response(200, json={
                "results": [
                    {"term": "Nausea", "count": 5000},
                    {"term": "Headache", "count": 200},
                ]
            })
        )
        alerts = get_adverse_events("metformin")
        assert len(alerts) == 2
        assert alerts[0].drug_name == "metformin"
        assert alerts[0].alert_type == "adverse_event"
        assert alerts[0].source == "openFDA/FAERS"
        assert "Nausea" in alerts[0].description

    @respx.mock
    def test_severity_high(self):
        respx.get("https://api.fda.gov/drug/event.json").mock(
            return_value=httpx.Response(200, json={
                "results": [{"term": "Death", "count": 5000}]
            })
        )
        alerts = get_adverse_events("warfarin")
        assert alerts[0].severity == "high"

    @respx.mock
    def test_severity_moderate(self):
        respx.get("https://api.fda.gov/drug/event.json").mock(
            return_value=httpx.Response(200, json={
                "results": [{"term": "Rash", "count": 500}]
            })
        )
        alerts = get_adverse_events("amoxicillin")
        assert alerts[0].severity == "moderate"

    @respx.mock
    def test_severity_low(self):
        respx.get("https://api.fda.gov/drug/event.json").mock(
            return_value=httpx.Response(200, json={
                "results": [{"term": "Dizziness", "count": 50}]
            })
        )
        alerts = get_adverse_events("aspirin")
        assert alerts[0].severity == "low"

    @respx.mock
    def test_api_error_returns_empty(self):
        respx.get("https://api.fda.gov/drug/event.json").mock(
            return_value=httpx.Response(404)
        )
        alerts = get_adverse_events("nonexistent_drug_xyz")
        assert alerts == []

    @respx.mock
    def test_empty_results(self):
        respx.get("https://api.fda.gov/drug/event.json").mock(
            return_value=httpx.Response(200, json={"results": []})
        )
        alerts = get_adverse_events("water")
        assert alerts == []


class TestGetLabelWarnings:
    @respx.mock
    def test_black_box_warning(self):
        respx.get("https://api.fda.gov/drug/label.json").mock(
            return_value=httpx.Response(200, json={
                "results": [{
                    "boxed_warning": ["WARNING: Risk of serious bleeding"],
                    "warnings_and_precautions": ["Use with caution"],
                    "contraindications": ["Active bleeding"],
                }]
            })
        )
        alerts = get_label_warnings("warfarin")
        types = [a.alert_type for a in alerts]
        assert all(t == "label_warning" for t in types)
        severities = [a.severity for a in alerts]
        assert "critical" in severities  # black box = critical
        assert any("BLACK BOX" in a.description for a in alerts)

    @respx.mock
    def test_no_warnings(self):
        respx.get("https://api.fda.gov/drug/label.json").mock(
            return_value=httpx.Response(200, json={"results": [{}]})
        )
        alerts = get_label_warnings("saline")
        assert alerts == []

    @respx.mock
    def test_api_error(self):
        respx.get("https://api.fda.gov/drug/label.json").mock(
            return_value=httpx.Response(500)
        )
        alerts = get_label_warnings("anything")
        assert alerts == []


class TestGetDrugRecalls:
    @respx.mock
    def test_class_i_recall(self):
        respx.get("https://api.fda.gov/drug/enforcement.json").mock(
            return_value=httpx.Response(200, json={
                "results": [{
                    "classification": "Class I",
                    "reason_for_recall": "Contamination with foreign substance",
                }]
            })
        )
        alerts = get_drug_recalls("valsartan")
        assert len(alerts) == 1
        assert alerts[0].severity == "critical"
        assert alerts[0].alert_type == "recall"

    @respx.mock
    def test_class_ii_recall(self):
        respx.get("https://api.fda.gov/drug/enforcement.json").mock(
            return_value=httpx.Response(200, json={
                "results": [{"classification": "Class II", "reason_for_recall": "Mislabeling"}]
            })
        )
        alerts = get_drug_recalls("generic_drug")
        assert alerts[0].severity == "high"

    @respx.mock
    def test_no_recalls(self):
        respx.get("https://api.fda.gov/drug/enforcement.json").mock(
            return_value=httpx.Response(200, json={"results": []})
        )
        alerts = get_drug_recalls("metformin")
        assert alerts == []


class TestGetSafetyProfile:
    @respx.mock
    def test_combines_all_sources(self):
        # Mock all three endpoints
        respx.get("https://api.fda.gov/drug/event.json").mock(
            return_value=httpx.Response(200, json={
                "results": [{"term": "Nausea", "count": 100}]
            })
        )
        respx.get("https://api.fda.gov/drug/label.json").mock(
            return_value=httpx.Response(200, json={
                "results": [{"boxed_warning": ["Serious risk"]}]
            })
        )
        respx.get("https://api.fda.gov/drug/enforcement.json").mock(
            return_value=httpx.Response(200, json={"results": []})
        )
        profile = get_safety_profile(["warfarin"])
        assert isinstance(profile, FDASafetyProfile)
        assert len(profile.alerts) >= 2
        assert profile.medications == ["warfarin"]
        assert profile.highest_severity in ("critical", "high", "moderate", "low")

    @respx.mock
    def test_empty_medication_list(self):
        profile = get_safety_profile([])
        assert profile.alerts == []
        assert profile.total_adverse_events == 0
        assert profile.highest_severity == "none"

    @respx.mock
    def test_black_box_collected(self):
        respx.get("https://api.fda.gov/drug/event.json").mock(
            return_value=httpx.Response(200, json={"results": []})
        )
        respx.get("https://api.fda.gov/drug/label.json").mock(
            return_value=httpx.Response(200, json={
                "results": [{"boxed_warning": ["Suicidal thoughts warning"]}]
            })
        )
        respx.get("https://api.fda.gov/drug/enforcement.json").mock(
            return_value=httpx.Response(200, json={"results": []})
        )
        profile = get_safety_profile(["fluoxetine"])
        assert len(profile.black_box_warnings) >= 1

    @respx.mock
    def test_sorted_by_severity(self):
        respx.get("https://api.fda.gov/drug/event.json").mock(
            return_value=httpx.Response(200, json={
                "results": [{"term": "Rash", "count": 50}]
            })
        )
        respx.get("https://api.fda.gov/drug/label.json").mock(
            return_value=httpx.Response(200, json={
                "results": [{"boxed_warning": ["Serious risk"]}]
            })
        )
        respx.get("https://api.fda.gov/drug/enforcement.json").mock(
            return_value=httpx.Response(200, json={"results": []})
        )
        profile = get_safety_profile(["drug_x"])
        if len(profile.alerts) >= 2:
            severity_rank = {"critical": 4, "high": 3, "moderate": 2, "low": 1}
            for i in range(len(profile.alerts) - 1):
                a = severity_rank.get(profile.alerts[i].severity, 0)
                b = severity_rank.get(profile.alerts[i + 1].severity, 0)
                assert a >= b


# ---------------------------------------------------------------------------
# Coverage gap tests — lines 48-49, 62, 97-99, 110, 128, 174-176, 187, 201,
#                       225-227
# ---------------------------------------------------------------------------


class TestGetHttpxImportError:
    """Cover lines 48-49: _get_httpx() ImportError path."""

    def test_returns_none_when_httpx_unavailable(self):
        """Simulate httpx not installed by making the import raise."""
        import builtins

        real_import = builtins.__import__

        def _mock_import(name, *args, **kwargs):
            if name == "httpx":
                raise ImportError("No module named 'httpx'")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=_mock_import):
            result = _get_httpx()
        assert result is None


class TestHttpxUnavailableReturnsEmpty:
    """Cover lines 62, 110, 187: each public function returns [] when
    _get_httpx() yields None (httpx not installed)."""

    @patch("engine.fda_client._get_httpx", return_value=None)
    def test_adverse_events_returns_empty(self, _mock):
        """Line 62."""
        assert get_adverse_events("metformin") == []

    @patch("engine.fda_client._get_httpx", return_value=None)
    def test_label_warnings_returns_empty(self, _mock):
        """Line 110."""
        assert get_label_warnings("metformin") == []

    @patch("engine.fda_client._get_httpx", return_value=None)
    def test_drug_recalls_returns_empty(self, _mock):
        """Line 187."""
        assert get_drug_recalls("metformin") == []


class TestExceptionHandlers:
    """Cover lines 97-99, 174-176, 225-227: exception paths inside each
    public function's try/except block."""

    @patch("engine.fda_client._get_httpx")
    def test_adverse_events_exception_returns_empty(self, mock_httpx):
        """Lines 97-99: get_adverse_events catches arbitrary exceptions."""
        mock_mod = mock_httpx.return_value
        mock_mod.get.side_effect = RuntimeError("network down")
        assert get_adverse_events("warfarin") == []

    @patch("engine.fda_client._get_httpx")
    def test_label_warnings_exception_returns_empty(self, mock_httpx):
        """Lines 174-176: get_label_warnings catches arbitrary exceptions."""
        mock_mod = mock_httpx.return_value
        mock_mod.get.side_effect = ConnectionError("timeout")
        assert get_label_warnings("warfarin") == []

    @patch("engine.fda_client._get_httpx")
    def test_drug_recalls_exception_returns_empty(self, mock_httpx):
        """Lines 225-227: get_drug_recalls catches arbitrary exceptions."""
        mock_mod = mock_httpx.return_value
        mock_mod.get.side_effect = OSError("DNS failure")
        assert get_drug_recalls("warfarin") == []


class TestLabelWarningsNoResults:
    """Cover line 128: get_label_warnings when API returns 200 but results
    list is empty."""

    @respx.mock
    def test_empty_results_list_returns_empty(self):
        respx.get("https://api.fda.gov/drug/label.json").mock(
            return_value=httpx.Response(200, json={"results": []})
        )
        alerts = get_label_warnings("saline")
        assert alerts == []


class TestDrugRecallsNon200:
    """Cover line 201: get_drug_recalls when API returns a non-200 status."""

    @respx.mock
    def test_non_200_returns_empty(self):
        respx.get("https://api.fda.gov/drug/enforcement.json").mock(
            return_value=httpx.Response(503)
        )
        alerts = get_drug_recalls("metformin")
        assert alerts == []
