"""Tests for FDA client — openFDA API integration with mocked HTTP."""
import pytest
import respx
import httpx

from engine.fda_client import (
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
