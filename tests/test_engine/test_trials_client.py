"""Tests for ClinicalTrials.gov API v2 client — mocked HTTP responses."""
import pytest
import respx
import httpx

from engine.trials_client import (
    search_trials,
    match_patient_to_trials,
    ClinicalTrial,
    TrialMatchResult,
    CTGOV_BASE,
    _ssrf_guard,
)


@pytest.fixture(autouse=True)
def clear_caches():
    """Clear lru_cache between tests."""
    search_trials.cache_clear()
    yield


def _mock_study(nct_id="NCT00000001", title="Test Trial", condition="Diabetes"):
    """Helper to build a mock ClinicalTrials.gov study JSON."""
    return {
        "protocolSection": {
            "identificationModule": {"nctId": nct_id, "briefTitle": title},
            "statusModule": {
                "overallStatus": "RECRUITING",
                "startDateStruct": {"date": "2026-01-01"},
            },
            "conditionsModule": {"conditions": [condition]},
            "armsInterventionsModule": {
                "interventions": [{"name": "Drug X"}]
            },
            "designModule": {
                "phases": ["PHASE3"],
                "enrollmentInfo": {"count": 500},
            },
            "contactsLocationsModule": {
                "locations": [{"city": "Boston", "country": "United States"}]
            },
        }
    }


class TestSsrfGuard:
    def test_allowed_host(self):
        # Should not raise
        _ssrf_guard("https://clinicaltrials.gov/api/v2/studies")

    def test_blocked_host(self):
        with pytest.raises(ValueError, match="SSRF blocked"):
            _ssrf_guard("https://evil.com/api")

    def test_blocked_internal(self):
        with pytest.raises(ValueError, match="SSRF blocked"):
            _ssrf_guard("http://169.254.169.254/latest/meta-data/")


class TestSearchTrials:
    @respx.mock
    def test_returns_trials(self):
        respx.get(f"{CTGOV_BASE}/studies").mock(
            return_value=httpx.Response(200, json={
                "studies": [_mock_study()]
            })
        )
        trials = search_trials("Diabetes")
        assert len(trials) == 1
        assert isinstance(trials[0], ClinicalTrial)
        assert trials[0].nct_id == "NCT00000001"
        assert trials[0].title == "Test Trial"
        assert trials[0].status == "RECRUITING"
        assert "Diabetes" in trials[0].conditions
        assert trials[0].phase == "PHASE3"
        assert trials[0].enrollment == 500

    @respx.mock
    def test_multiple_trials(self):
        respx.get(f"{CTGOV_BASE}/studies").mock(
            return_value=httpx.Response(200, json={
                "studies": [
                    _mock_study("NCT00000001", "Trial A"),
                    _mock_study("NCT00000002", "Trial B"),
                    _mock_study("NCT00000003", "Trial C"),
                ]
            })
        )
        trials = search_trials("Heart Failure")
        assert len(trials) == 3

    @respx.mock
    def test_empty_results(self):
        respx.get(f"{CTGOV_BASE}/studies").mock(
            return_value=httpx.Response(200, json={"studies": []})
        )
        trials = search_trials("RareConditionXYZ123")
        assert trials == ()

    @respx.mock
    def test_api_error_returns_empty(self):
        respx.get(f"{CTGOV_BASE}/studies").mock(
            return_value=httpx.Response(500)
        )
        trials = search_trials("Diabetes")
        assert trials == ()

    def test_empty_condition_returns_empty(self):
        trials = search_trials("")
        assert trials == ()

    def test_whitespace_condition_returns_empty(self):
        trials = search_trials("   ")
        assert trials == ()

    @respx.mock
    def test_returns_immutable_tuple(self):
        respx.get(f"{CTGOV_BASE}/studies").mock(
            return_value=httpx.Response(200, json={
                "studies": [_mock_study()]
            })
        )
        result = search_trials("Test")
        assert isinstance(result, tuple)

    @respx.mock
    def test_trial_url(self):
        respx.get(f"{CTGOV_BASE}/studies").mock(
            return_value=httpx.Response(200, json={
                "studies": [_mock_study("NCT12345678")]
            })
        )
        trials = search_trials("Cancer")
        assert trials[0].url == "https://clinicaltrials.gov/study/NCT12345678"

    @respx.mock
    def test_locations_extracted(self):
        respx.get(f"{CTGOV_BASE}/studies").mock(
            return_value=httpx.Response(200, json={
                "studies": [_mock_study()]
            })
        )
        trials = search_trials("Diabetes")
        assert "Boston" in trials[0].locations[0]

    @respx.mock
    def test_missing_fields_handled(self):
        """Studies with missing optional fields should not crash."""
        respx.get(f"{CTGOV_BASE}/studies").mock(
            return_value=httpx.Response(200, json={
                "studies": [{
                    "protocolSection": {
                        "identificationModule": {"nctId": "NCT99999999"},
                        "statusModule": {},
                        "conditionsModule": {},
                        "armsInterventionsModule": {},
                        "designModule": {},
                        "contactsLocationsModule": {},
                    }
                }]
            })
        )
        trials = search_trials("Anything")
        assert len(trials) == 1
        assert trials[0].nct_id == "NCT99999999"
        assert trials[0].phase == "N/A"
        assert trials[0].enrollment is None


class TestMatchPatientToTrials:
    @respx.mock
    def test_matches_multiple_conditions(self):
        call_count = 0

        def handler(request):
            nonlocal call_count
            call_count += 1
            return httpx.Response(200, json={
                "studies": [_mock_study(f"NCT{call_count:08d}", f"Trial {call_count}")]
            })

        respx.get(f"{CTGOV_BASE}/studies").mock(side_effect=handler)
        result = match_patient_to_trials(["Diabetes", "Hypertension"])
        assert isinstance(result, TrialMatchResult)
        assert result.total_found >= 1
        assert len(result.search_terms) == 2

    def test_empty_conditions(self):
        result = match_patient_to_trials([])
        assert result.total_found == 0
        assert result.matched_trials == ()

    @respx.mock
    def test_deduplicates_by_nct_id(self):
        """Same trial from multiple condition searches appears once."""
        respx.get(f"{CTGOV_BASE}/studies").mock(
            return_value=httpx.Response(200, json={
                "studies": [_mock_study("NCT00000001", "Same Trial")]
            })
        )
        result = match_patient_to_trials(["Diabetes", "Obesity"])
        nct_ids = [t.nct_id for t in result.matched_trials]
        assert len(nct_ids) == len(set(nct_ids))  # No duplicates


# ── search_trials exception handler (lines 154-156) ───────────────────────

class TestSearchTrialsException:
    """Cover lines 154-156: except Exception handler in search_trials."""

    @respx.mock
    def test_httpx_exception_returns_empty(self):
        """Lines 154-156: When httpx.get raises an exception, return empty tuple."""
        respx.get(f"{CTGOV_BASE}/studies").mock(
            side_effect=httpx.ConnectError("Connection refused")
        )
        trials = search_trials("ExceptionCondition_unique_1")
        assert trials == ()

    @respx.mock
    def test_timeout_exception_returns_empty(self):
        """Lines 154-156: Timeout exception also returns empty."""
        respx.get(f"{CTGOV_BASE}/studies").mock(
            side_effect=httpx.ReadTimeout("Read timed out")
        )
        trials = search_trials("TimeoutCondition_unique_1")
        assert trials == ()

    @respx.mock
    def test_json_decode_error_returns_empty(self):
        """Lines 154-156: Malformed JSON also caught by except Exception."""
        respx.get(f"{CTGOV_BASE}/studies").mock(
            return_value=httpx.Response(200, text="not json at all")
        )
        trials = search_trials("JsonErrorCondition_unique_1")
        assert trials == ()


# ── match_patient_to_trials: empty condition string (line 189) ─────────────

class TestMatchPatientEmptyCondition:
    """Cover line 189: continue when condition is empty string."""

    @respx.mock
    def test_empty_condition_skipped(self):
        """Line 189: Empty string conditions in the list are skipped."""
        respx.get(f"{CTGOV_BASE}/studies").mock(
            return_value=httpx.Response(200, json={
                "studies": [_mock_study("NCT99990001", "Real Trial")]
            })
        )
        result = match_patient_to_trials(["", "  ", "Diabetes"])
        # Only "Diabetes" should be in search_terms (empty strings skipped)
        assert "Diabetes" in result.search_terms
        assert "" not in result.search_terms
        assert result.total_found >= 1

    def test_all_empty_conditions(self):
        """When all conditions are empty strings, no searches are made."""
        result = match_patient_to_trials(["", "  ", ""])
        assert result.total_found == 0
        assert result.matched_trials == ()
