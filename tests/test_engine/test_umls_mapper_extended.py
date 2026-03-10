"""Extended coverage for engine/umls_mapper.py — crosswalk, find, are_same, enrich."""
import pytest
from unittest.mock import patch, MagicMock

from engine.umls_mapper import (
    crosswalk,
    find_concept,
    are_same_concept,
    _get_cui,
    enrich_with_crosswalk,
    UMLSConcept,
    UMLS_API_KEY,
)


@pytest.fixture(autouse=True)
def clear_caches():
    crosswalk.cache_clear()
    find_concept.cache_clear()
    _get_cui.cache_clear()
    yield
    crosswalk.cache_clear()
    find_concept.cache_clear()
    _get_cui.cache_clear()


# ── crosswalk ────────────────────────────────────────────────────────────────

class TestCrosswalk:
    @patch("engine.umls_mapper.UMLS_API_KEY", "")
    def test_no_key(self):
        assert crosswalk("ICD10CM", "E11.9", "SNOMEDCT_US") == []

    @patch("engine.umls_mapper.UMLS_API_KEY", "test-key")
    @patch("engine.umls_mapper.httpx")
    def test_success(self, mock_httpx):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "result": [
                {"ui": "C001", "name": "Diabetes"},
                {"ui": "NONE", "name": "NO RESULTS"},
            ]
        }
        mock_httpx.get.return_value = mock_resp
        results = crosswalk("ICD10CM", "E11.9_unique", "SNOMEDCT_US")
        assert len(results) == 1
        assert results[0].cui == "C001"

    @patch("engine.umls_mapper.UMLS_API_KEY", "key")
    @patch("engine.umls_mapper.httpx")
    def test_dict_result(self, mock_httpx):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "result": {"results": [{"ui": "C002", "name": "Test"}]}
        }
        mock_httpx.get.return_value = mock_resp
        results = crosswalk("ICD10CM", "test_unique", "SNOMEDCT_US")
        assert len(results) == 1

    @patch("engine.umls_mapper.UMLS_API_KEY", "key")
    @patch("engine.umls_mapper.httpx")
    def test_non_200(self, mock_httpx):
        mock_resp = MagicMock()
        mock_resp.status_code = 401
        mock_httpx.get.return_value = mock_resp
        assert crosswalk("ICD10CM", "fail_unique", "SNOMEDCT_US") == []

    @patch("engine.umls_mapper.UMLS_API_KEY", "key")
    @patch("engine.umls_mapper.httpx")
    def test_exception(self, mock_httpx):
        mock_httpx.get.side_effect = Exception("fail")
        assert crosswalk("ICD10CM", "exc_unique", "SNOMEDCT_US") == []


# ── find_concept ─────────────────────────────────────────────────────────────

class TestFindConcept:
    @patch("engine.umls_mapper.UMLS_API_KEY", "")
    def test_no_key(self):
        assert find_concept("diabetes") == []

    @patch("engine.umls_mapper.UMLS_API_KEY", "key")
    @patch("engine.umls_mapper.httpx")
    def test_success(self, mock_httpx):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "result": {
                "results": [
                    {"ui": "C011", "name": "Diabetes Mellitus", "rootSource": "SNOMEDCT_US"},
                    {"ui": "NONE", "name": "NO RESULTS"},
                ]
            }
        }
        mock_httpx.get.return_value = mock_resp
        results = find_concept("diabetes_unique")
        assert len(results) == 1

    @patch("engine.umls_mapper.UMLS_API_KEY", "key")
    @patch("engine.umls_mapper.httpx")
    def test_with_source_filter(self, mock_httpx):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"result": {"results": []}}
        mock_httpx.get.return_value = mock_resp
        find_concept("test_src_unique", source="ICD10CM")
        call_args = mock_httpx.get.call_args
        assert call_args[1]["params"]["sab"] == "ICD10CM"

    @patch("engine.umls_mapper.UMLS_API_KEY", "key")
    @patch("engine.umls_mapper.httpx")
    def test_non_200(self, mock_httpx):
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_httpx.get.return_value = mock_resp
        assert find_concept("fail_unique_2") == []

    @patch("engine.umls_mapper.UMLS_API_KEY", "key")
    @patch("engine.umls_mapper.httpx")
    def test_exception(self, mock_httpx):
        mock_httpx.get.side_effect = Exception("network")
        assert find_concept("exc_unique_2") == []


# ── _get_cui ─────────────────────────────────────────────────────────────────

class TestGetCui:
    @patch("engine.umls_mapper.UMLS_API_KEY", "")
    def test_no_key(self):
        assert _get_cui("SNOMEDCT_US", "123") is None

    @patch("engine.umls_mapper.UMLS_API_KEY", "key")
    @patch("engine.umls_mapper.httpx")
    def test_success(self, mock_httpx):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "result": {"concept": "https://uts.nlm.nih.gov/rest/content/current/CUI/C0011849"}
        }
        mock_httpx.get.return_value = mock_resp
        result = _get_cui("SNOMEDCT_US", "unique_code_1")
        assert result == "C0011849"

    @patch("engine.umls_mapper.UMLS_API_KEY", "key")
    @patch("engine.umls_mapper.httpx")
    def test_no_concept_uri(self, mock_httpx):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"result": {"concept": ""}}
        mock_httpx.get.return_value = mock_resp
        assert _get_cui("SNOMEDCT_US", "unique_code_2") is None

    @patch("engine.umls_mapper.UMLS_API_KEY", "key")
    @patch("engine.umls_mapper.httpx")
    def test_non_200(self, mock_httpx):
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        mock_httpx.get.return_value = mock_resp
        assert _get_cui("SNOMEDCT_US", "unique_code_3") is None

    @patch("engine.umls_mapper.UMLS_API_KEY", "key")
    @patch("engine.umls_mapper.httpx")
    def test_exception(self, mock_httpx):
        mock_httpx.get.side_effect = Exception("fail")
        assert _get_cui("SNOMEDCT_US", "unique_code_4") is None


# ── are_same_concept ─────────────────────────────────────────────────────────

class TestAreSameConcept:
    @patch("engine.umls_mapper.UMLS_API_KEY", "")
    def test_no_key(self):
        assert are_same_concept(("ICD10CM", "E11"), ("SNOMEDCT_US", "44054006")) is False

    @patch("engine.umls_mapper._get_cui")
    @patch("engine.umls_mapper.UMLS_API_KEY", "key")
    def test_same_cui(self, mock_get_cui):
        mock_get_cui.side_effect = lambda s, c: "C0011849"
        assert are_same_concept(("ICD10CM", "E11"), ("SNOMEDCT_US", "44054006")) is True

    @patch("engine.umls_mapper._get_cui")
    @patch("engine.umls_mapper.UMLS_API_KEY", "key")
    def test_different_cui(self, mock_get_cui):
        mock_get_cui.side_effect = lambda s, c: f"CUI_{c}"
        assert are_same_concept(("ICD10CM", "E11"), ("SNOMEDCT_US", "99999")) is False

    @patch("engine.umls_mapper._get_cui")
    @patch("engine.umls_mapper.UMLS_API_KEY", "key")
    def test_first_none(self, mock_get_cui):
        mock_get_cui.side_effect = [None, "C001"]
        assert are_same_concept(("ICD10CM", "X"), ("SNOMEDCT_US", "Y")) is False


# ── enrich_with_crosswalk ────────────────────────────────────────────────────

class TestEnrichWithCrosswalk:
    @patch("engine.umls_mapper.crosswalk")
    @patch("engine.umls_mapper.UMLS_API_KEY", "key")
    def test_enriches(self, mock_crosswalk):
        mock_crosswalk.side_effect = lambda src, code, target: (
            [UMLSConcept("C1", "test", target, "mapped_code")]
            if target == "SNOMEDCT_US" else []
        )
        result = enrich_with_crosswalk("ICD10CM", "E11.9", "Diabetes")
        assert "ICD10CM" in result
        assert "SNOMEDCT_US" in result

    @patch("engine.umls_mapper.crosswalk", return_value=[])
    @patch("engine.umls_mapper.UMLS_API_KEY", "key")
    def test_no_mappings(self, mock_crosswalk):
        result = enrich_with_crosswalk("ICD10CM", "E11.9", "Diabetes")
        assert result == {"ICD10CM": "E11.9"}


# ── are_same_concept: cui_b is None (line 147) ────────────────────────────

class TestAreSameConceptCuiBNone:
    """Cover line 147: return False when _get_cui for second code returns None."""

    @patch("engine.umls_mapper._get_cui")
    @patch("engine.umls_mapper.UMLS_API_KEY", "key")
    def test_second_cui_none_returns_false(self, mock_get_cui):
        """Line 147: code_a resolves to a CUI, but code_b does not → return False."""
        # First call (code_a) returns a CUI, second call (code_b) returns None
        mock_get_cui.side_effect = ["C0011849", None]
        result = are_same_concept(
            ("ICD10CM", "E11_test_b_none"),
            ("SNOMEDCT_US", "nonexistent_code_xyz"),
        )
        assert result is False
        assert mock_get_cui.call_count == 2
