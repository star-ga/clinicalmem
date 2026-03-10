"""Extended SNOMED CT tests — covers search_snomed, snowstorm, UMLS, and FHIR mapping."""
import pytest
from unittest.mock import patch, MagicMock

from engine.snomed_client import (
    search_snomed,
    _search_snowstorm,
    _search_umls_snomed,
    map_fhir_code_to_snomed,
    SnomedConcept,
    is_allergy_cross_reactive,
    get_allergy_cross_reactions,
)


class TestSearchSnowstorm:
    @patch("engine.snomed_client.httpx")
    def test_successful_search(self, mock_httpx):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "items": [
                {
                    "conceptId": "91936005",
                    "fsn": {"term": "Allergy to penicillin (finding)"},
                    "pt": {"term": "Allergy to penicillin"},
                }
            ]
        }
        mock_httpx.get.return_value = mock_resp
        results = _search_snowstorm("penicillin allergy")
        assert len(results) == 1
        assert results[0].concept_id == "91936005"
        assert "finding" in results[0].semantic_tag

    @patch("engine.snomed_client.httpx")
    def test_with_semantic_tag(self, mock_httpx):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"items": []}
        mock_httpx.get.return_value = mock_resp
        _search_snowstorm("diabetes", semantic_tag="disorder")
        call_args = mock_httpx.get.call_args
        assert call_args[1]["params"]["semanticTag"] == "disorder"

    @patch("engine.snomed_client.httpx")
    def test_non_200_returns_empty(self, mock_httpx):
        mock_resp = MagicMock()
        mock_resp.status_code = 503
        mock_httpx.get.return_value = mock_resp
        assert _search_snowstorm("test") == []

    @patch("engine.snomed_client.httpx")
    def test_exception_returns_empty(self, mock_httpx):
        mock_httpx.get.side_effect = Exception("timeout")
        assert _search_snowstorm("test") == []

    @patch("engine.snomed_client.httpx")
    def test_no_fsn_parentheses(self, mock_httpx):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "items": [{"conceptId": "123", "fsn": {"term": "Simple term"}, "pt": {"term": "Simple"}}]
        }
        mock_httpx.get.return_value = mock_resp
        results = _search_snowstorm("simple")
        assert results[0].semantic_tag == ""


class TestSearchUmlsSnomed:
    @patch("engine.snomed_client.UMLS_API_KEY", "")
    def test_no_api_key_returns_empty(self):
        assert _search_umls_snomed("test") == []

    @patch("engine.snomed_client.UMLS_API_KEY", "test-key")
    @patch("engine.snomed_client.httpx")
    def test_successful_search(self, mock_httpx):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "result": {
                "results": [
                    {"ui": "C0002792", "name": "Penicillin allergy"},
                    {"ui": "NONE", "name": "NO RESULTS"},
                ]
            }
        }
        mock_httpx.get.return_value = mock_resp
        results = _search_umls_snomed("penicillin")
        assert len(results) == 1
        assert results[0].concept_id == "C0002792"

    @patch("engine.snomed_client.UMLS_API_KEY", "test-key")
    @patch("engine.snomed_client.httpx")
    def test_non_200_returns_empty(self, mock_httpx):
        mock_resp = MagicMock()
        mock_resp.status_code = 401
        mock_httpx.get.return_value = mock_resp
        assert _search_umls_snomed("test") == []

    @patch("engine.snomed_client.UMLS_API_KEY", "test-key")
    @patch("engine.snomed_client.httpx")
    def test_exception_returns_empty(self, mock_httpx):
        mock_httpx.get.side_effect = Exception("network error")
        assert _search_umls_snomed("test") == []


class TestSearchSnomed:
    @patch("engine.snomed_client._search_snowstorm")
    def test_uses_snowstorm_first(self, mock_snow):
        concept = SnomedConcept(concept_id="123", term="test", semantic_tag="finding")
        mock_snow.return_value = [concept]
        # Clear lru_cache
        search_snomed.cache_clear()
        results = search_snomed("test_unique_term_1")
        assert len(results) == 1
        search_snomed.cache_clear()

    @patch("engine.snomed_client.UMLS_API_KEY", "key")
    @patch("engine.snomed_client._search_umls_snomed")
    @patch("engine.snomed_client._search_snowstorm")
    def test_falls_back_to_umls(self, mock_snow, mock_umls):
        mock_snow.return_value = []
        concept = SnomedConcept(concept_id="C123", term="test", semantic_tag="")
        mock_umls.return_value = [concept]
        search_snomed.cache_clear()
        results = search_snomed("test_unique_term_2")
        assert len(results) == 1
        assert results[0].concept_id == "C123"
        search_snomed.cache_clear()


class TestMapFhirCodeToSnomed:
    def test_snomed_system(self):
        coding = {"system": "http://snomed.info/sct", "code": "91936005", "display": "Penicillin allergy"}
        result = map_fhir_code_to_snomed(coding)
        assert result is not None
        assert result.concept_id == "91936005"

    def test_non_snomed_with_display(self):
        coding = {"system": "http://loinc.org", "code": "1234", "display": "Glucose"}
        with patch("engine.snomed_client.search_snomed") as mock_search:
            mock_search.return_value = [SnomedConcept("999", "Glucose", "observable")]
            result = map_fhir_code_to_snomed(coding)
            assert result is not None
            assert result.concept_id == "999"

    def test_non_snomed_no_display(self):
        coding = {"system": "http://loinc.org", "code": "1234", "display": ""}
        result = map_fhir_code_to_snomed(coding)
        assert result is None

    def test_non_snomed_search_returns_empty(self):
        coding = {"system": "http://loinc.org", "code": "1234", "display": "Unknown Thing"}
        with patch("engine.snomed_client.search_snomed") as mock_search:
            mock_search.return_value = []
            result = map_fhir_code_to_snomed(coding)
            assert result is None


class TestAllergyAliasExpansion:
    def test_pen_vk_alias(self):
        assert is_allergy_cross_reactive("pen-vk", "amoxicillin") is True

    def test_penicillin_v_alias(self):
        assert is_allergy_cross_reactive("penicillin v", "ampicillin") is True

    def test_fluoroquinolone_class(self):
        assert is_allergy_cross_reactive("fluoroquinolone", "ciprofloxacin") is True
        assert is_allergy_cross_reactive("fluoroquinolone", "levofloxacin") is True

    def test_get_reactions_fluoroquinolone(self):
        reactions = get_allergy_cross_reactions("fluoroquinolone")
        assert "ciprofloxacin" in reactions
        assert "moxifloxacin" in reactions
