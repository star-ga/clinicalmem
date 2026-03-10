"""Extended coverage for engine/rxnorm_client.py — resolve, properties, interactions."""
import pytest
from unittest.mock import patch, MagicMock

from engine.rxnorm_client import (
    resolve_rxcui,
    _get_concept_properties,
    get_ingredient_rxcui,
    get_interactions_for_list,
    normalize_medication_list,
    RxConcept,
    RxInteraction,
)


@pytest.fixture(autouse=True)
def clear_caches():
    resolve_rxcui.cache_clear()
    _get_concept_properties.cache_clear()
    yield
    resolve_rxcui.cache_clear()
    _get_concept_properties.cache_clear()


# ── resolve_rxcui ────────────────────────────────────────────────────────────

class TestResolveRxcui:
    def test_empty_name(self):
        assert resolve_rxcui("") is None
        assert resolve_rxcui("   ") is None

    @patch("engine.rxnorm_client.httpx")
    def test_exact_match(self, mock_httpx):
        # First call: exact match
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"idGroup": {"rxnormId": ["12345"]}}
        mock_httpx.get.return_value = mock_resp
        # Patch _get_concept_properties to return a concept
        with patch("engine.rxnorm_client._get_concept_properties") as mock_props:
            mock_props.return_value = RxConcept("12345", "warfarin", "IN")
            result = resolve_rxcui("warfarin_test_1")
            assert result is not None
            assert result.rxcui == "12345"

    @patch("engine.rxnorm_client.httpx")
    def test_exact_no_ids_approx_fallback(self, mock_httpx):
        call_count = [0]
        def side_effect(*args, **kwargs):
            call_count[0] += 1
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            if call_count[0] == 1:  # exact match returns empty
                mock_resp.json.return_value = {"idGroup": {}}
            else:  # approximate match
                mock_resp.json.return_value = {
                    "approximateGroup": {"candidate": [{"rxcui": "999", "score": "90"}]}
                }
            return mock_resp
        mock_httpx.get.side_effect = side_effect
        with patch("engine.rxnorm_client._get_concept_properties") as mock_props:
            mock_props.return_value = RxConcept("999", "test", "IN")
            result = resolve_rxcui("unique_drug_name_2")
            assert result is not None
            assert result.rxcui == "999"

    @patch("engine.rxnorm_client.httpx")
    def test_exact_exception(self, mock_httpx):
        mock_httpx.get.side_effect = Exception("timeout")
        result = resolve_rxcui("unique_drug_3")
        assert result is None

    @patch("engine.rxnorm_client.httpx")
    def test_approx_low_score(self, mock_httpx):
        call_count = [0]
        def side_effect(*args, **kwargs):
            call_count[0] += 1
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            if call_count[0] == 1:
                mock_resp.json.return_value = {"idGroup": {}}
            else:
                mock_resp.json.return_value = {
                    "approximateGroup": {"candidate": [{"rxcui": "111", "score": "30"}]}
                }
            return mock_resp
        mock_httpx.get.side_effect = side_effect
        result = resolve_rxcui("unique_drug_4")
        assert result is None


# ── _get_concept_properties ──────────────────────────────────────────────────

class TestGetConceptProperties:
    @patch("engine.rxnorm_client.httpx")
    def test_success(self, mock_httpx):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "properties": {"rxcui": "123", "name": "Warfarin", "tty": "IN"}
        }
        mock_httpx.get.return_value = mock_resp
        result = _get_concept_properties("123_unique")
        assert result is not None
        assert result.name == "Warfarin"

    @patch("engine.rxnorm_client.httpx")
    def test_non_200(self, mock_httpx):
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        mock_httpx.get.return_value = mock_resp
        result = _get_concept_properties("404_unique")
        assert result is None

    @patch("engine.rxnorm_client.httpx")
    def test_exception(self, mock_httpx):
        mock_httpx.get.side_effect = Exception("fail")
        result = _get_concept_properties("fail_unique")
        assert result is None


# ── get_ingredient_rxcui ─────────────────────────────────────────────────────

class TestGetIngredientRxcui:
    @patch("engine.rxnorm_client.httpx")
    def test_success(self, mock_httpx):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "allRelatedGroup": {
                "conceptGroup": [
                    {"tty": "IN", "conceptProperties": [{"rxcui": "base-123"}]},
                ]
            }
        }
        mock_httpx.get.return_value = mock_resp
        assert get_ingredient_rxcui("456") == "base-123"

    @patch("engine.rxnorm_client.httpx")
    def test_no_ingredient(self, mock_httpx):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"allRelatedGroup": {"conceptGroup": []}}
        mock_httpx.get.return_value = mock_resp
        assert get_ingredient_rxcui("456") == "456"

    @patch("engine.rxnorm_client.httpx")
    def test_exception(self, mock_httpx):
        mock_httpx.get.side_effect = Exception("fail")
        assert get_ingredient_rxcui("456") == "456"


# ── get_interactions_for_list ────────────────────────────────────────────────

class TestGetInteractionsForList:
    def test_less_than_two(self):
        assert get_interactions_for_list(["123"]) == []

    @patch("engine.rxnorm_client.httpx")
    def test_success(self, mock_httpx):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "fullInteractionTypeGroup": [
                {
                    "sourceName": "DrugBank",
                    "fullInteractionType": [
                        {
                            "interactionPair": [
                                {
                                    "interactionConcept": [
                                        {"minConceptItem": {"name": "warfarin", "rxcui": "123"}},
                                        {"minConceptItem": {"name": "ibuprofen", "rxcui": "456"}},
                                    ],
                                    "description": "Serious bleeding risk interaction",
                                    "severity": "N/A",
                                }
                            ]
                        }
                    ],
                }
            ]
        }
        mock_httpx.get.return_value = mock_resp
        results = get_interactions_for_list(["123", "456"])
        assert len(results) == 1
        assert results[0].severity == "serious"

    @patch("engine.rxnorm_client.httpx")
    def test_contraindicated(self, mock_httpx):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "fullInteractionTypeGroup": [
                {
                    "sourceName": "ONCHigh",
                    "fullInteractionType": [
                        {
                            "interactionPair": [
                                {
                                    "interactionConcept": [
                                        {"minConceptItem": {"name": "drug_a", "rxcui": "1"}},
                                        {"minConceptItem": {"name": "drug_b", "rxcui": "2"}},
                                    ],
                                    "description": "Avoid use",
                                    "severity": "contraindicated",
                                }
                            ]
                        }
                    ],
                }
            ]
        }
        mock_httpx.get.return_value = mock_resp
        results = get_interactions_for_list(["1", "2"])
        assert results[0].severity == "contraindicated"

    @patch("engine.rxnorm_client.httpx")
    def test_api_failure(self, mock_httpx):
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_httpx.get.return_value = mock_resp
        assert get_interactions_for_list(["1", "2"]) == []

    @patch("engine.rxnorm_client.httpx")
    def test_exception(self, mock_httpx):
        mock_httpx.get.side_effect = Exception("network error")
        assert get_interactions_for_list(["1", "2"]) == []


# ── normalize_medication_list ────────────────────────────────────────────────

class TestNormalizeMedicationList:
    @patch("engine.rxnorm_client.resolve_rxcui")
    def test_resolves(self, mock_resolve):
        mock_resolve.return_value = RxConcept("123", "warfarin", "IN")
        result = normalize_medication_list(["warfarin"])
        assert result["warfarin"] is not None

    @patch("engine.rxnorm_client.resolve_rxcui")
    def test_unresolved(self, mock_resolve):
        mock_resolve.return_value = None
        result = normalize_medication_list(["unknown_drug_xyz"])
        assert result["unknown_drug_xyz"] is None


# ── resolve_rxcui: exact match rxcui but no concept properties (line 72) ───

class TestResolveRxcuiFallbackConcept:
    """Cover line 72: exact match returns rxcui ID but _get_concept_properties returns None,
    so we fall back to creating a RxConcept with the raw rxcui, name, and tty='IN'."""

    @patch("engine.rxnorm_client.httpx")
    def test_exact_match_no_properties(self, mock_httpx):
        """Line 72: rxcui found via exact match but _get_concept_properties returns None."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"idGroup": {"rxnormId": ["77777"]}}
        mock_httpx.get.return_value = mock_resp

        with patch("engine.rxnorm_client._get_concept_properties", return_value=None):
            result = resolve_rxcui("unique_exact_no_props")
            assert result is not None
            assert result.rxcui == "77777"
            assert result.tty == "IN"


# ── resolve_rxcui: approx match rxcui but no concept properties (line 93) ──

class TestResolveRxcuiApproxFallback:
    """Cover line 93: approximate match returns rxcui but _get_concept_properties returns None."""

    @patch("engine.rxnorm_client.httpx")
    def test_approx_match_no_properties(self, mock_httpx):
        """Line 93: rxcui found via approximate match, properties are None, fallback to raw."""
        call_count = [0]

        def side_effect(*args, **kwargs):
            call_count[0] += 1
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            if call_count[0] == 1:
                # Exact match fails (no rxnormId)
                mock_resp.json.return_value = {"idGroup": {}}
            else:
                # Approximate match succeeds
                mock_resp.json.return_value = {
                    "approximateGroup": {
                        "candidate": [{"rxcui": "88888", "score": "85"}]
                    }
                }
            return mock_resp

        mock_httpx.get.side_effect = side_effect
        with patch("engine.rxnorm_client._get_concept_properties", return_value=None):
            result = resolve_rxcui("unique_approx_no_props")
            assert result is not None
            assert result.rxcui == "88888"
            assert result.tty == "IN"


# ── get_interactions_for_list: len(concepts) < 2 (line 170) ───────────────

class TestInteractionsConceptsParsing:
    """Cover lines 170, 178, 194 in get_interactions_for_list."""

    @patch("engine.rxnorm_client.httpx")
    def test_single_concept_skipped(self, mock_httpx):
        """Line 170: When interactionConcept has only 1 entry, skip (continue)."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "fullInteractionTypeGroup": [
                {
                    "sourceName": "DrugBank",
                    "fullInteractionType": [
                        {
                            "interactionPair": [
                                {
                                    "interactionConcept": [
                                        {"minConceptItem": {"name": "warfarin", "rxcui": "1"}},
                                    ],
                                    "description": "Incomplete",
                                    "severity": "N/A",
                                }
                            ]
                        }
                    ],
                }
            ]
        }
        mock_httpx.get.return_value = mock_resp
        results = get_interactions_for_list(["1", "2"])
        assert results == []

    @patch("engine.rxnorm_client.httpx")
    def test_empty_drug_names_skipped(self, mock_httpx):
        """Line 178: When drug_a or drug_b name is empty, skip (continue)."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "fullInteractionTypeGroup": [
                {
                    "sourceName": "DrugBank",
                    "fullInteractionType": [
                        {
                            "interactionPair": [
                                {
                                    "interactionConcept": [
                                        {"minConceptItem": {"name": "", "rxcui": "1"}},
                                        {"minConceptItem": {"name": "ibuprofen", "rxcui": "2"}},
                                    ],
                                    "description": "Interaction",
                                    "severity": "N/A",
                                }
                            ]
                        }
                    ],
                }
            ]
        }
        mock_httpx.get.return_value = mock_resp
        results = get_interactions_for_list(["1", "2"])
        assert results == []

    @patch("engine.rxnorm_client.httpx")
    def test_moderate_severity(self, mock_httpx):
        """Line 194: Interaction that is neither serious nor contraindicated -> moderate."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "fullInteractionTypeGroup": [
                {
                    "sourceName": "ONCHigh",
                    "fullInteractionType": [
                        {
                            "interactionPair": [
                                {
                                    "interactionConcept": [
                                        {"minConceptItem": {"name": "druga", "rxcui": "10"}},
                                        {"minConceptItem": {"name": "drugb", "rxcui": "20"}},
                                    ],
                                    "description": "Mild pharmacokinetic interaction noted",
                                    "severity": "N/A",
                                }
                            ]
                        }
                    ],
                }
            ]
        }
        mock_httpx.get.return_value = mock_resp
        results = get_interactions_for_list(["10", "20"])
        assert len(results) == 1
        assert results[0].severity == "moderate"
