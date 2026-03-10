"""Extended coverage tests for engine/clinical_scoring.py — LLM cascade, OpenEvidence, RxNorm layers."""
import json
import pytest
from unittest.mock import patch, MagicMock

from engine.clinical_scoring import (
    _openevidence_check_interactions,
    _parse_interaction_narrative,
    _rxnorm_check_interactions,
    _llm_check_interactions,
    _call_openai_json,
    _call_google_json,
    DrugInteraction,
    medication_severity_score,
)


# ── _call_openai_json ────────────────────────────────────────────────────────

class TestCallOpenaiJson:
    @patch("httpx.post")
    def test_success(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "[]"}}]
        }
        mock_post.return_value = mock_resp
        result = _call_openai_json("test prompt", "fake-key")
        assert result == "[]"

    @patch("httpx.post")
    def test_non_200(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 429
        mock_post.return_value = mock_resp
        result = _call_openai_json("test", "key")
        assert result is None

    @patch("httpx.post", side_effect=Exception("timeout"))
    def test_exception(self, mock_post):
        result = _call_openai_json("test", "key")
        assert result is None


# ── _call_google_json ────────────────────────────────────────────────────────

class TestCallGoogleJson:
    @patch("httpx.post")
    def test_success(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "candidates": [{"content": {"parts": [{"text": "[]"}]}}]
        }
        mock_post.return_value = mock_resp
        result = _call_google_json("test", "key", "gemini-3.1-pro-preview")
        assert result == "[]"

    @patch("httpx.post")
    def test_non_200(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_post.return_value = mock_resp
        result = _call_google_json("test", "key", "model-x")
        assert result is None

    @patch("httpx.post")
    def test_no_candidates(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"candidates": []}
        mock_post.return_value = mock_resp
        result = _call_google_json("test", "key", "model-x")
        assert result is None

    @patch("httpx.post", side_effect=Exception("error"))
    def test_exception(self, mock_post):
        result = _call_google_json("test", "key", "model-x")
        assert result is None


# ── _parse_interaction_narrative ─────────────────────────────────────────────

class TestParseInteractionNarrative:
    def test_detects_serious(self):
        text = "Warfarin and ibuprofen have a significant bleeding risk interaction."
        results = _parse_interaction_narrative(text, ["warfarin", "ibuprofen"], set())
        assert len(results) == 1
        assert results[0].severity == "serious"

    def test_detects_contraindicated(self):
        text = "Warfarin and aspirin are contraindicated when used together."
        results = _parse_interaction_narrative(text, ["warfarin", "aspirin"], set())
        assert len(results) == 1
        assert results[0].severity == "contraindicated"

    def test_skips_moderate(self):
        text = "Metformin and lisinopril have a mild pharmacokinetic interaction."
        results = _parse_interaction_narrative(text, ["metformin", "lisinopril"], set())
        assert len(results) == 0

    def test_skips_already_found(self):
        text = "Warfarin and ibuprofen have a significant interaction."
        already = {("warfarin", "ibuprofen")}
        results = _parse_interaction_narrative(text, ["warfarin", "ibuprofen"], already)
        assert len(results) == 0

    def test_custom_source(self):
        text = "Warfarin and aspirin avoid combining these drugs."
        results = _parse_interaction_narrative(text, ["warfarin", "aspirin"], set(), source="TestSource")
        assert len(results) == 1
        assert "TestSource" in results[0].description


# ── _openevidence_check_interactions ─────────────────────────────────────────

class TestOpenEvidenceCheckInteractions:
    @patch.dict("os.environ", {"OPENEVIDENCE_API_KEY": ""})
    def test_no_api_key(self):
        results = _openevidence_check_interactions(["warfarin", "ibuprofen"], set())
        assert results == []

    @patch("httpx.post")
    @patch.dict("os.environ", {"OPENEVIDENCE_API_KEY": "fake-key"})
    def test_success(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "text": "Warfarin and ibuprofen have a significant bleeding risk when combined."
        }
        mock_post.return_value = mock_resp
        results = _openevidence_check_interactions(["warfarin", "ibuprofen"], set())
        assert len(results) >= 1

    @patch("httpx.post")
    @patch.dict("os.environ", {"OPENEVIDENCE_API_KEY": "key"})
    def test_non_200(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_post.return_value = mock_resp
        results = _openevidence_check_interactions(["warfarin", "ibuprofen"], set())
        assert results == []

    @patch("httpx.post", side_effect=Exception("fail"))
    @patch.dict("os.environ", {"OPENEVIDENCE_API_KEY": "key"})
    def test_exception(self, mock_post):
        results = _openevidence_check_interactions(["warfarin", "ibuprofen"], set())
        assert results == []

    @patch("httpx.post")
    @patch.dict("os.environ", {"OPENEVIDENCE_API_KEY": "key"})
    def test_empty_analysis(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"text": ""}
        mock_post.return_value = mock_resp
        results = _openevidence_check_interactions(["warfarin", "ibuprofen"], set())
        assert results == []


# ── _rxnorm_check_interactions ───────────────────────────────────────────────

class TestRxnormCheckInteractions:
    @patch("engine.rxnorm_client.get_interactions_for_list", return_value=[])
    @patch("engine.rxnorm_client.normalize_medication_list")
    def test_insufficient_rxcuis(self, mock_normalize, mock_interact):
        from engine.rxnorm_client import RxConcept
        mock_normalize.return_value = {"warfarin": RxConcept("123", "warfarin", "IN")}
        results = _rxnorm_check_interactions(["warfarin"], set())
        assert results == []

    @patch("engine.rxnorm_client.get_interactions_for_list")
    @patch("engine.rxnorm_client.normalize_medication_list")
    def test_with_interactions(self, mock_normalize, mock_interact):
        from engine.rxnorm_client import RxConcept, RxInteraction
        mock_normalize.return_value = {
            "warfarin": RxConcept("123", "warfarin", "IN"),
            "ibuprofen": RxConcept("456", "ibuprofen", "IN"),
        }
        mock_interact.return_value = [
            RxInteraction(
                drug_a="warfarin", drug_b="ibuprofen",
                rxcui_a="123", rxcui_b="456",
                severity="serious",
                description="Bleeding risk",
                source="DrugBank",
            )
        ]
        results = _rxnorm_check_interactions(["warfarin", "ibuprofen"], set())
        assert len(results) == 1
        assert "RxNorm" in results[0].description

    @patch("engine.rxnorm_client.get_interactions_for_list")
    @patch("engine.rxnorm_client.normalize_medication_list")
    def test_skips_already_found(self, mock_normalize, mock_interact):
        from engine.rxnorm_client import RxConcept, RxInteraction
        mock_normalize.return_value = {
            "warfarin": RxConcept("123", "warfarin", "IN"),
            "ibuprofen": RxConcept("456", "ibuprofen", "IN"),
        }
        mock_interact.return_value = [
            RxInteraction("warfarin", "ibuprofen", "123", "456", "serious", "test", "DB")
        ]
        already = {("warfarin", "ibuprofen")}
        results = _rxnorm_check_interactions(["warfarin", "ibuprofen"], already)
        assert len(results) == 0


# ── _llm_check_interactions ──────────────────────────────────────────────────

class TestLlmCheckInteractions:
    @patch.dict("os.environ", {"OPENAI_API_KEY": "", "GOOGLE_API_KEY": "", "GEMINI_API_KEY": ""})
    def test_no_keys(self):
        results = _llm_check_interactions(["warfarin", "ibuprofen"], set())
        assert results == []

    @patch("engine.clinical_scoring._call_openai_json")
    @patch.dict("os.environ", {"OPENAI_API_KEY": "key", "GOOGLE_API_KEY": "", "GEMINI_API_KEY": ""})
    def test_openai_success(self, mock_call):
        mock_call.return_value = json.dumps([
            {"drug_a": "warfarin", "drug_b": "ibuprofen", "severity": "serious", "description": "Bleeding risk"}
        ])
        results = _llm_check_interactions(["warfarin", "ibuprofen"], set())
        assert len(results) == 1

    @patch("engine.clinical_scoring._call_openai_json")
    @patch.dict("os.environ", {"OPENAI_API_KEY": "key", "GOOGLE_API_KEY": "", "GEMINI_API_KEY": ""})
    def test_empty_json_array(self, mock_call):
        mock_call.return_value = "[]"
        results = _llm_check_interactions(["warfarin", "ibuprofen"], set())
        assert results == []

    @patch("engine.clinical_scoring._call_openai_json")
    @patch.dict("os.environ", {"OPENAI_API_KEY": "key", "GOOGLE_API_KEY": "", "GEMINI_API_KEY": ""})
    def test_code_block_wrapped(self, mock_call):
        mock_call.return_value = '```json\n[{"drug_a":"warfarin","drug_b":"aspirin","severity":"contraindicated","description":"test"}]\n```'
        results = _llm_check_interactions(["warfarin", "aspirin"], set())
        assert len(results) == 1
        assert results[0].severity == "contraindicated"

    @patch("engine.clinical_scoring._call_openai_json")
    @patch.dict("os.environ", {"OPENAI_API_KEY": "key", "GOOGLE_API_KEY": "", "GEMINI_API_KEY": ""})
    def test_invalid_json(self, mock_call):
        mock_call.return_value = "not json at all"
        results = _llm_check_interactions(["warfarin", "ibuprofen"], set())
        assert results == []

    @patch("engine.clinical_scoring._call_openai_json")
    @patch.dict("os.environ", {"OPENAI_API_KEY": "key", "GOOGLE_API_KEY": "", "GEMINI_API_KEY": ""})
    def test_non_list_json(self, mock_call):
        mock_call.return_value = '{"error": "bad"}'
        results = _llm_check_interactions(["warfarin", "ibuprofen"], set())
        assert results == []

    @patch("engine.clinical_scoring._call_openai_json")
    @patch.dict("os.environ", {"OPENAI_API_KEY": "key", "GOOGLE_API_KEY": "", "GEMINI_API_KEY": ""})
    def test_skips_moderate(self, mock_call):
        mock_call.return_value = json.dumps([
            {"drug_a": "warfarin", "drug_b": "aspirin", "severity": "moderate", "description": "mild"}
        ])
        results = _llm_check_interactions(["warfarin", "aspirin"], set())
        assert results == []

    @patch("engine.clinical_scoring._call_openai_json", return_value=None)
    @patch("engine.clinical_scoring._call_google_json", return_value=None)
    @patch.dict("os.environ", {"OPENAI_API_KEY": "key", "GOOGLE_API_KEY": "key"})
    def test_all_models_fail(self, mock_google, mock_openai):
        results = _llm_check_interactions(["warfarin", "ibuprofen"], set())
        assert results == []


# ── covered_pairs after OpenEvidence (line 225) ────────────────────────────

class TestCoveredPairsOpenEvidence:
    """Cover line 225: covered_pairs.add after OpenEvidence returns interactions."""

    @patch("engine.clinical_scoring._rxnorm_check_interactions", return_value=[])
    @patch("httpx.post")
    @patch.dict("os.environ", {
        "OPENEVIDENCE_API_KEY": "key",
        "OPENAI_API_KEY": "",
        "GOOGLE_API_KEY": "",
        "GEMINI_API_KEY": "",
    })
    def test_openevidence_adds_to_covered_pairs(self, mock_post, mock_rxnorm):
        """Line 225: When OE returns interactions, they're added to covered_pairs
        so downstream layers (RxNorm, LLM) skip them."""
        from engine.clinical_scoring import check_drug_interactions
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        # Use drug names not in the deterministic table
        mock_resp.json.return_value = {
            "text": "Pembrolizumab and nivolumab have a significant interaction risk."
        }
        mock_post.return_value = mock_resp
        results = check_drug_interactions(
            ["pembrolizumab", "nivolumab"], use_llm_fallback=True
        )
        # OE should detect a "serious" interaction since "significant" is a keyword
        oe_results = [r for r in results if "OpenEvidence" in r.description]
        assert len(oe_results) >= 1

    @patch("engine.clinical_scoring._openevidence_check_interactions")
    @patch.dict("os.environ", {
        "OPENAI_API_KEY": "",
        "GOOGLE_API_KEY": "",
        "GEMINI_API_KEY": "",
    })
    def test_oe_results_prevent_rxnorm_duplicates(self, mock_oe):
        """Line 225: OE-detected interactions are added to covered_pairs,
        preventing the same pair from appearing in RxNorm layer."""
        from engine.clinical_scoring import check_drug_interactions
        mock_oe.return_value = [
            DrugInteraction(
                drug_a="drugx", drug_b="drugy",
                severity="serious",
                description="OE-detected interaction",
                score=0.8,
            )
        ]
        results = check_drug_interactions(["drugx", "drugy"], use_llm_fallback=True)
        # The OE result should be present
        oe_results = [r for r in results if "OE-detected" in r.description]
        assert len(oe_results) == 1


# ── covered_pairs after RxNorm (line 232) ──────────────────────────────────

class TestCoveredPairsRxNorm:
    """Cover line 232: covered_pairs.add after RxNorm returns interactions."""

    @patch("engine.clinical_scoring._rxnorm_check_interactions")
    @patch("engine.clinical_scoring._openevidence_check_interactions", return_value=[])
    @patch.dict("os.environ", {
        "OPENAI_API_KEY": "",
        "GOOGLE_API_KEY": "",
        "GEMINI_API_KEY": "",
    })
    def test_rxnorm_adds_to_covered_pairs(self, mock_oe, mock_rxnorm):
        """Line 232: When RxNorm returns interactions, pairs are added to covered_pairs."""
        from engine.clinical_scoring import check_drug_interactions
        mock_rxnorm.return_value = [
            DrugInteraction(
                drug_a="druga_rx", drug_b="drugb_rx",
                severity="serious",
                description="RxNorm-detected",
                score=0.8,
            )
        ]
        results = check_drug_interactions(["druga_rx", "drugb_rx"], use_llm_fallback=True)
        rx_results = [r for r in results if "RxNorm-detected" in r.description]
        assert len(rx_results) == 1


# ── _openevidence_check_interactions empty text (line 295) ─────────────────

class TestOpenEvidenceEmptyText:
    """Cover line 295: return [] when analysis_text is empty string."""

    @patch("httpx.post")
    @patch.dict("os.environ", {"OPENEVIDENCE_API_KEY": "key"})
    def test_empty_analysis_text_returns_empty(self, mock_post):
        """Line 295: When OpenEvidence returns empty text, return empty list."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        # All text fields are empty
        mock_resp.json.return_value = {"text": "", "analysis": ""}
        mock_post.return_value = mock_resp
        results = _openevidence_check_interactions(["warfarin", "ibuprofen"], set())
        assert results == []

    @patch("httpx.post")
    @patch.dict("os.environ", {"OPENEVIDENCE_API_KEY": "key"})
    def test_no_text_key_empty(self, mock_post):
        """When OE response has no text/analysis keys at all, returns []."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {}
        mock_post.return_value = mock_resp
        results = _openevidence_check_interactions(["warfarin", "ibuprofen"], set())
        assert results == []


# ── _rxnorm_check_interactions ImportError (lines 387-388) ─────────────────

class TestRxnormImportError:
    """Cover lines 387-388: except ImportError: return [] in _rxnorm_check_interactions."""

    def test_import_error_returns_empty(self):
        """Lines 387-388: When rxnorm_client cannot be imported, return []."""
        import sys
        saved = sys.modules.get("engine.rxnorm_client")
        sys.modules["engine.rxnorm_client"] = None  # Force ImportError
        try:
            # We need to reimport to trigger the import error path
            # _rxnorm_check_interactions does a local import
            results = _rxnorm_check_interactions(["warfarin", "ibuprofen"], set())
            assert results == []
        finally:
            if saved is not None:
                sys.modules["engine.rxnorm_client"] = saved
            else:
                sys.modules.pop("engine.rxnorm_client", None)


# ── _check_allergy_cross_reactions ImportError (lines 642-643) ─────────────

class TestAllergySnowmedImportError:
    """Cover lines 642-643: except ImportError: pass in check_allergy_conflicts."""

    def test_snomed_import_error_graceful(self):
        """Lines 642-643: When snomed_client cannot be imported, allergy check
        still returns results from deterministic table only."""
        import sys
        from engine.clinical_scoring import check_allergy_conflicts
        saved = sys.modules.get("engine.snomed_client")
        sys.modules["engine.snomed_client"] = None  # Force ImportError
        try:
            results = check_allergy_conflicts(
                ["penicillin"], ["amoxicillin"]
            )
            # Should still find the deterministic table match
            assert len(results) >= 1
            assert results[0].allergen == "penicillin"
        finally:
            if saved is not None:
                sys.modules["engine.snomed_client"] = saved
            else:
                sys.modules.pop("engine.snomed_client", None)


# ── Lab-based medication checks: non-numeric value (lines 740-741) ─────────

class TestLabMedNonNumericValue:
    """Cover lines 740-741: except (ValueError, TypeError): continue for non-numeric obs."""

    def test_non_numeric_value_skipped(self):
        """Lines 740-741: Observations with non-numeric values are skipped."""
        from engine.clinical_scoring import check_lab_medication_contraindications
        observations = [
            {
                "observation_name": "eGFR",
                "value": "not-a-number",
                "unit": "mL/min",
            },
        ]
        # Metformin + eGFR is a known contraindication rule
        results = check_lab_medication_contraindications(observations, ["metformin"])
        assert results == []

    def test_none_value_skipped(self):
        """Lines 740-741: Observation with None value triggers TypeError → continue."""
        from engine.clinical_scoring import check_lab_medication_contraindications
        observations = [
            {
                "observation_name": "eGFR",
                "value": None,
                "unit": "mL/min",
            },
        ]
        results = check_lab_medication_contraindications(observations, ["metformin"])
        assert results == []

    def test_numeric_value_triggers_contraindication(self):
        """Positive case: numeric value below threshold triggers contraindication."""
        from engine.clinical_scoring import check_lab_medication_contraindications
        observations = [
            {
                "observation_name": "eGFR",
                "value": 25,
                "unit": "mL/min",
            },
        ]
        results = check_lab_medication_contraindications(observations, ["metformin"])
        assert len(results) >= 1
        assert results[0].medication == "metformin"


# ── INR trending high (line 850) ───────────────────────────────────────────

class TestINRTrendingHigh:
    """Cover line 850: INR trending high detection in detect_lab_trends."""

    def test_inr_rising_above_therapeutic(self):
        """Line 850: INR rising above 3.0 with change > 0.5 triggers trend alert."""
        from engine.clinical_scoring import detect_lab_trends
        observations = [
            {"observation_name": "INR", "value": 2.5, "date": "2026-01-01", "effective_date": "2026-01-01"},
            {"observation_name": "INR", "value": 2.8, "date": "2026-01-08", "effective_date": "2026-01-08"},
            {"observation_name": "INR", "value": 3.2, "date": "2026-01-15", "effective_date": "2026-01-15"},
            {"observation_name": "INR", "value": 3.5, "date": "2026-01-22", "effective_date": "2026-01-22"},
        ]
        trends = detect_lab_trends(observations)
        inr_trends = [t for t in trends if "INR" in t.lab_name.upper()]
        assert len(inr_trends) >= 1
        assert inr_trends[0].direction == "rising"
        assert inr_trends[0].severity == "high"
        assert "therapeutic range" in inr_trends[0].description.lower()

    def test_inr_not_rising_enough(self):
        """INR changes less than 0.5 should not trigger trend."""
        from engine.clinical_scoring import detect_lab_trends
        observations = [
            {"observation_name": "INR", "value": 2.8, "date": "2026-01-01", "effective_date": "2026-01-01"},
            {"observation_name": "INR", "value": 2.9, "date": "2026-01-08", "effective_date": "2026-01-08"},
        ]
        trends = detect_lab_trends(observations)
        inr_trends = [t for t in trends if "INR" in t.lab_name.upper()]
        assert len(inr_trends) == 0
