"""Tests for LLM Clinical Synthesizer — covers template fallbacks, prompts, and abstention."""
import pytest
from unittest.mock import patch, MagicMock

from engine.llm_synthesizer import (
    ClinicalNarrative,
    _build_conflict_prompt,
    _build_handoff_prompt,
    _template_conflict_explanation,
    _template_handoff,
    explain_conflict,
    generate_clinical_handoff,
    _call_medical_llm_sync,
    _SYSTEM_PROMPT,
)


# ── Fixtures ────────────────────────────────────────────────────────────────

CONFLICT = {
    "type": "drug_interaction",
    "severity": "critical",
    "description": "Warfarin + Ibuprofen: increased bleeding risk",
    "recommendation": "Discontinue NSAID or switch to acetaminophen",
}

PATIENT_CTX = {
    "conditions": [{"name": "Atrial Fibrillation"}, {"name": "CKD Stage 3b"}],
    "medications": [{"name": "Warfarin"}, {"name": "Ibuprofen"}],
    "allergies": [{"allergen": "Penicillin"}],
}

EVIDENCE_BLOCKS = [
    {"block_id": "ev-001", "title": "Warfarin Rx", "content": "Patient on warfarin 5mg daily"},
    {"block_id": "ev-002", "title": "Ibuprofen Rx", "content": "Ibuprofen 400mg for knee pain"},
    {"block_id": "ev-003", "title": "INR Result", "content": "INR 3.8, above therapeutic range"},
]

CONTRADICTIONS = [
    {"severity": "critical", "description": "Warfarin + Ibuprofen", "recommendation": "Stop NSAID"},
    {"severity": "high", "description": "Declining GFR + Metformin", "recommendation": "Monitor renal function"},
]

SAFETY_REPORT = {"interaction_count": 2, "allergy_conflict_count": 1}


# ── Prompt Builders ─────────────────────────────────────────────────────────

class TestBuildConflictPrompt:
    def test_includes_conflict_details(self):
        prompt = _build_conflict_prompt(CONFLICT, PATIENT_CTX, EVIDENCE_BLOCKS)
        assert "Warfarin + Ibuprofen" in prompt
        assert "critical" in prompt
        assert "drug_interaction" in prompt

    def test_includes_patient_context(self):
        prompt = _build_conflict_prompt(CONFLICT, PATIENT_CTX, EVIDENCE_BLOCKS)
        assert "2 conditions" in prompt
        assert "2 medications" in prompt
        assert "1 allergies" in prompt
        assert "Atrial Fibrillation" in prompt

    def test_includes_evidence_blocks(self):
        prompt = _build_conflict_prompt(CONFLICT, PATIENT_CTX, EVIDENCE_BLOCKS)
        assert "[ev-001]" in prompt
        assert "[ev-002]" in prompt
        assert "warfarin 5mg" in prompt

    def test_empty_conditions(self):
        ctx = {"conditions": [], "medications": [], "allergies": []}
        prompt = _build_conflict_prompt(CONFLICT, ctx, [])
        assert "0 conditions" in prompt

    def test_includes_recommendation(self):
        prompt = _build_conflict_prompt(CONFLICT, PATIENT_CTX, EVIDENCE_BLOCKS)
        assert "acetaminophen" in prompt


class TestBuildHandoffPrompt:
    def test_includes_medications(self):
        prompt = _build_handoff_prompt(PATIENT_CTX, CONTRADICTIONS, SAFETY_REPORT, EVIDENCE_BLOCKS)
        assert "Warfarin" in prompt
        assert "Ibuprofen" in prompt

    def test_includes_contradictions(self):
        prompt = _build_handoff_prompt(PATIENT_CTX, CONTRADICTIONS, SAFETY_REPORT, EVIDENCE_BLOCKS)
        assert "CRITICAL" in prompt
        assert "Warfarin + Ibuprofen" in prompt
        assert "2 total" in prompt

    def test_includes_safety_counts(self):
        prompt = _build_handoff_prompt(PATIENT_CTX, CONTRADICTIONS, SAFETY_REPORT, EVIDENCE_BLOCKS)
        assert "Drug interactions: 2" in prompt
        assert "Allergy conflicts: 1" in prompt

    def test_empty_contradictions(self):
        prompt = _build_handoff_prompt(PATIENT_CTX, [], SAFETY_REPORT, EVIDENCE_BLOCKS)
        assert "None detected" in prompt

    def test_limits_evidence_blocks(self):
        many_blocks = [{"block_id": f"b-{i}", "title": f"Block {i}", "content": f"Content {i}"} for i in range(30)]
        prompt = _build_handoff_prompt(PATIENT_CTX, CONTRADICTIONS, SAFETY_REPORT, many_blocks)
        # Should limit to 20
        assert "[b-19]" in prompt
        assert "[b-20]" not in prompt

    def test_includes_allergies(self):
        prompt = _build_handoff_prompt(PATIENT_CTX, CONTRADICTIONS, SAFETY_REPORT, EVIDENCE_BLOCKS)
        assert "Penicillin" in prompt


# ── Template Fallbacks ──────────────────────────────────────────────────────

class TestTemplateConflictExplanation:
    def test_includes_severity(self):
        result = _template_conflict_explanation(CONFLICT, PATIENT_CTX)
        assert "[CRITICAL]" in result

    def test_includes_description(self):
        result = _template_conflict_explanation(CONFLICT, PATIENT_CTX)
        assert "Warfarin + Ibuprofen" in result

    def test_includes_comorbidities(self):
        result = _template_conflict_explanation(CONFLICT, PATIENT_CTX)
        assert "Atrial Fibrillation" in result

    def test_includes_recommendation(self):
        result = _template_conflict_explanation(CONFLICT, PATIENT_CTX)
        assert "acetaminophen" in result


class TestTemplateHandoff:
    def test_has_sections(self):
        result = _template_handoff(PATIENT_CTX, CONTRADICTIONS)
        assert "CARE HANDOFF NOTE" in result
        assert "CRITICAL ALERTS:" in result
        assert "ACTIVE MEDICATIONS:" in result
        assert "ACTIVE CONDITIONS:" in result
        assert "ALLERGIES:" in result

    def test_critical_alerts(self):
        result = _template_handoff(PATIENT_CTX, CONTRADICTIONS)
        assert "[CRITICAL]" in result
        assert "[HIGH]" in result

    def test_medications_listed(self):
        ctx = {"medications": [{"name": "Warfarin", "dosage": "5mg"}], "conditions": [], "allergies": []}
        result = _template_handoff(ctx, [])
        assert "Warfarin" in result
        assert "5mg" in result

    def test_no_critical(self):
        result = _template_handoff(PATIENT_CTX, [{"severity": "low", "description": "Minor"}])
        assert "CRITICAL ALERTS:" not in result


# ── LLM Sync Cascade ───────────────────────────────────────────────────────

class TestCallMedicalLlmSync:
    def test_no_keys_returns_none(self):
        with patch.dict("os.environ", {}, clear=True):
            result, model = _call_medical_llm_sync("test", "system")
            assert result is None
            assert model == "none"

    @patch("httpx.post")
    def test_openai_success(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "Clinical analysis: [ev-001] shows risk."}}]
        }
        mock_post.return_value = mock_resp
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}, clear=True):
            result, model = _call_medical_llm_sync("test", "system")
            assert result == "Clinical analysis: [ev-001] shows risk."
            assert model == "OpenAI-GPT-5.4"

    @patch("httpx.post")
    def test_openai_failure_falls_to_gemini(self, mock_post):
        openai_resp = MagicMock()
        openai_resp.status_code = 500
        gemini_resp = MagicMock()
        gemini_resp.status_code = 200
        gemini_resp.json.return_value = {
            "candidates": [{"content": {"parts": [{"text": "Gemini analysis"}]}}]
        }
        mock_post.side_effect = [openai_resp, gemini_resp]
        with patch.dict("os.environ", {"OPENAI_API_KEY": "k1", "GOOGLE_API_KEY": "k2"}, clear=True):
            result, model = _call_medical_llm_sync("test", "system")
            assert result == "Gemini analysis"
            assert model == "Gemini-3.1-Pro"

    @patch("httpx.post")
    def test_openai_exception_falls_to_gemini(self, mock_post):
        gemini_resp = MagicMock()
        gemini_resp.status_code = 200
        gemini_resp.json.return_value = {
            "candidates": [{"content": {"parts": [{"text": "Fallback"}]}}]
        }
        mock_post.side_effect = [Exception("timeout"), gemini_resp]
        with patch.dict("os.environ", {"OPENAI_API_KEY": "k1", "GOOGLE_API_KEY": "k2"}, clear=True):
            result, model = _call_medical_llm_sync("test", "system")
            assert result == "Fallback"
            assert model == "Gemini-3.1-Pro"

    @patch("httpx.post")
    def test_gemini_pro_fails_falls_to_flash(self, mock_post):
        pro_resp = MagicMock()
        pro_resp.status_code = 429
        flash_resp = MagicMock()
        flash_resp.status_code = 200
        flash_resp.json.return_value = {
            "candidates": [{"content": {"parts": [{"text": "Flash result"}]}}]
        }
        mock_post.side_effect = [pro_resp, flash_resp]
        with patch.dict("os.environ", {"GOOGLE_API_KEY": "k2"}, clear=True):
            result, model = _call_medical_llm_sync("test", "system")
            assert result == "Flash result"
            assert model == "Gemini-3.1-Flash-Lite"

    @patch("httpx.post")
    def test_all_fail_returns_none(self, mock_post):
        mock_post.side_effect = Exception("all down")
        with patch.dict("os.environ", {"OPENAI_API_KEY": "k1", "GOOGLE_API_KEY": "k2"}, clear=True):
            result, model = _call_medical_llm_sync("test", "system")
            assert result is None
            assert model == "none"

    @patch("httpx.post")
    def test_openai_empty_text(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"choices": [{"message": {"content": ""}}]}
        mock_post.return_value = mock_resp
        with patch.dict("os.environ", {"OPENAI_API_KEY": "k1"}, clear=True):
            result, model = _call_medical_llm_sync("test", "system")
            assert result is None

    @patch("httpx.post")
    def test_gemini_empty_candidates(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"candidates": []}
        mock_post.return_value = mock_resp
        with patch.dict("os.environ", {"GOOGLE_API_KEY": "k2"}, clear=True):
            result, model = _call_medical_llm_sync("test", "system")
            assert result is None

    @patch("httpx.post")
    def test_gemini_no_text_in_parts(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"candidates": [{"content": {"parts": [{"text": ""}]}}]}
        mock_post.return_value = mock_resp
        with patch.dict("os.environ", {"GOOGLE_API_KEY": "k2"}, clear=True):
            result, model = _call_medical_llm_sync("test", "system")
            assert result is None


# ── explain_conflict ────────────────────────────────────────────────────────

class TestExplainConflict:
    def test_abstention_insufficient_evidence(self):
        result = explain_conflict(CONFLICT, PATIENT_CTX, [], confidence_threshold=0.3)
        assert result.abstained is True
        assert "ABSTAIN" in result.narrative
        assert result.model_used == "abstention_gate"
        assert result.confidence_score == 0.0

    def test_abstention_single_block(self):
        result = explain_conflict(CONFLICT, PATIENT_CTX, EVIDENCE_BLOCKS[:1], confidence_threshold=0.5)
        assert result.abstained is True

    def test_template_fallback_no_api_keys(self):
        with patch.dict("os.environ", {}, clear=True):
            result = explain_conflict(CONFLICT, PATIENT_CTX, EVIDENCE_BLOCKS)
            assert result.model_used == "template_fallback"
            assert result.abstained is False
            assert "CRITICAL" in result.narrative

    @patch("engine.llm_synthesizer._call_medical_llm_sync")
    def test_llm_success_with_citations(self, mock_llm):
        mock_llm.return_value = ("Risk: [ev-001] warfarin + [ev-002] ibuprofen.", "OpenAI-GPT-5.4")
        result = explain_conflict(CONFLICT, PATIENT_CTX, EVIDENCE_BLOCKS)
        assert result.abstained is False
        assert result.model_used == "OpenAI-GPT-5.4"
        assert len(result.evidence_citations) == 2
        assert result.confidence_score == 1.0

    @patch("engine.llm_synthesizer._call_medical_llm_sync")
    def test_llm_abstains(self, mock_llm):
        mock_llm.return_value = ("ABSTAIN: Insufficient evidence.", "OpenAI-GPT-5.4")
        result = explain_conflict(CONFLICT, PATIENT_CTX, EVIDENCE_BLOCKS)
        assert result.abstained is True
        assert result.model_used == "OpenAI-GPT-5.4"

    @patch("engine.llm_synthesizer._call_medical_llm_sync")
    def test_citations_only_valid_block_ids(self, mock_llm):
        mock_llm.return_value = ("Risk: [ev-001] real and [fake-999] not real.", "GPT")
        result = explain_conflict(CONFLICT, PATIENT_CTX, EVIDENCE_BLOCKS)
        # Only ev-001 should be in citations, not fake-999
        ids = [c["block_id"] for c in result.evidence_citations]
        assert "ev-001" in ids
        assert "fake-999" not in ids

    def test_confidence_calculation(self):
        # 2 blocks → confidence = 2/3 ≈ 0.667
        with patch.dict("os.environ", {}, clear=True):
            result = explain_conflict(CONFLICT, PATIENT_CTX, EVIDENCE_BLOCKS[:2])
            assert abs(result.confidence_score - 2 / 3) < 0.01


# ── generate_clinical_handoff ───────────────────────────────────────────────

class TestGenerateClinicalHandoff:
    def test_abstention_insufficient_evidence(self):
        result = generate_clinical_handoff(PATIENT_CTX, CONTRADICTIONS, SAFETY_REPORT, [])
        assert result.abstained is True
        assert "ABSTAIN" in result.narrative

    def test_template_fallback_no_keys(self):
        with patch.dict("os.environ", {}, clear=True):
            blocks = [{"block_id": f"b-{i}", "title": f"T{i}", "content": f"C{i}"} for i in range(5)]
            result = generate_clinical_handoff(PATIENT_CTX, CONTRADICTIONS, SAFETY_REPORT, blocks)
            assert result.model_used == "template_fallback"
            assert "CARE HANDOFF NOTE" in result.narrative

    @patch("engine.llm_synthesizer._call_medical_llm_sync")
    def test_llm_success(self, mock_llm):
        mock_llm.return_value = ("Handoff: [b-0] shows risk. [b-1] confirms.", "Gemini-3.1-Pro")
        blocks = [{"block_id": f"b-{i}", "title": f"T{i}", "content": f"C{i}"} for i in range(5)]
        result = generate_clinical_handoff(PATIENT_CTX, CONTRADICTIONS, SAFETY_REPORT, blocks)
        assert result.model_used == "Gemini-3.1-Pro"
        assert result.abstained is False
        assert len(result.evidence_citations) == 2

    @patch("engine.llm_synthesizer._call_medical_llm_sync")
    def test_llm_abstains(self, mock_llm):
        mock_llm.return_value = ("ABSTAIN: Not enough data.", "GPT")
        blocks = [{"block_id": f"b-{i}", "title": f"T{i}", "content": f"C{i}"} for i in range(5)]
        result = generate_clinical_handoff(PATIENT_CTX, CONTRADICTIONS, SAFETY_REPORT, blocks)
        assert result.abstained is True

    def test_handoff_confidence_calculation(self):
        # 3 blocks → confidence = 3/5 = 0.6
        with patch.dict("os.environ", {}, clear=True):
            blocks = [{"block_id": f"b-{i}", "title": f"T{i}", "content": f"C{i}"} for i in range(3)]
            result = generate_clinical_handoff(PATIENT_CTX, CONTRADICTIONS, SAFETY_REPORT, blocks)
            assert abs(result.confidence_score - 0.6) < 0.01


# ── ClinicalNarrative dataclass ─────────────────────────────────────────────

class TestClinicalNarrative:
    def test_frozen(self):
        n = ClinicalNarrative(
            narrative="test", evidence_citations=[], confidence_score=0.5,
            abstained=False, model_used="test", audit_context={},
        )
        with pytest.raises(AttributeError):
            n.narrative = "changed"

    def test_fields(self):
        n = ClinicalNarrative(
            narrative="text", evidence_citations=[{"id": "1"}],
            confidence_score=0.9, abstained=False, model_used="GPT",
            audit_context={"key": "val"},
        )
        assert n.narrative == "text"
        assert n.confidence_score == 0.9
        assert n.model_used == "GPT"
