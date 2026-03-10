"""Extended coverage tests for engine/llm_synthesizer.py — sync LLM cascade, handoff, async."""
import importlib
import json
import sys
import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from engine.llm_synthesizer import (
    _call_medical_llm_async,
    _call_medical_llm_sync,
    _template_conflict_explanation,
    _template_handoff,
    explain_conflict,
    generate_clinical_handoff,
    ClinicalNarrative,
)


# ── _call_medical_llm_sync ──────────────────────────────────────────────────

class TestCallMedicalLlmSync:
    @patch.dict("os.environ", {"OPENAI_API_KEY": "", "GOOGLE_API_KEY": "", "GEMINI_API_KEY": ""})
    def test_no_keys(self):
        text, model = _call_medical_llm_sync("test", "system")
        assert text is None
        assert model == "none"

    @patch("httpx.post")
    @patch.dict("os.environ", {"OPENAI_API_KEY": "key", "GOOGLE_API_KEY": ""})
    def test_openai_success(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "LLM response text"}}]
        }
        mock_post.return_value = mock_resp
        text, model = _call_medical_llm_sync("test", "system")
        assert text == "LLM response text"
        assert model == "OpenAI-GPT-5.4"

    @patch("httpx.post")
    @patch.dict("os.environ", {"OPENAI_API_KEY": "", "GOOGLE_API_KEY": "key"})
    def test_gemini_success(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "candidates": [{"content": {"parts": [{"text": "Gemini says hello"}]}}]
        }
        mock_post.return_value = mock_resp
        text, model = _call_medical_llm_sync("test", "system")
        assert text == "Gemini says hello"
        assert "Gemini" in model

    @patch("httpx.post", side_effect=Exception("fail"))
    @patch.dict("os.environ", {"OPENAI_API_KEY": "key", "GOOGLE_API_KEY": ""})
    def test_openai_exception_fallthrough(self, mock_post):
        text, model = _call_medical_llm_sync("test", "system")
        assert text is None
        assert model == "none"

    @patch("httpx.post")
    @patch.dict("os.environ", {"OPENAI_API_KEY": "key", "GOOGLE_API_KEY": "key"})
    def test_openai_non_200_falls_to_gemini(self, mock_post):
        call_count = [0]
        def side_effect(*args, **kwargs):
            call_count[0] += 1
            mock_resp = MagicMock()
            if call_count[0] == 1:
                mock_resp.status_code = 500
            else:
                mock_resp.status_code = 200
                mock_resp.json.return_value = {
                    "candidates": [{"content": {"parts": [{"text": "Gemini fallback"}]}}]
                }
            return mock_resp
        mock_post.side_effect = side_effect
        text, model = _call_medical_llm_sync("test", "system")
        assert text == "Gemini fallback"
        assert "Gemini" in model


# ── _template_conflict_explanation ───────────────────────────────────────────

class TestTemplateConflictExplanation:
    def test_basic(self):
        conflict = {"severity": "critical", "description": "Warfarin + ibuprofen", "recommendation": "Stop ibuprofen"}
        patient = {"conditions": [{"name": "Diabetes"}, {"name": "CKD"}]}
        text = _template_conflict_explanation(conflict, patient)
        assert "CRITICAL" in text
        assert "Diabetes" in text


# ── _template_handoff ────────────────────────────────────────────────────────

class TestTemplateHandoff:
    def test_basic_handoff(self):
        patient = {
            "medications": [{"name": "Warfarin", "dosage": "5mg"}],
            "conditions": [{"name": "AF"}],
            "allergies": [{"allergen": "Penicillin", "criticality": "high"}],
        }
        contradictions = [
            {"severity": "critical", "description": "Allergy conflict", "recommendation": "Stop drug"},
        ]
        text = _template_handoff(patient, contradictions)
        assert "CARE HANDOFF" in text
        assert "CRITICAL ALERTS" in text
        assert "Warfarin" in text
        assert "Penicillin" in text

    def test_no_critical_findings(self):
        patient = {"medications": [], "conditions": [], "allergies": []}
        text = _template_handoff(patient, [])
        assert "CARE HANDOFF" in text


# ── explain_conflict (full pipeline) ─────────────────────────────────────────

class TestExplainConflict:
    def test_insufficient_evidence_abstains(self):
        conflict = {"type": "test", "severity": "high", "description": "test"}
        patient = {"conditions": []}
        result = explain_conflict(conflict, patient, [])  # 0 evidence blocks
        assert result.abstained is True

    @patch("engine.llm_synthesizer._call_medical_llm_sync", return_value=(None, "none"))
    def test_llm_unavailable_falls_to_template(self, mock_llm):
        conflict = {"type": "test", "severity": "high", "description": "Drug conflict", "recommendation": "Review"}
        patient = {"conditions": [{"name": "DM"}]}
        evidence = [{"block_id": "b1", "content": "med data"}]
        result = explain_conflict(conflict, patient, evidence)
        assert result.model_used == "template_fallback"
        assert result.abstained is False

    @patch("engine.llm_synthesizer._call_medical_llm_sync")
    def test_llm_returns_abstain(self, mock_llm):
        mock_llm.return_value = ("ABSTAIN: Insufficient data", "OpenAI-GPT-5.4")
        conflict = {"type": "test", "severity": "high", "description": "test"}
        patient = {"conditions": []}
        evidence = [{"block_id": "b1"}, {"block_id": "b2"}, {"block_id": "b3"}]
        result = explain_conflict(conflict, patient, evidence)
        assert result.abstained is True
        assert result.model_used == "OpenAI-GPT-5.4"

    @patch("engine.llm_synthesizer._call_medical_llm_sync")
    def test_llm_success_with_citations(self, mock_llm):
        mock_llm.return_value = (
            "Patient has warfarin allergy conflict [b1] and bleeding risk [b2].",
            "OpenAI-GPT-5.4",
        )
        conflict = {"type": "allergy_medication_conflict", "severity": "critical", "description": "test"}
        patient = {"conditions": []}
        evidence = [
            {"block_id": "b1", "content": "allergy data"},
            {"block_id": "b2", "content": "med data"},
        ]
        result = explain_conflict(conflict, patient, evidence)
        assert result.abstained is False
        assert result.model_used == "OpenAI-GPT-5.4"
        assert len(result.evidence_citations) == 2

    @patch("engine.llm_synthesizer._call_medical_llm_sync")
    def test_citations_only_valid_block_ids(self, mock_llm):
        mock_llm.return_value = (
            "Issue [b1] [nonexistent_id] found",
            "Gemini-3.1-Pro",
        )
        evidence = [{"block_id": "b1"}]
        result = explain_conflict({"type": "t"}, {}, evidence)
        assert len(result.evidence_citations) == 1
        assert result.evidence_citations[0]["block_id"] == "b1"


# ── generate_clinical_handoff (full pipeline) ────────────────────────────────

class TestGenerateClinicalHandoff:
    def test_insufficient_evidence_abstains(self):
        result = generate_clinical_handoff({}, [], {}, [])
        assert result.abstained is True

    @patch("engine.llm_synthesizer._call_medical_llm_sync", return_value=(None, "none"))
    def test_llm_unavailable_falls_to_template(self, mock_llm):
        patient = {"medications": [{"name": "M1", "dosage": ""}], "conditions": [], "allergies": []}
        evidence = [{"block_id": f"b{i}"} for i in range(5)]
        result = generate_clinical_handoff(patient, [], {}, evidence)
        assert result.model_used == "template_fallback"
        assert "CARE HANDOFF" in result.narrative

    @patch("engine.llm_synthesizer._call_medical_llm_sync")
    def test_llm_success(self, mock_llm):
        mock_llm.return_value = ("Full handoff note [b1] [b2]", "OpenAI-GPT-5.4")
        evidence = [{"block_id": f"b{i}"} for i in range(5)]
        result = generate_clinical_handoff({}, [], {}, evidence)
        assert result.abstained is False
        assert len(result.evidence_citations) == 2

    @patch("engine.llm_synthesizer._call_medical_llm_sync")
    def test_llm_returns_abstain(self, mock_llm):
        mock_llm.return_value = ("ABSTAIN: Insufficient data for handoff", "Gemini-3.1-Pro")
        evidence = [{"block_id": f"b{i}"} for i in range(5)]
        result = generate_clinical_handoff({}, [], {}, evidence)
        assert result.abstained is True


# ── _call_medical_llm_async (lines 131-201) ─────────────────────────────────

class TestCallMedicalLlmAsync:
    """Cover the entire async cascade function (lines 133-201)."""

    @patch.dict(
        "os.environ",
        {"OPENAI_API_KEY": "", "GOOGLE_API_KEY": "", "GEMINI_API_KEY": ""},
    )
    async def test_no_keys_returns_none(self):
        """Line 136-137: both keys absent → (None, 'none')."""
        text, model = await _call_medical_llm_async("prompt", "system")
        assert text is None
        assert model == "none"

    @patch.dict(
        "os.environ",
        {"OPENAI_API_KEY": "test-key", "GOOGLE_API_KEY": "", "GEMINI_API_KEY": ""},
    )
    @patch.dict("sys.modules", {"httpx": None})
    async def test_httpx_import_error_async(self):
        """Lines 140-142: httpx not installed → (None, 'none').

        We must reload the module so the local import inside the function
        actually re-executes.  But since the function does `import httpx`
        at call time, patching sys.modules is sufficient.
        """
        # Re-import to pick up the patched sys.modules
        import engine.llm_synthesizer as mod

        importlib.reload(mod)
        text, model = await mod._call_medical_llm_async("prompt", "system")
        assert text is None
        assert model == "none"
        # Restore module so later tests aren't affected
        importlib.reload(mod)

    @patch.dict(
        "os.environ",
        {"OPENAI_API_KEY": "oai-key", "GOOGLE_API_KEY": "", "GEMINI_API_KEY": ""},
    )
    async def test_openai_success_async(self):
        """Lines 144-164: OpenAI succeeds with 200 and valid content."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "Async OpenAI answer"}}]
        }

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_resp
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client):
            text, model = await _call_medical_llm_async("prompt", "system")

        assert text == "Async OpenAI answer"
        assert model == "OpenAI-GPT-5.4"

    @patch.dict(
        "os.environ",
        {"OPENAI_API_KEY": "oai-key", "GOOGLE_API_KEY": "", "GEMINI_API_KEY": ""},
    )
    async def test_openai_non_200_no_google_key(self):
        """Lines 165-166: OpenAI returns non-200, no Google key → falls through to (None, 'none')."""
        mock_resp = MagicMock()
        mock_resp.status_code = 429

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_resp
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client):
            text, model = await _call_medical_llm_async("prompt", "system")

        assert text is None
        assert model == "none"

    @patch.dict(
        "os.environ",
        {"OPENAI_API_KEY": "oai-key", "GOOGLE_API_KEY": "", "GEMINI_API_KEY": ""},
    )
    async def test_openai_exception_async(self):
        """Lines 167-168: OpenAI raises exception, no Google key → (None, 'none')."""
        mock_client = AsyncMock()
        mock_client.post.side_effect = Exception("connection timeout")
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client):
            text, model = await _call_medical_llm_async("prompt", "system")

        assert text is None
        assert model == "none"

    @patch.dict(
        "os.environ",
        {"OPENAI_API_KEY": "", "GOOGLE_API_KEY": "goog-key", "GEMINI_API_KEY": ""},
    )
    async def test_gemini_pro_success_async(self):
        """Lines 170-196: Google key only, first Gemini model succeeds."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "candidates": [{"content": {"parts": [{"text": "Gemini Pro async"}]}}]
        }

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_resp
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client):
            text, model = await _call_medical_llm_async("prompt", "system")

        assert text == "Gemini Pro async"
        assert model == "Gemini-3.1-Pro"

    @patch.dict(
        "os.environ",
        {"OPENAI_API_KEY": "", "GOOGLE_API_KEY": "goog-key", "GEMINI_API_KEY": ""},
    )
    async def test_gemini_pro_fails_flash_succeeds(self):
        """Lines 188-196: First Gemini returns non-200, second succeeds."""
        call_count = [0]

        def post_side_effect(*args, **kwargs):
            call_count[0] += 1
            resp = MagicMock()
            if call_count[0] == 1:
                resp.status_code = 503
            else:
                resp.status_code = 200
                resp.json.return_value = {
                    "candidates": [
                        {"content": {"parts": [{"text": "Flash Lite async"}]}}
                    ]
                }
            return resp

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=post_side_effect)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client):
            text, model = await _call_medical_llm_async("prompt", "system")

        assert text == "Flash Lite async"
        assert model == "Gemini-3.1-Flash-Lite"

    @patch.dict(
        "os.environ",
        {"OPENAI_API_KEY": "", "GOOGLE_API_KEY": "goog-key", "GEMINI_API_KEY": ""},
    )
    async def test_gemini_exception_cascades(self):
        """Lines 197-199: Gemini Pro throws, Flash throws → (None, 'none')."""
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=Exception("network error"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client):
            text, model = await _call_medical_llm_async("prompt", "system")

        assert text is None
        assert model == "none"

    @patch.dict(
        "os.environ",
        {"OPENAI_API_KEY": "", "GOOGLE_API_KEY": "goog-key", "GEMINI_API_KEY": ""},
    )
    async def test_gemini_empty_candidates(self):
        """Lines 192-193: Gemini returns 200 but empty candidates list."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"candidates": []}

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_resp
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client):
            text, model = await _call_medical_llm_async("prompt", "system")

        # Both models return empty candidates → falls through to line 201
        assert text is None
        assert model == "none"

    @patch.dict(
        "os.environ",
        {"OPENAI_API_KEY": "", "GOOGLE_API_KEY": "goog-key", "GEMINI_API_KEY": ""},
    )
    async def test_gemini_no_text_in_parts(self):
        """Lines 194-195: Gemini returns candidates but parts have no text."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "candidates": [{"content": {"parts": [{"text": ""}]}}]
        }

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_resp
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client):
            text, model = await _call_medical_llm_async("prompt", "system")

        assert text is None
        assert model == "none"

    @patch.dict(
        "os.environ",
        {
            "OPENAI_API_KEY": "oai-key",
            "GOOGLE_API_KEY": "goog-key",
            "GEMINI_API_KEY": "",
        },
    )
    async def test_openai_fails_gemini_succeeds_async(self):
        """Lines 145-168 + 170-196: Full cascade OpenAI fail → Gemini succeed."""
        call_count = [0]

        def post_side_effect(*args, **kwargs):
            call_count[0] += 1
            resp = MagicMock()
            if call_count[0] == 1:
                # OpenAI call fails with non-200
                resp.status_code = 500
            else:
                # Gemini Pro succeeds
                resp.status_code = 200
                resp.json.return_value = {
                    "candidates": [
                        {"content": {"parts": [{"text": "Gemini cascade"}]}}
                    ]
                }
            return resp

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=post_side_effect)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client):
            text, model = await _call_medical_llm_async("prompt", "system")

        assert text == "Gemini cascade"
        assert model == "Gemini-3.1-Pro"

    @patch.dict(
        "os.environ",
        {"OPENAI_API_KEY": "oai-key", "GOOGLE_API_KEY": "", "GEMINI_API_KEY": ""},
    )
    async def test_openai_200_but_empty_content(self):
        """Lines 162-163: OpenAI returns 200 but content is empty/None."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": ""}}]
        }

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_resp
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client):
            text, model = await _call_medical_llm_async("prompt", "system")

        # Empty content → does not return, falls through to line 201
        assert text is None
        assert model == "none"

    @patch.dict(
        "os.environ",
        {"OPENAI_API_KEY": "", "GOOGLE_API_KEY": "", "GEMINI_API_KEY": "gem-key"},
    )
    async def test_gemini_api_key_fallback(self):
        """Line 134: GEMINI_API_KEY used when GOOGLE_API_KEY is empty."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "candidates": [{"content": {"parts": [{"text": "Via GEMINI_API_KEY"}]}}]
        }

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_resp
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client):
            text, model = await _call_medical_llm_async("prompt", "system")

        assert text == "Via GEMINI_API_KEY"
        assert "Gemini" in model


# ── _call_medical_llm_sync httpx ImportError (lines 222-223) ─────────────────

class TestCallMedicalLlmSyncHttpxImport:
    """Cover the httpx ImportError branch in the sync function."""

    @patch.dict(
        "os.environ",
        {"OPENAI_API_KEY": "key", "GOOGLE_API_KEY": "", "GEMINI_API_KEY": ""},
    )
    def test_httpx_import_error_sync(self):
        """Lines 220-223: httpx not installed → (None, 'none')."""
        import engine.llm_synthesizer as mod

        # Temporarily remove httpx from sys.modules to force ImportError
        with patch.dict("sys.modules", {"httpx": None}):
            importlib.reload(mod)
            text, model = mod._call_medical_llm_sync("prompt", "system")

        assert text is None
        assert model == "none"
        # Restore module
        importlib.reload(mod)


# ── explain_conflict phi_detector ImportError (lines 374-375) ────────────────

class TestExplainConflictPhiDetectorImport:
    """Cover the phi_detector ImportError path inside explain_conflict."""

    @patch("engine.llm_synthesizer._call_medical_llm_sync", return_value=(None, "none"))
    @patch.dict("sys.modules", {"engine.phi_detector": None})
    def test_phi_detector_import_error_explain_conflict(self, mock_llm):
        """Lines 374-375: phi_detector not available → gracefully skips redaction,
        falls through to LLM call (which returns None → template fallback).
        """
        conflict = {
            "type": "drug_interaction",
            "severity": "high",
            "description": "Warfarin + aspirin",
            "recommendation": "Monitor INR",
        }
        patient = {"conditions": [{"name": "AF"}]}
        evidence = [
            {"block_id": "ev1", "title": "Warfarin", "content": "anticoagulant data"},
        ]
        result = explain_conflict(conflict, patient, evidence)
        # The function should still work — just without PHI redaction
        assert hasattr(result, "narrative")
        assert result.model_used == "template_fallback"
        assert result.abstained is False

    @patch("engine.llm_synthesizer._call_medical_llm_sync")
    @patch.dict("sys.modules", {"engine.phi_detector": None})
    def test_phi_detector_missing_but_llm_succeeds(self, mock_llm):
        """Lines 374-375: phi_detector missing, LLM still returns a good response."""
        mock_llm.return_value = (
            "Clinical concern [ev1] due to bleeding risk.",
            "OpenAI-GPT-5.4",
        )
        conflict = {"type": "drug_interaction", "severity": "critical", "description": "test"}
        patient = {"conditions": []}
        evidence = [{"block_id": "ev1", "content": "data"}]
        result = explain_conflict(conflict, patient, evidence)
        assert result.abstained is False
        assert result.model_used == "OpenAI-GPT-5.4"
        assert len(result.evidence_citations) == 1


# ── generate_clinical_handoff phi_detector ImportError (lines 462-463) ───────

class TestGenerateHandoffPhiDetectorImport:
    """Cover the phi_detector ImportError path inside generate_clinical_handoff."""

    @patch("engine.llm_synthesizer._call_medical_llm_sync", return_value=(None, "none"))
    @patch.dict("sys.modules", {"engine.phi_detector": None})
    def test_phi_detector_import_error_handoff(self, mock_llm):
        """Lines 462-463: phi_detector not available → gracefully skips redaction,
        falls through to LLM call (which returns None → template fallback).
        """
        patient = {
            "medications": [{"name": "Metformin", "dosage": "500mg"}],
            "conditions": [{"name": "T2DM"}],
            "allergies": [],
        }
        contradictions = [
            {"severity": "high", "description": "Renal dosing needed"},
        ]
        safety_report = {"interaction_count": 1, "allergy_conflict_count": 0}
        evidence = [{"block_id": f"b{i}", "content": f"data {i}"} for i in range(5)]

        result = generate_clinical_handoff(
            patient, contradictions, safety_report, evidence
        )
        assert hasattr(result, "narrative")
        assert result.model_used == "template_fallback"
        assert "CARE HANDOFF" in result.narrative

    @patch("engine.llm_synthesizer._call_medical_llm_sync")
    @patch.dict("sys.modules", {"engine.phi_detector": None})
    def test_phi_detector_missing_but_handoff_llm_succeeds(self, mock_llm):
        """Lines 462-463: phi_detector missing, LLM returns a valid handoff."""
        mock_llm.return_value = (
            "Handoff note: Patient stable [b0] [b1]. Monitor renal function [b2].",
            "Gemini-3.1-Pro",
        )
        patient = {"medications": [], "conditions": [], "allergies": []}
        evidence = [{"block_id": f"b{i}", "content": f"data {i}"} for i in range(5)]

        result = generate_clinical_handoff(patient, [], {}, evidence)
        assert result.abstained is False
        assert result.model_used == "Gemini-3.1-Pro"
        assert len(result.evidence_citations) == 3  # b0, b1, b2


# ── Additional edge cases for full line coverage ─────────────────────────────

class TestAsyncGeminiEdgeCases:
    """Additional async edge cases ensuring all branches in lines 170-201 are hit."""

    @patch.dict(
        "os.environ",
        {"OPENAI_API_KEY": "", "GOOGLE_API_KEY": "goog-key", "GEMINI_API_KEY": ""},
    )
    async def test_gemini_candidates_missing_content_key(self):
        """Lines 194: candidates exist but 'content' key missing → no parts → skip."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "candidates": [{"finishReason": "STOP"}]  # no 'content' key
        }

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_resp
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client):
            text, model = await _call_medical_llm_async("prompt", "system")

        assert text is None
        assert model == "none"

    @patch.dict(
        "os.environ",
        {"OPENAI_API_KEY": "", "GOOGLE_API_KEY": "goog-key", "GEMINI_API_KEY": ""},
    )
    async def test_gemini_parts_empty_list(self):
        """Lines 195: parts list is empty → skip."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "candidates": [{"content": {"parts": []}}]
        }

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_resp
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client):
            text, model = await _call_medical_llm_async("prompt", "system")

        assert text is None
        assert model == "none"

    @patch.dict(
        "os.environ",
        {
            "OPENAI_API_KEY": "oai-key",
            "GOOGLE_API_KEY": "goog-key",
            "GEMINI_API_KEY": "",
        },
    )
    async def test_full_cascade_all_fail_async(self):
        """Lines 133-201: OpenAI 500, Gemini Pro exception, Gemini Flash 503 → (None, 'none')."""
        call_count = [0]

        def post_side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                # OpenAI returns 500
                resp = MagicMock()
                resp.status_code = 500
                return resp
            elif call_count[0] == 2:
                # Gemini Pro throws
                raise Exception("Gemini Pro down")
            else:
                # Gemini Flash returns 503
                resp = MagicMock()
                resp.status_code = 503
                return resp

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=post_side_effect)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client):
            text, model = await _call_medical_llm_async("prompt", "system")

        assert text is None
        assert model == "none"
