"""Tests for multi-LLM consensus verification engine."""
import asyncio
import json
import os
import pytest
import respx
import httpx

from engine.consensus_engine import (
    _build_prompt,
    _parse_verdict,
    verify_finding_consensus,
    verify_finding_consensus_sync,
    LLMVerdict,
    ConsensusResult,
)


class TestBuildPrompt:
    def test_includes_finding(self):
        prompt = _build_prompt("Drug interaction detected", [], {})
        assert "Drug interaction detected" in prompt

    def test_includes_evidence(self):
        evidence = [
            {"block_id": "b1", "title": "Lab Result", "content": "Potassium 5.8"},
        ]
        prompt = _build_prompt("High potassium", evidence, {})
        assert "b1" in prompt
        assert "Potassium 5.8" in prompt

    def test_includes_patient_context(self):
        ctx = {"medications": ["warfarin"], "patient_id": "P001"}
        prompt = _build_prompt("Bleeding risk", [], ctx)
        assert "warfarin" in prompt

    def test_limits_evidence(self):
        evidence = [
            {"block_id": f"b{i}", "content": f"Evidence {i}"}
            for i in range(20)
        ]
        prompt = _build_prompt("Finding", evidence, {})
        # Only first 10 evidence blocks
        assert "b9" in prompt
        assert "b10" not in prompt


class TestParseVerdict:
    def test_parses_valid_json(self):
        text = '{"agrees": true, "confidence": 0.9, "reasoning": "Clearly dangerous"}'
        verdict = _parse_verdict(text, "TestModel")
        assert verdict.agrees is True
        assert verdict.confidence == 0.9
        assert verdict.model == "TestModel"
        assert "Clearly dangerous" in verdict.reasoning

    def test_parses_disagreement(self):
        text = '{"agrees": false, "confidence": 0.3, "reasoning": "Insufficient evidence"}'
        verdict = _parse_verdict(text, "ModelA")
        assert verdict.agrees is False
        assert verdict.confidence == 0.3

    def test_handles_markdown_code_block(self):
        text = '```json\n{"agrees": true, "confidence": 0.8, "reasoning": "Valid"}\n```'
        verdict = _parse_verdict(text, "Model")
        assert verdict.agrees is True

    def test_clamps_confidence(self):
        text = '{"agrees": true, "confidence": 1.5, "reasoning": "Overconfident"}'
        verdict = _parse_verdict(text, "Model")
        assert verdict.confidence == 1.0

    def test_clamps_negative_confidence(self):
        text = '{"agrees": true, "confidence": -0.5, "reasoning": "Negative"}'
        verdict = _parse_verdict(text, "Model")
        assert verdict.confidence == 0.0

    def test_fallback_keyword_agree(self):
        text = "I agree this is a genuine clinical concern."
        verdict = _parse_verdict(text, "Model")
        assert verdict.agrees is True
        assert verdict.confidence == 0.5

    def test_fallback_keyword_disagree(self):
        text = "The evidence does not support this finding at all."
        verdict = _parse_verdict(text, "Model")
        assert verdict.agrees is False

    def test_truncates_long_reasoning(self):
        text = json.dumps({"agrees": True, "confidence": 0.5, "reasoning": "x" * 500})
        verdict = _parse_verdict(text, "Model")
        assert len(verdict.reasoning) <= 300

    def test_missing_fields_default(self):
        text = "{}"
        verdict = _parse_verdict(text, "Model")
        assert verdict.agrees is False
        assert verdict.confidence == 0.5


@pytest.fixture(autouse=True)
def _clear_all_api_keys(monkeypatch):
    """Ensure no real API keys leak into tests."""
    for key in (
        "OPENAI_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY",
        "XAI_API_KEY", "DEEPSEEK_API_KEY", "MISTRAL_API_KEY",
    ):
        monkeypatch.delenv(key, raising=False)


class TestVerifyFindingConsensus:
    """Tests use the sync wrapper to avoid pytest-asyncio dependency."""

    def test_no_api_keys_returns_none(self, monkeypatch):

        result = verify_finding_consensus_sync("Test finding", [], {})
        assert isinstance(result, ConsensusResult)
        assert result.consensus_level == "NONE"
        assert result.should_report is False
        assert result.total_models == 0

    @respx.mock
    def test_openai_only_limited(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")

        respx.post("https://api.openai.com/v1/chat/completions").mock(
            return_value=httpx.Response(200, json={
                "choices": [{"message": {"content": json.dumps({
                    "agrees": True, "confidence": 0.9, "reasoning": "Valid concern"
                })}}]
            })
        )

        result = verify_finding_consensus_sync("Drug interaction", [], {})
        assert result.total_models == 1
        assert result.consensus_level == "LIMITED"

    @respx.mock
    def test_full_consensus_high(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("GOOGLE_API_KEY", "gk-test")

        agree_json = json.dumps({"agrees": True, "confidence": 0.9, "reasoning": "Valid"})

        respx.post("https://api.openai.com/v1/chat/completions").mock(
            return_value=httpx.Response(200, json={
                "choices": [{"message": {"content": agree_json}}]
            })
        )
        respx.post(
            url__regex=r"https://generativelanguage\.googleapis\.com/.*"
        ).mock(
            return_value=httpx.Response(200, json={
                "candidates": [{"content": {"parts": [{"text": agree_json}]}}]
            })
        )

        result = verify_finding_consensus_sync("Warfarin-aspirin interaction", [], {})
        assert result.agreement_count == 3
        assert result.consensus_level == "HIGH"
        assert result.should_report is True

    @respx.mock
    def test_consensus_none_all_disagree(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("GOOGLE_API_KEY", "gk-test")

        disagree_json = json.dumps({"agrees": False, "confidence": 0.2, "reasoning": "Not valid"})

        respx.post("https://api.openai.com/v1/chat/completions").mock(
            return_value=httpx.Response(200, json={
                "choices": [{"message": {"content": disagree_json}}]
            })
        )
        respx.post(
            url__regex=r"https://generativelanguage\.googleapis\.com/.*"
        ).mock(
            return_value=httpx.Response(200, json={
                "candidates": [{"content": {"parts": [{"text": disagree_json}]}}]
            })
        )

        result = verify_finding_consensus_sync("Dubious finding", [], {})
        assert result.consensus_level == "NONE"
        assert result.should_report is False

    @respx.mock
    def test_api_failure_handled_gracefully(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

        respx.post("https://api.openai.com/v1/chat/completions").mock(
            return_value=httpx.Response(500)
        )

        result = verify_finding_consensus_sync("Some finding", [], {})
        assert result.total_models == 1
        # Failed API call treated as non-agreement
        assert any("failed" in v.reasoning.lower() for v in result.verdicts)

    @respx.mock
    def test_medium_consensus_2_of_3(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("GOOGLE_API_KEY", "gk-test")

        agree_json = json.dumps({"agrees": True, "confidence": 0.8, "reasoning": "Valid"})
        disagree_json = json.dumps({"agrees": False, "confidence": 0.3, "reasoning": "Unsure"})

        call_count = 0

        def openai_handler(request):
            return httpx.Response(200, json={
                "choices": [{"message": {"content": agree_json}}]
            })

        def google_handler(request):
            nonlocal call_count
            call_count += 1
            # First Google call agrees, second disagrees
            content = agree_json if call_count == 1 else disagree_json
            return httpx.Response(200, json={
                "candidates": [{"content": {"parts": [{"text": content}]}}]
            })

        respx.post("https://api.openai.com/v1/chat/completions").mock(
            side_effect=openai_handler
        )
        respx.post(
            url__regex=r"https://generativelanguage\.googleapis\.com/.*"
        ).mock(side_effect=google_handler)

        result = verify_finding_consensus_sync("Moderate finding", [], {})
        assert result.agreement_count == 2
        assert result.consensus_level == "MEDIUM"
        assert result.should_report is True

    def test_immutable_result(self):
        result = verify_finding_consensus_sync("Test", [], {})
        assert isinstance(result.verdicts, tuple)  # Immutable

    @respx.mock
    def test_xai_grok_included(self, monkeypatch):
        """xAI Grok joins consensus when XAI_API_KEY is set."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.setenv("XAI_API_KEY", "xai-test")

        agree_json = json.dumps({"agrees": True, "confidence": 0.85, "reasoning": "Valid"})
        respx.post("https://api.x.ai/v1/chat/completions").mock(
            return_value=httpx.Response(200, json={
                "choices": [{"message": {"content": agree_json}}]
            })
        )

        result = verify_finding_consensus_sync("Test finding", [], {})
        assert result.total_models == 1
        assert result.verdicts[0].model == "xAI-Grok-4.1"
        assert result.verdicts[0].agrees is True

    @respx.mock
    def test_deepseek_included(self, monkeypatch):
        """DeepSeek Reasoner joins consensus when DEEPSEEK_API_KEY is set."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.setenv("DEEPSEEK_API_KEY", "ds-test")

        agree_json = json.dumps({"agrees": True, "confidence": 0.9, "reasoning": "Confirmed"})
        respx.post("https://api.deepseek.com/v1/chat/completions").mock(
            return_value=httpx.Response(200, json={
                "choices": [{"message": {"content": agree_json}}]
            })
        )

        result = verify_finding_consensus_sync("Test finding", [], {})
        assert result.total_models == 1
        assert result.verdicts[0].model == "DeepSeek-Reasoner"

    @respx.mock
    def test_mistral_included(self, monkeypatch):
        """Mistral Large joins consensus when MISTRAL_API_KEY is set."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.setenv("MISTRAL_API_KEY", "mis-test")

        agree_json = json.dumps({"agrees": False, "confidence": 0.4, "reasoning": "Uncertain"})
        respx.post("https://api.mistral.ai/v1/chat/completions").mock(
            return_value=httpx.Response(200, json={
                "choices": [{"message": {"content": agree_json}}]
            })
        )

        result = verify_finding_consensus_sync("Test finding", [], {})
        assert result.total_models == 1
        assert result.verdicts[0].model == "Mistral-Large"
        assert result.verdicts[0].agrees is False

    @respx.mock
    def test_six_model_consensus(self, monkeypatch):
        """All 6 models fire in parallel when all keys are available."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("GOOGLE_API_KEY", "gk-test")
        monkeypatch.setenv("XAI_API_KEY", "xai-test")
        monkeypatch.setenv("DEEPSEEK_API_KEY", "ds-test")
        monkeypatch.setenv("MISTRAL_API_KEY", "mis-test")

        agree_json = json.dumps({"agrees": True, "confidence": 0.9, "reasoning": "Valid"})

        # Mock all API endpoints
        respx.post("https://api.openai.com/v1/chat/completions").mock(
            return_value=httpx.Response(200, json={
                "choices": [{"message": {"content": agree_json}}]
            })
        )
        respx.post(
            url__regex=r"https://generativelanguage\.googleapis\.com/.*"
        ).mock(
            return_value=httpx.Response(200, json={
                "candidates": [{"content": {"parts": [{"text": agree_json}]}}]
            })
        )
        respx.post("https://api.x.ai/v1/chat/completions").mock(
            return_value=httpx.Response(200, json={
                "choices": [{"message": {"content": agree_json}}]
            })
        )
        respx.post("https://api.deepseek.com/v1/chat/completions").mock(
            return_value=httpx.Response(200, json={
                "choices": [{"message": {"content": agree_json}}]
            })
        )
        respx.post("https://api.mistral.ai/v1/chat/completions").mock(
            return_value=httpx.Response(200, json={
                "choices": [{"message": {"content": agree_json}}]
            })
        )

        result = verify_finding_consensus_sync("Critical interaction", [], {})
        assert result.total_models == 6
        assert result.agreement_count == 6
        assert result.consensus_level == "HIGH"
        assert result.should_report is True
        models = {v.model for v in result.verdicts}
        assert "OpenAI-GPT-5.4" in models
        assert "MedGemma-27B" in models
        assert "xAI-Grok-4.1" in models
        assert "DeepSeek-Reasoner" in models
        assert "Mistral-Large" in models

    @respx.mock
    def test_partial_failure_still_works(self, monkeypatch):
        """Consensus works even when some models fail."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("XAI_API_KEY", "xai-test")
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        monkeypatch.delenv("MISTRAL_API_KEY", raising=False)

        agree_json = json.dumps({"agrees": True, "confidence": 0.9, "reasoning": "Valid"})

        # OpenAI succeeds, xAI fails
        respx.post("https://api.openai.com/v1/chat/completions").mock(
            return_value=httpx.Response(200, json={
                "choices": [{"message": {"content": agree_json}}]
            })
        )
        respx.post("https://api.x.ai/v1/chat/completions").mock(
            return_value=httpx.Response(500)
        )

        result = verify_finding_consensus_sync("Finding", [], {})
        assert result.total_models == 2
        # One agrees, one failed (treated as disagree)
        assert result.agreement_count == 1
