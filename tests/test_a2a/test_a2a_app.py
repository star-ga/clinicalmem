"""Tests for A2A app — agent card, middleware, and configuration.

These tests require the a2a and google-adk packages.
They skip gracefully if dependencies are not installed.
"""
import json
import pytest
from unittest.mock import MagicMock

a2a_types = pytest.importorskip("a2a.types")
google_adk = pytest.importorskip("google.adk")


class TestAgentCard:
    def test_agent_card_fields(self):
        from a2a_agent.app import agent_card
        assert agent_card.name == "clinicalmem_agent"
        assert "ClinicalMem" in agent_card.description
        assert "STARGA" in agent_card.description
        assert agent_card.version == "0.1.0"
        assert len(agent_card.skills) == 5

    def test_agent_card_skills(self):
        from a2a_agent.app import agent_card
        skill_ids = [s.id for s in agent_card.skills]
        assert "medication-safety-review" in skill_ids
        assert "clinical-context-recall" in skill_ids
        assert "contradiction-assessment" in skill_ids
        assert "care-transition-summary" in skill_ids
        assert "explain-conflict" in skill_ids

    def test_agent_card_security(self):
        from a2a_agent.app import agent_card
        assert "apiKey" in agent_card.security_schemes
        assert agent_card.security == [{"apiKey": []}]

    def test_agent_card_capabilities(self):
        from a2a_agent.app import agent_card
        assert agent_card.capabilities.streaming is True
        assert agent_card.capabilities.state_transition_history is True
        assert len(agent_card.capabilities.extensions) == 1
        assert "fhir-context" in agent_card.capabilities.extensions[0].uri

    def test_default_modes(self):
        from a2a_agent.app import agent_card
        assert "text/plain" in agent_card.default_input_modes
        assert "text/plain" in agent_card.default_output_modes


class TestRateLimitMiddleware:
    def test_middleware_init(self):
        from a2a_agent.app import RateLimitMiddleware
        mock_app = MagicMock()
        mw = RateLimitMiddleware(mock_app)
        assert mw.MAX_REQUESTS == 60
        assert mw.WINDOW_SECONDS == 60


class TestExtractFhirContext:
    def test_fallback_with_json_string(self):
        from a2a_agent.agent import extract_fhir_context
        ctx = MagicMock()
        ctx.metadata = {
            "fhir-context": json.dumps({"fhirUrl": "http://fhir.test", "fhirToken": "tok123", "patientId": "p-1"})
        }
        ctx.state = {}
        result = extract_fhir_context(ctx, MagicMock())
        assert ctx.state.get("fhir_url") == "http://fhir.test"
        assert ctx.state.get("fhir_token") == "tok123"
        assert ctx.state.get("patient_id") == "p-1"

    def test_fallback_with_dict_value(self):
        from a2a_agent.agent import extract_fhir_context
        ctx = MagicMock()
        ctx.metadata = {
            "fhir-context": {"fhirUrl": "http://fhir.test", "fhirToken": "tok", "patientId": "p-2"}
        }
        ctx.state = {}
        extract_fhir_context(ctx, MagicMock())
        assert ctx.state.get("fhir_url") == "http://fhir.test"

    def test_fallback_no_metadata(self):
        from a2a_agent.agent import extract_fhir_context
        ctx = MagicMock()
        ctx.metadata = None
        ctx.run_config = None
        ctx.state = {}
        result = extract_fhir_context(ctx, MagicMock())
        assert result is None

    def test_fallback_invalid_json(self):
        from a2a_agent.agent import extract_fhir_context
        ctx = MagicMock()
        ctx.metadata = {"fhir-context": "not-json"}
        ctx.state = {}
        extract_fhir_context(ctx, MagicMock())
        assert "fhir_url" not in ctx.state


class TestRootAgent:
    def test_agent_exists(self):
        from a2a_agent.agent import root_agent
        assert root_agent.name == "clinicalmem_agent"
        assert len(root_agent.tools) == 13

    def test_agent_model(self):
        from a2a_agent.agent import root_agent
        assert "gemini" in root_agent.model
