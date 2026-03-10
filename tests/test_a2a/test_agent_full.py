"""Full coverage tests for a2a_agent/agent.py — extract_fhir_context edge cases."""
import pytest
from unittest.mock import MagicMock

a2a_types = pytest.importorskip("a2a.types")
google_adk = pytest.importorskip("google.adk")


class TestExtractFhirContextLine55:
    """Line 55: return None when metadata is not a dict after fallback attempts."""

    def test_metadata_not_dict_no_run_config(self):
        """metadata is not a dict, run_config is None -> line 55 returns None."""
        from a2a_agent.agent import extract_fhir_context

        ctx = MagicMock()
        ctx.metadata = "not-a-dict-string"
        ctx.run_config = None
        ctx.state = {}
        result = extract_fhir_context(ctx, MagicMock())
        assert result is None
        # No FHIR state should have been set
        assert "fhir_url" not in ctx.state

    def test_metadata_not_dict_run_config_no_custom(self):
        """metadata not a dict, run_config exists but custom_metadata is None -> line 55."""
        from a2a_agent.agent import extract_fhir_context

        ctx = MagicMock()
        ctx.metadata = 42  # not a dict
        run_config = MagicMock()
        run_config.custom_metadata = None
        ctx.run_config = run_config
        ctx.state = {}
        result = extract_fhir_context(ctx, MagicMock())
        assert result is None

    def test_metadata_not_dict_custom_metadata_not_dict(self):
        """metadata not a dict, custom_metadata exists but is not a dict -> line 55."""
        from a2a_agent.agent import extract_fhir_context

        ctx = MagicMock()
        ctx.metadata = False  # not a dict
        run_config = MagicMock()
        run_config.custom_metadata = "also-not-a-dict"
        ctx.run_config = run_config
        ctx.state = {}
        result = extract_fhir_context(ctx, MagicMock())
        assert result is None

    def test_metadata_not_dict_custom_metadata_dict_no_a2a_key(self):
        """custom_metadata is a dict but has no 'a2a_metadata' key -> empty dict fallback -> line 55."""
        from a2a_agent.agent import extract_fhir_context

        ctx = MagicMock()
        ctx.metadata = []  # not a dict
        run_config = MagicMock()
        run_config.custom_metadata = {"other_key": "value"}
        ctx.run_config = run_config
        ctx.state = {}
        result = extract_fhir_context(ctx, MagicMock())
        # a2a_metadata key is missing, .get returns None which is not a dict
        assert result is None

    def test_metadata_not_dict_a2a_metadata_also_not_dict(self):
        """custom_metadata['a2a_metadata'] exists but is not a dict -> line 55."""
        from a2a_agent.agent import extract_fhir_context

        ctx = MagicMock()
        ctx.metadata = 0  # not a dict
        run_config = MagicMock()
        run_config.custom_metadata = {"a2a_metadata": "still-not-a-dict"}
        ctx.run_config = run_config
        ctx.state = {}
        result = extract_fhir_context(ctx, MagicMock())
        assert result is None
