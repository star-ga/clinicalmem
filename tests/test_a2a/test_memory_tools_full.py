"""Full coverage tests for a2a_agent/tools/memory_tools.py."""
import json
import os
import pytest
from unittest.mock import patch, MagicMock

a2a_types = pytest.importorskip("a2a.types")
google_adk = pytest.importorskip("google.adk")


def _make_ctx(state=None, patient_id="patient-sarah-mitchell"):
    ctx = MagicMock()
    ctx.state = state if state is not None else {"patient_id": patient_id}
    ctx.metadata = None
    ctx.run_config = None
    return ctx


@pytest.fixture(autouse=True)
def reset_engine():
    from a2a_agent.tools import memory_tools
    from engine.clinical_memory import ClinicalMemEngine
    memory_tools._engine = ClinicalMemEngine()
    yield


@pytest.fixture
def loaded():
    from a2a_agent.tools.memory_tools import _engine
    fixture_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "tests", "fixtures", "sarah_mitchell_bundle.json",
    )
    with open(fixture_path) as f:
        bundle = json.load(f)
    _engine.ingest_from_bundle(bundle, "patient-sarah-mitchell")
    return _engine


class TestDemoPatientId:
    @patch.dict(os.environ, {"DEMO_MODE": "true"})
    def test_demo_mode_fallback(self):
        from a2a_agent.tools.memory_tools import _demo_patient_id
        ctx = _make_ctx(state={})
        assert _demo_patient_id(ctx) == "patient-sarah-mitchell"

    def test_no_demo(self):
        from a2a_agent.tools.memory_tools import _demo_patient_id
        ctx = _make_ctx(state={})
        assert _demo_patient_id(ctx) == ""

    def test_from_state(self):
        from a2a_agent.tools.memory_tools import _demo_patient_id
        ctx = _make_ctx(state={"patient_id": "p-1"})
        assert _demo_patient_id(ctx) == "p-1"


class TestAutoIngest:
    def test_no_credentials(self):
        from a2a_agent.tools.memory_tools import _auto_ingest
        ctx = _make_ctx(state={})
        _auto_ingest(ctx)  # should not raise

    def test_already_has_data(self, loaded):
        from a2a_agent.tools.memory_tools import _auto_ingest
        ctx = _make_ctx(state={
            "fhir_url": "https://fhir.test",
            "fhir_token": "tok",
            "patient_id": "patient-sarah-mitchell",
        })
        _auto_ingest(ctx)

    @patch("a2a_agent.tools.memory_tools.FHIRClient")
    def test_ingest_failure(self, mock_cls):
        from a2a_agent.tools.memory_tools import _auto_ingest
        mock_cls.side_effect = Exception("fail")
        ctx = _make_ctx(state={"fhir_url": "https://fhir", "fhir_token": "t", "patient_id": "p-new"})
        _auto_ingest(ctx)  # logs warning


class TestRecallClinicalContext:
    def test_no_patient(self):
        from a2a_agent.tools.memory_tools import recall_clinical_context
        ctx = _make_ctx(state={})
        result = recall_clinical_context("test", ctx)
        assert result["status"] == "error"

    def test_with_data(self, loaded):
        from a2a_agent.tools.memory_tools import recall_clinical_context
        ctx = _make_ctx()
        result = recall_clinical_context("medication allergy", ctx)
        assert result["status"] == "success"
        assert result["result_count"] > 0
        assert "confidence" in result


class TestStoreClinicalNote:
    def test_no_patient(self):
        from a2a_agent.tools.memory_tools import store_clinical_note
        ctx = _make_ctx(state={})
        result = store_clinical_note("title", "content", "note", ctx)
        assert result["status"] == "error"

    def test_store_note(self, loaded):
        from a2a_agent.tools.memory_tools import store_clinical_note
        ctx = _make_ctx()
        result = store_clinical_note("Test", "Content", "clinical_note", ctx)
        assert result["status"] == "success"
        assert "block_id" in result
        assert "audit_hash" in result


# ── Additional coverage for missing lines ──────────────────────────────────


class TestDemoModeAutoLoad:
    """Cover lines 28-41: DEMO_MODE auto-load at module level."""

    def test_demo_mode_loads_fixture(self):
        """Re-importing memory_tools with DEMO_MODE=true triggers auto-load.

        Lines 28-41 are module-level code that execute on import when
        DEMO_MODE is set. We reimport the module to exercise this path.
        """
        import importlib
        import sys

        # Remove cached module so reimport triggers top-level code
        mods_to_remove = [k for k in sys.modules if "memory_tools" in k]
        saved = {}
        for mod_name in mods_to_remove:
            saved[mod_name] = sys.modules.pop(mod_name)

        try:
            with patch.dict(os.environ, {"DEMO_MODE": "true"}):
                import a2a_agent.tools.memory_tools as mt
                importlib.reload(mt)

                # Verify fixture data was loaded for demo patient
                engine = mt._engine
                assert "patient-sarah-mitchell" in engine._patient_blocks
                assert len(engine._patient_blocks["patient-sarah-mitchell"]) > 0
        finally:
            # Restore original modules
            for mod_name, mod in saved.items():
                sys.modules[mod_name] = mod

    def test_demo_mode_missing_fixture_logs_warning(self):
        """Lines 40-41: When fixture file doesn't exist, exception is caught."""
        import importlib
        import sys

        mods_to_remove = [k for k in sys.modules if "memory_tools" in k]
        saved = {}
        for mod_name in mods_to_remove:
            saved[mod_name] = sys.modules.pop(mod_name)

        try:
            with patch.dict(os.environ, {"DEMO_MODE": "1"}):
                with patch("os.path.exists", return_value=False):
                    import a2a_agent.tools.memory_tools as mt
                    importlib.reload(mt)
                    # Should not raise — just skips loading
                    assert mt._engine is not None
        finally:
            for mod_name, mod in saved.items():
                sys.modules[mod_name] = mod


class TestAutoIngestFhirFailure:
    """Cover lines 61-62: engine.ingest_from_fhir raises an exception."""

    @patch("a2a_agent.tools.memory_tools.FHIRClient")
    def test_ingest_from_fhir_raises(self, mock_fhir_cls):
        """Line 61-62: FHIRClient succeeds but engine.ingest_from_fhir fails."""
        from a2a_agent.tools.memory_tools import _auto_ingest, _engine

        mock_fhir_instance = MagicMock()
        mock_fhir_cls.return_value = mock_fhir_instance

        # Make ingest_from_fhir raise
        with patch.object(_engine, "ingest_from_fhir", side_effect=RuntimeError("FHIR ingest boom")):
            ctx = _make_ctx(state={
                "fhir_url": "https://fhir.example.com",
                "fhir_token": "bearer-token",
                "patient_id": "patient-new-test",
            })
            # Should not raise — exception is caught and logged
            _auto_ingest(ctx)

        # Verify FHIRClient was instantiated but ingest failed gracefully
        mock_fhir_cls.assert_called_once()

    @patch("a2a_agent.tools.memory_tools.FHIRClient")
    def test_ingest_from_fhir_generic_exception(self, mock_fhir_cls):
        """Line 63-64: Generic Exception in the try block is caught."""
        from a2a_agent.tools.memory_tools import _auto_ingest, _engine

        mock_fhir_instance = MagicMock()
        mock_fhir_cls.return_value = mock_fhir_instance

        with patch.object(_engine, "ingest_from_fhir", side_effect=Exception("connection refused")):
            ctx = _make_ctx(state={
                "fhir_url": "https://fhir.example.com",
                "fhir_token": "bearer-token",
                "patient_id": "patient-another-test",
            })
            _auto_ingest(ctx)  # Should log warning, not raise


# ── Line 40-41: DEMO_MODE auto-load exception handler ───────────────────────


class TestDemoModeAutoLoadException:
    """Lines 40-41: Exception during DEMO_MODE auto-load is caught and logged."""

    def test_demo_mode_json_load_fails(self):
        """Trigger exception in the DEMO_MODE try block (lines 28-41).

        We force the json.load to raise by patching builtins.open to return
        a file-like object that causes json.load to fail.
        """
        import importlib
        import sys

        mods_to_remove = [k for k in sys.modules if "memory_tools" in k]
        saved = {}
        for mod_name in mods_to_remove:
            saved[mod_name] = sys.modules.pop(mod_name)

        try:
            with patch.dict(os.environ, {"DEMO_MODE": "true", "GEMINI_API_KEY": ""}):
                # Patch json.load to raise inside the module-level try block
                with patch("json.load", side_effect=Exception("corrupt JSON")):
                    import a2a_agent.tools.memory_tools as mt
                    importlib.reload(mt)
                    # Module should still load — exception is caught at line 40-41
                    assert mt._engine is not None
        finally:
            for mod_name, mod in saved.items():
                sys.modules[mod_name] = mod


# ── Line 62: _auto_ingest success path (engine.ingest_from_fhir succeeds) ───


class TestAutoIngestSuccessPath:
    """Line 61-62: FHIRClient succeeds AND engine.ingest_from_fhir succeeds."""

    @patch("a2a_agent.tools.memory_tools.FHIRClient")
    @patch("a2a_agent.tools.memory_tools.FHIRContext")
    def test_ingest_from_fhir_succeeds(self, mock_ctx_cls, mock_fhir_cls):
        from a2a_agent.tools.memory_tools import _auto_ingest, _engine

        mock_fhir_instance = MagicMock()
        mock_fhir_cls.return_value = mock_fhir_instance
        mock_ctx_cls.return_value = MagicMock()

        # Mock ingest_from_fhir to succeed (return counts dict)
        with patch.object(
            _engine, "ingest_from_fhir", return_value={"MedicationRequest": 3, "Condition": 2}
        ) as mock_ingest:
            ctx = _make_ctx(state={
                "fhir_url": "https://fhir.example.com",
                "fhir_token": "bearer-token",
                "patient_id": "patient-success-test",
            })
            _auto_ingest(ctx)

        # Verify the full success path was executed
        mock_ctx_cls.assert_called_once_with(
            url="https://fhir.example.com",
            token="bearer-token",
            patient_id="patient-success-test",
        )
        mock_fhir_cls.assert_called_once()
        mock_ingest.assert_called_once_with(mock_fhir_instance)
