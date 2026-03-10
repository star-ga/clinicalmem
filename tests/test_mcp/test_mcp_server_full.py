"""Full coverage tests for mcp_server/server.py — all MCP tool functions."""
import json
import os
import time
import pytest
from unittest.mock import patch, MagicMock

import mcp_server.server as srv

# FastMCP @mcp.tool() wraps functions as FunctionTool objects.
# Access the original function via .fn attribute.
_store = srv.store_clinical_observation.fn
_recall = srv.recall_patient_context.fn
_check_meds = srv.check_medication_conflicts.fn
_check_allergy = srv.check_allergy_conflicts.fn
_treatment_deps = srv.get_treatment_dependencies.fn
_audit_trail = srv.get_clinical_audit_trail.fn
_summarize = srv.summarize_patient_history.fn
_detect_drift = srv.detect_belief_drift.fn
_ingest = srv.ingest_patient_data.fn
_explain = srv.explain_clinical_conflict.fn
_handoff = srv.clinical_care_handoff.fn
_whatif = srv.what_if_medication_change.fn
_verify_claims = srv.verify_clinical_claims.fn
_scan_phi = srv.scan_for_phi.fn
_fda = srv.check_fda_safety_alerts.fn
_trials = srv.find_matching_trials.fn
_consensus = srv.consensus_verify_finding.fn
_health = srv.health_check.fn


@pytest.fixture(autouse=True)
def reset_engine():
    """Reset the shared engine between tests."""
    from engine.clinical_memory import ClinicalMemEngine
    srv._engine = ClinicalMemEngine()
    srv._rate_limit_window.clear()
    yield


@pytest.fixture
def loaded_engine():
    """Engine pre-loaded with Sarah Mitchell fixture data."""
    fixture_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "tests", "fixtures", "sarah_mitchell_bundle.json",
    )
    with open(fixture_path) as f:
        bundle = json.load(f)
    srv._engine.ingest_from_bundle(bundle, "patient-sarah-mitchell")
    return srv._engine


# ── Rate limiter ──────────────────────────────────────────────────────────────

class TestRateLimiter:
    def test_allows_under_limit(self):
        for _ in range(59):
            srv._check_rate_limit("test_tool")

    def test_raises_at_limit(self):
        for _ in range(60):
            srv._check_rate_limit("test_tool2")
        with pytest.raises(ValueError, match="Rate limit exceeded"):
            srv._check_rate_limit("test_tool2")

    def test_separate_tools_have_separate_limits(self):
        for _ in range(60):
            srv._check_rate_limit("tool_a")
        srv._check_rate_limit("tool_b")


# ── FHIR context extraction ──────────────────────────────────────────────────

class TestGetFhirContextFromHeaders:
    def test_valid_headers(self):
        headers = {"x-fhir-server-url": "https://fhir.test/r4", "x-fhir-access-token": "tok123", "x-patient-id": "p-1"}
        ctx = srv._get_fhir_context_from_headers(headers)
        assert ctx is not None
        assert ctx.patient_id == "p-1"

    def test_missing_headers(self):
        assert srv._get_fhir_context_from_headers({}) is None

    def test_none_headers(self):
        assert srv._get_fhir_context_from_headers(None) is None

    def test_uppercase_headers(self):
        headers = {"X-FHIR-Server-URL": "https://fhir.test/r4", "X-FHIR-Access-Token": "tok", "X-Patient-ID": "p-2"}
        ctx = srv._get_fhir_context_from_headers(headers)
        assert ctx.patient_id == "p-2"

    def test_partial_headers_returns_none(self):
        """Line 86: some headers present but not all → returns None."""
        headers = {"x-fhir-server-url": "https://fhir.test/r4"}
        assert srv._get_fhir_context_from_headers(headers) is None


# ── Store Clinical Observation ────────────────────────────────────────────────

class TestStoreObservation:
    def test_store_basic(self):
        result = _store(patient_id="p-1", observation_type="clinical_note", title="Test Note", content="Test content")
        assert result["status"] == "success"
        assert "block_id" in result

    def test_store_with_source(self):
        result = _store(patient_id="p-1", observation_type="lab_result", title="Lab", content="Result", source="Dr. Smith")
        assert result["status"] == "success"


# ── Recall Patient Context ───────────────────────────────────────────────────

class TestRecallPatientContext:
    def test_recall_empty(self):
        result = _recall(patient_id="unknown", query="medications")
        assert result["status"] == "success"
        assert result["result_count"] == 0

    def test_recall_with_data(self, loaded_engine):
        result = _recall(patient_id="patient-sarah-mitchell", query="medication allergy")
        assert result["status"] == "success"
        assert result["result_count"] > 0

    def test_recall_with_fhir_auto_ingest(self):
        with patch("mcp_server.server.FHIRClient") as mock_fhir:
            mock_fhir.return_value.get_medications.return_value = []
            mock_fhir.return_value.get_conditions.return_value = []
            mock_fhir.return_value.get_allergies.return_value = []
            mock_fhir.return_value.get_observations.return_value = []
            mock_fhir.return_value.patient_id = "p-auto"
            result = _recall(patient_id="p-auto", query="test", fhir_server_url="https://fhir.test", fhir_access_token="tok")
            assert result["status"] == "success"


# ── Check Medication Conflicts ────────────────────────────────────────────────

class TestCheckMedicationConflicts:
    def test_no_patient_data(self):
        result = _check_meds(patient_id="unknown")
        assert result["status"] == "success"
        assert result["interaction_count"] == 0

    def test_with_data(self, loaded_engine):
        result = _check_meds(patient_id="patient-sarah-mitchell")
        assert result["status"] == "success"
        assert "recommendations" in result


# ── Check Allergy Conflicts ──────────────────────────────────────────────────

class TestCheckAllergyConflicts:
    def test_no_data(self):
        result = _check_allergy(patient_id="unknown")
        assert result["status"] == "success"
        assert result["conflict_count"] == 0

    def test_with_data(self, loaded_engine):
        result = _check_allergy(patient_id="patient-sarah-mitchell")
        assert result["status"] == "success"


# ── Treatment Dependencies ───────────────────────────────────────────────────

class TestTreatmentDependencies:
    def test_no_data(self):
        result = _treatment_deps(patient_id="unknown")
        assert result["status"] == "success"
        assert result["dependencies"] == []

    def test_with_data(self, loaded_engine):
        result = _treatment_deps(patient_id="patient-sarah-mitchell")
        assert result["status"] == "success"
        assert "audit_hash" in result


# ── Audit Trail ──────────────────────────────────────────────────────────────

class TestAuditTrail:
    def test_empty_trail(self):
        result = _audit_trail()
        assert result["status"] == "success"

    def test_trail_after_operations(self, loaded_engine):
        _store("p-1", "note", "title", "content")
        result = _audit_trail()
        assert result["entry_count"] > 0


# ── Summarize Patient History ────────────────────────────────────────────────

class TestSummarizePatientHistory:
    def test_empty_patient(self):
        result = _summarize(patient_id="unknown")
        assert result["status"] == "success"
        assert result["total_blocks"] == 0

    def test_with_data(self, loaded_engine):
        result = _summarize(patient_id="patient-sarah-mitchell")
        assert result["total_blocks"] > 0


# ── Detect Belief Drift ─────────────────────────────────────────────────────

class TestDetectBeliefDrift:
    def test_empty_patient(self):
        result = _detect_drift(patient_id="unknown")
        assert result["status"] == "success"
        assert result["contradiction_count"] == 0

    def test_with_data(self, loaded_engine):
        result = _detect_drift(patient_id="patient-sarah-mitchell")
        assert result["status"] == "success"
        assert "escalation" in result


# ── Ingest Patient Data ──────────────────────────────────────────────────────

class TestIngestPatientData:
    def test_ingest_fhir_failure(self):
        result = _ingest(patient_id="p-1", fhir_server_url="ftp://invalid", fhir_access_token="tok")
        assert result["status"] == "error"

    @patch("mcp_server.server.FHIRClient")
    def test_ingest_success(self, mock_fhir_cls):
        mock_fhir = MagicMock()
        mock_fhir.patient_id = "p-test"
        mock_fhir.get_medications.return_value = []
        mock_fhir.get_conditions.return_value = []
        mock_fhir.get_allergies.return_value = []
        mock_fhir.get_observations.return_value = []
        mock_fhir_cls.return_value = mock_fhir
        result = _ingest(patient_id="p-test", fhir_server_url="https://fhir.test", fhir_access_token="tok")
        assert result["status"] == "success"

    @patch("mcp_server.server.FHIRClient")
    def test_ingest_general_exception(self, mock_fhir_cls):
        mock_fhir_cls.side_effect = RuntimeError("Boom")
        result = _ingest(patient_id="p-1", fhir_server_url="https://fhir.test", fhir_access_token="tok")
        assert result["status"] == "error"


# ── Explain Clinical Conflict ────────────────────────────────────────────────

class TestExplainClinicalConflict:
    def test_no_conflicts(self):
        result = _explain(patient_id="unknown")
        assert result["status"] == "success"
        assert result["abstained"] is True

    def test_with_data(self, loaded_engine):
        result = _explain(patient_id="patient-sarah-mitchell", conflict_index=0)
        assert result["status"] == "success"
        assert "narrative" in result


# ── Clinical Care Handoff ────────────────────────────────────────────────────

class TestClinicalCareHandoff:
    def test_handoff_no_data(self):
        result = _handoff(patient_id="unknown")
        assert result["status"] == "success"

    def test_handoff_with_data(self, loaded_engine):
        result = _handoff(patient_id="patient-sarah-mitchell")
        assert result["status"] == "success"
        assert "handoff_note" in result


# ── What-If Medication Change ────────────────────────────────────────────────

class TestWhatIfMedicationChange:
    def test_add_medication(self, loaded_engine):
        result = _whatif(patient_id="patient-sarah-mitchell", action="add", medication="ibuprofen")
        assert result["status"] == "success"

    def test_remove_medication(self, loaded_engine):
        result = _whatif(patient_id="patient-sarah-mitchell", action="remove", medication="warfarin")
        assert result["status"] == "success"

    def test_swap_medication(self, loaded_engine):
        result = _whatif(patient_id="patient-sarah-mitchell", action="swap", medication="warfarin", swap_to="apixaban")
        assert result["status"] == "success"

    def test_invalid_action(self, loaded_engine):
        result = _whatif(patient_id="patient-sarah-mitchell", action="invalid", medication="test")
        assert result["status"] == "error"


# ── Verify Clinical Claims ──────────────────────────────────────────────────

class TestVerifyClinicalClaims:
    def test_no_data(self):
        result = _verify_claims(text="Patient has diabetes", patient_id="unknown")
        assert result["status"] == "success"

    def test_with_data(self, loaded_engine):
        result = _verify_claims(text="Patient takes warfarin and has penicillin allergy", patient_id="patient-sarah-mitchell")
        assert result["status"] == "success"


# ── Scan for PHI ─────────────────────────────────────────────────────────────

class TestScanForPhi:
    def test_safe_text(self):
        result = _scan_phi(text="The patient has hypertension.")
        assert result["status"] == "success"
        assert result["is_safe"] is True

    def test_text_with_phi(self):
        result = _scan_phi(text="Patient SSN: 123-45-6789, phone 555-123-4567")
        assert result["phi_count"] > 0


# ── FDA Safety Alerts ────────────────────────────────────────────────────────

class TestFDASafetyAlerts:
    def test_no_medications(self):
        result = _fda(patient_id="unknown")
        assert result["status"] == "success"
        assert result["medications"] == []

    @patch("engine.fda_client.get_safety_profile")
    def test_with_medications(self, mock_profile, loaded_engine):
        from engine.fda_client import FDASafetyProfile
        mock_profile.return_value = FDASafetyProfile(
            medications=["warfarin"], alerts=(), black_box_warnings=(), total_adverse_events=0, highest_severity="none",
        )
        result = _fda(patient_id="patient-sarah-mitchell")
        assert result["status"] == "success"


# ── Find Matching Trials ────────────────────────────────────────────────────

class TestFindMatchingTrials:
    def test_no_conditions(self):
        result = _trials(patient_id="unknown")
        assert result["status"] == "success"
        assert result["trials"] == []

    @patch("engine.trials_client.match_patient_to_trials")
    def test_with_conditions(self, mock_match, loaded_engine):
        from engine.trials_client import TrialMatchResult
        mock_match.return_value = TrialMatchResult(patient_conditions=("T2DM",), matched_trials=(), total_found=0, search_terms=("T2DM",))
        result = _trials(patient_id="patient-sarah-mitchell")
        assert result["status"] == "success"


# ── Consensus Verify Finding ────────────────────────────────────────────────

class TestConsensusVerifyFinding:
    @patch("engine.consensus_engine.verify_finding_consensus_sync")
    def test_with_data(self, mock_verify, loaded_engine):
        from engine.consensus_engine import ConsensusResult, LLMVerdict
        mock_verify.return_value = ConsensusResult(
            finding="test", consensus_level="strong", agreement_count=3, total_models=3,
            confidence_score=0.95, should_report=True,
            verdicts=(LLMVerdict(model="gpt", agrees=True, confidence=0.9, reasoning="ok"),),
            reasoning_summary="ok",
        )
        result = _consensus(finding="Warfarin + Ibuprofen bleeding", patient_id="patient-sarah-mitchell")
        assert result["status"] == "success"


# ── Health Check ─────────────────────────────────────────────────────────────

class TestHealthCheck:
    def test_health_check(self):
        result = _health()
        assert result["status"] == "healthy"
        assert result["engine_ready"] is True


# ── Helpers ──────────────────────────────────────────────────────────────────

class TestIsRelated:
    def test_diabetes_metformin(self):
        assert srv._is_related("diabetes mellitus", "metformin") is True

    def test_unrelated(self):
        assert srv._is_related("headache", "metformin") is False


class TestAutoIngest:
    def test_no_credentials(self):
        srv._auto_ingest("p-1", "", "")

    def test_already_has_data(self, loaded_engine):
        srv._auto_ingest("patient-sarah-mitchell", "https://fhir", "tok")

    @patch("mcp_server.server.FHIRClient")
    def test_auto_ingest_failure(self, mock_fhir_cls):
        mock_fhir_cls.side_effect = Exception("Boom")
        srv._auto_ingest("p-new", "https://fhir", "tok")

    @patch("mcp_server.server.FHIRClient")
    @patch("mcp_server.server.FHIRContext")
    def test_auto_ingest_success_path(self, mock_ctx_cls, mock_fhir_cls):
        """Line 589: _engine.ingest_from_fhir(fhir) executes successfully."""
        mock_fhir_instance = MagicMock()
        mock_fhir_cls.return_value = mock_fhir_instance
        mock_ctx_cls.return_value = MagicMock()

        with patch.object(
            srv._engine, "ingest_from_fhir",
            return_value={"MedicationRequest": 2, "Condition": 1},
        ) as mock_ingest:
            srv._auto_ingest("p-new-success", "https://fhir.test", "tok")

        mock_ctx_cls.assert_called_once()
        mock_fhir_cls.assert_called_once()
        mock_ingest.assert_called_once_with(mock_fhir_instance)


# ── Recall auto-ingest exception (lines 170-171) ────────────────────────────

class TestRecallAutoIngestException:
    @patch("mcp_server.server.FHIRClient")
    def test_recall_auto_ingest_fails_gracefully(self, mock_fhir_cls):
        """Lines 170-171: auto-ingest failure during recall is caught."""
        mock_fhir_cls.side_effect = Exception("Connection refused")
        result = _recall(
            patient_id="p-new", query="test",
            fhir_server_url="https://fhir.test", fhir_access_token="tok",
        )
        assert result["status"] == "success"
        assert result["result_count"] == 0


# ── Contraindicated recommendation (line 217) ────────────────────────────────

class TestContraindicatedRecommendation:
    def test_contraindicated_interaction_recommendation(self, loaded_engine):
        """Line 217: contraindicated interactions produce STOP recommendation."""
        from engine.clinical_scoring import DrugInteraction
        mock_report = MagicMock()
        mock_report.interactions = [
            DrugInteraction(
                drug_a="warfarin", drug_b="aspirin",
                severity="contraindicated",
                description="Life-threatening combination",
                score=1.0,
            )
        ]
        mock_report.allergy_conflicts = []
        mock_report.score = 1.0
        with patch.object(srv._engine, "medication_safety_check", return_value=mock_report):
            result = _check_meds(patient_id="patient-sarah-mitchell")
        assert result["status"] == "success"
        assert any("STOP" in r for r in result["recommendations"])


# ── Line 443: elif high branch in detect_belief_drift ────────────────────────


class TestDetectBeliefDriftHighOnly:
    """Line 442-446: contradictions with only high severity (no critical)."""

    def test_high_only_escalation(self, loaded_engine):
        high_only = [
            {
                "type": "drug_interaction",
                "severity": "high",
                "description": "Warfarin + aspirin increases bleeding risk",
            },
            {
                "type": "lab_medication",
                "severity": "high",
                "description": "Declining renal function with NSAID use",
            },
        ]

        with patch.object(
            srv._engine, "detect_contradictions", return_value=high_only
        ):
            result = _detect_drift(patient_id="patient-sarah-mitchell")

        assert result["status"] == "success"
        assert result["critical_count"] == 0
        assert result["high_count"] == 2
        assert result["has_critical"] is False
        assert result["escalation"] is not None
        assert "PRIORITY REVIEW RECOMMENDED" in result["escalation"]
        assert "2 high-severity" in result["escalation"]


# ── Lines 51-62: DEMO_MODE module-level init ─────────────────────────────────


class TestDemoModeInit:
    """Lines 50-62: DEMO_MODE environment variable triggers fixture loading."""

    def test_demo_mode_loads_fixture(self):
        """Reimport server module with DEMO_MODE=true to trigger lines 50-62."""
        import importlib
        import sys

        mods_to_remove = [k for k in sys.modules if k.startswith("mcp_server.server")]
        saved = {}
        for mod_name in mods_to_remove:
            saved[mod_name] = sys.modules.pop(mod_name)

        try:
            with patch.dict(os.environ, {"DEMO_MODE": "true", "GEMINI_API_KEY": ""}):
                import mcp_server.server as srv_mod
                importlib.reload(srv_mod)
                # Verify fixture data was loaded for demo patient
                assert "patient-sarah-mitchell" in srv_mod._engine._patient_blocks
                assert len(srv_mod._engine._patient_blocks["patient-sarah-mitchell"]) > 0
        finally:
            for mod_name, mod in saved.items():
                sys.modules[mod_name] = mod

    def test_demo_mode_exception_caught(self):
        """Lines 61-62: Exception during demo auto-load is caught and logged."""
        import importlib
        import sys

        mods_to_remove = [k for k in sys.modules if k.startswith("mcp_server.server")]
        saved = {}
        for mod_name in mods_to_remove:
            saved[mod_name] = sys.modules.pop(mod_name)

        try:
            with patch.dict(os.environ, {"DEMO_MODE": "1", "GEMINI_API_KEY": ""}):
                with patch("json.load", side_effect=Exception("corrupt fixture")):
                    import mcp_server.server as srv_mod
                    importlib.reload(srv_mod)
                    # Module should still load — exception is caught at line 61-62
                    assert srv_mod._engine is not None
        finally:
            for mod_name, mod in saved.items():
                sys.modules[mod_name] = mod


# ── Lines 990-993: if __name__ == "__main__" block ───────────────────────────


class TestMainBlock:
    """Lines 990-993: __main__ guard and __main__.py entry point."""

    def test_main_block_exists_in_source(self):
        """Verify the __main__ guard exists in server.py source."""
        import inspect
        source = inspect.getsource(srv)
        assert 'if __name__ == "__main__"' in source

    def test_main_module_entry_point_exists(self):
        """Verify mcp_server/__main__.py exists and imports correctly."""
        import mcp_server.__main__ as main_mod
        assert hasattr(main_mod, "main")


# ── __main__.py lines 6-7 via runpy ─────────────────────────────────────────


class TestMainModuleRunpy:
    """Cover mcp_server/__main__.py entry point via direct main() call."""

    def test_main_module_calls_main(self):
        """__main__.py imports and delegates to server.main()."""
        import mcp_server.server as srv
        with patch.object(srv.mcp, "run") as mock_run:
            # Call main() directly — this is what __main__.py's if-guard invokes
            srv.main()
            mock_run.assert_called_once()
            kw = mock_run.call_args.kwargs
            assert kw["transport"] == "streamable-http"
            assert kw["host"] == "0.0.0.0"


# ── server.py main() function ───────────────────────────────────────────────


class TestServerMainFunction:
    """Cover server.py main() function (extracted from __main__ guard)."""

    def test_main_default_port(self):
        import mcp_server.server as srv
        with patch.object(srv.mcp, "run") as mock_run:
            srv.main()
            mock_run.assert_called_once()
            assert mock_run.call_args.kwargs["port"] == 8080

    def test_main_custom_port(self):
        import mcp_server.server as srv
        with patch.object(srv.mcp, "run") as mock_run:
            with patch.dict("os.environ", {"PORT": "3000"}):
                srv.main()
                assert mock_run.call_args.kwargs["port"] == 3000
