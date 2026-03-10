"""Full coverage tests for a2a_agent/tools/safety_tools.py."""
import json
import os
import pytest
from unittest.mock import patch, MagicMock

a2a_types = pytest.importorskip("a2a.types")
google_adk = pytest.importorskip("google.adk")


def _make_tool_context(state=None, patient_id="patient-sarah-mitchell"):
    ctx = MagicMock()
    ctx.state = state if state is not None else {"patient_id": patient_id}
    ctx.metadata = None
    ctx.run_config = None
    return ctx


def _load_engine():
    """Load Sarah Mitchell fixture data into the shared engine."""
    from a2a_agent.tools.memory_tools import _engine
    fixture_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "tests", "fixtures", "sarah_mitchell_bundle.json",
    )
    with open(fixture_path) as f:
        bundle = json.load(f)
    _engine.ingest_from_bundle(bundle, "patient-sarah-mitchell")
    return _engine


@pytest.fixture(autouse=True)
def reset_engine():
    from a2a_agent.tools import memory_tools
    from engine.clinical_memory import ClinicalMemEngine
    memory_tools._engine = ClinicalMemEngine()
    yield


@pytest.fixture
def loaded():
    return _load_engine()


# ── _demo_patient_id ─────────────────────────────────────────────────────────

class TestDemoPatientId:
    def test_from_state(self):
        from a2a_agent.tools.safety_tools import _demo_patient_id
        ctx = _make_tool_context(state={"patient_id": "p-1"})
        assert _demo_patient_id(ctx) == "p-1"

    @patch.dict(os.environ, {"DEMO_MODE": "true"})
    def test_demo_fallback(self):
        from a2a_agent.tools.safety_tools import _demo_patient_id
        ctx = _make_tool_context(state={})
        assert _demo_patient_id(ctx) == "patient-sarah-mitchell"

    def test_no_patient_no_demo(self):
        from a2a_agent.tools.safety_tools import _demo_patient_id
        ctx = _make_tool_context(state={})
        assert _demo_patient_id(ctx) == ""


# ── _auto_ingest ─────────────────────────────────────────────────────────────

class TestAutoIngest:
    def test_no_credentials(self):
        from a2a_agent.tools.safety_tools import _auto_ingest
        ctx = _make_tool_context(state={})
        _auto_ingest(ctx)  # should not raise

    def test_already_has_data(self, loaded):
        from a2a_agent.tools.safety_tools import _auto_ingest
        ctx = _make_tool_context(state={
            "fhir_url": "https://fhir.test",
            "fhir_token": "tok",
            "patient_id": "patient-sarah-mitchell",
        })
        _auto_ingest(ctx)  # should not re-ingest

    @patch("a2a_agent.tools.safety_tools.FHIRClient")
    def test_ingest_failure(self, mock_fhir_cls):
        from a2a_agent.tools.safety_tools import _auto_ingest
        mock_fhir_cls.side_effect = Exception("network error")
        ctx = _make_tool_context(state={
            "fhir_url": "https://fhir.test",
            "fhir_token": "tok",
            "patient_id": "p-new",
        })
        _auto_ingest(ctx)  # should not raise


# ── medication_safety_review ─────────────────────────────────────────────────

class TestMedicationSafetyReview:
    def test_no_patient(self):
        from a2a_agent.tools.safety_tools import medication_safety_review
        ctx = _make_tool_context(state={})
        result = medication_safety_review(ctx)
        assert result["status"] == "error"

    def test_with_data(self, loaded):
        from a2a_agent.tools.safety_tools import medication_safety_review
        ctx = _make_tool_context()
        result = medication_safety_review(ctx)
        assert result["status"] == "success"
        assert "medications_reviewed" in result
        assert "drug_interactions" in result
        assert "allergy_conflicts" in result
        assert "critical_findings" in result


# ── detect_record_contradictions ─────────────────────────────────────────────

class TestDetectRecordContradictions:
    def test_no_patient(self):
        from a2a_agent.tools.safety_tools import detect_record_contradictions
        ctx = _make_tool_context(state={})
        result = detect_record_contradictions(ctx)
        assert result["status"] == "error"

    def test_with_data(self, loaded):
        from a2a_agent.tools.safety_tools import detect_record_contradictions
        ctx = _make_tool_context()
        result = detect_record_contradictions(ctx)
        assert result["status"] == "success"
        assert "contradictions" in result
        assert "chain_integrity" in result
        assert "escalation" in result


# ── what_if_scenario ─────────────────────────────────────────────────────────

class TestWhatIfScenario:
    def test_no_patient(self):
        from a2a_agent.tools.safety_tools import what_if_scenario
        ctx = _make_tool_context(state={})
        result = what_if_scenario(ctx, action="add", medication="ibuprofen")
        assert result["status"] == "error"

    def test_add(self, loaded):
        from a2a_agent.tools.safety_tools import what_if_scenario
        ctx = _make_tool_context()
        result = what_if_scenario(ctx, action="add", medication="ibuprofen")
        assert result["status"] == "success"
        assert "safe_to_proceed" in result

    def test_remove(self, loaded):
        from a2a_agent.tools.safety_tools import what_if_scenario
        ctx = _make_tool_context()
        result = what_if_scenario(ctx, action="remove", medication="warfarin")
        assert result["status"] == "success"

    def test_swap(self, loaded):
        from a2a_agent.tools.safety_tools import what_if_scenario
        ctx = _make_tool_context()
        result = what_if_scenario(ctx, action="swap", medication="warfarin", swap_to="apixaban")
        assert result["status"] == "success"

    def test_invalid_action(self, loaded):
        from a2a_agent.tools.safety_tools import what_if_scenario
        ctx = _make_tool_context()
        result = what_if_scenario(ctx, action="invalid", medication="test")
        assert result["status"] == "error"


# ── check_fda_alerts ─────────────────────────────────────────────────────────

class TestCheckFdaAlerts:
    def test_no_patient(self):
        from a2a_agent.tools.safety_tools import check_fda_alerts
        ctx = _make_tool_context(state={})
        result = check_fda_alerts(ctx)
        assert result["status"] == "error"

    def test_no_meds(self):
        from a2a_agent.tools.safety_tools import check_fda_alerts
        ctx = _make_tool_context(state={"patient_id": "empty-patient"})
        result = check_fda_alerts(ctx)
        assert result["status"] == "success"
        assert result["medications"] == []

    @patch("engine.fda_client.get_safety_profile")
    def test_with_meds(self, mock_profile, loaded):
        from a2a_agent.tools.safety_tools import check_fda_alerts
        from engine.fda_client import FDASafetyProfile
        mock_profile.return_value = FDASafetyProfile(
            medications=["warfarin"], alerts=[], black_box_warnings=[],
            total_adverse_events=0, highest_severity="none",
        )
        ctx = _make_tool_context()
        result = check_fda_alerts(ctx)
        assert result["status"] == "success"


# ── find_clinical_trials ─────────────────────────────────────────────────────

class TestFindClinicalTrials:
    def test_no_patient(self):
        from a2a_agent.tools.safety_tools import find_clinical_trials
        ctx = _make_tool_context(state={})
        result = find_clinical_trials(ctx)
        assert result["status"] == "error"

    def test_no_conditions(self):
        from a2a_agent.tools.safety_tools import find_clinical_trials
        ctx = _make_tool_context(state={"patient_id": "empty-patient"})
        result = find_clinical_trials(ctx)
        assert result["status"] == "success"
        assert result["trials"] == []

    @patch("engine.trials_client.match_patient_to_trials")
    def test_with_conditions(self, mock_match, loaded):
        from a2a_agent.tools.safety_tools import find_clinical_trials
        from engine.trials_client import TrialMatchResult
        mock_match.return_value = TrialMatchResult(
            patient_conditions=("T2DM",), matched_trials=(), total_found=0,
            search_terms=("T2DM",),
        )
        ctx = _make_tool_context()
        result = find_clinical_trials(ctx)
        assert result["status"] == "success"


# ── consensus_verify ─────────────────────────────────────────────────────────

class TestConsensusVerify:
    def test_no_patient(self):
        from a2a_agent.tools.safety_tools import consensus_verify
        ctx = _make_tool_context(state={})
        result = consensus_verify(ctx, finding="test")
        assert result["status"] == "error"

    def test_no_finding(self, loaded):
        from a2a_agent.tools.safety_tools import consensus_verify
        ctx = _make_tool_context()
        result = consensus_verify(ctx, finding="")
        assert result["status"] == "error"

    @patch("engine.consensus_engine.verify_finding_consensus_sync")
    def test_with_data(self, mock_verify, loaded):
        from a2a_agent.tools.safety_tools import consensus_verify
        from engine.consensus_engine import ConsensusResult, LLMVerdict
        mock_verify.return_value = ConsensusResult(
            finding="test", consensus_level="strong",
            agreement_count=3, total_models=3,
            confidence_score=0.95, should_report=True,
            verdicts=(LLMVerdict(model="gpt", agrees=True, confidence=0.9, reasoning="ok"),),
            reasoning_summary="ok",
        )
        ctx = _make_tool_context()
        result = consensus_verify(ctx, finding="Warfarin + Ibuprofen bleeding risk")
        assert result["status"] == "success"
        assert result["consensus_level"] == "strong"


# ── explain_clinical_conflict ────────────────────────────────────────────────

class TestExplainClinicalConflict:
    def test_no_patient(self):
        from a2a_agent.tools.safety_tools import explain_clinical_conflict
        ctx = _make_tool_context(state={})
        result = explain_clinical_conflict(ctx)
        assert result["status"] == "error"

    def test_with_data(self, loaded):
        from a2a_agent.tools.safety_tools import explain_clinical_conflict
        ctx = _make_tool_context()
        result = explain_clinical_conflict(ctx, conflict_index=0)
        assert result["status"] == "success"
        assert "narrative" in result
        assert "chain_integrity" in result


# ── Line 47: _auto_ingest exception when engine.ingest_from_fhir fails ──────


class TestAutoIngestIngestFromFhirException:
    """Line 47-49: FHIRClient/FHIRContext succeed but engine.ingest_from_fhir raises."""

    @patch("a2a_agent.tools.safety_tools.FHIRClient")
    @patch("a2a_agent.tools.safety_tools.FHIRContext")
    def test_ingest_from_fhir_raises_in_safety_auto_ingest(
        self, mock_ctx_cls, mock_fhir_cls
    ):
        from a2a_agent.tools.safety_tools import _auto_ingest
        from a2a_agent.tools.memory_tools import _engine

        mock_fhir_instance = MagicMock()
        mock_fhir_cls.return_value = mock_fhir_instance
        mock_ctx_cls.return_value = MagicMock()

        # Make engine.ingest_from_fhir raise — this hits line 47-48
        with patch.object(
            _engine, "ingest_from_fhir", side_effect=RuntimeError("FHIR ingest boom")
        ):
            ctx = _make_tool_context(state={
                "fhir_url": "https://fhir.test",
                "fhir_token": "tok",
                "patient_id": "p-brand-new",
            })
            # Should not raise — exception is caught at line 48-49
            _auto_ingest(ctx)

        # Verify both FHIRContext and FHIRClient were instantiated
        mock_ctx_cls.assert_called_once()
        mock_fhir_cls.assert_called_once()


# ── Line 147: elif high branch in detect_record_contradictions ───────────────


class TestDetectRecordContradictionsHighOnly:
    """Line 146-149: contradictions with high severity but no critical."""

    def test_high_only_escalation(self, loaded):
        from a2a_agent.tools.safety_tools import detect_record_contradictions
        from a2a_agent.tools.memory_tools import _engine

        # Mock detect_contradictions to return only high-severity items (no critical)
        high_only_contradictions = [
            {
                "type": "drug_interaction",
                "severity": "high",
                "description": "Warfarin + aspirin increases bleeding risk",
                "drug_a": "warfarin",
                "drug_b": "aspirin",
            },
            {
                "type": "lab_medication",
                "severity": "high",
                "description": "Declining renal function with NSAID use",
                "drug_a": "ibuprofen",
                "drug_b": "",
            },
        ]

        with patch.object(
            _engine, "detect_contradictions", return_value=high_only_contradictions
        ):
            ctx = _make_tool_context()
            result = detect_record_contradictions(ctx)

        assert result["status"] == "success"
        assert result["critical_count"] == 0
        assert result["high_count"] == 2
        assert result["has_critical"] is False
        assert result["has_high"] is True
        # The escalation should be the "PRIORITY REVIEW" message (elif high branch)
        assert result["escalation"] is not None
        assert "PRIORITY REVIEW RECOMMENDED" in result["escalation"]
        assert "2 high-severity" in result["escalation"]
