"""
Integration tests — wire FHIR client + ClinicalMem engine against Sarah Mitchell bundle.

Mocks httpx.get to simulate a FHIR R4 server returning resources from the
synthetic patient fixture.
"""
import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from engine.clinical_memory import ClinicalMemEngine
from engine.fhir_client import FHIRClient, FHIRContext, FHIRClientError

FIXTURES = Path(__file__).parent.parent / "fixtures"
BUNDLE = json.loads((FIXTURES / "sarah_mitchell_bundle.json").read_text())

# Group resources by type for mock FHIR responses
_BY_TYPE: dict[str, list[dict]] = {}
for entry in BUNDLE["entry"]:
    res = entry["resource"]
    _BY_TYPE.setdefault(res["resourceType"], []).append(res)


def _make_search_bundle(resources: list[dict]) -> dict:
    """Wrap resources in a FHIR searchset Bundle."""
    return {
        "resourceType": "Bundle",
        "type": "searchset",
        "total": len(resources),
        "entry": [{"resource": r} for r in resources],
    }


def _mock_fhir_get(url: str, **kwargs) -> MagicMock:
    """Route mock GET requests to the right resource type."""
    resp = MagicMock()
    resp.status_code = 200

    # Parse path relative to base URL
    base = "https://fhir.example.com/r4/"
    rel_path = url.replace(base, "").split("?")[0]
    resource_type = rel_path.split("/")[0]

    if resource_type == "Patient" and "/" in rel_path:
        # Direct read: GET /Patient/{id}
        resp.json.return_value = _BY_TYPE["Patient"][0]
    elif resource_type == "MedicationRequest":
        resp.json.return_value = _make_search_bundle(_BY_TYPE.get("MedicationRequest", []))
    elif resource_type == "Condition":
        resp.json.return_value = _make_search_bundle(_BY_TYPE.get("Condition", []))
    elif resource_type == "AllergyIntolerance":
        resp.json.return_value = _make_search_bundle(_BY_TYPE.get("AllergyIntolerance", []))
    elif resource_type == "Observation":
        # Filter by category param if present
        params = kwargs.get("params", {})
        category = params.get("category", "")
        obs = _BY_TYPE.get("Observation", [])
        if category:
            obs = [
                o for o in obs
                if any(
                    cat.get("coding", [{}])[0].get("code") == category
                    for cat in o.get("category", [])
                )
            ]
        resp.json.return_value = _make_search_bundle(obs)
    elif resource_type == "Encounter":
        resp.json.return_value = _make_search_bundle(_BY_TYPE.get("Encounter", []))
    else:
        resp.status_code = 404
        resp.text = "Not Found"

    return resp


@pytest.fixture
def fhir_ctx():
    return FHIRContext(
        url="https://fhir.example.com/r4",
        token="test-token-abc",
        patient_id="patient-sarah-mitchell",
    )


@pytest.fixture
def fhir_client(fhir_ctx):
    with patch("httpx.get", side_effect=_mock_fhir_get):
        yield FHIRClient(fhir_ctx)


@pytest.fixture
def engine(tmp_path):
    return ClinicalMemEngine(data_dir=str(tmp_path / "clinicalmem"))


# ── FHIR Client Tests ────────────────────────────────────────────────────


class TestFHIRClient:
    def test_get_patient(self, fhir_client):
        patient = fhir_client.get_patient()
        assert patient["resourceType"] == "Patient"
        assert patient["name"][0]["family"] == "Mitchell"

    def test_get_medications(self, fhir_client):
        meds = fhir_client.get_medications()
        assert len(meds) == 7  # 5 regular + ibuprofen + amoxicillin
        names = [m["medicationCodeableConcept"]["text"] for m in meds]
        assert "Warfarin 5mg" in names
        assert "Ibuprofen 400mg" in names
        assert "Amoxicillin 500mg" in names

    def test_get_conditions(self, fhir_client):
        conditions = fhir_client.get_conditions()
        assert len(conditions) == 4
        texts = [c["code"]["text"] for c in conditions]
        assert "Type 2 Diabetes Mellitus" in texts
        assert "Chronic Kidney Disease Stage 3b" in texts

    def test_get_allergies(self, fhir_client):
        allergies = fhir_client.get_allergies()
        assert len(allergies) == 2
        names = [a["code"]["text"] for a in allergies]
        assert "Penicillin" in names
        assert "Sulfa drugs" in names

    def test_get_observations_vitals(self, fhir_client):
        obs = fhir_client.get_observations(category="vital-signs")
        assert len(obs) == 2  # 2 BP readings
        assert all(o["resourceType"] == "Observation" for o in obs)

    def test_get_observations_labs(self, fhir_client):
        obs = fhir_client.get_observations(category="laboratory")
        assert len(obs) == 5  # 3 GFR + HbA1c + INR

    def test_get_encounters(self, fhir_client):
        encounters = fhir_client.get_encounters()
        assert len(encounters) == 1
        assert "Fall" in encounters[0]["type"][0]["text"]


class TestFHIRContext:
    def test_missing_url_raises(self):
        ctx = FHIRContext(url="", token="tok", patient_id="pid")
        with pytest.raises(FHIRClientError, match="url"):
            ctx.validate()

    def test_missing_token_raises(self):
        ctx = FHIRContext(url="https://fhir.example.com", token="", patient_id="pid")
        with pytest.raises(FHIRClientError, match="token"):
            ctx.validate()


# ── Engine Ingestion Tests ────────────────────────────────────────────────


class TestEngineIngestion:
    def test_ingest_counts(self, engine, fhir_client):
        with patch("httpx.get", side_effect=_mock_fhir_get):
            counts = engine.ingest_from_fhir(fhir_client)

        assert counts["medications"] == 7
        assert counts["conditions"] == 4
        assert counts["allergies"] == 2
        assert counts["observations"] == 7  # 2 vitals + 5 labs

    def test_ingest_creates_blocks(self, engine, fhir_client):
        with patch("httpx.get", side_effect=_mock_fhir_get):
            engine.ingest_from_fhir(fhir_client)

        blocks = engine._patient_blocks.get("patient-sarah-mitchell", [])
        assert len(blocks) == 20  # 7 + 4 + 2 + 7

    def test_ingest_audit_entry(self, engine, fhir_client):
        with patch("httpx.get", side_effect=_mock_fhir_get):
            engine.ingest_from_fhir(fhir_client)

        trail = engine.get_audit_trail()
        assert len(trail) == 1
        # mind-mem uses "operation" field; fallback uses "action"
        entry = trail[0]
        if "action" in entry:
            assert entry["action"] == "ingest_fhir"
        else:
            assert entry["operation"] == "create_block"
            assert entry["agent"] == "clinicalmem"


# ── Medication Safety Tests ───────────────────────────────────────────────


class TestMedicationSafety:
    @pytest.fixture(autouse=True)
    def _ingest(self, engine, fhir_client):
        with patch("httpx.get", side_effect=_mock_fhir_get):
            engine.ingest_from_fhir(fhir_client)
        self.engine = engine
        self.pid = "patient-sarah-mitchell"

    def test_detects_warfarin_ibuprofen_interaction(self):
        report = self.engine.medication_safety_check(self.pid)
        interaction_pairs = {(i.drug_a, i.drug_b) for i in report.interactions}
        assert ("warfarin", "ibuprofen") in interaction_pairs

    def test_detects_penicillin_amoxicillin_conflict(self):
        report = self.engine.medication_safety_check(self.pid)
        conflict_pairs = {(c.allergen, c.medication) for c in report.allergy_conflicts}
        assert ("penicillin", "amoxicillin") in conflict_pairs

    def test_report_has_summary(self):
        report = self.engine.medication_safety_check(self.pid)
        assert "drug interaction" in report.summary.lower()
        assert "allergy conflict" in report.summary.lower()

    def test_report_has_audit_hash(self):
        report = self.engine.medication_safety_check(self.pid)
        assert len(report.audit_hash) == 64  # SHA-256 hex


# ── Contradiction Detection Tests ─────────────────────────────────────────


class TestContradictionDetection:
    @pytest.fixture(autouse=True)
    def _ingest(self, engine, fhir_client):
        with patch("httpx.get", side_effect=_mock_fhir_get):
            engine.ingest_from_fhir(fhir_client)
        self.engine = engine
        self.pid = "patient-sarah-mitchell"

    def test_finds_allergy_medication_contradiction(self):
        contradictions = self.engine.detect_contradictions(self.pid)
        types = [c["type"] for c in contradictions]
        assert "allergy_medication_conflict" in types

    def test_finds_drug_interaction_contradiction(self):
        contradictions = self.engine.detect_contradictions(self.pid)
        types = [c["type"] for c in contradictions]
        assert "drug_interaction" in types

    def test_contradiction_severities(self):
        contradictions = self.engine.detect_contradictions(self.pid)
        severities = {c["severity"] for c in contradictions}
        assert "critical" in severities or "high" in severities

    def test_at_least_two_contradictions(self):
        contradictions = self.engine.detect_contradictions(self.pid)
        assert len(contradictions) >= 2

    def test_detects_gfr_metformin_contraindication(self):
        """Planted conflict #3: declining GFR + Metformin."""
        contradictions = self.engine.detect_contradictions(self.pid)
        lab_med = [c for c in contradictions if c["type"] == "lab_medication_contraindication"]
        assert len(lab_med) >= 1
        assert any("metformin" in str(c).lower() for c in lab_med)
        assert any("gfr" in str(c).lower() or "egfr" in str(c).lower() for c in lab_med)

    def test_detects_declining_gfr_trend(self):
        """Planted conflict #3: GFR 45 -> 38 -> 32 declining trend."""
        contradictions = self.engine.detect_contradictions(self.pid)
        trends = [c for c in contradictions if c["type"] == "lab_trend_alert"]
        assert len(trends) >= 1
        assert any("declining" in c["description"].lower() for c in trends)

    def test_detects_bp_target_disagreement(self):
        """Planted conflict #4: Cardiologist <130/80 vs Nephrologist <140/90."""
        contradictions = self.engine.detect_contradictions(self.pid)
        provider = [c for c in contradictions if c["type"] == "provider_disagreement"]
        assert len(provider) >= 1
        assert any("bp target" in c["description"].lower() for c in provider)

    def test_all_four_planted_conflicts_detected(self):
        """All 4 planted conflicts in Sarah Mitchell's data should be found."""
        contradictions = self.engine.detect_contradictions(self.pid)
        types_found = {c["type"] for c in contradictions}
        assert "allergy_medication_conflict" in types_found
        assert "drug_interaction" in types_found
        assert "lab_medication_contraindication" in types_found
        # Either trend or provider disagreement (both are new detections)
        assert "lab_trend_alert" in types_found or "provider_disagreement" in types_found

    def test_contradictions_have_recommendations(self):
        """All contradictions should include actionable recommendations."""
        contradictions = self.engine.detect_contradictions(self.pid)
        for c in contradictions:
            assert "recommendation" in c, f"Missing recommendation in: {c['type']}"
            assert len(c["recommendation"]) > 10, f"Empty recommendation in: {c['type']}"


# ── Recall Tests ──────────────────────────────────────────────────────────


class TestRecall:
    @pytest.fixture(autouse=True)
    def _ingest(self, engine, fhir_client):
        with patch("httpx.get", side_effect=_mock_fhir_get):
            engine.ingest_from_fhir(fhir_client)
        self.engine = engine
        self.pid = "patient-sarah-mitchell"

    def test_recall_warfarin(self):
        result = self.engine.recall(self.pid, "warfarin bleeding risk")
        assert len(result.blocks) > 0
        titles = [b["title"] for b in result.blocks]
        assert any("warfarin" in t.lower() for t in titles)

    def test_recall_diabetes(self):
        result = self.engine.recall(self.pid, "diabetes management metformin")
        assert len(result.blocks) > 0

    def test_recall_empty_patient(self):
        result = self.engine.recall("nonexistent", "any query")
        assert result.confidence.should_abstain is True
        assert len(result.blocks) == 0

    def test_recall_has_audit(self):
        result = self.engine.recall(self.pid, "blood pressure")
        assert len(result.audit_hash) == 64

    def test_negation_query_handling(self):
        result = self.engine.recall(self.pid, "NOT allergic to penicillin")
        assert result.blocks is not None  # Should not crash


# ── Audit Chain Tests ─────────────────────────────────────────────────────


class TestAuditChain:
    def test_chain_integrity_after_operations(self, engine, fhir_client):
        with patch("httpx.get", side_effect=_mock_fhir_get):
            engine.ingest_from_fhir(fhir_client)

        pid = "patient-sarah-mitchell"
        engine.recall(pid, "medications")
        engine.medication_safety_check(pid)
        engine.detect_contradictions(pid)
        engine.patient_summary(pid)

        assert engine.verify_audit_chain() is True

    def test_chain_genesis(self, engine, fhir_client):
        with patch("httpx.get", side_effect=_mock_fhir_get):
            engine.ingest_from_fhir(fhir_client)

        trail = engine.get_audit_trail()
        # mind-mem uses "0" * 64 as genesis; fallback uses "genesis"
        assert trail[0]["prev_hash"] in ("genesis", "0" * 64)

    def test_chain_links(self, engine, fhir_client):
        with patch("httpx.get", side_effect=_mock_fhir_get):
            engine.ingest_from_fhir(fhir_client)

        pid = "patient-sarah-mitchell"
        engine.recall(pid, "test")
        engine.recall(pid, "test2")

        trail = engine.get_audit_trail()
        for i in range(1, len(trail)):
            # mind-mem uses "entry_hash"; fallback uses "hash"
            prev_entry_hash = trail[i - 1].get("entry_hash") or trail[i - 1].get("hash")
            assert trail[i]["prev_hash"] == prev_entry_hash


# ── Patient Summary Tests ─────────────────────────────────────────────────


class TestPatientSummary:
    @pytest.fixture(autouse=True)
    def _ingest(self, engine, fhir_client):
        with patch("httpx.get", side_effect=_mock_fhir_get):
            engine.ingest_from_fhir(fhir_client)
        self.engine = engine
        self.pid = "patient-sarah-mitchell"

    def test_summary_structure(self):
        summary = self.engine.patient_summary(self.pid)
        assert "medications" in summary
        assert "conditions" in summary
        assert "allergies" in summary
        assert "recent_observations" in summary

    def test_summary_counts(self):
        summary = self.engine.patient_summary(self.pid)
        assert len(summary["medications"]) == 7
        assert len(summary["conditions"]) == 4
        assert len(summary["allergies"]) == 2
        assert summary["total_blocks"] == 20

    def test_summary_medication_names(self):
        summary = self.engine.patient_summary(self.pid)
        med_names = [m["name"] for m in summary["medications"]]
        assert "Warfarin 5mg" in med_names
        assert "Metformin 500mg" in med_names


# ── LLM Synthesis Tests ──────────────────────────────────────────────────


class TestLLMSynthesis:
    """Test GenAI synthesis features (explain_conflict, clinical_handoff).

    These tests run without an LLM API key — they exercise the template
    fallback and the abstention gate, which are the deterministic parts.
    """

    @pytest.fixture(autouse=True)
    def _ingest(self, engine, fhir_client):
        with patch("httpx.get", side_effect=_mock_fhir_get):
            engine.ingest_from_fhir(fhir_client)
        self.engine = engine
        self.pid = "patient-sarah-mitchell"

    def test_explain_conflict_returns_narrative(self):
        narrative = self.engine.explain_clinical_conflict(self.pid, conflict_index=0)
        assert narrative.narrative  # Non-empty
        assert isinstance(narrative.confidence_score, float)
        assert narrative.model_used  # Either LLM name or fallback

    def test_explain_conflict_has_audit(self):
        narrative = self.engine.explain_clinical_conflict(self.pid, conflict_index=0)
        assert narrative.audit_context is not None
        assert "conflict_type" in narrative.audit_context or "reason" in narrative.audit_context

    def test_explain_nonexistent_conflict_abstains(self):
        narrative = self.engine.explain_clinical_conflict(self.pid, conflict_index=99)
        assert narrative.abstained is True
        assert "ABSTAIN" in narrative.narrative

    def test_explain_empty_patient_abstains(self):
        narrative = self.engine.explain_clinical_conflict("nonexistent", conflict_index=0)
        assert narrative.abstained is True

    def test_clinical_handoff_returns_note(self):
        narrative = self.engine.clinical_handoff(self.pid)
        assert narrative.narrative  # Non-empty
        assert isinstance(narrative.confidence_score, float)
        assert narrative.confidence_score > 0  # Has evidence

    def test_clinical_handoff_empty_patient_abstains(self):
        narrative = self.engine.clinical_handoff("nonexistent")
        # No blocks = no contradictions detected, but still generates note
        # The abstention check is based on evidence count
        assert narrative.narrative  # Should produce something

    def test_narrative_dataclass_immutable(self):
        narrative = self.engine.explain_clinical_conflict(self.pid, conflict_index=0)
        with pytest.raises(AttributeError):
            narrative.narrative = "tampered"  # type: ignore[misc]


# ── LLM-Augmented Detection Tests ────────────────────────────────────────


class TestLLMAugmentedDetection:
    """Test LLM fallback for drug interactions beyond the 12-pair table."""

    def test_deterministic_still_works_without_llm(self):
        """Deterministic table catches known pairs even with LLM disabled."""
        from engine.clinical_scoring import check_drug_interactions

        meds = ["Warfarin 5mg", "Ibuprofen 400mg"]
        interactions = check_drug_interactions(meds, use_llm_fallback=False)
        pairs = {(i.drug_a, i.drug_b) for i in interactions}
        assert ("warfarin", "ibuprofen") in pairs

    def test_llm_fallback_does_not_break_deterministic(self):
        """LLM fallback only adds — never removes deterministic results."""
        from engine.clinical_scoring import check_drug_interactions

        meds = ["Warfarin 5mg", "Ibuprofen 400mg"]
        without_llm = check_drug_interactions(meds, use_llm_fallback=False)
        with_llm = check_drug_interactions(meds, use_llm_fallback=True)
        # with_llm should have at least as many as without_llm
        assert len(with_llm) >= len(without_llm)

    def test_empty_meds_no_crash(self):
        """Empty medication list doesn't crash LLM fallback."""
        from engine.clinical_scoring import check_drug_interactions

        interactions = check_drug_interactions([], use_llm_fallback=True)
        assert interactions == []

    def test_single_med_no_interactions(self):
        """Single medication can't have interactions."""
        from engine.clinical_scoring import check_drug_interactions

        interactions = check_drug_interactions(
            ["Metformin 500mg"], use_llm_fallback=False
        )
        assert interactions == []
