"""Tests for what-if scenario engine — digital twin clinical simulation."""
import pytest
from unittest.mock import patch

from engine.what_if import (
    simulate_add_medication,
    simulate_remove_medication,
    simulate_swap_medication,
    WhatIfResult,
)
from engine.clinical_scoring import DrugInteraction, AllergyConflict, LabMedContraindication


def _interaction(drug_a, drug_b, severity="moderate", description="interaction"):
    return DrugInteraction(
        drug_a=drug_a,
        drug_b=drug_b,
        severity=severity,
        description=description,
        score=0.8,
    )


def _allergy(allergen, medication, severity="moderate"):
    return AllergyConflict(
        allergen=allergen,
        medication=medication,
        cross_reaction_group="test_group",
        description=f"{allergen} allergy with {medication}",
    )


def _lab_contra(lab_name, medication, severity="moderate"):
    return LabMedContraindication(
        lab_name=lab_name,
        lab_value=5.8,
        lab_unit="mEq/L",
        medication=medication,
        threshold=5.0,
        direction="above",
        severity=severity,
        description=f"{lab_name} contraindication with {medication}",
        recommendation="Monitor closely",
    )


class TestSimulateAddMedication:
    @patch("engine.what_if.check_lab_medication_contraindications", return_value=[])
    @patch("engine.what_if.check_allergy_conflicts", return_value=[])
    @patch("engine.what_if.check_drug_interactions")
    def test_no_new_risks(self, mock_interactions, mock_allergy, mock_lab):
        mock_interactions.return_value = []
        result = simulate_add_medication(
            "P001", "metformin", ["lisinopril"], [], []
        )
        assert isinstance(result, WhatIfResult)
        assert result.risk_delta == 0
        assert result.safe_to_proceed is True
        assert "NO new interactions" in result.recommendation

    @patch("engine.what_if.check_lab_medication_contraindications", return_value=[])
    @patch("engine.what_if.check_allergy_conflicts", return_value=[])
    @patch("engine.what_if.check_drug_interactions")
    def test_new_interaction_detected(self, mock_interactions, mock_allergy, mock_lab):
        # Baseline: no interactions
        # Scenario: warfarin+aspirin interaction
        mock_interactions.side_effect = [
            [],  # baseline
            [_interaction("warfarin", "aspirin", "moderate")],  # scenario
        ]
        result = simulate_add_medication(
            "P001", "aspirin", ["warfarin"], [], []
        )
        assert result.risk_delta == 1
        assert len(result.new_interactions) == 1
        assert result.new_interactions[0].drug_a == "warfarin"

    @patch("engine.what_if.check_lab_medication_contraindications", return_value=[])
    @patch("engine.what_if.check_allergy_conflicts")
    @patch("engine.what_if.check_drug_interactions", return_value=[])
    def test_allergy_conflict_blocks(self, mock_interactions, mock_allergy, mock_lab):
        mock_allergy.side_effect = [
            [],  # baseline
            [_allergy("penicillin", "amoxicillin", "critical")],  # scenario
        ]
        result = simulate_add_medication(
            "P001", "amoxicillin", ["metformin"], ["penicillin"], []
        )
        assert result.safe_to_proceed is False
        assert len(result.new_allergy_conflicts) == 1
        assert "DO NOT ADD" in result.recommendation

    @patch("engine.what_if.check_lab_medication_contraindications")
    @patch("engine.what_if.check_allergy_conflicts", return_value=[])
    @patch("engine.what_if.check_drug_interactions", return_value=[])
    def test_critical_lab_contraindication(self, mock_interactions, mock_allergy, mock_lab):
        mock_lab.side_effect = [
            [],  # baseline
            [_lab_contra("potassium", "spironolactone", "critical")],  # scenario
        ]
        result = simulate_add_medication(
            "P001", "spironolactone", ["lisinopril"], [], [{"code": "K", "value": 5.8}]
        )
        assert result.safe_to_proceed is False
        assert "DO NOT ADD" in result.recommendation

    @patch("engine.what_if.check_lab_medication_contraindications", return_value=[])
    @patch("engine.what_if.check_allergy_conflicts", return_value=[])
    @patch("engine.what_if.check_drug_interactions")
    def test_contraindicated_interaction(self, mock_interactions, mock_allergy, mock_lab):
        mock_interactions.side_effect = [
            [],
            [_interaction("methotrexate", "trimethoprim", "contraindicated")],
        ]
        result = simulate_add_medication(
            "P001", "trimethoprim", ["methotrexate"], [], []
        )
        assert result.safe_to_proceed is False
        assert "CONTRAINDICATED" in result.recommendation

    def test_scenario_label(self):
        with patch("engine.what_if.check_drug_interactions", return_value=[]), \
             patch("engine.what_if.check_allergy_conflicts", return_value=[]), \
             patch("engine.what_if.check_lab_medication_contraindications", return_value=[]):
            result = simulate_add_medication("P001", "aspirin", [], [], [])
        assert result.scenario == "Add aspirin"
        assert result.patient_id == "P001"


class TestSimulateRemoveMedication:
    @patch("engine.what_if.check_lab_medication_contraindications", return_value=[])
    @patch("engine.what_if.check_drug_interactions")
    def test_removes_interaction(self, mock_interactions, mock_lab):
        interaction = _interaction("warfarin", "aspirin")
        mock_interactions.side_effect = [
            [interaction],  # baseline with interaction
            [],  # scenario without aspirin
        ]
        result = simulate_remove_medication(
            "P001", "aspirin", ["warfarin", "aspirin"], [], []
        )
        assert result.risk_delta < 0
        assert len(result.removed_risks) > 0
        assert "eliminates" in result.recommendation.lower()

    @patch("engine.what_if.check_lab_medication_contraindications", return_value=[])
    @patch("engine.what_if.check_drug_interactions", return_value=[])
    def test_no_change_in_risk(self, mock_interactions, mock_lab):
        result = simulate_remove_medication(
            "P001", "metformin", ["metformin", "lisinopril"], [], []
        )
        assert result.risk_delta == 0
        assert result.safe_to_proceed is True

    @patch("engine.what_if.check_lab_medication_contraindications", return_value=[])
    @patch("engine.what_if.check_drug_interactions", return_value=[])
    def test_always_safe(self, mock_interactions, mock_lab):
        result = simulate_remove_medication(
            "P001", "drug_x", ["drug_x"], [], []
        )
        assert result.safe_to_proceed is True


class TestSimulateSwapMedication:
    @patch("engine.what_if.check_lab_medication_contraindications", return_value=[])
    @patch("engine.what_if.check_allergy_conflicts", return_value=[])
    @patch("engine.what_if.check_drug_interactions")
    def test_swap_safer(self, mock_interactions, mock_allergy, mock_lab):
        interaction = _interaction("warfarin", "aspirin")
        # simulate_swap calls simulate_add first, then simulate_remove
        # simulate_add: baseline(meds_after_remove), scenario(meds_after_remove+new)
        # simulate_remove: baseline(current_meds), scenario(current_meds-removed)
        mock_interactions.side_effect = [
            [],  # add baseline: ["warfarin"] — no interactions
            [],  # add scenario: ["warfarin", "clopidogrel"] — no new
            [interaction],  # remove baseline: ["warfarin", "aspirin"] — has interaction
            [],  # remove scenario: ["warfarin"] — interaction gone
        ]
        result = simulate_swap_medication(
            "P001", "aspirin", "clopidogrel", ["warfarin", "aspirin"], [], []
        )
        assert result.risk_delta < 0
        assert "SAFER" in result.recommendation

    @patch("engine.what_if.check_lab_medication_contraindications", return_value=[])
    @patch("engine.what_if.check_allergy_conflicts", return_value=[])
    @patch("engine.what_if.check_drug_interactions", return_value=[])
    def test_swap_neutral(self, mock_interactions, mock_allergy, mock_lab):
        result = simulate_swap_medication(
            "P001", "drug_a", "drug_b", ["drug_a"], [], []
        )
        assert result.risk_delta == 0
        assert "NEUTRAL" in result.recommendation

    @patch("engine.what_if.check_lab_medication_contraindications", return_value=[])
    @patch("engine.what_if.check_allergy_conflicts", return_value=[])
    @patch("engine.what_if.check_drug_interactions")
    def test_swap_increases_risk(self, mock_interactions, mock_allergy, mock_lab):
        new_interaction = _interaction("warfarin", "new_drug")
        # simulate_swap calls simulate_add first, then simulate_remove
        mock_interactions.side_effect = [
            [],  # add baseline: ["warfarin"] — clean
            [new_interaction],  # add scenario: ["warfarin", "new_drug"] — NEW risk
            [],  # remove baseline: ["warfarin", "aspirin"] — clean
            [],  # remove scenario: ["warfarin"] — no change
        ]
        result = simulate_swap_medication(
            "P001", "aspirin", "new_drug", ["warfarin", "aspirin"], [], []
        )
        assert result.risk_delta > 0
        assert "INCREASES" in result.recommendation

    @patch("engine.what_if.check_lab_medication_contraindications", return_value=[])
    @patch("engine.what_if.check_allergy_conflicts", return_value=[])
    @patch("engine.what_if.check_drug_interactions", return_value=[])
    def test_scenario_label(self, mock_interactions, mock_allergy, mock_lab):
        result = simulate_swap_medication(
            "P001", "drug_a", "drug_b", ["drug_a"], [], []
        )
        assert result.scenario == "Swap drug_a → drug_b"
