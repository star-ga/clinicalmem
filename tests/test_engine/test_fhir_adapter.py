# Copyright 2026 STARGA Inc. — Apache-2.0
"""Tests for engine.fhir_adapter — SMART-on-FHIR R4 ingress adapter.

Coverage targets
----------------
* Happy-path ingestion of the Sarah Mitchell demo bundle
* Malformed / invalid input rejection
* PHI scrubbing (MRN stripping from Patient.identifier)
* NPI Luhn validation of Practitioner resources
* Bundle SHA-256 canonical determinism
* Graceful handling of missing optional fields
"""
from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

_FIXTURE_PATH = _REPO_ROOT / "docs" / "sarah_mitchell_fhir.json"

from engine.fhir_adapter import (  # noqa: E402
    ClinicalIngestResult,
    NormalizedAllergy,
    NormalizedCondition,
    NormalizedMedication,
    NormalizedObservation,
    ingest_bundle,
)


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def sarah_bundle() -> dict:
    return json.loads(_FIXTURE_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def sarah_result(sarah_bundle: dict) -> ClinicalIngestResult:
    return ingest_bundle(sarah_bundle)


# ── happy path ────────────────────────────────────────────────────────────────

class TestIngestSarahMitchellBundle:
    """test_ingest_sarah_mitchell_bundle_returns_complete_result."""

    def test_patient_id_extracted(self, sarah_result: ClinicalIngestResult) -> None:
        assert sarah_result.patient_id == "sarah-mitchell"

    def test_four_conditions_returned(self, sarah_result: ClinicalIngestResult) -> None:
        assert len(sarah_result.normalized_conditions) == 4

    def test_condition_names_present(self, sarah_result: ClinicalIngestResult) -> None:
        names = {c.display_text for c in sarah_result.normalized_conditions}
        # Each condition has a text field in the fixture
        assert "Type 2 Diabetes Mellitus" in names
        assert "Hypertension" in names
        assert "Chronic Kidney Disease Stage 3b" in names
        assert "Atrial Fibrillation" in names

    def test_conditions_have_snomed_codes(self, sarah_result: ClinicalIngestResult) -> None:
        snomed_codes = {c.snomed_code for c in sarah_result.normalized_conditions
                        if c.snomed_code}
        assert "44054006" in snomed_codes   # T2DM
        assert "59621000" in snomed_codes   # HTN
        assert "433146006" in snomed_codes  # CKD stage 3
        assert "49436004" in snomed_codes   # AFib

    def test_conditions_have_icd10_codes(self, sarah_result: ClinicalIngestResult) -> None:
        icd10_codes = {c.icd10_code for c in sarah_result.normalized_conditions
                       if c.icd10_code}
        assert "E11.9" in icd10_codes
        assert "I10" in icd10_codes

    def test_four_medications_returned(self, sarah_result: ClinicalIngestResult) -> None:
        assert len(sarah_result.normalized_medications) == 4

    def test_medication_names_present(self, sarah_result: ClinicalIngestResult) -> None:
        names_lower = {m.lower() for m in sarah_result.normalized_medications}
        assert any("warfarin" in n for n in names_lower)
        assert any("metformin" in n for n in names_lower)
        assert any("lisinopril" in n for n in names_lower)
        assert any("ibuprofen" in n for n in names_lower)

    def test_medications_have_rxnorm_codes(self, sarah_result: ClinicalIngestResult) -> None:
        rxnorm_codes = {m.rxnorm_code for m in sarah_result.medications if m.rxnorm_code}
        assert "11289" in rxnorm_codes   # warfarin
        assert "6809" in rxnorm_codes    # metformin
        assert "29046" in rxnorm_codes   # lisinopril
        assert "5640" in rxnorm_codes    # ibuprofen

    def test_one_allergy_returned(self, sarah_result: ClinicalIngestResult) -> None:
        assert len(sarah_result.allergies) == 1

    def test_allergy_is_penicillin_high_criticality(
        self, sarah_result: ClinicalIngestResult
    ) -> None:
        allergy = sarah_result.allergies[0]
        assert allergy.criticality == "high"
        assert allergy.allergy_type == "allergy"
        assert "medication" in allergy.category

    def test_four_observations_returned(self, sarah_result: ClinicalIngestResult) -> None:
        assert len(sarah_result.observations) == 4

    def test_gfr_observation_value_and_unit(
        self, sarah_result: ClinicalIngestResult
    ) -> None:
        gfr_obs = [o for o in sarah_result.observations
                   if o.loinc_code == "62238-1"]
        assert len(gfr_obs) == 2  # two GFR readings in the fixture
        values = {o.value for o in gfr_obs}
        assert 32 in values
        assert 45 in values

    def test_four_practitioner_npis_returned(
        self, sarah_result: ClinicalIngestResult
    ) -> None:
        assert len(sarah_result.practitioner_npis) == 4

    def test_practitioner_npis_pass_luhn(
        self, sarah_result: ClinicalIngestResult
    ) -> None:
        from engine.npi_registry import validate_npi

        for npi in sarah_result.practitioner_npis:
            assert validate_npi(npi), f"NPI {npi} failed Luhn"

    def test_expected_npi_values(self, sarah_result: ClinicalIngestResult) -> None:
        npi_set = set(sarah_result.practitioner_npis)
        assert "1431602063" in npi_set   # Dr. Patel (PCP)
        assert "1642442994" in npi_set   # Dr. Liu (cardio)
        assert "1040623971" in npi_set   # Dr. Santos (nephro)
        assert "1857610905" in npi_set   # Dr. O'Connor (ER)

    def test_no_rejected_resources_for_valid_bundle(
        self, sarah_result: ClinicalIngestResult
    ) -> None:
        assert sarah_result.rejected_resources == []

    def test_bundle_sha256_is_hex_string(
        self, sarah_result: ClinicalIngestResult
    ) -> None:
        sha = sarah_result.bundle_sha256
        assert len(sha) == 64
        assert all(c in "0123456789abcdef" for c in sha)


# ── SHA-256 determinism ───────────────────────────────────────────────────────

class TestBundleSha256Determinism:
    """test_bundle_sha256_is_canonical_and_deterministic."""

    def test_same_bundle_same_hash_twice(self, sarah_bundle: dict) -> None:
        result_a = ingest_bundle(copy.deepcopy(sarah_bundle))
        result_b = ingest_bundle(copy.deepcopy(sarah_bundle))
        assert result_a.bundle_sha256 == result_b.bundle_sha256

    def test_hash_matches_canonical_json(self, sarah_bundle: dict) -> None:
        result = ingest_bundle(copy.deepcopy(sarah_bundle))
        canonical = json.dumps(
            sarah_bundle, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        )
        expected = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        assert result.bundle_sha256 == expected

    def test_json_string_input_produces_same_hash(self, sarah_bundle: dict) -> None:
        result_dict = ingest_bundle(copy.deepcopy(sarah_bundle))
        result_str = ingest_bundle(json.dumps(sarah_bundle))
        assert result_dict.bundle_sha256 == result_str.bundle_sha256


# ── invalid resource type rejection ──────────────────────────────────────────

class TestIngestRejectsInvalidResourceType:
    """test_ingest_rejects_invalid_resource_type."""

    def test_unknown_type_in_strict_mode_is_rejected(self) -> None:
        bundle = {
            "resourceType": "Bundle",
            "type": "collection",
            "entry": [
                {
                    "resource": {
                        "resourceType": "InventedNonFHIRResource",
                        "id": "bad-resource",
                    }
                }
            ],
        }
        result = ingest_bundle(bundle, strict_resource_types=True)
        assert len(result.rejected_resources) == 1
        rtype, reason = result.rejected_resources[0]
        assert rtype == "InventedNonFHIRResource"
        assert "unrecognised" in reason.lower()

    def test_unknown_type_in_tolerant_mode_is_silently_skipped(self) -> None:
        bundle = {
            "resourceType": "Bundle",
            "type": "collection",
            "entry": [
                {
                    "resource": {
                        "resourceType": "InventedNonFHIRResource",
                        "id": "bad-resource",
                    }
                }
            ],
        }
        result = ingest_bundle(bundle, strict_resource_types=False)
        assert result.rejected_resources == []

    def test_missing_resource_type_is_always_rejected(self) -> None:
        bundle = {
            "resourceType": "Bundle",
            "type": "collection",
            "entry": [{"resource": {"id": "no-type"}}],
        }
        result = ingest_bundle(bundle)
        # The adapter rejects entries with no resourceType — check it's recorded
        assert len(result.rejected_resources) == 1
        rtype, reason = result.rejected_resources[0]
        # The reason must mention the missing field somehow
        assert rtype == "(unknown)" or "resourceType" in rtype.lower() or "resourcetype" in reason.lower() or "no resourcetype" in reason.lower()

    def test_non_bundle_resource_type_raises(self) -> None:
        with pytest.raises(ValueError, match="Expected resourceType='Bundle'"):
            ingest_bundle({"resourceType": "Patient", "id": "p1", "entry": []})

    def test_missing_entry_array_raises(self) -> None:
        with pytest.raises(ValueError, match="entry must be an array"):
            ingest_bundle({"resourceType": "Bundle", "type": "collection"})

    def test_invalid_json_string_raises(self) -> None:
        with pytest.raises(ValueError, match="not valid JSON"):
            ingest_bundle("{not: valid json}")


# ── MRN stripping ─────────────────────────────────────────────────────────────

class TestIngestStripsMrnFromIdentifierArray:
    """test_ingest_strips_mrn_from_identifier_array."""

    def _bundle_with_patient_identifiers(
        self, identifiers: list[dict]
    ) -> dict:
        return {
            "resourceType": "Bundle",
            "type": "collection",
            "entry": [
                {
                    "resource": {
                        "resourceType": "Patient",
                        "id": "test-patient",
                        "identifier": identifiers,
                        "name": [{"family": "Test", "given": ["Patient"]}],
                        "gender": "unknown",
                        "birthDate": "1990-01-01",
                    }
                }
            ],
        }

    def test_mrn_identifier_by_type_code_is_stripped(self) -> None:
        bundle = self._bundle_with_patient_identifiers(
            [
                {
                    "system": "http://hospital.example.org/mrn",
                    "value": "MRN-987654",
                    "type": {
                        "coding": [
                            {
                                "system": "http://terminology.hl7.org/CodeSystem/v2-0203",
                                "code": "MR",
                                "display": "Medical Record Number",
                            }
                        ]
                    },
                }
            ]
        )
        result = ingest_bundle(bundle)
        assert result.phi_redactions == 1

    def test_ssn_identifier_is_stripped(self) -> None:
        bundle = self._bundle_with_patient_identifiers(
            [
                {
                    "system": "http://hl7.org/fhir/sid/us-ssn",
                    "value": "123-45-6789",
                }
            ]
        )
        result = ingest_bundle(bundle)
        assert result.phi_redactions == 1

    def test_ssn_oid_identifier_is_stripped(self) -> None:
        bundle = self._bundle_with_patient_identifiers(
            [
                {
                    "system": "urn:oid:2.16.840.1.113883.4.1",
                    "value": "987654321",
                }
            ]
        )
        result = ingest_bundle(bundle)
        assert result.phi_redactions == 1

    def test_non_phi_identifier_is_not_stripped(self) -> None:
        bundle = self._bundle_with_patient_identifiers(
            [
                {
                    "system": "http://hospital.example.org/account",
                    "value": "ACC-001",
                }
            ]
        )
        result = ingest_bundle(bundle)
        assert result.phi_redactions == 0

    def test_mixed_identifiers_only_phi_stripped(self) -> None:
        bundle = self._bundle_with_patient_identifiers(
            [
                {
                    "system": "http://hl7.org/fhir/sid/us-ssn",
                    "value": "123-45-6789",
                },
                {
                    "system": "http://hospital.example.org/account",
                    "value": "ACC-001",
                },
            ]
        )
        result = ingest_bundle(bundle)
        assert result.phi_redactions == 1


# ── NPI Luhn validation ───────────────────────────────────────────────────────

class TestIngestValidatesPractitionerNPIAgainstLuhn:
    """test_ingest_validates_practitioner_npi_against_luhn."""

    def _bundle_with_practitioner(self, npi_value: str) -> dict:
        return {
            "resourceType": "Bundle",
            "type": "collection",
            "entry": [
                {
                    "resource": {
                        "resourceType": "Practitioner",
                        "id": "prac-test",
                        "identifier": [
                            {
                                "system": "http://hl7.org/fhir/sid/us-npi",
                                "value": npi_value,
                            }
                        ],
                    }
                }
            ],
        }

    def test_luhn_valid_npi_is_accepted(self) -> None:
        # 1234567893 is the CMS spec example NPI
        result = ingest_bundle(self._bundle_with_practitioner("1234567893"))
        assert "1234567893" in result.practitioner_npis

    def test_luhn_invalid_npi_is_rejected(self) -> None:
        # 1234567890 — bad check digit (last digit should be 3)
        result = ingest_bundle(self._bundle_with_practitioner("1234567890"))
        assert "1234567890" not in result.practitioner_npis
        assert len(result.rejected_resources) == 1
        rtype, reason = result.rejected_resources[0]
        assert rtype == "Practitioner"
        assert "Luhn" in reason

    def test_non_npi_identifier_system_ignored(self) -> None:
        bundle = {
            "resourceType": "Bundle",
            "type": "collection",
            "entry": [
                {
                    "resource": {
                        "resourceType": "Practitioner",
                        "id": "prac-test",
                        "identifier": [
                            {
                                "system": "http://nucc.org/provider-taxonomy",
                                "value": "207R00000X",
                            }
                        ],
                    }
                }
            ],
        }
        result = ingest_bundle(bundle)
        assert result.practitioner_npis == []
        assert result.rejected_resources == []

    def test_sarah_bundle_all_four_npis_accepted(
        self, sarah_result: ClinicalIngestResult
    ) -> None:
        assert len(sarah_result.practitioner_npis) == 4
        assert len(sarah_result.rejected_resources) == 0


# ── missing optional fields ───────────────────────────────────────────────────

class TestIngestHandlesMissingOptionalFieldsGracefully:
    """test_ingest_handles_missing_optional_fields_gracefully."""

    def test_condition_without_text_uses_coding_display(self) -> None:
        bundle = {
            "resourceType": "Bundle",
            "type": "collection",
            "entry": [
                {
                    "resource": {
                        "resourceType": "Condition",
                        "id": "cond-1",
                        "subject": {"reference": "Patient/p1"},
                        "code": {
                            # No "text" field — should fall back to coding display
                            "coding": [
                                {
                                    "system": "http://snomed.info/sct",
                                    "code": "73211009",
                                    "display": "Diabetes mellitus",
                                }
                            ]
                        },
                        "clinicalStatus": {"coding": [{"code": "active"}]},
                    }
                }
            ],
        }
        result = ingest_bundle(bundle)
        assert len(result.normalized_conditions) == 1
        assert result.normalized_conditions[0].display_text == "Diabetes mellitus"

    def test_observation_without_value_quantity(self) -> None:
        bundle = {
            "resourceType": "Bundle",
            "type": "collection",
            "entry": [
                {
                    "resource": {
                        "resourceType": "Observation",
                        "id": "obs-1",
                        "subject": {"reference": "Patient/p1"},
                        "status": "final",
                        "code": {
                            "coding": [
                                {
                                    "system": "http://loinc.org",
                                    "code": "85354-9",
                                    "display": "Blood pressure target",
                                }
                            ]
                        },
                        # No valueQuantity — observation has a note instead
                        "note": [{"text": "Target <130/80 mmHg"}],
                    }
                }
            ],
        }
        result = ingest_bundle(bundle)
        assert len(result.observations) == 1
        obs = result.observations[0]
        assert obs.value is None
        assert obs.unit is None
        assert obs.loinc_code == "85354-9"

    def test_allergy_without_criticality(self) -> None:
        bundle = {
            "resourceType": "Bundle",
            "type": "collection",
            "entry": [
                {
                    "resource": {
                        "resourceType": "AllergyIntolerance",
                        "id": "allergy-1",
                        "patient": {"reference": "Patient/p1"},
                        "code": {
                            "text": "Latex allergy",
                            "coding": [],
                        },
                        # No criticality field
                    }
                }
            ],
        }
        result = ingest_bundle(bundle)
        assert len(result.allergies) == 1
        assert result.allergies[0].criticality is None

    def test_medication_without_information_source(self) -> None:
        bundle = {
            "resourceType": "Bundle",
            "type": "collection",
            "entry": [
                {
                    "resource": {
                        "resourceType": "MedicationStatement",
                        "id": "med-1",
                        "subject": {"reference": "Patient/p1"},
                        "status": "active",
                        "medicationCodeableConcept": {
                            "text": "Aspirin 81mg",
                            "coding": [
                                {
                                    "system": "http://www.nlm.nih.gov/research/umls/rxnorm",
                                    "code": "1191",
                                    "display": "aspirin",
                                }
                            ],
                        },
                        # No informationSource
                    }
                }
            ],
        }
        result = ingest_bundle(bundle)
        assert len(result.medications) == 1
        assert result.medications[0].prescriber_reference is None

    def test_empty_bundle_returns_defaults(self) -> None:
        bundle = {
            "resourceType": "Bundle",
            "type": "collection",
            "entry": [],
        }
        result = ingest_bundle(bundle)
        assert result.patient_id == "unknown-patient"
        assert result.normalized_medications == []
        assert result.normalized_conditions == []
        assert result.allergies == []
        assert result.observations == []
        assert result.practitioner_npis == []
        assert result.phi_redactions == 0
        assert len(result.bundle_sha256) == 64

    def test_json_string_bundle_accepted(self, sarah_bundle: dict) -> None:
        json_str = json.dumps(sarah_bundle)
        result = ingest_bundle(json_str)
        assert result.patient_id == "sarah-mitchell"
        assert len(result.normalized_conditions) == 4
