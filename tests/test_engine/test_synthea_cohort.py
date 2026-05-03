"""Tests for the Synthea 10-patient synthetic FHIR R4 demo cohort.

Validates:
- Bundle is structurally valid FHIR R4
- At least 10 patients present
- Each patient has at least one Condition
- All NPIs pass the CMS Luhn algorithm
- No Patient resource contains a real-looking SSN or MRN from a real system
- Each patient demonstrates a distinct conflict type (via _conflict_type annotation)
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pytest

from engine.npi_registry import validate_npi

_COHORT_JSON = Path(__file__).resolve().parents[2] / "docs" / "synthea_demo_cohort.json"

# SSN patterns (US): 9 digits with optional separators, or explicit XXX-XX-XXXX format.
_SSN_RE = re.compile(r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b")
# Real MRN patterns: we disallow anything that looks like a real EHR MRN
# (our synthetic ones are prefixed SYN-MRN-).
_REAL_MRN_SYSTEMS = {
    "http://hl7.org/fhir/sid/us-ssn",
    "urn:oid:2.16.840.1.113883.4.1",  # SSN OID
}


def _load_bundle() -> dict:
    with open(_COHORT_JSON, encoding="utf-8") as fh:
        return json.load(fh)


def _all_resources(bundle: dict) -> list[dict]:
    return [e["resource"] for e in bundle.get("entry", []) if "resource" in e]


def _resources_of_type(bundle: dict, rtype: str) -> list[dict]:
    return [r for r in _all_resources(bundle) if r.get("resourceType") == rtype]


# ---------------------------------------------------------------------------
# Basic FHIR R4 structure
# ---------------------------------------------------------------------------


class TestBundleIsValidFhirR4:
    def test_bundle_resource_type(self):
        bundle = _load_bundle()
        assert bundle.get("resourceType") == "Bundle"

    def test_bundle_type_is_collection(self):
        bundle = _load_bundle()
        assert bundle.get("type") == "collection"

    def test_bundle_has_id(self):
        bundle = _load_bundle()
        assert bundle.get("id") == "synthea-clinicalmem-demo-cohort"

    def test_bundle_has_meta_synthetic_flag(self):
        bundle = _load_bundle()
        meta = bundle.get("meta", {})
        assert "_synthetic" in meta, "meta._synthetic field missing"

    def test_bundle_has_entry_list(self):
        bundle = _load_bundle()
        assert isinstance(bundle.get("entry"), list)
        assert len(bundle["entry"]) > 0

    def test_all_resources_have_resource_type(self):
        bundle = _load_bundle()
        for entry in bundle["entry"]:
            r = entry.get("resource", {})
            assert r.get("resourceType"), f"Entry missing resourceType: {entry}"

    def test_all_resources_have_id(self):
        bundle = _load_bundle()
        for entry in bundle["entry"]:
            r = entry.get("resource", {})
            assert r.get("id"), f"Resource missing id: {r.get('resourceType')}"

    def test_practitioners_have_npi_identifier(self):
        bundle = _load_bundle()
        practitioners = _resources_of_type(bundle, "Practitioner")
        assert len(practitioners) >= 10, f"Expected >=10 practitioners, got {len(practitioners)}"
        for p in practitioners:
            identifiers = p.get("identifier", [])
            npi_ids = [
                i for i in identifiers
                if i.get("system") == "http://hl7.org/fhir/sid/us-npi"
            ]
            assert npi_ids, f"Practitioner {p['id']} missing NPI identifier"

    def test_medication_statements_have_rxnorm_codes(self):
        bundle = _load_bundle()
        meds = _resources_of_type(bundle, "MedicationStatement")
        assert len(meds) >= 10
        for med in meds:
            codings = (
                med.get("medicationCodeableConcept", {})
                   .get("coding", [])
            )
            rxnorm = [c for c in codings if c.get("system") == "http://www.nlm.nih.gov/research/umls/rxnorm"]
            assert rxnorm, f"MedicationStatement {med['id']} missing RxNorm code"

    def test_conditions_have_snomed_codes(self):
        bundle = _load_bundle()
        conditions = _resources_of_type(bundle, "Condition")
        for cond in conditions:
            codings = cond.get("code", {}).get("coding", [])
            snomed = [c for c in codings if c.get("system") == "http://snomed.info/sct"]
            assert snomed, f"Condition {cond['id']} missing SNOMED CT code"

    def test_observations_have_loinc_codes(self):
        bundle = _load_bundle()
        obs = _resources_of_type(bundle, "Observation")
        for o in obs:
            codings = o.get("code", {}).get("coding", [])
            loinc = [c for c in codings if c.get("system") == "http://loinc.org"]
            assert loinc, f"Observation {o['id']} missing LOINC code"


# ---------------------------------------------------------------------------
# Patient count and per-patient requirements
# ---------------------------------------------------------------------------


class TestAtLeast10PatientsPresent:
    def test_patient_count_at_least_10(self):
        bundle = _load_bundle()
        patients = _resources_of_type(bundle, "Patient")
        assert len(patients) >= 10, f"Expected >=10 patients, got {len(patients)}"

    def test_each_patient_has_gender_and_birthdate(self):
        bundle = _load_bundle()
        for pt in _resources_of_type(bundle, "Patient"):
            assert pt.get("gender") in ("male", "female", "other", "unknown"), \
                f"Patient {pt['id']} missing or invalid gender"
            assert pt.get("birthDate"), f"Patient {pt['id']} missing birthDate"

    def test_each_patient_has_us_address(self):
        bundle = _load_bundle()
        for pt in _resources_of_type(bundle, "Patient"):
            addresses = pt.get("address", [])
            assert addresses, f"Patient {pt['id']} has no address"
            assert any(a.get("country") == "US" for a in addresses), \
                f"Patient {pt['id']} has no US address"


class TestEachPatientHasAtLeastOneCondition:
    def test_each_patient_has_condition(self):
        bundle = _load_bundle()
        patients = _resources_of_type(bundle, "Patient")
        conditions = _resources_of_type(bundle, "Condition")

        for pt in patients:
            pt_ref = f"Patient/{pt['id']}"
            pt_conditions = [
                c for c in conditions
                if c.get("subject", {}).get("reference") == pt_ref
            ]
            assert pt_conditions, \
                f"Patient {pt['id']} has no associated Condition resource"

    def test_each_patient_has_medication_statement(self):
        bundle = _load_bundle()
        patients = _resources_of_type(bundle, "Patient")
        meds = _resources_of_type(bundle, "MedicationStatement")

        for pt in patients:
            pt_ref = f"Patient/{pt['id']}"
            pt_meds = [
                m for m in meds
                if m.get("subject", {}).get("reference") == pt_ref
            ]
            assert pt_meds, \
                f"Patient {pt['id']} has no MedicationStatement"


# ---------------------------------------------------------------------------
# NPI Luhn validation
# ---------------------------------------------------------------------------


class TestAllNPIsPassLuhn:
    def test_all_practitioner_npis_are_luhn_valid(self):
        bundle = _load_bundle()
        practitioners = _resources_of_type(bundle, "Practitioner")
        failures = []
        for p in practitioners:
            for ident in p.get("identifier", []):
                if ident.get("system") == "http://hl7.org/fhir/sid/us-npi":
                    npi_val = ident.get("value", "")
                    if not validate_npi(npi_val):
                        failures.append((p["id"], npi_val))
        assert failures == [], f"NPIs failing Luhn check: {failures}"

    def test_all_practitioners_marked_demo_npi_source(self):
        bundle = _load_bundle()
        practitioners = _resources_of_type(bundle, "Practitioner")
        bad = []
        for p in practitioners:
            npi_source = p.get("meta", {}).get("npi_source")
            if npi_source != "DEMO_LUHN_GENERATED":
                bad.append((p["id"], npi_source))
        assert bad == [], \
            f"Practitioners not marked DEMO_LUHN_GENERATED: {bad}"


# ---------------------------------------------------------------------------
# PHI / SSN / real identifier absence
# ---------------------------------------------------------------------------


class TestNoPatientResourceContainsRealSSNOrMRN:
    def _check_no_ssn_in_text(self, text: str, context: str) -> None:
        matches = _SSN_RE.findall(text)
        # Filter out values that are clearly not SSNs (e.g. LOINC codes like 33914-8)
        # by requiring the match to be long enough and not contain letters.
        genuine = [m for m in matches if re.fullmatch(r"[\d\-\s]{9,11}", m.strip())]
        # Exclude known safe patterns: 33914-8, 8867-4, etc. (LOINC codes <= 6 chars)
        genuine = [m for m in genuine if len(m.replace("-", "").replace(" ", "")) == 9]
        assert not genuine, f"Possible SSN pattern in {context}: {genuine}"

    def test_no_ssn_in_patient_identifiers(self):
        bundle = _load_bundle()
        for pt in _resources_of_type(bundle, "Patient"):
            for ident in pt.get("identifier", []):
                system = ident.get("system", "")
                assert system not in _REAL_MRN_SYSTEMS, \
                    f"Patient {pt['id']} uses real SSN system OID: {system}"
                value = ident.get("value", "")
                self._check_no_ssn_in_text(value, f"Patient/{pt['id']} identifier")

    def test_no_real_mrn_system_oids(self):
        bundle = _load_bundle()
        for r in _all_resources(bundle):
            for ident in r.get("identifier", []):
                assert ident.get("system") not in _REAL_MRN_SYSTEMS, \
                    f"Resource {r.get('id')} uses real SSN/MRN system OID"

    def test_mrn_values_use_synthetic_prefix(self):
        bundle = _load_bundle()
        for pt in _resources_of_type(bundle, "Patient"):
            for ident in pt.get("identifier", []):
                if "mrn" in ident.get("system", "").lower():
                    val = ident.get("value", "")
                    assert val.startswith("SYN-MRN-"), \
                        f"Patient {pt['id']} MRN does not use SYN-MRN- prefix: {val}"

    def test_no_real_npi_registry_lookups_needed(self):
        """All practitioners must have npi_source = DEMO_LUHN_GENERATED
        so no system ever accidentally looks them up in real NPPES."""
        bundle = _load_bundle()
        practitioners = _resources_of_type(bundle, "Practitioner")
        for p in practitioners:
            source = p.get("meta", {}).get("npi_source", "")
            assert source == "DEMO_LUHN_GENERATED", \
                f"Practitioner {p['id']} has npi_source={source!r}"


# ---------------------------------------------------------------------------
# Distinct conflict types
# ---------------------------------------------------------------------------


class TestEachPatientDemonstratesDistinctClinicalConflict:
    def test_conflict_type_annotations_present(self):
        bundle = _load_bundle()
        conflict_entries = [
            e for e in bundle.get("entry", [])
            if "_conflict_type" in e
        ]
        assert len(conflict_entries) >= 10, \
            f"Expected >=10 _conflict_type annotations, got {len(conflict_entries)}"

    def test_conflict_types_are_distinct(self):
        bundle = _load_bundle()
        conflict_types = [
            e["_conflict_type"]
            for e in bundle.get("entry", [])
            if "_conflict_type" in e
        ]
        assert len(conflict_types) == len(set(conflict_types)), \
            f"Duplicate conflict types: {[t for t in conflict_types if conflict_types.count(t) > 1]}"

    def test_warfarin_nsaid_conflict_present(self):
        bundle = _load_bundle()
        types = {e.get("_conflict_type") for e in bundle["entry"]}
        assert "warfarin_plus_nsaid_bleeding" in types

    def test_penicillin_allergy_conflict_present(self):
        bundle = _load_bundle()
        types = {e.get("_conflict_type") for e in bundle["entry"]}
        assert "penicillin_allergy_amoxicillin_cross_reactivity" in types

    def test_metformin_gfr_conflict_present(self):
        bundle = _load_bundle()
        types = {e.get("_conflict_type") for e in bundle["entry"]}
        assert "declining_gfr_metformin_contraindication" in types

    def test_ssri_tramadol_conflict_present(self):
        bundle = _load_bundle()
        types = {e.get("_conflict_type") for e in bundle["entry"]}
        assert "ssri_tramadol_serotonin_syndrome" in types

    def test_beta_blocker_ccb_conflict_present(self):
        bundle = _load_bundle()
        types = {e.get("_conflict_type") for e in bundle["entry"]}
        assert "beta_blocker_non_dhp_ccb_bradycardia" in types

    def test_hyperkalemia_conflict_present(self):
        bundle = _load_bundle()
        types = {e.get("_conflict_type") for e in bundle["entry"]}
        assert "ace_inhibitor_potassium_sparing_diuretic_hyperkalemia" in types
