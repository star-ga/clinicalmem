"""Pin: structured logging in `engine.fhir_adapter` is PHI-safe.

The FHIR R4 ingress is the entry boundary where raw EHR bundles cross
the PHI line. Any log call here can leak patient narrative if not
disciplined. Pinned discipline:

  - Log resourceType names (FHIR R4 standard, public configuration).
  - Log counts (entry_count, medication_count, etc.).
  - Log hash prefixes (bundle_sha256[:16]) — non-reversible by design.
  - Log internal patient_id (post-PHI-scrub identifier).
  - NEVER log: medication display text, condition codes, observation
    values, allergy text, raw exception messages from parsing.

Pinned event surface:
  - fhir_bundle_ingest_start (DEBUG)
  - fhir_bundle_unknown_resource_type_skipped (DEBUG, tolerant mode)
  - fhir_bundle_unknown_resource_type_rejected (WARNING, strict mode)
  - fhir_bundle_parse_failure (WARNING, error TYPE only)
  - fhir_bundle_ingest_complete (INFO when no rejections / WARNING otherwise)
"""
import logging

import pytest

from engine.fhir_adapter import ingest_bundle


def _record(caplog, event_name):
    for r in caplog.records:
        if r.name == "engine.fhir_adapter" and r.message == event_name:
            return r
    return None


@pytest.fixture
def minimal_bundle():
    return {
        "resourceType": "Bundle",
        "type": "collection",
        "entry": [
            {
                "resource": {
                    "resourceType": "Patient",
                    "id": "test-pt-01",
                    "name": [{"family": "TestSurname", "given": ["Test"]}],
                    "gender": "female",
                    "birthDate": "1980-01-01",
                },
            },
            {
                "resource": {
                    "resourceType": "MedicationStatement",
                    "subject": {"reference": "Patient/test-pt-01"},
                    "status": "active",
                    "medicationCodeableConcept": {
                        "coding": [{
                            "system": "http://www.nlm.nih.gov/research/umls/rxnorm",
                            "code": "10324",
                            "display": "SECRET_TEST_DRUG_TOKEN",
                        }],
                    },
                },
            },
        ],
    }


def test_ingest_emits_start_log(caplog, minimal_bundle):
    caplog.set_level(logging.DEBUG, logger="engine.fhir_adapter")
    ingest_bundle(minimal_bundle)
    rec = _record(caplog, "fhir_bundle_ingest_start")
    assert rec is not None
    assert rec.bundle_type == "collection"
    assert rec.entry_count == 2
    assert isinstance(rec.bundle_sha256_prefix, str)
    assert len(rec.bundle_sha256_prefix) == 16


def test_ingest_emits_complete_log(caplog, minimal_bundle):
    caplog.set_level(logging.DEBUG, logger="engine.fhir_adapter")
    ingest_bundle(minimal_bundle)
    rec = _record(caplog, "fhir_bundle_ingest_complete")
    assert rec is not None
    assert rec.levelno == logging.INFO  # no rejections in this bundle
    assert rec.medication_count == 1
    assert rec.condition_count == 0
    assert rec.allergy_count == 0


def test_ingest_unknown_resource_type_warning_in_strict(caplog):
    caplog.set_level(logging.DEBUG, logger="engine.fhir_adapter")
    bundle = {
        "resourceType": "Bundle",
        "type": "collection",
        "entry": [{"resource": {"resourceType": "Bogus", "id": "x"}}],
    }
    ingest_bundle(bundle, strict_resource_types=True)
    rec = _record(caplog, "fhir_bundle_unknown_resource_type_rejected")
    assert rec is not None
    assert rec.levelno == logging.WARNING
    assert rec.resource_type == "Bogus"
    assert rec.strict is True


def test_ingest_unknown_resource_type_debug_in_tolerant(caplog):
    caplog.set_level(logging.DEBUG, logger="engine.fhir_adapter")
    bundle = {
        "resourceType": "Bundle",
        "type": "collection",
        "entry": [{"resource": {"resourceType": "Bogus", "id": "x"}}],
    }
    ingest_bundle(bundle)  # default tolerant
    rec = _record(caplog, "fhir_bundle_unknown_resource_type_skipped")
    assert rec is not None
    assert rec.levelno == logging.DEBUG
    assert rec.resource_type == "Bogus"
    assert rec.strict is False


def test_ingest_logs_never_emit_secret_drug_token(caplog, minimal_bundle):
    """PHI scrub: a sentinel drug-display string passed in must NEVER
    appear in any engine.fhir_adapter log record.
    """
    caplog.set_level(logging.DEBUG, logger="engine.fhir_adapter")
    ingest_bundle(minimal_bundle)
    for rec in caplog.records:
        if rec.name != "engine.fhir_adapter":
            continue
        for value in vars(rec).values():
            text = repr(value)
            assert "SECRET_TEST_DRUG_TOKEN" not in text, (
                f"medication display name leaked into log {rec.message}"
            )
            assert "TestSurname" not in text, (
                f"patient family name leaked into log {rec.message}"
            )
