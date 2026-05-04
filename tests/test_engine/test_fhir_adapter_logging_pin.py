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

Pinned event surface (10 events post-iter-108):
  - fhir_bundle_ingest_start (DEBUG)
  - fhir_bundle_unknown_resource_type_skipped (DEBUG, tolerant mode)
  - fhir_bundle_unknown_resource_type_rejected (WARNING, strict mode)
  - fhir_bundle_parse_failure (WARNING, error TYPE only)
  - fhir_bundle_ingest_complete (INFO when no rejections / WARNING otherwise)
  - fhir_bundle_invalid_json (ERROR)             — iter 108
  - fhir_bundle_wrong_python_type (ERROR)        — iter 108
  - fhir_bundle_wrong_resource_type (ERROR)      — iter 108
  - fhir_bundle_entry_not_array (ERROR)          — iter 108
  - fhir_bundle_unsupported_type (ERROR)         — iter 108
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


# ─── Iter 108: 5 silent pre-raise validation paths now logged ──────────────


def test_invalid_json_logs_decode_msg_not_raw_input(caplog):
    """A malformed JSON bundle string logs `fhir_bundle_invalid_json`
    with the canned decode_msg only. The raw bundle string (which can
    contain attacker-supplied content) must NOT appear in the log."""
    caplog.set_level(logging.ERROR, logger="engine.fhir_adapter")
    sentinel = "ZZZ_RAW_BUNDLE_BODY_LEAK_TOKEN"
    bad_json = '{"resourceType": "Bundle", "type": "collection",' + sentinel
    with pytest.raises(ValueError):
        ingest_bundle(bad_json)

    rec = _record(caplog, "fhir_bundle_invalid_json")
    assert rec is not None, "fhir_bundle_invalid_json event missing"
    assert rec.error_type == "JSONDecodeError"
    assert isinstance(rec.decode_msg, str)
    assert isinstance(rec.input_size_bytes, int)
    for value in vars(rec).values():
        assert sentinel not in repr(value), (
            f"fhir_bundle_invalid_json leaked raw bundle content"
        )


def test_wrong_resource_type_logs_length_only(caplog):
    """A bundle with a non-'Bundle' resourceType logs `fhir_bundle_
    wrong_resource_type` with the LENGTH of the bad string, never
    the value."""
    caplog.set_level(logging.ERROR, logger="engine.fhir_adapter")
    sentinel_rt = "ZZZ_FAKE_RESOURCE_TYPE_PAYLOAD_LEAK"
    bundle = {"resourceType": sentinel_rt, "type": "collection", "entry": []}
    with pytest.raises(ValueError):
        ingest_bundle(bundle)

    rec = _record(caplog, "fhir_bundle_wrong_resource_type")
    assert rec is not None, "fhir_bundle_wrong_resource_type event missing"
    assert rec.resource_type_len == len(sentinel_rt)
    assert rec.resource_type_class == "str"
    for value in vars(rec).values():
        assert sentinel_rt not in repr(value), (
            f"fhir_bundle_wrong_resource_type leaked raw resourceType"
        )


def test_unsupported_bundle_type_logs_length_only(caplog):
    """An unsupported Bundle.type logs `fhir_bundle_unsupported_type`
    with the LENGTH of the bad value, never the value itself."""
    caplog.set_level(logging.ERROR, logger="engine.fhir_adapter")
    sentinel_type = "ZZZ_MALICIOUS_BUNDLE_TYPE_LEAK"
    bundle = {"resourceType": "Bundle", "type": sentinel_type, "entry": []}
    with pytest.raises(ValueError):
        ingest_bundle(bundle)

    rec = _record(caplog, "fhir_bundle_unsupported_type")
    assert rec is not None, "fhir_bundle_unsupported_type event missing"
    assert rec.bundle_type_len == len(sentinel_type)
    for value in vars(rec).values():
        assert sentinel_type not in repr(value), (
            f"fhir_bundle_unsupported_type leaked raw bundle type"
        )


def test_fhir_adapter_has_at_least_ten_structured_logs():
    """Iter-108 floor: fhir_adapter.py grew from 5 to 10 logger
    calls. Future evidence-chain regressions can't quietly slip
    below the floor."""
    import re
    from pathlib import Path
    src = (Path(__file__).resolve().parent.parent.parent / "engine" / "fhir_adapter.py").read_text()
    pat = re.compile(
        r"\blogger\.(?:debug|info|warning|error|exception|critical)\("
    )
    count = len(pat.findall(src))
    assert count >= 10, (
        f"engine/fhir_adapter.py has {count} logger calls; "
        f"floor is 10 (iter-108 baseline)."
    )
