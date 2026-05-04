"""Pin: structured logging in `engine.audit_export_part11` is PHI-safe.

The 21 CFR Part 11 export module is the FDA-grade audit-trail export.
Every event in the input may carry actor_id (could be PHI), action
text (clinical narrative), and resource_ref (FHIR resource path).
Logging discipline:

  - Log: event_count, attestation_signer_npi (public NPI registry),
    output_path, chain_root_hash[:16], anomaly_count, signature_valid.
  - NEVER log: actor_id, action text, resource_ref, raw exception
    messages from JSON parse failures (which may quote file content).

Pinned event surface:
  - audit_export_part11_start (INFO)
  - audit_export_part11_complete (INFO)
  - audit_verify_start (DEBUG)
  - audit_verify_read_failure (WARNING, error_type only)
  - audit_verify_complete_clean (INFO)
  - audit_verify_tampering_detected (WARNING)
"""
import json
import logging
from pathlib import Path

import pytest

from engine.audit_export_part11 import (
    AuditEvent,
    export_audit_trail,
    verify_audit_trail,
)


_VALID_NPI = "1932159530"  # the long-published clinical-validation NPI
_SECRET_ACTOR = "ZZZ_SECRET_ACTOR_TOKEN"
_SECRET_ACTION = "ZZZ_SECRET_ACTION_NARRATIVE"


def _record(caplog, event_name):
    for r in caplog.records:
        if r.name == "engine.audit_export_part11" and r.message == event_name:
            return r
    return None


@pytest.fixture
def sample_events():
    return [
        AuditEvent(
            timestamp_iso="2026-05-01T10:00:00+00:00",
            event_type="DECISION",
            actor_id=_SECRET_ACTOR,
            action=_SECRET_ACTION,
            resource_ref="Patient/test-pt",
            before_hash="0" * 64,
            after_hash="0" * 64,
            plan_hash="abc" + "0" * 61,
        ),
        AuditEvent(
            timestamp_iso="2026-05-01T10:00:01+00:00",
            event_type="OBSERVATION",
            actor_id=_SECRET_ACTOR,
            action=_SECRET_ACTION,
            resource_ref="Patient/test-pt",
            before_hash="1" * 64,
            after_hash="1" * 64,
            plan_hash="def" + "0" * 61,
        ),
    ]


def test_export_emits_start_log(caplog, sample_events, tmp_path):
    caplog.set_level(logging.DEBUG, logger="engine.audit_export_part11")
    out = tmp_path / "trail.json"
    export_audit_trail(sample_events, out, _VALID_NPI)
    rec = _record(caplog, "audit_export_part11_start")
    assert rec is not None
    assert rec.event_count == 2
    assert rec.attestation_signer_npi == _VALID_NPI


def test_export_emits_complete_log(caplog, sample_events, tmp_path):
    caplog.set_level(logging.DEBUG, logger="engine.audit_export_part11")
    out = tmp_path / "trail.json"
    result = export_audit_trail(sample_events, out, _VALID_NPI)
    rec = _record(caplog, "audit_export_part11_complete")
    assert rec is not None
    assert rec.event_count == 2
    assert rec.chain_root_prefix == result.chain_root_hash[:16]


def test_verify_clean_emits_info(caplog, sample_events, tmp_path):
    caplog.set_level(logging.DEBUG, logger="engine.audit_export_part11")
    out = tmp_path / "trail.json"
    export_audit_trail(sample_events, out, _VALID_NPI)
    caplog.clear()
    result = verify_audit_trail(out)
    assert result.is_valid
    rec = _record(caplog, "audit_verify_complete_clean")
    assert rec is not None
    assert rec.levelno == logging.INFO


def test_verify_tampering_emits_warning(caplog, sample_events, tmp_path):
    caplog.set_level(logging.DEBUG, logger="engine.audit_export_part11")
    out = tmp_path / "trail.json"
    export_audit_trail(sample_events, out, _VALID_NPI)
    # Tamper with the file
    document = json.loads(out.read_text())
    document["audit_events"][0]["action"] = "TAMPERED"
    out.write_text(json.dumps(document, indent=2))
    caplog.clear()
    result = verify_audit_trail(out)
    assert not result.is_valid
    rec = _record(caplog, "audit_verify_tampering_detected")
    assert rec is not None
    assert rec.levelno == logging.WARNING
    assert rec.anomaly_count > 0


def test_verify_read_failure_emits_warning_with_error_type(caplog, tmp_path):
    caplog.set_level(logging.DEBUG, logger="engine.audit_export_part11")
    bad_path = tmp_path / "nonexistent.json"
    result = verify_audit_trail(bad_path)
    assert not result.is_valid
    rec = _record(caplog, "audit_verify_read_failure")
    assert rec is not None
    assert rec.levelno == logging.WARNING
    # error_type should be a name like "FileNotFoundError"; ensure it's a
    # short identifier, not a full exception message body.
    assert isinstance(rec.error_type, str)
    assert " " not in rec.error_type  # no spaces → no message body


def test_export_logs_never_emit_secret_actor_or_action(caplog, sample_events, tmp_path):
    caplog.set_level(logging.DEBUG, logger="engine.audit_export_part11")
    out = tmp_path / "trail.json"
    export_audit_trail(sample_events, out, _VALID_NPI)
    verify_audit_trail(out)
    for rec in caplog.records:
        if rec.name != "engine.audit_export_part11":
            continue
        for value in vars(rec).values():
            text = repr(value)
            assert _SECRET_ACTOR not in text, (
                f"actor_id leaked into log {rec.message}"
            )
            assert _SECRET_ACTION not in text, (
                f"action narrative leaked into log {rec.message}"
            )
