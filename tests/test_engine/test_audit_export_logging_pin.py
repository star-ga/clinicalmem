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
  - audit_signature_invalid (WARNING, iter-118)        — bytes parse OK,
                                                          crypto verify fails
  - audit_signature_verification_exception (WARNING, iter-118) — malformed
                                                          payload / key shape
  - audit_export_npi_invalid (WARNING, iter-118)       — pre-raise on bad NPI
  - audit_verify_pubkey_setup_failed (WARNING, iter-118) — pubkey b64
                                                            decode failure
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


def test_logger_call_floor():
    """audit_export_part11.py logger-call floor (iter-118 ratchet 6 -> 10).

    iter-118 closed three silent paths:
      _verify_signature InvalidSignature exc -> WARNING (with error_type)
      _verify_signature general Exception   -> WARNING (with error_type)
      _assert_npi_valid pre-raise            -> WARNING (with public NPI)
      verify_audit_trail pubkey setup        -> WARNING (with error_type)

    Future audits may add more (would raise this floor). A regression
    that drops below 10 means a structured event got silently removed.
    """
    import re
    src = (
        Path(__file__).resolve().parent.parent.parent
        / "engine"
        / "audit_export_part11.py"
    ).read_text()
    calls = re.findall(r"\blogger\.(debug|info|warning|error|critical)\(", src)
    assert len(calls) >= 10, (
        f"engine/audit_export_part11.py logger-call count regressed: "
        f"{len(calls)} < floor 10. Some structured event was silently "
        f"removed."
    )


def test_invalid_npi_emits_warning_before_raise(caplog, sample_events, tmp_path):
    """Bad NPI submission must emit a structured WARNING before raising.

    iter-118 added the pre-raise log so operators can spot bad-NPI
    submissions before the ValueError propagates. NPI is public CMS
    registry data — safe to log alongside the failure reason.
    """
    caplog.set_level(logging.DEBUG, logger="engine.audit_export_part11")
    bad_npi = "0000000000"  # all zeros — fails CMS Luhn check
    out = tmp_path / "trail.json"
    with pytest.raises(ValueError, match="failed CMS Luhn"):
        export_audit_trail(sample_events, out, bad_npi)
    rec = _record(caplog, "audit_export_npi_invalid")
    assert rec is not None, (
        f"expected 'audit_export_npi_invalid' WARNING log; got "
        f"{[r.message for r in caplog.records]}"
    )
    assert rec.levelno == logging.WARNING
    assert getattr(rec, "npi", None) == bad_npi
    assert getattr(rec, "reason", None) == "cms_luhn_validation_failed"


def test_invalid_signature_emits_warning_with_error_type_only(caplog, sample_events, tmp_path):
    """Tampered signature must emit a structured WARNING; PHI-safe metadata only.

    iter-118 closed the silent `except (InvalidSignature, Exception): return False`
    path in `_verify_signature`. Verification failures must surface as WARNING
    with `error_type` only — never the raw signature bytes (attacker-
    controlled payload would otherwise echo into the log).
    """
    caplog.set_level(logging.DEBUG, logger="engine.audit_export_part11")
    out = tmp_path / "trail.json"
    export_audit_trail(sample_events, out, _VALID_NPI)

    # Tamper the signature to force InvalidSignature.
    document = json.loads(out.read_text())
    # Replace the attestation signature with a different valid-shape b64
    # (Ed25519 signatures are 64 bytes / ~88 b64url chars).
    sentinel = "ZZZ_TAMPERED_SIG_PAYLOAD_LEAK_iter118"
    document["attestation"]["signature"] = (
        "AAECAwQFBgcICQoLDA0ODxAREhMUFRYXGBkaGxwdHh8gISIjJCUmJygpKissLS4vMA"
        "ECAwQFBgcICQoLDA"
    )
    out.write_text(json.dumps(document, indent=2))

    caplog.clear()
    result = verify_audit_trail(out)
    assert not result.is_valid

    sig_record = _record(caplog, "audit_signature_invalid")
    assert sig_record is not None, (
        f"expected 'audit_signature_invalid' WARNING; got "
        f"{[r.message for r in caplog.records]}"
    )
    assert sig_record.levelno == logging.WARNING
    assert isinstance(sig_record.error_type, str)
    assert " " not in sig_record.error_type  # short identifier, no message body

    # No sentinel leak across any record.
    for rec in caplog.records:
        for value in vars(rec).values():
            assert sentinel not in repr(value), (
                f"sentinel leaked into log {rec.message}"
            )
