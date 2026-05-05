# Copyright 2026 STARGA Inc. — Apache-2.0
"""Pin: structured logging in `engine.audit_export_part11` is PHI-safe.

Iter 163 T4 round-32 logger ratchet on `engine/audit_export_part11.py`
— the next-lowest manageable density file in the engine after
iter-160 closed `llm_synthesizer.py` (15.8 → 23.1/kloc).

Pre-iter-163 audit:
  * 609 LOC, 10 logger calls (16.4/kloc) — silent paths through the
    Merkle-construction loop, demo-keypair slot, attestation-signing,
    file serialisation, and the verifier `@context` happy-path.
  * Demo-keypair slot is REGULATION-CRITICAL — production audits
    must verify the demo path is NOT firing in prod (HSM keys load
    via the production slot).

Iter-163 ratchet adds 5 PHI-safe DEBUG events:

  * audit_export_demo_keypair_generated (slot guard for prod audits)
  * audit_export_chain_built (Merkle construction summary)
  * audit_export_attestation_signed (Ed25519 signature lifecycle)
  * audit_export_serialised (file-write metrics)
  * audit_verify_context_match (verifier early-step success)

PHI-safety contract:
  * Cryptographic hashes (chain seed/root, public-key fingerprint) are
    one-way SHA-256 outputs — safe to log as 16-hex prefixes.
  * NPI is public CMS-registry data — safe to log.
  * Event/byte counts are cardinality signals — safe to log.
  * Algorithm names (Ed25519, SHA-256) are constants — safe to log.
  * NEVER: actor_id, action text, resource_ref, before/after hashes
    of memory blocks, signed bytes, or any audit-event narrative.
"""
from __future__ import annotations

import logging
import re
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from engine.audit_export_part11 import (  # noqa: E402
    AuditEvent,
    export_audit_trail,
    verify_audit_trail,
)

_VALID_NPI = "1234567893"
_EMPTY_SHA = "0" * 64

_PHI_SENTINEL_ACTOR = "ZZZ_PHI_LEAK_ACTOR_ID"
_PHI_SENTINEL_ACTION = "ZZZ_PHI_LEAK_ACTION_TEXT"
_PHI_SENTINEL_RESOURCE = "ZZZ_PHI_LEAK_RESOURCE_REF"


def _make_events_with_phi_sentinels(n: int = 3) -> list[AuditEvent]:
    return [
        AuditEvent(
            timestamp_iso=f"2026-05-0{i + 1}T10:00:00Z",
            event_type="MEMORY_WRITE",
            actor_id=f"{_PHI_SENTINEL_ACTOR}-{i}",
            action=f"{_PHI_SENTINEL_ACTION}-{i}",
            resource_ref=f"{_PHI_SENTINEL_RESOURCE}-{i}",
            before_hash=_EMPTY_SHA,
            after_hash=_EMPTY_SHA,
            plan_hash=_EMPTY_SHA,
        )
        for i in range(n)
    ]


def _record(caplog, event_name):
    for r in caplog.records:
        if r.name == "engine.audit_export_part11" and r.message == event_name:
            return r
    return None


# ── new-event coverage tests ──────────────────────────────────────────────────

class TestAuditExportPart11LoggerRatchetIter163:
    """Pin the 5 new structured DEBUG events added in iter-163."""

    def test_demo_keypair_generation_emits_debug_event(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Production audit slot: a DEBUG event MUST fire every time
        the demo keypair generator runs. If this event is observed in
        a production environment it indicates HSM-backed keys are NOT
        being used and the export is signed by an ephemeral key."""
        caplog.set_level(logging.DEBUG, logger="engine.audit_export_part11")
        events = _make_events_with_phi_sentinels(2)
        export_audit_trail(events, tmp_path / "a.jsonld", _VALID_NPI)

        rec = _record(caplog, "audit_export_demo_keypair_generated")
        assert rec is not None, "demo keypair production-slot event missing"
        assert rec.levelno == logging.DEBUG
        assert isinstance(rec.public_key_fp, str)
        assert len(rec.public_key_fp) == 16
        assert re.match(r"^[0-9a-f]{16}$", rec.public_key_fp)
        assert rec.production_slot == "_generate_demo_keypair"

    def test_chain_built_emits_debug_with_event_count(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """After Merkle construction completes, a DEBUG event captures
        chain_seed_prefix + chain_root_prefix + event_count for ops
        correlation against the runtime audit chain."""
        caplog.set_level(logging.DEBUG, logger="engine.audit_export_part11")
        events = _make_events_with_phi_sentinels(4)
        result = export_audit_trail(events, tmp_path / "a.jsonld", _VALID_NPI)

        rec = _record(caplog, "audit_export_chain_built")
        assert rec is not None
        assert rec.levelno == logging.DEBUG
        assert rec.event_count == 4
        assert len(rec.chain_seed_prefix) == 16
        assert len(rec.chain_root_prefix) == 16
        assert result.chain_root_hash.startswith(rec.chain_root_prefix)

    def test_attestation_signed_emits_debug_with_payload_size(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Signature lifecycle DEBUG: payload bytes + signature length
        only. NEVER the signed bytes or the signature itself."""
        caplog.set_level(logging.DEBUG, logger="engine.audit_export_part11")
        events = _make_events_with_phi_sentinels(2)
        export_audit_trail(events, tmp_path / "a.jsonld", _VALID_NPI)

        rec = _record(caplog, "audit_export_attestation_signed")
        assert rec is not None
        assert rec.levelno == logging.DEBUG
        assert isinstance(rec.signed_payload_bytes, int)
        assert rec.signed_payload_bytes > 0
        assert rec.signature_b64url_chars > 0
        assert rec.algorithm == "Ed25519"

    def test_serialised_emits_debug_with_total_bytes(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Serialisation DEBUG: total bytes + format choices. Cardinal
        signal for ops to spot drift in audit-document size."""
        caplog.set_level(logging.DEBUG, logger="engine.audit_export_part11")
        events = _make_events_with_phi_sentinels(3)
        out = tmp_path / "a.jsonld"
        result = export_audit_trail(events, out, _VALID_NPI)

        rec = _record(caplog, "audit_export_serialised")
        assert rec is not None
        assert rec.levelno == logging.DEBUG
        assert rec.total_bytes == result.total_bytes
        assert rec.indent_spaces == 2
        assert rec.ensure_ascii is True

    def test_verify_context_match_emits_debug_on_happy_path(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Verifier early-step DEBUG: @context match fires before the
        chain replay so audit-replay debugging can tell where a
        verification ran vs where it failed early."""
        caplog.set_level(logging.DEBUG, logger="engine.audit_export_part11")
        events = _make_events_with_phi_sentinels(2)
        out = tmp_path / "a.jsonld"
        export_audit_trail(events, out, _VALID_NPI)

        caplog.clear()
        result = verify_audit_trail(out)
        assert result.is_valid

        rec = _record(caplog, "audit_verify_context_match")
        assert rec is not None
        assert rec.levelno == logging.DEBUG
        assert rec.path == str(out)


# ── density floor + PHI-paranoia scan ────────────────────────────────────────

_LOGGER_PATTERN = re.compile(
    r"\blogger\.(?:debug|info|warning|error|exception|critical)\("
)
_LOGGER_FLOOR = 15  # iter-163 baseline (10 → 15)


def test_audit_export_part11_logger_floor_iter_163() -> None:
    """Iter-163 floor: 15 direct logger calls. Future regressions can't
    quietly slip below without firing this gate."""
    src = (_REPO_ROOT / "engine" / "audit_export_part11.py").read_text()
    count = len(_LOGGER_PATTERN.findall(src))
    assert count >= _LOGGER_FLOOR, (
        f"engine/audit_export_part11.py has {count} direct logger calls; "
        f"floor is {_LOGGER_FLOOR} (iter-163 baseline)."
    )


def test_audit_export_part11_logger_density_iter_163() -> None:
    """Logger density floor: post-iter-163 must stay >= 15/kloc.

    Density = 1000 * (logger_calls / non-blank-non-comment LOC).
    """
    src_path = _REPO_ROOT / "engine" / "audit_export_part11.py"
    raw = src_path.read_text().splitlines()
    code_lines = [
        ln for ln in raw
        if ln.strip() and not ln.strip().startswith("#")
    ]
    loc = max(len(code_lines), 1)
    count = len(_LOGGER_PATTERN.findall(src_path.read_text()))
    density_per_kloc = (1000.0 * count) / loc
    assert density_per_kloc >= 15.0, (
        f"engine/audit_export_part11.py logger density "
        f"{density_per_kloc:.1f}/kloc < floor 15/kloc "
        f"({count} calls / {loc} LOC)."
    )


def test_no_phi_sentinels_in_log_records(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """End-to-end paranoia: run a full export+verify with PHI sentinels
    in actor_id / action / resource_ref. NO log record may contain any
    sentinel — neither in `message` nor in any structured `extra` field.
    """
    caplog.set_level(logging.DEBUG, logger="engine.audit_export_part11")
    events = _make_events_with_phi_sentinels(3)
    out = tmp_path / "a.jsonld"
    export_audit_trail(events, out, _VALID_NPI)
    verify_audit_trail(out)

    sentinels = (
        _PHI_SENTINEL_ACTOR,
        _PHI_SENTINEL_ACTION,
        _PHI_SENTINEL_RESOURCE,
    )
    for rec in caplog.records:
        if rec.name != "engine.audit_export_part11":
            continue
        for value in vars(rec).values():
            for sentinel in sentinels:
                assert sentinel not in repr(value), (
                    f"PHI sentinel {sentinel!r} leaked into log record "
                    f"{rec.message!r} via field "
                    f"{[k for k, v in vars(rec).items() if sentinel in repr(v)]}"
                )
