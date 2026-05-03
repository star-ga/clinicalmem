# Copyright 2026 STARGA Inc. — Apache-2.0
"""Tests for engine.audit_export_part11 — 21 CFR Part 11 audit trail export.

Coverage targets
----------------
* Round-trip: export then verify succeeds
* Tampered event breaks verification
* Invalid NPI rejected at export time
* Chronological ordering enforced regardless of input order
* JSON-LD @context is present and correct
* Chain root hash is SHA-256 of canonical encoding
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from engine.audit_export_part11 import (  # noqa: E402
    AuditEvent,
    ExportResult,
    VerificationResult,
    _JSONLD_CONTEXT,
    _CHAIN_SEED,
    _chain_hash,
    _event_canonical,
    export_audit_trail,
    verify_audit_trail,
)

# ── valid demo NPI (CMS spec example, passes Luhn) ───────────────────────────
_VALID_NPI = "1234567893"
# Invalid NPI — wrong check digit
_INVALID_NPI = "1234567890"

_EMPTY_SHA = "0" * 64


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_events(n: int = 3) -> list[AuditEvent]:
    """Generate n synthetic audit events in chronological order."""
    return [
        AuditEvent(
            timestamp_iso=f"2026-05-0{i + 1}T10:00:00Z",
            event_type="MEMORY_WRITE",
            actor_id=f"actor-{i}",
            action=f"Clinical action {i}",
            resource_ref=f"MedicationStatement/med-{i}",
            before_hash=_EMPTY_SHA,
            after_hash=_EMPTY_SHA,  # will be replaced during export
            plan_hash=_EMPTY_SHA,
        )
        for i in range(n)
    ]


# ── round trip ────────────────────────────────────────────────────────────────

class TestExportThenVerifyRoundTripSucceeds:
    """test_export_then_verify_round_trip_succeeds."""

    def test_basic_round_trip(self, tmp_path: Path) -> None:
        events = _make_events(5)
        out = tmp_path / "audit.jsonld"
        export_result = export_audit_trail(events, out, _VALID_NPI)

        assert export_result.event_count == 5
        assert export_result.attestation_event_count == 1
        assert out.exists()

        verify_result = verify_audit_trail(out)
        assert verify_result.is_valid, f"Tampering detected: {verify_result.tampering_detected}"
        assert verify_result.event_count == 5
        assert verify_result.attesting_npi == _VALID_NPI
        assert verify_result.signature_valid

    def test_chain_root_matches_between_export_and_verify(
        self, tmp_path: Path
    ) -> None:
        events = _make_events(3)
        out = tmp_path / "audit2.jsonld"
        export_result = export_audit_trail(events, out, _VALID_NPI)
        verify_result = verify_audit_trail(out)
        assert export_result.chain_root_hash == verify_result.chain_root_hash

    def test_export_writes_file_with_nonzero_bytes(self, tmp_path: Path) -> None:
        events = _make_events(2)
        out = tmp_path / "audit3.jsonld"
        export_result = export_audit_trail(events, out, _VALID_NPI)
        assert export_result.total_bytes > 0
        assert out.stat().st_size == export_result.total_bytes

    def test_empty_event_list_round_trip(self, tmp_path: Path) -> None:
        out = tmp_path / "empty.jsonld"
        export_result = export_audit_trail([], out, _VALID_NPI)
        assert export_result.event_count == 0
        verify_result = verify_audit_trail(out)
        assert verify_result.is_valid


# ── tamper detection ──────────────────────────────────────────────────────────

class TestTamperedEventBreaksVerification:
    """test_tampered_event_breaks_verification."""

    def test_modify_action_field_breaks_chain(self, tmp_path: Path) -> None:
        events = _make_events(4)
        out = tmp_path / "tampered.jsonld"
        export_audit_trail(events, out, _VALID_NPI)

        # Read, mutate middle event, write back
        doc = json.loads(out.read_text(encoding="utf-8"))
        doc["audit_events"][1]["action"] = "TAMPERED — unauthorized modification"
        out.write_text(json.dumps(doc), encoding="utf-8")

        result = verify_audit_trail(out)
        assert not result.is_valid
        assert len(result.tampering_detected) > 0

    def test_delete_event_breaks_chain(self, tmp_path: Path) -> None:
        events = _make_events(4)
        out = tmp_path / "deleted.jsonld"
        export_audit_trail(events, out, _VALID_NPI)

        doc = json.loads(out.read_text(encoding="utf-8"))
        # Remove event at index 1 — breaks predecessor chain from index 2 onward
        del doc["audit_events"][1]
        out.write_text(json.dumps(doc), encoding="utf-8")

        result = verify_audit_trail(out)
        assert not result.is_valid

    def test_insert_event_breaks_chain(self, tmp_path: Path) -> None:
        events = _make_events(3)
        out = tmp_path / "inserted.jsonld"
        export_audit_trail(events, out, _VALID_NPI)

        doc = json.loads(out.read_text(encoding="utf-8"))
        # Insert a synthetic event in the middle
        fake = copy.deepcopy(doc["audit_events"][0])
        fake["action"] = "INJECTED EVENT"
        doc["audit_events"].insert(1, fake)
        out.write_text(json.dumps(doc), encoding="utf-8")

        result = verify_audit_trail(out)
        assert not result.is_valid

    def test_after_hash_mutation_breaks_chain(self, tmp_path: Path) -> None:
        events = _make_events(3)
        out = tmp_path / "hash_tampered.jsonld"
        export_audit_trail(events, out, _VALID_NPI)

        doc = json.loads(out.read_text(encoding="utf-8"))
        # Flip one character in an after_hash
        original = doc["audit_events"][0]["after_hash"]
        doc["audit_events"][0]["after_hash"] = original[:-1] + (
            "0" if original[-1] != "0" else "1"
        )
        out.write_text(json.dumps(doc), encoding="utf-8")

        result = verify_audit_trail(out)
        assert not result.is_valid

    def test_self_verification_root_mutation_detected(self, tmp_path: Path) -> None:
        events = _make_events(3)
        out = tmp_path / "sv_tampered.jsonld"
        export_audit_trail(events, out, _VALID_NPI)

        doc = json.loads(out.read_text(encoding="utf-8"))
        original = doc["self_verification"]["chain_root_hash"]
        doc["self_verification"]["chain_root_hash"] = "a" * 64
        out.write_text(json.dumps(doc), encoding="utf-8")

        result = verify_audit_trail(out)
        assert not result.is_valid

    def test_returns_false_not_exception_on_tamper(self, tmp_path: Path) -> None:
        events = _make_events(2)
        out = tmp_path / "tampered2.jsonld"
        export_audit_trail(events, out, _VALID_NPI)

        doc = json.loads(out.read_text(encoding="utf-8"))
        doc["audit_events"][0]["actor_id"] = "CHANGED"
        out.write_text(json.dumps(doc), encoding="utf-8")

        # Must not raise; must return a result with is_valid=False
        result = verify_audit_trail(out)
        assert isinstance(result, VerificationResult)
        assert result.is_valid is False


# ── NPI validation ────────────────────────────────────────────────────────────

class TestAttestationSignerNPIMustBeLuhnValid:
    """test_attestation_signer_npi_must_be_luhn_valid."""

    def test_invalid_npi_raises_value_error(self, tmp_path: Path) -> None:
        out = tmp_path / "bad_npi.jsonld"
        with pytest.raises(ValueError, match="Luhn"):
            export_audit_trail(_make_events(1), out, _INVALID_NPI)

    def test_non_digit_npi_raises_value_error(self, tmp_path: Path) -> None:
        out = tmp_path / "bad_npi2.jsonld"
        with pytest.raises(ValueError, match="Luhn"):
            export_audit_trail(_make_events(1), out, "NOT-AN-NPI")

    def test_short_npi_raises_value_error(self, tmp_path: Path) -> None:
        out = tmp_path / "bad_npi3.jsonld"
        with pytest.raises(ValueError, match="Luhn"):
            export_audit_trail(_make_events(1), out, "123456789")

    def test_valid_npi_export_succeeds(self, tmp_path: Path) -> None:
        out = tmp_path / "good_npi.jsonld"
        result = export_audit_trail(_make_events(1), out, _VALID_NPI)
        assert result.attesting_npi == _VALID_NPI


# ── chronological ordering ────────────────────────────────────────────────────

class TestChronologicalOrderEnforced:
    """test_chronological_order_enforced."""

    def test_out_of_order_events_are_sorted(self, tmp_path: Path) -> None:
        events = [
            AuditEvent(
                timestamp_iso="2026-05-03T10:00:00Z",
                event_type="DRUG_INTERACTION_DETECTED",
                actor_id="engine",
                action="Third event",
                resource_ref="MedicationStatement/med-3",
                before_hash=_EMPTY_SHA,
                after_hash=_EMPTY_SHA,
                plan_hash=_EMPTY_SHA,
            ),
            AuditEvent(
                timestamp_iso="2026-05-01T08:00:00Z",
                event_type="MEMORY_WRITE",
                actor_id="engine",
                action="First event",
                resource_ref="Patient/p1",
                before_hash=_EMPTY_SHA,
                after_hash=_EMPTY_SHA,
                plan_hash=_EMPTY_SHA,
            ),
            AuditEvent(
                timestamp_iso="2026-05-02T14:30:00Z",
                event_type="FHIR_INGEST",
                actor_id="fhir_adapter",
                action="Second event",
                resource_ref="Bundle/b1",
                before_hash=_EMPTY_SHA,
                after_hash=_EMPTY_SHA,
                plan_hash=_EMPTY_SHA,
            ),
        ]
        out = tmp_path / "ordered.jsonld"
        export_audit_trail(events, out, _VALID_NPI)

        doc = json.loads(out.read_text(encoding="utf-8"))
        timestamps = [e["timestamp_iso"] for e in doc["audit_events"]]
        assert timestamps == sorted(timestamps), "Events not in chronological order"

    def test_sorted_and_unsorted_same_chain_root(self, tmp_path: Path) -> None:
        """Same events produce the same chain root regardless of input order."""
        e1 = AuditEvent("2026-05-01T00:00:00Z", "T1", "a", "A", "R/1",
                         _EMPTY_SHA, _EMPTY_SHA, _EMPTY_SHA)
        e2 = AuditEvent("2026-05-02T00:00:00Z", "T2", "b", "B", "R/2",
                         _EMPTY_SHA, _EMPTY_SHA, _EMPTY_SHA)

        out_sorted = tmp_path / "sorted.jsonld"
        out_reversed = tmp_path / "reversed.jsonld"

        r_sorted = export_audit_trail([e1, e2], out_sorted, _VALID_NPI)
        r_reversed = export_audit_trail([e2, e1], out_reversed, _VALID_NPI)

        assert r_sorted.chain_root_hash == r_reversed.chain_root_hash

    def test_verification_still_passes_after_unsorted_input(
        self, tmp_path: Path
    ) -> None:
        events = list(reversed(_make_events(4)))
        out = tmp_path / "reverse_verify.jsonld"
        export_audit_trail(events, out, _VALID_NPI)
        result = verify_audit_trail(out)
        assert result.is_valid


# ── JSON-LD context ───────────────────────────────────────────────────────────

class TestJsonLdContextPresentAndCorrect:
    """test_jsonld_context_present_and_correct."""

    def test_context_is_fda_guidance_url(self, tmp_path: Path) -> None:
        out = tmp_path / "ctx.jsonld"
        export_audit_trail(_make_events(1), out, _VALID_NPI)
        doc = json.loads(out.read_text(encoding="utf-8"))
        assert doc["@context"] == _JSONLD_CONTEXT

    def test_context_url_contains_fda_domain(self, tmp_path: Path) -> None:
        out = tmp_path / "ctx2.jsonld"
        export_audit_trail(_make_events(1), out, _VALID_NPI)
        doc = json.loads(out.read_text(encoding="utf-8"))
        assert "fda.gov" in doc["@context"]

    def test_context_url_references_part11_subject(self, tmp_path: Path) -> None:
        out = tmp_path / "ctx3.jsonld"
        export_audit_trail(_make_events(1), out, _VALID_NPI)
        doc = json.loads(out.read_text(encoding="utf-8"))
        # The URL should reference electronic records / electronic signatures
        assert "electronic-records" in doc["@context"].lower() or \
               "electronic_records" in doc["@context"].lower() or \
               "electronic-signatures" in doc["@context"].lower()

    def test_type_is_clinical_audit_trail(self, tmp_path: Path) -> None:
        out = tmp_path / "type.jsonld"
        export_audit_trail(_make_events(1), out, _VALID_NPI)
        doc = json.loads(out.read_text(encoding="utf-8"))
        assert doc["@type"] == "ClinicalAuditTrail"

    def test_regulation_field_is_21_cfr_part_11(self, tmp_path: Path) -> None:
        out = tmp_path / "reg.jsonld"
        export_audit_trail(_make_events(1), out, _VALID_NPI)
        doc = json.loads(out.read_text(encoding="utf-8"))
        assert "21 CFR Part 11" in doc.get("regulation", "")

    def test_verify_checks_context_mismatch(self, tmp_path: Path) -> None:
        out = tmp_path / "bad_ctx.jsonld"
        export_audit_trail(_make_events(1), out, _VALID_NPI)
        doc = json.loads(out.read_text(encoding="utf-8"))
        doc["@context"] = "https://example.com/wrong-context"
        out.write_text(json.dumps(doc), encoding="utf-8")
        result = verify_audit_trail(out)
        assert not result.is_valid
        assert any("context" in m.lower() for m in result.tampering_detected)


# ── chain root hash ───────────────────────────────────────────────────────────

class TestChainRootHashIsSha256OfCanonicalEncoding:
    """test_chain_root_hash_is_sha256_of_canonical_encoding."""

    def test_chain_root_is_64_char_hex(self, tmp_path: Path) -> None:
        events = _make_events(3)
        out = tmp_path / "root.jsonld"
        result = export_audit_trail(events, out, _VALID_NPI)
        assert len(result.chain_root_hash) == 64
        assert all(c in "0123456789abcdef" for c in result.chain_root_hash)

    def test_chain_root_deterministic_for_same_events(
        self, tmp_path: Path
    ) -> None:
        events = _make_events(3)
        out_a = tmp_path / "root_a.jsonld"
        out_b = tmp_path / "root_b.jsonld"
        r_a = export_audit_trail(events, out_a, _VALID_NPI)
        r_b = export_audit_trail(events, out_b, _VALID_NPI)
        assert r_a.chain_root_hash == r_b.chain_root_hash

    def test_different_events_produce_different_roots(
        self, tmp_path: Path
    ) -> None:
        events_a = _make_events(3)
        events_b = _make_events(3)
        events_b[0] = AuditEvent(
            timestamp_iso="2026-05-01T10:00:00Z",
            event_type="DIFFERENT_TYPE",
            actor_id="actor-0",
            action="Different action",
            resource_ref="Patient/different",
            before_hash=_EMPTY_SHA,
            after_hash=_EMPTY_SHA,
            plan_hash=_EMPTY_SHA,
        )
        out_a = tmp_path / "diff_a.jsonld"
        out_b = tmp_path / "diff_b.jsonld"
        r_a = export_audit_trail(events_a, out_a, _VALID_NPI)
        r_b = export_audit_trail(events_b, out_b, _VALID_NPI)
        assert r_a.chain_root_hash != r_b.chain_root_hash

    def test_manual_chain_replay_matches_export(self, tmp_path: Path) -> None:
        """Manually replay the chain and confirm the root matches."""
        import hashlib as hl

        events = _make_events(2)
        out = tmp_path / "manual.jsonld"
        export_result = export_audit_trail(events, out, _VALID_NPI)

        # Replay manually
        doc = json.loads(out.read_text(encoding="utf-8"))
        running = _CHAIN_SEED
        for ev in doc["audit_events"]:
            payload_keys = {
                "timestamp_iso", "event_type", "actor_id", "action",
                "resource_ref", "before_hash", "plan_hash",
            }
            payload = {k: ev[k] for k in payload_keys if k in ev}
            canonical = _event_canonical(payload)
            running = _chain_hash(running, canonical)

        assert running == export_result.chain_root_hash

    def test_self_verification_block_present(self, tmp_path: Path) -> None:
        events = _make_events(2)
        out = tmp_path / "sv.jsonld"
        result = export_audit_trail(events, out, _VALID_NPI)
        doc = json.loads(out.read_text(encoding="utf-8"))
        sv = doc["self_verification"]
        assert sv["chain_root_hash"] == result.chain_root_hash
        assert sv["event_count"] == result.event_count
        assert sv["attestation_signer_npi"] == _VALID_NPI

    def test_metadata_block_included_by_default(self, tmp_path: Path) -> None:
        out = tmp_path / "meta.jsonld"
        export_audit_trail(_make_events(1), out, _VALID_NPI)
        doc = json.loads(out.read_text(encoding="utf-8"))
        assert "metadata" in doc
        assert "public_key_b64url" in doc["metadata"]
        assert "hsm_note" in doc["metadata"]

    def test_metadata_excluded_when_flag_false(self, tmp_path: Path) -> None:
        out = tmp_path / "no_meta.jsonld"
        export_audit_trail(_make_events(1), out, _VALID_NPI,
                           include_metadata=False)
        doc = json.loads(out.read_text(encoding="utf-8"))
        assert "metadata" not in doc
