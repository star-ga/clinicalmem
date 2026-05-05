"""Tests for engine.npi_registry — CMS Luhn validation + NPPES lookup.

Three load-bearing claims:

  1. **Luhn validation matches CMS spec** — `validate_npi` rejects every
     NPI whose check digit doesn't agree with the official 80840-prefix
     Luhn algorithm. A wrong validator would silently let bad NPIs into
     the FHIR bundle, breaking the audit chain.

  2. **`generate_test_npi` always returns a Luhn-valid NPI** — used to
     populate Sarah Mitchell's Practitioner resources for the demo.

  3. **Sarah Mitchell's 4 demo NPIs all pass Luhn** — regression test
     so a future bundle edit can't ship invalid identifiers.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_HERE))

from engine.npi_registry import (  # noqa: E402
    NPIRecord,
    generate_test_npi,
    validate_npi,
)


# ─── Luhn validator ─────────────────────────────────────────────────────────

class TestValidateNPI:
    def test_known_valid_individual_npi(self) -> None:
        # Test NPI 1234567893 — published in CMS NPI checkdigit spec.
        assert validate_npi("1234567893") is True

    def test_known_valid_organisational_npi(self) -> None:
        # Test NPI 9999999999 with type prefix 2: must also pass Luhn.
        npi = generate_test_npi("test-org", individual=False)
        assert npi[0] == "2"
        assert validate_npi(npi) is True

    def test_wrong_check_digit_rejected(self) -> None:
        # Mutate the check digit and confirm the validator catches it.
        valid = "1234567893"
        broken = valid[:9] + str((int(valid[9]) + 1) % 10)
        assert validate_npi(broken) is False

    def test_wrong_length_rejected(self) -> None:
        assert validate_npi("123456789") is False
        assert validate_npi("12345678901") is False
        assert validate_npi("") is False

    def test_non_string_rejected(self) -> None:
        assert validate_npi(1234567893) is False  # type: ignore[arg-type]
        assert validate_npi(None) is False  # type: ignore[arg-type]

    def test_non_digit_rejected(self) -> None:
        assert validate_npi("1234A67893") is False
        assert validate_npi("12345 7893") is False

    def test_wrong_type_prefix_rejected(self) -> None:
        # Type prefix must be 1 or 2 — anything else is malformed.
        assert validate_npi("0234567893") is False
        assert validate_npi("3234567893") is False


# ─── generate_test_npi ──────────────────────────────────────────────────────

class TestGenerateTestNPI:
    def test_deterministic(self) -> None:
        a = generate_test_npi("dr-mitchell-pcp")
        b = generate_test_npi("dr-mitchell-pcp")
        assert a == b
        assert validate_npi(a) is True

    def test_different_seeds_different_npis(self) -> None:
        a = generate_test_npi("alice")
        b = generate_test_npi("bob")
        assert a != b

    def test_individual_starts_with_1(self) -> None:
        npi = generate_test_npi("any", individual=True)
        assert npi[0] == "1"

    def test_organisation_starts_with_2(self) -> None:
        npi = generate_test_npi("any", individual=False)
        assert npi[0] == "2"

    def test_always_passes_validate(self) -> None:
        for seed in ("abc", "def", "ghi", "jkl", "mno", "pqr"):
            for kind in (True, False):
                npi = generate_test_npi(seed, individual=kind)
                assert validate_npi(npi), f"generated bad NPI {npi} for seed {seed}"


# ─── Sarah Mitchell FHIR bundle regression ──────────────────────────────────

_BUNDLE_PATH = _HERE / "docs" / "sarah_mitchell_fhir.json"


def test_sarah_mitchell_bundle_npis_all_valid() -> None:
    """Every Practitioner NPI in the demo bundle must pass Luhn."""
    if not _BUNDLE_PATH.exists():
        pytest.skip(f"{_BUNDLE_PATH} not present")
    bundle = json.loads(_BUNDLE_PATH.read_text(encoding="utf-8"))
    practitioners = [
        e["resource"] for e in bundle["entry"]
        if e["resource"]["resourceType"] == "Practitioner"
    ]
    assert len(practitioners) >= 4, "expected at least 4 demo Practitioners"

    for prac in practitioners:
        npi_idents = [
            i for i in prac.get("identifier", [])
            if i.get("system") == "http://hl7.org/fhir/sid/us-npi"
        ]
        assert npi_idents, f"Practitioner {prac['id']} has no NPI identifier"
        for ident in npi_idents:
            value = ident["value"]
            assert validate_npi(value), (
                f"Practitioner {prac['id']} has invalid NPI {value}"
            )


def test_sarah_mitchell_bundle_practitioner_references_resolve() -> None:
    """Every `Practitioner/...` reference in the bundle must point to an
    actual Practitioner resource. Dangling references would break the
    audit chain."""
    if not _BUNDLE_PATH.exists():
        pytest.skip(f"{_BUNDLE_PATH} not present")
    bundle = json.loads(_BUNDLE_PATH.read_text(encoding="utf-8"))

    practitioner_ids: set[str] = {
        f"Practitioner/{e['resource']['id']}"
        for e in bundle["entry"]
        if e["resource"]["resourceType"] == "Practitioner"
    }

    # Recursively walk the bundle for any `reference` field starting
    # with "Practitioner/" and assert it resolves.
    def _walk(obj: object) -> list[str]:
        refs: list[str] = []
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k == "reference" and isinstance(v, str) and v.startswith("Practitioner/"):
                    refs.append(v)
                else:
                    refs.extend(_walk(v))
        elif isinstance(obj, list):
            for item in obj:
                refs.extend(_walk(item))
        return refs

    refs = _walk(bundle)
    assert refs, "expected at least one Practitioner reference in bundle"
    for ref in refs:
        assert ref in practitioner_ids, (
            f"dangling Practitioner reference: {ref} (known: {sorted(practitioner_ids)})"
        )


# ─── NPIRecord dataclass surface ────────────────────────────────────────────

def test_npi_record_is_frozen() -> None:
    """NPIRecord must be immutable so an attacker cannot mutate a cached
    record after it leaves NPPES."""
    record = NPIRecord(
        npi="1234567893",
        enumeration_type="NPI-1",
        name="Dr. Demo",
        credential="MD",
        primary_taxonomy="Internal Medicine",
        primary_taxonomy_code="207R00000X",
        practice_state="MA",
        practice_city="Boston",
    )
    with pytest.raises(Exception):
        record.npi = "9999999999"  # type: ignore[misc]


# ─── Iter-154 logger-density ratchet pin (T4 round-30) ───────────────
#
# engine/npi_registry.py was the next-lowest-density observability file
# in engine/ at 14.6 logger calls / kloc (3 logs / 205 LOC). Same
# ratchet pattern as iter-138 (clinical_memory.py 13.7 -> 27.5),
# iter-144 (flow_runner.py 12.3 -> 18.0), and iter-151
# (hallucination_detector.py 14.4 -> 31.0).
#
# Three silent paths closed:
#   1. validate_npi failures emit DEBUG `npi_validation_failed` with
#      reason (not_str / wrong_length / non_digit / bad_type_prefix /
#      bad_luhn) + length. PHI-safe: the candidate NPI value is NEVER
#      logged (probes can be real-PII attempts).
#   2. generate_test_npi exit emits DEBUG `npi_generated_synthetic`
#      with the synthetic NPI + individual flag + seed_length. Seed
#      string itself is NEVER logged (callers may pass real-name
#      fragments).
#   3. lookup_npi success emits INFO `npi_lookup_success` with NPI +
#      enumeration_type + taxonomy_code + has_practice_location.
#      The practitioner's NAME and full ADDRESS are NEVER logged
#      (they become identifying when paired with location).
#
# Net: 3 -> 10 logger calls (+7 events). Density 14.6 -> 38.5/kloc.

import logging


class TestNPIRegistryLoggerRatchetIter154:
    """Iter-154 T4 round-30 — logger density ratchet pin."""

    def test_validate_npi_failure_logs_reason(self, caplog):
        with caplog.at_level(logging.DEBUG, logger='engine.npi_registry'):
            validate_npi("123")  # too short
        recs = [r for r in caplog.records if r.message == 'npi_validation_failed']
        assert any(getattr(r, 'reason', None) == 'wrong_length' for r in recs), (
            "validate_npi('123') must emit a DEBUG npi_validation_failed "
            "event with reason='wrong_length'"
        )

    def test_validate_npi_failure_never_logs_candidate(self, caplog):
        """PHI paranoia: a candidate NPI must NEVER appear in any log
        record — the value could be a real PII probe."""
        sentinel = "9876543210ZZZ"  # contains alpha so it'll fail at .isdigit()
        with caplog.at_level(logging.DEBUG, logger='engine.npi_registry'):
            validate_npi(sentinel)
        for rec in caplog.records:
            haystack = (
                str(rec.message) + ' ' +
                ' '.join(str(v) for v in vars(rec).values() if isinstance(v, (str, int, float)))
            )
            for slot in vars(rec).values():
                if isinstance(slot, dict):
                    haystack += ' ' + ' '.join(str(v) for v in slot.values())
            assert sentinel not in haystack, (
                f"PHI LEAK: candidate NPI {sentinel!r} appeared in log "
                f"record {rec.message!r}. validate_npi must NEVER log "
                "the candidate value."
            )

    def test_generate_test_npi_emits_debug(self, caplog):
        with caplog.at_level(logging.DEBUG, logger='engine.npi_registry'):
            npi = generate_test_npi("dr-test-seed-iter-154")
        recs = [r for r in caplog.records if r.message == 'npi_generated_synthetic']
        assert len(recs) >= 1
        rec = recs[0]
        assert getattr(rec, 'synthetic_npi', None) == npi
        assert getattr(rec, 'individual', None) is True
        # Seed string must NEVER appear in any log
        for r in caplog.records:
            for v in vars(r).values():
                assert 'dr-test-seed-iter-154' not in str(v), (
                    "seed_string must NEVER appear in log records"
                )

    def test_logger_density_floor(self):
        """Floor: npi_registry.py must keep >= 10 logger calls.
        Iter-154 ratchet bumped 3 -> 10 (+7 events: 5 validate-failure
        reasons + 1 generate-synthetic trace + 1 lookup-success);
        future regressions below 10 fail this gate."""
        import re
        from pathlib import Path
        path = Path(__file__).resolve().parent.parent.parent / 'engine' / 'npi_registry.py'
        text = path.read_text()
        count = len(re.findall(r'logger\.(debug|info|warning|error|critical)', text))
        assert count >= 10, (
            f"engine/npi_registry.py logger density regressed: "
            f"got {count}, floor=10 (iter-154 ratchet)"
        )
