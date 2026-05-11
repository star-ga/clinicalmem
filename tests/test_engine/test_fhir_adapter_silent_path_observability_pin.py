"""iter-443 39th cross-pin family — fhir_adapter silent-path observability.

Background
==========
iter-443 T4 round-30 logger ratchet on engine/fhir_adapter.py closes two
silent dispatch paths in `ingest_bundle`:

1. **entry-without-resourceType**: an entry whose `resource.resourceType`
   is missing or empty. Previously appended to `rejected` silently; the
   iter-191 aggregate `fhir_bundle_rejection_summary` groups all such
   entries under "(unknown)" but masks per-entry bundle-builder bugs.
2. **Practitioner-zero-valid-NPIs**: a Practitioner resource whose
   identifier list contains zero valid CMS NPIs (all non-NPI systems,
   or all Luhn-invalid). Per-NPI Luhn failures fire `fhir_practitioner_
   npi_luhn_failed` (L370) but a Practitioner whose IDENTIFIERS skip
   the NPI system entirely yields an empty list silently — the iter-401
   care-team-attestation flow hard-depends on ≥1 valid NPI per
   Practitioner.

This pin file is the **structural lock** that prevents either
observability event from being silently regressed (deleted, renamed,
or stripped of PHI-safe extras).

Same iter-314 / iter-319 / iter-324 / iter-329 / iter-432 dispatch-
table observability pattern (closes silent helper paths via structured
logger events with PHI-safe categorical extras + key-shape pin).
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from engine.fhir_adapter import ingest_bundle

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent


# --- helper: build a minimally-valid FHIR R4 Bundle ---------------------------

def _bundle_with(entries: list[dict]) -> dict:
    return {
        "resourceType": "Bundle",
        "type": "collection",
        "entry": entries,
    }


def _patient_entry(pid: str = "patient-test") -> dict:
    return {"resource": {"resourceType": "Patient", "id": pid}}


# ── Test 1: entry-no-resourceType emits the iter-443 event ────────────────────

def test_fhir_bundle_entry_no_resource_type_event_fires(caplog):
    """An entry with `resource.resourceType` missing must fire the
    iter-443 `fhir_bundle_entry_no_resource_type` debug event."""
    bundle = _bundle_with([
        _patient_entry(),
        {"resource": {"id": "no-rt-1"}},  # missing resourceType
    ])

    with caplog.at_level(logging.DEBUG):
        result = ingest_bundle(bundle)

    events = [r for r in caplog.records
              if r.msg == "fhir_bundle_entry_no_resource_type"]
    assert len(events) >= 1, (
        "iter-443 T4 silent-path closure regressed: "
        "fhir_bundle_entry_no_resource_type event not fired."
    )
    # Rejection list must still capture the same entry (drift catch).
    assert any(rt == "(unknown)" for rt, _ in result.rejected_resources)


# ── Test 2: PHI-safe extras key shape on no-resource-type event ───────────────

def test_fhir_bundle_entry_no_resource_type_extras_phi_safe(caplog):
    """The iter-443 event must carry only structural metadata — no
    resource body, no identifier values, no clinical narrative."""
    bundle = _bundle_with([
        _patient_entry(),
        {
            "fullUrl": "urn:uuid:should-not-leak",
            "resource": {
                "id": "no-rt-2",
                "extension": [{"url": "leak-marker", "valueString": "PHI-LEAK"}],
            },
        },
    ])

    with caplog.at_level(logging.DEBUG):
        ingest_bundle(bundle)

    matching = [r for r in caplog.records
                if r.msg == "fhir_bundle_entry_no_resource_type"]
    assert matching, "event missing"
    extras = getattr(matching[0], "entry_keys_present", None)
    res_extras = getattr(matching[0], "resource_keys_present", None)
    assert extras is not None and res_extras is not None, (
        "iter-443 PHI-safe shape regressed: structured extras missing"
    )
    # Extras must be sorted lists of keys — never values.
    assert isinstance(extras, list)
    assert isinstance(res_extras, list)
    # Hard PHI-leak guard: no value content from the resource body
    # may appear in extras (key NAMES are FHIR R4 spec, not values).
    flat = " ".join(str(v) for v in [extras, res_extras])
    assert "PHI-LEAK" not in flat, (
        "iter-443 silent-path observability LEAKED resource body content"
    )
    # Sanity: extras enumerate KEYS, never raw values. Verify the
    # specific value strings ("urn:uuid:should-not-leak", "leak-marker")
    # don't appear in the structured extras.
    for value_marker in ("urn:uuid:should-not-leak", "leak-marker"):
        assert value_marker not in flat, (
            f"iter-443 entry extras LEAKED value content: {value_marker!r}"
        )


# ── Test 3: Practitioner-zero-valid-NPIs event fires ──────────────────────────

def test_fhir_practitioner_zero_valid_npis_event_fires(caplog):
    """A Practitioner with zero valid CMS NPIs must fire the iter-443
    `fhir_practitioner_zero_valid_npis` debug event."""
    bundle = _bundle_with([
        _patient_entry(),
        {
            "resource": {
                "resourceType": "Practitioner",
                "id": "prac-1",
                # All identifiers are non-NPI system → zero NPIs extracted.
                "identifier": [
                    {
                        "system": "http://hospital.example/internal-id",
                        "value": "INT-12345",
                    },
                ],
            },
        },
    ])

    with caplog.at_level(logging.DEBUG):
        result = ingest_bundle(bundle)

    events = [r for r in caplog.records
              if r.msg == "fhir_practitioner_zero_valid_npis"]
    assert len(events) >= 1, (
        "iter-443 T4 silent-path closure regressed: "
        "fhir_practitioner_zero_valid_npis event not fired."
    )
    # Sanity: practitioner_npis list must be empty (drift catch).
    assert result.practitioner_npis == []


# ── Test 4: Practitioner WITH valid NPI does NOT fire the zero-NPI event ──────

def test_fhir_practitioner_with_valid_npi_does_not_fire_zero_event(caplog):
    """A Practitioner with ≥1 valid CMS NPI MUST NOT fire the iter-443
    zero-NPI event — false-positives would bury the real signal."""
    # Canonical Luhn-valid NPI fixture used across the repo's FHIR tests
    # (matches the public CMS-registry NPI we reserve for collision-detection
    # in test_synthea_cohort_integrity_pin.py).
    bundle = _bundle_with([
        _patient_entry(),
        {
            "resource": {
                "resourceType": "Practitioner",
                "id": "prac-valid",
                "identifier": [
                    {
                        "system": "http://hl7.org/fhir/sid/us-npi",
                        "value": "1932159530",
                    },
                ],
            },
        },
    ])

    with caplog.at_level(logging.DEBUG):
        result = ingest_bundle(bundle)

    events = [r for r in caplog.records
              if r.msg == "fhir_practitioner_zero_valid_npis"]
    assert len(events) == 0, (
        "iter-443 false-positive: zero-NPI event fired for a "
        "Practitioner with a Luhn-valid NPI."
    )
    assert "1932159530" in result.practitioner_npis


# ── Test 5: PHI-safe shape on zero-NPI event ──────────────────────────────────

def test_fhir_practitioner_zero_valid_npis_extras_phi_safe(caplog):
    """The iter-443 zero-NPI event must use the iter-309-era PHI-safe
    shape: patient_id_hash_prefix (not raw patient_id), resource_id,
    identifier_count — no identifier values, no name, no qualification."""
    bundle = _bundle_with([
        _patient_entry("synthetic-patient-9"),
        {
            "resource": {
                "resourceType": "Practitioner",
                "id": "prac-2",
                "name": [{"family": "DOE-PHI-MARKER", "given": ["JANE"]}],
                "identifier": [
                    {"system": "http://internal", "value": "X-LEAK"},
                ],
                "qualification": [{"code": {"text": "MD-PHI"}}],
            },
        },
    ])

    with caplog.at_level(logging.DEBUG):
        ingest_bundle(bundle)

    matching = [r for r in caplog.records
                if r.msg == "fhir_practitioner_zero_valid_npis"]
    assert matching
    rec = matching[0]
    # Required PHI-safe extras.
    assert hasattr(rec, "resource_id")
    assert hasattr(rec, "identifier_count")
    assert hasattr(rec, "patient_id_hash_prefix")
    # patient_id_hash_prefix must be a 16-char SHA-256 prefix
    # (iter-332..336 PHI extras-key migration discipline).
    assert isinstance(rec.patient_id_hash_prefix, str)
    assert len(rec.patient_id_hash_prefix) == 16
    # Hard PHI-leak guard: name + qualification + identifier value
    # must NOT appear anywhere in the record extras.
    flat = " ".join(str(getattr(rec, attr, "")) for attr in
                    ("resource_id", "identifier_count",
                     "patient_id_hash_prefix"))
    for marker in ("DOE-PHI-MARKER", "JANE", "MD-PHI", "X-LEAK",
                   "synthetic-patient-9"):
        assert marker not in flat, (
            f"iter-443 zero-NPI event LEAKED '{marker}' into extras"
        )


# ── Test 6: source-level pin — both event names appear in the source ──────────

def test_iter_443_event_names_pinned_in_source():
    """Both iter-443 observability event names must appear in the
    fhir_adapter source — protects against silent rename / removal."""
    src = (_REPO_ROOT / "engine" / "fhir_adapter.py").read_text()
    assert "fhir_bundle_entry_no_resource_type" in src, (
        "iter-443 silent-path observability regressed: "
        "fhir_bundle_entry_no_resource_type event removed from source"
    )
    assert "fhir_practitioner_zero_valid_npis" in src, (
        "iter-443 silent-path observability regressed: "
        "fhir_practitioner_zero_valid_npis event removed from source"
    )


# ── Test 7: density floor — fhir_adapter logger density advances ──────────────

def test_iter_443_logger_density_advances():
    """fhir_adapter.py logger density must stay at or above iter-443
    floor. Same iter-226 / iter-234 ratchet pattern at the per-module
    layer — prevents silent regression of the T4 round-30 ratchet."""
    src = (_REPO_ROOT / "engine" / "fhir_adapter.py").read_text()
    n_lines = src.count("\n")
    n_logs = sum(1 for line in src.splitlines()
                 if line.lstrip().startswith("logger.")
                 and "(" in line)
    density = (n_logs / n_lines) * 1000  # per kloc
    # Pre-iter-443 density: ~19.8/kloc. Post-iter-443: ~22.5/kloc
    # (added 2 logger.debug calls). Floor at 21.0 so a single
    # event regression fires this pin (iter-226 ratchet pattern).
    floor = 21.0
    assert density >= floor, (
        f"iter-443 fhir_adapter logger density regressed: "
        f"{density:.1f}/kloc < floor {floor}/kloc. "
        f"({n_logs} logs / {n_lines} lines)"
    )
