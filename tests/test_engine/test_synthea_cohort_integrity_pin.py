"""Pin the structural integrity of the Synthea-style FHIR demo cohort.

Iter 125 (round 25 T1 substitution).

`docs/synthea_demo_cohort.json` is a load-bearing artifact: every
demo finding ("4 conflicts detected automatically", every patient
case study, every per-patient drug-pair → cache-entry trace) rides
on this bundle being structurally well-formed. The demo + JUDGES
make multiple claims about the bundle:

  1. "29 synthetic patients · 46 NPIs"
  2. "every Practitioner has an NPI" (Luhn-valid)
  3. "Same shape Epic, Cerner, and every certified EHR speak"
  4. "All NPIs Luhn-valid" (FHIR R4 spec compliance)
  5. The demo's `_synthetic` meta flag distinguishes Synthea-generated
     fictional resources from any future real-EHR ingest

Without a pin, silent drift could:
  - Drop a Patient resource (size shrinks, demo claims look wrong)
  - Add a Practitioner without an NPI (demo NPI-Luhn claim breaks)
  - Add an NPI that fails Luhn (claim becomes false)
  - Forget the `_meta._synthetic` tag on a new resource (the
    iter-iter NPI_SOURCE = "DEMO_LUHN_GENERATED" guard is the
    fail-safe against fictional NPIs polluting real CMS lookups)
  - Reduce evidence URLs / break NPI coverage in any EHR-shape way

Same drift class as iter-100 pharmacology_flags coverage pin and
iter-120 negative-control cohort integrity pin: load-bearing
artifact must be auditable from the test suite.
"""
from __future__ import annotations

import json
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_BUNDLE = _REPO_ROOT / "docs" / "synthea_demo_cohort.json"
_DEMO = _REPO_ROOT / "docs" / "demo.html"


def _load_bundle() -> dict:
    return json.loads(_BUNDLE.read_text())


def _resources_by_type(bundle: dict) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for entry in bundle["entry"]:
        rt = entry["resource"]["resourceType"]
        out.setdefault(rt, []).append(entry["resource"])
    return out


def _luhn_validate_npi(npi: str) -> bool:
    """CMS NPI Luhn check digit algorithm (CMS NPI checkdigit spec)."""
    if not npi.isdigit() or len(npi) != 10:
        return False
    # NPI uses constant prefix 80840 + 9 digits + check digit.
    digits = [int(c) for c in "80840" + npi[:9]]
    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 0:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    expected_check = (10 - (total % 10)) % 10
    return expected_check == int(npi[9])


def test_bundle_top_level_shape():
    """The bundle must be a FHIR R4 Bundle (resourceType + type + entry list)."""
    bundle = _load_bundle()
    assert bundle.get("resourceType") == "Bundle", (
        "Top-level resourceType must be 'Bundle' for FHIR R4 compliance."
    )
    assert bundle.get("type") in ("collection", "transaction"), (
        f"FHIR Bundle.type must be 'collection' or 'transaction'; "
        f"got {bundle.get('type')!r}"
    )
    assert isinstance(bundle.get("entry"), list), (
        "Bundle.entry must be a list of FHIR resources."
    )
    assert len(bundle["entry"]) >= 1, "Bundle must contain at least one entry."


def test_patient_count_floor():
    """Patient count must be >= 30 (iter-129 floor).

    Future cohort growth is healthy but must be visible — adding a new
    patient without bumping this floor means the patient-count claims
    in the demo + JUDGES + IRB are silently outdated.
    """
    bundle = _load_bundle()
    by = _resources_by_type(bundle)
    patients = by.get("Patient", [])
    assert len(patients) >= 30, (
        f"Patient count regressed: live={len(patients)}, "
        f"floor=30 (iter-129). Either a Patient was deleted, or the "
        f"floor needs lifting in the same commit as a deliberate "
        f"cohort change."
    )


def test_practitioner_count_floor():
    """Practitioner count must be >= 47 (iter-129 floor)."""
    bundle = _load_bundle()
    by = _resources_by_type(bundle)
    practitioners = by.get("Practitioner", [])
    assert len(practitioners) >= 47, (
        f"Practitioner count regressed: live={len(practitioners)}, "
        f"floor=47 (iter-129)."
    )


def test_every_practitioner_has_npi():
    """Every Practitioner must carry an NPI identifier.

    The demo's NPI Luhn claim ("every Practitioner has an HHS NPI,
    Luhn-validated") is meaningless if any Practitioner is missing
    its identifier.
    """
    bundle = _load_bundle()
    by = _resources_by_type(bundle)
    offenders: list[str] = []
    for prac in by.get("Practitioner", []):
        prac_id = prac.get("id", "<no-id>")
        identifiers = prac.get("identifier", [])
        npi_id = next(
            (
                i for i in identifiers
                if i.get("system") == "http://hl7.org/fhir/sid/us-npi"
            ),
            None,
        )
        if npi_id is None or not npi_id.get("value"):
            offenders.append(prac_id)
    assert not offenders, (
        f"Practitioner(s) without an NPI identifier: {offenders}. "
        f"Add `identifier: [{{system: 'http://hl7.org/fhir/sid/us-npi', "
        f"value: '<10-digit>'}}]` to each."
    )


def test_every_npi_passes_cms_luhn():
    """Every Practitioner NPI must pass the CMS Luhn check digit algorithm.

    The demo's "Luhn-validated" claim is computed by Joel's CMS
    checkdigit algorithm: 80840 prefix + 9 digits + check. A
    fail-Luhn NPI in the bundle would invalidate the demo's "every
    Practitioner has an NPI, Luhn-validated" claim outright.
    """
    bundle = _load_bundle()
    by = _resources_by_type(bundle)
    offenders: list[tuple[str, str]] = []
    for prac in by.get("Practitioner", []):
        prac_id = prac.get("id", "<no-id>")
        for identifier in prac.get("identifier", []):
            if identifier.get("system") != "http://hl7.org/fhir/sid/us-npi":
                continue
            npi = identifier.get("value", "")
            if not _luhn_validate_npi(npi):
                offenders.append((prac_id, npi))
    assert not offenders, (
        f"NPI(s) failing CMS Luhn validation: {offenders}. Generate "
        f"replacements via "
        f"`engine.npi_registry.generate_test_npi(<seed>)` which "
        f"computes the correct check digit."
    )


def test_every_patient_and_practitioner_marked_synthetic():
    """Every Patient + Practitioner resource carries `_meta._synthetic = true`.

    Convention: Patient + Practitioner are the resources that carry
    *identifying* metadata (names, NPIs, MRNs) — those MUST be tagged
    synthetic so a future EHR-ingest path never confuses them with
    real registry data.

    Conditions / Medications / Observations / AllergyIntolerances all
    reference a synthetic Patient via `subject.reference` and inherit
    the synthetic provenance through that link — so the convention
    only requires the explicit tag on identity-bearing resources.
    Future iter could extend this to all resources via a backfill;
    until then, this pin enforces the existing observed convention.
    """
    bundle = _load_bundle()
    offenders: list[str] = []
    for entry in bundle["entry"]:
        resource = entry["resource"]
        rt = resource.get("resourceType")
        if rt not in ("Patient", "Practitioner"):
            continue
        rid = resource.get("id", "<no-id>")
        meta = resource.get("meta", {})
        if not meta.get("_synthetic"):
            offenders.append(f"{rt}/{rid}")
    assert not offenders, (
        f"Patient/Practitioner resource(s) missing `meta._synthetic = "
        f"true`: {offenders}. Identity-bearing resources MUST be "
        f"tagged so a future EHR-ingest path never confuses synthetic "
        f"demo identifiers with real CMS / hospital registry data."
    )


def test_every_practitioner_has_demo_luhn_npi_source_tag():
    """Every Practitioner.meta must carry npi_source = 'DEMO_LUHN_GENERATED'.

    iter-iter convention: the synthetic-NPI guard. A real CMS NPI
    lookup finding npi_source = 'CMS_REGISTRY' would treat the value
    as canonical; finding 'DEMO_LUHN_GENERATED' tells the lookup to
    NOT contact CMS NPPES. The tag is the integrity primitive
    distinguishing demo data from real data.
    """
    bundle = _load_bundle()
    by = _resources_by_type(bundle)
    offenders: list[str] = []
    for prac in by.get("Practitioner", []):
        prac_id = prac.get("id", "<no-id>")
        meta = prac.get("meta", {})
        if meta.get("npi_source") != "DEMO_LUHN_GENERATED":
            offenders.append(prac_id)
    assert not offenders, (
        f"Practitioner(s) missing `meta.npi_source = "
        f"'DEMO_LUHN_GENERATED'`: {offenders}. Required to prevent "
        f"a future real-EHR ingest path from accidentally querying "
        f"CMS NPPES with these synthetic NPIs."
    )


def test_demo_cites_this_pin_file():
    """Demo must cite this pin file near the cohort-integrity callout.

    Iter 126 (T2 surfacing) added a post-FHIR-grid blue-rule callout
    in docs/demo.html that names the 8 cohort-integrity invariants
    and references this pin file. A future copy edit could silently
    drop the reference, making the cohort-integrity claim look
    unguarded even though the tests still pass internally. This pin
    enforces the demo surfacing.

    Same pattern as iter-110 + iter-115 + iter-121's "demo cites pin
    file" cross-checks: test layer and user-facing surface stay in
    sync.
    """
    text = _DEMO.read_text()
    pin_filename = "test_synthea_cohort_integrity_pin.py"
    assert pin_filename in text, (
        f"docs/demo.html must cite "
        f"`tests/test_engine/{pin_filename}` near the FHIR cohort-"
        f"integrity callout so judges can trace the 29-patient / "
        f"46-NPI / Luhn-validated claims to their enforcing pin file. "
        f"Same drift class as iter-107 JUDGES manifest description fix."
    )
    # Anchor: the callout must appear with a recognizable phrase so
    # a copy edit can't strip the rationale and leave just the filename
    # (which would still pass the substring check above but lose
    # judge-readable context).
    locality_anchors = (
        "8 invariants",
        "Cohort integrity is gated",
        "FHIR R4 Bundle top-level shape",
    )
    has_anchor = any(a in text for a in locality_anchors)
    assert has_anchor, (
        f"Demo's pin-file citation must appear with a locality anchor "
        f"such as '8 invariants' / 'Cohort integrity is gated' / "
        f"'FHIR R4 Bundle top-level shape'. None found near the FHIR "
        f"cohort callout — copy edit may have stripped the rationale."
    )


def test_no_real_npi_substring_collisions():
    """No demo NPI may match a publicly-known real NPI (paranoia check).

    `engine.npi_registry.generate_test_npi(seed)` produces Luhn-valid
    10-digit NPIs from a deterministic seed; the check-digit math
    means most generated NPIs land in the same number space as real
    CMS-registered NPIs. This isn't a privacy issue for demo purposes
    (we explicitly tag DEMO_LUHN_GENERATED) but if a generated NPI
    happens to collide with a known-real NPI from the very small
    publicly-disclosed set in our project context, that's worth
    surfacing.

    The reserved-known-real list below is project-scoped — the NPI was
    used in earlier clinical-review materials and is kept here as a
    collision-detection sentinel so cohort generation cannot silently
    duplicate it.
    """
    KNOWN_REAL_NPIS = frozenset({"1932159530"})
    bundle = _load_bundle()
    by = _resources_by_type(bundle)
    cohort_npis = set()
    for prac in by.get("Practitioner", []):
        for identifier in prac.get("identifier", []):
            if identifier.get("system") == "http://hl7.org/fhir/sid/us-npi":
                cohort_npis.add(identifier.get("value", ""))
    collisions = cohort_npis & KNOWN_REAL_NPIS
    assert not collisions, (
        f"Demo cohort NPI(s) collide with known-real NPIs: "
        f"{sorted(collisions)}. The clinical-validation NPI must NOT "
        f"appear inside the demo bundle — it lives only in "
        f"docs/clinical_validation.md as a real-clinician attestation."
    )
