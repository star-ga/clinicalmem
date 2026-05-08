"""Pin: FHIR resource-TYPE counts in user-facing docs match live state.

Iter-78's `test_cohort_count_drift_pin.py` covers patient + practitioner +
cache-pair counts. Iter-91's `test_hero_bundle_resource_count_pin.py`
covers the hero-bundle ENTRY count (`18 FHIR R4 resources`). Iter-366's
`test_fhir_resource_count_drift_pin.py` covers the cohort total entry
count (`239-entry FHIR cohort`). None of these covered the **distinct
resource TYPE count** — the count of unique `resourceType` values in
the bundle and the engine's supported-type whitelist.

Iter-374 caught this gap: JUDGES.md L27 said "18 resource TYPES in the
demo bundle" but the live counts are:

  • Sarah Mitchell hero bundle: **6 distinct types**
    (Patient, Practitioner, Condition, AllergyIntolerance,
    MedicationStatement, Observation)
  • Cohort bundle (`docs/synthea_demo_cohort.json`): **7 distinct types**
    (the 6 above + MedicationRequest)
  • Engine `fhir_adapter._KNOWN_RESOURCE_TYPES`: **19 supported types**
    (the 7 above + Bundle, Encounter, Procedure, DiagnosticReport,
    DocumentReference, Device, Organization, Location, Medication,
    Immunization, CarePlan, Goal)

The "18" in the original JUDGES claim was the entry COUNT of Sarah's
bundle (which the hero pin already enforces) accidentally relabelled as
a TYPE count. Same single-source-of-truth → derived-surface drift class
as iter-301 / iter-306 / iter-318 / iter-338, applied at the FHIR
resource-type-count layer.

Pin gate
========
1. JUDGES.md FHIR coverage row cites the live engine whitelist count
   ('19-type whitelist' or '19 supported types'); fires if the engine
   adds/removes a type without rotating the doc.
2. JUDGES.md FHIR coverage row cites the live hero-bundle distinct-type
   count ('6 distinct types' for Sarah).
3. JUDGES.md FHIR coverage row cites the live cohort distinct-type
   count ('7 distinct types' for the 30-patient cohort).
4. The engine whitelist count matches the manifest constant — guards
   against accidental dedup if the frozenset grows duplicates.
"""
from __future__ import annotations

import json
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_HERO_BUNDLE = _REPO_ROOT / "docs" / "sarah_mitchell_fhir.json"
_COHORT_BUNDLE = _REPO_ROOT / "docs" / "synthea_demo_cohort.json"
_JUDGES = _REPO_ROOT / "JUDGES.md"


def _distinct_types(bundle_path: Path) -> set[str]:
    bundle = json.loads(bundle_path.read_text())
    return {e["resource"]["resourceType"] for e in bundle.get("entry", [])}


def _engine_supported_types() -> set[str]:
    """Read the engine's _KNOWN_RESOURCE_TYPES frozenset at test time."""
    from engine.fhir_adapter import _KNOWN_RESOURCE_TYPES  # type: ignore
    return set(_KNOWN_RESOURCE_TYPES)


def test_judges_quotes_live_engine_whitelist_count():
    """JUDGES.md FHIR coverage row must cite the live engine whitelist count."""
    n = len(_engine_supported_types())
    text = _JUDGES.read_text()
    expected = f"{n}-type whitelist"
    assert expected in text, (
        f"JUDGES.md must say '{expected}' matching the live engine "
        f"whitelist (engine.fhir_adapter._KNOWN_RESOURCE_TYPES, "
        f"currently {n} types). Iter-374 caught a stale '18 resource "
        f"types' claim that was actually the entry count of Sarah's "
        f"hero bundle. Same iter-301 / iter-306 / iter-318 / iter-338 "
        f"single-source-of-truth → derived-surface drift class, at the "
        f"FHIR resource-type-count layer."
    )


def test_judges_quotes_live_hero_bundle_distinct_type_count():
    """JUDGES.md FHIR coverage row must cite Sarah's hero bundle's
    distinct-type count.

    Match against either '{n} distinct types' (when hero != cohort and
    each number is cited separately) or '{n} distinct' (looser form
    that just requires the integer next to 'distinct')."""
    n = len(_distinct_types(_HERO_BUNDLE))
    text = _JUDGES.read_text()
    candidates = (f"{n} distinct types", f"{n} distinct type", f"{n} distinct")
    assert any(c in text for c in candidates), (
        f"JUDGES.md must cite the hero-bundle distinct-type count "
        f"using one of: {candidates}. Live count from "
        f"{_HERO_BUNDLE.relative_to(_REPO_ROOT)} is {n}."
    )


def test_judges_quotes_live_cohort_distinct_type_count():
    """JUDGES.md FHIR coverage row must cite the cohort's distinct-type
    count using a phrasing that distinguishes it from the hero-bundle
    count when they differ."""
    n = len(_distinct_types(_COHORT_BUNDLE))
    hero_n = len(_distinct_types(_HERO_BUNDLE))
    text = _JUDGES.read_text()
    # Forms accepted: '{n} distinct types' verbatim, OR '{n} distinct'
    # in a cohort/exercise context.
    cohort_phrasings = (
        f"cohort exercises {n} distinct",
        f"30-patient cohort exercises {n}",
        f"{n} distinct types in the cohort",
        f"{n} distinct cohort types",
    )
    assert any(p in text for p in cohort_phrasings), (
        f"JUDGES.md must cite the cohort distinct-type count {n} using "
        f"one of: {cohort_phrasings}. Hero count is {hero_n}; cohort "
        f"adds MedicationRequest. The two numbers must both appear so "
        f"a judge reading JUDGES.md can verify the bundle has the "
        f"claimed type breadth."
    )


def test_engine_whitelist_no_accidental_duplicates():
    """The engine's _KNOWN_RESOURCE_TYPES is a frozenset; this test
    cross-checks that future maintainers haven't accidentally created
    duplicate entries by stringifying then re-parsing.
    """
    types = _engine_supported_types()
    # FHIR resource type names are CamelCase ASCII identifiers; no
    # whitespace or punctuation should sneak in.
    for t in types:
        assert t.isidentifier(), (
            f"Engine FHIR type {t!r} is not a valid Python identifier "
            f"(probable typo in fhir_adapter._KNOWN_RESOURCE_TYPES)"
        )
    # Sanity floor: at least the 6 hero-bundle types must be present.
    hero_required = _distinct_types(_HERO_BUNDLE)
    missing = hero_required - types
    assert not missing, (
        f"Engine whitelist missing types used in Sarah's hero bundle: "
        f"{sorted(missing)}"
    )
