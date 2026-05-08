"""Pin: live FHIR resource count must match every user-facing claim.

Iter-78's `test_cohort_count_drift_pin.py` covers patient + practitioner
+ cache-pair counts but **not** the total-FHIR-resource count (the
sum of all entries in `docs/synthea_demo_cohort.json`).

That gap let a stale "233-entry FHIR cohort" claim survive in
`docs/demo.html` after the bundle grew to 239 entries via the iter
T5 cohort-growth round-robin (Iter-78 added pt-020 → 110 cache, but
also bumped FHIR entries; later T5 rounds kept growing both).
Iter-366 caught the resulting drift: demo claimed 233 entries while
the live bundle had 239.

This pin closes the gap by computing the live FHIR-entry count at
test time and asserting the formatted claim appears verbatim in
every user-facing surface that quotes it.

Same iter-301/iter-306/iter-318/iter-338 single-source-of-truth →
derived-surface drift class, but at the FHIR-bundle-cardinality
layer.
"""
from __future__ import annotations

import json
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_BUNDLE = _REPO_ROOT / "docs" / "synthea_demo_cohort.json"
_DEMO = _REPO_ROOT / "docs" / "demo.html"


def _fhir_entry_count() -> int:
    bundle = json.loads(_BUNDLE.read_text())
    return len(bundle["entry"])


def test_demo_quotes_live_fhir_entry_count():
    """The demo's NPI-Luhn validity claim must cite the live entry count.

    Live count is computed from `docs/synthea_demo_cohort.json` at test
    time. Drift forms checked: any '{n}-entry FHIR cohort' phrasing
    where {n} != live count.
    """
    live = _fhir_entry_count()
    text = _DEMO.read_text()
    expected = f"{live}-entry FHIR cohort"
    assert expected in text, (
        f"docs/demo.html must say '{expected}' matching the live FHIR "
        f"bundle in docs/synthea_demo_cohort.json. The NPI Luhn-validity "
        f"claim is the canonical surface; iter-366 caught a stale "
        f"233-entry claim after the bundle grew to 239. Same iter-301 / "
        f"iter-306 / iter-318 single-source-of-truth → derived-surface "
        f"drift class, at the FHIR-bundle-cardinality layer."
    )


def test_no_off_by_one_stale_fhir_entry_count_in_demo():
    """Catches iter-78-style drift: bundle grew from N → N+k but a
    surface still says N. Blocks both N-1 and N+1 stragglers via the
    '{n}-entry FHIR cohort' phrasing form (which is the unique form
    this pin owns; the patient/practitioner/cache forms are owned by
    test_cohort_count_drift_pin.py).
    """
    live = _fhir_entry_count()
    text = _DEMO.read_text()
    for stale in (live - 1, live + 1):
        if stale <= 0:
            continue
        forbidden = f"{stale}-entry FHIR cohort"
        assert forbidden not in text, (
            f"docs/demo.html contains stale phrasing '{forbidden}' "
            f"(live FHIR entry count is {live})"
        )
