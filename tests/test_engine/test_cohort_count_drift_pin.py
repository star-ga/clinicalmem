"""Pin: live cohort + cache counts must match every user-facing claim.

Iter-78 cohort growth (pt-020 / 110 → 111 cache pairs / 19 → 20 patients
/ 36 → 37 NPIs) had to manually rotate ~12 hard-coded numbers across
demo.html, JUDGES.md, irb_exemption.md, cohort_coverage_matrix.md, plus
regenerate the manifest, calibration, and confusion-matrix artifacts.
A future iter that adds the next patient and forgets one of these
surfaces would silently ship stale claims to judges.

These tests close that drift class: the live cohort + cache counts
are computed at test time and asserted to appear (literally, as
formatted strings) in every user-facing doc that quotes them.

Pin checklist:
  • patient count: demo trust-bar "N patients", JUDGES "N synthetic
    patients", IRB "All N patients are entirely fictional"
  • practitioner / NPI count: JUDGES "· N NPIs"
  • cache pair count: demo "PCCP cohort N pairs", JUDGES "(N entries)",
    "(N-pair", calibration artifact "total_pairs"
"""
from __future__ import annotations

import json
import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_BUNDLE = _REPO_ROOT / "docs" / "synthea_demo_cohort.json"
_CACHE = _REPO_ROOT / "docs" / "openevidence_cache.json"
_DEMO = _REPO_ROOT / "docs" / "demo.html"
_JUDGES = _REPO_ROOT / "JUDGES.md"
_IRB = _REPO_ROOT / "docs" / "irb_exemption.md"
_COHORT_MATRIX = _REPO_ROOT / "docs" / "cohort_coverage_matrix.md"
_CALIBRATION = _REPO_ROOT / "docs" / "bitnet_calibration.json"


def _live_counts() -> dict[str, int]:
    bundle = json.loads(_BUNDLE.read_text())
    cache = json.loads(_CACHE.read_text())
    patients = sum(
        1 for e in bundle["entry"]
        if e["resource"].get("resourceType") == "Patient"
    )
    practitioners = sum(
        1 for e in bundle["entry"]
        if e["resource"].get("resourceType") == "Practitioner"
    )
    return {
        "patients": patients,
        "practitioners": practitioners,
        "cache_entries": len(cache),
    }


def test_patient_count_pinned_in_demo_judges_and_irb():
    counts = _live_counts()
    n = counts["patients"]
    demo = _DEMO.read_text()
    judges = _JUDGES.read_text()
    irb = _IRB.read_text()

    # demo.html provenance footer: "FHIR cohort  N patients"
    assert f">{n} patients</span>" in demo, (
        f"docs/demo.html provenance footer must say '{n} patients' "
        f"matching the live synthea bundle"
    )
    # JUDGES audit-trail row uses "N synthetic patients"
    assert f"{n} synthetic patients" in judges, (
        f"JUDGES.md audit-trail must say '{n} synthetic patients' "
        f"matching the live synthea bundle"
    )
    # IRB exemption locks the count: "All N patients are entirely fictional"
    assert f"All {n} patients are entirely fictional" in irb, (
        f"docs/irb_exemption.md must say 'All {n} patients are "
        f"entirely fictional' matching the live synthea bundle"
    )


def test_practitioner_count_pinned_in_judges():
    counts = _live_counts()
    n = counts["practitioners"]
    judges = _JUDGES.read_text()
    # JUDGES row format: "N synthetic patients · M NPIs"
    assert f"· {n} NPIs" in judges, (
        f"JUDGES.md audit-trail must say '· {n} NPIs' matching "
        f"the live synthea bundle's Practitioner resources"
    )


def test_cache_pair_count_pinned_in_demo_and_judges():
    counts = _live_counts()
    n = counts["cache_entries"]
    demo = _DEMO.read_text()
    judges = _JUDGES.read_text()

    # demo.html provenance footer: "PCCP cohort  N pairs"
    assert f">{n} pairs</span>" in demo, (
        f"docs/demo.html provenance footer must say '{n} pairs' "
        f"matching the live OpenEvidence cache size"
    )
    # JUDGES.md uses "(N entries)" and "N-pair" forms in multiple rows
    assert f"({n} entries)" in judges, (
        f"JUDGES.md must say '({n} entries)' for the cache size"
    )
    assert f"{n}-pair" in judges, (
        f"JUDGES.md must use '{n}-pair' formulation in audit-trail "
        f"references to the live cache"
    )


def test_calibration_total_pairs_matches_live_cache():
    counts = _live_counts()
    cal = json.loads(_CALIBRATION.read_text())
    assert cal["total_pairs"] == counts["cache_entries"], (
        f"docs/bitnet_calibration.json total_pairs={cal['total_pairs']} "
        f"but live cache has {counts['cache_entries']} entries — "
        f"calibration is stale, regenerate via "
        f"scripts/build_bitnet_calibration.py"
    )


def test_cohort_matrix_header_quotes_live_counts():
    counts = _live_counts()
    text = _COHORT_MATRIX.read_text()
    # Header line: "...({patients} patients) × ...({cache_entries} pairs)"
    pattern = rf"\({counts['patients']} patients\).*\({counts['cache_entries']} pairs\)"
    assert re.search(pattern, text, re.DOTALL), (
        f"docs/cohort_coverage_matrix.md header must quote "
        f"({counts['patients']} patients) and ({counts['cache_entries']} "
        f"pairs); regenerate via scripts/build_cohort_coverage.py"
    )


def test_evidence_url_count_in_judges_matches_live_cache():
    """Iter-78 cohort growth + iter-81 audit caught a drift: when pt-020
    landed (110 → 111 cache entries, +3 evidence URLs), JUDGES.md was
    rotated to '229 URLs across 111 entries, avg 2.06' but the live
    count was 232 / 2.09. T3 round 16 corrected. Pinning the live
    URL-count claim against the JUDGES prose closes that drift class."""
    cache = json.loads(_CACHE.read_text())
    total_urls = sum(len(e.get("evidence_urls", [])) for e in cache)
    n = len(cache)
    avg = total_urls / max(n, 1)
    judges = _JUDGES.read_text()
    expected = f"{total_urls} URLs across {n} entries"
    assert expected in judges, (
        f"JUDGES.md must quote live URL count: {expected!r} "
        f"(found something else; live avg = {avg:.2f} URLs/pair)"
    )
    expected_avg = f"avg {avg:.2f} URLs/pair"
    assert expected_avg in judges, (
        f"JUDGES.md must quote live URL/pair average {expected_avg!r}"
    )


def test_no_off_by_one_stale_cache_count_in_user_docs():
    """Catches the iter-78-style drift: cohort grew to 111 entries but
    a doc still says 110. Blocks both N-1 and N+1 stragglers."""
    counts = _live_counts()
    n = counts["cache_entries"]
    stale_off_by_one = (n - 1, n + 1)
    surfaces = (_DEMO, _JUDGES)
    for stale in stale_off_by_one:
        if stale <= 0:
            continue
        # Match exactly "stale-pair" or "(stale entries)" or "(stale pairs)".
        # The drift pin already covers test counts; this covers cache size.
        forbidden_forms = (
            f"{stale}-pair",
            f"({stale} entries)",
            f"({stale} pairs)",
            f">{stale} pairs</span>",
        )
        for path in surfaces:
            text = path.read_text()
            for form in forbidden_forms:
                assert form not in text, (
                    f"{path.relative_to(_REPO_ROOT)} contains stale "
                    f"cache-count form {form!r} (live count is {n})"
                )
