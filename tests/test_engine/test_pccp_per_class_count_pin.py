"""Pin the per-class PCCP cohort counts displayed on the dashboard.

The BitNet sparkline on `docs/demo.html` shows per-class recall
labels of the form `100% · N / N` for each severity class. Those
N values are the count of pairs in the live OpenEvidence cache for
each class.

Iter-30/31 audit caught FOUR such labels going stale across cohort
growth (iter-19 added pt-013, iter-9 added 5 cache entries before
that). This test pins the per-class counts so any future cache
growth fails the gate until the labels are updated.

Pinned values (iter 72):

  contraindicated: 20 (was 14 → 15 → 16 → 17 → 18 → 19 → 20 across cohort growth)
  serious        : 68 (was 64 → 66 → 67 → 68 — iter 78 added azithromycin+warfarin)
  moderate       : 22 (was 20 → 22 — sparkline lagged)
  major          : 1  (iter 39: tamoxifen + paroxetine — first major)
"""
from __future__ import annotations

import json
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_CACHE = _REPO_ROOT / "docs" / "openevidence_cache.json"
_DEMO_HTML = _REPO_ROOT / "docs" / "demo.html"

_EXPECTED_CONTRAINDICATED = 37
_EXPECTED_SERIOUS = 69
_EXPECTED_MODERATE = 22
_EXPECTED_MAJOR = 4  # iter 109: 4th major (dronedarone + dabigatran) joined iter-93 voriconazole+tacrolimus, iter-83 clarithromycin+digoxin, iter-39 tamoxifen+paroxetine


def _live_counts() -> dict[str, int]:
    cache = json.loads(_CACHE.read_text())
    counts: dict[str, int] = {}
    for it in cache:
        sev = (it.get("severity") or "").lower()
        counts[sev] = counts.get(sev, 0) + 1
    return counts


def test_live_per_class_counts_match_pinned():
    counts = _live_counts()
    assert counts.get("contraindicated", 0) == _EXPECTED_CONTRAINDICATED, (
        f"contraindicated count drifted: live={counts.get('contraindicated', 0)}, "
        f"pinned={_EXPECTED_CONTRAINDICATED}. Update both this constant AND "
        f"the demo.html sparkline label '100% · N / N' for the contraindicated bar."
    )
    assert counts.get("serious", 0) == _EXPECTED_SERIOUS, (
        f"serious count drifted: live={counts.get('serious', 0)}, "
        f"pinned={_EXPECTED_SERIOUS}"
    )
    assert counts.get("moderate", 0) == _EXPECTED_MODERATE, (
        f"moderate count drifted: live={counts.get('moderate', 0)}, "
        f"pinned={_EXPECTED_MODERATE}"
    )
    assert counts.get("major", 0) == _EXPECTED_MAJOR, (
        f"major count drifted: live={counts.get('major', 0)}, "
        f"pinned={_EXPECTED_MAJOR}"
    )


def test_dashboard_sparkline_displays_pinned_counts():
    html = _DEMO_HTML.read_text()
    expected = (
        f"100% · {_EXPECTED_CONTRAINDICATED} / {_EXPECTED_CONTRAINDICATED}",
        f"100% · {_EXPECTED_SERIOUS} / {_EXPECTED_SERIOUS}",
        f"100% · {_EXPECTED_MODERATE} / {_EXPECTED_MODERATE}",
    )
    for label in expected:
        assert label in html, (
            f"docs/demo.html BitNet sparkline must display {label!r} "
            f"(matches the live cache count)"
        )


def test_no_stale_per_class_counts_remain():
    """Old per-class counts must not linger after cohort growth.

    Iter 60 NOTE: removed the "moderate 100% · 20 / 20" historical from
    this list. As of iter 60 the *contraindicated* class hit 20, so the
    label "100% · 20 / 20" is now the LIVE contraindicated count and
    blocking it would fire on legitimate copy. Past historicals are kept
    because their values haven't been re-claimed by another class.
    """
    historical = (
        ("contraindicated", "100% · 14 / 14"),
        ("contraindicated", "100% · 15 / 15"),
        ("contraindicated", "100% · 16 / 16"),
        ("serious", "100% · 64 / 64"),
    )
    html = _DEMO_HTML.read_text()
    for label, stale in historical:
        assert stale not in html, (
            f"Stale {label} sparkline label {stale!r} still in docs/demo.html "
            f"— replace with the live count."
        )


def test_no_vacuous_major_recall_claim():
    """The cohort has 0 'major' class entries; any "major 100%" claim is vacuous.

    Iteration 35 discovery: the eval reports `recall_major: 1.000` because
    `class_counts.get("major", {"total": 0})["total"] == 0` triggers the
    `if bucket["total"] == 0: return 1.0` short-circuit in
    `scripts/run_clinical_regression_eval.py`. That's mathematically
    convention-driven, not empirical evidence. Any dashboard claim that
    asserts "100% recall on major" is misleading until a major-class
    pair is added to the cohort.
    """
    cache = json.loads(_CACHE.read_text())
    major_count = sum(1 for it in cache if it.get("severity") == "major")
    if major_count > 0:
        # Once major-class pairs are added, this test becomes a no-op.
        return

    html = _DEMO_HTML.read_text()
    # Specific stale claims to catch
    forbidden = (
        "major 100%",
        "100% · all observed",  # iter-32-vintage major bar label
    )
    for snippet in forbidden:
        assert snippet not in html, (
            f"docs/demo.html contains the misleading {snippet!r} for the "
            f"major class while the cohort has 0 major-class pairs. Either "
            f"add major examples to docs/openevidence_cache.json or remove "
            f"the inflated claim."
        )

    # Affirmative check: the honest "n/a · 0 in cohort" label is shown.
    assert "n/a · 0 in cohort" in html, (
        "docs/demo.html sparkline must label the empty major class as "
        "'n/a · 0 in cohort' rather than implying 100% recall."
    )
