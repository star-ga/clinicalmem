"""Pin Sarah Mitchell's patient-header DDI severity badges.

Iter-41 audit: the dashboard's Sarah Mitchell patient header
displayed `2 CRITICAL · 2 HIGH` — but the live pipeline (Layer 1
deterministic table + Layer 2 cache fallback + Layer 4.5 BitNet)
flags exactly **1 serious + 1 moderate** on her medication list:

    warfarin + ibuprofen        : serious   (NSAID + anticoag bleeding)
    metformin + lisinopril      : moderate  (lactic-acidosis surveillance in CKD)

The other med pairs in her chart (amlodipine, amoxicillin,
atorvastatin combinations) classify as "none" or "minor" today.

This test asserts the live counts match the displayed badges so a
future cache addition that surfaces a new Sarah-relevant flag can't
silently leave the badges stale, and a future cache rebalance that
demotes warfarin + ibuprofen can't silently leave a "1 SERIOUS"
claim.
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_DEMO_HTML = _REPO_ROOT / "docs" / "demo.html"

# Sarah Mitchell's medication list — kept in sync with the patient
# scenario card in docs/demo.html. If meds are added or removed, update
# this list AND the badge text in demo.html in the same commit.
_SARAH_MEDS = (
    "warfarin",
    "ibuprofen",
    "metformin",
    "lisinopril",
    "amlodipine",
    "amoxicillin",
    "atorvastatin",
)


def _live_severity_counts() -> Counter:
    sys.path.insert(0, str(_REPO_ROOT))
    from scripts.run_clinical_regression_eval import _pipeline_severity

    counts: Counter = Counter()
    n = len(_SARAH_MEDS)
    for i in range(n):
        for j in range(i + 1, n):
            v = _pipeline_severity(_SARAH_MEDS[i], _SARAH_MEDS[j]).lower()
            if v not in ("none", "minor"):
                counts[v] += 1
    return counts


def test_sarah_badges_match_live_pipeline():
    counts = _live_severity_counts()
    html = _DEMO_HTML.read_text()
    # Pinned counts (iter 41).
    assert counts.get("contraindicated", 0) == 0, (
        f"Cache surfaced a new contraindicated pair on Sarah's chart: "
        f"{counts}. Update the patient-header badge in docs/demo.html."
    )
    assert counts.get("serious", 0) == 1, (
        f"serious count drift: live={counts.get('serious', 0)}, pinned=1"
    )
    assert counts.get("moderate", 0) == 1, (
        f"moderate count drift: live={counts.get('moderate', 0)}, pinned=1"
    )
    # Affirmative: badges in demo.html match.
    assert "1 SERIOUS" in html, (
        "patient header must show '1 SERIOUS' badge (warfarin + ibuprofen)"
    )
    assert "1 MODERATE" in html, (
        "patient header must show '1 MODERATE' badge (metformin + lisinopril)"
    )


def test_no_stale_inflated_badges_remain():
    """Old over-stated counts must not linger after rotation."""
    historical = (
        "2 CRITICAL",
        "2 HIGH",
        "3 CRITICAL",
        "3 HIGH",
    )
    html = _DEMO_HTML.read_text()
    for stale in historical:
        assert stale not in html, (
            f"Stale inflated badge {stale!r} still in docs/demo.html — "
            f"replace with the live count from the pipeline."
        )
