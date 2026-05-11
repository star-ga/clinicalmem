"""Pin: BitNet body section copy stays in sync with live confusion matrix.

Iter 101 (T2 round 20): the BitNet section paragraph contains two
hard-coded numbers that judges read first:

    "on the live OpenEvidence cache (n=N contraindicated) the
    deployment-side numbers are precision 100% / recall R%"

These were 20 / 30.0% at iter-72 baseline. Iter-99 cohort growth
(pt-024) bumped contra to 21, recall to 28.6%, but this line was not
rotated alongside the sparkline / confusion-matrix surfaces. Same
silent-drift class as iter-89 caught for the SVG row comments and
iter-91 caught for the FHIR genesis block.

This pin reads the live `docs/bitnet_confusion_matrix.json` and
asserts both numbers in the BitNet body match. Future cohort growth
that rotates the matrix without rotating the body copy fails the
gate immediately.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_DEMO = _REPO_ROOT / "docs" / "demo.html"
_MATRIX = _REPO_ROOT / "docs" / "bitnet_confusion_matrix.json"

_CONTRA_TOTAL_PATTERN = re.compile(
    r'id="bitnet-body-contra-total"[^>]*>n=(\d+)'
)
_CONTRA_RECALL_PATTERN = re.compile(
    r'id="bitnet-body-contra-recall"[^>]*>([\d.]+)%</span>'
)


def _live_contra_stats():
    matrix = json.loads(_MATRIX.read_text())
    pc = matrix["per_class"]["contraindicated"]
    return pc["ground_truth_total"], pc["recall"]


def test_bitnet_body_contra_total_matches_live_matrix():
    html = _DEMO.read_text()
    m = _CONTRA_TOTAL_PATTERN.search(html)
    assert m is not None, (
        "demo.html BitNet body must contain "
        "<span id='bitnet-body-contra-total'>n=N contraindicated</span>; "
        "pattern not found."
    )
    displayed = int(m.group(1))
    live_total, _ = _live_contra_stats()
    assert displayed == live_total, (
        f"BitNet body says n={displayed} contraindicated but live "
        f"confusion matrix has {live_total}. Re-run "
        f"`scripts/build_bitnet_confusion_matrix.py` then update "
        f"the body span value."
    )


def test_bitnet_body_contra_recall_matches_live_matrix():
    html = _DEMO.read_text()
    m = _CONTRA_RECALL_PATTERN.search(html)
    assert m is not None, (
        "demo.html BitNet body must contain "
        "<span id='bitnet-body-contra-recall'>R%</span>; pattern not found."
    )
    displayed = float(m.group(1))
    _, live_recall = _live_contra_stats()
    expected = round(live_recall * 100, 1)
    assert abs(displayed - expected) < 0.1, (
        f"BitNet body says recall {displayed}% but live confusion "
        f"matrix has {expected}% ({live_recall:.4f}). "
        f"Update the body span value to match."
    )


def test_no_stale_recall_30_lurking_after_growth():
    """Iter-99 contra growth bumped denominator 20 → 21. The
    historical 'recall 30.0%' must not linger in plain text after
    the cohort exceeded 20 contraindicated entries."""
    matrix = json.loads(_MATRIX.read_text())
    contra_total = matrix["per_class"]["contraindicated"]["ground_truth_total"]
    if contra_total == 20:
        # If a future restructure rolls back to 20 contras, the
        # historical stops being historical.
        return
    html = _DEMO.read_text()
    forbidden = (
        "recall 30.0%</strong>",  # iter-72 wording
        "n=20 contraindicated",   # iter-72 + iter-93 wording
    )
    for snippet in forbidden:
        assert snippet not in html, (
            f"Stale snippet {snippet!r} in docs/demo.html — cohort grew "
            f"past 20 contraindicated (now {contra_total})."
        )
