"""Pin the README's Layer 4.5 BitNet recall + precision claims to live calibration.

Iteration 112 forensic: README.md L130 + L134 contained the
BitNet-on-contraindicated numbers from iter-99 era (21-pair cache,
6/6 = 100% precision, 28.6% recall). iter 104 grew the contra cohort
22, rotated demo + JUDGES + 5 pin tests, but the README was missed —
8 iterations of stale numbers shipped before this catch.

Same silent-drift class as:
  iter 89  — SVG row comments
  iter 91  — FHIR genesis block
  iter 101 — BitNet body recall paragraph
  iter 102 — README "85.7% precision on contraindicated" mislabel
  iter 107 — JUDGES manifest description (4 gates → 5)

README.md is the FIRST doc a developer reads on the GitHub repo page
(it renders by default). Drift here is judge-visible AND
package-visible: PyPI / pip / cargo / npm registries display the
README. Stale numbers in README erode trust faster than any other
artifact.

This pin reads `docs/bitnet_calibration.json` at test time and
asserts that the README numbers match the live counts:

  contra_total       = ground_truth_total of contraindicated class
  contra_tp          = correctly classified
  contra_recall_pct  = TP / total * 100, rounded to 1 dp

The README must contain the literal phrasings:
  "{contra_tp} / {contra_tp} of its contraindicated predictions"
  "{contra_total}-pair cache"
  "{contra_recall_pct}% recall"

Future cohort growth that bumps any of these numbers without rotating
the README fails this gate immediately.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_README = _REPO_ROOT / "README.md"
_CALIB = _REPO_ROOT / "docs" / "bitnet_calibration.json"


def _live_contra_metrics() -> dict[str, float]:
    calib = json.loads(_CALIB.read_text())
    contra = [
        e for e in calib["entries"] if e.get("ground_truth") == "contraindicated"
    ]
    pred_contra = [
        e for e in calib["entries"] if e.get("predicted") == "contraindicated"
    ]
    contra_total = len(contra)
    contra_tp = sum(1 for e in contra if e.get("correct"))
    pred_total = len(pred_contra)
    return {
        "total": contra_total,
        "tp": contra_tp,
        "pred_total": pred_total,
        "recall_pct": round(contra_tp / contra_total * 100, 1),
    }


def test_readme_contra_count_matches_live():
    """README's `N-pair cache` for the contra class must match live."""
    metrics = _live_contra_metrics()
    expected = f"{metrics['total']}-pair"
    text = _README.read_text()
    assert expected in text, (
        f"README.md must reference the live contraindicated cohort size "
        f"{metrics['total']!r} (as `{expected} cache`). The README contains "
        f"a stale count. Update README.md L130 + L134 to match the live "
        f"`docs/bitnet_calibration.json` numbers."
    )


def test_readme_contra_precision_matches_live():
    """README's `N / N of its contraindicated predictions` must match live.

    Live precision is always TP/pred_total = X/X (precision = 1.0 by
    design — Layer 4.5 is a high-precision veto, not a recall layer).
    The phrasing 'M / M of its contraindicated predictions' must match
    the live TP count.
    """
    metrics = _live_contra_metrics()
    tp = metrics["tp"]
    text = _README.read_text()
    # Match `7 / 7` AND `7/7` — both forms appear in README prose.
    spaced = f"{tp} / {tp} of its contraindicated predictions"
    compact = f"{tp}/{tp} of its contraindicated predictions"
    assert (spaced in text) or (compact in text), (
        f"README.md must reference the live contra-class TP count "
        f"({tp}) in the precision phrasing 'N / N of its contraindicated "
        f"predictions' (or compact form 'N/N'). The README contains a "
        f"stale TP count. Live: TP={tp}, total={metrics['total']}, "
        f"recall={metrics['recall_pct']}%."
    )


def test_readme_contra_recall_matches_live():
    """README's `N% recall` must match live recall (rounded to 1 dp)."""
    metrics = _live_contra_metrics()
    rec = metrics["recall_pct"]
    text = _README.read_text()
    # `31.8%`, `28.6%`, etc. — must contain the live value
    expected = f"{rec}%"
    # Anchored: must be near "recall" within ~80 chars to avoid matching
    # an unrelated 31.8% somewhere.
    pattern = re.compile(
        rf"(?s)recall.{{0,80}}{re.escape(expected)}|{re.escape(expected)}.{{0,80}}recall"
    )
    assert pattern.search(text), (
        f"README.md must reference the live BitNet contraindicated "
        f"recall ({expected}) within ~80 chars of the word 'recall'. "
        f"Live: TP={metrics['tp']}/{metrics['total']} = {expected}. "
        f"The README contains a stale recall percentage. Update "
        f"README.md L130 + L134."
    )


def test_no_historical_bitnet_recall_phrases_in_readme():
    """Block specific historical phrasings from re-appearing in README.

    These are the iter-89 SVG / iter-91 FHIR / iter-101 BitNet body /
    iter-102 conflated-precision drift class: prevent past values
    from haunting future revisions.
    """
    historical = (
        # iter-99 era (pre-iter-104 sumatriptan+phenelzine cohort growth)
        "21-pair cache",
        "6 / 6 of its contraindicated predictions",
        "6/6 of its contraindicated predictions",
        # iter-99 recall (28.6% = 6/21)
        # NOTE: do not block the bare "28.6%" — could match other places.
        # Block it only with adjacent recall+contra context.
        "28.6% recall",
        # iter-19 era (pre-iter-78 azithromycin+warfarin)
        # 6/6 was the same TP, but cohort was 14-pair and recall was 42.9%
        "14-pair cache",
        "42.9% recall",
        # iter-19 era second drift point
        "20-pair cache",
        "30.0% recall",
        "30 / 20",
    )
    text = _README.read_text()
    for stale in historical:
        assert stale not in text, (
            f"README.md contains the historical Layer 4.5 BitNet phrasing "
            f"{stale!r} which is no longer correct. Update to live values "
            f"from `docs/bitnet_calibration.json`."
        )
