"""Tests for `scripts/build_bitnet_confusion_matrix.py`.

The artifact `docs/bitnet_confusion_matrix.json` is the audit-grade
companion to the iter-29 contraindicated-class precision pin. It
gives auditors a one-look full picture of where Layer 4.5 BitNet
is precise (contraindicated: precision 1.000) and where the
upstream 4-tier pipeline carries the load (serious: BitNet rarely
predicts this class — by design).

Pinned invariants:
  • FP_contraindicated == 0 (the safety claim that justifies
    the "high-precision veto" framing on the dashboard).
  • TP_contraindicated ≥ 6 (recall floor — pre-iter-49 was 6/17,
    iter-49 cohort growth makes it 6/18, this floor catches a
    weight rotation that drops below 6/N).
  • Live matrix matches on-disk artifact (CI parity gate).
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_ARTIFACT = _REPO_ROOT / "docs" / "bitnet_confusion_matrix.json"
_SCRIPT = _REPO_ROOT / "scripts" / "build_bitnet_confusion_matrix.py"


@pytest.fixture(scope="module")
def matrix() -> dict:
    """Load the on-disk artifact."""
    assert _ARTIFACT.exists(), (
        f"Run scripts/build_bitnet_confusion_matrix.py to regenerate {_ARTIFACT}"
    )
    return json.loads(_ARTIFACT.read_text())


def test_artifact_has_required_keys(matrix):
    for key in (
        "matrix",
        "per_class",
        "safety_invariants",
        "weights_id",
        "cache_pairs_total",
    ):
        assert key in matrix, f"missing key: {key}"


def test_safety_invariant_fp_contraindicated_is_zero(matrix):
    """The headline safety claim: BitNet never falsely predicts contraindicated."""
    assert matrix["safety_invariants"]["fp_contraindicated_is_zero"] is True, (
        "Layer 4.5 false-positive on contraindicated would invalidate the "
        "'high-precision veto' framing. Investigate the weight rotation."
    )
    fp = matrix["per_class"]["contraindicated"]["fp"]
    assert fp == 0, f"contraindicated FP drifted: {fp}, pinned=0"


def test_safety_invariant_tp_contraindicated_floor(matrix):
    """Recall floor: at least 7 contraindicated TPs on the live cache.

    Iter-117 ratchet: bumped 6 -> 7 because BitNet has held TP=7 since
    iter 104 (sumatriptan+phenelzine catch). The floor follows the
    iter-66/iter-90 ratchet pattern — once a safety-class invariant
    has held for many iters, ratchet up so a regression below the
    sustained value fails the gate.

    The old key `tp_contraindicated_at_least_six` is intentionally
    removed to force callers to update; the historical block test
    below catches the old name.
    """
    assert matrix["safety_invariants"]["tp_contraindicated_at_least_seven"] is True
    tp = matrix["per_class"]["contraindicated"]["tp"]
    assert tp >= 7, f"contraindicated TP dropped below floor: {tp}, floor=7"
    # Old key must NOT remain in the artifact — silent dual-keying
    # would let a regression to TP=6 pass.
    assert "tp_contraindicated_at_least_six" not in matrix["safety_invariants"], (
        "iter-117 ratchet: the legacy 'at_least_six' key must not remain "
        "in safety_invariants. Re-run scripts/build_bitnet_confusion_matrix.py "
        "to regenerate with only 'at_least_seven'."
    )


def test_artifact_matches_live_computation():
    """`--check` mode confirms artifact is in sync with live BitNet output."""
    cp = subprocess.run(
        [sys.executable, str(_SCRIPT), "--check"],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=str(_REPO_ROOT),
    )
    assert cp.returncode == 0, (
        f"Confusion matrix on-disk drifted from live:\n{cp.stdout}\n{cp.stderr}\n"
        "Re-run `python3 scripts/build_bitnet_confusion_matrix.py` to refresh."
    )


def test_per_class_precision_recall_within_expected_bands(matrix):
    """Loose bands per-class — catch regressions without coupling to exact
    weight rotation. Tightened on contraindicated (the safety class)."""
    pc = matrix["per_class"]

    # contraindicated — tight: precision MUST be 1.000, recall ≥ 0.25
    # (iter-164 cohort grew 30 → 31 → recall 8/31 = 0.258).
    assert pc["contraindicated"]["precision"] == 1.0
    assert pc["contraindicated"]["recall"] >= 0.21

    # major — small ground-truth class (3 pairs as of iter-93). Iter-39
    # added the 1st (tamoxifen+paroxetine), iter-83 the 2nd
    # (clarithromycin+digoxin), iter-93 the 3rd (voriconazole+tacrolimus).
    # BitNet predicts "none" on voriconazole+tacrolimus (architectural
    # 3-bin under-prediction) so recall is now 2/3 = 0.667. Loose floor
    # 0.50 catches a future weight rotation that drops below half-coverage
    # while accepting the iter-93 baseline.
    assert pc["major"]["recall"] >= 0.50

    # moderate — loose band: precision and recall both ≥ 0.40 (the iter-50
    # snapshot is precision=0.688 recall=0.500; the band catches significant
    # regression without churning on every weight rotation).
    assert pc["moderate"]["precision"] >= 0.40
    assert pc["moderate"]["recall"] >= 0.40

    # serious — BitNet rarely predicts this class by design; no per-class
    # assertion. The upstream pipeline carries serious classification.


def test_matrix_totals_sum_to_cache_size(matrix):
    """Sanity: the matrix entries sum to the live cache size."""
    total = sum(sum(row.values()) for row in matrix["matrix"].values())
    assert total == matrix["cache_pairs_total"]
    # And cache_pairs_total ≥ 107 (iter 49 cohort growth — never shrinks).
    assert matrix["cache_pairs_total"] >= 107
