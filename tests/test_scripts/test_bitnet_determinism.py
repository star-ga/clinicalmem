"""Pytest version of the iter-61 BitNet determinism stress test.

Lighter than `scripts/run_bitnet_determinism_stress.py` (which runs 1200
classifier calls): runs ~10 iterations on a small fixture so it stays
under 1 second in the standard scope, but exercises the same
`severity_name + repro_hash + logits_q16` triple-equality contract.

The full stress run (100 iterations × 12 pairs) is the auditor's
artifact — pinned by `test_full_stress_script_runs_clean` below via
subprocess, capped to the smaller iteration count to keep CI fast.

Pinned invariant: bit-identical output across iterations on the same
machine. Cross-machine determinism is implied by the Q16.16 fixed-point
math (no floating-point ops); the script's docstring notes this and
the pin file references it.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_SCRIPT = _REPO_ROOT / "scripts" / "run_bitnet_determinism_stress.py"

# Small fast fixture for the inline pytest runs.
_FAST_PAIRS = (
    ("warfarin", "ibuprofen"),
    ("ciprofloxacin", "tizanidine"),
    ("clarithromycin", "ergotamine"),
    ("lisinopril", "sacubitril"),
    ("acetaminophen", "lisinopril"),
)


@pytest.fixture(scope="module")
def _bitnet():
    sys.path.insert(0, str(_REPO_ROOT))
    from engine.bitnet_classifier import classify, load_weights  # noqa: PLC0415

    return classify, load_weights()


def test_bitnet_repro_hash_stable_within_process(_bitnet):
    """50 iterations × 5 pairs = 250 calls. Every pair's repro_hash
    must be identical across all 50 iterations."""
    classify, weights = _bitnet
    for pair in _FAST_PAIRS:
        baseline = classify(pair[0], pair[1], weights)
        for _ in range(49):
            r = classify(pair[0], pair[1], weights)
            assert r.repro_hash == baseline.repro_hash
            assert r.severity_name == baseline.severity_name
            assert r.logits_q16 == baseline.logits_q16


def test_bitnet_case_insensitivity(_bitnet):
    """warfarin + ibuprofen and Warfarin + IBUPROFEN must produce the
    same repro_hash (case-insensitive tokenization)."""
    classify, weights = _bitnet
    a = classify("warfarin", "ibuprofen", weights)
    b = classify("Warfarin", "IBUPROFEN", weights)
    assert a.repro_hash == b.repro_hash
    assert a.severity_name == b.severity_name


def test_full_stress_script_runs_clean():
    """Run the auditor's stress script with a reduced iteration count
    (10) so it stays under 2s. Asserts exit 0 (no drift events)."""
    cp = subprocess.run(
        [sys.executable, str(_SCRIPT), "--iterations", "10"],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=str(_REPO_ROOT),
    )
    assert cp.returncode == 0, (
        f"Determinism stress reported drift:\n{cp.stdout}\n{cp.stderr}\n"
        "Investigate any classify() codepath that mutates module-level "
        "state."
    )
    assert "PASS" in cp.stdout
    assert "Drift events: 0" in cp.stdout
