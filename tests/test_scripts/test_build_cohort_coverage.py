"""Tests for scripts/build_cohort_coverage.py.

Two tests:
  1. Generated matrix matches every patient in the bundle.
  2. --check mode exits 0 when the on-disk doc is in sync.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

_REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
_SCRIPT = os.path.join(_REPO_ROOT, "scripts", "build_cohort_coverage.py")
_MATRIX = os.path.join(_REPO_ROOT, "docs", "cohort_coverage_matrix.md")
_BUNDLE = os.path.join(_REPO_ROOT, "docs", "synthea_demo_cohort.json")


def _run(*extra: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, _SCRIPT, *extra],
        capture_output=True,
        text=True,
        timeout=30,
        cwd=_REPO_ROOT,
    )


def test_matrix_lists_every_patient():
    """The matrix must contain a row for every Patient in the bundle."""
    # Regenerate first to ensure the on-disk file is fresh.
    result = _run()
    assert result.returncode == 0, f"build failed: {result.stderr}"

    matrix = open(_MATRIX).read()
    bundle = json.load(open(_BUNDLE))
    patient_ids = [
        e["resource"]["id"]
        for e in bundle["entry"]
        if e["resource"]["resourceType"] == "Patient"
    ]
    assert patient_ids, "bundle has no Patients"
    for pid in patient_ids:
        assert f"`{pid}`" in matrix, (
            f"patient {pid} missing from cohort coverage matrix"
        )


def test_check_mode_passes_after_fresh_build():
    """--check mode must exit 0 when the doc is in sync with sources."""
    # Build first
    build = _run()
    assert build.returncode == 0
    # Then verify --check sees no drift
    check = _run("--check")
    assert check.returncode == 0, (
        f"--check failed after fresh build:\n{check.stdout}\n{check.stderr}"
    )
