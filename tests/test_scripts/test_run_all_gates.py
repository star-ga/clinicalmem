"""Tests for scripts/run_all_gates.py.

Verifies the unified eval driver invokes every individual gate and
reports a coherent PASS/FAIL summary. The arch-mind sub-gate is
allowed to be `skipped` when the binary isn't installed (CI-friendly).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

_REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
_SCRIPT = os.path.join(_REPO_ROOT, "scripts", "run_all_gates.py")


def _run(*extra: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, _SCRIPT, *extra],
        capture_output=True,
        text=True,
        timeout=180,
        cwd=_REPO_ROOT,
    )


def test_all_gates_passes_when_skipping_optional_arch_mind():
    """The mandatory four (PCCP / precision / federation / audit-replay) must pass."""
    result = _run("--skip-arch-mind")
    assert result.returncode == 0, (
        f"Expected exit 0; got {result.returncode}.\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "ALL-GATES RESULT: PASS" in result.stdout
    # Four passes + one skipped (PCCP / precision / federation / audit-replay + arch-mind skipped)
    assert "4 pass" in result.stdout
    assert "1 skipped" in result.stdout


def test_all_gates_json_output_shape():
    result = _run("--json", "--skip-arch-mind")
    assert result.returncode == 0
    report = json.loads(result.stdout)
    assert report["ok"] is True
    assert report["fail"] == 0
    assert report["pass"] == 4
    assert report["skipped"] == 1
    assert isinstance(report["gates"], list)
    assert len(report["gates"]) == 5
    # Each gate has the required keys
    for gate in report["gates"]:
        assert {"name", "status", "duration_ms", "summary"}.issubset(gate.keys())
    # Audit-replay must be one of the gates
    assert any(
        "audit" in g["name"].lower() or "replay" in g["name"].lower()
        for g in report["gates"]
    ), "Audit-replay gate must appear in run_all_gates output"


def test_skip_federation_still_passes():
    result = _run("--skip-federation", "--skip-arch-mind")
    assert result.returncode == 0
    assert "ALL-GATES RESULT: PASS" in result.stdout
    # Three pass (PCCP + precision + audit-replay) + two skipped
    assert "3 pass" in result.stdout
    assert "2 skipped" in result.stdout
