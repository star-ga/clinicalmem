"""Tests for scripts/run_negative_control_eval.py.

Three tests:
  1. test_negative_control_runs_to_completion
     — Run the script as a subprocess; assert exit code 0 (no false
       positives expected on the curated cohort).
  2. test_negative_control_json_output_shape
     — `--json` mode emits a valid JSON report with the documented keys.
  3. test_negative_control_pipeline_returns_none_for_every_pair
     — Direct in-process call: every pair classifies as "none" or
       "minor" so precision is 1.0.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

_SCRIPTS_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "scripts")
)
_SCRIPT = os.path.join(_SCRIPTS_DIR, "run_negative_control_eval.py")
_REPO_ROOT = os.path.normpath(os.path.join(_SCRIPTS_DIR, ".."))


def _run(*extra: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, _SCRIPT, *extra],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=_REPO_ROOT,
    )


def test_negative_control_runs_to_completion():
    result = _run()
    assert result.returncode == 0, (
        f"Expected exit code 0, got {result.returncode}.\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "PRECISION GATE: PASS" in result.stdout
    assert "False positives : 0" in result.stdout


def test_negative_control_json_output_shape():
    result = _run("--json")
    assert result.returncode == 0
    report = json.loads(result.stdout)
    expected_keys = {
        "total_pairs",
        "false_positives",
        "precision",
        "fpr",
        "mean_latency_ms",
        "per_pair_results",
    }
    assert expected_keys.issubset(report.keys())
    assert report["false_positives"] == 0
    assert report["precision"] == 1.0
    assert report["fpr"] == 0.0
    assert report["total_pairs"] == len(report["per_pair_results"])


def test_negative_control_pipeline_returns_none_or_minor_for_every_pair():
    """In-process import: ensures every cohort pair is classified ≤ minor."""
    sys.path.insert(0, _REPO_ROOT)
    from scripts.run_negative_control_eval import _run as run_inproc  # noqa: E402

    report = run_inproc()
    assert report["false_positives"] == 0
    for pair in report["per_pair_results"]:
        assert pair["pipeline_verdict"].lower() in {"none", "minor"}, (
            f"{pair['drug_a']} + {pair['drug_b']} unexpectedly returned "
            f"{pair['pipeline_verdict']!r}"
        )
