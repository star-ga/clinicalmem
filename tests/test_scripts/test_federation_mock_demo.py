"""
tests/test_scripts/test_federation_mock_demo.py

Test suite for scripts/federation_mock_demo.py.

Three tests:
  1. test_federation_mock_demo_runs_to_completion
     — Run the script as a subprocess; assert exit code 0.

  2. test_federation_mock_demo_invariants_satisfied
     — Parse the captured stdout; confirm all 16 invariants from
       JointMemoryFederation.flow.mind are logged as PASS.

  3. test_federation_mock_demo_phi_blocked
     — Run the script in PHI-test mode; assert the egress path
       quarantines the finding before transport.
"""
from __future__ import annotations

import re
import subprocess
import sys
import os

import pytest

# Absolute path to the demo script — no relative imports.
_SCRIPTS_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "scripts")
)
_DEMO_SCRIPT = os.path.join(_SCRIPTS_DIR, "federation_mock_demo.py")
_REPO_ROOT = os.path.normpath(os.path.join(_SCRIPTS_DIR, ".."))

# Total invariants defined in JointMemoryFederation.flow.mind
_EXPECTED_INVARIANT_COUNT = 16


def _run_demo(*extra_args: str) -> subprocess.CompletedProcess:
    """Run the demo script as a subprocess and capture output."""
    return subprocess.run(
        [sys.executable, _DEMO_SCRIPT, *extra_args],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=_REPO_ROOT,
    )


# ── Test 1: happy-path completion ─────────────────────────────────────────────

def test_federation_mock_demo_runs_to_completion():
    """The demo script must exit with code 0 on the happy path."""
    result = _run_demo()
    assert result.returncode == 0, (
        f"Expected exit code 0, got {result.returncode}.\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    # Sanity: both sites appear in the output
    assert "Mass General" in result.stdout
    assert "Mayo Clinic"  in result.stdout
    assert "FEDERATION DEMO COMPLETE" in result.stdout


# ── Test 2: all 16 invariants logged as PASS ─────────────────────────────────

def test_federation_mock_demo_invariants_satisfied():
    """
    Parse stdout and confirm every invariant from
    JointMemoryFederation.flow.mind is explicitly logged as PASS.

    The script emits lines like:
      ✓ INVARIANT 01 PASS  classify.lane in [...]
    We look for the pattern "INVARIANT NN PASS" for each N in 1..16.
    """
    result = _run_demo()
    assert result.returncode == 0, f"Demo did not exit cleanly:\n{result.stderr}"

    stdout = result.stdout
    # Strip ANSI escape codes before matching
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    clean = ansi_escape.sub("", stdout)

    missing = []
    for i in range(1, _EXPECTED_INVARIANT_COUNT + 1):
        # Match "INVARIANT 01 PASS" or "INVARIANT 1 PASS"
        pattern = rf"INVARIANT\s+0?{i}\s+PASS"
        if not re.search(pattern, clean):
            missing.append(i)

    assert not missing, (
        f"The following invariants were not logged as PASS: {missing}\n"
        f"Full output:\n{clean}"
    )
    # Final summary line
    assert f"All {_EXPECTED_INVARIANT_COUNT} JointMemoryFederation" in clean
    assert "AUDIT CHAIN MATCH" in clean


# ── Test 3: PHI gate quarantines the finding before transport ─────────────────

def test_federation_mock_demo_phi_blocked():
    """
    Run with --phi-test; assert the egress path quarantines the
    finding before reaching the mock transport.

    Expected indicators in stdout:
      - "PHI GATE TRIGGERED"
      - "phi_lane" quarantine event
      - Script still exits 0 (gate is working correctly, not an error)
    """
    result = _run_demo("--phi-test")

    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    clean = ansi_escape.sub("", result.stdout)

    assert result.returncode == 0, (
        f"PHI test should exit 0 (gate passed), got {result.returncode}.\n"
        f"stderr:\n{result.stderr}"
    )
    assert "PHI GATE TRIGGERED" in clean, (
        f"Expected 'PHI GATE TRIGGERED' in output.\nOutput:\n{clean}"
    )
    assert "PHI GATE TEST PASSED" in clean, (
        f"Expected 'PHI GATE TEST PASSED' in output.\nOutput:\n{clean}"
    )
    # Invariant 5 (phi_lane → empty) must fire
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    assert re.search(r"INVARIANT\s+05?\s+PASS", clean), (
        f"Invariant 5 (phi_lane gate) must fire in PHI-test mode.\nOutput:\n{clean}"
    )
    # Transport must NOT be called — no "Record published" line
    assert "Record published to mock transport" not in clean, (
        "Transport should not be called when PHI gate fires.\nOutput:\n{clean}"
    )
