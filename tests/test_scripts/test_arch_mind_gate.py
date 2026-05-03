"""Tests for scripts/run_arch_mind_gate.py.

The gate auto-generates `docs/arch_mind/clinicalmem.fixture.json`
from a fresh ast walk of `engine/*.py`, then runs `arch-mind scan +
rules --mode enforce` against `docs/arch_mind/clinicalmem_rules.mind`.

These tests are skipped when the arch-mind binary is not installed
(CI without the toolchain) so the test suite stays portable.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_SCRIPT = _REPO_ROOT / "scripts" / "run_arch_mind_gate.py"
_FIXTURE = _REPO_ROOT / "docs" / "arch_mind" / "clinicalmem.fixture.json"

_arch_mind_present = (
    shutil.which("arch-mind") is not None
    or Path("~/arch-mind/bin/arch-mind").exists()
)


def _run(*extra: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(_SCRIPT), *extra],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=str(_REPO_ROOT),
    )


def test_fixture_only_mode_writes_summary():
    """`--fixture-only` is portable — it never invokes arch-mind."""
    result = _run("--fixture-only")
    assert result.returncode == 0, (
        f"--fixture-only exit {result.returncode}\n{result.stdout}\n{result.stderr}"
    )
    assert _FIXTURE.exists()
    payload = json.loads(_FIXTURE.read_text())
    summary = payload["_aggregated_for_phase_a"]
    # Spot-check the documented invariants of the engine v0.0.1 baseline:
    assert summary["module_count"] >= 20, "engine/ should ship 20+ modules"
    assert summary["cyclic_edges"] == 0, "no cycles allowed"
    assert summary["intra_package_edges"] == summary["total_edges"], (
        "every internal edge must stay intra-engine"
    )
    assert summary["pure_modules"] == summary["module_count"], (
        "every engine module must be pure on import"
    )
    assert summary["total_mcp_tools"] == 18, (
        "mcp_server.server registers exactly 18 @mcp.tool() decorators"
    )


def test_fixture_only_json_mode_emits_summary_object():
    result = _run("--fixture-only", "--json")
    assert result.returncode == 0
    report = json.loads(result.stdout)
    assert "summary" in report
    assert "fixture_written" in report


@pytest.mark.skipif(
    not _arch_mind_present,
    reason="arch-mind binary not installed; gate tests skip on CI without the toolchain",
)
def test_arch_mind_gate_passes_in_enforce_mode():
    """End-to-end: regenerate fixture, run scan + rules, expect exit 0."""
    result = _run()
    assert result.returncode == 0, (
        f"arch-mind gate failed (exit {result.returncode})\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "OK: every rule passed." in result.stdout
    assert "PASS" in result.stdout
