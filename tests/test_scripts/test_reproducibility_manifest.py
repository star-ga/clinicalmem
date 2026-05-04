"""Tests for `scripts/build_reproducibility_manifest.py`.

The reproducibility manifest is the **single audit artifact** an FDA
SaMD reviewer drops into a compliance review to verify every load-
bearing deterministic surface at once: cache + weights + confusion
matrix + cohort + flow plan_hashes + gate verdicts + test count + git
HEAD.

These tests pin the manifest's structural invariants:
  - All required artifact SHAs are present and well-formed (64-char hex).
  - The flow plan_hashes map exactly matches `flows/*.flow.mind` files.
  - Safety invariants from the BitNet confusion matrix flow through.
  - All 4 gate verdicts are PASS.
  - test_count ≥ a fixed floor.
  - The on-disk manifest matches the live computation (parity check).
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_MANIFEST = _REPO_ROOT / "docs" / "reproducibility_manifest.json"
_SCRIPT = _REPO_ROOT / "scripts" / "build_reproducibility_manifest.py"
_FLOWS_DIR = _REPO_ROOT / "flows"

_HEX64 = re.compile(r"^[0-9a-f]{64}$")


@pytest.fixture(scope="module")
def manifest() -> dict:
    assert _MANIFEST.exists(), (
        f"Run scripts/build_reproducibility_manifest.py to regenerate "
        f"{_MANIFEST}"
    )
    return json.loads(_MANIFEST.read_text())


def test_manifest_has_required_top_level_keys(manifest):
    for key in (
        "@context",
        "@type",
        "name",
        "version",
        "dateCreated",
        "license",
        "git_head",
        "artifacts",
        "gates",
        "test_count",
        "audit_replay_hint",
    ):
        assert key in manifest, f"missing top-level key: {key}"


def test_manifest_artifact_shas_are_well_formed(manifest):
    """Every artifact SHA-256 must be a 64-char hex digest."""
    for name, info in manifest["artifacts"].items():
        if name == "flow_plan_hashes":
            # nested mapping handled by separate test
            continue
        assert "sha256" in info, f"artifact {name}: missing sha256"
        assert _HEX64.match(info["sha256"]), (
            f"artifact {name}: malformed sha256 {info['sha256']!r}"
        )


def test_manifest_flow_plan_hashes_match_flow_sources(manifest):
    """The manifest's flow_plan_hashes dict must exactly match
    flows/*.flow.mind on disk (one entry per flow, all 64-char hex)."""
    flow_files = {p.stem.replace(".flow", "") for p in _FLOWS_DIR.glob("*.flow.mind")}
    manifest_flows = set(manifest["artifacts"]["flow_plan_hashes"].keys())
    assert flow_files == manifest_flows, (
        f"flow set drift: on-disk={sorted(flow_files)} "
        f"manifest={sorted(manifest_flows)}"
    )
    for name, h in manifest["artifacts"]["flow_plan_hashes"].items():
        assert _HEX64.match(h), f"flow {name}: malformed plan_hash {h!r}"


def test_manifest_safety_invariants_pass_through(manifest):
    """The fp_contraindicated_is_zero + tp_contraindicated_at_least_six
    booleans from the BitNet confusion matrix must propagate into the
    manifest. If either flips, the entire manifest is invalid."""
    inv = manifest["artifacts"]["bitnet_confusion_matrix"]["safety_invariants"]
    assert inv.get("fp_contraindicated_is_zero") is True, (
        "Layer 4.5 false-positive on contraindicated invalidates the "
        "manifest. Re-run scripts/build_bitnet_confusion_matrix.py and "
        "investigate the weight rotation before re-building the manifest."
    )
    assert inv.get("tp_contraindicated_at_least_six") is True, (
        "Layer 4.5 contraindicated TP dropped below the floor of 6. "
        "Re-run scripts/build_bitnet_confusion_matrix.py and investigate."
    )


def test_manifest_all_four_gates_pass(manifest):
    """The four-gate audit sequence must show PASS on every line."""
    gates = manifest["gates"]
    expected = {
        "pccp_recall",
        "negative_control_precision",
        "federation_invariant",
        "arch_mind_l1",
    }
    assert set(gates.keys()) == expected
    for name, verdict in gates.items():
        assert verdict == "PASS", (
            f"Gate {name} reported {verdict!r} when manifest was built. "
            f"Re-run that gate, fix the regression, then rebuild."
        )


def test_manifest_test_count_at_or_above_floor(manifest):
    """The collected-test count must stay at or above the iter-54 floor.
    If it ever drops below, either tests were deleted or the manifest is
    stale."""
    assert manifest["test_count"] >= 869, (
        f"manifest test_count = {manifest['test_count']}, floor = 869"
    )


def test_manifest_git_head_present(manifest):
    """git_head can be a 40-char SHA or 'unavailable' (CI without git)
    — never empty."""
    head = manifest.get("git_head", "")
    assert head, "git_head must not be empty"
    assert head == "unavailable" or re.match(r"^[0-9a-f]{40}$", head)


def test_manifest_matches_live_computation():
    """`--check` mode confirms artifact is in sync with live state."""
    cp = subprocess.run(
        [sys.executable, str(_SCRIPT), "--check"],
        capture_output=True, text=True, timeout=180, cwd=str(_REPO_ROOT),
    )
    assert cp.returncode == 0, (
        f"Reproducibility manifest drifted from live state.\n{cp.stdout}\n{cp.stderr}\n"
        "Re-run `python3 scripts/build_reproducibility_manifest.py` to refresh."
    )
