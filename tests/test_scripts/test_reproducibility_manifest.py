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


def test_manifest_all_gates_pass(manifest):
    """The five-gate audit sequence must show PASS on every line.

    Iter 90 promoted the audit-replay verifier from a stand-alone
    `verify_audit_replay.py --check` into run_all_gates.py, and the
    manifest's `gates` dict now includes `audit_replay` alongside the
    iter-20 four. Future gate additions extend `expected` here.
    """
    gates = manifest["gates"]
    expected = {
        "pccp_recall",
        "negative_control_precision",
        "federation_invariant",
        "arch_mind_l1",
        "audit_replay",
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


def test_manifest_tracks_all_load_bearing_artifacts(manifest):
    """Iter-75 added `docs/bitnet_calibration.json` — a load-bearing
    audit artifact (linked from JUDGES + demo, pinned by 8 tests). It
    must be in the reproducibility manifest's tracked artifacts so the
    SHA is content-addressed; an attacker who modifies the file but
    leaves the manifest stale would otherwise go undetected by the
    --check parity gate. This test catches that drift."""
    expected_artifacts = {
        "openevidence_cache",
        "bitnet_weights",
        "bitnet_confusion_matrix",
        "cohort_coverage_matrix",
        "synthea_demo_cohort",
        "bitnet_calibration",
        "audit_replay_pins",
        "pharmacology_flags",
        "flow_plan_hashes",
    }
    actual = set(manifest["artifacts"].keys())
    missing = expected_artifacts - actual
    assert not missing, (
        f"Reproducibility manifest is missing tracked artifacts: {sorted(missing)}. "
        f"Update scripts/build_reproducibility_manifest.py to include them, "
        f"then regenerate the manifest."
    )


def test_manifest_calibration_weights_id_matches_engine(manifest):
    """The calibration entry in the manifest must record the same
    weights_id (bundle_id) as the engine weights bundle. If they
    drift, the calibration was computed against stale weights."""
    calib = manifest["artifacts"].get("bitnet_calibration", {})
    weights = manifest["artifacts"].get("bitnet_weights", {})
    cal_wid = calib.get("weights_id")
    eng_bid = weights.get("bundle_id")
    if cal_wid is None or eng_bid is None:
        return  # bootstrap path — nothing to check
    assert cal_wid == eng_bid, (
        f"calibration weights_id={cal_wid[:16]}... but engine "
        f"bundle_id={eng_bid[:16]}... — calibration is stale, "
        f"re-run scripts/build_bitnet_calibration.py"
    )


def test_manifest_audit_replay_bundle_id_matches_engine(manifest):
    """The audit_replay_pins entry must record the same bundle_id as
    the engine weights bundle. If they drift, the audit-replay pins
    were captured against stale weights and `--check` would either
    show 'bundle_id_rotated' or repro_hash mismatches. Same drift
    class as test_manifest_calibration_weights_id_matches_engine."""
    audit = manifest["artifacts"].get("audit_replay_pins", {})
    weights = manifest["artifacts"].get("bitnet_weights", {})
    audit_bid = audit.get("bundle_id")
    eng_bid = weights.get("bundle_id")
    if audit_bid is None or eng_bid is None:
        return  # bootstrap path
    assert audit_bid == eng_bid, (
        f"audit_replay_pins bundle_id={audit_bid[:16]}... but engine "
        f"bundle_id={eng_bid[:16]}... — pins are stale, "
        f"re-run scripts/verify_audit_replay.py"
    )


def test_manifest_pharmacology_flags_integrity(manifest):
    """The pharmacology_flags entry must record the live drug count and
    flag-key set. If they drift, the demo's `13-flag ATC table` claim
    is hollow. Iter-97 catch — same gap class as iter-76 / iter-86 for
    bitnet_calibration / audit_replay_pins respectively.

    Floors:
      - drug_count >= 50 (covers the cohort + cache drug surface area)
      - len(flag_keys) >= 12 (the 12-flag minimum we ship)
      - schema_version is set (audit-trail invariant)
    """
    pf = manifest["artifacts"].get("pharmacology_flags", {})
    if not pf:
        return  # bootstrap path
    drug_count = pf.get("drug_count", 0)
    flag_keys = pf.get("flag_keys", [])
    schema_version = pf.get("schema_version")
    assert drug_count >= 50, (
        f"pharmacology_flags drug_count={drug_count} < 50; the curated "
        f"table must cover at least the cohort + cache drug surface."
    )
    assert len(flag_keys) >= 12, (
        f"pharmacology_flags has only {len(flag_keys)} flag keys; "
        f"floor is 12 (the iter-96 baseline schema)."
    )
    assert schema_version is not None, (
        "pharmacology_flags must carry a schema_version field — "
        "without it the audit chain can't pin the flag-set shape."
    )
    # Critical flag presence — the load-bearing pharmacology classes
    # the demo claims to flag.
    required = {
        "is_cyp3a4_strong_inhibitor",
        "is_cyp3a4_substrate",
        "is_p_gp_inhibitor",
        "is_p_gp_substrate",
        "is_statin",
        "is_anticoagulant",
        "is_maoi",
        "is_serotonergic",
    }
    missing = required - set(flag_keys)
    assert not missing, (
        f"pharmacology_flags missing required flag keys: {sorted(missing)}. "
        f"These are referenced in demo + JUDGES copy; removing them "
        f"breaks the published schema."
    )


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
