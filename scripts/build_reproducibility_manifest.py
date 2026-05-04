#!/usr/bin/env python3
"""Build the single-file reproducibility manifest for ClinicalMem.

Produces ``docs/reproducibility_manifest.json``: one JSON file an FDA
SaMD auditor (or a CI gate) can drop into a compliance review and
verify every load-bearing artifact at once.

The manifest is the **content-addressed snapshot** of the entire
deterministic surface:

  - openevidence_cache.json — SHA-256 + entry count + per-class counts
  - bitnet_weights.json — SHA-256 + bundle_id + param count
  - bitnet_confusion_matrix.json — safety-invariant booleans
  - cohort_coverage_matrix.md — SHA-256 + patient count
  - bitnet_calibration.json — SHA-256 + weights_id + total_pairs + recall
  - flow plan_hashes for all 6 .flow.mind files
  - gate verdicts (PCCP / negative-control / federation / arch-mind)
  - test count
  - git HEAD

Run::

    python3 scripts/build_reproducibility_manifest.py

Use ``--check`` to verify the on-disk manifest matches the live
computation (CI parity gate).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_OUT = _REPO_ROOT / "docs" / "reproducibility_manifest.json"


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_head() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_REPO_ROOT, text=True, timeout=5,
        )
        return out.strip()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return "unavailable"


def _per_class_counts(cache: list[dict]) -> dict[str, int]:
    out: dict[str, int] = {}
    for entry in cache:
        sev = entry.get("severity", "").lower()
        out[sev] = out.get(sev, 0) + 1
    return out


def _flow_plan_hashes() -> dict[str, str]:
    sys.path.insert(0, str(_REPO_ROOT))
    from engine.flow_runner import compute_plan_hash, list_flow_names  # noqa: PLC0415

    flows_dir = _REPO_ROOT / "flows"
    return {
        name: compute_plan_hash(name, flows_dir=flows_dir)
        for name in list_flow_names(flows_dir)
    }


def _live_test_count() -> int:
    """Run pytest --collect-only on the standard scope and return count."""
    cp = subprocess.run(
        [
            sys.executable, "-m", "pytest",
            str(_REPO_ROOT / "tests" / "test_engine"),
            str(_REPO_ROOT / "tests" / "test_scripts"),
            "--collect-only", "-q",
        ],
        capture_output=True, text=True, timeout=120, cwd=str(_REPO_ROOT),
    )
    if cp.returncode != 0:
        return -1
    m = re.search(r"(\d+)\s+tests?\s+collected", cp.stdout)
    return int(m.group(1)) if m else -1


def _gate_verdict(script_name: str) -> str:
    """Run a gate script and return PASS / FAIL / SKIP."""
    script = _REPO_ROOT / "scripts" / script_name
    if not script.exists():
        return "SKIP"
    cp = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True, text=True, timeout=120, cwd=str(_REPO_ROOT),
    )
    return "PASS" if cp.returncode == 0 else "FAIL"


def _build() -> dict:
    """Build the manifest dict from live state."""
    cache_path = _REPO_ROOT / "docs" / "openevidence_cache.json"
    weights_path = _REPO_ROOT / "engine" / "bitnet_weights.json"
    confusion_path = _REPO_ROOT / "docs" / "bitnet_confusion_matrix.json"
    coverage_path = _REPO_ROOT / "docs" / "cohort_coverage_matrix.md"
    cohort_path = _REPO_ROOT / "docs" / "synthea_demo_cohort.json"
    calibration_path = _REPO_ROOT / "docs" / "bitnet_calibration.json"
    audit_replay_path = _REPO_ROOT / "docs" / "audit_replay_pins.json"

    cache = json.loads(cache_path.read_text())
    weights = json.loads(weights_path.read_text())
    confusion = json.loads(confusion_path.read_text())
    cohort = json.loads(cohort_path.read_text())
    calibration = json.loads(calibration_path.read_text()) if calibration_path.exists() else None
    audit_replay = json.loads(audit_replay_path.read_text()) if audit_replay_path.exists() else None

    patients = sum(
        1 for e in cohort.get("entry", [])
        if e.get("resource", {}).get("resourceType") == "Patient"
    )
    practitioners = sum(
        1 for e in cohort.get("entry", [])
        if e.get("resource", {}).get("resourceType") == "Practitioner"
    )

    bitnet_meta = weights.get("_meta", {})

    return {
        "@context": "https://schema.org",
        "@type": "Dataset",
        "name": "ClinicalMem Reproducibility Manifest",
        "version": "1.0.0",
        "dateCreated": datetime.now(timezone.utc).isoformat(),
        "license": "Apache-2.0",
        "description": (
            "Content-addressed snapshot of every load-bearing deterministic "
            "artifact in ClinicalMem. An FDA SaMD auditor (or a CI gate) "
            "verifies all eight artifacts + four gate verdicts + flow plan "
            "hashes by checking this single file. Re-run "
            "`scripts/build_reproducibility_manifest.py --check` to verify "
            "the on-disk manifest matches the live computation."
        ),
        "git_head": _git_head(),
        "artifacts": {
            "openevidence_cache": {
                "path": "docs/openevidence_cache.json",
                "sha256": _sha256_file(cache_path),
                "entry_count": len(cache),
                "per_class_counts": _per_class_counts(cache),
            },
            "bitnet_weights": {
                "path": "engine/bitnet_weights.json",
                "sha256": _sha256_file(weights_path),
                "bundle_id": bitnet_meta.get("bundle_id", "unknown"),
                "param_count": bitnet_meta.get("param_count", "unknown"),
            },
            "bitnet_confusion_matrix": {
                "path": "docs/bitnet_confusion_matrix.json",
                "sha256": _sha256_file(confusion_path),
                "safety_invariants": confusion.get("safety_invariants", {}),
                "cache_pairs_total": confusion.get("cache_pairs_total", 0),
            },
            "cohort_coverage_matrix": {
                "path": "docs/cohort_coverage_matrix.md",
                "sha256": _sha256_file(coverage_path),
            },
            "synthea_demo_cohort": {
                "path": "docs/synthea_demo_cohort.json",
                "sha256": _sha256_file(cohort_path),
                "patient_count": patients,
                "practitioner_count": practitioners,
                "entry_count": len(cohort.get("entry", [])),
            },
            "bitnet_calibration": {
                "path": "docs/bitnet_calibration.json",
                "sha256": _sha256_file(calibration_path) if calibration_path.exists() else None,
                "weights_id": calibration.get("weights_id") if calibration else None,
                "total_pairs": calibration.get("total_pairs") if calibration else None,
                "contraindicated_recall": (
                    calibration.get("by_class", {}).get("contraindicated", {}).get("recall")
                    if calibration else None
                ),
            },
            "audit_replay_pins": {
                "path": "docs/audit_replay_pins.json",
                "sha256": _sha256_file(audit_replay_path) if audit_replay_path.exists() else None,
                "bundle_id": audit_replay.get("bundle_id") if audit_replay else None,
                "pair_count": len(audit_replay.get("pairs", [])) if audit_replay else 0,
            },
            "flow_plan_hashes": _flow_plan_hashes(),
        },
        "gates": {
            "pccp_recall": _gate_verdict("run_clinical_regression_eval.py"),
            "negative_control_precision": _gate_verdict(
                "run_negative_control_eval.py",
            ),
            "federation_invariant": _gate_verdict("federation_mock_demo.py"),
            "arch_mind_l1": _gate_verdict("run_arch_mind_gate.py"),
        },
        "test_count": _live_test_count(),
        "audit_replay_hint": (
            "To verify: (1) check git_head matches the auditor's expected "
            "commit, (2) recompute SHA-256 of every artifact path and "
            "compare, (3) recompute flow plan_hashes via "
            "engine.flow_runner.compute_plan_hash, (4) re-run the four "
            "gate scripts, (5) run pytest tests/test_engine tests/test_scripts "
            "and confirm test_count is at or above the manifested value."
        ),
    }


def _diff(live: dict, on_disk: dict) -> list[str]:
    """Return non-trivial differences (ignoring dateCreated and test_count).

    test_count is allowed to grow as new tests land; the floor pin in
    test_reproducibility_manifest.py enforces the lower bound. A `--check`
    fail on test_count would force regeneration on every test addition,
    which is wasteful churn — the floor pin catches the real signal
    (test_count drops below floor)."""
    diffs: list[str] = []

    def _walk(a, b, path):
        if path in ("dateCreated", "test_count"):
            return
        if isinstance(a, dict) and isinstance(b, dict):
            for k in set(a) | set(b):
                _walk(a.get(k), b.get(k), f"{path}.{k}" if path else k)
        elif a != b:
            diffs.append(f"{path}: live={a!r} on_disk={b!r}")

    _walk(live, on_disk, "")
    return diffs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify the on-disk manifest matches live; exit 1 on diff",
    )
    args = parser.parse_args()

    live = _build()

    if args.check:
        if not _OUT.exists():
            print(f"FAIL — {_OUT} does not exist")
            return 1
        on_disk = json.loads(_OUT.read_text())
        diffs = _diff(live, on_disk)
        if diffs:
            print("FAIL — on-disk manifest differs from live computation:")
            for d in diffs:
                print(f"  • {d}")
            return 1
        print(f"PASS — {_OUT} matches live computation")
        return 0

    _OUT.write_text(json.dumps(live, indent=2) + "\n")
    print(f"wrote {_OUT}")

    # Summary
    print(f"\n  git_head: {live['git_head'][:16]}…")
    print(f"  test_count: {live['test_count']}")
    print("  gate verdicts:")
    for name, verdict in live["gates"].items():
        print(f"    {name:35s}: {verdict}")
    print("  artifact SHAs:")
    for name, info in live["artifacts"].items():
        if isinstance(info, dict) and "sha256" in info:
            print(f"    {name:30s}: {info['sha256'][:16]}…")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
