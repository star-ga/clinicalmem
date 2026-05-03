"""Negative-control precision check.

Complements scripts/run_clinical_regression_eval.py (recall on
positive interactions) by running the pipeline against a curated
set of clinically-known non-interactions and reporting false-positive
rate.

Goal: every pair in docs/negative_control_cohort.json should yield
``severity in {"none", "minor"}`` from the deterministic pipeline.
Anything stricter is a false positive (FP) — a precision miss.

Usage:
    python3 scripts/run_negative_control_eval.py
    python3 scripts/run_negative_control_eval.py --json   # machine-readable

Exit code:
    0  -- all pairs pass (no false positives at moderate-or-stricter)
    1  -- one or more false positives detected

Apache-2.0 — STARGA, Inc.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Make the engine package importable when run directly.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from scripts.run_clinical_regression_eval import _pipeline_severity  # noqa: E402

_COHORT = _REPO_ROOT / "docs" / "negative_control_cohort.json"
# Severity rank: anything moderate-or-stricter on a non-interaction is a FP.
_FP_THRESHOLD_RANK = 3  # moderate
_RANK = {"none": 0, "minor": 1, "moderate": 3, "serious": 4, "major": 4, "contraindicated": 5}


def _is_fp(verdict: str) -> bool:
    return _RANK.get(verdict.lower(), 0) >= _FP_THRESHOLD_RANK


def _run() -> dict:
    cohort = json.loads(_COHORT.read_text())
    results = []
    for pair in cohort:
        a, b = pair["drug_a"], pair["drug_b"]
        t0 = time.perf_counter()
        verdict = _pipeline_severity(a, b)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        results.append(
            {
                "drug_a": a,
                "drug_b": b,
                "expected_severity": pair["expected_severity"],
                "pipeline_verdict": verdict,
                "is_fp": _is_fp(verdict),
                "latency_ms": round(dt_ms, 3),
            }
        )
    total = len(results)
    fp = sum(1 for r in results if r["is_fp"])
    return {
        "total_pairs": total,
        "false_positives": fp,
        "precision": (total - fp) / total if total else 1.0,
        "fpr": fp / total if total else 0.0,
        "mean_latency_ms": round(
            sum(r["latency_ms"] for r in results) / total if total else 0.0, 3
        ),
        "per_pair_results": results,
    }


def _print_human(report: dict) -> None:
    print("=" * 72)
    print("ClinicalMem — Negative-Control Precision Check")
    print("=" * 72)
    print(f"  Cohort        : {_COHORT}")
    print(f"  Total pairs   : {report['total_pairs']}")
    print(f"  False positives : {report['false_positives']}")
    print(f"  Precision     : {report['precision']:.4f}")
    print(f"  FPR           : {report['fpr']:.4f}")
    print(f"  Mean latency  : {report['mean_latency_ms']:.3f} ms")
    print("-" * 72)
    for r in report["per_pair_results"]:
        marker = "FAIL" if r["is_fp"] else "OK"
        print(
            f"  [{marker}] {r['drug_a']:20} + {r['drug_b']:25} "
            f"→ {r['pipeline_verdict']:15} "
            f"(expected ≤ minor)"
        )
    print("=" * 72)
    if report["false_positives"] == 0:
        print("  PRECISION GATE: PASS — no false positives.")
    else:
        print(f"  PRECISION GATE: FAIL — {report['false_positives']} FP detected.")
    print("=" * 72)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = ap.parse_args()
    report = _run()
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        _print_human(report)
    return 0 if report["false_positives"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
