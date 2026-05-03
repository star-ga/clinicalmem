"""
PCCP Empirical Regression Harness — ClinicalMem

Predetermined Change Control Plan (PCCP) gate: verifies that no retraining
event causes a regression on the safety-critical OpenEvidence ground-truth
evaluation set, with a hard-fail on any contraindicated false-negative.

Usage:
    python3 scripts/run_clinical_regression_eval.py
    python3 scripts/run_clinical_regression_eval.py --update-baseline

Exit codes:
    0 — all PCCP gates pass; build may proceed
    1 — release-blocking degradation detected; build MUST NOT ship

Constraints:
    - No LLM calls (use_llm_fallback=False + cache is allowed; no live APIs)
    - Must complete in under 60 seconds
    - Writes structured report to docs/pccp_eval_latest.json
    - Establishes or compares against docs/pccp_baseline.json

Copyright 2026 STARGA, Inc. — Apache-2.0 License.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s  %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_CACHE_JSON = _REPO_ROOT / "docs" / "openevidence_cache.json"
_WEIGHTS_JSON = _REPO_ROOT / "engine" / "bitnet_weights.json"
_BASELINE_JSON = _REPO_ROOT / "docs" / "pccp_baseline.json"
_REPORT_JSON = _REPO_ROOT / "docs" / "pccp_eval_latest.json"

# Safety-class severity ordering (higher index = more severe).
# A pipeline verdict at >= the ground-truth level is safe-side (not a FN).
_SEVERITY_ORDER: dict[str, int] = {
    "none": 0,
    "minor": 1,
    "moderate": 2,
    "serious": 3,
    "major": 3,      # treated as equal to serious for safety ordering
    "contraindicated": 4,
}

# PCCP gate thresholds (Section 4.2 Algorithm Change Protocol)
_CONTRAINDICATED_RECALL_GATE: float = 1.0    # must be 100%
_CONTRAINDICATED_FNR_GATE: float = 0.0       # must be 0%
_MAJOR_FNR_GATE: float = 0.15                # must be < 15%
_BASELINE_REGRESSION_TOLERANCE: float = 0.001  # 0.1% — even a whisker triggers abort


def _load_cache() -> list[dict[str, Any]]:
    """Load OpenEvidence ground-truth pairs from docs/openevidence_cache.json."""
    with open(_CACHE_JSON, encoding="utf-8") as fh:
        return json.load(fh)


def _load_weights_metadata() -> dict[str, str]:
    """Return bundle_id and preimage_hash from the current bitnet_weights.json.

    The bundle_id is the SHA-256 over the canonical weight matrices; any
    auditor can re-derive it from engine/bitnet_classifier.py::_bundle_id.
    The preimage_hash is SHA-256 over the raw JSON bytes (byte-level pin).
    """
    import hashlib

    raw_bytes = _WEIGHTS_JSON.read_bytes()
    preimage_hash = hashlib.sha256(raw_bytes).hexdigest()

    payload = json.loads(raw_bytes.decode("utf-8"))
    bundle_id = payload.get("_meta", {}).get("bundle_id", "unknown")
    return {"bundle_id": bundle_id, "preimage_hash": preimage_hash}


def _pipeline_severity(drug_a: str, drug_b: str) -> str:
    """Run Layers 1 + 2-cache + 4.5 of the pipeline for one pair.

    No live API calls are made:
    - All live API keys are unset before invocation.
    - The RxNorm client (Layer 3) is patched to a no-op so it returns an
      empty list without touching the network (the NIH RxNav endpoint has
      no test mode; mocking is the correct CI pattern).
    - Layer 2 uses the OpenEvidence *cache* fallback (docs/openevidence_cache.json)
      — zero HTTP traffic because no OPENEVIDENCE_API_KEY is set.
    - Layers 4 (LLM) is skipped because no LLM API keys are set.
    - Layer 4.5 (BitNet) always runs (pure integer math, no network).

    This covers all 15 ground-truth pairs:
    - 12 pairs from the deterministic table (Layer 1) → always resolved.
    - 3 cache-only pairs (amoxicillin+penicillin, iodine+metformin,
      atorvastatin+grapefruit) → resolved by Layer 2 cache fallback.
    """
    import unittest.mock as mock  # stdlib — no extra deps  # noqa: PLC0415

    _api_env_keys = [
        "OPENEVIDENCE_API_KEY",
        "OPENAI_API_KEY",
        "GOOGLE_API_KEY",
        "GEMINI_API_KEY",
    ]
    saved = {k: os.environ.pop(k, None) for k in _api_env_keys}
    try:
        sys.path.insert(0, str(_REPO_ROOT))
        from engine.clinical_scoring import check_drug_interactions  # noqa: PLC0415
        import engine.rxnorm_client as rxnorm_mod  # noqa: PLC0415

        # Patch RxNorm client: return empty normalization so it contributes
        # zero interactions — no HTTP calls, deterministic output.
        empty_rxnorm: list = []
        with (
            mock.patch.object(rxnorm_mod, "normalize_medication_list", return_value={}),
            mock.patch.object(rxnorm_mod, "get_interactions_for_list", return_value=empty_rxnorm),
        ):
            interactions = check_drug_interactions(
                [drug_a, drug_b], use_llm_fallback=True
            )
        if not interactions:
            return "none"
        return interactions[0].severity
    finally:
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v


def _is_false_negative(ground_truth: str, pipeline_verdict: str) -> bool:
    """Return True if pipeline_verdict is a safety downgrade from ground_truth.

    A downgrade is when the pipeline reports a severity LOWER than the
    ground truth. Reporting higher is safe-side (conservative). Reporting
    equal is correct.

    Examples:
        ground_truth=contraindicated, verdict=none       -> True  (FN, release-blocking)
        ground_truth=contraindicated, verdict=moderate   -> True  (FN, release-blocking)
        ground_truth=serious, verdict=major              -> False (safe-side, OK)
        ground_truth=moderate, verdict=serious           -> False (safe-side, OK)
        ground_truth=serious, verdict=moderate           -> True  (FN on major class)
    """
    gt_rank = _SEVERITY_ORDER.get(ground_truth.lower(), 0)
    vd_rank = _SEVERITY_ORDER.get(pipeline_verdict.lower(), 0)
    return vd_rank < gt_rank


def _compute_metrics(
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compute per-class and aggregate metrics from the per-pair result list."""
    total = len(results)
    agreements = sum(1 for r in results if not r["is_fn"])

    # Per-class counts
    class_counts: dict[str, dict[str, int]] = {}
    for r in results:
        gt = r["ground_truth"]
        class_counts.setdefault(gt, {"total": 0, "fn": 0})
        class_counts[gt]["total"] += 1
        if r["is_fn"]:
            class_counts[gt]["fn"] += 1

    def _recall(cls: str) -> float:
        bucket = class_counts.get(cls, {"total": 0, "fn": 0})
        if bucket["total"] == 0:
            return 1.0
        return (bucket["total"] - bucket["fn"]) / bucket["total"]

    def _fnr(cls: str) -> float:
        bucket = class_counts.get(cls, {"total": 0, "fn": 0})
        if bucket["total"] == 0:
            return 0.0
        return bucket["fn"] / bucket["total"]

    mean_latency = (
        sum(r["latency_ms"] for r in results) / total if total > 0 else 0.0
    )

    return {
        "total_pairs": total,
        "total_agreements": agreements,
        "agreement_rate": agreements / total if total > 0 else 0.0,
        "mean_latency_ms": round(mean_latency, 3),
        "recall_contraindicated": _recall("contraindicated"),
        "recall_major": _recall("major"),
        "recall_serious": _recall("serious"),
        "recall_moderate": _recall("moderate"),
        "fnr_contraindicated": _fnr("contraindicated"),
        "fnr_major": _fnr("major"),
        "fnr_serious": _fnr("serious"),
        "fnr_moderate": _fnr("moderate"),
        "per_class_counts": class_counts,
    }


def _check_absolute_gates(metrics: dict[str, Any]) -> list[str]:
    """Return a list of gate failure messages (empty = all gates pass)."""
    failures: list[str] = []

    recall_ci = metrics["recall_contraindicated"]
    if recall_ci < _CONTRAINDICATED_RECALL_GATE:
        failures.append(
            f"GATE FAIL: contraindicated recall {recall_ci:.4f} "
            f"< required {_CONTRAINDICATED_RECALL_GATE:.4f} (100%)"
        )

    fnr_ci = metrics["fnr_contraindicated"]
    if fnr_ci > _CONTRAINDICATED_FNR_GATE:
        failures.append(
            f"GATE FAIL: contraindicated FNR {fnr_ci:.4f} "
            f"> allowed {_CONTRAINDICATED_FNR_GATE:.4f} (0%)"
        )

    fnr_maj = metrics.get("fnr_major", 0.0)
    if fnr_maj > _MAJOR_FNR_GATE:
        failures.append(
            f"GATE FAIL: major FNR {fnr_maj:.4f} "
            f"> allowed {_MAJOR_FNR_GATE:.4f} (15%)"
        )

    return failures


def _check_baseline_regression(
    current: dict[str, Any],
    baseline: dict[str, Any],
) -> list[str]:
    """Return regression failure messages relative to the saved baseline."""
    failures: list[str] = []
    safety_metrics = [
        "recall_contraindicated",
        "recall_major",
        "recall_serious",
    ]
    for key in safety_metrics:
        cur_val = current.get(key, 0.0)
        base_val = baseline.get(key, 0.0)
        drop = base_val - cur_val
        if drop > _BASELINE_REGRESSION_TOLERANCE:
            failures.append(
                f"REGRESSION: {key} dropped {drop:.4f} "
                f"(baseline {base_val:.4f} -> current {cur_val:.4f}); "
                f"tolerance is {_BASELINE_REGRESSION_TOLERANCE:.4f}"
            )
    return failures


def _print_summary_table(
    results: list[dict[str, Any]],
    metrics: dict[str, Any],
    weights_meta: dict[str, str],
    gate_failures: list[str],
    baseline_failures: list[str],
) -> None:
    """Print a human-readable summary table to stdout."""
    SEP = "-" * 72

    print()
    print("=" * 72)
    print("  ClinicalMem PCCP Empirical Regression Harness")
    print("=" * 72)
    print(f"  BitNet bundle_id    : {weights_meta['bundle_id']}")
    print(f"  Weights preimage    : {weights_meta['preimage_hash']}")
    print(f"  Ground-truth source : docs/openevidence_cache.json")
    print(SEP)

    # Per-pair results
    print(f"  {'PAIR':<40}  {'GT':<15}  {'PIPELINE':<15}  {'STATUS'}")
    print(SEP)
    for r in results:
        pair_label = f"{r['drug_a']}+{r['drug_b']}"[:38]
        status = "OK" if not r["is_fn"] else "FALSE-NEGATIVE"
        print(
            f"  {pair_label:<40}  {r['ground_truth']:<15}  "
            f"{r['pipeline_verdict']:<15}  {status}"
        )

    print(SEP)
    print(f"  Total pairs evaluated : {metrics['total_pairs']}")
    print(f"  Total agreements      : {metrics['total_agreements']}")
    print(f"  Agreement rate        : {metrics['agreement_rate']:.1%}")
    print(f"  Mean per-pair latency : {metrics['mean_latency_ms']:.1f} ms")
    print()
    print(f"  Recall  contraindicated : {metrics['recall_contraindicated']:.3f}  "
          f"(GATE: must be 1.000)")
    print(f"  Recall  major           : {metrics['recall_major']:.3f}")
    print(f"  Recall  serious         : {metrics['recall_serious']:.3f}")
    print(f"  Recall  moderate        : {metrics['recall_moderate']:.3f}")
    print()
    print(f"  FNR contraindicated : {metrics['fnr_contraindicated']:.3f}  "
          f"(GATE: must be 0.000)")
    print(f"  FNR major           : {metrics['fnr_major']:.3f}  "
          f"(GATE: must be < 0.150)")
    print(f"  FNR serious         : {metrics['fnr_serious']:.3f}")
    print(f"  FNR moderate        : {metrics['fnr_moderate']:.3f}")
    print(SEP)

    if not gate_failures and not baseline_failures:
        print("  PCCP GATE: PASS — all safety-class gates satisfied")
    else:
        for msg in gate_failures + baseline_failures:
            print(f"  {msg}")
        print("  PCCP GATE: FAIL — release BLOCKED")

    print("=" * 72)
    print()


def _build_report(
    results: list[dict[str, Any]],
    metrics: dict[str, Any],
    weights_meta: dict[str, str],
    gate_failures: list[str],
    baseline_failures: list[str],
    is_baseline_run: bool,
) -> dict[str, Any]:
    """Build the structured JSON-LD report written to docs/pccp_eval_latest.json."""
    import datetime

    return {
        "@context": "https://schema.org/",
        "@type": "SoftwareApplication",
        "name": "ClinicalMem PCCP Regression Harness",
        "version": "1.0.0",
        "dateCreated": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "license": "Apache-2.0",
        "description": (
            "Empirical PCCP gate verifying zero contraindicated false-negatives "
            "against the OpenEvidence ground-truth cache. Per ClinicalMem PCCP "
            "Section 4.2 Algorithm Change Protocol."
        ),
        "pccp_gate_pass": (not gate_failures and not baseline_failures),
        "is_baseline_run": is_baseline_run,
        "weights": {
            "bundle_id": weights_meta["bundle_id"],
            "preimage_hash": weights_meta["preimage_hash"],
            "path": "engine/bitnet_weights.json",
        },
        "ground_truth_source": "docs/openevidence_cache.json",
        "metrics": metrics,
        "gate_failures": gate_failures,
        "baseline_failures": baseline_failures,
        "per_pair_results": results,
    }


def run_eval(update_baseline: bool = False) -> int:
    """Execute the full evaluation and return the exit code (0=pass, 1=fail)."""
    logger.info("Loading ground-truth pairs from %s", _CACHE_JSON)
    cache_entries = _load_cache()
    logger.info("Loaded %d ground-truth drug pairs", len(cache_entries))

    logger.info("Reading BitNet weights metadata from %s", _WEIGHTS_JSON)
    weights_meta = _load_weights_metadata()
    logger.info(
        "BitNet bundle_id=%s preimage=%s",
        weights_meta["bundle_id"][:16] + "...",
        weights_meta["preimage_hash"][:16] + "...",
    )

    # Evaluate each pair
    per_pair_results: list[dict[str, Any]] = []
    for entry in cache_entries:
        drug_a, drug_b = entry["drug_pair_canonical"]
        ground_truth = entry["severity"]

        t0 = time.perf_counter()
        verdict = _pipeline_severity(drug_a, drug_b)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        is_fn = _is_false_negative(ground_truth, verdict)
        per_pair_results.append({
            "drug_a": drug_a,
            "drug_b": drug_b,
            "ground_truth": ground_truth,
            "pipeline_verdict": verdict,
            "is_fn": is_fn,
            "latency_ms": round(latency_ms, 3),
        })

    metrics = _compute_metrics(per_pair_results)
    gate_failures = _check_absolute_gates(metrics)

    # Baseline comparison
    baseline_failures: list[str] = []
    is_baseline_run = False

    if update_baseline or not _BASELINE_JSON.exists():
        is_baseline_run = True
        baseline_payload = {
            "recall_contraindicated": metrics["recall_contraindicated"],
            "recall_major": metrics["recall_major"],
            "recall_serious": metrics["recall_serious"],
            "recall_moderate": metrics["recall_moderate"],
            "fnr_contraindicated": metrics["fnr_contraindicated"],
            "fnr_major": metrics["fnr_major"],
            "total_pairs": metrics["total_pairs"],
            "weights_bundle_id": weights_meta["bundle_id"],
            "weights_preimage_hash": weights_meta["preimage_hash"],
        }
        _BASELINE_JSON.write_text(
            json.dumps(baseline_payload, indent=2), encoding="utf-8"
        )
        logger.info(
            "Baseline written to %s (bundle_id=%s...)",
            _BASELINE_JSON,
            weights_meta["bundle_id"][:16],
        )
    else:
        existing_baseline: dict[str, Any] = json.loads(
            _BASELINE_JSON.read_text(encoding="utf-8")
        )
        baseline_failures = _check_baseline_regression(metrics, existing_baseline)

    report = _build_report(
        per_pair_results,
        metrics,
        weights_meta,
        gate_failures,
        baseline_failures,
        is_baseline_run,
    )
    _REPORT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Structured report written to %s", _REPORT_JSON)

    _print_summary_table(
        per_pair_results, metrics, weights_meta, gate_failures, baseline_failures
    )

    all_failures = gate_failures + baseline_failures
    if all_failures:
        logger.error(
            "PCCP gate FAILED — %d release-blocking issue(s) detected", len(all_failures)
        )
        return 1

    logger.info("PCCP gate PASS — all safety-class invariants satisfied")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ClinicalMem PCCP empirical regression harness"
    )
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help=(
            "Overwrite docs/pccp_baseline.json with the current metrics. "
            "Run after a deliberate, reviewed weights update."
        ),
    )
    args = parser.parse_args()
    sys.exit(run_eval(update_baseline=args.update_baseline))


if __name__ == "__main__":
    main()
