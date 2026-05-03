"""
Tests for scripts/run_clinical_regression_eval.py.

Five load-bearing tests:

1. test_regression_eval_runs_to_completion
   - Subprocess invocation; asserts exit code 0 and all gate metrics present
     in the human-readable stdout on a healthy weights bundle.

2. test_regression_eval_writes_baseline_on_first_run
   - With a temp directory for the baseline JSON, asserts the file is
     created on first run and contains the required metric keys.

3. test_regression_eval_compares_against_baseline_on_subsequent_run
   - Writes a baseline that matches current metrics exactly, then reruns;
     asserts exit code 0 (no regression).

4. test_regression_eval_aborts_on_simulated_contraindicated_regression
   - Monkeypatches the pipeline to return "none" for all contraindicated
     cache entries; asserts exit code 1 and appropriate gate-failure
     messages in stdout.

5. test_pccp_baseline_json_is_human_readable_jsonld
   - Asserts the report written to docs/pccp_eval_latest.json is valid
     JSON-LD with the required @context and @type fields and all mandatory
     metric keys.

Copyright 2026 STARGA, Inc. — Apache-2.0 License.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any
import unittest.mock as mock

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _REPO_ROOT / "scripts" / "run_clinical_regression_eval.py"
_CACHE_JSON = _REPO_ROOT / "docs" / "openevidence_cache.json"
_REPORT_JSON = _REPO_ROOT / "docs" / "pccp_eval_latest.json"

# Derive expected pair count from the actual ground-truth file so the
# tests remain correct as the cache is expanded over time.
_EXPECTED_PAIR_COUNT = len(json.loads(_CACHE_JSON.read_text()))

# ─── helpers ──────────────────────────────────────────────────────────────────

def _run_script(*extra_args: str, extra_env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    """Run the harness script as a subprocess and capture all output."""
    env = {**os.environ, **(extra_env or {})}
    # Remove live API keys so no network escapes
    for k in ["OPENEVIDENCE_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY"]:
        env.pop(k, None)
    return subprocess.run(
        [sys.executable, str(_SCRIPT), *extra_args],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=str(_REPO_ROOT),
        env=env,
    )


# ─── Test 1: happy-path completion ────────────────────────────────────────────

def test_regression_eval_runs_to_completion() -> None:
    """The harness must exit 0 with healthy weights and report PCCP GATE: PASS."""
    result = _run_script("--update-baseline")

    assert result.returncode == 0, (
        f"Expected exit 0 but got {result.returncode}.\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )

    stdout = result.stdout
    # Summary table must appear
    assert "PCCP Empirical Regression Harness" in stdout, "Missing harness title"
    assert "PCCP GATE: PASS" in stdout, "Expected PCCP GATE: PASS in output"
    assert "Recall  contraindicated : 1.000" in stdout, "Missing contraindicated recall metric"
    assert "FNR contraindicated : 0.000" in stdout, "Missing contraindicated FNR metric"
    assert "Agreement rate" in stdout, "Missing agreement rate"
    # All 15 pairs from the cache should appear
    assert "aspirin+warfarin" in stdout
    assert "ciprofloxacin+tizanidine" in stdout
    assert "iodine+metformin" in stdout


# ─── Test 2: baseline creation on first run ───────────────────────────────────

def test_regression_eval_writes_baseline_on_first_run(tmp_path: Path) -> None:
    """On first run, docs/pccp_baseline.json must be created with required keys."""
    import importlib
    import sys

    # Patch the baseline path to a temp location so we don't clobber production
    baseline_tmp = tmp_path / "pccp_baseline.json"
    report_tmp = tmp_path / "pccp_eval_latest.json"

    # Load the module under test in-process with paths patched
    if "scripts.run_clinical_regression_eval" in sys.modules:
        del sys.modules["scripts.run_clinical_regression_eval"]
    # Force fresh import from the scripts directory
    spec_dir = str(_REPO_ROOT / "scripts")
    if spec_dir not in sys.path:
        sys.path.insert(0, spec_dir)

    import run_clinical_regression_eval as harness  # type: ignore[import]

    # Ensure env keys are absent
    for k in ["OPENEVIDENCE_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY"]:
        os.environ.pop(k, None)

    import engine.rxnorm_client as rxnorm_mod  # type: ignore[import]

    with (
        mock.patch.object(harness, "_BASELINE_JSON", baseline_tmp),
        mock.patch.object(harness, "_REPORT_JSON", report_tmp),
        mock.patch.object(rxnorm_mod, "normalize_medication_list", return_value={}),
        mock.patch.object(rxnorm_mod, "get_interactions_for_list", return_value=[]),
    ):
        exit_code = harness.run_eval(update_baseline=True)

    assert exit_code == 0, "Expected exit 0 on healthy weights"
    assert baseline_tmp.exists(), "pccp_baseline.json was not written"

    baseline = json.loads(baseline_tmp.read_text())

    required_keys = {
        "recall_contraindicated",
        "recall_serious",
        "recall_moderate",
        "fnr_contraindicated",
        "fnr_major",
        "total_pairs",
        "weights_bundle_id",
        "weights_preimage_hash",
    }
    for key in required_keys:
        assert key in baseline, f"Missing key '{key}' in pccp_baseline.json"

    assert baseline["recall_contraindicated"] == 1.0, "Contraindicated recall must be 1.0"
    assert baseline["fnr_contraindicated"] == 0.0, "Contraindicated FNR must be 0.0"
    assert baseline["total_pairs"] == _EXPECTED_PAIR_COUNT


# ─── Test 3: subsequent run compares against existing baseline ─────────────────

def test_regression_eval_compares_against_baseline_on_subsequent_run(tmp_path: Path) -> None:
    """When a baseline exists and metrics are unchanged, exit code must be 0."""
    import sys

    if "run_clinical_regression_eval" in sys.modules:
        del sys.modules["run_clinical_regression_eval"]
    spec_dir = str(_REPO_ROOT / "scripts")
    if spec_dir not in sys.path:
        sys.path.insert(0, spec_dir)

    import run_clinical_regression_eval as harness  # type: ignore[import]
    import engine.rxnorm_client as rxnorm_mod  # type: ignore[import]

    # Write a passing baseline — matches what the healthy pipeline produces
    baseline_tmp = tmp_path / "pccp_baseline.json"
    report_tmp = tmp_path / "pccp_eval_latest.json"
    baseline_content = {
        "recall_contraindicated": 1.0,
        "recall_major": 1.0,
        "recall_serious": 1.0,
        "recall_moderate": 1.0,
        "fnr_contraindicated": 0.0,
        "fnr_major": 0.0,
        "total_pairs": _EXPECTED_PAIR_COUNT,
        "weights_bundle_id": "any",
        "weights_preimage_hash": "any",
    }
    baseline_tmp.write_text(json.dumps(baseline_content), encoding="utf-8")

    for k in ["OPENEVIDENCE_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY"]:
        os.environ.pop(k, None)

    with (
        mock.patch.object(harness, "_BASELINE_JSON", baseline_tmp),
        mock.patch.object(harness, "_REPORT_JSON", report_tmp),
        mock.patch.object(rxnorm_mod, "normalize_medication_list", return_value={}),
        mock.patch.object(rxnorm_mod, "get_interactions_for_list", return_value=[]),
    ):
        exit_code = harness.run_eval(update_baseline=False)

    assert exit_code == 0, (
        "Expected exit 0 when metrics match baseline; subsequent comparison should pass"
    )
    assert report_tmp.exists(), "Report JSON not written"
    report = json.loads(report_tmp.read_text())
    assert report["pccp_gate_pass"] is True
    assert report["baseline_failures"] == []


# ─── Test 4: abort on simulated contraindicated regression ────────────────────

def test_regression_eval_aborts_on_simulated_contraindicated_regression(
    tmp_path: Path,
) -> None:
    """When pipeline returns 'none' for all contraindicated pairs, exit code must be 1.

    This simulates a catastrophic weights regression where the classifier
    can no longer identify contraindicated drug pairs — the primary
    PCCP gate violation.
    """
    import sys

    if "run_clinical_regression_eval" in sys.modules:
        del sys.modules["run_clinical_regression_eval"]
    spec_dir = str(_REPO_ROOT / "scripts")
    if spec_dir not in sys.path:
        sys.path.insert(0, spec_dir)

    import run_clinical_regression_eval as harness  # type: ignore[import]
    import engine.rxnorm_client as rxnorm_mod  # type: ignore[import]

    # Baseline reflects "passing" metrics — so the abort comes from
    # the absolute PCCP gate (recall < 1.0), not just baseline regression.
    baseline_tmp = tmp_path / "pccp_baseline.json"
    report_tmp = tmp_path / "pccp_eval_latest.json"
    baseline_content = {
        "recall_contraindicated": 1.0,
        "recall_major": 1.0,
        "recall_serious": 1.0,
        "recall_moderate": 1.0,
        "fnr_contraindicated": 0.0,
        "fnr_major": 0.0,
        "total_pairs": _EXPECTED_PAIR_COUNT,
        "weights_bundle_id": "any",
        "weights_preimage_hash": "any",
    }
    baseline_tmp.write_text(json.dumps(baseline_content), encoding="utf-8")

    # Patch _pipeline_severity so any call for a contraindicated pair returns "none"
    _cache = json.loads(_CACHE_JSON.read_text())
    _contraindicated_pairs = {
        tuple(e["drug_pair_canonical"])
        for e in _cache
        if e["severity"] == "contraindicated"
    }

    original_severity_fn = harness._pipeline_severity

    def _patched_severity(drug_a: str, drug_b: str) -> str:
        pair = tuple(sorted((drug_a, drug_b)))
        if pair in _contraindicated_pairs:
            return "none"  # simulate catastrophic regression
        return original_severity_fn(drug_a, drug_b)

    for k in ["OPENEVIDENCE_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY"]:
        os.environ.pop(k, None)

    with (
        mock.patch.object(harness, "_BASELINE_JSON", baseline_tmp),
        mock.patch.object(harness, "_REPORT_JSON", report_tmp),
        mock.patch.object(harness, "_pipeline_severity", side_effect=_patched_severity),
        mock.patch.object(rxnorm_mod, "normalize_medication_list", return_value={}),
        mock.patch.object(rxnorm_mod, "get_interactions_for_list", return_value=[]),
    ):
        exit_code = harness.run_eval(update_baseline=False)

    assert exit_code == 1, (
        f"Expected exit 1 when contraindicated pairs return 'none', got {exit_code}"
    )

    # Report must exist and record the failure
    assert report_tmp.exists(), "Report JSON not written even on gate failure"
    report = json.loads(report_tmp.read_text())
    assert report["pccp_gate_pass"] is False, "pccp_gate_pass must be False on failure"
    assert len(report["gate_failures"]) > 0, "gate_failures list must be non-empty"

    # At least one failure message must mention contraindicated
    failure_text = " ".join(report["gate_failures"] + report.get("baseline_failures", []))
    assert "contraindicated" in failure_text.lower(), (
        f"Expected 'contraindicated' in failure messages; got:\n{failure_text}"
    )


# ─── Test 5: JSON-LD report is human-readable and schema-complete ─────────────

def test_pccp_baseline_json_is_human_readable_jsonld(tmp_path: Path) -> None:
    """The report docs/pccp_eval_latest.json must be valid JSON-LD with required fields.

    This is the artifact referenced in the FDA Q-sub as the structured
    evidence record. Auditors must be able to parse it with any standard
    JSON parser and find the canonical weight bundle_id and preimage hash.
    """
    import sys

    if "run_clinical_regression_eval" in sys.modules:
        del sys.modules["run_clinical_regression_eval"]
    spec_dir = str(_REPO_ROOT / "scripts")
    if spec_dir not in sys.path:
        sys.path.insert(0, spec_dir)

    import run_clinical_regression_eval as harness  # type: ignore[import]
    import engine.rxnorm_client as rxnorm_mod  # type: ignore[import]

    baseline_tmp = tmp_path / "pccp_baseline.json"
    report_tmp = tmp_path / "pccp_eval_latest.json"

    for k in ["OPENEVIDENCE_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY"]:
        os.environ.pop(k, None)

    with (
        mock.patch.object(harness, "_BASELINE_JSON", baseline_tmp),
        mock.patch.object(harness, "_REPORT_JSON", report_tmp),
        mock.patch.object(rxnorm_mod, "normalize_medication_list", return_value={}),
        mock.patch.object(rxnorm_mod, "get_interactions_for_list", return_value=[]),
    ):
        exit_code = harness.run_eval(update_baseline=True)

    assert exit_code == 0

    raw = report_tmp.read_text(encoding="utf-8")

    # Must be valid JSON
    report: dict[str, Any] = json.loads(raw)

    # JSON-LD required fields
    assert report.get("@context") == "https://schema.org/", (
        f"Expected @context='https://schema.org/', got {report.get('@context')!r}"
    )
    assert report.get("@type") == "SoftwareApplication", (
        f"Expected @type='SoftwareApplication', got {report.get('@type')!r}"
    )

    # Structural fields
    assert "pccp_gate_pass" in report
    assert isinstance(report["pccp_gate_pass"], bool)

    assert "weights" in report
    weights_section = report["weights"]
    assert "bundle_id" in weights_section, "weights.bundle_id must be present"
    assert "preimage_hash" in weights_section, "weights.preimage_hash must be present"
    assert len(weights_section["bundle_id"]) == 64, "bundle_id must be 64-char SHA-256"
    assert len(weights_section["preimage_hash"]) == 64, "preimage_hash must be 64-char SHA-256"

    # Metrics section
    metrics = report.get("metrics", {})
    for key in [
        "total_pairs",
        "total_agreements",
        "recall_contraindicated",
        "fnr_contraindicated",
        "mean_latency_ms",
    ]:
        assert key in metrics, f"Missing key 'metrics.{key}' in report"

    assert metrics["total_pairs"] == _EXPECTED_PAIR_COUNT

    # Per-pair results must cover all cache entries
    per_pair = report.get("per_pair_results", [])
    assert len(per_pair) == _EXPECTED_PAIR_COUNT, (
        f"Expected {_EXPECTED_PAIR_COUNT} per-pair results, got {len(per_pair)}"
    )

    # Each pair entry must have the required fields
    for entry in per_pair:
        for field in ["drug_a", "drug_b", "ground_truth", "pipeline_verdict", "is_fn", "latency_ms"]:
            assert field in entry, f"Missing field '{field}' in per-pair result {entry}"

    # Human-readable: pretty-printed (indent=2 produces newlines)
    assert "\n" in raw, "Report JSON must be pretty-printed (indented)"

    # gate_failures and baseline_failures must be present
    assert "gate_failures" in report
    assert "baseline_failures" in report
    assert isinstance(report["gate_failures"], list)
    assert isinstance(report["baseline_failures"], list)
