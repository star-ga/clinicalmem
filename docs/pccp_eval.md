# PCCP Empirical Regression Harness

## What It Does

`scripts/run_clinical_regression_eval.py` is the runnable proof behind the
Predetermined Change Control Plan (PCCP) described in `docs/fda_q_sub_draft.md`.
It verifies that no retraining event causes a regression on the safety-critical
contraindicated drug-pair class by running the full ClinicalMem deterministic
pipeline against the curated OpenEvidence ground-truth cache.

The script:

1. Loads `engine/bitnet_weights.json` and records its `bundle_id` (SHA-256 over
   the canonical weight matrices) and `preimage_hash` (SHA-256 over the raw JSON
   bytes). These two hashes together pin the exact weights version evaluated.

2. Loads `docs/openevidence_cache.json` as the ground-truth evaluation set (drug
   pairs with severity labels drawn from PubMed, FDA prescribing information, and
   Lexicomp citations; the cache size grows over time as new pairs are curated).

3. For every drug pair in the cache, runs the 6-tier ClinicalMem pipeline
   (Layer 1 deterministic table + Layer 2 OpenEvidence cache + Layer 4.5 BitNet
   stamp) with all live API keys suppressed and the RxNorm client mocked to empty.
   No LLM calls are made; no network traffic leaves the machine.

4. Compares each pipeline verdict against the ground-truth severity using a
   safety-side ordering: reporting a severity equal to or higher than ground truth
   counts as correct; reporting lower is a false-negative.

5. Computes per-class metrics:
   - Recall on contraindicated (THE load-bearing metric — must be 100%)
   - Recall on major, serious, moderate
   - False-negative rate (FNR) on contraindicated (must be 0%)
   - FNR on major (must be < 15%)

6. Computes aggregate metrics: total pairs, total agreements, agreement rate,
   mean per-pair latency.

7. Either establishes or compares against `docs/pccp_baseline.json`:
   - First run or `--update-baseline`: writes the current metrics as the new
     baseline.
   - Subsequent runs: checks that no safety-class recall has dropped by more
     than 0.1% from the baseline. Any drop triggers an abort.

8. Prints a human-readable summary table and writes a structured JSON-LD report
   to `docs/pccp_eval_latest.json`.

9. Exits 0 if all gates pass, exits 1 if any gate fails.

## Usage

```bash
# Normal CI run (compare against baseline):
python3 scripts/run_clinical_regression_eval.py

# After a deliberate, reviewed weights rotation:
python3 scripts/run_clinical_regression_eval.py --update-baseline
```

## Release-Blocking Metrics

| Metric | Threshold | Rationale |
|--------|-----------|-----------|
| Contraindicated recall | 1.000 (100%) | Zero false-negatives on contraindicated is an FDA SaMD safety invariant (PCCP Section 4.1) |
| Contraindicated FNR | 0.000 (0%) | Same as above — dual assertion |
| Major FNR | < 0.150 (15%) | Major interactions require conservative safety margin |
| Baseline regression (safety recalls) | < 0.001 drop (0.1%) | Even a marginal drop from the established baseline aborts the release |

## When to Update the Baseline

Run `--update-baseline` only when:

1. A new weights bundle has been trained from a clinician-curated expanded corpus
   (per PCCP Section 4.2 permitted change triggers).
2. All five frozen regression anchor pairs from
   `tests/test_engine/test_bitnet_classifier.py` still produce a non-`none`
   severity verdict with the new bundle.
3. The absolute gates pass (contraindicated recall = 100%, contraindicated FNR = 0%).
4. The new `bundle_id` has been published in the release notes.

Never update the baseline to escape a gate failure. A gate failure means the
weights are unsafe to ship.

## Secondary Regression Set

The `tests/test_engine/test_bitnet_classifier.py` test file includes five frozen
anchor pairs as a secondary regression set:

```python
@pytest.mark.parametrize("drug_a,drug_b", [
    ("warfarin", "ibuprofen"),
    ("amoxicillin", "penicillin"),
    ("metformin", "iodine"),
    ("atorvastatin", "grapefruit"),
    ("aspirin", "warfarin"),
])
def test_known_pair_produces_severity(drug_a, drug_b):
    ...
```

These five pairs must each return a non-`none` severity on every weights bundle
update. They are the "determinism anchors" that the BitNet repro_hash is pinned
against in the FDA Q-sub. The PCCP harness covers these same pairs as a superset
(all five are in `openevidence_cache.json`).

## CI Integration

Add to `.github/workflows/ci.yml` (or your CI configuration):

```yaml
- name: PCCP gate
  run: python3 scripts/run_clinical_regression_eval.py
```

The script completes in under 60 seconds (typically under 5 seconds with the
RxNorm layer mocked). No API keys or network access are required.

If the step exits 1, the CI run fails and the release is blocked. The structured
report at `docs/pccp_eval_latest.json` can be uploaded as a CI artifact for
audit purposes.

## Output Files

| File | Description |
|------|-------------|
| `docs/pccp_baseline.json` | Baseline metrics against which future runs are compared. Committed to the repository after each deliberate weights update. |
| `docs/pccp_eval_latest.json` | Structured JSON-LD report from the most recent run. Contains `bundle_id`, `preimage_hash`, all metrics, per-pair verdicts, and gate results. Suitable for FDA audit trail inclusion. |

## Report Schema

`docs/pccp_eval_latest.json` is a JSON-LD document:

```json
{
  "@context": "https://schema.org/",
  "@type": "SoftwareApplication",
  "pccp_gate_pass": true,
  "weights": {
    "bundle_id": "cfadb4f6...",
    "preimage_hash": "4078396f...",
    "path": "engine/bitnet_weights.json"
  },
  "metrics": {
    "total_pairs": 15,
    "recall_contraindicated": 1.0,
    "fnr_contraindicated": 0.0,
    ...
  },
  "gate_failures": [],
  "baseline_failures": [],
  "per_pair_results": [...]
}
```

The `bundle_id` and `preimage_hash` in the `weights` section are the canonical
preimage anchors that the FDA Q-sub references (Section 5.4 of
`docs/fda_q_sub_draft.md`). Any future audit can verify a historical decision by:

1. Obtaining the `bundle_id` from the persisted audit record.
2. Loading the corresponding `bitnet_weights.json` (identifiable by its `_meta.bundle_id`).
3. Replaying the forward pass using `engine/bitnet_classifier.py` — the output
   is bit-identical on any machine because the entire forward pass is integer
   arithmetic.
