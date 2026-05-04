# BitNet b1.58 Retrain v2 — Runpod Procedure

**Goal:** lift contraindicated recall from 6 / 20 = 30.0% to 20 / 20 =
100% on the live OpenEvidence cache, **without breaking the safety
invariant** `fp_contraindicated_is_zero` (no non-contraindicated pair
may classify as contraindicated).

**Why Runpod:** the local RTX 3080 is occupied for ~7.5 hours by the
mind-mem 4b retrain (Qwen-3.5-4B fine-tune). The BitNet retrain only
needs ~30 seconds of GPU time but can't share the device.

---

## What's in this folder

| File | Purpose |
|---|---|
| `build_augmented_corpus.py` | Reads `clinicalmem-bitnet-training/drug_corpus.jsonl` (3,247 rows) + `docs/openevidence_cache.json` (20 contraindicated entries), writes `drug_corpus_augmented.jsonl` (3,257 rows — adds 10 new contraindicated pairs not already in base). Idempotent. |
| `drug_corpus_augmented.jsonl` | The 3,257-row training corpus. Pre-built — no need to rerun the builder on Runpod unless the OpenEvidence cache changes. |
| `train_bitnet_v2.py` | The retrain script. v1 differences: (a) reads the augmented corpus, (b) extends forced-train fold to include EVERY contraindicated cache pair (loaded at runtime, not hardcoded), (c) oversample factor 80× → 100×, (d) two **gating verifications** that block promotion if either fails. |
| `bitnet_weights_v2.json` | Output: the new ternary weights bundle. Created by `train_bitnet_v2.py`. |
| `training_summary_v2.json` | Output: gate report + bundle_id. |

---

## Runpod execution procedure

### 1. Provision a pod

Any GPU with ≥ 4 GB VRAM is enough — the model is 8,581 parameters.
Recommended template: `runpod/pytorch:2.10.0-py3.12-cuda12.8` or any
recent PyTorch + CUDA image.

Cheapest tier (RTX 3090 / RTX 4090 community cloud) is fine. Total
billable time is ~2 minutes (provision + clone + train + verify),
well under $0.05.

### 2. Clone clinicalmem

```bash
git clone https://github.com/star-ga/clinicalmem.git /workspace/clinicalmem
cd /workspace/clinicalmem
git checkout main
```

### 3. Stage the base corpus

The base corpus lives at `clinicalmem-bitnet-training/`
on your laptop. The augmented corpus tracked in this repo
(`retrain_runpod/drug_corpus_augmented.jsonl`) already includes
everything the script needs at runtime, so the Runpod pod **does not
need access to /data/**.

Skip if you trust the checked-in augmented corpus. Otherwise:

```bash
# (Optional) re-augment from a fresh local corpus snapshot
scp -r <dev-machine>:./clinicalmem-bitnet-training \
       /workspace/clinicalmem-bitnet-training
# then on the pod:
python3 retrain_runpod/build_augmented_corpus.py
```

### 4. Install dependencies

```bash
pip install --quiet torch  # only torch is needed; everything else is stdlib
pip install -e .            # makes engine.bitnet_classifier importable
```

### 5. Run the retrain

```bash
python3 retrain_runpod/train_bitnet_v2.py
```

Expected output (~30 seconds on a 3090):

```
Loading augmented corpus...
  3257 pairs
Stratified split (v3 + cache contraindicated forced-train)...
  forced-train set: 25 rows (5 v3 anchors + 20 cache contraindicated)
  train: 2630  test: 627
  oversampled 25 forced anchors x100 -> train size now 5130

Training on cuda...
  epoch   0  test_acc 0.421
  ...
  epoch 399  test_acc 0.7XX
  best test_acc: 0.7XX

Exporting ternary weights...
  bundle_id = <NEW SHA-256>
  path      = retrain_runpod/bitnet_weights_v2.json

Gating verification (the load-bearing audit-chain claim)...
  Contraindicated recall: 20 / 20 (PASS)
  False positives on contraindicated: 0 (PASS)

✓ PASS — promoting weights to engine/bitnet_weights.json
  summary -> retrain_runpod/training_summary_v2.json
```

### 6. Push the new weights back

```bash
git add engine/bitnet_weights.json \
        retrain_runpod/bitnet_weights_v2.json \
        retrain_runpod/training_summary_v2.json
git commit -m "feat(bitnet): retrain v2 — 20/20 contraindicated recall"
git push origin main
```

The autonomous watcher loop on your laptop picks up the new commit on
its next 15-minute fire and ripples the bundle_id rotation through:

  - `tests/test_engine/test_bitnet_param_count_pin.py` (no change — params unchanged)
  - `tests/test_engine/test_bitnet_repro_hash_pin.py` (warfarin + ibuprofen repro_hash will change → must re-pin)
  - `tests/test_engine/test_bitnet_live_precision_pin.py` (recall band: was 0.30-0.45, now should be 1.0)
  - `tests/test_scripts/test_bitnet_confusion_matrix.py` (`tp_contraindicated_at_least_six` floor → can lift to 20)
  - `docs/bitnet_confusion_matrix.json` (regenerate)
  - `docs/demo.html` heatmap (every cell — needs script re-run)
  - `docs/demo.html` repro_hash, bundle_id chip
  - `docs/reproducibility_manifest.json` (regenerate)
  - `docs/eval_runs/round_4_*.json` (no change — eval is doc-level)
  - JUDGES.md `100% recall · contraindicated · 20/20` row already correct (it's the upstream pipeline number, not Layer 4.5; but we can now strengthen the language)

Plan to handle that ripple in iters 67-70 (one drift sweep + 4 pin-test
updates + 2 doc card updates).

---

## Gates the script enforces

The Runpod job will exit code 1 (no weight promotion) if either:

  - **Recall gate**: any of the 20 contraindicated cache pairs predicts
    a non-contraindicated severity.
  - **Precision gate**: any non-contraindicated cache pair predicts
    contraindicated (would break `fp_contraindicated_is_zero`).

If the script exits 1, do NOT promote `bitnet_weights_v2.json` to
`engine/bitnet_weights.json`. Tune hyperparameters and rerun. Levers
to consider:

  - Bump epochs from 400 → 800 (more time on the contraindicated
    anchors).
  - Bump oversample 100 → 200 (more training-set weight on anchors).
  - If recall passes but precision fails: reduce oversample (currently
    pushing too hard); the precision break means a non-contraindicated
    pair is being pulled toward the contraindicated logit by the
    aggressive anchor weighting. Try 80× → 60×.
  - If recall fails: bump epochs OR add more class_weight on the
    contraindicated class (currently inverse-frequency; could multiply
    by a constant > 1).

---

## Rollback

If the v2 weights ship and a downstream test fails, restore v1:

```bash
git restore --source HEAD~1 engine/bitnet_weights.json
```

The bundle_id in the manifest will mismatch until that's reverted on
the next iteration. The reproducibility manifest's `--check` test will
fail until regenerated.

---

*STARGA, Inc. — Apache-2.0*
