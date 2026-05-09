"""Pin iter-421 Path B tier-2 specialist bundle integrity.

iter-425 (round T1) — **36th cross-pin family** in the discipline lineage.
iter-421 shipped the Path B 2-bundle ensemble with a NEW bundle (B,
``5f7ed5f6…``, h=64) cascading after the frozen v8 contra/major gate
(A, ``1f0f8859…``, h=256). Bundle A is content-pinned by the iter-255 +
iter-265 ``test_v8_bundle_integrity_pin.py`` (10 invariants). Bundle B
had ZERO content protection — only its existence was implicit (engine
loader's graceful-fallback would silently degrade to A-only mode if B
went missing or got mis-edited, and demo chips would silently regress
to 84%/91% serious/moderate without any test catching it).

Same drift-prevention shape as the iter-255 v8 pin, applied to the new
B bundle. Catches:

- Hand-edit of ``_meta`` values without rotating ``bundle_id`` (since
  ``bundle_id`` is hashed only over the canonical-form weight payload,
  not over ``_meta``)
- File replacement with a non-iter-421-gate-passing candidate (e.g.,
  a v11 attempt that regressed contra; gate-failure preserves the
  weights as audit-trail-only and engine MUST stay on A-only fallback)
- JSON pretty-print drift past the expected ~30 KB band
- Sparsity collapse signaling STE thresholding broke
- Stray top-level keys inflating audit-trail confusion
- Engine ↔ retrain_runpod copy divergence (same content + SHA-256
  byte-identical between ``engine/`` and ``retrain_runpod/`` copies)
- Self-referenced ``ensemble_partner_bundle_id_a`` pointing at a stale
  v8 hash (would mean A and B drifted out of sync — composite weights_id
  in audit chain would still fire but auditor reconstructing the
  ensemble decision would load the WRONG A bundle)

iter-421 ensemble cascade: A predicts contra → return contra (A's
verdict); otherwise B's constrained argmax over {moderate, serious,
major} replaces A's prediction. A frozen at 100% contra/major recall;
B's job is the serious/moderate gap closure (84% → 100% / 91% → 100%).

Cross-pin lineage: 36th cross-pin family
(iter-178/183/188/193/198/203/223/228/232/234/236/239/240/244/246/247
 + iter-255 + iter-260 + iter-279 + iter-281 + iter-285 + iter-286 +
 iter-295 + iter-296 + iter-301 + iter-304 + iter-306 + iter-311 +
 iter-314 + iter-319 + iter-324 + iter-329 + iter-366 + iter-374 +
 iter-390 + iter-425).

Same shape: every load-bearing artifact gets a content-integrity pin
that ties artifact + provenance metadata + cross-file consistency
together so silent drift fails the gate at commit time.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_BUNDLE_B_ENGINE = _REPO_ROOT / "engine" / "bitnet_weights_b_specialist.json"
_BUNDLE_B_RETRAIN = _REPO_ROOT / "retrain_runpod" / "bitnet_weights_b_specialist.json"

_PINNED_BUNDLE_ID_B = (
    "5f7ed5f67f4db0d55d89c63f00b340ebbea598ea861669a85a69cdf6376e44b8"
)
_B_SHORT = _PINNED_BUNDLE_ID_B[:8]   # "5f7ed5f6"

# Bundle A (frozen v8) hash — B's _meta must point here as the ensemble
# partner. Drift here means A and B got out of sync.
_PINNED_BUNDLE_ID_A = (
    "1f0f88591c05af57c62d844b667639b29c7d1f0eb1b213073d158101611f76e6"
)

# B bundle expected size band: ~30 KB (live: 30,587 bytes).
# h=64 vs v8's h=256 gives ~4x fewer ternary weights, hence ~4x smaller.
# 12,672 ternary weights serialized as JSON arrays of -1/0/1 ints.
# Hard ceiling at 60 KB so a deliberate metadata expansion fails loud.
_B_SIZE_LO = 28 * 1024   # 28 KB
_B_SIZE_HI = 35 * 1024   # 35 KB
_B_SIZE_HARD_CEILING = 60 * 1024  # 60 KB

# Ternary weight sparsity floor: ~42.5% live. Same band as A (40-60%)
# since both bundles use the same STE quantization recipe.
_B_SPARSITY_LO = 0.40
_B_SPARSITY_HI = 0.60

# Required _meta provenance fields. iter-421 sweep recorded all 22.
# A future regen that drops any of these breaks FDA SaMD audit-trail.
_B_META_REQUIRED = frozenset({
    "augmentation",
    "best_test_acc_non_contra",
    "bias_dtype",
    "bundle_id",
    "dispatcher_rule",
    "ensemble_partner_bundle_id_a",
    "feature_breakdown",
    "flag_keys_count",
    "hidden_features",
    "in_features",
    "major_fp_within_non_contra",
    "major_recall",
    "moderate_recall",
    "out_features",
    "pair_derived_rule_count",
    "role",
    "schema",
    "serious_recall",
    "trained_with",
    "training_corpus",
    "training_iter",
    "weight_dtype",
})

# Top-level JSON key set MUST be exactly these 5 (canonical bundle
# shape, identical to A). Adding any new top-level key would not
# change bundle_id (since bundle_id is hashed only over the
# canonical-form weight matrices) but would inflate file size + risk
# audit confusion.
_B_TOPLEVEL_REQUIRED = frozenset({
    "_meta",
    "hidden_b",
    "hidden_w",
    "output_b",
    "output_w",
})

# Architecture pinned by iter-421 multi-LLM consensus (DeepSeek recipe):
# h=64 hidden, 193-dim feature input, 5-class output. A future B
# variant with different dims would require a new bundle_id (since
# matrix shapes change canonical-form bytes), so this is also a
# bundle_id self-consistency cross-check.
_B_HIDDEN_FEATURES = 64
_B_IN_FEATURES = 193
_B_OUT_FEATURES = 5


def _load_b() -> dict:
    return json.loads(_BUNDLE_B_ENGINE.read_text())


def test_b_bundle_id_short_prefix_pinned():
    """Live ``_meta.bundle_id`` short prefix MUST match ``5f7ed5f6``."""
    bundle = _load_b()
    live_id = bundle["_meta"]["bundle_id"]
    assert live_id == _PINNED_BUNDLE_ID_B, (
        f"B bundle_id drifted: live={live_id[:16]}…, "
        f"pinned={_PINNED_BUNDLE_ID_B[:16]}… The B weights have changed; "
        f"this pin AND every iter-421 doc reference must rotate in lockstep."
    )
    assert live_id.startswith(_B_SHORT), (
        f"B bundle short-prefix mismatch: {live_id[:8]} vs pinned {_B_SHORT}"
    )


def test_b_bundle_file_size_within_pinned_band():
    """File size stays within ~30 KB ±2 KB; never exceeds 60 KB.

    Catches: stray-key inflation, JSON pretty-print drift, hostile
    or buggy build pipeline that doubles the file."""
    size = os.path.getsize(_BUNDLE_B_ENGINE)
    assert size <= _B_SIZE_HARD_CEILING, (
        f"B bundle size {size} bytes exceeds hard ceiling "
        f"{_B_SIZE_HARD_CEILING} bytes. Something inflated the JSON "
        f"beyond a deliberate refactor."
    )
    assert _B_SIZE_LO <= size <= _B_SIZE_HI, (
        f"B bundle size {size} bytes outside expected band "
        f"[{_B_SIZE_LO}, {_B_SIZE_HI}] (~30 KB ±2 KB). Either weight "
        f"count changed (architectural shift requires deliberate band "
        f"rotation) or JSON encoding drifted."
    )


def test_b_ternary_sparsity_within_band():
    """Ternary weight sparsity (fraction of zeros) MUST stay between
    40% and 60%.

    Same band as A bundle since both use the same STE quantization
    recipe. A retrained B bundle that accidentally produced all-±1
    weights (sparsity → 0) would break the BitNet b1.58 thesis;
    sparsity > 60% would indicate over-pruning that hurts recall."""
    bundle = _load_b()
    ternary: list[int] = []
    for row in bundle["hidden_w"]:
        ternary.extend(row)
    for row in bundle["output_w"]:
        ternary.extend(row)
    zeros = sum(1 for x in ternary if x == 0)
    sparsity = zeros / len(ternary)
    assert _B_SPARSITY_LO <= sparsity <= _B_SPARSITY_HI, (
        f"B ternary sparsity {sparsity:.4f} outside band "
        f"[{_B_SPARSITY_LO}, {_B_SPARSITY_HI}]. Retrain produced "
        f"{zeros}/{len(ternary)} zero weights — investigate STE "
        f"thresholding."
    )


def test_b_toplevel_key_set_canonical():
    """Top-level JSON keys MUST be exactly the 5-element canonical set,
    identical to A's shape."""
    bundle = _load_b()
    live_keys = frozenset(bundle.keys())
    assert live_keys == _B_TOPLEVEL_REQUIRED, (
        f"B bundle top-level keys drifted:\n"
        f"  expected: {sorted(_B_TOPLEVEL_REQUIRED)}\n"
        f"  live:     {sorted(live_keys)}\n"
        f"  added:    {sorted(live_keys - _B_TOPLEVEL_REQUIRED)}\n"
        f"  missing:  {sorted(_B_TOPLEVEL_REQUIRED - live_keys)}"
    )


def test_b_meta_provenance_fields_present():
    """``_meta`` MUST carry all 22 provenance fields recorded at the
    iter-421 sweep. FDA SaMD audit-trail integrity depends on these
    being permanent."""
    bundle = _load_b()
    meta = bundle["_meta"]
    live = frozenset(meta.keys())
    missing = _B_META_REQUIRED - live
    assert not missing, (
        f"B _meta missing FDA SaMD provenance fields: {sorted(missing)}. "
        f"These are part of the audit trail; restoring them requires "
        f"re-running the iter-421 sweep with the same seed."
    )


def test_b_meta_self_referenced_bundle_id_consistent():
    """``_meta.bundle_id`` must equal the SHA-256 of the canonical-form
    weight payload. A self-inconsistent bundle is a corruption signal."""
    bundle = _load_b()
    payload = {
        "hidden_w": bundle["hidden_w"],
        "hidden_b": bundle["hidden_b"],
        "output_w": bundle["output_w"],
        "output_b": bundle["output_b"],
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    computed = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    assert bundle["_meta"]["bundle_id"] == computed, (
        f"B _meta.bundle_id self-inconsistent:\n"
        f"  recorded: {bundle['_meta']['bundle_id'][:16]}…\n"
        f"  computed: {computed[:16]}…\n"
        f"Either weights were edited without rotating bundle_id, or "
        f"the canonical-form serialization changed (different json "
        f"library?). Investigate before promoting B to engine."
    )


def test_b_meta_serious_recall_is_one():
    """``_meta.serious_recall = 1.0`` is the iter-421 breakthrough.
    B was trained EXCLUSIVELY on the 95 non-contra samples; its job is
    to close the v8 standalone serious gap (84% → 100%) on the live
    cohort. A non-1.0 value would indicate the iter-421 gate failed."""
    bundle = _load_b()
    recall = bundle["_meta"]["serious_recall"]
    assert recall == 1.0, (
        f"B _meta.serious_recall = {recall}, expected 1.0. iter-421 gate "
        f"requires 100% on serious; if this fires the wrong bundle was "
        f"committed (e.g., a non-gate-passing seed)."
    )


def test_b_meta_moderate_recall_is_one():
    """``_meta.moderate_recall = 1.0`` — closes v8 standalone moderate
    gap (91% → 100%) on the live cohort."""
    bundle = _load_b()
    recall = bundle["_meta"]["moderate_recall"]
    assert recall == 1.0, (
        f"B _meta.moderate_recall = {recall}, expected 1.0. iter-421 gate "
        f"requires 100% on moderate; if this fires the wrong bundle was "
        f"committed."
    )


def test_b_meta_major_recall_is_one():
    """``_meta.major_recall = 1.0`` — A still gets 4/4 majors but B
    reinforces this on its own cohort."""
    bundle = _load_b()
    recall = bundle["_meta"]["major_recall"]
    assert recall == 1.0, (
        f"B _meta.major_recall = {recall}, expected 1.0. iter-421 gate "
        f"requires 100% on major (cascades through to ensemble result)."
    )


def test_b_meta_major_fp_zero():
    """``_meta.major_fp_within_non_contra = 0`` — under iter-421's
    constrained argmax B never over-vetoes serious/moderate as major
    on the non-contra cohort. A non-zero value would indicate the
    over-veto pattern v8 had on warfarin/NTI cluster wasn't fixed."""
    bundle = _load_b()
    fp = bundle["_meta"]["major_fp_within_non_contra"]
    assert fp == 0, (
        f"B _meta.major_fp_within_non_contra = {fp}, expected 0. The "
        f"NTI-cluster over-veto fix didn't land or was reverted."
    )


def test_b_meta_provenance_values_pinned():
    """Pin EXACT VALUES of safety-critical provenance fields, not just
    existence. Catches drift mode where someone hand-edits _meta values
    without rotating bundle_id (which is hashed only over weight
    matrices)."""
    bundle = _load_b()
    meta = bundle["_meta"]
    pinned = {
        "schema": "bitnet_classifier_v3_atc_flags",
        "training_iter": "iter-421-path-b-bitnet-b-specialist",
        "role": "tier_2_serious_moderate_major_specialist",
        "weight_dtype": "ternary",
        "bias_dtype": "q16.16",
        "trained_with": "PyTorch + STE",
        "in_features": _B_IN_FEATURES,
        "hidden_features": _B_HIDDEN_FEATURES,
        "out_features": _B_OUT_FEATURES,
        "flag_keys_count": 26,
        "pair_derived_rule_count": 13,
        "ensemble_partner_bundle_id_a": _PINNED_BUNDLE_ID_A,
    }
    for key, expected in pinned.items():
        live = meta.get(key)
        assert live == expected, (
            f"B _meta.{key} drifted:\n"
            f"  pinned: {expected!r}\n"
            f"  live:   {live!r}\n"
            f"Hand-editing _meta values without retraining the bundle "
            f"breaks the FDA SaMD provenance chain."
        )


def test_b_architecture_dims_match_meta():
    """Weight matrix shapes MUST match _meta dim claims. A mismatch
    means either _meta drifted from the actual weights or the weights
    were partially regenerated."""
    bundle = _load_b()
    meta = bundle["_meta"]

    hidden_w = bundle["hidden_w"]
    output_w = bundle["output_w"]
    hidden_b = bundle["hidden_b"]
    output_b = bundle["output_b"]

    assert len(hidden_w) == meta["hidden_features"], (
        f"hidden_w rows {len(hidden_w)} != _meta.hidden_features "
        f"{meta['hidden_features']}"
    )
    assert all(len(row) == meta["in_features"] for row in hidden_w), (
        f"hidden_w cols not all == _meta.in_features {meta['in_features']}"
    )
    assert len(hidden_b) == meta["hidden_features"], (
        f"hidden_b length {len(hidden_b)} != _meta.hidden_features "
        f"{meta['hidden_features']}"
    )
    assert len(output_w) == meta["out_features"], (
        f"output_w rows {len(output_w)} != _meta.out_features "
        f"{meta['out_features']}"
    )
    assert all(len(row) == meta["hidden_features"] for row in output_w), (
        f"output_w cols not all == _meta.hidden_features "
        f"{meta['hidden_features']}"
    )
    assert len(output_b) == meta["out_features"], (
        f"output_b length {len(output_b)} != _meta.out_features "
        f"{meta['out_features']}"
    )


def test_b_engine_retrain_copies_byte_identical():
    """Engine + retrain_runpod copies of bundle B MUST be byte-identical
    SHA-256. Drift here means one was edited without the other —
    auditor reconstruction would diverge depending on which copy they
    found first."""
    engine_sha = hashlib.sha256(_BUNDLE_B_ENGINE.read_bytes()).hexdigest()
    retrain_sha = hashlib.sha256(_BUNDLE_B_RETRAIN.read_bytes()).hexdigest()
    assert engine_sha == retrain_sha, (
        f"Bundle B engine ↔ retrain copies diverged:\n"
        f"  engine:           {engine_sha[:16]}…  ({_BUNDLE_B_ENGINE})\n"
        f"  retrain_runpod:   {retrain_sha[:16]}…  ({_BUNDLE_B_RETRAIN})\n"
        f"One copy was edited without the other. Re-sync via "
        f"`cp retrain_runpod/bitnet_weights_b_specialist.json "
        f"engine/bitnet_weights_b_specialist.json` (or vice versa "
        f"depending on which is the source-of-truth)."
    )
