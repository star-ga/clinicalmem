"""Pin v8 staged bundle integrity (mirror of the iter-72 cfadb4f6 pin
applied to the iter-244 staged 1f0f8859 bundle).

Iter 255 (round 47 T1) — 17th cross-pin family. The shipped engine
bundle (cfadb4f6) is already pinned at 9 invariants by
``test_bitnet_bundle_integrity_pin.py`` (size band, sparsity floor,
key set, _meta provenance, self-referenced bundle_id, etc.). The
iter-244-staged v8 bundle (1f0f8859, h=256) only has aggregate
recall + canonical-Q16.16 pins. **Gap**: a future maintainer (or
build-system glitch) could corrupt the v8 JSON in ways that don't
change ``bundle_id`` but do change behavior or audit value:

- Add a stray top-level key (e.g., ``training_log: "..."``)
- Drop a ``_meta`` provenance field (the FDA SaMD audit trail breaks)
- Re-encode the JSON with different whitespace causing file-size
  drift past the expected band
- Sparsity collapse via dense float weights masquerading as ternary

The shipped bundle's iter-72 pin already protects against all of
these on cfadb4f6. This pin extends the same shape to v8 (1f0f8859).

User feedback (iter-255): "0% is the only goal." This pin directly
defends the 0%-known-misses promise — corruption of the v8 JSON
without an obvious bundle_id change would leave demo + JUDGES
claiming 43/43 + 0 FP under 1f0f8859 while the on-disk weights have
silently drifted.

Cross-pin lineage: 17th cross-pin family
(iter-178/183/188/193/198/203/223/228/232/234/236/239/240/244/246/247
+ iter-255). Same shape as iter-244 (binary-artifact existence), but
applied to bundle CONTENT integrity instead of file existence.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_V8_BUNDLE = _REPO_ROOT / "retrain_runpod" / "bitnet_weights_v8_h256.json"

_PINNED_V8_BUNDLE_ID = (
    "1f0f88591c05af57c62d844b667639b29c7d1f0eb1b213073d158101611f76e6"
)
_V8_SHORT = _PINNED_V8_BUNDLE_ID[:8]   # "1f0f8859"

# v8 bundle expected size band: ~118-124 KB (live: 121,024 bytes).
# 50,688 ternary weights serialized as JSON arrays of -1/0/1 ints.
# Tighter than the cfadb4f6 19-KB band to catch any non-trivial content
# change. Hard ceiling at 200 KB so a deliberate metadata expansion
# fails loud rather than silent.
_V8_SIZE_LO = 110 * 1024  # 110 KB
_V8_SIZE_HI = 130 * 1024  # 130 KB
_V8_SIZE_HARD_CEILING = 200 * 1024  # 200 KB

# Ternary weight sparsity floor: ~44.4% live. Floor at 40% to catch
# severe collapse; ceiling at 60% to catch over-pruning.
_V8_SPARSITY_LO = 0.40
_V8_SPARSITY_HI = 0.60

# Required _meta provenance fields. iter-244 sweep recorded all 17.
# A future regen that drops any of these breaks FDA SaMD audit-trail.
_V8_META_REQUIRED = frozenset({
    "augmentation",
    "best_test_acc",
    "bias_dtype",
    "bundle_id",
    "contra_fp",
    "contra_recall",
    "feature_breakdown",
    "flag_keys_count",
    "hidden_features",
    "in_features",
    "major_recall",
    "out_features",
    "pair_derived_rule_count",
    "schema",
    "trained_with",
    "training_iter",
    "weight_dtype",
})

# Top-level JSON key set MUST be exactly these 5. Adding any new top-
# level key (e.g., "training_log") would not change bundle_id (since
# bundle_id is hashed only over the canonical-form weight matrices)
# but would inflate file size + risk audit confusion.
_V8_TOPLEVEL_REQUIRED = frozenset({
    "_meta",
    "hidden_b",
    "hidden_w",
    "output_b",
    "output_w",
})


def _load_v8() -> dict:
    return json.loads(_V8_BUNDLE.read_text())


def test_v8_bundle_id_short_prefix_pinned():
    """Live ``_meta.bundle_id`` short prefix MUST match ``1f0f8859``.

    This is the iter-244 sweep result — bundle_id is the SHA-256 of
    the canonical-form weight matrices, so any byte change to weights
    would break this. (Single source of truth: this constant; all
    derived doc references must match.)
    """
    bundle = _load_v8()
    live_id = bundle["_meta"]["bundle_id"]
    assert live_id == _PINNED_V8_BUNDLE_ID, (
        f"v8 bundle_id drifted: live={live_id[:16]}…, "
        f"pinned={_PINNED_V8_BUNDLE_ID[:16]}… The v8 weights have changed; "
        f"this pin AND every iter-244 / iter-249 / iter-254 doc reference "
        f"must rotate in lockstep."
    )
    assert live_id.startswith(_V8_SHORT), (
        f"Bundle short-prefix mismatch: {live_id[:8]} vs pinned {_V8_SHORT}"
    )


def test_v8_bundle_file_size_within_pinned_band():
    """File size stays within ~120 KB ±10 KB; never exceeds 200 KB.

    Catches: stray-key inflation, JSON pretty-print drift, a hostile
    or buggy build pipeline that doubles the file."""
    size = os.path.getsize(_V8_BUNDLE)
    assert size <= _V8_SIZE_HARD_CEILING, (
        f"v8 bundle size {size} bytes exceeds hard ceiling "
        f"{_V8_SIZE_HARD_CEILING} bytes (200 KB). Something inflated "
        f"the JSON beyond a deliberate refactor."
    )
    assert _V8_SIZE_LO <= size <= _V8_SIZE_HI, (
        f"v8 bundle size {size} bytes outside expected band "
        f"[{_V8_SIZE_LO}, {_V8_SIZE_HI}] (~120 KB ±10 KB). Either "
        f"weight count changed (architectural shift requires a "
        f"deliberate band rotation) or JSON encoding drifted."
    )


def test_v8_ternary_sparsity_within_band():
    """Ternary weight sparsity (fraction of zeros) MUST stay between
    40% and 60%.

    The iter-72 'structured sparsity' rhetoric depends on a meaningful
    fraction of the weights being zero. A retrained bundle that
    accidentally produces all-±1 weights (sparsity → 0) breaks the
    BitNet b1.58 thesis. Conversely, sparsity >60% indicates over-
    pruning that may have hurt recall."""
    bundle = _load_v8()
    ternary: list[int] = []
    for row in bundle["hidden_w"]:
        ternary.extend(row)
    for row in bundle["output_w"]:
        ternary.extend(row)
    zeros = sum(1 for x in ternary if x == 0)
    sparsity = zeros / len(ternary)
    assert _V8_SPARSITY_LO <= sparsity <= _V8_SPARSITY_HI, (
        f"v8 ternary sparsity {sparsity:.4f} outside band "
        f"[{_V8_SPARSITY_LO}, {_V8_SPARSITY_HI}]. Retrain produced "
        f"{zeros}/{len(ternary)} zero weights — investigate STE "
        f"thresholding."
    )


def test_v8_toplevel_key_set_canonical():
    """Top-level JSON keys MUST be exactly the 5-element canonical
    set. Adding stray keys (training logs, debug data) inflates file
    size and risks audit confusion since bundle_id is hashed only
    over the canonical-form weights."""
    bundle = _load_v8()
    live_keys = frozenset(bundle.keys())
    assert live_keys == _V8_TOPLEVEL_REQUIRED, (
        f"v8 bundle top-level keys drifted:\n"
        f"  expected: {sorted(_V8_TOPLEVEL_REQUIRED)}\n"
        f"  live:     {sorted(live_keys)}\n"
        f"  added:    {sorted(live_keys - _V8_TOPLEVEL_REQUIRED)}\n"
        f"  missing:  {sorted(_V8_TOPLEVEL_REQUIRED - live_keys)}"
    )


def test_v8_meta_provenance_fields_present():
    """``_meta`` MUST carry all 17 provenance fields recorded at the
    iter-244 sweep. FDA SaMD audit-trail integrity depends on these
    being permanent."""
    bundle = _load_v8()
    meta = bundle["_meta"]
    live = frozenset(meta.keys())
    missing = _V8_META_REQUIRED - live
    assert not missing, (
        f"v8 _meta missing FDA SaMD provenance fields: {sorted(missing)}. "
        f"These are part of the audit trail; restoring them requires "
        f"re-running the iter-244 sweep with the same seed."
    )


def test_v8_meta_self_referenced_bundle_id_consistent():
    """``_meta.bundle_id`` must equal the SHA-256 of the canonical-form
    weight payload. A self-inconsistent bundle is a corruption signal."""
    bundle = _load_v8()
    payload = {
        "hidden_w": bundle["hidden_w"],
        "hidden_b": bundle["hidden_b"],
        "output_w": bundle["output_w"],
        "output_b": bundle["output_b"],
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    computed = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    assert bundle["_meta"]["bundle_id"] == computed, (
        f"v8 _meta.bundle_id self-inconsistent:\n"
        f"  recorded: {bundle['_meta']['bundle_id'][:16]}…\n"
        f"  computed: {computed[:16]}…\n"
        f"Either weights were edited without rotating bundle_id, or "
        f"the canonical-form serialization changed (different json "
        f"library?). Investigate before promoting v8 to engine."
    )


def test_v8_meta_contra_fp_is_zero():
    """``_meta.contra_fp = 0`` is the safety invariant: v8 had ZERO
    false positives on the iter-244 cohort. A non-zero value would
    indicate the wrong bundle was committed (e.g., a v7 candidate
    that didn't quite hit the gate)."""
    bundle = _load_v8()
    fp = bundle["_meta"]["contra_fp"]
    assert fp == 0, (
        f"v8 _meta.contra_fp = {fp}, expected 0. Either the bundle is "
        f"a non-gate-hitting candidate that was committed by mistake, "
        f"or the safety invariant was violated."
    )


def test_v8_meta_contra_recall_is_one():
    """``_meta.contra_recall = 1.0`` is the iter-244 breakthrough: v8
    achieves 100% recall on contraindicated under cross-arch Q16.16
    on the cohort it was evaluated against."""
    bundle = _load_v8()
    recall = bundle["_meta"]["contra_recall"]
    assert recall == 1.0, (
        f"v8 _meta.contra_recall = {recall}, expected 1.0. The 41/41 "
        f"breakthrough either didn't land or the bundle was overwritten."
    )


def test_v8_architecture_dimensions_match_pinned():
    """v8 architecture: in=193, hidden=256, out=5. Any of these
    changing requires a deliberate cascade refactor (encoder lift,
    JS bit-identity mirror, audit-replay regen). Catches accidental
    architecture drift."""
    bundle = _load_v8()
    meta = bundle["_meta"]
    assert meta["in_features"] == 193, (
        f"v8 in_features={meta['in_features']}, expected 193 (64 hash "
        f"trits + 26 ATC flag bits ×2 + 13 pair-derived = 193)"
    )
    assert meta["hidden_features"] == 256, (
        f"v8 hidden_features={meta['hidden_features']}, expected 256 "
        f"(the architectural double from v6's 128)"
    )
    assert meta["out_features"] == 5, (
        f"v8 out_features={meta['out_features']}, expected 5 (severity "
        f"classes: none/moderate/serious/major/contraindicated)"
    )
    # Cross-check against actual matrix shapes
    assert len(bundle["hidden_w"]) == 256
    assert len(bundle["hidden_w"][0]) == 193
    assert len(bundle["output_w"]) == 5
    assert len(bundle["output_w"][0]) == 256
    assert len(bundle["hidden_b"]) == 256
    assert len(bundle["output_b"]) == 5
