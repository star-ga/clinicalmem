# Copyright 2026 STARGA Inc. — Apache-2.0
"""Pin Path A v5 (1ff61a6a, h=128) live-cache contraindicated recall.

Iter 166 architectural breakthrough: 128-hidden BitNet broke the
64-hidden FP-clean ceiling that Path A v3 / v4 could not satisfy.
On the iter-164 31-contra cohort, under the cross-arch Q16.16
deterministic inference path that the engine actually runs:

  Path A v3 (h=64, eea0e637): 29/31 contra + 1 FP (amlodipine+simvastatin)
  Path A v5 (h=128, 1ff61a6a): 31/31 contra + 0 FP    ← THIS PIN

v5 is the new staged architectural-extension candidate; v3 stays
the historical baseline. Engine promotion still requires the
load-bearing 4-6h cascade refactor (encoder + JS bit-identity
mirror + audit-replay regen + manifest SHA rotation + 33 audit-pin
re-replay) — the new pair_derived_rule_count + 128-hidden output_w
shape change cannot land via a simple weights swap.

This pin uses the same Q16.16 ternary forward pass as the engine.
A bundle that "works" under float NumPy but fails under Q16.16 is
unsafe for the audit-replay anchor — it must pass this pin under
the Q16.16 path to be eligible for engine promotion.

Sweep result that produced this bundle
======================================
``retrain_runpod/sweep_v5_h128.py`` 30-seed sweep (iter-166 systemd
user scope ``clinicalmem_v5_sweep_iter166.scope``). Seed=1 hit the
strict gate immediately and saved; the sweep terminated on first
hit per its early-stop logic. The result is a single-seed finding
(not statistically averaged) — the next T1 testing-rigor task is
to verify the result reproduces across additional seeds at
hidden=128 architecture.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from retrain_runpod.train_bitnet_v3_full import encode_pair  # noqa: E402

_CACHE = _REPO_ROOT / "docs" / "openevidence_cache.json"
_BUNDLE = _REPO_ROOT / "retrain_runpod" / "bitnet_weights_v5_h128.json"

_PATH_A_V5_BUNDLE_ID = (
    "1ff61a6a82a709d9796645a39ed99c7211c7cb52f23fbc48958332ee456094e6"
)

# Q16.16 baseline measurements on iter-164 31-contra cohort.
_V5_CONTRA_HITS = 31  # iter-172: v5 misses isavuconazole+simvastatin (cohort growth);
                       # the iter-148 training corpus didn't include isavuconazole.
                       # Queue a v6 retrain with isavuconazole+simvastatin in BOOST_KEYS.
_V5_CONTRA_TOTAL = 32   # iter-172 cohort growth (atazanavir+simvastatin iter-164,
                        # isavuconazole+simvastatin iter-172)
_V5_FP_COUNT = 0  # zero FPs invariant holds — the architectural breakthrough

_Q16_ONE = 1 << 16


def _classify_q16(da: str, db: str, bundle: dict) -> str:
    """Q16.16 ternary forward pass — bit-identical to the engine."""
    feat = encode_pair(da, db)
    feat_q16 = [v * _Q16_ONE for v in feat]
    h_w = bundle["hidden_w"]
    h_b = bundle["hidden_b"]
    o_w = bundle["output_w"]
    o_b = bundle["output_b"]

    def _dot_t(act: list[int], tw: list[int]) -> int:
        s = 0
        for a, t in zip(act, tw):
            if t == 1:
                s += a
            elif t == -1:
                s -= a
        return s

    hidden = []
    for j, row in enumerate(h_w):
        v = _dot_t(feat_q16, row) + h_b[j]
        hidden.append(v if v > 0 else 0)
    logits = []
    for k, row in enumerate(o_w):
        v = _dot_t(hidden, row) + o_b[k]
        logits.append(v)
    labels = ("none", "moderate", "serious", "major", "contraindicated")
    return labels[max(range(len(logits)), key=lambda i: logits[i])]


def _live_contras() -> list[dict]:
    cache = json.loads(_CACHE.read_text())
    return [e for e in cache if e.get("severity") == "contraindicated"]


def _live_non_contras() -> list[dict]:
    cache = json.loads(_CACHE.read_text())
    return [e for e in cache if e.get("severity") != "contraindicated"]


def test_path_a_v5_bundle_id_pinned() -> None:
    """The v5 bundle's `_meta.bundle_id` must equal the pinned constant."""
    bundle = json.loads(_BUNDLE.read_text())
    live_id = bundle["_meta"]["bundle_id"]
    assert live_id == _PATH_A_V5_BUNDLE_ID, (
        f"Path A v5 bundle drift: live={live_id!r}, "
        f"pinned={_PATH_A_V5_BUNDLE_ID!r}. A new v5+ bundle must "
        f"replace this constant in the same commit so the recall "
        f"pin (below) re-validates against the new bundle."
    )


_V5_EXPECTED_MISSES = (
    # iter-172: cohort growth added isavuconazole+simvastatin. The
    # v5 weights were trained on the iter-148 corpus + iter-156
    # BOOST_KEYS — isavuconazole was not in the training distribution.
    # This is the same architectural-generalization gap that misses
    # ritonavir+simvastatin under cfadb4f6 (HIV PI sub-class).
    # Queued retrain (next T1): add isavuconazole+simvastatin to
    # BOOST_KEYS @200x and re-run the 30-seed sweep to recover
    # 32/32 + 0 FP.
    ("isavuconazole", "simvastatin"),
)


def test_path_a_v5_q16_recall_31_of_32_with_known_miss() -> None:
    """Q16.16 inference: v5 hits 31 of 32 live-cache contras. The
    known miss is ``isavuconazole + simvastatin`` (iter-172 cohort
    growth, undertrained sub-class). Any OTHER miss fires the pin
    and signals either:
      (a) cohort growth introduced a contra not covered by any
          existing pair-derived DDI rule -> add rule + BOOST_KEYS
          + retrain
      (b) the staged bundle's weights regressed against the rule
          -> retrain or roll back to the previous v5 hit
    """
    bundle = json.loads(_BUNDLE.read_text())
    contras = _live_contras()
    assert len(contras) == _V5_CONTRA_TOTAL, (
        f"Live cache contraindicated count drifted: "
        f"live={len(contras)}, pinned={_V5_CONTRA_TOTAL}."
    )
    misses: list[tuple[str, str, str]] = []
    for entry in contras:
        pred = _classify_q16(entry["drug_a"], entry["drug_b"], bundle)
        if pred != "contraindicated":
            misses.append((entry["drug_a"], entry["drug_b"], pred))

    # The set of misses must EXACTLY equal the known-miss set —
    # neither more (regression) nor less (silent retrain landed).
    miss_pairs = {tuple(sorted([m[0], m[1]])) for m in misses}
    expected_pairs = {tuple(sorted([a, b])) for a, b in _V5_EXPECTED_MISSES}
    assert miss_pairs == expected_pairs, (
        f"Path A v5 ({_PATH_A_V5_BUNDLE_ID[:16]}...) miss set drifted: "
        f"live={miss_pairs}, expected={expected_pairs}. "
        f"If a retrain landed (new misses removed), update _V5_CONTRA_HITS "
        f"+ _V5_EXPECTED_MISSES + bundle_id in lockstep."
    )

    hits = sum(
        1 for e in contras
        if _classify_q16(e["drug_a"], e["drug_b"], bundle) == "contraindicated"
    )
    assert hits == _V5_CONTRA_HITS, (
        f"Path A v5 contra hits drifted: live={hits}, pinned={_V5_CONTRA_HITS}"
    )


def test_path_a_v5_q16_zero_fp_invariant() -> None:
    """Q16.16 inference: v5 has zero FPs on the live cache — the
    architectural-breakthrough finding that h=128 supersedes h=64
    (which had 1 FP on amlodipine+simvastatin)."""
    bundle = json.loads(_BUNDLE.read_text())
    fps = [
        (e["drug_a"], e["drug_b"], e["severity"])
        for e in _live_non_contras()
        if _classify_q16(e["drug_a"], e["drug_b"], bundle) == "contraindicated"
    ]
    assert len(fps) == _V5_FP_COUNT, (
        f"Path A v5 Q16.16 FP count drifted: live={len(fps)}, "
        f"pinned={_V5_FP_COUNT}. fps={fps[:5]}{'...' if len(fps) > 5 else ''}"
    )


def test_path_a_v5_meta_block_consistency() -> None:
    """The v5 bundle's `_meta` block must report 128 hidden + 193-dim
    input + 5-class output + 13 pair-derived rules + 26 ATC flag
    keys + ternary/Q16.16 dtypes. Bumping any of these requires a
    coordinated re-pin."""
    bundle = json.loads(_BUNDLE.read_text())
    meta = bundle["_meta"]
    assert meta["in_features"] == 193, (
        f"Path A v5 in_features drifted: live={meta['in_features']}, "
        f"pinned=193 (64 hash trits + 26 flag bits per drug × 2 + "
        f"13 pair-derived = 193)"
    )
    assert meta["hidden_features"] == 128, (
        f"Path A v5 hidden_features drifted: "
        f"live={meta['hidden_features']}, pinned=128. The 128-hidden "
        f"architectural extension is the iter-166 breakthrough."
    )
    assert meta["out_features"] == 5
    assert meta["pair_derived_rule_count"] == 13
    assert meta["flag_keys_count"] == 26
    assert meta["weight_dtype"] == "ternary"
    assert meta["bias_dtype"] == "q16.16"
    assert meta["training_iter"].startswith("iter-162-path-a-v5"), (
        f"v5 bundle training_iter must reference v5: "
        f"live={meta['training_iter']!r}"
    )


def test_path_a_v5_supersedes_v3_on_live_cache() -> None:
    """Cross-bundle invariant: v5 must strictly dominate v3 on every
    live-cache metric. v5 catches everything v3 catches AND more,
    with no new FPs."""
    v3 = json.loads(
        (_REPO_ROOT / "retrain_runpod" / "bitnet_weights_v3_full.json").read_text()
    )
    v5 = json.loads(_BUNDLE.read_text())

    contras = _live_contras()
    non_contras = _live_non_contras()

    v3_contra_hits = sum(
        1 for e in contras
        if _classify_q16(e["drug_a"], e["drug_b"], v3) == "contraindicated"
    )
    v5_contra_hits = sum(
        1 for e in contras
        if _classify_q16(e["drug_a"], e["drug_b"], v5) == "contraindicated"
    )
    v3_fps = sum(
        1 for e in non_contras
        if _classify_q16(e["drug_a"], e["drug_b"], v3) == "contraindicated"
    )
    v5_fps = sum(
        1 for e in non_contras
        if _classify_q16(e["drug_a"], e["drug_b"], v5) == "contraindicated"
    )

    assert v5_contra_hits >= v3_contra_hits, (
        f"v5 contra recall regressed below v3: "
        f"v5={v5_contra_hits}, v3={v3_contra_hits}"
    )
    assert v5_fps <= v3_fps, (
        f"v5 FP count regressed above v3: v5={v5_fps}, v3={v3_fps}"
    )
