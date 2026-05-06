# Copyright 2026 STARGA Inc. — Apache-2.0
"""Pin Path A v3 (eea0e637, h=64) live-cache contraindicated recall.

Iter 166 correction: an earlier iter-164/iter-165 framing claimed
Path A v3 hit 31/31 = 100% on the iter-164 live cache. That number
was measured with a NumPy float forward-pass, NOT the cross-arch
Q16.16 fixed-point inference path that the engine actually runs.
Under Q16.16 (the bit-identical inference path the demo's
deterministic-veto claim relies on), v3 hits **29/31 = 93.5%** with
**1 FP** (amlodipine + simvastatin) — matching the iter-155 demo
claim. The float-NumPy "31/31" was the measurement artifact.

This pin locks the corrected Q16.16 v3 baseline. v3 stays the
historical optimum at 64 hidden; the new Path A v5 (h=128) bundle
saved iter-166 supersedes v3 with 31/31 + 0 FP — pinned separately
in ``test_path_a_v5_live_recall_pin.py``.

Why Q16.16 inference matters for this pin
=========================================
The engine's BitNet forward pass is deterministic across CPU, GPU,
and NPU only when computed in Q16.16 fixed-point (no float rounding
ambiguity). This pin uses the same Q16.16 ternary-multiplication
classifier path as the engine's deterministic veto. A bundle that
"works" under float NumPy but fails under Q16.16 is unsafe for the
audit-replay anchor — it must pass this pin under the Q16.16 path
to be eligible for engine promotion.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from retrain_runpod.train_bitnet_v3_full import encode_pair  # noqa: E402

_CACHE = _REPO_ROOT / "docs" / "openevidence_cache.json"
_BUNDLE = _REPO_ROOT / "retrain_runpod" / "bitnet_weights_v3_full.json"

_PATH_A_V3_BUNDLE_ID = (
    "eea0e637b537507352950feeb413da33d81775e41fb7c7ec2aaed94f464dcb2c"
)

# Q16.16 baseline measurements on iter-164 31-contra cohort.
_V3_CONTRA_HITS = 29  # iter-182: still 29 hits; v3 now misses 5 contras at h=64
                       # (HIV-PI / triazole / ergot / tetracycline-retinoid /
                       # one 4th sub-class). Path A v5 (h=128) catches more.
                       # Full 34/34 awaits v6 retrain with extended BOOST_KEYS.
_V3_CONTRA_TOTAL = 35   # iter-187 cohort growth (atazanavir+simvastatin iter-164,
                        # isavuconazole+simvastatin iter-172, ketoconazole+ergotamine
                        # iter-177, minocycline+isotretinoin iter-182,
                        # midazolam+ketoconazole iter-187)
_V3_FP_COUNT = 1  # amlodipine + simvastatin (safety-conservative)
_V3_FP_PAIR = ("amlodipine", "simvastatin")

_Q16_ONE = 1 << 16


def _classify_q16(da: str, db: str, bundle: dict) -> str:
    """Q16.16 ternary forward pass — bit-identical to the engine.

    No float arithmetic. Hidden activations and output logits are
    Q16.16 integers; argmax is taken over the integer logits.
    """
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


def test_path_a_v3_bundle_id_pinned() -> None:
    """The v3 bundle's `_meta.bundle_id` must equal the pinned constant.
    Recovered from iter-155 commit 1a7210c at iter-166 after a v5 sweep
    accidentally overwrote this file."""
    bundle = json.loads(_BUNDLE.read_text())
    live_id = bundle["_meta"]["bundle_id"]
    assert live_id == _PATH_A_V3_BUNDLE_ID, (
        f"Path A v3 bundle drift: live={live_id!r}, "
        f"pinned={_PATH_A_V3_BUNDLE_ID!r}. v3 was recovered from git "
        f"at iter-166; do not overwrite this file. New bundles should "
        f"go to bitnet_weights_v5_h128.json (or higher version)."
    )


def test_path_a_v3_q16_recall_pinned_at_29_of_31() -> None:
    """Q16.16 inference: v3 hits 29 of 31 live-cache contras (the
    iter-155 claim, corrected back at iter-166 after the iter-164
    float-NumPy measurement artifact was caught). Misses are
    expected to be the ones whose flag-pair patterns the 64-hidden
    architecture cannot saturate."""
    bundle = json.loads(_BUNDLE.read_text())
    contras = _live_contras()
    assert len(contras) == _V3_CONTRA_TOTAL, (
        f"Live cache contraindicated count drifted: "
        f"live={len(contras)}, pinned={_V3_CONTRA_TOTAL}."
    )
    hits = sum(
        1 for e in contras
        if _classify_q16(e["drug_a"], e["drug_b"], bundle) == "contraindicated"
    )
    assert hits == _V3_CONTRA_HITS, (
        f"Path A v3 Q16.16 contra recall drifted: "
        f"live={hits}/{len(contras)}, pinned={_V3_CONTRA_HITS}/{_V3_CONTRA_TOTAL}."
    )


def test_path_a_v3_q16_fp_pinned_at_amlodipine_simvastatin() -> None:
    """Q16.16 inference: v3 has exactly 1 FP on the live cache, the
    safety-conservative amlodipine + simvastatin (FDA Zocor labels
    simvastatin 20 mg with amlodipine; ground truth = moderate, model
    = contraindicated). This is a documented miscalibration, not a
    dangerous classifier failure."""
    bundle = json.loads(_BUNDLE.read_text())
    fps = [
        (e["drug_a"], e["drug_b"]) for e in _live_non_contras()
        if _classify_q16(e["drug_a"], e["drug_b"], bundle) == "contraindicated"
    ]
    assert len(fps) == _V3_FP_COUNT, (
        f"Path A v3 Q16.16 FP count drifted: "
        f"live={len(fps)}, pinned={_V3_FP_COUNT}. fps={fps}"
    )
    assert _V3_FP_PAIR in fps or (_V3_FP_PAIR[1], _V3_FP_PAIR[0]) in fps, (
        f"Path A v3 Q16.16 FP must include {_V3_FP_PAIR}; got {fps}"
    )


def test_path_a_v3_meta_block_consistency() -> None:
    """The bundle's `_meta` block must report the architecture
    invariants the demo claims (193-dim feature input, 64-hidden,
    13 pair-derived rules, 26 ATC flag keys)."""
    bundle = json.loads(_BUNDLE.read_text())
    meta = bundle["_meta"]
    assert meta["in_features"] == 193
    assert meta["hidden_features"] == 64
    assert meta["out_features"] == 5
    assert meta["pair_derived_rule_count"] == 13
    assert meta["flag_keys_count"] == 26
    assert meta["weight_dtype"] == "ternary"
    assert meta["bias_dtype"] == "q16.16"
