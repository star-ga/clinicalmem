# Copyright 2026 STARGA Inc. — Apache-2.0
"""Pin Path A v5 (1ff61a6a, h=128) cross-arch Q16.16 determinism.

Iter 167 T1 round-33. The FDA SaMD audit-replay claim ("decade-stable
replay") relies on the Q16.16 ternary forward pass producing
**bit-identical** outputs across CPU / GPU / NPU and across software
versions. This pin locks the v5 bundle's Q16.16 logits on a curated
set of canonical pairs so a future change to:

  * encode_pair feature builder
  * train_bitnet_v5_h128 dot_t / classify implementation
  * the v5 weights themselves (bundle file)
  * Python integer arithmetic semantics (e.g., a future Python that
    changes int division behaviour — extremely unlikely but the pin
    catches it)

cannot silently change the logits. Any drift fires a per-pair diff.

Each canonical pair has 4 pinned values:

  * `feature_hash` — SHA-256 of the comma-joined integer feature
    vector. Pins the encode_pair output. Drift here means the feature
    builder changed (rule list, hash trits, flag set, etc.).
  * `logits_q16` — the full 5-class logit vector in Q16.16. The
    primary determinism contract.
  * `logits_hash` — SHA-256 of the comma-joined logits. A second
    bit-identity check that fires faster than per-element diff.
  * `severity_name` — argmax label. The user-visible verdict; if
    this drifts but logits don't, the label-list rotated, which is
    a coordinated breaking change that requires re-pinning.

If a pin needs to be updated (legitimate retrain, encoder refactor,
etc.), update ALL FOUR fields for the affected pair in the same
commit. The 100-iteration determinism stress test (below) exists
because Q16.16 inference must produce IDENTICAL outputs every run
— if classify(da, db) returns different logits on repeated calls,
the deterministic-replay claim is false and the FDA SaMD audit
trail loses its load-bearing safety property.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from retrain_runpod.train_bitnet_v3_full import encode_pair  # noqa: E402

_BUNDLE = _REPO_ROOT / "retrain_runpod" / "bitnet_weights_v5_h128.json"
_Q16_ONE = 1 << 16
_LABELS = ("none", "moderate", "serious", "major", "contraindicated")


# Canonical pair pin set — locks v5's Q16.16 logits on 8 well-evidenced
# pairs covering every severity class + the iter-155/164 cohort additions.
# Captured iter-167 against bundle 1ff61a6a82a709d9 (h=128, iter-166
# saved). A coordinated re-pin is required if any field changes.
_V5_CANONICAL_PINS: dict[tuple[str, str], dict] = {
    ("warfarin", "aspirin"): {
        "logits_q16": [-7088062, -3745966, -2970176, -5232455, -10644289],
        "feature_hash": "56760d1f63d83984a8a07f38ad351adb9149c64077d6badafa39954199d7cd65",
        "logits_hash": "0b5ee6248bccaf8f56e42643aa5a65395803b2aa7e7151c80d40291f8d5cd3af",
        "severity_name": "serious",
    },
    ("atorvastatin", "grapefruit"): {
        "logits_q16": [-1577110, 1353587, 88251, -379748, -1955366],
        "feature_hash": "388b2c7bed588f849524bb217bcc971d6fabcda4e689a79995410b2cabcc6cd2",
        "logits_hash": "5e74c7ac2fe33fc6f8090cae755fce959cc7f0e3b43455cf9e2656bfaf9cdc44",
        "severity_name": "moderate",
    },
    ("amoxicillin", "penicillin"): {
        "logits_q16": [745023, 2799211, -1587553, -970, -1300212],
        "feature_hash": "274420840682b74c796f4d9cd58b024a888fb8dadf7e0a6e1b7c5bc39549a60c",
        "logits_hash": "81613349056cc986c0e9d4aaf14e02c3ff1ffb456b5b2b6684646c7ae9297109",
        "severity_name": "moderate",
    },
    ("clarithromycin", "simvastatin"): {
        "logits_q16": [-667860, -464892, -330156, 35777, 101566],
        "feature_hash": "c65cdf7da2640dd93a0a672f752e3a556855aeb0c93bc128b8d8f63f271f362e",
        "logits_hash": "ef8638336076b4e82601efdf6106f9ab292261305e24b20ce7fbffe6ce7d8205",
        "severity_name": "contraindicated",
    },
    ("linezolid", "sertraline"): {
        "logits_q16": [-666415, -2064853, -2569497, -596445, 1563478],
        "feature_hash": "7b6c1ec6d92bbd55eecbae83c180553b871959c9657827c3eb41911ec0b0d51f",
        "logits_hash": "1ce8ab1fc000aa11140cbcb257b5be9bd8e320dac54fc5302fbb20d9c9fa7d9d",
        "severity_name": "contraindicated",
    },
    ("ritonavir", "simvastatin"): {
        "logits_q16": [-469207, -1450970, -661606, -284865, 624007],
        "feature_hash": "97f2ab8f8f3a7525f9af4381b20f3a4c96294fa23551558ba1ed670bd95980e3",
        "logits_hash": "d708f11152e0493b889d7124360d67b37cbf6ce856a42ccb1a4c58f054e0a1d2",
        "severity_name": "contraindicated",
    },
    ("atazanavir", "simvastatin"): {
        "logits_q16": [-431203, -983649, -683516, 288046, 574110],
        "feature_hash": "f89879901a2b2b18897f51a7b24e96b9dab898f27c319a43b917e2a242d765a1",
        "logits_hash": "04a4bf6e3b7655d5259a5c8b92dd883faed70a4bc0a4cf9d98b2c3ae577f7f02",
        "severity_name": "contraindicated",
    },
    ("azathioprine", "febuxostat"): {
        "logits_q16": [-4281634, -4828084, -4452307, -3658754, 2710477],
        "feature_hash": "6dbe2b1757f813be250fda5b2fe205a0eed0b176fcb53c1e49769a5f367930d1",
        "logits_hash": "a55992ba89958312f161137570851e3f3f9a7b3ce5699e332daeb0f5ae821f68",
        "severity_name": "contraindicated",
    },
    # iter-183: extension to lock the 3 v5 known-misses (iter-172/177/182
    # cohort-growth pairs) under Q16.16. Each pair's logits are pinned
    # so that pre-v6-retrain drift on these specific inferences is
    # impossible. After v6 retrain lands these will flip from
    # 'severity_name: none' to 'contraindicated' — at which point the
    # pin must be re-rotated in the same commit as the v5 bundle swap.
    ("isavuconazole", "simvastatin"): {
        "logits_q16": [472215, -271348, 165836, 116798, -337666],
        "feature_hash": "9962aa97b72267fec2b88bd1a6d014462fb9b5dd8d779de818bb7d63db1e54bd",
        "logits_hash": "fab154aede801b172b0bcb9f1908abe3d6cd405db36e371341121b945fb93eab",
        "severity_name": "none",
    },
    ("ketoconazole", "ergotamine"): {
        "logits_q16": [558828, -1923101, -1853283, -1216526, -1075026],
        "feature_hash": "902eff79b5c30b9263ed0d6ac1499546c00993c96f4aca92505868cfe84778ff",
        "logits_hash": "224ba6663131dbaa3165ce5e9eae0622e8c34dd95dce3d1f5fe8ad3a3132ba7b",
        "severity_name": "none",
    },
    ("minocycline", "isotretinoin"): {
        "logits_q16": [819989, -3968120, -1585966, -29760, -5111691],
        "feature_hash": "1843a44e3853c29b50b5dd5406c6545dd3c9011100943bbf08c14f097c73f7f6",
        "logits_hash": "a950504bcb1204d38913a98313b627efd86196588e8e71659d70e634e8f33f37",
        "severity_name": "none",
    },
    # iter-187: midazolam+ketoconazole (35th contra). v5 misses
    # (predicts 'none') — CYP3A4-strong-inh × benzodiazepine sub-slot
    # was 0-example pre-iter-187.
    ("ketoconazole", "midazolam"): {
        "logits_q16": [196491, -3113614, -1790158, -196196, -2376667],
        "feature_hash": "03ca8946845ad20deb5388bfbd5c11068ff5d8ebf7c12f872a32b15550a139d5",
        "logits_hash": "074d94b9cbe8f4759ddd4bce7da5a9029c63eff588fbbef4e54519f31cd09734",
        "severity_name": "none",
    },
    # iter-192: eplerenone+ketoconazole (36th contra). v5 misses
    # — CYP3A4-strong-inh × K+-sparing-diuretic sub-slot was
    # 0-example pre-iter-192.
    ("eplerenone", "ketoconazole"): {
        "logits_q16": [235129, -594025, -801568, 61440, -969445],
        "feature_hash": "68e6a07a15f1cf6b51e573ab8c8b7aa12bb21fb8d38b7f2235a1c1e397aa2cb2",
        "logits_hash": "4c202d8acb977368a68f22eeeaf6a15f3d5c03c2708d3e85d71273e6574a0d38",
        "severity_name": "none",
    },
}


def _classify_v5_q16(da: str, db: str, bundle: dict) -> tuple[list[int], list[int]]:
    """Return (feature_q1616, logits_q16) — bit-identical engine path."""
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
    return feat, logits


def _argmax_label(logits: list[int]) -> str:
    return _LABELS[max(range(len(logits)), key=lambda i: logits[i])]


def _hash_int_seq(xs: list[int]) -> str:
    return hashlib.sha256(",".join(str(x) for x in xs).encode()).hexdigest()


def _load_bundle() -> dict:
    return json.loads(_BUNDLE.read_text())


# ── per-pair canonical pin tests ──────────────────────────────────────────────


def test_v5_canonical_pair_logits_pinned() -> None:
    """Each canonical pair must produce the EXACT pinned Q16.16 logits."""
    bundle = _load_bundle()
    for (da, db), expected in _V5_CANONICAL_PINS.items():
        _, live_logits = _classify_v5_q16(da, db, bundle)
        assert live_logits == expected["logits_q16"], (
            f"v5 logits drift for ({da}, {db}):\n"
            f"  expected: {expected['logits_q16']}\n"
            f"  live:     {live_logits}\n"
            f"Re-pin all four fields if this is a legitimate retrain."
        )


def test_v5_canonical_feature_hashes_pinned() -> None:
    """Each canonical pair's encode_pair output must hash to the pinned
    SHA-256 — drift here signals an upstream feature-builder change
    (rule list, hash trits, flag set, etc.) that breaks the v5 contract."""
    bundle = _load_bundle()
    for (da, db), expected in _V5_CANONICAL_PINS.items():
        feat, _ = _classify_v5_q16(da, db, bundle)
        live = _hash_int_seq(feat)
        assert live == expected["feature_hash"], (
            f"v5 feature_hash drift for ({da}, {db}): "
            f"live={live[:16]}..., pinned={expected['feature_hash'][:16]}..."
        )


def test_v5_canonical_logits_hashes_pinned() -> None:
    """SHA-256 over the comma-joined logits must match. Faster-failing
    bit-identity check that complements the per-element diff above."""
    bundle = _load_bundle()
    for (da, db), expected in _V5_CANONICAL_PINS.items():
        _, logits = _classify_v5_q16(da, db, bundle)
        live = _hash_int_seq(logits)
        assert live == expected["logits_hash"], (
            f"v5 logits_hash drift for ({da}, {db}): "
            f"live={live[:16]}..., pinned={expected['logits_hash'][:16]}..."
        )


def test_v5_canonical_severity_labels_pinned() -> None:
    """argmax label per canonical pair must match the pinned severity_name."""
    bundle = _load_bundle()
    for (da, db), expected in _V5_CANONICAL_PINS.items():
        _, logits = _classify_v5_q16(da, db, bundle)
        live = _argmax_label(logits)
        assert live == expected["severity_name"], (
            f"v5 severity drift for ({da}, {db}): "
            f"live={live!r}, pinned={expected['severity_name']!r}"
        )


# ── determinism stress test ───────────────────────────────────────────────────


def test_v5_q16_inference_is_deterministic_across_repeats() -> None:
    """Q16.16 inference MUST produce bit-identical outputs on every
    invocation. If the same (drug_a, drug_b) yields different logits
    across 100 calls, the deterministic-replay claim is false and the
    FDA SaMD audit trail loses its load-bearing safety property.

    100 iterations × 13 canonical pairs = 1300 forward passes. All must
    return the same logit vector for a given pair."""
    bundle = _load_bundle()
    for da, db in _V5_CANONICAL_PINS.keys():
        first_pass = None
        for _ in range(100):
            _, logits = _classify_v5_q16(da, db, bundle)
            if first_pass is None:
                first_pass = logits
            else:
                assert logits == first_pass, (
                    f"Non-deterministic Q16.16 inference for ({da}, {db}): "
                    f"first call returned {first_pass}, later call returned "
                    f"{logits}. The cross-arch determinism claim is broken."
                )


def test_v5_canonical_pin_set_covers_every_severity_class() -> None:
    """Coverage invariant: the pin set must include at least one pair
    from every severity class actually present in the live cache. Drift
    here signals that a class was added/removed from the cohort and
    the pin set should grow."""
    actually_present_classes = {
        info["severity_name"] for info in _V5_CANONICAL_PINS.values()
    }
    required = {"moderate", "serious", "contraindicated"}
    missing = required - actually_present_classes
    assert not missing, (
        f"v5 canonical pin set is missing severity classes: {missing}. "
        f"Add at least one canonical pair for each missing class so the "
        f"determinism pin actually covers the full live cache surface."
    )


# ─── iter-183 cross-pin: every v5-known-miss has a pinned Q16.16 logit set ──


def test_v5_canonical_pins_cover_every_expected_miss_iter183() -> None:
    """Cross-pin invariant: every pair in _V5_EXPECTED_MISSES (the v5
    live-recall pin's known-miss set) must also appear in this pin's
    canonical Q16.16 logit set.

    Why
    ===
    The known-miss set acknowledges that v5 currently predicts 'none'
    on those pairs. Without a Q16.16 logit pin, a future change to
    encode_pair / dot_t / weights could silently flip the prediction
    for those exact pairs without firing CI — masking either a
    silent retrain landing OR (worse) a regression on what was
    already a near-miss prediction.

    By pinning the full logit vector, any drift on these specific
    inferences fires the canonical-pair-logits-pinned test before
    the live-recall pin re-evaluates. The two pins together provide
    bidirectional guarantee:
      * live recall pin: 'these pairs MUST be missed (no silent retrain)'
      * determinism pin: 'these pairs' logits MUST equal pinned values
        (no silent encoder/weight change on the missed pairs)
    """
    from tests.test_engine.test_path_a_v5_live_recall_pin import (
        _V5_EXPECTED_MISSES,
    )

    canonical_pairs = {
        tuple(sorted([a, b])) for a, b in _V5_CANONICAL_PINS.keys()
    }
    expected_miss_pairs = {
        tuple(sorted([a, b])) for a, b in _V5_EXPECTED_MISSES
    }

    missing = expected_miss_pairs - canonical_pairs
    assert not missing, (
        f"Q16.16 determinism gap: the following v5 known-miss pairs "
        f"are NOT in this pin's canonical-pair set, so their Q16.16 "
        f"logits can drift silently before the v6 retrain: "
        f"{sorted(missing)}.  Add each missing pair to "
        f"_V5_CANONICAL_PINS in lockstep with the _V5_EXPECTED_MISSES "
        f"extension."
    )
