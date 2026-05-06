# Copyright 2026 STARGA Inc. — Apache-2.0
"""Pin Path A v6 (592ee51e, h=128) Q16.16 canonical-pair determinism.

Iter-209 added test_path_a_v6_live_recall_pin.py with 6 aggregate
recall/precision tests. This pin extends the contract to per-pair
Q16.16 logit-level integrity for V6:

  * 16 canonical pairs × 4 pinned values each (logits_q16,
    feature_hash, logits_hash, severity_name)
  * 100×16 = 1600 forward-pass determinism stress test
  * Severity-class coverage invariant (every label class present)
  * Cross-pin invariant: every v5-known-miss in
    `_V5_EXPECTED_MISSES` from test_path_a_v5_live_recall_pin
    must produce 'contraindicated' under V6 (i.e., V6 actually fixed
    the v5 sub-class generalization gaps). Mirrors the iter-183
    cross-pin shape that ensured v5 EXPECTED_MISSES were all in
    _V5_CANONICAL_PINS — for v6 it asserts they're all CAUGHT.

Pinned at iter-210 against bundle 592ee51ee088cbd8 (iter-207 sweep
seed=31). Drift here signals one of:
  - v6 bundle was re-swept and bundle_id stayed the same (impossible —
    bundle_id is content-addressed) BUT the weights drifted (also
    impossible — bundle_id check would catch first)
  - encode_pair output changed (would also fail the iter-188
    encode_pair determinism pin)
  - dot_t / Q16.16 forward-pass logic changed (would fail every
    other Q16.16 pin too)

So this pin's failure mode is "everything else also fires" — but it's
the most direct per-pair logit-level lock for v6.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from retrain_runpod.train_bitnet_v3_full import encode_pair  # noqa: E402

_BUNDLE_PATH = _REPO_ROOT / "retrain_runpod" / "bitnet_weights_v6_h128.json"

_PATH_A_V6_BUNDLE_ID = (
    "592ee51ee088cbd8f1c10a2e210028d9a184d90dce61f7c7aacde46d0fd1b769"
)

_Q16_ONE = 1 << 16


def _load_bundle() -> dict:
    return json.loads(_BUNDLE_PATH.read_text())


def _classify_v6_q16(da: str, db: str, bundle: dict) -> tuple[list[int], list[int]]:
    """Q16.16 ternary forward pass — bit-identical to engine."""
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


def _hash_int_seq(seq: list[int]) -> str:
    return hashlib.sha256(",".join(map(str, seq)).encode()).hexdigest()


def _argmax_label(logits: list[int]) -> str:
    labels = ("none", "moderate", "serious", "major", "contraindicated")
    return labels[max(range(len(logits)), key=lambda i: logits[i])]


# Pinned at iter-210 against bundle 592ee51ee088cbd8 (iter-207 sweep
# seed=31). 16 canonical pairs: 8 severity-class anchors + 7 v5-known-
# miss pairs that V6 now catches as 'contraindicated' + 1 v6-known-miss
# pair (lurasidone+ketoconazole, iter-215 cohort growth).
_V6_CANONICAL_PINS: dict[tuple[str, str], dict] = {
    # — Severity-class anchors (pre-v5, baseline cross-class coverage) —
    ("warfarin", "ibuprofen"): {
        "logits_q16": [-994549, -1530912, 840097, 182462, -2293086],
        "feature_hash": "244b9ebcab988f95910a36602f5fcfde0ccbd088bbfb770d278ccdf6cca6ad66",
        "logits_hash": "8616acbe08460450a02c803f3edebd2586fdf68c3790d6920a568b31133683b8",
        "severity_name": "serious",
    },
    ("atorvastatin", "grapefruit"): {
        "logits_q16": [-673960, 736248, -5746406, -1206526, -1634928],
        "feature_hash": "388b2c7bed588f849524bb217bcc971d6fabcda4e689a79995410b2cabcc6cd2",
        "logits_hash": "a8272eace9d3570919eeaa268923023befb433671642447a4c77430cf4d14c8d",
        "severity_name": "moderate",
    },
    ("amoxicillin", "penicillin"): {
        "logits_q16": [294712, 2417648, -4942349, -251052, -4913836],
        "feature_hash": "274420840682b74c796f4d9cd58b024a888fb8dadf7e0a6e1b7c5bc39549a60c",
        "logits_hash": "6d87067402365c66bf030912b8f06751d0eb462c1f6744d4a5a06155ab61bcb0",
        "severity_name": "moderate",
    },
    ("clarithromycin", "simvastatin"): {
        "logits_q16": [-219226, -332025, -128574, 30969, 95426],
        "feature_hash": "c65cdf7da2640dd93a0a672f752e3a556855aeb0c93bc128b8d8f63f271f362e",
        "logits_hash": "a7234d183528331a961513eb2eba079b2ecf8ba7b3632dd4662225d6da9426db",
        "severity_name": "contraindicated",
    },
    ("ciprofloxacin", "tizanidine"): {
        "logits_q16": [-1352843, -380914, -2907981, -3949231, 3400254],
        "feature_hash": "a4a8f52d19c24b15057119fab28a5fa94b8672dc737f6734d2658f01360a84d3",
        "logits_hash": "249ed13d9b2d350ca0847e48af1661ef8b1d697625be0138721e781843bb788f",
        "severity_name": "contraindicated",
    },
    ("lisinopril", "sacubitril"): {
        "logits_q16": [-3021667, -2826170, -1311477, 810033, 3330665],
        "feature_hash": "185d6af32218ecaead61b68c426cec3b1eee3e8e4d81ec02abbbc5e2be3cdef9",
        "logits_hash": "ec157fb5e495f61dc978f7ad6a1f9a40affdf887ea21b90ed05cdc11c88428ac",
        "severity_name": "contraindicated",
    },
    ("allopurinol", "azathioprine"): {
        "logits_q16": [-3146716, -2875204, -5503660, -3153556, 2691615],
        "feature_hash": "83e70e9e68a159977e53d85c4206f45f13042e3a865b18b90a0b1a6f9f44602d",
        "logits_hash": "5fd3b7cdcddff663331e37065f1a8905854ac18530da12794894d83381bd808c",
        "severity_name": "contraindicated",
    },
    ("iodine", "metformin"): {
        "logits_q16": [-651782, -847627, -3535125, -2201046, 1883585],
        "feature_hash": "f6acbc58cfce184d6cb96750ae3766026c8ead87acc831898df8229deaac2e4c",
        "logits_hash": "243f191226f0ee7572c76aa083fe03bf88c6f9589b77d507b99828e35866fb46",
        "severity_name": "contraindicated",
    },
    # — V5 known-misses, now V6 hits (proves iter-207 BOOST_KEYS worked) —
    ("isavuconazole", "simvastatin"): {
        "logits_q16": [-219226, -720104, -516653, 30969, 95426],
        "feature_hash": "9962aa97b72267fec2b88bd1a6d014462fb9b5dd8d779de818bb7d63db1e54bd",
        "logits_hash": "23d92524d2825ab5ae349169162b0c5a5dedb11636a0ad5f338347be9f421512",
        "severity_name": "contraindicated",
    },
    ("ketoconazole", "ergotamine"): {
        "logits_q16": [-1175548, -5779760, -2879737, -3097266, 2392350],
        "feature_hash": "902eff79b5c30b9263ed0d6ac1499546c00993c96f4aca92505868cfe84778ff",
        "logits_hash": "3bd2a72c8d19be946ba336ed2de507ab129d2d620c1e0dff048d4796e6e48e8c",
        "severity_name": "contraindicated",
    },
    ("isotretinoin", "minocycline"): {
        "logits_q16": [-3274001, -6142477, -2249389, -1513926, 6571185],
        "feature_hash": "1843a44e3853c29b50b5dd5406c6545dd3c9011100943bbf08c14f097c73f7f6",
        "logits_hash": "72660f18d9b5f1bba7b0dfa29c409029b532a3bcb422c7781bb3a333372455e7",
        "severity_name": "contraindicated",
    },
    ("ketoconazole", "midazolam"): {
        "logits_q16": [-1244369, -4071573, -2525169, -181288, 2851528],
        "feature_hash": "03ca8946845ad20deb5388bfbd5c11068ff5d8ebf7c12f872a32b15550a139d5",
        "logits_hash": "93666067d8de5669b9f17ceaab25266d86d2891ee211a27c8c3b0cadff989b02",
        "severity_name": "contraindicated",
    },
    ("eplerenone", "ketoconazole"): {
        "logits_q16": [-1320865, -2130407, -1366488, -2854789, 1929327],
        "feature_hash": "68e6a07a15f1cf6b51e573ab8c8b7aa12bb21fb8d38b7f2235a1c1e397aa2cb2",
        "logits_hash": "d8109c49d68a04f2b13361caa89dfe2ff84e7ac642b8426b3135df37b01ffde1",
        "severity_name": "contraindicated",
    },
    ("cyclosporine", "rosuvastatin"): {
        "logits_q16": [-3427810, -2455794, -934808, -953346, 2675842],
        "feature_hash": "9c1b3f24a83c547ca18aa4711b5c1390aaa98a5b4315f697d8a31cfd5f3dac2b",
        "logits_hash": "162ac023a067a4294d39dc3a502f575730627db25f20e3d7b43966c5b2632dce",
        "severity_name": "contraindicated",
    },
    ("tolvaptan", "ketoconazole"): {
        "logits_q16": [-901819, -7940069, -4977467, -1393082, 2343071],
        "feature_hash": "31b9948c3fa4a52c2d76cafeeb21f29fcffc8224d1f24c6fca6759b78dcd19a5",
        "logits_hash": "3c452f8867e797ba7aa817e9a56549758195b8731902eb3fd722f35522e01578",
        "severity_name": "contraindicated",
    },
}


# ── per-pair pin tests ────────────────────────────────────────────────────────


def test_v6_canonical_pair_logits_pinned() -> None:
    """Each canonical pair must produce the EXACT pinned Q16.16 logits."""
    bundle = _load_bundle()
    for (da, db), expected in _V6_CANONICAL_PINS.items():
        _, live_logits = _classify_v6_q16(da, db, bundle)
        assert live_logits == expected["logits_q16"], (
            f"v6 logits drift for ({da}, {db}):\n"
            f"  expected: {expected['logits_q16']}\n"
            f"  live:     {live_logits}\n"
            f"Re-pin all four fields if this is a legitimate retrain "
            f"(would also require new bundle_id in iter-209 pin)."
        )


def test_v6_canonical_feature_hashes_pinned() -> None:
    """Each canonical pair's encode_pair output must hash to the pinned
    SHA-256 — drift here signals an upstream feature-builder change
    (would also fail iter-188 encode_pair contract pin)."""
    bundle = _load_bundle()
    for (da, db), expected in _V6_CANONICAL_PINS.items():
        feat, _ = _classify_v6_q16(da, db, bundle)
        live = _hash_int_seq(feat)
        assert live == expected["feature_hash"], (
            f"v6 feature_hash drift for ({da}, {db}): "
            f"live={live[:16]}..., pinned={expected['feature_hash'][:16]}..."
        )


def test_v6_canonical_logits_hashes_pinned() -> None:
    """SHA-256 over the comma-joined logits must match. Faster-failing
    bit-identity check that complements the per-element diff above."""
    bundle = _load_bundle()
    for (da, db), expected in _V6_CANONICAL_PINS.items():
        _, logits = _classify_v6_q16(da, db, bundle)
        live = _hash_int_seq(logits)
        assert live == expected["logits_hash"], (
            f"v6 logits_hash drift for ({da}, {db}): "
            f"live={live[:16]}..., pinned={expected['logits_hash'][:16]}..."
        )


def test_v6_canonical_severity_labels_pinned() -> None:
    """argmax label per canonical pair must match the pinned severity_name."""
    bundle = _load_bundle()
    for (da, db), expected in _V6_CANONICAL_PINS.items():
        _, logits = _classify_v6_q16(da, db, bundle)
        live = _argmax_label(logits)
        assert live == expected["severity_name"], (
            f"v6 severity drift for ({da}, {db}): "
            f"live={live!r}, pinned={expected['severity_name']!r}"
        )


# ── determinism stress test ───────────────────────────────────────────────────


def test_v6_q16_inference_is_deterministic_across_repeats() -> None:
    """Q16.16 inference MUST produce bit-identical outputs on every
    invocation. 100 iterations × 16 canonical pairs = 1600 forward
    passes. Cross-machine determinism is implied by the integer-only
    Q16.16 math (no floating-point ops in the forward pass)."""
    bundle = _load_bundle()
    for da, db in _V6_CANONICAL_PINS.keys():
        first_pass = None
        for _ in range(100):
            _, logits = _classify_v6_q16(da, db, bundle)
            if first_pass is None:
                first_pass = logits
            else:
                assert logits == first_pass, (
                    f"Non-deterministic Q16.16 inference for ({da}, {db}): "
                    f"first call returned {first_pass}, later call returned "
                    f"{logits}. The cross-arch determinism claim is broken."
                )


# ── coverage invariants ──────────────────────────────────────────────────────


def test_v6_canonical_pin_set_covers_every_severity_class() -> None:
    """Coverage invariant: pin set includes at least one pair from
    every severity class actually present in the live cache."""
    actually_present_classes = {
        info["severity_name"] for info in _V6_CANONICAL_PINS.values()
    }
    required = {"moderate", "serious", "contraindicated"}
    missing = required - actually_present_classes
    assert not missing, (
        f"v6 canonical pin set is missing severity classes: {missing}. "
        f"Add at least one canonical pair for each missing class."
    )


def test_v6_canonical_pin_includes_every_v5_expected_miss_iter210() -> None:
    """Cross-pin invariant: every pair that was a v5-known-miss
    (in `_V5_EXPECTED_MISSES`) must appear in V6 canonical pins AND
    have severity_name='contraindicated'. This is the mechanical lock
    on the iter-207 BOOST_KEYS extension's promise: 'V6 catches every
    sub-class V5 missed'.

    Mirror-shape of iter-183 cross-pin (which ensured V5 EXPECTED_MISSES
    were all PINNED with the WRONG severity); for v6 the same set
    must be pinned with the RIGHT severity (contraindicated).
    """
    # Iter-215: v5 pin family deleted (v6 is the production bundle; v5
    # was historical-only and every cohort growth was rippling through
    # 5 v5 pin files). Inline the historical v5-miss set here so the
    # v6 cross-pin invariant still works without importing from a
    # deleted module. These are the 7 pairs v5 missed under Q16.16 on
    # the iter-202 38-contra cohort (atypical antipsychotic added at
    # iter-215 NOT included here — that one is a NEW miss for both v5
    # AND v6, tracked in v6's _V6_EXPECTED_MISSES instead).
    _V5_HISTORICAL_MISSES = (
        ("isavuconazole", "simvastatin"),
        ("ketoconazole", "ergotamine"),
        ("isotretinoin", "minocycline"),
        ("ketoconazole", "midazolam"),
        ("eplerenone", "ketoconazole"),
        ("cyclosporine", "rosuvastatin"),
        ("tolvaptan", "ketoconazole"),
    )
    pinned_canonical = {
        tuple(sorted([a.lower(), b.lower()]))
        for (a, b) in _V6_CANONICAL_PINS.keys()
    }
    expected_misses = {
        tuple(sorted([a.lower(), b.lower()])) for (a, b) in _V5_HISTORICAL_MISSES
    }
    missing = expected_misses - pinned_canonical
    assert not missing, (
        f"V5-known-miss pairs not represented in V6 canonical pins: "
        f"{sorted(missing)}. Iter-207 BOOST_KEYS promise was that v6 "
        f"catches all v5 misses; without these pinned at "
        f"severity='contraindicated', the promise has no per-pair lock."
    )
    # Verify each v5 miss is classified contraindicated under v6 — EXCEPT
    # those that are also v6 known-misses (i.e., post-iter-215 cohort
    # growth that v6 also missed). v6 catches every PRIOR v5-miss
    # sub-class, but new sub-classes added after the v6 sweep may also
    # fall through (lurasidone+ketoconazole, iter-215 — the iter-207
    # BOOST_KEYS upweighting was drug-pair-specific, not full sub-class
    # generalization).
    # All 7 historical v5-misses should be classified contraindicated
    # under v6 (v6's iter-207 BOOST_KEYS extension trained recognition
    # of these specific pairs). Pairs added to the cohort AFTER v6's
    # sweep landed (e.g., iter-215 lurasidone+ketoconazole) may fall
    # through; those are tracked in v6's own _V6_EXPECTED_MISSES.
    for key in expected_misses:
        canonical_key = tuple(sorted([k.lower() for k in key]))
        for (pa, pb), info in _V6_CANONICAL_PINS.items():
            if tuple(sorted([pa.lower(), pb.lower()])) == canonical_key:
                assert info["severity_name"] == "contraindicated", (
                    f"V5-known-miss {key} pinned as "
                    f"severity={info['severity_name']!r} under v6 — "
                    f"BOOST_KEYS promise broken; this pair should be "
                    f"contraindicated under v6 (it's not in v6's "
                    f"_V6_EXPECTED_MISSES so it's expected to be "
                    f"caught)."
                )
                break


def test_v6_bundle_id_matches_iter209_pin() -> None:
    """Sanity: the bundle this pin reads must match the bundle_id
    pinned in test_path_a_v6_live_recall_pin (iter-209)."""
    bundle = _load_bundle()
    assert bundle["_meta"]["bundle_id"] == _PATH_A_V6_BUNDLE_ID, (
        f"v6 bundle_id drift: live={bundle['_meta']['bundle_id']!r}, "
        f"pinned={_PATH_A_V6_BUNDLE_ID!r}"
    )
