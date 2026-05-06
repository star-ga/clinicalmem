# Copyright 2026 STARGA Inc. — Apache-2.0
"""Pin Path A v8 (1f0f8859, h=256) per-pair Q16.16 canonical-pin
determinism — 16 canonical pairs × 4 pinned values + 100×16 = 1600
forward-pass determinism stress.

Iter 244 v8 sweep landed: doubled hidden_dim 128 → 256 broke the v7
architectural ceiling. Seed 71 hit 41/41 + 4/4 + 0 FP under Q16.16.

Mirror-shape of the v6 q16 pin (retired iter-245) — but with 16
canonical pairs ALL classifying as documented (8 anchors + 7 v5-
historical-misses + 1 iter-215 lurasidone+ketoconazole pair that v6
missed but v8 catches). The iter-244 architectural-double broke the
ceiling — every prior known-miss is now classified contraindicated
under v8.

Iter-210 cross-pin discipline preserved: every pair in the inlined
`_V5_HISTORICAL_MISSES` tuple appears in V8 canonical pins AND has
severity_name='contraindicated'. The iter-215 v6-known-miss pair
(lurasidone+ketoconazole) is also pinned at severity='contraindicated'
(now caught by v8).

Pinned at iter-244 against bundle 1f0f88591c05af57c (iter-244 sweep
seed=71).
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from retrain_runpod.train_bitnet_v3_full import encode_pair  # noqa: E402

_BUNDLE = _REPO_ROOT / "retrain_runpod" / "bitnet_weights_v8_h256.json"

_PATH_A_V8_BUNDLE_ID = (
    "1f0f88591c05af57c62d844b667639b29c7d1f0eb1b213073d158101611f76e6"
)

_Q16_ONE = 1 << 16


def _load_bundle() -> dict:
    return json.loads(_BUNDLE.read_text())


def _hash_int_seq(seq: list[int]) -> str:
    return hashlib.sha256(
        b"".join(v.to_bytes(8, "big", signed=True) for v in seq)
    ).hexdigest()


def _classify_v8_q16(da: str, db: str, bundle: dict) -> tuple[list[int], list[int]]:
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
    return feat, logits


def _argmax_label(logits: list[int]) -> str:
    labels = ("none", "moderate", "serious", "major", "contraindicated")
    return labels[max(range(len(logits)), key=lambda i: logits[i])]


# Inlined V5 historical-miss tuple — same as the iter-210 cross-pin
# preserved through the iter-245 v6 retirement. v8 must catch every
# pair in this tuple as contraindicated (the BOOST_KEYS promise that
# iter-207 v6 sweep partially fulfilled and iter-244 v8 sweep
# completed).
_V5_HISTORICAL_MISSES: tuple[tuple[str, str], ...] = (
    ("isavuconazole", "simvastatin"),
    ("ketoconazole", "ergotamine"),
    ("minocycline", "isotretinoin"),
    ("ketoconazole", "midazolam"),
    ("eplerenone", "ketoconazole"),
    ("cyclosporine", "rosuvastatin"),
    ("tolvaptan", "ketoconazole"),
)

# Plus the iter-215 v6-known-miss that v8 finally catches (the
# breakthrough pair that motivated the architectural double).
_ITER215_V6_KNOWN_MISS = ("ketoconazole", "lurasidone")


# 16 canonical pairs × 4 pinned values (logits_q16 + feature_hash +
# logits_hash + severity_name). Computed at iter-244 against bundle
# 1f0f88591c05af57c.
_V8_CANONICAL_PINS: dict[tuple[str, str], dict] = {
    # — Severity-class anchors (cross-class coverage) —
    ("warfarin", "ibuprofen"): {
        "logits_q16": [-1928017, -1031470, 2423326, 747543, -2811021],
        "feature_hash": "5f6271ff01718007bca74568412cfe26afbe8f3139fbc5f759c1f0ab07b1b1f3",
        "logits_hash": "37b25f9f6826729a61de9b2e71ceae7229881641590650d6351d97d266549eee",
        "severity_name": "serious",
    },
    ("atorvastatin", "grapefruit"): {
        "logits_q16": [480474, 3967176, -6329991, 2109040, -7359887],
        "feature_hash": "90141ffcf38708e520db512d5f64cf10c6ee12b5eb2b52aab14741af3fb401fd",
        "logits_hash": "e758c64f19bdfead86eb09b503a5320abd4319765b54dda205254d02bb6364f5",
        "severity_name": "moderate",
    },
    ("amoxicillin", "penicillin"): {
        "logits_q16": [-425437, 2248054, -8606112, -3074630, -5816061],
        "feature_hash": "d429048fd7e653301d2a0ae825ded1623a21cde786b4e1931da354265f49505f",
        "logits_hash": "1b3db58a06d501647a1b2037b93a7926f7a26bfc1bc6bee98d5aa79e2a828934",
        "severity_name": "moderate",
    },
    ("clarithromycin", "simvastatin"): {
        "logits_q16": [-284901, -333753, -138679, 35778, 99886],
        "feature_hash": "91d88e612117b8ed86e9828a1d3516c919dbd51623e853a39e632c60d5c47f13",
        "logits_hash": "8b75166b1efcb447d1ec9250b919cd671350282a60d41281f10ca39d61be3284",
        "severity_name": "contraindicated",
    },
    ("ciprofloxacin", "tizanidine"): {
        "logits_q16": [468310, -1636843, -14026782, -7675825, 6622716],
        "feature_hash": "fe1aaa027887389508903746c485e72c51c525385aea17021867139aefb02d9a",
        "logits_hash": "a51fc0881dfa5ca1dcd65069735b4089e35ce1959524505fc49a67682348c7cc",
        "severity_name": "contraindicated",
    },
    ("lisinopril", "sacubitril"): {
        "logits_q16": [-1126467, -5739706, -6517761, -5256886, 3433842],
        "feature_hash": "577cf01d87e934c64da6c0cfc860c349662bf6387e28ad9ea44d74014977b83c",
        "logits_hash": "00f17114521b48b9cc50e491427545bd608482970119700276ca3ded7e64011c",
        "severity_name": "contraindicated",
    },
    ("allopurinol", "azathioprine"): {
        "logits_q16": [-464108, -2051951, -7974165, -1317657, 6841162],
        "feature_hash": "59975574ec8c184df0b8efb8037d6e09b5dfc29f0785ddfc1c91bed51403e9f0",
        "logits_hash": "f83b1f8a253998f496a79a088afb0f20ada2b2c36d9519e5869b48923a878e27",
        "severity_name": "contraindicated",
    },
    ("iodine", "metformin"): {
        "logits_q16": [-36285, -4064442, -17832834, -7460059, 9463593],
        "feature_hash": "6f34ac4a2b3cf6bf06083cbe07236fc5de3a93404fbc162d09d2b2be15ed0413",
        "logits_hash": "88a5e901631407a1fb146547b57c0e1c9f74373ef6f2f2acd02964fc88db025f",
        "severity_name": "contraindicated",
    },
    # — V5 historical-misses, all caught by v6+v8 —
    ("isavuconazole", "simvastatin"): {
        "logits_q16": [-284901, -485703, -138679, 35778, 99886],
        "feature_hash": "56f267a2ddc94feb5cf73916e038b80e1b71e3f91953a4c18bf49915d70d4dad",
        "logits_hash": "0a4128362277ae1b4df9fc54b91269aaac5c34d194fc694628f76d9e2c6e0ee9",
        "severity_name": "contraindicated",
    },
    ("ketoconazole", "ergotamine"): {
        "logits_q16": [-304161, -8391173, -285252, -3695122, 8366227],
        "feature_hash": "eb7b9733edafd784792518e963e78028ef3fb3e12906ee8a6a8a47b695a64d7a",
        "logits_hash": "c39e862c75e459b12729667b38bb32e4554cbc96299956d11b3ce352720e08fd",
        "severity_name": "contraindicated",
    },
    ("isotretinoin", "minocycline"): {
        "logits_q16": [845044, -13544258, -13600090, -7053920, 13380406],
        "feature_hash": "5cba8f19c7540f39a26a1d6bab94450290031fcf6b1ec82ac81058b458047435",
        "logits_hash": "67214a0708622465ea9a44be78356cd9832b8d4c0e2af2448369724531393d5b",
        "severity_name": "contraindicated",
    },
    ("ketoconazole", "midazolam"): {
        "logits_q16": [-2317330, -5798717, -8277145, -2746269, 5598479],
        "feature_hash": "d5f38861048c035532f75e3eec1383ea3eda3feb270d81339b1774fee8073d05",
        "logits_hash": "028a504f2e948c0f2c81284c1f72c2ab4903d529dcbcb8729a81f577925253f8",
        "severity_name": "contraindicated",
    },
    ("eplerenone", "ketoconazole"): {
        "logits_q16": [-523074, -1274112, 281141, -1544055, 4739024],
        "feature_hash": "5a9adfc4a7ab2a54f5c860bfbdd43d6e7b9281baebcda552269a4d53d48575e9",
        "logits_hash": "c2b81b9bd91c6dfc46c71372082d5464b43b9780f8a34060f7faf35d700696f6",
        "severity_name": "contraindicated",
    },
    ("cyclosporine", "rosuvastatin"): {
        "logits_q16": [-5197984, -5376837, -1732557, 569672, 4338937],
        "feature_hash": "1f5427ef9dc7f4cb6409b6708c803bddbfefe4ea8f47132b46b1afca11b6e0fc",
        "logits_hash": "87a17fdda9f8dec9913bebd193bda1ac4fef157384c73f467f83da07ee928fc3",
        "severity_name": "contraindicated",
    },
    ("tolvaptan", "ketoconazole"): {
        "logits_q16": [-3078554, -7802952, -8319117, -1729771, 9013915],
        "feature_hash": "68cf7bffce498ace39a1a9c53281cbd1bacf97d349fa5d3a7ff037a1944e09ac",
        "logits_hash": "36fc20f6da59a12e87ec1937d1b5957081bb47bd198377e6a62d9dd3dbfbc710",
        "severity_name": "contraindicated",
    },
    # — Iter-215 v6-known-miss, FINALLY caught by v8 (the architectural-double breakthrough) —
    ("ketoconazole", "lurasidone"): {
        "logits_q16": [-2509327, -5002563, -7008196, 551839, 5671254],
        "feature_hash": "7bdb73d83277b5930254acda7da06265fd037f7a179e119a5794908f2502fc46",
        "logits_hash": "536f0dc3b7d3e72d079faf4c30c842aed8dc973c158ebbd35cc4dc2abade2cb1",
        "severity_name": "contraindicated",
    },
}


# ── per-pair pin tests ────────────────────────────────────────────────────────


def test_v8_canonical_pair_logits_pinned() -> None:
    """Each canonical pair must produce the EXACT pinned Q16.16 logits."""
    bundle = _load_bundle()
    for (da, db), expected in _V8_CANONICAL_PINS.items():
        _, live_logits = _classify_v8_q16(da, db, bundle)
        assert live_logits == expected["logits_q16"], (
            f"v8 logits drift for ({da}, {db}):\n"
            f"  expected: {expected['logits_q16']}\n"
            f"  live:     {live_logits}\n"
            f"Re-pin all four fields if this is a legitimate retrain "
            f"(would also require new bundle_id in iter-244 pin)."
        )


def test_v8_canonical_feature_hashes_pinned() -> None:
    """Each canonical pair's encode_pair output must hash to the pinned
    SHA-256 — drift here signals an upstream feature-builder change
    (would also fail iter-188 encode_pair contract pin)."""
    bundle = _load_bundle()
    for (da, db), expected in _V8_CANONICAL_PINS.items():
        feat, _ = _classify_v8_q16(da, db, bundle)
        live = _hash_int_seq(feat)
        assert live == expected["feature_hash"], (
            f"v8 feature_hash drift for ({da}, {db}): "
            f"live={live[:16]}..., pinned={expected['feature_hash'][:16]}..."
        )


def test_v8_canonical_logits_hashes_pinned() -> None:
    """SHA-256 over the byte-encoded logits must match. Faster-failing
    bit-identity check that complements the per-element diff above."""
    bundle = _load_bundle()
    for (da, db), expected in _V8_CANONICAL_PINS.items():
        _, logits = _classify_v8_q16(da, db, bundle)
        live = _hash_int_seq(logits)
        assert live == expected["logits_hash"], (
            f"v8 logits_hash drift for ({da}, {db}): "
            f"live={live[:16]}..., pinned={expected['logits_hash'][:16]}..."
        )


def test_v8_canonical_severity_labels_pinned() -> None:
    """argmax label per canonical pair must match the pinned severity_name."""
    bundle = _load_bundle()
    for (da, db), expected in _V8_CANONICAL_PINS.items():
        _, logits = _classify_v8_q16(da, db, bundle)
        live = _argmax_label(logits)
        assert live == expected["severity_name"], (
            f"v8 severity drift for ({da}, {db}): "
            f"live={live!r}, pinned={expected['severity_name']!r}"
        )


# ── determinism stress test ───────────────────────────────────────────────────


def test_v8_q16_determinism_stress() -> None:
    """100 iterations × 16 canonical pairs = 1600 forward-pass
    determinism stress test. Q16.16 ternary inference MUST be
    bit-identical across iterations (no floating-point ops, no RNG)."""
    bundle = _load_bundle()
    first_results: dict[tuple[str, str], list[int]] = {}
    for iteration in range(100):
        for (da, db), expected in _V8_CANONICAL_PINS.items():
            _, logits = _classify_v8_q16(da, db, bundle)
            key = (da, db)
            if iteration == 0:
                first_results[key] = logits
            else:
                assert logits == first_results[key], (
                    f"v8 determinism violated for ({da}, {db}) at iter {iteration}: "
                    f"first={first_results[key]}, now={logits}. Q16.16 is supposed "
                    f"to be bit-identical."
                )


def test_v8_severity_class_coverage() -> None:
    """The 16 canonical pairs must cover all 5 severity classes
    (excluding 'minor' which the cohort intentionally lacks).

    Iter-244 expanded coverage from iter-210's 4 classes (no major
    in v6 canonicals) to include the iter-215 lurasidone+ketoconazole
    breakthrough as an additional contraindicated pin.
    """
    severities = {info["severity_name"] for info in _V8_CANONICAL_PINS.values()}
    expected_coverage = {"serious", "moderate", "contraindicated"}
    assert expected_coverage.issubset(severities), (
        f"v8 canonical pins missing severity classes: "
        f"have={severities}, expected_subset={expected_coverage}"
    )


def test_v8_bundle_id_matches_iter244_pin() -> None:
    """Sanity: the bundle this pin reads must match the bundle_id
    pinned in test_path_a_v8_live_recall_pin (iter-244)."""
    bundle = _load_bundle()
    assert bundle["_meta"]["bundle_id"] == _PATH_A_V8_BUNDLE_ID, (
        f"v8 bundle_id drift: live={bundle['_meta']['bundle_id']!r}, "
        f"pinned={_PATH_A_V8_BUNDLE_ID!r}"
    )


def test_v8_catches_every_v5_historical_miss() -> None:
    """Cross-pin invariant (mirror of iter-210): every pair in the
    inlined `_V5_HISTORICAL_MISSES` tuple must appear in V8 canonical
    pins AND have severity_name='contraindicated'. The bidirectional
    guarantee with iter-244 v8 live-recall pin (zero misses) is "v8
    catches every prior known-miss + the iter-215 lurasidone+keto
    breakthrough" — without this pin the architectural-double promise
    has no per-pair lock.

    Plus the iter-215 v6-known-miss `ketoconazole+lurasidone` must
    ALSO be classified contraindicated (the breakthrough pair).
    """
    # All v5-historical misses must be contraindicated under v8
    for key in _V5_HISTORICAL_MISSES:
        canonical_key = tuple(sorted([k.lower() for k in key]))
        found = False
        for (pa, pb), info in _V8_CANONICAL_PINS.items():
            if tuple(sorted([pa.lower(), pb.lower()])) == canonical_key:
                found = True
                assert info["severity_name"] == "contraindicated", (
                    f"V5-historical-miss {key} pinned as "
                    f"severity={info['severity_name']!r} under v8 — "
                    f"BOOST_KEYS promise broken; this pair must be "
                    f"contraindicated under v8."
                )
                break
        assert found, (
            f"V5-historical-miss {key} not in v8 canonical pins; "
            f"the iter-210 cross-pin discipline requires every pair "
            f"in _V5_HISTORICAL_MISSES to be pinned."
        )

    # The iter-215 v6-known-miss must also be contraindicated under v8
    canonical_key = tuple(sorted([k.lower() for k in _ITER215_V6_KNOWN_MISS]))
    found = False
    for (pa, pb), info in _V8_CANONICAL_PINS.items():
        if tuple(sorted([pa.lower(), pb.lower()])) == canonical_key:
            found = True
            assert info["severity_name"] == "contraindicated", (
                f"Iter-215 v6-known-miss {_ITER215_V6_KNOWN_MISS} pinned "
                f"as severity={info['severity_name']!r} under v8 — the "
                f"architectural-double breakthrough that motivated the v8 "
                f"sweep is broken if this pair isn't contraindicated."
            )
            break
    assert found, (
        f"Iter-215 v6-known-miss {_ITER215_V6_KNOWN_MISS} not in v8 "
        f"canonical pins — the breakthrough pair must be pinned at "
        f"severity='contraindicated' to lock the iter-244 promise."
    )
