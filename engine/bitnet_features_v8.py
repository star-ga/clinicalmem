"""V8 (193-dim) Path A feature encoder for engine/bitnet_classifier.py.

Layout
======
Per-drug encoding (90 features × 2 = 180):
    [0..64)   64 BLAKE2b ternary hash trits ∈ {-1, 0, +1}
    [64..90)  26 ATC pharmacology flag bits ∈ {0, 1}, ordered by
              `docs/pharmacology_flags.json` ``flag_keys``.

Pair-level encoding (13 features):
    [180..193)  13 pair-derived DDI-rule bits ∈ {0, 1}.

Total: 64 + 26 + 64 + 26 + 13 = 193 trits/bits.

Order canonicalisation
----------------------
Drug pairs are sorted lexicographically before encoding so that
`{warfarin, ibuprofen}` and `{ibuprofen, warfarin}` produce the same
193-dim vector. Same canonicalisation as `engine/bitnet_classifier`.

Source of truth
---------------
The encoder is bit-identical to `retrain_runpod/train_bitnet_v8_h256.py`
since the v8 ternary weights bundle (1f0f8859…) was trained against this
exact pipeline. Any divergence here would silently change forward-pass
output and invalidate the audit-chain bundle_id binding.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_PHARM_FLAGS_PATH = _REPO_ROOT / "docs" / "pharmacology_flags.json"

# Distribution-balanced trit table (50% zeros, 25% +1, 25% -1) — matches
# the table used in `engine/bitnet_classifier._encode_drug_token` and in
# the v8 trainer.
_TRIT_LOOKUP: tuple[int, ...] = (
    0, 0, 0, 0, 0, 0, 0, 0,
    1, 1, 1, 1,
    -1, -1, -1, -1,
)

# 64-byte hash digest size hits 64 trits cleanly via the 4-trit-per-byte
# extraction below; 16-byte BLAKE2b key matches the v8 trainer.
_BLAKE2B_DIGEST_SIZE = 16

_NITRATE_NAMES = frozenset({
    "isosorbide mononitrate",
    "isosorbide dinitrate",
    "nitroglycerin",
})

# Cached pharmacology flag table — read once at module import.
_FLAGS_DOC = json.loads(_PHARM_FLAGS_PATH.read_text(encoding="utf-8"))
FLAG_KEYS: tuple[str, ...] = tuple(_FLAGS_DOC["flag_keys"])
_FLAG_DRUGS: dict[str, dict] = _FLAGS_DOC["drugs"]

N_HASH_TRITS = 64
N_FLAG_BITS = len(FLAG_KEYS)
N_PER_DRUG = N_HASH_TRITS + N_FLAG_BITS
N_PAIR_DERIVED = 13  # iter-140: 6 baseline + 7 closure rules
FEAT_DIM = N_PER_DRUG * 2 + N_PAIR_DERIVED


def _canonical(name: str) -> str:
    """Lowercase + whitespace-collapse — same canonicalisation as
    `_encode_drug_token` in bitnet_classifier."""
    return " ".join(name.strip().lower().split())


def hash_trits(name: str) -> list[int]:
    """64-dim ternary hash trits ∈ {-1, 0, +1} via BLAKE2b-128 digest.

    Bit-identical to `engine.bitnet_classifier._encode_drug_token` and to
    the v8 trainer — both produce the same vector for the same canonical
    drug name on every machine.
    """
    digest = hashlib.blake2b(
        _canonical(name).encode("utf-8"),
        digest_size=_BLAKE2B_DIGEST_SIZE,
    ).digest()
    out: list[int] = []
    for byte in digest:
        out.append(_TRIT_LOOKUP[(byte >> 0) & 0xF])
        out.append(_TRIT_LOOKUP[(byte >> 4) & 0xF])
        out.append(_TRIT_LOOKUP[byte & 0xF])
        out.append(_TRIT_LOOKUP[(byte >> 2) & 0xF])
    return out[:N_HASH_TRITS]


def flag_bits(name: str) -> list[int]:
    """26 ATC pharmacology flag bits ∈ {0, 1} per drug.

    Unknown drugs → all zeros (the v8 trainer was trained against this
    same fall-through, so the model handles it as "no known class
    membership").
    """
    entry = _FLAG_DRUGS.get(_canonical(name), {"flags": []})
    set_flags = set(entry["flags"])
    return [1 if k in set_flags else 0 for k in FLAG_KEYS]


def pair_derived_flags(da: str, db: str) -> list[int]:
    """13 pair-derived DDI-rule bits encoding canonical interaction
    rules directly. These bypass hash noise to make the decision
    boundary explicit.

    Each bit fires iff the corresponding rule applies to the (drug_a,
    drug_b) pair. Indices match the v8 trainer (and the iter-140
    pair-derived rule set):

      [0]  cyp3a4_inhib_substrate
      [1]  oatp1b1_inhib_statin
      [2]  p_gp_inhib_substrate
      [3]  cyp2c9_inhib_anticoag
      [4]  maoi_serotonergic
      [5]  pde5_nitrate            (special: nitrate via name suffix)
      [6]  iodinated_contrast_metformin
      [7]  cyp1a2_inhib_substrate
      [8]  xo_thiopurine
      [9]  folate_antagonist_pair  (both drugs same flag)
      [10] tetracycline_retinoid
      [11] ace_neprilysin
      [12] metformin_renal
    """
    fa = set(_FLAG_DRUGS.get(_canonical(da), {"flags": []})["flags"])
    fb = set(_FLAG_DRUGS.get(_canonical(db), {"flags": []})["flags"])

    def has_pair(flag_x: str, flag_y: str) -> bool:
        return (flag_x in fa and flag_y in fb) or (flag_x in fb and flag_y in fa)

    def both_have(flag: str) -> bool:
        return flag in fa and flag in fb

    a_norm = _canonical(da)
    b_norm = _canonical(db)
    pde5_nitrate = (
        ("is_pde5_inhibitor" in fa and b_norm in _NITRATE_NAMES)
        or ("is_pde5_inhibitor" in fb and a_norm in _NITRATE_NAMES)
    )

    return [
        1 if has_pair("is_cyp3a4_strong_inhibitor", "is_cyp3a4_substrate") else 0,
        1 if has_pair("is_oatp1b1_inhibitor", "is_statin") else 0,
        1 if has_pair("is_p_gp_inhibitor", "is_p_gp_substrate") else 0,
        1 if has_pair("is_cyp2c9_inhibitor", "is_anticoagulant") else 0,
        1 if has_pair("is_maoi", "is_serotonergic") else 0,
        1 if pde5_nitrate else 0,
        1 if has_pair("is_iodinated_contrast", "is_metformin") else 0,
        1 if has_pair("is_cyp1a2_inhibitor", "is_cyp1a2_substrate") else 0,
        1 if has_pair("is_xanthine_oxidase_inhibitor", "is_thiopurine") else 0,
        1 if both_have("is_folate_antagonist") else 0,
        1 if has_pair("is_tetracycline", "is_retinoid") else 0,
        1 if has_pair("is_ace_inhibitor", "is_neprilysin_inhibitor") else 0,
        1 if has_pair("is_metformin", "is_renal_state") else 0,
    ]


def encode_pair_v8(drug_a: str, drug_b: str) -> list[int]:
    """V8 193-dim feature vector for an order-canonicalised drug pair.

    Layout: hash_trits(a) + flag_bits(a) + hash_trits(b) + flag_bits(b)
    + pair_derived_flags(a, b). Bit-identical to the v8 trainer's
    ``encode_pair``.
    """
    a, b = sorted((drug_a, drug_b))
    out = (
        hash_trits(a)
        + flag_bits(a)
        + hash_trits(b)
        + flag_bits(b)
        + pair_derived_flags(a, b)
    )
    if len(out) != FEAT_DIM:
        raise RuntimeError(
            f"v8 encoder produced {len(out)}-dim vector, expected {FEAT_DIM}"
        )
    return out
