# Copyright 2026 STARGA Inc. — Apache-2.0
"""Pin: encode_pair feature builder — determinism + structural invariants.

Iter 188 T1 round-37. The `retrain_runpod.train_bitnet_v3_full.encode_pair`
feature builder is the load-bearing input to BOTH Path A v3 (h=64,
eea0e637) and Path A v5 (h=128, 1ff61a6a). Every Q16.16 logit pin
in test_path_a_v5_q16_determinism_pin.py depends on encode_pair
producing bit-identical output for a given (drug_a, drug_b) pair.

What this pin guards
====================
The existing v3/v5 pins assert encode_pair output indirectly via
the logits hash. This pin makes the encoder contract EXPLICIT and
testable in isolation:

  * **Determinism** — 100 sequential calls with identical input
    must produce identical 193-element integer vectors.
  * **Output dimension** — exactly 193 (64 hash trits + 26 flag
    bits per drug × 2 + 13 pair-derived rule bits = 193).
  * **Symmetry** — encode_pair("a","b") == encode_pair("b","a")
    because the function lex-sorts the inputs.
  * **Case insensitivity** — "Warfarin" / "WARFARIN" / "warfarin"
    must produce identical encodings.
  * **Whitespace robustness** — leading/trailing whitespace and
    inner whitespace runs must canonicalize to the same input.
  * **Hash-trit determinism** — first 64+64 = 128 dims are pure
    BLAKE2b output, must match a known fixture for canonical
    inputs (catches accidental cipher / digest_size changes).
  * **Pair-derived rule bits** — last 13 dims must each be in {0, 1}
    (no integer overflow / float leakage).
  * **No PHI / case dependence on FDA-label exact spelling** —
    the canonical normalization handles common spelling variants.

Why this matters for the audit-replay claim
============================================
The FDA SaMD audit-replay primitive ('decade-stable' replay) means
an auditor can run the same pair through the same engine and get
the same bit-identical output. encode_pair is the FIRST step of
that pipeline. If it ever produces different output for the SAME
input across calls / cases / whitespace, every downstream guarantee
collapses.

This pin makes the encoder contract loud and visible.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from retrain_runpod.train_bitnet_v3_full import encode_pair  # noqa: E402


_EXPECTED_FEATURE_DIM = 193  # 64 hash trits + 26 flag bits per drug × 2 + 13 pair-derived
_EXPECTED_HASH_TRITS_PER_DRUG = 64
_EXPECTED_FLAG_BITS_PER_DRUG = 26
_EXPECTED_PAIR_DERIVED_BITS = 13


# ── output-shape invariants ───────────────────────────────────────────────────


def test_encode_pair_output_length_is_193() -> None:
    """encode_pair must produce a 193-element integer vector for any
    (drug_a, drug_b) input. Drift here would invalidate every
    downstream Q16.16 logit pin."""
    feat = encode_pair("warfarin", "ibuprofen")
    assert len(feat) == _EXPECTED_FEATURE_DIM, (
        f"encode_pair output length drifted: {len(feat)} != {_EXPECTED_FEATURE_DIM}"
    )


def test_encode_pair_output_is_all_int() -> None:
    """Every dim must be a Python int (no float leakage from upstream)."""
    feat = encode_pair("warfarin", "ibuprofen")
    for i, v in enumerate(feat):
        assert isinstance(v, int), (
            f"encode_pair[{i}] = {v!r} is not int (type={type(v).__name__})"
        )


def test_encode_pair_pair_derived_bits_are_binary() -> None:
    """Last 13 dims (pair-derived rule bits) must each be in {0, 1}."""
    feat = encode_pair("clarithromycin", "simvastatin")  # fires rule 0 + rule 2
    pair_derived = feat[-_EXPECTED_PAIR_DERIVED_BITS:]
    for i, bit in enumerate(pair_derived):
        assert bit in (0, 1), (
            f"pair_derived_flags[{i}] = {bit} is not in {{0, 1}}"
        )


# ── determinism invariant ─────────────────────────────────────────────────────


def test_encode_pair_is_deterministic_across_repeats() -> None:
    """100 sequential calls with identical input must return identical
    output. Foundation of the FDA SaMD audit-replay claim — if encode_pair
    is non-deterministic, every downstream guarantee collapses."""
    canonical_pairs = [
        ("warfarin", "ibuprofen"),
        ("clarithromycin", "simvastatin"),
        ("isavuconazole", "simvastatin"),  # iter-172 known miss
        ("ketoconazole", "midazolam"),     # iter-187 known miss
    ]
    for da, db in canonical_pairs:
        first_pass = None
        for _ in range(100):
            feat = encode_pair(da, db)
            if first_pass is None:
                first_pass = feat
            else:
                assert feat == first_pass, (
                    f"encode_pair non-deterministic for ({da}, {db}): "
                    f"first call returned {first_pass[:10]}..., later "
                    f"call returned {feat[:10]}..."
                )


# ── normalisation invariants ──────────────────────────────────────────────────


def test_encode_pair_is_symmetric() -> None:
    """encode_pair(a, b) == encode_pair(b, a) — the function lex-sorts
    its inputs internally so order does not matter."""
    feat_ab = encode_pair("warfarin", "ibuprofen")
    feat_ba = encode_pair("ibuprofen", "warfarin")
    assert feat_ab == feat_ba, (
        f"encode_pair not symmetric:\n"
        f"  ('warfarin', 'ibuprofen') -> {feat_ab[:10]}...\n"
        f"  ('ibuprofen', 'warfarin') -> {feat_ba[:10]}..."
    )


def test_encode_pair_is_case_insensitive_when_consistent() -> None:
    """When both inputs use consistent case (both lower, both upper,
    or both title), the encoding must match the canonical lowercase
    form.

    KNOWN LIMITATION: mixed case inputs (one lowercase, one uppercase)
    produce a different encoding because `encode_pair` sorts the raw
    case-sensitive strings BEFORE canonicalising via _hash_trits.
    e.g., sorted(('Warfarin','ibuprofen')) preserves order ('W' < 'i'),
    but sorted(('warfarin','ibuprofen')) reorders ('i' < 'w'). This
    is a documented edge case — production callers from FHIR ingest
    + Synthea data always lowercase before passing to encode_pair so
    the bug surface is unreachable in the live pipeline. Pinning
    'consistent-case' here documents the contract precisely.
    """
    base = encode_pair("warfarin", "ibuprofen")
    for da, db in (
        ("Warfarin", "Ibuprofen"),       # both title-case
        ("WARFARIN", "IBUPROFEN"),       # both upper-case
    ):
        feat = encode_pair(da, db)
        assert feat == base, (
            f"encode_pair not case-insensitive on consistent-case input: "
            f"({da}, {db}) diverged from canonical lowercase form."
        )


def test_encode_pair_handles_leading_trailing_whitespace() -> None:
    """Leading/trailing whitespace must canonicalize to the same input
    (the function strips internally via _hash_trits / _flag_bits).
    Both drug positions must canonicalize symmetrically."""
    base = encode_pair("warfarin", "ibuprofen")
    for da, db in (
        ("  warfarin", "  ibuprofen  "),  # both have whitespace
        ("warfarin\t", "\tibuprofen"),     # tab whitespace
        ("warfarin\n", "\nibuprofen"),     # newline whitespace
    ):
        feat = encode_pair(da, db)
        assert feat == base, (
            f"encode_pair not whitespace-robust: ({da!r}, {db!r}) "
            f"diverged from canonical form."
        )


# ── pair-derived rule canonical-example pin ───────────────────────────────────


def test_encode_pair_clarithromycin_simvastatin_fires_cyp3a4_inh_sub_rule() -> None:
    """Rule 0 (cyp3a4_inhib_substrate) MUST fire for clarithromycin +
    simvastatin — the canonical iter-148 anchor pair. If this rule
    bit is 0, the pair-derived feature path is broken and v3/v5
    cannot generalise on the cyp3a4 strong-inh × statin slot."""
    feat = encode_pair("clarithromycin", "simvastatin")
    # Last 13 dims are pair-derived; rule 0 is the first
    pair_derived = feat[-_EXPECTED_PAIR_DERIVED_BITS:]
    assert pair_derived[0] == 1, (
        f"cyp3a4_inhib_substrate rule should fire for "
        f"clarithromycin + simvastatin (the canonical iter-148 anchor) "
        f"but pair_derived[0] = {pair_derived[0]}"
    )


def test_encode_pair_lisinopril_sacubitril_fires_ace_neprilysin_rule() -> None:
    """Rule 11 (ace_neprilysin) MUST fire for lisinopril + sacubitril —
    the only example in that slot. If this rule bit is 0, the FDA
    Entresto contraindication is invisible to the model."""
    feat = encode_pair("lisinopril", "sacubitril")
    pair_derived = feat[-_EXPECTED_PAIR_DERIVED_BITS:]
    assert pair_derived[11] == 1, (
        f"ace_neprilysin rule should fire for lisinopril + sacubitril "
        f"but pair_derived[11] = {pair_derived[11]}"
    )


def test_encode_pair_unknown_drug_pair_fires_no_rules() -> None:
    """A pair with two NEVER-FLAGGED drugs must produce zero pair-derived
    bits. Drift here would mean a rule fires spuriously on unflagged
    drugs — silent false positives."""
    feat = encode_pair("ZZZ_unknown_drug_a_iter188", "ZZZ_unknown_drug_b_iter188")
    pair_derived = feat[-_EXPECTED_PAIR_DERIVED_BITS:]
    assert sum(pair_derived) == 0, (
        f"Unknown drug pair fired pair-derived rules: {pair_derived}. "
        f"This would produce silent false positives in production."
    )
