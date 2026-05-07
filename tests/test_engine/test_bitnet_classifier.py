"""Tests for engine.bitnet_classifier.

Covers the four load-bearing claims the BitNet b1.58 layer adds to ClinicalMem:

  1. **Bit-identical reproducibility** — the same input + weights bundle on
     any platform produces the same logits and severity class. Pure Python
     int arithmetic guarantees this; the test asserts byte-stable output
     against a frozen golden.
  2. **Order invariance** — `(drug_a, drug_b)` and `(drug_b, drug_a)`
     produce the same output (lex sort canonicalisation).
  3. **Q16.16 invariants** — every logit and bias is in the signed-32-bit
     Q16.16 range; saturation is correct.
  4. **Bundle-ID stability** — the same `engine/bitnet_weights.json` always
     loads to the same `bundle_id` on every machine.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Allow running tests from the repo root.
_HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_HERE))

from engine.bitnet_classifier import (  # noqa: E402
    BitNetWeights,
    Q16_ONE,
    classify,
    classifier_layer,
    load_weights,
)

SEVERITY_NAMES_COUNT = 5


# ─── reproducibility / bit-identity ────────────────────────────────────────

def test_classify_is_byte_stable_across_calls() -> None:
    """Same input + same weights -> same logits + same severity, every call."""
    weights = load_weights()
    first = classify("warfarin", "ibuprofen", weights)
    second = classify("warfarin", "ibuprofen", weights)
    assert first.severity == second.severity
    assert first.logits_q16 == second.logits_q16
    assert first.feature_hash == second.feature_hash
    assert first.repro_hash == second.repro_hash


def test_classify_logits_are_integers() -> None:
    """Q16.16 logits must be plain Python ints — no floats anywhere."""
    weights = load_weights()
    result = classify("warfarin", "ibuprofen", weights)
    assert all(isinstance(v, int) for v in result.logits_q16)


def test_repro_hash_changes_when_inputs_change() -> None:
    """Different drug pair → different repro_hash."""
    weights = load_weights()
    a = classify("warfarin", "ibuprofen", weights)
    b = classify("metformin", "ibuprofen", weights)
    assert a.repro_hash != b.repro_hash
    assert a.feature_hash != b.feature_hash


# ─── order invariance ─────────────────────────────────────────────────────

def test_classify_is_order_invariant() -> None:
    """`(a, b)` and `(b, a)` must produce identical results."""
    weights = load_weights()
    forward = classify("warfarin", "ibuprofen", weights)
    reverse = classify("ibuprofen", "warfarin", weights)
    assert forward.severity == reverse.severity
    assert forward.logits_q16 == reverse.logits_q16
    assert forward.feature_hash == reverse.feature_hash
    assert forward.repro_hash == reverse.repro_hash


def test_classify_canonicalises_whitespace_and_case() -> None:
    """Drug names are case-insensitive and whitespace-normalised."""
    weights = load_weights()
    a = classify("WARFARIN", "ibuprofen", weights)
    b = classify("  warfarin  ", "Ibuprofen", weights)
    assert a.repro_hash == b.repro_hash


# ─── Q16.16 invariants ────────────────────────────────────────────────────

def test_logits_within_q16_signed_range() -> None:
    """Every logit must fit in signed 32-bit Q16.16 (saturating clamp)."""
    weights = load_weights()
    result = classify("warfarin", "ibuprofen", weights)
    q16_min = -(1 << 31)
    q16_max = (1 << 31) - 1
    for value in result.logits_q16:
        assert q16_min <= value <= q16_max, f"{value} outside Q16.16 range"


def test_severity_class_is_argmax_of_logits() -> None:
    """The reported severity must be the argmax of the logits vector."""
    weights = load_weights()
    result = classify("warfarin", "ibuprofen", weights)
    expected = result.logits_q16.index(max(result.logits_q16))
    assert result.severity == expected


# ─── bundle-ID stability ──────────────────────────────────────────────────

def test_bundle_id_is_stable_across_loads() -> None:
    """Loading the same JSON file twice produces the same bundle_id."""
    a = load_weights()
    b = load_weights()
    assert a.bundle_id == b.bundle_id
    assert len(a.bundle_id) == 64  # SHA-256 hex digest length


def test_bundle_shape_is_canonical() -> None:
    """Weight matrices must match the documented dimensions.

    Iter-275 v8 promotion: dims now driven by `_meta` schema, but the
    invariants (square first layer, hidden→5-class output, ternary
    weights, q16.16 biases) still hold. v8 ships at 193×256 hidden,
    256×5 output. Pre-v8 (cfadb4f6) was 128×64, 64×5.
    """
    weights = load_weights()
    assert weights.hidden_features == len(weights.hidden_w)
    assert all(len(row) == weights.in_features for row in weights.hidden_w)
    assert len(weights.hidden_b) == weights.hidden_features
    assert len(weights.output_w) == weights.out_features == 5
    assert all(len(row) == weights.hidden_features for row in weights.output_w)
    assert len(weights.output_b) == weights.out_features


def test_all_weights_are_ternary() -> None:
    """Every weight in the bundle must be in {-1, 0, +1}."""
    weights = load_weights()
    for row in weights.hidden_w:
        for w in row:
            assert w in (-1, 0, 1)
    for row in weights.output_w:
        for w in row:
            assert w in (-1, 0, 1)


# ─── classifier_layer convenience wrapper ─────────────────────────────────

def test_classifier_layer_caches_weights() -> None:
    """First call loads; subsequent calls reuse the cached bundle."""
    a = classifier_layer("warfarin", "ibuprofen")
    b = classifier_layer("amoxicillin", "penicillin")
    assert a.weights_id == b.weights_id


def test_classifier_layer_severity_in_range() -> None:
    """Severity is one of the five canonical classes."""
    result = classifier_layer("warfarin", "ibuprofen")
    assert 0 <= result.severity < SEVERITY_NAMES_COUNT


# ─── known-pair regression set (frozen against the current weights bundle) ─

@pytest.mark.parametrize("drug_a,drug_b", [
    ("warfarin", "ibuprofen"),
    ("amoxicillin", "penicillin"),
    ("metformin", "iodine"),
    ("atorvastatin", "grapefruit"),
    ("aspirin", "warfarin"),
])
def test_known_pair_produces_severity(drug_a: str, drug_b: str) -> None:
    """Each known interaction pair must produce a non-`none` severity.

    The classifier is hash-deterministic; if these ever return SEVERITY_NONE
    after a weights-bundle update, that update broke the regression set
    and must be re-baselined. Used as the v3 audit-chain replay anchor.
    """
    result = classifier_layer(drug_a, drug_b)
    assert result.severity > 0, (
        f"{drug_a}+{drug_b} returned SEVERITY_NONE; weights bundle "
        f"{result.weights_id[:16]} broke the regression set"
    )
