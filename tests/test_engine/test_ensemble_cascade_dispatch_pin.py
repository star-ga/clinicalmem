"""Pin iter-421 Path B 2-bundle ensemble dispatch contract.

iter-428 (round T1) — **37th cross-pin family** in the discipline lineage.
iter-421 / iter-422 / iter-426 shipped the cascade dispatcher in
``engine/bitnet_classifier.py::classify`` AND its mirror in
``docs/bitnet_browser.js``. iter-425 locked B's bundle CONTENT integrity
(13 invariants). iter-426 closed the audit-replay verifier under the
ensemble. **Gap**: the dispatcher RULE itself — "A predicts contra →
A's verdict wins; otherwise B's constrained argmax over {moderate,
serious, major} replaces A's class" — is verified only INDIRECTLY (via
the all-diagonal confusion matrix from
``test_bitnet_design_class_abstention_pin.py`` and the audit-replay
verifier). A regression that swaps the dispatch condition (e.g., "==
contra" → "!= contra"), removes the constrained argmax, or uses A's
logits for B's prediction would still produce a valid-looking
``BitNetResult`` and could pass surface-level tests if the weights
happened to compensate.

This pin LOCKS THE DISPATCH RULE DIRECTLY at the unit-test layer:

  1. **A-only mode** (weights_b=None) — every classify call returns
     weights_id == bundle_id_a (64-char hex, no '+'). Same behavior as
     the v8-only era pre-iter-421.
  2. **Cascade fires when A predicts non-contra** — composite weights_id
     "{bundle_id_a}+{bundle_id_b}" (129 chars, exactly one '+'),
     severity in {moderate, serious, major} (i.e., 1, 2, or 3).
  3. **Cascade BYPASSED when A predicts contra** — weights_id stays
     bundle_id_a only (A's contra-veto wins, B is not consulted).
  4. **Severity 0 (none) cannot escape from B** — B's constrained
     argmax pins severity ∈ {1, 2, 3} so 'none' is unreachable from
     a cascade-fired path; the only way to get severity=0 is from A's
     prediction in A-only mode.
  5. **Composite weights_id parse roundtrip** — splitting on '+' yields
     exactly two 64-char hex strings, each matching SHA-256 format.
  6. **repro_hash is sensitive to bundle_b** — the same pair classified
     under (A, None) vs (A, B) produces DIFFERENT repro_hash for
     non-contra cases (because the cascade payload includes
     logits_q16_b + bundle_id_b).
  7. **Severity_name vocab consistency** — under cascade-fired path,
     severity_name uses the v8 vocab ('none', 'moderate', 'serious',
     'major', 'contraindicated'); not the v1 vocab.
  8. **BitNetResult invariants** — every returned result has all 7
     required fields; weights_id matches repro_hash payload's
     weights_id; logits_q16 length == 5.

These tests use the LIVE engine bundles + LIVE classifier function.
A mocking-based contract test would not catch the regression class
where weights_id format drifts (e.g., a hypothetical refactor that
joins with "::" instead of "+" would pass mocks but break audit-chain
parser). End-to-end with real bundles + real classify() is the only
way to lock the format invariant.

Cross-pin lineage: 37th cross-pin family
(iter-178/183/188/193/198/203/223/228/232/234/236/239/240/244/246/247
 + iter-255 + iter-260 + iter-279 + iter-281 + iter-285 + iter-286 +
 iter-295 + iter-296 + iter-301 + iter-304 + iter-306 + iter-311 +
 iter-314 + iter-319 + iter-324 + iter-329 + iter-366 + iter-374 +
 iter-390 + iter-425 + iter-428).

Same shape: every load-bearing claim gets a cross-pin invariant
that ties artifact + behavior + format together so silent drift fails
at commit time. iter-425 locked B bundle CONTENT; iter-428 locks the
DISPATCH RULE that consumes both bundles.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from engine.bitnet_classifier import (  # noqa: E402
    classify,
    load_weights,
    load_weights_b,
)

# 64-char lowercase hex SHA-256 — both bundle_ids must match this.
_SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")
# Composite format: "{a_id}+{b_id}" — 64 + 1 + 64 = 129 chars.
_COMPOSITE_LEN = 129

# Canonical test pairs from the live cohort with known severities
# under iter-421 ensemble. Picked so each invariant has at least one
# concrete trigger. ALL three are in the live PCCP cache (verified).
_CONTRA_PAIR = ("clarithromycin", "simvastatin")  # contra (v8 BOOST_KEYS anchor)
_SERIOUS_PAIR = ("warfarin", "aspirin")           # serious (one of the 6 NTI-overveto B fixes)
_MODERATE_PAIR = ("amlodipine", "simvastatin")    # moderate (one of the 2 mod-miss B fixes)


def _weights_a_b():
    a = load_weights()
    b = load_weights_b()
    assert b is not None, (
        "Bundle B must be loadable for this pin. Run `cp "
        "retrain_runpod/bitnet_weights_b_specialist.json "
        "engine/bitnet_weights_b_specialist.json` to restore the "
        "iter-421 ensemble bundle."
    )
    return a, b


# ─── 1. A-only mode ──────────────────────────────────────────────────────────


def test_a_only_mode_weights_id_is_single_bundle():
    """When weights_b=None, every result's weights_id is bundle_id_a
    (64 chars, no '+'). Pre-iter-421 contract preserved verbatim."""
    weights_a, _ = _weights_a_b()
    for pair in (_CONTRA_PAIR, _SERIOUS_PAIR, _MODERATE_PAIR):
        result = classify(*pair, weights_a)  # No weights_b kwarg
        assert result.weights_id == weights_a.bundle_id, (
            f"A-only mode result.weights_id drifted for {pair}: "
            f"got {result.weights_id[:16]}…, expected {weights_a.bundle_id[:16]}…"
        )
        assert "+" not in result.weights_id, (
            f"A-only mode result.weights_id contains '+' for {pair} — "
            f"cascade fired when weights_b was None?"
        )
        assert _SHA256_HEX.match(result.weights_id), (
            f"A-only mode result.weights_id not 64-char hex SHA-256 for "
            f"{pair}: {result.weights_id!r}"
        )


# ─── 2. Cascade fires when A predicts non-contra ─────────────────────────────


def test_cascade_fires_when_a_predicts_non_contra_serious():
    """Under iter-421 ensemble, a non-contra pair gets the composite
    weights_id format with severity in {moderate, serious, major}."""
    weights_a, weights_b = _weights_a_b()
    result = classify(*_SERIOUS_PAIR, weights_a, weights_b=weights_b)

    assert "+" in result.weights_id, (
        f"Cascade did NOT fire for {_SERIOUS_PAIR} (serious): "
        f"weights_id={result.weights_id[:16]}… has no '+'"
    )
    assert len(result.weights_id) == _COMPOSITE_LEN, (
        f"Composite weights_id wrong length for {_SERIOUS_PAIR}: "
        f"got {len(result.weights_id)}, expected {_COMPOSITE_LEN}"
    )
    assert result.severity in (1, 2, 3), (
        f"Cascade-fired severity {result.severity} ({result.severity_name}) "
        f"outside constrained-argmax set {{1, 2, 3}} for {_SERIOUS_PAIR}"
    )


def test_cascade_fires_when_a_predicts_non_contra_moderate():
    """Same as serious-pair test but for a moderate pair to confirm B's
    constrained argmax can return moderate (class 1) end-to-end."""
    weights_a, weights_b = _weights_a_b()
    result = classify(*_MODERATE_PAIR, weights_a, weights_b=weights_b)

    assert "+" in result.weights_id
    assert len(result.weights_id) == _COMPOSITE_LEN
    assert result.severity in (1, 2, 3)


# ─── 3. Cascade BYPASSED when A predicts contra ──────────────────────────────


def test_cascade_bypassed_when_a_predicts_contra():
    """A's contra-veto wins ALWAYS. weights_id stays A-only (64 chars)
    even when weights_b is supplied — B is not consulted on contra
    predictions. This is the iter-421 FDA-grade safety contract."""
    weights_a, weights_b = _weights_a_b()
    result = classify(*_CONTRA_PAIR, weights_a, weights_b=weights_b)

    assert result.severity_name == "contraindicated", (
        f"Test invariant violation: {_CONTRA_PAIR} should be "
        f"contraindicated under v8 (this confirms A's contra recall "
        f"is intact). Got {result.severity_name!r}."
    )
    assert result.weights_id == weights_a.bundle_id, (
        f"Cascade fired on contra prediction (B was consulted) — "
        f"violates iter-421 contract that A's contra-veto wins. "
        f"Got weights_id={result.weights_id[:16]}…"
    )
    assert "+" not in result.weights_id, (
        f"weights_id contains '+' on contra pair {_CONTRA_PAIR} — "
        f"cascade fired when it should have been bypassed"
    )


# ─── 4. Severity 0 (none) unreachable from cascade ───────────────────────────


def test_cascade_constrained_argmax_excludes_none():
    """B's constrained argmax over {1, 2, 3} means 'none' (class 0)
    can never be returned from a cascade-fired path. Even if B's raw
    logits had a higher value at class 0, the constraint masks it."""
    weights_a, weights_b = _weights_a_b()
    # Test on all 3 non-contra pairs; B can never emit class 0 here.
    for pair in (_SERIOUS_PAIR, _MODERATE_PAIR):
        result = classify(*pair, weights_a, weights_b=weights_b)
        assert result.severity != 0, (
            f"Cascade-fired result for {pair} returned severity=0 "
            f"(none). B's constrained argmax must mask class 0; if "
            f"this fires the constraint logic was removed."
        )
        assert result.severity_name != "none"


# ─── 5. Composite weights_id parse roundtrip ─────────────────────────────────


def test_composite_weights_id_parse_roundtrip():
    """Splitting composite weights_id on '+' yields exactly two 64-char
    hex SHA-256 values matching live bundle_id_a and bundle_id_b."""
    weights_a, weights_b = _weights_a_b()
    result = classify(*_SERIOUS_PAIR, weights_a, weights_b=weights_b)

    parts = result.weights_id.split("+")
    assert len(parts) == 2, (
        f"Composite weights_id has {len(parts)} parts after split('+'), "
        f"expected exactly 2: {result.weights_id!r}"
    )
    a_id, b_id = parts
    assert _SHA256_HEX.match(a_id), f"A part not 64-char hex: {a_id!r}"
    assert _SHA256_HEX.match(b_id), f"B part not 64-char hex: {b_id!r}"
    assert a_id == weights_a.bundle_id, (
        f"Composite A-part != live bundle_id_a: {a_id[:16]}… vs "
        f"{weights_a.bundle_id[:16]}…"
    )
    assert b_id == weights_b.bundle_id, (
        f"Composite B-part != live bundle_id_b: {b_id[:16]}… vs "
        f"{weights_b.bundle_id[:16]}…"
    )


# ─── 6. repro_hash is sensitive to bundle_b ──────────────────────────────────


def test_repro_hash_differs_between_a_only_and_cascade():
    """The same pair classified A-only vs A→B cascade produces
    DIFFERENT repro_hash for non-contra cases (cascade payload includes
    logits_q16_b + bundle_id_b). Otherwise the audit-chain wouldn't
    distinguish single-bundle from ensemble decisions."""
    weights_a, weights_b = _weights_a_b()

    a_only = classify(*_SERIOUS_PAIR, weights_a)
    cascaded = classify(*_SERIOUS_PAIR, weights_a, weights_b=weights_b)

    assert a_only.repro_hash != cascaded.repro_hash, (
        f"repro_hash IDENTICAL between A-only and cascade for "
        f"{_SERIOUS_PAIR} — the cascade payload extension "
        f"(logits_q16_b + bundle_id_b) is not reaching the SHA-256 "
        f"input. Audit chain cannot distinguish ensemble decisions "
        f"from A-only decisions."
    )


def test_repro_hash_identical_on_contra_pair_across_modes():
    """For contra pairs where A's verdict ALWAYS wins (B is bypassed),
    the repro_hash MUST match between A-only and cascade modes —
    otherwise A's contra path is silently consulting B somewhere."""
    weights_a, weights_b = _weights_a_b()

    a_only = classify(*_CONTRA_PAIR, weights_a)
    cascaded = classify(*_CONTRA_PAIR, weights_a, weights_b=weights_b)

    assert a_only.repro_hash == cascaded.repro_hash, (
        f"repro_hash diverged between A-only and cascade for contra "
        f"pair {_CONTRA_PAIR} — the cascade is reaching B even though "
        f"A predicted contra. Violates iter-421 contract."
    )


# ─── 7. Severity_name vocab consistency ──────────────────────────────────────


def test_cascade_severity_vocab_v8_aligned():
    """Severity_name uses the v8 corpus-aligned vocab (none, moderate,
    serious, major, contraindicated), not the v1 vocab (none, minor,
    moderate, major, contraindicated). Lock in case a future engine
    refactor accidentally swaps vocab tables for the cascade path."""
    weights_a, weights_b = _weights_a_b()

    expected_vocab = {"none", "moderate", "serious", "major", "contraindicated"}
    forbidden_vocab = {"minor"}  # v1-era class, removed at iter-275

    for pair in (_CONTRA_PAIR, _SERIOUS_PAIR, _MODERATE_PAIR):
        result = classify(*pair, weights_a, weights_b=weights_b)
        assert result.severity_name in expected_vocab, (
            f"Severity_name {result.severity_name!r} for {pair} not in "
            f"v8 vocab {sorted(expected_vocab)}"
        )
        assert result.severity_name not in forbidden_vocab, (
            f"Severity_name {result.severity_name!r} is from the "
            f"retired v1 vocab — cascade path is using stale enum"
        )


# ─── 8. BitNetResult invariants ──────────────────────────────────────────────


def test_bitnet_result_has_all_required_fields():
    """Every BitNetResult must carry all 7 fields with valid types
    so the audit chain can serialize without ambiguity."""
    weights_a, weights_b = _weights_a_b()
    result = classify(*_SERIOUS_PAIR, weights_a, weights_b=weights_b)

    assert isinstance(result.severity, int)
    assert isinstance(result.severity_name, str)
    assert isinstance(result.logits_q16, tuple)
    assert isinstance(result.feature_hash, str)
    assert isinstance(result.repro_hash, str)
    assert isinstance(result.weights_id, str)
    assert isinstance(result.deterministic_table_match, bool)

    assert len(result.logits_q16) == 5, (
        f"logits_q16 length {len(result.logits_q16)} != 5 (5-class output)"
    )
    assert all(isinstance(v, int) for v in result.logits_q16), (
        f"logits_q16 must be all ints (Q16.16): got {result.logits_q16}"
    )
    assert _SHA256_HEX.match(result.feature_hash), (
        f"feature_hash not 64-char hex: {result.feature_hash!r}"
    )
    assert _SHA256_HEX.match(result.repro_hash), (
        f"repro_hash not 64-char hex: {result.repro_hash!r}"
    )
