# Copyright 2026 STARGA Inc. — Apache-2.0
"""Structural-integrity pin for ``docs/audit_replay_pins.json``.

The existing audit-replay pins (``tests/test_scripts/
test_audit_replay_verifier.py``) enforce a **coverage floor**:
``len(pins) >= contra_cohort_size`` and ``contra_keys ⊆ pin_keys``.
That catches the "forgot to regen audit-replay after cohort growth"
class.

What the floor pin does NOT catch:

1. **Over-pinning drift** — a contributor stuffs extra unrelated pairs
   into ``audit_replay_pins.json``. The set is still ⊇ cohort but the
   "39-pair re-replay" claim drifts upward without any T3 catching it.
2. **Lost anchors** — the 3 hand-picked demo anchors (warfarin+ibuprofen,
   atorvastatin+grapefruit, amoxicillin+penicillin) anchor the demo's
   classifier-replay walkthrough. A regenerator that only emits
   contra-class pairs would silently drop them.
3. **Empty / malformed fields** — a pin entry with empty ``logits_q16``
   or non-hex ``repro_hash`` would still pass the coverage floor but
   would crash the audit-replay verifier at runtime.

This pin closes those three gaps with **5 structural invariants**:

* T1: ``len(pairs) == 3 anchors + len(contra cohort)`` (exact equality,
  not a floor).
* T2: 3 demo anchors are ALWAYS present and have ``severity_name`` ≠ "none".
* T3: every pin has all 6 required fields and they pass type/shape contracts:
  - ``drug_a``, ``drug_b``: non-empty lower-stripped strings
  - ``feature_hash``, ``repro_hash``: 64-char lowercase hex (SHA-256)
  - ``logits_q16``: list of exactly 5 ints (5-class output)
  - ``severity_name`` ∈ {"none","moderate","serious","major","contraindicated"}
* T4: every pin's pair-key is **unique** (no duplicate drug pairs in
  the replay set — direction-flip canonicalised).
* T5: the JSON-LD wrapper carries the contractual top-level fields
  (``@context``, ``@type``, ``bundle_id``, ``pairs``, ``version``).

This is a **T1 round-38 testing-rigor pin** companion to:
- iter-178 BOOST_KEYS coverage (cross-pin shape: subset)
- iter-183 Q16.16 canonical-pins coverage (cross-pin shape: subset)
- iter-188 encode_pair encoder contract (unit-level structural)

Same drift-prevention pattern, but for the audit-replay JSON contract.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_PINS = _REPO_ROOT / "docs" / "audit_replay_pins.json"
_CACHE = _REPO_ROOT / "docs" / "openevidence_cache.json"

_ANCHOR_PAIRS = (
    ("amoxicillin", "penicillin"),
    ("atorvastatin", "grapefruit"),
    ("warfarin", "ibuprofen"),
)
_VALID_SEVERITIES = frozenset(
    {"none", "moderate", "serious", "major", "contraindicated"}
)
_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")


def _canon_key(a: str, b: str) -> tuple[str, str]:
    return tuple(sorted([a.lower().strip(), b.lower().strip()]))


@pytest.fixture
def pins() -> dict:
    return json.loads(_PINS.read_text())


@pytest.fixture
def contra_cohort_size() -> int:
    cache = json.loads(_CACHE.read_text())
    return sum(1 for e in cache if e.get("severity") == "contraindicated")


def test_pin_count_equals_anchors_plus_cohort_iter193(
    pins: dict, contra_cohort_size: int
) -> None:
    """Exact equality: pin count == 3 anchors + contra cohort size.

    Existing test_pins_size_at_least_full_contra_coverage uses ``>=``;
    that catches under-pinning but not over-pinning. Cohort growth
    flow (iter-172/177/182/187/192) requires +1 pin per +1 cohort
    entry. Anything else is drift."""
    expected = len(_ANCHOR_PAIRS) + contra_cohort_size
    actual = len(pins["pairs"])
    assert actual == expected, (
        f"audit_replay_pins.json pair count drifted: live={actual}, "
        f"expected={expected} (= {len(_ANCHOR_PAIRS)} demo anchors "
        f"+ {contra_cohort_size} contraindicated cohort entries). "
        f"Re-run scripts/verify_audit_replay.py."
    )


def test_demo_anchors_present_with_non_none_severity_iter193(
    pins: dict,
) -> None:
    """The 3 demo anchors (warfarin+ibuprofen, atorvastatin+grapefruit,
    amoxicillin+penicillin) MUST always be in the replay set, and must
    have non-'none' severity_name (these anchor the demo's BitNet 4.5
    Q16.16 replay walkthrough — 'none'-severity anchor would defeat
    the safety-class demonstration)."""
    pin_keys = {
        _canon_key(p["drug_a"], p["drug_b"]) for p in pins["pairs"]
    }
    for a, b in _ANCHOR_PAIRS:
        anchor_key = _canon_key(a, b)
        assert anchor_key in pin_keys, (
            f"Demo anchor pair {anchor_key} missing from audit-replay "
            f"pin set. The demo's BitNet 4.5 walkthrough uses these 3 "
            f"pairs to demonstrate cross-class severity replay — none "
            f"can be silently dropped during cohort regen."
        )
    # Also verify severity_name is not 'none' on any anchor
    for p in pins["pairs"]:
        if _canon_key(p["drug_a"], p["drug_b"]) in {
            _canon_key(*pair) for pair in _ANCHOR_PAIRS
        }:
            assert p.get("severity_name") not in (None, "none"), (
                f"Demo anchor {p['drug_a']}+{p['drug_b']} regressed to "
                f"severity_name='{p.get('severity_name')}' — anchors "
                f"must always carry a non-none severity to demonstrate "
                f"cross-class replay."
            )


def test_every_pin_has_valid_field_shapes_iter193(pins: dict) -> None:
    """Each pin must have all 6 required fields with type/shape
    contracts. A pin entry with missing/empty fields would still pass
    coverage-floor pins but would crash scripts/verify_audit_replay.py
    at runtime — this pin makes the JSON contract enforceable."""
    for i, p in enumerate(pins["pairs"]):
        # Strings: drug_a, drug_b
        for k in ("drug_a", "drug_b"):
            assert k in p and isinstance(p[k], str) and p[k].strip(), (
                f"Pin {i}: missing/empty {k!r}. Got: {p.get(k)!r}"
            )
            assert p[k] == p[k].lower().strip(), (
                f"Pin {i}: {k}={p[k]!r} not canonicalised "
                f"(must be lower-stripped)."
            )
        # Hashes: feature_hash, repro_hash — 64-char lowercase hex
        for k in ("feature_hash", "repro_hash"):
            assert k in p and isinstance(p[k], str), (
                f"Pin {i}: missing/non-string {k!r}."
            )
            assert _HEX64_RE.match(p[k]), (
                f"Pin {i}: {k}={p[k]!r} is not a valid 64-char "
                f"lowercase SHA-256 hex digest."
            )
        # logits_q16: list of exactly 5 ints (5-class severity output)
        assert "logits_q16" in p, f"Pin {i}: missing logits_q16."
        logits = p["logits_q16"]
        assert isinstance(logits, list) and len(logits) == 5, (
            f"Pin {i}: logits_q16 must be a 5-element list "
            f"(5-class severity output). Got: {logits!r}"
        )
        assert all(isinstance(v, int) for v in logits), (
            f"Pin {i}: every element of logits_q16 must be int "
            f"(Q16.16 fixed-point). Got: {logits!r}"
        )
        # severity_name: enum (allow None or absent for legacy entries
        # but if present must be in the 5-class set)
        sev = p.get("severity_name")
        assert sev is None or sev in _VALID_SEVERITIES, (
            f"Pin {i}: severity_name={sev!r} not in valid 5-class set "
            f"{sorted(_VALID_SEVERITIES)}."
        )


def test_pin_pair_keys_are_unique_iter193(pins: dict) -> None:
    """No drug pair may appear twice in the audit-replay set. The
    coverage-floor pin (test_pins_cover_every_contraindicated_cache_entry)
    works with sets so it's blind to duplicates — a regenerator that
    accidentally double-emits a pair would still pass coverage but
    would inflate the '39 pairs' claim spuriously. Direction-flip
    collapsed (warfarin+ibuprofen == ibuprofen+warfarin)."""
    seen: list[tuple[str, str]] = []
    for p in pins["pairs"]:
        key = _canon_key(p["drug_a"], p["drug_b"])
        assert key not in seen, (
            f"Duplicate audit-replay pin: {key} appears more than "
            f"once (direction-flip canonicalised). Regenerator bug."
        )
        seen.append(key)


def test_top_level_jsonld_envelope_intact_iter193(pins: dict) -> None:
    """The JSON-LD wrapper carries the FDA-SaMD contractual fields:
    @context (Schema.org), @type, bundle_id, pairs, version. A
    regenerator that emitted only `pairs` (no envelope) would defeat
    the Schema.org Dataset claim documented in JUDGES.md."""
    required = ("@context", "@type", "bundle_id", "pairs", "version")
    for k in required:
        assert k in pins, (
            f"audit_replay_pins.json missing required top-level "
            f"field {k!r}. Required envelope: {required}."
        )
    assert isinstance(pins["pairs"], list), (
        f"`pairs` must be a list. Got: {type(pins['pairs']).__name__}"
    )
    # bundle_id must be a 64-char hex (sha256 of engine weights bundle)
    assert _HEX64_RE.match(pins["bundle_id"]), (
        f"bundle_id={pins['bundle_id']!r} is not a 64-char "
        f"lowercase SHA-256 hex (engine weights digest)."
    )
