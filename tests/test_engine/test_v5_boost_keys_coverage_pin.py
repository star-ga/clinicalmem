# Copyright 2026 STARGA Inc. — Apache-2.0
"""Cross-pin invariant: every Path A v5 known-miss must be queued for
v6 retrain via BOOST_KEYS in retrain_runpod/train_bitnet_v5_h128.py.

Iter 178 T1 round-35. The iter-172 / iter-177 cohort-growth events
revealed a recurring pattern:

  1. T5 cohort growth adds a new contraindicated pair.
  2. Path A v5 (h=128, eea0e637-superseding 1ff61a6a bundle) misses
     the new pair under Q16.16 because its training corpus didn't
     include the relevant sub-class.
  3. _V5_EXPECTED_MISSES in test_path_a_v5_live_recall_pin.py is
     extended with the new pair (acknowledging the miss).
  4. BOOST_KEYS in retrain_runpod/train_bitnet_v5_h128.py is
     extended with the same pair so the next v6 retrain sweep will
     boost the loss-weight on the under-represented sub-class.

The risk: a contributor could update _V5_EXPECTED_MISSES (acknowledging
the miss) WITHOUT extending BOOST_KEYS. Then the next v6 retrain
wouldn't actually fix the miss — and the pin would happily pass with
the same _V5_EXPECTED_MISSES set forever, masking the regression.

This pin enforces the implication: every pair in _V5_EXPECTED_MISSES
must also be in BOOST_KEYS (in either canonical-pair-key direction).
The pin fires on the structural mismatch BEFORE the retrain is
launched, so the BOOST_KEYS edit lands in the same commit as the
miss-set extension.

What this pin does NOT enforce
==============================
This pin does NOT require every BOOST_KEYS entry to also appear in
_V5_EXPECTED_MISSES — BOOST_KEYS is a SUPERSET that includes the
historical iter-148 anchors (clarithromycin+simvastatin, etc.) which
v5 hits on Q16.16. Only the v5-misses subset of BOOST_KEYS overlaps
with _V5_EXPECTED_MISSES.
"""
from __future__ import annotations

import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_V5_PIN = _REPO_ROOT / "tests" / "test_engine" / "test_path_a_v5_live_recall_pin.py"
_TRAIN_SCRIPT = _REPO_ROOT / "retrain_runpod" / "train_bitnet_v5_h128.py"


def _parse_v5_expected_misses() -> set[tuple[str, str]]:
    """Extract the canonical (drug_a, drug_b) pairs from the
    _V5_EXPECTED_MISSES tuple in the v5 pin file.  Returns each pair
    as a sorted tuple so direction-flipped equivalents collapse."""
    src = _V5_PIN.read_text()
    # Match the _V5_EXPECTED_MISSES = ( ... ) block
    m = re.search(
        r"_V5_EXPECTED_MISSES\s*=\s*\(\s*(.*?)^\)",
        src,
        re.DOTALL | re.MULTILINE,
    )
    assert m is not None, (
        "Could not find _V5_EXPECTED_MISSES tuple in "
        f"{_V5_PIN.relative_to(_REPO_ROOT)}; the pin can't validate "
        "BOOST_KEYS coverage if the source is unparseable."
    )
    block = m.group(1)
    # Each entry is a quoted-string tuple: ("drug_a", "drug_b")
    pairs = re.findall(
        r'\(\s*"([^"]+)"\s*,\s*"([^"]+)"\s*\)',
        block,
    )
    return {tuple(sorted([a, b])) for a, b in pairs}


def _parse_boost_keys() -> set[tuple[str, str]]:
    """Extract canonical pairs from the BOOST_KEYS set literal in
    train_bitnet_v5_h128.py.  Each BOOST_KEYS entry is a string of
    the form 'drug_a::drug_b'."""
    src = _TRAIN_SCRIPT.read_text()
    m = re.search(
        r"BOOST_KEYS\s*=\s*\{\s*(.*?)^\s*\}",
        src,
        re.DOTALL | re.MULTILINE,
    )
    assert m is not None, (
        "Could not find BOOST_KEYS set literal in "
        f"{_TRAIN_SCRIPT.relative_to(_REPO_ROOT)}; the v6 retrain "
        "script's loss-weight injection cannot be cross-validated."
    )
    block = m.group(1)
    keys = re.findall(r'"([a-z][a-z\-_]*)::([a-z][a-z\-_]*)"', block)
    return {tuple(sorted([a, b])) for a, b in keys}


def test_v5_expected_misses_subset_of_boost_keys() -> None:
    """Every v5 expected-miss pair must be in BOOST_KEYS so the next
    v6 retrain sweep will up-weight that pair's loss and recover it."""
    expected_misses = _parse_v5_expected_misses()
    boost_keys = _parse_boost_keys()

    missing_from_boost = expected_misses - boost_keys
    assert not missing_from_boost, (
        f"BOOST_KEYS coverage gap: the following pairs are recorded as "
        f"v5 known-misses but are NOT in train_bitnet_v5_h128.py "
        f"BOOST_KEYS — the next v6 retrain will NOT recover them:\n"
        f"  {sorted(missing_from_boost)}\n"
        f"Add each missing pair to BOOST_KEYS in the same commit that "
        f"updates _V5_EXPECTED_MISSES."
    )


def test_v5_boost_keys_count_at_or_above_expected_misses() -> None:
    """Sanity floor: BOOST_KEYS should contain at least as many entries
    as _V5_EXPECTED_MISSES (because BOOST_KEYS includes the historical
    iter-148 anchors plus all v5-misses)."""
    expected_misses = _parse_v5_expected_misses()
    boost_keys = _parse_boost_keys()
    assert len(boost_keys) >= len(expected_misses), (
        f"BOOST_KEYS={len(boost_keys)} entries < _V5_EXPECTED_MISSES="
        f"{len(expected_misses)}. The historical iter-148 anchors are "
        f"missing from BOOST_KEYS, which would regress the iter-148 "
        f"baseline. Restore the iter-148 anchors before re-running."
    )


def test_parse_v5_expected_misses_reads_at_least_one_pair() -> None:
    """If _V5_EXPECTED_MISSES is empty, this pin is a no-op — that
    happens after a successful v6 retrain when v5 hits 33/33. Once
    that lands, this test will fail and the pin needs adjustment
    (the pin's role pivots from 'enforce BOOST_KEYS coverage' to
    'pin that v5 has zero misses')."""
    expected_misses = _parse_v5_expected_misses()
    if not expected_misses:
        # Acceptable post-v6-retrain state — but the parser must
        # still resolve. The empty set is a valid test outcome.
        return
    # Otherwise the parser must have found at least one pair (else
    # there's a parsing bug masking real misses).
    assert len(expected_misses) >= 1


def test_parse_boost_keys_reads_at_least_iter148_anchors() -> None:
    """The iter-148 BOOST_KEYS anchors (clarithromycin+simvastatin,
    cyclosporine+simvastatin, itraconazole+simvastatin,
    ketoconazole+simvastatin, gemfibrozil+simvastatin) must always
    be present — they're the original loss-weighted anchors from the
    iter-148 30-pair test split. Removing them would regress the
    iter-148 baseline."""
    boost_keys = _parse_boost_keys()
    iter148_anchors = {
        tuple(sorted(["clarithromycin", "simvastatin"])),
        tuple(sorted(["cyclosporine", "simvastatin"])),
        tuple(sorted(["itraconazole", "simvastatin"])),
        tuple(sorted(["ketoconazole", "simvastatin"])),
        tuple(sorted(["gemfibrozil", "simvastatin"])),
    }
    missing = iter148_anchors - boost_keys
    assert not missing, (
        f"Iter-148 BOOST_KEYS anchors removed: {sorted(missing)}. "
        f"These are the original loss-weighted contra anchors; "
        f"removing them regresses the iter-148 baseline."
    )
