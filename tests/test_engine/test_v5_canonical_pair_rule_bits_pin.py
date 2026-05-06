# Copyright 2026 STARGA Inc. — Apache-2.0
"""Per-pair pair-derived-rule-bit pin for the 14 V5 canonical pins.

Iter-197 cohort growth (cyclosporine+rosuvastatin) made the
"OATP1B1 × statin pure-isolation" framing load-bearing in JUDGES.md
+ demo.html: the pair fires rule 1 (`is_oatp1b1_inhibitor × is_statin`)
ALONE because rosuvastatin is NOT a CYP3A4 substrate so rule 0 does
not fire — exposing OATP1B1×statin as the most-undertrained sub-class
in the cohort. That claim has no corresponding pin, leaving it
verbal-only and drift-prone.

This pin locks the **encoder→rule mapping per canonical pair**:

* T1: each `_V5_CANONICAL_PINS` pair fires the EXACT documented set of
  pair-derived rule bits (last 13 dims of encode_pair output). Catches:
  - silent rule-table refactor that flips a bit pattern
  - flag-set drift in `pharmacology_flags.json` that breaks an
    expected rule firing
  - encoder-internal rule definition divergence between v3 and v5

* T2: the iter-197 "rule-1-only signal" claim is mechanically
  enforceable — cyclosporine+rosuvastatin must fire rule 1 (and only
  rule 1). The "most-undertrained sub-class" framing in JUDGES + demo
  becomes runtime-verified.

* T3: the 3 demo anchors (warfarin+ibuprofen, atorvastatin+grapefruit,
  amoxicillin+penicillin) fire ZERO pair-derived rules — they're
  cross-class severity demonstrations not rule-derived. A regression
  here would make the demo's BitNet 4.5 walkthrough lie about how
  the classifier actually arrived at its severity verdict.

* T4: cross-pin invariant — every `_V5_EXPECTED_MISSES` pair fires
  AT LEAST 1 pair-derived rule. A miss that fires 0 rules has no
  signal for BOOST_KEYS upweighting to learn from; the v6 retrain
  would be unable to fix it. This makes the queued-fix promise
  mechanical.

This is a **T1 round-39 testing-rigor pin** companion to:
- iter-178 BOOST_KEYS coverage (subset over expected misses)
- iter-183 Q16.16 canonical-pins coverage (subset over expected misses)
- iter-188 encode_pair encoder contract (output shape + symmetry)
- iter-193 audit-replay structural integrity (envelope contract)

Same drift-prevention shape, applied to the per-pair rule-firing
contract that connects pharmacology_flags → encoder → V5 model.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from retrain_runpod.train_bitnet_v3_full import encode_pair  # noqa: E402

# 13 pair-derived rule indices (in encode_pair output[-13:]) — must
# match retrain_runpod/train_bitnet_v5_h128.py::_pair_derived_rules
# return order. Locked at iter-140 (rule 6-12 added; rule 0-5 from
# iter-96 baseline).
_RULE_NAMES = (
    "cyp3a4_inhib_substrate",     # 0
    "oatp1b1_inhib_statin",       # 1
    "p_gp_inhib_substrate",       # 2
    "cyp2c9_inhib_anticoag",      # 3
    "maoi_serotonergic",          # 4
    "pde5_nitrate",               # 5
    "iodinated_metformin",        # 6 (iter-140)
    "cyp1a2_inhib_substrate",     # 7 (iter-140)
    "xanthine_thiopurine",        # 8 (iter-140)
    "folate_pair",                # 9 (iter-140)
    "tetracycline_retinoid",      # 10 (iter-140)
    "ace_neprilysin",             # 11 (iter-140)
    "metformin_renal",            # 12 (iter-140)
)

# Per-pair expected rule-bit set (frozenset of rule indices that fire).
# Computed by inspecting pharmacology_flags.json + the rule definitions
# at iter-197; pin freezes the contract.
_EXPECTED_RULE_FIRINGS: dict[tuple[str, str], frozenset[int]] = {
    # Demo anchors — cross-class severity demonstration, no rule-derived
    # signal. Severity comes from hash-trit + flag-bit contributions
    # alone; rule bits all zero.
    ("warfarin", "ibuprofen"): frozenset(),
    ("atorvastatin", "grapefruit"): frozenset(),
    ("amoxicillin", "penicillin"): frozenset(),
    # Severity-class anchors — at least 1 rule fires.
    ("clarithromycin", "simvastatin"): frozenset({0, 2}),  # CYP3A4 + P-gp
    ("ciprofloxacin", "tizanidine"): frozenset({7}),       # CYP1A2
    ("lisinopril", "sacubitril"): frozenset({11}),         # ACE x neprilysin
    ("allopurinol", "azathioprine"): frozenset({8}),       # xanthine/thiopurine
    ("iodine", "metformin"): frozenset({6}),               # iodinated/metformin
    # iter-172 v5 known-miss — triazole sub-class (CYP3A4 strong inh).
    ("isavuconazole", "simvastatin"): frozenset({0}),
    # iter-177 v5 known-miss — ergot derivative (CYP3A4 substrate).
    ("ketoconazole", "ergotamine"): frozenset({0}),
    # iter-182 v5 known-miss — tetracycline x retinoid pair.
    ("isotretinoin", "minocycline"): frozenset({10}),
    # iter-187 v5 known-miss — benzodiazepine (CYP3A4 substrate).
    ("ketoconazole", "midazolam"): frozenset({0}),
    # iter-192 v5 known-miss — K+-sparing-diuretic (CYP3A4 substrate).
    ("eplerenone", "ketoconazole"): frozenset({0}),
    # iter-197 v5 known-miss — OATP1B1 x statin in PURE ISOLATION.
    # Rosuvastatin is NOT a CYP3A4 substrate so rule 0 does not fire;
    # ONLY rule 1 (OATP1B1 inhib x statin) fires. This is the
    # mechanical lock on the "most-undertrained sub-class" framing
    # in JUDGES.md row 102 + demo.html L1357 callout.
    ("cyclosporine", "rosuvastatin"): frozenset({1}),
    # iter-202 v5 known-miss — V2-receptor-antagonist sub-class
    # (tolvaptan, FDA Samsca/Jynarque § 4 dual-label vs strong
    # CYP3A4 inh). Tolvaptan is is_cyp3a4_substrate so rule 0 fires
    # alone. Same shape as iter-187 midazolam + iter-192 eplerenone
    # — well-trained slot (rule 0) but new substrate sub-class.
    ("tolvaptan", "ketoconazole"): frozenset({0}),
}


def _rule_firings(da: str, db: str) -> frozenset[int]:
    """Extract the pair-derived rule-bit set from encode_pair output."""
    feat = encode_pair(da, db)
    rule_bits = feat[-13:]
    return frozenset(i for i, v in enumerate(rule_bits) if v)


def test_every_canonical_pair_has_pinned_rule_firings() -> None:
    """Each of the 14 V5 canonical pairs must fire EXACTLY the pinned
    set of pair-derived rules. Drift here signals that pharmacology_flags
    lost a flag, the rule-table function got refactored, OR encode_pair
    returned a different output for a known pair."""
    for (da, db), expected in _EXPECTED_RULE_FIRINGS.items():
        live = _rule_firings(da, db)
        assert live == expected, (
            f"Rule-bit drift for ({da}, {db}):\n"
            f"  expected: {sorted(expected)} = "
            f"{[_RULE_NAMES[i] for i in sorted(expected)]}\n"
            f"  live:     {sorted(live)} = "
            f"{[_RULE_NAMES[i] for i in sorted(live)]}\n"
            f"Either pharmacology_flags.json drifted, the rule-table "
            f"function got refactored, or encode_pair changed."
        )


def test_cyclosporine_rosuvastatin_fires_only_oatp1b1_rule_iter197() -> None:
    """Locks the iter-197 'rule-1-only signal' framing: cyclosporine +
    rosuvastatin fires rule 1 (OATP1B1 x statin) AND ONLY rule 1.

    This is the mechanical anchor for the JUDGES.md row 102 claim
    'OATP1B1 x statin slot is the most-undertrained sub-class (all
    three classifiers default to major on this rule-1-only signal)'
    and the demo.html L1357 callout's matching framing.

    If a future flag addition makes this pair fire any other rule
    (e.g., adding is_cyp3a4_substrate to rosuvastatin would fire rule
    0 as well), the iter-197 framing breaks and this pin fires."""
    live = _rule_firings("cyclosporine", "rosuvastatin")
    assert live == frozenset({1}), (
        f"cyclosporine+rosuvastatin must fire ONLY rule 1 "
        f"(oatp1b1_inhib_statin) — pure-isolation test of OATP1B1 x "
        f"statin slot. Got: {sorted(live)} = "
        f"{[_RULE_NAMES[i] for i in sorted(live)]}. "
        f"If rosuvastatin gained is_cyp3a4_substrate flag, rule 0 "
        f"would now fire too — update JUDGES.md + demo.html iter-197 "
        f"framing to remove the 'rule-1-only signal' claim."
    )


def test_demo_anchors_fire_zero_pair_derived_rules() -> None:
    """The 3 demo anchors (warfarin+ibuprofen, atorvastatin+grapefruit,
    amoxicillin+penicillin) demonstrate cross-class severity through
    hash-trit + flag-bit contributions ALONE. None of them fire any
    pair-derived rule.

    This is the mechanical anchor for the demo's BitNet 4.5
    walkthrough — if a future flag addition causes any anchor to
    suddenly fire a rule, the walkthrough's 'these are
    rule-independent severity classifications' framing breaks."""
    anchors = [
        ("warfarin", "ibuprofen"),
        ("atorvastatin", "grapefruit"),
        ("amoxicillin", "penicillin"),
    ]
    for da, db in anchors:
        live = _rule_firings(da, db)
        assert live == frozenset(), (
            f"Demo anchor {da}+{db} unexpectedly fires rules "
            f"{sorted(live)} = {[_RULE_NAMES[i] for i in sorted(live)]}. "
            f"Anchors must remain rule-independent so the BitNet 4.5 "
            f"demo walkthrough's framing stays accurate."
        )


def test_every_v5_expected_miss_fires_at_least_one_rule_iter197() -> None:
    """Cross-pin invariant: every pair in `_V5_EXPECTED_MISSES` (the
    set of known v5 misses queued for v6 retrain via BOOST_KEYS @200x)
    must fire AT LEAST 1 pair-derived rule.

    Why: BOOST_KEYS upweighting only helps if the model has a non-
    trivial signal to learn — a pair that fires 0 rules has no rule-
    derived bits set, leaving the model with only hash-trit + flag-
    bit contributions which are already balanced across classes by
    the iter-148 corpus. A 0-rule miss in BOOST_KEYS is a wasted
    upweight slot.

    Mirror-shape of iter-178 (BOOST_KEYS coverage) + iter-183 (Q16.16
    canonical-pins coverage) — extends cross-pin discipline to the
    encoder-rule-firing contract."""
    # Import the live miss set rather than duplicating it.
    from tests.test_engine.test_path_a_v5_live_recall_pin import (  # noqa: E402
        _V5_EXPECTED_MISSES,
    )
    misses_with_zero_rules: list[tuple[str, str]] = []
    for da, db in _V5_EXPECTED_MISSES:
        live = _rule_firings(da, db)
        if not live:
            misses_with_zero_rules.append((da, db))
    assert not misses_with_zero_rules, (
        f"v5 expected-miss pairs that fire 0 pair-derived rules: "
        f"{misses_with_zero_rules}. BOOST_KEYS @200x upweighting "
        f"on these pairs has no rule-derived signal to learn from "
        f"— v6 retrain would be unable to fix them. Either add a "
        f"new rule to cover the mechanism OR remove from "
        f"_V5_EXPECTED_MISSES + BOOST_KEYS."
    )
