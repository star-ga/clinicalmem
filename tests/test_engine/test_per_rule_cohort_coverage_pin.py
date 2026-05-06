# Copyright 2026 STARGA Inc. — Apache-2.0
"""Per-rule cohort-coverage pin for the 13 pair-derived rules.

Iter-198 added a per-pair rule-bit pin for the 14 V5 canonical pins
+ a cross-pin invariant (every `_V5_EXPECTED_MISSES` pair fires ≥ 1
rule). This pin extends the contract to the **entire contraindicated
cohort**:

* T1: **encoder-level coverage** — every contraindicated cache entry
  fires AT LEAST 1 pair-derived rule (encoder-side pin; complements
  the iter-100 + iter-140 explanation-coverage pin which checks the
  curated-table explanation logic, not the encoder). Catches the
  drift class where a new contra is added that fires NO rule because
  pharmacology_flags is missing a key flag for one of the drugs.

* T2: **per-rule distribution snapshot** — pins the count of contras
  firing each of the 13 rules. Catches a refactor that silently flips
  which rule a given pair fires (e.g., changing `is_cyp3a4_substrate`
  to a stricter-criteria flag would drop rule 0 from 15 → 14 silently).

* T3: **rule 3 (CYP2C9 × anticoag) zero-cohort design state** — the
  rule is structurally present in the encoder for a reason: warfarin
  + fluconazole + miconazole etc. interact via CYP2C9 inhibition, but
  the FDA Coumadin label classifies these as § 5.4 "monitor INR + dose
  reduce" rather than § 4 absolute contraindications. The cohort
  therefore has 0 examples by design (anticoagulation interactions are
  monitored not blocked). A future cohort growth that adds a CYP2C9 ×
  anticoag contra MUST update this pin in lockstep — preventing
  silent drift away from the documented design state.

This is a **T1 round-40 testing-rigor pin** companion to:
- iter-178 BOOST_KEYS coverage cross-pin
- iter-183 Q16.16 canonical-pins coverage cross-pin
- iter-188 encode_pair encoder contract
- iter-193 audit-replay structural integrity
- iter-198 per-pair rule-bit cardinality

Same drift-prevention shape, applied at the cohort-aggregate level
(rather than per-pair) so a cohort-growth event that drops a rule's
count to 0 cannot silently land.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from retrain_runpod.train_bitnet_v3_full import encode_pair  # noqa: E402

_CACHE = _REPO_ROOT / "docs" / "openevidence_cache.json"

_RULE_NAMES = (
    "cyp3a4_inhib_substrate",     # 0
    "oatp1b1_inhib_statin",       # 1
    "p_gp_inhib_substrate",       # 2
    "cyp2c9_inhib_anticoag",      # 3
    "maoi_serotonergic",          # 4
    "pde5_nitrate",               # 5
    "iodinated_metformin",        # 6
    "cyp1a2_inhib_substrate",     # 7
    "xanthine_thiopurine",        # 8
    "folate_pair",                # 9
    "tetracycline_retinoid",      # 10
    "ace_neprilysin",             # 11
    "metformin_renal",            # 12
)

# Per-rule expected count of contraindicated cache entries that fire
# the rule. Pinned at iter-202 cohort = 38 contras.
_EXPECTED_PER_RULE_CONTRA_COUNTS: tuple[int, ...] = (
    19,  # 0  cyp3a4_inhib_substrate    (saturated; iter-235 +1 ritonavir+ergotamine; iter-249 +1 quinidine+ritonavir)
    3,   # 1  oatp1b1_inhib_statin       (gemfibrozil+simva, cyclo+simva,
         #                                 cyclosporine+rosuvastatin iter-197)
    7,   # 2  p_gp_inhib_substrate       (iter-230 +1: clari+tacrolimus also fires
         #                                 rule 2 because tacrolimus is P-gp substrate)
    0,   # 3  cyp2c9_inhib_anticoag      (BY DESIGN — see test below)
    8,   # 4  maoi_serotonergic          (well-evidenced class)
    2,   # 5  pde5_nitrate
    2,   # 6  iodinated_metformin        (contrast dye + iodine)
    2,   # 7  cyp1a2_inhib_substrate
    2,   # 8  xanthine_thiopurine
    1,   # 9  folate_pair                (only methotrexate+TMP-SMX)
    2,   # 10 tetracycline_retinoid      (doxycycline+isotretinoin,
         #                                 minocycline+isotretinoin iter-182)
    1,   # 11 ace_neprilysin             (only lisinopril+sacubitril)
    1,   # 12 metformin_renal            (only metformin+renal-impairment)
)

assert len(_EXPECTED_PER_RULE_CONTRA_COUNTS) == 13, "rule count drift"


def _live_contras() -> list[dict]:
    cache = json.loads(_CACHE.read_text())
    return [e for e in cache if e.get("severity") == "contraindicated"]


def _per_rule_counts(contras: list[dict]) -> list[int]:
    counts = [0] * 13
    for e in contras:
        feat = encode_pair(e["drug_a"], e["drug_b"])
        for i, v in enumerate(feat[-13:]):
            counts[i] += v
    return counts


def test_every_contra_fires_at_least_one_rule_iter202() -> None:
    """Encoder-level coverage: every contraindicated cache entry must
    fire AT LEAST 1 pair-derived rule. Catches the drift where a new
    contra is added without ensuring pharmacology_flags has the
    necessary flag for one of the drugs (e.g., adding a new drug
    pair without flagging the inhibitor side, leaving rule firings
    silent and the v6 retrain unable to learn from the example)."""
    contras = _live_contras()
    zero_rule_pairs: list[tuple[str, str]] = []
    for e in contras:
        feat = encode_pair(e["drug_a"], e["drug_b"])
        if sum(feat[-13:]) == 0:
            zero_rule_pairs.append((e["drug_a"], e["drug_b"]))
    assert not zero_rule_pairs, (
        f"Contraindicated cache entries that fire 0 pair-derived rules: "
        f"{zero_rule_pairs}. Either add the missing pharmacology_flags "
        f"flag for one of the drugs OR document this pair as "
        f"unexplained (which would also fail the iter-140 explanation-"
        f"coverage pin). Encoder-level coverage must be 100%."
    )


def test_per_rule_contra_distribution_pinned_iter202() -> None:
    """Per-rule count of contras firing the rule must match the pinned
    distribution. Catches:
      - silent flag rename or removal that drops a rule's firing count
      - cohort-growth drift that changes the distribution without
        updating this pin in lockstep
    """
    contras = _live_contras()
    live_counts = _per_rule_counts(contras)
    expected = list(_EXPECTED_PER_RULE_CONTRA_COUNTS)
    if live_counts != expected:
        diffs = []
        for i in range(13):
            if live_counts[i] != expected[i]:
                diffs.append(
                    f"rule {i:2d} ({_RULE_NAMES[i]:25s}): "
                    f"live={live_counts[i]} expected={expected[i]}"
                )
        pytest.fail(
            "Per-rule contra distribution drifted:\n  "
            + "\n  ".join(diffs)
            + f"\nLive total: {sum(live_counts)} contras, "
            + f"expected {sum(expected)}. "
            + "If this is intentional cohort growth, update "
            + "_EXPECTED_PER_RULE_CONTRA_COUNTS in lockstep."
        )


def test_rule3_cyp2c9_anticoag_has_zero_contras_by_design_iter202() -> None:
    """Rule 3 (CYP2C9 × anticoag) has 0 contraindicated cache entries
    BY DESIGN. The rule is structurally available in the encoder
    because warfarin (anticoag) + fluconazole/miconazole/voriconazole
    (CYP2C9 inhibitors) interact via CYP2C9 inhibition. However, the
    FDA Coumadin label classifies these interactions as **§ 5.4 "monitor
    INR + dose reduce"** rather than § 4 absolute contraindications —
    anticoagulation interactions are managed by monitoring, not blocked.

    A future cohort growth that adds a CYP2C9 × anticoag contra (e.g.,
    a hypothetical FDA label change classifying fluconazole+warfarin
    as § 4) MUST update this pin in lockstep — preventing silent drift
    away from the documented design state.

    This pin documents the intentional 0-coverage state mechanically.
    """
    contras = _live_contras()
    live_counts = _per_rule_counts(contras)
    rule3_count = live_counts[3]
    assert rule3_count == 0, (
        f"Rule 3 (cyp2c9_inhib_anticoag) fires on {rule3_count} contras "
        f"but the pinned design state is 0 (anticoagulation interactions "
        f"are § 5.4 monitor-and-adjust, not § 4 absolute contras). "
        f"If a new FDA label changes the classification AND this pair is "
        f"now in cohort, update both this pin AND "
        f"_EXPECTED_PER_RULE_CONTRA_COUNTS[3] in lockstep."
    )


def test_total_per_rule_firings_at_least_cohort_size_iter202() -> None:
    """Sanity floor: sum of per-rule firings must be >= cohort size
    (every contra fires ≥ 1 rule, some fire ≥ 2). Catches a regression
    where the rule-bit reduction in some encoder refactor leaves more
    than half the cohort uncovered."""
    contras = _live_contras()
    total_firings = sum(_per_rule_counts(contras))
    n = len(contras)
    assert total_firings >= n, (
        f"Total per-rule firings {total_firings} < cohort size {n}. "
        f"At least {n - total_firings} contras fire 0 rules — "
        f"encoder coverage broke."
    )
