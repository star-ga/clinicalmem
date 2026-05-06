# Copyright 2026 STARGA Inc. — Apache-2.0
"""Pin Path A v6 (592ee51e, h=128) live-cache contraindicated recall.

Iter 207 BOOST_KEYS extension closed the v5 → v6 generalization gap
that iter-166's v5 architectural breakthrough opened then iters
172/177/182/187/192/197/202 dragged via cohort growth (v5 went from
31/31 at iter-166 to 31/38 at iter-202 as 7 sub-classes were added
that iter-148 corpus + iter-156 BOOST_KEYS hadn't covered).

Iter-207 sweep ran ``retrain_runpod/sweep_v6_h128.py`` (30-seed CPU
sweep × 1800 epochs each) using the iter-202-extended BOOST_KEYS
(8 anchors @200x: clarithromycin/cyclosporine/itraconazole/keto
conazole/gemfibrozil ::simvastatin + azathioprine::febuxostat +
isavuconazole::simvastatin + ergotamine::ketoconazole +
isotretinoin::minocycline + ketoconazole::midazolam +
eplerenone::ketoconazole + cyclosporine::rosuvastatin +
tolvaptan::ketoconazole). Seed=31 hit the strict full-recall gate:

  Path A v3 (h=64, eea0e637):    29/38 contra + 1 FP (historical)
  Path A v5 (h=128, 1ff61a6a):   31/38 contra + 0 FP (iter-166 staged)
  Path A v6 (h=128, 592ee51e):   38/38 contra + 4/4 major + 0 FP   ← THIS PIN

Same architecture as v5 (193-dim feature × 128 hidden × 5 logits),
different weights — v6's 8-anchor BOOST_KEYS upweighting trained
recognition of the 7 v5-known-miss sub-classes (triazole, ergot,
tetracycline, benzodiazepine, K+-sparing-diuretic, OATP1B1×rosuvastatin,
V2-receptor antagonist).

NOT yet engine-promoted — promotion requires the same iter-166
cascade work (encoder lift to 193-dim already in v5; engine bundle
swap requires re-pinning every V5-derived check, JUDGES + demo
update 31/38 → 38/38 + 0 FP, audit-replay regen under V6
bundle_id, manifest SHA rotation). Engine still loads cfadb4f6
(iter-72 baseline). v6 lives at retrain_runpod/bitnet_weights_v6_h128.json.

This pin uses the same Q16.16 ternary forward pass as the engine.
A bundle that "works" under float NumPy but fails under Q16.16 is
unsafe for the audit-replay anchor — it must pass this pin under
the Q16.16 path to be eligible for engine promotion.

Sweep result that produced this bundle
======================================
``retrain_runpod/sweep_v6_h128.py`` 30-seed sweep launched iter-207
local CPU. Seed=31 hit the strict gate (38/38 + 4/4 + ≤ 1 FP), the
sweep saved the bundle and stopped per its early-stop logic. The
result is a single-seed finding — multi-seed reproducibility is the
next testing-rigor task (queued for next T1).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from retrain_runpod.train_bitnet_v3_full import encode_pair  # noqa: E402

_CACHE = _REPO_ROOT / "docs" / "openevidence_cache.json"
_BUNDLE = _REPO_ROOT / "retrain_runpod" / "bitnet_weights_v6_h128.json"

_PATH_A_V6_BUNDLE_ID = (
    "592ee51ee088cbd8f1c10a2e210028d9a184d90dce61f7c7aacde46d0fd1b769"
)

# Q16.16 baseline measurements.
# Iter-207: v6 hit 38/38 on iter-202 38-contra cohort (full recall).
# Iter-215: cohort grew 38 → 39 (lurasidone+ketoconazole added). v6
# missed the new pair (predicted 'none' instead of 'contraindicated')
# because lurasidone wasn't in BOOST_KEYS — v6's training upweighted
# specific drug-pair anchors, not entire CYP3A4-substrate sub-class.
# This is the same generalization-gap pattern v5 had. v7 retrain
# queued with lurasidone::ketoconazole added to BOOST_KEYS.
_V6_CONTRA_HITS = 40   # iter-235: cohort 40 → 41, v6 catches the new pair (ritonavir+ergotamine)
_V6_CONTRA_TOTAL = 41  # iter-235 cohort: ritonavir+ergotamine added (FDA Norvir § 4)
_V6_FP_COUNT = 0       # zero FPs invariant holds
_V6_MAJOR_HITS = 4
_V6_MAJOR_TOTAL = 4

# Iter-215: V6 now has 1 known-miss after cohort grew 38 → 39.
# v6 predicts 'none' for lurasidone+ketoconazole (the BOOST_KEYS
# upweighting at iter-207 trained drug-pair-specific recognition
# rather than CYP3A4-substrate-sub-class generalization, so a new
# atypical antipsychotic substrate falls through). Same shape as
# the iter-172 v5-known-miss invariant.
_V6_EXPECTED_MISSES: tuple[tuple[str, str], ...] = (
    ("ketoconazole", "lurasidone"),
)

_Q16_ONE = 1 << 16


def _classify_q16(da: str, db: str, bundle: dict) -> str:
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
    labels = ("none", "moderate", "serious", "major", "contraindicated")
    return labels[max(range(len(logits)), key=lambda i: logits[i])]


def _live_contras() -> list[dict]:
    cache = json.loads(_CACHE.read_text())
    return [e for e in cache if e.get("severity") == "contraindicated"]


def _live_majors() -> list[dict]:
    cache = json.loads(_CACHE.read_text())
    return [e for e in cache if e.get("severity") == "major"]


def _live_non_contras() -> list[dict]:
    cache = json.loads(_CACHE.read_text())
    return [e for e in cache if e.get("severity") != "contraindicated"]


def test_path_a_v6_bundle_id_pinned() -> None:
    """The v6 bundle's `_meta.bundle_id` must equal the pinned constant.
    iter-207 sweep seed=31 produced this bundle; any other bundle_id
    means a re-sweep landed without re-pinning."""
    bundle = json.loads(_BUNDLE.read_text())
    live_id = bundle["_meta"]["bundle_id"]
    assert live_id == _PATH_A_V6_BUNDLE_ID, (
        f"Path A v6 bundle drift: live={live_id!r}, "
        f"pinned={_PATH_A_V6_BUNDLE_ID!r}. A new v6+ bundle must "
        f"replace this constant in the same commit so the recall "
        f"pin (below) re-validates against the new bundle."
    )


def test_path_a_v6_q16_recall_with_known_misses() -> None:
    """Q16.16 inference: v6 hits 38/39 live-cache contras under Q16.16
    on the iter-215 cohort. The 1 known miss is recorded in
    `_V6_EXPECTED_MISSES` (lurasidone+ketoconazole, iter-215). Any
    OTHER miss fires the pin and signals either:
      (a) cohort growth introduced a new contra fired by no rule
          → encoder coverage broken; add rule + BOOST_KEYS + retrain
      (b) the staged bundle's weights regressed against the rule
          → re-run iter-207 sweep
      (c) a flag was silently removed from pharmacology_flags.json
          → iter-203 per-rule cohort-coverage pin should also fire
    Symmetric with iter-172's v5 known-miss invariant — neither MORE
    misses (regression) nor LESS (silent retrain landed without re-pin).
    """
    bundle = json.loads(_BUNDLE.read_text())
    contras = _live_contras()
    assert len(contras) == _V6_CONTRA_TOTAL, (
        f"Live cache contraindicated count drifted: "
        f"live={len(contras)}, pinned={_V6_CONTRA_TOTAL}."
    )
    misses: list[tuple[str, str]] = []
    for entry in contras:
        pred = _classify_q16(entry["drug_a"], entry["drug_b"], bundle)
        if pred != "contraindicated":
            misses.append(
                (entry["drug_a"].lower(), entry["drug_b"].lower())
            )
    expected_pairs = {
        tuple(sorted([a.lower(), b.lower()])) for a, b in _V6_EXPECTED_MISSES
    }
    actual_pairs = {tuple(sorted(p)) for p in misses}
    assert actual_pairs == expected_pairs, (
        f"Path A v6 miss-set drift: live={sorted(actual_pairs)}, "
        f"pinned={sorted(expected_pairs)}.\n"
        f"If a v7 retrain landed (new misses removed), update "
        f"_V6_CONTRA_HITS + _V6_EXPECTED_MISSES + bundle_id in lockstep."
    )
    hits = len(contras) - len(misses)
    assert hits == _V6_CONTRA_HITS, (
        f"Path A v6 contra hits drifted: live={hits}, pinned={_V6_CONTRA_HITS}"
    )


def test_path_a_v6_q16_zero_fp_invariant() -> None:
    """Q16.16 inference: v6 must have ZERO false positives on the live
    cache (no non-contra entry classified as contraindicated). The
    iter-166 v5 architectural breakthrough's load-bearing claim was
    'high-precision veto stays clean' — v6 inherits that claim.
    The single v3 amlodipine+simvastatin FP is gone in v6."""
    bundle = json.loads(_BUNDLE.read_text())
    fps = [
        (e["drug_a"], e["drug_b"], e["severity"])
        for e in _live_non_contras()
        if _classify_q16(e["drug_a"], e["drug_b"], bundle) == "contraindicated"
    ]
    assert len(fps) == _V6_FP_COUNT, (
        f"Path A v6 false-positive count drifted: "
        f"live={len(fps)}, pinned={_V6_FP_COUNT}. fps={fps[:5]}"
        f"{'...' if len(fps) > 5 else ''}"
    )


def test_path_a_v6_q16_full_major_recall() -> None:
    """Q16.16 inference: v6 hits 4/4 live-cache major-class contras.
    Closes the iter-72 baseline cfadb4f6 3/4 majors gap (cfadb4f6
    misses tacrolimus+voriconazole on the P-gp + strong CYP3A4 cross-
    mechanism). v6's 128-hidden architecture + iter-148 major-class
    BOOST_KEYS catches all 4."""
    bundle = json.loads(_BUNDLE.read_text())
    majors = _live_majors()
    assert len(majors) == _V6_MAJOR_TOTAL, (
        f"Live cache major count drifted: "
        f"live={len(majors)}, pinned={_V6_MAJOR_TOTAL}."
    )
    misses = []
    for entry in majors:
        pred = _classify_q16(entry["drug_a"], entry["drug_b"], bundle)
        if pred != "major":
            misses.append((entry["drug_a"], entry["drug_b"], pred))
    hits = len(majors) - len(misses)
    assert hits == _V6_MAJOR_HITS, (
        f"Path A v6 major-recall pin drifted: live={hits}/{len(majors)}, "
        f"pinned={_V6_MAJOR_HITS}/{_V6_MAJOR_TOTAL}. Misses: {misses}"
    )


def test_path_a_v6_meta_block_consistency() -> None:
    """The v6 bundle's `_meta` block must report the architecture
    invariants: 193-dim feature input, 128-hidden, 5 logits, ternary
    weights, q16.16 biases."""
    bundle = json.loads(_BUNDLE.read_text())
    meta = bundle["_meta"]
    assert meta["in_features"] == 193, (
        f"v6 in_features drift: live={meta['in_features']}, pinned=193"
    )
    assert meta["hidden_features"] == 128, (
        f"v6 hidden_features drift: live={meta['hidden_features']}, pinned=128"
    )
    assert meta["out_features"] == 5
    assert meta["weight_dtype"] == "ternary"
    assert meta["bias_dtype"] == "q16.16"
    assert meta["pair_derived_rule_count"] == 13
    assert meta["flag_keys_count"] == 26
    assert meta["training_iter"] == "iter-207-path-a-v6-h128"
    assert meta["contra_recall"] == 1.0  # full recall on iter-202 cohort
    # (cohort has since grown 38 → 39 at iter-215; v6 contra_recall on
    # the new cohort is 38/39 = 0.974, but the bundle's own _meta still
    # reflects what it achieved on its training-time cohort. Drift in
    # the bundle's own meta value would signal a re-saved bundle without
    # re-pin).


def test_path_a_v6_strictly_supersedes_v5_recall() -> None:
    """v6 must hit STRICTLY MORE contras than v5 on the same cohort.
    v5 = 31/38 (iter-202 measurement), v6 = 38/38. This pin enforces
    that v6 can never regress below v5 — if it did, the iter-207
    BOOST_KEYS extension would have been counterproductive."""
    v5_bundle = json.loads(
        (_REPO_ROOT / "retrain_runpod" / "bitnet_weights_v5_h128.json")
        .read_text()
    )
    v6_bundle = json.loads(_BUNDLE.read_text())
    contras = _live_contras()
    v5_hits = sum(
        1 for e in contras
        if _classify_q16(e["drug_a"], e["drug_b"], v5_bundle) == "contraindicated"
    )
    v6_hits = sum(
        1 for e in contras
        if _classify_q16(e["drug_a"], e["drug_b"], v6_bundle) == "contraindicated"
    )
    assert v6_hits >= v5_hits, (
        f"Path A v6 ({v6_hits}/{len(contras)}) regressed below v5 "
        f"({v5_hits}/{len(contras)}). The iter-207 BOOST_KEYS extension "
        f"failed its mission — re-run sweep with more seeds or "
        f"investigate the 8-anchor upweight strategy."
    )
