# Copyright 2026 STARGA Inc. — Apache-2.0
"""Pin Path A v3 (eea0e637) live-cache contraindicated recall.

Iter 164 added the 31st contraindicated entry (atazanavir + simvastatin)
and corrected an iter-155 measurement artifact. The original iter-155
demo claim — "Path A v3 = 29/30 = 96.7% (regressed by febuxostat +
azathioprine cohort growth)" — was incorrect. The
``is_xanthine_oxidase_inhibitor × is_thiopurine`` pair-derived rule
fires for both ``allopurinol+azathioprine`` AND
``febuxostat+azathioprine``, and v3's trained weights project both
through the same hidden activation, so v3 actually catches the
febuxostat+azathioprine pair.

This pin locks the corrected measurement: Path A v3 (eea0e637) hits
**31/31 = 100%** on the iter-164 live cache. The pin fires on any
of the following:

  * The staged Path A v3 bundle's contraindicated recall regresses
    below the iter-164 floor (live-cache contras → 30 or fewer hits)
  * The bundle's ``bundle_id_hex`` is swapped without updating this
    pin — a future Path A v4/v5/etc. promotion must replace the
    pinned bundle_id explicitly
  * Cohort growth introduces a contra that the existing v3 weights
    do NOT catch via the existing pair-derived rules — that's a
    BOOST_KEYS-add or retrain signal, NOT a silent regression

Why this pin matters
====================
Path A v3 is the staged engine candidate (deferred behind the 4-6 h
encoder cascade refactor). Live-cache recall on the staged candidate
is one of the most load-bearing claims in the demo. Pinning it
prevents a measurement-artifact recurrence and gives the engine
promotion path a hard contract: the bundle must hit 31/31 before it
can be promoted, and any cohort growth that v3 misses fires the
pin in CI.

Pair-derived rule coverage
==========================
v3's 100% live-cache recall is achieved entirely through the 13
pair-derived DDI-rule bits in ``encode_pair``. Each contra in the
cohort fires at least one rule; the trained 64-hidden weights
project all rule-firing pairs through a hidden-activation that
projects to the contraindicated logit being argmax. A future cohort
growth that does NOT fire any existing pair-derived rule will fail
this pin — that's the design signal to add a new pair-derived rule
+ BOOST_KEYS entry + retrain.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from retrain_runpod.train_bitnet_v3_full import encode_pair  # noqa: E402

_CACHE = _REPO_ROOT / "docs" / "openevidence_cache.json"
_BUNDLE = _REPO_ROOT / "retrain_runpod" / "bitnet_weights_v3_full.json"

# Frozen bundle identity for Path A v3. A future Path A v4 promotion
# replaces this constant; CI fires the pin until the constant matches
# the live bundle so the test must be re-pinned in the same commit.
_PATH_A_V3_BUNDLE_ID = (
    "eea0e637b537507352950feeb413da33d81775e41fb7c7ec2aaed94f464dcb2c"
)

# Iter-164 baseline: 31/31 contras on the live cache.
_IT164_CONTRA_FLOOR = 31

_LABEL_NAMES = ("none", "moderate", "serious", "major", "contraindicated")


def _load_bundle() -> dict:
    return json.loads(_BUNDLE.read_text())


def _classify(da: str, db: str, weights: dict) -> str:
    feat = np.array(encode_pair(da, db), dtype=np.float32)
    W1 = np.array(weights["hidden_w"], dtype=np.float32)
    b1 = np.array(weights["hidden_b"], dtype=np.float32)
    W2 = np.array(weights["output_w"], dtype=np.float32)
    b2 = np.array(weights["output_b"], dtype=np.float32)
    h = np.maximum(0.0, W1 @ feat + b1)
    out = W2 @ h + b2
    return _LABEL_NAMES[int(np.argmax(out))]


def _live_contras() -> list[dict]:
    cache = json.loads(_CACHE.read_text())
    return [e for e in cache if e.get("severity") == "contraindicated"]


def test_path_a_v3_bundle_id_pinned() -> None:
    """The staged Path A v3 bundle's `_meta.bundle_id` must equal the
    pinned constant. A future bundle rotation must replace this pin
    in the same commit so the recall-pin (below) is re-validated
    against the new bundle."""
    bundle = _load_bundle()
    live_id = bundle["_meta"]["bundle_id"]
    assert live_id == _PATH_A_V3_BUNDLE_ID, (
        f"Path A v3 bundle drift: live={live_id!r}, "
        f"pinned={_PATH_A_V3_BUNDLE_ID!r}. A new bundle was staged "
        f"without updating this pin. Update _PATH_A_V3_BUNDLE_ID + "
        f"_IT164_CONTRA_FLOOR in the same commit so recall is "
        f"re-validated against the new bundle."
    )


def test_path_a_v3_hits_all_live_contras() -> None:
    """Path A v3 must catch every contra in the live cache. iter-164
    cohort: 31/31 = 100%. A miss fires the pin and signals either:
      (a) cohort growth introduced a contra not covered by any
          existing pair-derived DDI rule (add the rule + BOOST_KEYS
          + retrain), OR
      (b) the staged bundle's weights regressed against the rule
          (retrain or roll back).
    """
    bundle = _load_bundle()
    contras = _live_contras()
    assert len(contras) == _IT164_CONTRA_FLOOR, (
        f"Live cache contraindicated count drifted: "
        f"live={len(contras)}, pinned={_IT164_CONTRA_FLOOR}. "
        f"Cohort growth requires bumping _IT164_CONTRA_FLOOR + "
        f"re-validating that v3 still hits all of them."
    )
    misses: list[tuple[str, str, str]] = []
    for entry in contras:
        da = entry["drug_a"]
        db = entry["drug_b"]
        pred = _classify(da, db, bundle)
        if pred != "contraindicated":
            misses.append((da, db, pred))
    assert not misses, (
        f"Path A v3 ({_PATH_A_V3_BUNDLE_ID[:16]}...) regressed on "
        f"{len(misses)} live-cache contraindicated pair(s): {misses}. "
        f"Either add the missed pair to BOOST_KEYS + retrain, or roll "
        f"back to the bundle that achieved 31/31 at iter-164."
    )


def test_path_a_v3_recall_floor_invariant() -> None:
    """Iter-164 baseline locks 100% live-cache recall. Any future
    cohort growth that v3 doesn't catch fires this floor pin first
    (the per-pair miss list above is the diagnostic; this is the
    aggregate)."""
    bundle = _load_bundle()
    contras = _live_contras()
    hits = sum(
        1 for e in contras
        if _classify(e["drug_a"], e["drug_b"], bundle) == "contraindicated"
    )
    assert hits >= _IT164_CONTRA_FLOOR, (
        f"Path A v3 live-cache contra recall regressed: "
        f"hits={hits}, floor={_IT164_CONTRA_FLOOR}. The staged "
        f"bundle no longer satisfies the iter-164 live-cache "
        f"recall claim shipped to demo + JUDGES."
    )


def test_path_a_v3_meta_block_consistency() -> None:
    """The bundle's `_meta` block must report the architecture
    invariants the demo claims (193-dim feature input, 64-hidden,
    13 pair-derived rules, 26 ATC flag keys). A future promotion
    must re-pin all four numbers in lockstep."""
    bundle = _load_bundle()
    meta = bundle["_meta"]
    assert meta["in_features"] == 193, (
        f"Path A v3 in_features drifted: live={meta['in_features']}, "
        f"pinned=193 (64 hash trits + 26 flag bits per drug × 2 + "
        f"13 pair-derived = 193)"
    )
    assert meta["hidden_features"] == 64, (
        f"Path A v3 hidden_features drifted: "
        f"live={meta['hidden_features']}, pinned=64. The 64-hidden "
        f"ceiling on the strict curated 30/30+4/4+≤1FP gate is the "
        f"architectural finding shipped in iter-161; bumping to 128 "
        f"requires the engine cascade refactor."
    )
    assert meta["out_features"] == 5, (
        f"Path A v3 out_features drifted: "
        f"live={meta['out_features']}, pinned=5 (none, moderate, "
        f"serious, major, contraindicated)"
    )
    assert meta["pair_derived_rule_count"] == 13, (
        f"Path A v3 pair-derived rule count drifted: "
        f"live={meta['pair_derived_rule_count']}, pinned=13. Adding "
        f"a new rule requires re-encoding + retraining the bundle."
    )
    assert meta["flag_keys_count"] == 26, (
        f"Path A v3 flag_keys_count drifted: "
        f"live={meta['flag_keys_count']}, pinned=26 (13 baseline "
        f"iter-96 + 12 iter-140 closure + 1 iter-146 FDA-correct "
        f"re-tier)"
    )
    assert meta["weight_dtype"] == "ternary", (
        f"Path A v3 weight_dtype drifted: "
        f"live={meta['weight_dtype']}, pinned=ternary (BitNet b1.58)"
    )
    assert meta["bias_dtype"] == "q16.16", (
        f"Path A v3 bias_dtype drifted: "
        f"live={meta['bias_dtype']}, pinned=q16.16 (Q16.16 fixed "
        f"point for cross-arch determinism)"
    )
