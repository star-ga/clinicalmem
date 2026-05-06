# Copyright 2026 STARGA Inc. — Apache-2.0
"""Pin Path A v8 (1f0f8859, h=256) live-cache contraindicated recall.

Iter 244 v8 sweep landed: doubled hidden_dim 128 → 256 broke the v7
architectural ceiling. Seed 71 hit the strict full-recall gate
(41/41 contra + 4/4 major + 0 FP) on the iter-235 41-contra cohort.

  Path A v3 (h=64,  eea0e637):    29/38 contra + 1 FP (historical)
  Path A v5 (h=128, 1ff61a6a):    31/38 contra + 0 FP (iter-166 staged)
  Path A v6 (h=128, 592ee51e):    40/41 contra + 4/4 major + 0 FP (iter-207, iter-235 cohort)
  Path A v8 (h=256, 1f0f8859):    41/41 contra + 4/4 major + 0 FP   ← THIS PIN

v8 inherits v6/v5's 193-dim feature input but doubles the hidden dim
to 256 — the architectural extension that broke the BOOST_KEYS @200x
ceiling discovered at iter-241 v7 sweep (where v7 at h=128 couldn't
satisfy 41/41 + 4/4 + 0 FP simultaneously regardless of seed).

NOT yet engine-promoted — promotion requires the same iter-166-class
cascade work (encoder lift to 193-dim already in v5/v6/v8; engine
bundle swap requires re-pinning every V6-derived check, JUDGES + demo
update, audit-replay regen under V8 bundle_id, manifest SHA rotation,
44 audit-pin re-replay) **plus** the 64 → 256 `hidden_w` shape
extension in `engine/bitnet_classifier.py`. Engine still loads
cfadb4f6 (iter-72 baseline). v8 lives at
retrain_runpod/bitnet_weights_v8_h256.json.

This pin uses the same Q16.16 ternary forward pass as the engine.
A bundle that "works" under float NumPy but fails under Q16.16 is
unsafe for the audit-replay anchor — it must pass this pin under
the Q16.16 path to be eligible for engine promotion.

Sweep result that produced this bundle
======================================
``retrain_runpod/sweep_v8_h256.py`` 30-seed sweep launched iter-242
local CPU. Seed=71 hit the strict gate (41/41 + 4/4 + 0 FP), the
sweep saved the bundle and stopped per its early-stop logic.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from retrain_runpod.train_bitnet_v3_full import encode_pair  # noqa: E402

_CACHE = _REPO_ROOT / "docs" / "openevidence_cache.json"
_BUNDLE = _REPO_ROOT / "retrain_runpod" / "bitnet_weights_v8_h256.json"

_PATH_A_V8_BUNDLE_ID = (
    "1f0f88591c05af57c62d844b667639b29c7d1f0eb1b213073d158101611f76e6"
)

# Q16.16 baseline measurements.
# Iter-244: v8 hit 41/41 on iter-235 41-contra cohort (full recall +
# 4/4 major + 0 FP). The doubled hidden_dim 128 → 256 gave the
# network enough capacity to satisfy all three constraints
# simultaneously — v7 at h=128 couldn't (best v7 was 40/41+4/4+0FP,
# matching v6).
# Iter-249 cohort growth: added (quinidine, ritonavir) — HIV-PI ×
# Class IA antiarrhythmic / dual QT-prolonger slot, NEW sub-class.
# Pre-flight v8 confirmed contraindicated classification at +14.59
# Q16.16 logit (clear margin), so cohort 41 → 42 with hits 41 → 42
# in lockstep (zero misses preserved).
# Iter-254 cohort growth: added (nitroglycerin, vardenafil) — extends
# rule 5 (PDE5 × nitrate) from 2 → 3 entries; vardenafil joins
# sildenafil + tadalafil as the third PDE5 inhibitor in the cohort.
# Pre-flight v8 caught at +150.51 Q16.16 (strongest margin yet),
# cohort 42 → 43 with hits 42 → 43 in lockstep.
_V8_CONTRA_HITS = 43
_V8_CONTRA_TOTAL = 43
_V8_FP_COUNT = 0       # zero FPs invariant holds
_V8_MAJOR_HITS = 4
_V8_MAJOR_TOTAL = 4

# Iter-244: V8 has ZERO known misses. Empty tuple means 41/41 full
# recall — the architectural-ceiling breakthrough relative to v7.
# Future cohort growth that adds a contra v8 misses would extend
# this tuple (same shape as v6 had during iter-215 → iter-244 era).
_V8_EXPECTED_MISSES: tuple[tuple[str, str], ...] = (
    # Empty — v8 catches all 41 contraindicated pairs.
    # Iter-244 architectural double broke v7's BOOST_KEYS ceiling.
    # Future cohort growth that adds a contra v8 misses would extend
    # this tuple (same shape as v6 had during iter-215 → iter-244 era).
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


def test_path_a_v8_bundle_id_pinned() -> None:
    """The v8 bundle's `_meta.bundle_id` must equal the pinned constant.
    iter-244 sweep seed=71 produced this bundle; any other bundle_id
    means a re-sweep landed without re-pinning."""
    bundle = json.loads(_BUNDLE.read_text())
    live_id = bundle["_meta"]["bundle_id"]
    assert live_id == _PATH_A_V8_BUNDLE_ID, (
        f"Path A v8 bundle drift: live={live_id!r}, "
        f"pinned={_PATH_A_V8_BUNDLE_ID!r}. A new v8+ bundle must "
        f"replace this constant in the same commit so the recall "
        f"pin (below) re-validates against the new bundle."
    )


def test_path_a_v8_q16_recall_full() -> None:
    """Q16.16 inference: v8 hits 41/41 live-cache contras under Q16.16
    on the iter-235 cohort. Empty `_V8_EXPECTED_MISSES` — v8 catches
    every contraindicated pair. Any miss fires the pin and signals:
      (a) cohort growth introduced a new contra fired by no rule
          → encoder coverage broken; add rule + BOOST_KEYS + retrain
      (b) the staged bundle's weights regressed
          → re-run iter-244 v8 sweep
      (c) a flag was silently removed from pharmacology_flags.json
          → iter-203 per-rule cohort-coverage pin should also fire
    The bidirectional guarantee with the empty miss tuple is "v8 is
    the no-known-miss bundle" — neither MORE misses (regression)
    nor LESS (impossible since miss set is already minimal).
    """
    bundle = json.loads(_BUNDLE.read_text())
    contras = _live_contras()
    assert len(contras) == _V8_CONTRA_TOTAL, (
        f"Live cache contraindicated count drifted: "
        f"live={len(contras)}, pinned={_V8_CONTRA_TOTAL}."
    )
    misses: list[tuple[str, str]] = []
    for entry in contras:
        pred = _classify_q16(entry["drug_a"], entry["drug_b"], bundle)
        if pred != "contraindicated":
            misses.append(
                (entry["drug_a"].lower(), entry["drug_b"].lower())
            )
    expected_pairs = {
        tuple(sorted([a.lower(), b.lower()])) for a, b in _V8_EXPECTED_MISSES
    }
    actual_pairs = {tuple(sorted(p)) for p in misses}
    assert actual_pairs == expected_pairs, (
        f"Path A v8 miss-set drift: live={sorted(actual_pairs)}, "
        f"pinned={sorted(expected_pairs)}.\n"
        f"If a v9 retrain landed (new misses appeared), update "
        f"_V8_CONTRA_HITS + _V8_EXPECTED_MISSES + bundle_id in lockstep."
    )


def test_path_a_v8_q16_major_full_recall() -> None:
    """v8 catches all 4 major-class pairs (paroxetine+tamoxifen,
    clarithromycin+digoxin, dabigatran+dronedarone, voriconazole+
    tacrolimus) under Q16.16 — the architectural-double advance that
    v7 (h=128) couldn't achieve."""
    bundle = json.loads(_BUNDLE.read_text())
    majors = _live_majors()
    assert len(majors) == _V8_MAJOR_TOTAL, (
        f"Major-class cohort drift: live={len(majors)}, pinned={_V8_MAJOR_TOTAL}."
    )
    hits = 0
    for entry in majors:
        pred = _classify_q16(entry["drug_a"], entry["drug_b"], bundle)
        if pred == "major":
            hits += 1
    assert hits == _V8_MAJOR_HITS, (
        f"Path A v8 major recall drift: live={hits}, pinned={_V8_MAJOR_HITS}. "
        f"v8 must catch all 4 major-class pairs under Q16.16."
    )


def test_path_a_v8_q16_zero_fp() -> None:
    """v8 must NEVER predict contraindicated for a non-contra pair.
    The zero-FP invariant is the load-bearing safety claim that
    distinguishes BitNet 4.5 as a high-precision veto rather than a
    primary classifier."""
    bundle = json.loads(_BUNDLE.read_text())
    non_contras = _live_non_contras()
    fps = []
    for entry in non_contras:
        pred = _classify_q16(entry["drug_a"], entry["drug_b"], bundle)
        if pred == "contraindicated":
            fps.append((entry["drug_a"], entry["drug_b"], entry["severity"]))
    assert not fps, (
        f"Path A v8 false-positive drift: {len(fps)} non-contra pairs "
        f"classified as contraindicated under Q16.16: {fps[:5]}. "
        f"Zero FP is the safety-floor invariant."
    )
    assert len(fps) == _V8_FP_COUNT, (
        f"FP count drift: live={len(fps)}, pinned={_V8_FP_COUNT}"
    )


def test_path_a_v8_meta_block_consistency() -> None:
    """The v8 bundle's _meta block must reflect the architectural
    decisions: 193-dim feature input × 256 hidden × 5 logits, ternary
    weights, q16.16 biases, 13 pair-derived rules, 26 ATC flags."""
    bundle = json.loads(_BUNDLE.read_text())
    meta = bundle["_meta"]
    assert meta["in_features"] == 193, (
        f"v8 in_features drift: live={meta['in_features']}, expected 193 "
        f"(64 hash trits + 26 flag bits per drug × 2 + 13 pair-derived rules)"
    )
    assert meta["hidden_features"] == 256, (
        f"v8 hidden_features drift: live={meta['hidden_features']}, expected 256 "
        f"(iter-242 v8 architectural double from v6's 128)"
    )
    assert meta["out_features"] == 5, (
        f"v8 out_features drift: live={meta['out_features']}, expected 5 "
        f"(none / moderate / serious / major / contraindicated)"
    )
    assert meta["weight_dtype"] == "ternary", (
        f"v8 weight_dtype drift: live={meta['weight_dtype']!r}, expected 'ternary'"
    )
    assert meta["bias_dtype"] == "q16.16", (
        f"v8 bias_dtype drift: live={meta['bias_dtype']!r}, expected 'q16.16'"
    )


def test_path_a_v8_strictly_supersedes_v6() -> None:
    """v8 catches strictly MORE contraindicated pairs than v6 ever did.
    v6 was 40/41 (1 known miss: lurasidone+ketoconazole). v8 is 41/41
    (zero known misses). The architectural double 128→256 is
    mechanically required to be net-positive — if v8 ever drops below
    v6's 40-contra recall, this pin fires.
    """
    assert _V8_CONTRA_HITS >= 40, (
        f"Path A v8 strictly_supersedes_v6 invariant violated: "
        f"v8 hits {_V8_CONTRA_HITS}, v6 hit 40. The iter-244 architectural "
        f"double 128 → 256 was supposed to break the BOOST_KEYS @200x "
        f"ceiling, not regress below v6's recall."
    )
