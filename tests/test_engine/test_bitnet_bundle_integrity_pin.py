"""Pin the BitNet weights bundle integrity (size + hash + sparsity).

Iter 130 (round 26 T1 substitution).

`engine/bitnet_weights.json` is the load-bearing reproducibility
primitive: every clinical decision Layer 4.5 makes carries a
`weights_id` (SHA-256 of the canonical-JSON form of the 4 weight
matrices). The bundle is judge-visible in three places:

  1. JUDGES.md row 100: "8,512 ternary weights + 69 Q16.16 biases =
     8,581 params / 19 KB + bundle hash `cfadb4f6…`"
  2. demo.html L1047: `<span>bundle_id cfadb4f6…0b3f</span>`
  3. demo.html L1895: `<div>...build... cfadb4f6</div>`
  4. README L130: "...19 KB weights bundle..."

The iter-29 param-count pin
(`tests/test_engine/test_bitnet_param_count_pin.py`) covers the
8,512 / 69 / 8,581 numbers, but THREE bundle-level integrity claims
are NOT pinned:

  - 19 KB file size — tied to the "Pi Zero 2 W can ship the whole
    weights bundle on a $15 board" edge claim.
  - `cfadb4f6` short-form bundle_id prefix — tied to every
    audit-replay claim ("any auditor can re-verify in <1 ms with
    the 19 KB weights bundle").
  - ~48% structural sparsity (weights collapsed to 0) — tied to the
    iter-72-era demo claim "structured sparsity from quantization-
    aware training (STE) is what lets the 8,581-parameter model fit
    in 19 KB" (demo.html L1072).

Without a pin, a future weight rotation could:
  - Bloat the file >> 19 KB (silently breaks the "Pi Zero 2 W ships
    the whole bundle" hardware claim)
  - Match the displayed `cfadb4f6` short prefix only by accident
  - Drop sparsity well below 40% (silently invalidates the demo's
    "structured sparsity" rhetoric)

This pin enforces all three.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_BUNDLE = _REPO_ROOT / "engine" / "bitnet_weights.json"
_DEMO = _REPO_ROOT / "docs" / "demo.html"
_JUDGES = _REPO_ROOT / "JUDGES.md"


# Pinned values (iter-130).
_EXPECTED_SHORT_BUNDLE_ID = "cfadb4f6"
_EXPECTED_BUNDLE_ID_TAIL = "0b3f"  # Last 4 chars from the demo span "cfadb4f6…0b3f"
_EXPECTED_FILE_SIZE_BYTES = 20391  # 19.9 KB raw on-disk size as of iter-130
# Tolerance: a deliberate weights rotation may shift bytes ±1024
# without invalidating the "19 KB" rhetorical claim. A jump beyond
# this band IS load-bearing — the Pi Zero 2 W edge claim depends on
# the bundle staying small.
_FILE_SIZE_TOLERANCE_BYTES = 2048
_SPARSITY_FLOOR = 0.40  # iter-72 lower bound on zero-weight fraction


def _payload() -> dict:
    return json.loads(_BUNDLE.read_text())


def _canonical_bundle_id() -> str:
    """SHA-256 over the canonical-JSON encoding of the four weight matrices.

    Mirrors `engine.bitnet_classifier._bundle_id` so this pin tests
    bundle identity end-to-end without importing the engine module.
    """
    payload = _payload()
    canonical = json.dumps(
        {
            "hidden_w": payload["hidden_w"],
            "hidden_b": payload["hidden_b"],
            "output_w": payload["output_w"],
            "output_b": payload["output_b"],
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def test_bundle_id_short_prefix_matches_pinned():
    """The first 8 chars of the live bundle_id must match the
    `cfadb4f6` claim displayed in demo + JUDGES + audit-replay
    rhetoric. A weight rotation that breaks this prefix MUST update
    the demo + JUDGES displayed values in the same commit.
    """
    bundle_id = _canonical_bundle_id()
    actual = bundle_id[:8]
    assert actual == _EXPECTED_SHORT_BUNDLE_ID, (
        f"bundle_id short prefix drifted: live={actual!r}, "
        f"pinned={_EXPECTED_SHORT_BUNDLE_ID!r}. The demo + JUDGES "
        f"display 'cfadb4f6…0b3f' as the audit-replay anchor; "
        f"a weight rotation needs to update those display values + "
        f"this constant in the same commit."
    )
    # Tail anchor — demo span shows `cfadb4f6…0b3f`, must still match.
    actual_tail = bundle_id[-4:]
    assert actual_tail == _EXPECTED_BUNDLE_ID_TAIL, (
        f"bundle_id short tail drifted: live={actual_tail!r}, "
        f"pinned={_EXPECTED_BUNDLE_ID_TAIL!r}. "
        f"docs/demo.html L1047 displays the …0b3f tail."
    )


def test_demo_displays_pinned_bundle_id():
    """Cross-check: demo.html actually contains the pinned short form."""
    text = _DEMO.read_text()
    short_form = f"{_EXPECTED_SHORT_BUNDLE_ID}…{_EXPECTED_BUNDLE_ID_TAIL}"
    assert short_form in text, (
        f"docs/demo.html missing the pinned bundle_id short form "
        f"{short_form!r}. The L1047 chip + the audit-replay "
        f"rhetoric reference this value."
    )


def test_judges_displays_pinned_bundle_id():
    """Cross-check: JUDGES.md actually contains the pinned short form."""
    text = _JUDGES.read_text()
    # JUDGES uses a slightly shorter form: `cfadb4f6…` (just the prefix).
    assert _EXPECTED_SHORT_BUNDLE_ID in text, (
        f"JUDGES.md missing the pinned bundle_id short prefix "
        f"{_EXPECTED_SHORT_BUNDLE_ID!r}. The runnable-claims row "
        f"100 references this value."
    )


def test_bundle_file_size_within_19kb_band():
    """File must remain within ~19 KB band (Pi Zero 2 W edge claim).

    A ballooned bundle (e.g. 32 KB after a 128-hidden retrain or
    accidental float weights) silently invalidates the "$15 Pi Zero
    2 W ships the whole bundle" rhetoric in the edge-offline section
    of the demo.

    Tolerance ± 2 KB allows for legitimate weight rotations that
    keep ~48% sparsity. A jump beyond this band IS the load-bearing
    signal that the edge claim needs updating.
    """
    actual_bytes = _BUNDLE.stat().st_size
    delta = abs(actual_bytes - _EXPECTED_FILE_SIZE_BYTES)
    assert delta <= _FILE_SIZE_TOLERANCE_BYTES, (
        f"bitnet_weights.json size drifted: live={actual_bytes} bytes, "
        f"pinned={_EXPECTED_FILE_SIZE_BYTES} bytes, "
        f"tolerance={_FILE_SIZE_TOLERANCE_BYTES} bytes (delta={delta}). "
        f"The '19 KB / Pi Zero 2 W edge claim' rhetoric breaks at "
        f">= 32 KB; update the demo + edge_pi_offline.md hardware "
        f"profile or revert the weight change."
    )
    # Also: must still be under 32 KB hard limit (the edge-claim ceiling).
    assert actual_bytes < 32 * 1024, (
        f"bitnet_weights.json grew past the 32 KB hard ceiling "
        f"(live={actual_bytes} bytes). The 'fits on a $15 Pi Zero "
        f"2 W' demo claim is now false."
    )


def test_ternary_weight_sparsity_floor():
    """At least 40% of ternary weights must be zero.

    Demo L1072 claim: "~48% of weights collapse to 0 — structured
    sparsity from quantization-aware training (STE) is what lets
    the 8,581-parameter model fit in 19 KB."

    A weight rotation that drops sparsity below 40% (e.g. dense
    floating-point reset that got accidentally clamped to {-1, 0, 1})
    silently invalidates this rhetoric without anyone noticing
    because the param count + bundle hash both look correct.
    """
    payload = _payload()
    all_weights: list[int] = []
    for row in payload["hidden_w"]:
        all_weights.extend(row)
    for row in payload["output_w"]:
        all_weights.extend(row)
    zeros = sum(1 for w in all_weights if w == 0)
    sparsity = zeros / len(all_weights)
    assert sparsity >= _SPARSITY_FLOOR, (
        f"Ternary weight sparsity dropped below the iter-72 floor: "
        f"live={sparsity:.1%}, floor={_SPARSITY_FLOOR:.0%}. "
        f"The demo's 'structured sparsity ~48%' claim becomes false. "
        f"A dense weight rotation needs to update the demo + "
        f"why_bitnet_b158.md sparsity rhetoric in the same commit."
    )


def test_bundle_keys_canonical():
    """The bundle JSON must have exactly the 4 weight keys + meta.

    Adding a stray key silently changes the canonical-JSON encoding
    AND the bundle_id, but the param-count pin would not catch the
    extra key (it only counts weights). This pin enforces shape.
    """
    payload = _payload()
    expected_keys = {"_meta", "hidden_b", "hidden_w", "output_b", "output_w"}
    actual_keys = set(payload.keys())
    extra = actual_keys - expected_keys
    missing = expected_keys - actual_keys
    assert not extra and not missing, (
        f"engine/bitnet_weights.json key set drifted: "
        f"extra={sorted(extra)}, missing={sorted(missing)}. "
        f"The 4 weight keys + _meta are the canonical shape; "
        f"changing this changes the bundle_id and breaks every "
        f"prior audit-replay claim."
    )


def test_bundle_meta_records_provenance():
    """`_meta` must carry at least one provenance field so an auditor
    can correlate a bundle_id back to its training context.

    The live bundle ships rich provenance: `framework_version`,
    `paper` (BitNet b1.58 arXiv ref), `schema`, `trained_with`,
    `bundle_id` (self-reference). Any of these counts — what
    matters is the provenance trail exists.
    """
    payload = _payload()
    meta = payload.get("_meta")
    assert isinstance(meta, dict), (
        "engine/bitnet_weights.json must carry a `_meta` dict so a "
        "future auditor can correlate a bundle_id back to a STARGA "
        "build context. Missing or non-dict meta found."
    )
    # Accept the iter-72-shipped schema OR an iteration-style schema.
    # Any of these provenance fields counts; the test just ensures
    # the trail isn't empty.
    provenance_keys = (
        "iteration", "version", "build_iter", "iter", "build", "ship_iter",
        "framework_version", "paper", "schema", "trained_with", "bundle_id",
    )
    has_provenance = any(k in meta for k in provenance_keys)
    assert has_provenance, (
        f"engine/bitnet_weights.json `_meta` must carry at least one "
        f"provenance field from {provenance_keys}. Live keys: "
        f"{sorted(meta.keys())}."
    )


def test_demo_cites_this_pin_file():
    """Demo must cite this pin file near the trained-model section.

    iter-131 (T2 surfacing) added a purple-rule callout below the
    ternary-weight distribution paragraph that names all 8
    integrity invariants and references this pin file. Same pattern
    as iter-110/115/121/126's "demo cites pin file" cross-checks:
    test layer and user-facing surface stay in sync.
    """
    text = _DEMO.read_text()
    pin_filename = "test_bitnet_bundle_integrity_pin.py"
    assert pin_filename in text, (
        f"docs/demo.html must cite "
        f"`tests/test_engine/{pin_filename}` near the trained-model "
        f"section so judges can trace the bundle_id + 19 KB + sparsity "
        f"claims to their enforcing pin file."
    )
    # Anchor: callout must appear with a recognizable phrase so a copy
    # edit can't strip the rationale and leave just the filename.
    locality_anchors = (
        "Bundle integrity",
        "eight invariants",
        "bundle_id first-8",
    )
    has_anchor = any(a in text for a in locality_anchors)
    assert has_anchor, (
        f"Demo's pin-file citation must appear with a locality anchor "
        f"such as 'Bundle integrity' / 'eight invariants' / "
        f"'bundle_id first-8'. None found near the trained-model "
        f"section — copy edit may have stripped the rationale."
    )


def test_bundle_meta_self_referenced_bundle_id_is_consistent():
    """If `_meta.bundle_id` is present, it must match the live SHA-256
    of the canonical-form weight matrices.

    A self-referenced bundle_id that doesn't match the live hash is
    a release-blocking integrity failure — the bundle is claiming
    one identity in metadata while computing another at runtime.
    The iter-29 / iter-72 / iter-117 bundle conventions all carry
    a `bundle_id` field; this pin enforces that it stays in sync.
    """
    meta = _payload().get("_meta", {})
    if not isinstance(meta, dict) or "bundle_id" not in meta:
        # Optional field — only test when present.
        return
    declared = meta["bundle_id"]
    actual = _canonical_bundle_id()
    assert declared == actual, (
        f"engine/bitnet_weights.json `_meta.bundle_id` is INCONSISTENT "
        f"with the live SHA-256: declared={declared!r}, "
        f"live={actual!r}. The bundle is claiming one identity in "
        f"metadata while computing another at runtime — this would "
        f"silently break audit-replay. Recompute `_meta.bundle_id` "
        f"to match `engine.bitnet_classifier._bundle_id(payload)`."
    )
