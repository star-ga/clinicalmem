"""Pin: the iter-275 v1 backup bundle is independently loadable + replayable.

**Iter-286 T1 round-60 — 22nd cross-pin family.**

The iter-275 v8 promotion archived the pre-promotion v1 baseline at
`engine/bitnet_weights.v1.cfadb4f6.bak.json`. The README + JUDGES both
claim "any auditor can replay decisions made before the iter-275
promotion under the prior bundle without proprietary toolchain". The
iter-281 cross-file consistency pin verifies the file EXISTS on disk;
this pin verifies it's actually FUNCTIONAL — load + classify + repro_hash.

Without this pin, the v1 backup could become corrupt (truncated, JSON
malformed, schema field missing, weight matrix shape drift) and the
"decade-replay claim" would silently break. The cross-file consistency
pin would still pass because the file is present; only an actual
forward-pass test catches functional corruption.

Pinned invariants
=================
- v1 backup loads via the same `engine.bitnet_classifier.load_weights`
  the live engine uses (no separate compatibility shim required).
- Loaded weights report `schema = bitnet_classifier_v1`, dims =
  (in=128, hidden=64, out=5), bundle_id starts with `cfadb4f6`.
- warfarin + ibuprofen under v1 produces severity = `major` (the
  iter-26 historical anchor pre-iter-275) AND repro_hash starts with
  `bdaf385a` (the iter-26 pinned 16-char short form).
- v1 backup repro_hash for warfarin + ibuprofen DIFFERS from the
  live v8 engine's repro_hash for the same pair (catches the case
  where someone accidentally copies the live v8 bundle over the v1
  backup — bundle_id would then match v8 instead of cfadb4f6).
- v1 backup is byte-stable across two consecutive loads (no
  module-state pollution between calls).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_V1_BACKUP = _REPO_ROOT / "engine" / "bitnet_weights.v1.cfadb4f6.bak.json"

# iter-26 pinned warfarin + ibuprofen result under v1 (cfadb4f6).
_V1_WARFARIN_IBUPROFEN_REPRO_PREFIX = "bdaf385a"
_V1_WARFARIN_IBUPROFEN_SEVERITY = "major"


@pytest.fixture(scope="module")
def v1_weights():
    sys.path.insert(0, str(_REPO_ROOT))
    from engine.bitnet_classifier import load_weights  # noqa: PLC0415
    return load_weights(_V1_BACKUP)


def test_v1_backup_loads_with_engine_loader(v1_weights):
    """v1 backup MUST load via the live engine's `load_weights` (not a
    parallel compatibility path). Catches the regression class where
    schema-aware loader logic accidentally drops v1 support."""
    assert v1_weights is not None


def test_v1_backup_schema_dims_match_documented(v1_weights):
    """Schema + dims match the iter-275 documentation: schema=v1,
    in_features=128, hidden_features=64, out_features=5, bundle_id
    starts with `cfadb4f6` (the historical short prefix cited in
    JUDGES + README + bitnet_training.md)."""
    assert v1_weights.schema == "bitnet_classifier_v1"
    assert v1_weights.in_features == 128
    assert v1_weights.hidden_features == 64
    assert v1_weights.out_features == 5
    assert v1_weights.bundle_id.startswith("cfadb4f6"), (
        f"v1 backup bundle_id {v1_weights.bundle_id[:16]!r} doesn't "
        f"start with 'cfadb4f6' — wrong bundle archived."
    )


def test_v1_backup_replays_warfarin_ibuprofen_with_iter26_anchor(v1_weights):
    """The iter-26 historical anchor: warfarin + ibuprofen under v1 →
    severity='major' + repro_hash starts with `bdaf385a`. This was
    the live engine output for ~250 iters before the iter-275 v8
    promotion rotated it; an FDA reviewer replaying any pre-iter-275
    audit row MUST get the same answer."""
    sys.path.insert(0, str(_REPO_ROOT))
    from engine.bitnet_classifier import classify  # noqa: PLC0415

    r = classify("warfarin", "ibuprofen", v1_weights)
    assert r.severity_name == _V1_WARFARIN_IBUPROFEN_SEVERITY, (
        f"v1 backup classifies warfarin+ibuprofen as {r.severity_name!r}, "
        f"iter-26 anchor was {_V1_WARFARIN_IBUPROFEN_SEVERITY!r}. v1 "
        f"backup is corrupted — pre-iter-275 audit replay broken."
    )
    assert r.repro_hash.startswith(_V1_WARFARIN_IBUPROFEN_REPRO_PREFIX), (
        f"v1 backup repro_hash {r.repro_hash[:16]!r} doesn't start "
        f"with iter-26 anchor {_V1_WARFARIN_IBUPROFEN_REPRO_PREFIX!r}."
    )


def test_v1_backup_repro_hash_differs_from_live_v8():
    """v1 backup MUST produce a DIFFERENT repro_hash than the live v8
    engine for the same pair. Catches the failure mode where someone
    accidentally copies the live `engine/bitnet_weights.json` over the
    v1 backup file (which would still pass the `_meta.bundle_id starts
    with cfadb4f6` check if the copy script edited the _meta block but
    left the weight matrices as v8).
    """
    sys.path.insert(0, str(_REPO_ROOT))
    from engine.bitnet_classifier import classify, load_weights  # noqa: PLC0415

    v1 = load_weights(_V1_BACKUP)
    v8 = load_weights()  # live engine bundle
    v1_result = classify("warfarin", "ibuprofen", v1)
    v8_result = classify("warfarin", "ibuprofen", v8)

    assert v1_result.repro_hash != v8_result.repro_hash, (
        f"v1 backup and live v8 produce IDENTICAL repro_hash for "
        f"warfarin+ibuprofen ({v1_result.repro_hash[:16]!r}). The v1 "
        f"backup was likely overwritten with the v8 bundle — "
        f"audit-chain reconstruction broken. Restore from "
        f"git history."
    )
    assert v1_result.severity_name != v8_result.severity_name or (
        v1_result.repro_hash != v8_result.repro_hash
    ), "v1 and v8 must differ on at least one observable output"


def test_v1_backup_forward_pass_deterministic(v1_weights):
    """Two consecutive forward passes on the v1 backup MUST produce
    identical output. Catches module-state pollution between calls
    (e.g., the iter-279 `_LOAD_CONTEXT_LOGGED` latch accidentally
    affecting determinism)."""
    sys.path.insert(0, str(_REPO_ROOT))
    from engine.bitnet_classifier import classify  # noqa: PLC0415

    r1 = classify("warfarin", "ibuprofen", v1_weights)
    r2 = classify("warfarin", "ibuprofen", v1_weights)
    assert r1.repro_hash == r2.repro_hash
    assert r1.severity_name == r2.severity_name
    assert list(r1.logits_q16) == list(r2.logits_q16)
    assert r1.feature_hash == r2.feature_hash


def test_v1_backup_bundle_id_self_reference_matches_canonical_sha(v1_weights):
    """The v1 backup's `_meta.bundle_id` self-reference MUST equal the
    canonical SHA over its own weight matrices. Catches the failure
    mode where _meta is hand-edited (e.g., bundle_id rewritten to
    cfadb4f6 even though weights changed). Mirror of iter-281 cross-
    file consistency pin applied to the v1 archive specifically.
    """
    import hashlib
    import json

    payload = json.loads(_V1_BACKUP.read_text())
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
    expected_sha = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    declared = payload.get("_meta", {}).get("bundle_id", "")
    assert declared == expected_sha, (
        f"v1 backup _meta.bundle_id {declared[:16]!r} doesn't match "
        f"canonical SHA over weight matrices {expected_sha[:16]!r}. "
        f"_meta was hand-edited or weights were swapped without "
        f"rotating bundle_id."
    )
    assert v1_weights.bundle_id == expected_sha
