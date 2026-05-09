"""Pin: bundle_id consistency across pin files, docs, and the live engine.

**Iter-281 T1 round-59 — 20th cross-pin family.**

Iter-275 v8 promotion cascade (cfadb4f6 → 1f0f8859) touched 26 files;
iter-278 caught 6 stale v1 references on judge-visible surfaces that
the iter-275 commit missed. This pin family forward-protects against
that drift class: every "live bundle_id" reference across pin files,
docs, and demo HTML MUST match the live engine bundle.

The iter-275 cascade made it clear that bundle_id rotation needs to
be **mechanically traceable across every claim site**. This pin lists
the canonical sites + the historical-ref allowlist (cfadb4f6 = v1,
592ee51e = v6, 1ff61a6a = v5, eea0e637 = v3 — kept on disk for FDA
SaMD audit-trail rigor).

Drift modes this pin catches
============================
1. **Half-completed bundle rotation** — pin file or doc says
   `1f0f8859` but the engine ships `cfadb4f6` (or vice versa).
2. **Typo in pin constant** — `1f0f88591c…` mistyped as `1f0f88591d…`.
3. **Demo prose stale post-rotation** — README/JUDGES/demo cite an
   old short prefix without lockstep update.

Allowlisted historical bundles (do NOT need to match live engine):
- `cfadb4f6` — v1 baseline (kept at engine/bitnet_weights.v1.cfadb4f6.bak.json
  for audit-chain reconstruction)
- `592ee51e` — v6 staged (40/41, kept on disk)
- `1ff61a6a` — v5 staged (31/38, kept on disk)
- `eea0e637` — v3 historical (29/31, kept on disk)
"""
from __future__ import annotations

import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# Historical bundles preserved on disk for audit-chain reconstruction.
# These short prefixes may appear in user-facing docs as historical
# context but MUST NOT be the engine's live bundle_id.
_HISTORICAL_BUNDLE_PREFIXES = frozenset({
    "cfadb4f6",   # v1 baseline (engine/bitnet_weights.v1.cfadb4f6.bak.json)
    "592ee51e",   # v6 staged (40/41 + 0 FP, h=128, retrain_runpod/bitnet_weights_v6_h128.json)
    "1ff61a6a",   # v5 staged (31/38 + 0 FP, h=128, retrain_runpod/bitnet_weights_v5_h128.json)
    "eea0e637",   # v3 historical (29/31 + 1 FP, h=64, retrain_runpod/bitnet_weights_v3_full.json)
    "5f7ed5f6",   # iter-421 Path B tier-2 specialist (engine/bitnet_weights_b_specialist.json)
                  # — ensemble partner alongside live A bundle 1f0f8859, not a replacement
})

# Files that pin-or-cite the LIVE bundle_id (post-iter-275 = 1f0f8859).
# Each entry MUST contain the live short prefix at least once.
_FILES_REQUIRING_LIVE_BUNDLE_ID = (
    _REPO_ROOT / "engine" / "bitnet_weights.json",
    _REPO_ROOT / "tests" / "test_engine" / "test_bitnet_bundle_integrity_pin.py",
    _REPO_ROOT / "tests" / "test_engine" / "test_path_a_v8_live_recall_pin.py",
    _REPO_ROOT / "tests" / "test_engine" / "test_path_a_v8_q16_determinism_pin.py",
    _REPO_ROOT / "tests" / "test_engine" / "test_v8_bundle_integrity_pin.py",
    _REPO_ROOT / "tests" / "test_engine" / "test_demo_retrain_callout_pin.py",
    _REPO_ROOT / "docs" / "demo.html",
    _REPO_ROOT / "JUDGES.md",
    _REPO_ROOT / "README.md",
)


def _live_bundle_id() -> str:
    """Return the engine's live bundle_id (full 64-char SHA-256)."""
    import sys
    sys.path.insert(0, str(_REPO_ROOT))
    from engine.bitnet_classifier import load_weights  # noqa: PLC0415
    return load_weights().bundle_id


def test_live_bundle_id_appears_in_every_required_file():
    """Every load-bearing pin file or doc that cites a bundle_id MUST
    contain the live engine bundle's short prefix (first 8 hex chars).
    """
    live_prefix = _live_bundle_id()[:8]
    missing = []
    for path in _FILES_REQUIRING_LIVE_BUNDLE_ID:
        if not path.exists():
            missing.append((path, "FILE_NOT_FOUND"))
            continue
        text = path.read_text(encoding="utf-8")
        if live_prefix not in text:
            missing.append((path, "live_prefix_absent"))
    assert not missing, (
        f"Live bundle_id prefix {live_prefix!r} missing from "
        f"{len(missing)} required files. The cross-file consistency "
        f"discipline is broken — a bundle rotation may have left "
        f"stale references. Sites: "
        f"{[(str(p.relative_to(_REPO_ROOT)), reason) for p, reason in missing]}"
    )


def test_no_orphan_eight_hex_bundle_prefixes_in_pin_files():
    """No pin file may reference an 8-hex bundle short prefix that is
    neither the live engine bundle nor a known historical bundle.

    Catches typos (e.g., 1f0f88591d instead of 1f0f88591c) and
    abandoned post-rotation drift (e.g., a stale 1f0f88592c left from
    a botched cherry-pick).
    """
    live_prefix = _live_bundle_id()[:8]
    allowed = frozenset({live_prefix}) | _HISTORICAL_BUNDLE_PREFIXES

    # Match `bundle_id` literal followed by 8-hex prefix patterns the
    # codebase actually uses: backtick-wrapped, code-tag-wrapped,
    # bundle_id NN_HEX, etc. Conservative pattern: an 8-hex prefix
    # followed by `…`/`...` or sandwiched between known bundle-id
    # delimiters. Avoids false-positives from arbitrary 8-hex strings
    # like SHA-256 prefixes of OTHER artifacts.
    # Require a non-hex char (or string start) BEFORE the 8-hex prefix
    # so partial-substring matches of longer bundle_ids (e.g. matching
    # an inner 8-hex window inside a 9+ hex bundle_id) don't
    # false-positive.
    # AFTER the 8-hex prefix: ellipsis / triple-dot / backtick / quote.
    bundle_prefix_re = re.compile(
        r"(?:^|[^0-9a-f])([0-9a-f]{8})(?:…|\.\.\.|`|\"|')",
        re.IGNORECASE | re.MULTILINE,
    )

    pin_dir = _REPO_ROOT / "tests" / "test_engine"
    orphans: list[tuple[str, str]] = []
    for pyfile in pin_dir.glob("test_*pin*.py"):
        text = pyfile.read_text(encoding="utf-8")
        for m in bundle_prefix_re.finditer(text):
            prefix = m.group(1).lower()
            # Only flag prefixes that look like bundle_id (8 hex with at
            # least 2 of each digit class to avoid arbitrary 8-hex hashes
            # of other artifacts).
            if prefix in allowed:
                continue
            # Heuristic — only flag if it co-occurs with bundle_id
            # tightly. Iter-286 narrowed the window from 200 → 60
            # chars to eliminate false positives where a repro_hash
            # prefix (e.g., the iter-26 warfarin+ibuprofen anchor
            # `bdaf385a`) appears in a pin file whose docstring
            # mentions `bundle_id` elsewhere. The 60-char window
            # captures legitimate `bundle_id 1f0f8859…` adjacency
            # without false-matching distant docstring co-occurrences.
            window_start = max(0, m.start() - 60)
            window = text[window_start:m.start() + 20]
            if "bundle_id" in window.lower():
                orphans.append((str(pyfile.relative_to(_REPO_ROOT)), prefix))

    assert not orphans, (
        f"Found {len(orphans)} orphan 8-hex bundle prefixes in pin "
        f"files (neither the live engine bundle {live_prefix!r} nor a "
        f"known historical bundle {sorted(_HISTORICAL_BUNDLE_PREFIXES)}). "
        f"Likely a typo or post-rotation drift: {orphans[:5]}"
    )


def test_engine_bitnet_weights_self_referenced_bundle_id_matches_live():
    """The `_meta.bundle_id` field embedded in `engine/bitnet_weights.json`
    MUST equal the live engine bundle_id. Catches manual edits to the
    bundle's _meta block that are NOT also reflected in the canonical
    weight-matrix SHA-256."""
    import json
    payload = json.loads(
        (_REPO_ROOT / "engine" / "bitnet_weights.json").read_text()
    )
    self_ref = payload.get("_meta", {}).get("bundle_id", "")
    live = _live_bundle_id()
    assert self_ref == live, (
        f"engine/bitnet_weights.json _meta.bundle_id self-reference "
        f"{self_ref[:16]!r} does NOT match live engine bundle_id "
        f"{live[:16]!r}. The _meta block was hand-edited without "
        f"rotating the canonical SHA — bundle integrity broken."
    )


def test_v1_backup_bundle_preserved_for_audit_chain():
    """The pre-promotion v1 bundle (cfadb4f6) MUST stay on disk at
    engine/bitnet_weights.v1.cfadb4f6.bak.json for full audit-chain
    reconstruction. Without it, decisions made before iter-275 cannot
    be replayed under the prior bundle and the FDA SaMD claim breaks.
    """
    backup = _REPO_ROOT / "engine" / "bitnet_weights.v1.cfadb4f6.bak.json"
    assert backup.exists(), (
        f"v1 baseline backup missing: {backup.relative_to(_REPO_ROOT)}. "
        f"The iter-275 v8 promotion preserved the v1 bundle for audit-"
        f"chain reconstruction; deleting it breaks decade-replay claims "
        f"on every pre-iter-275 audit row."
    )
    import json
    payload = json.loads(backup.read_text())
    backup_bundle_id = payload.get("_meta", {}).get("bundle_id", "")
    assert backup_bundle_id.startswith("cfadb4f6"), (
        f"v1 backup bundle_id {backup_bundle_id[:16]!r} doesn't start "
        f"with cfadb4f6 — wrong bundle archived as v1 backup."
    )


def test_historical_bundles_documented_in_judges():
    """JUDGES.md must document every historical bundle preserved on
    disk (so an FDA reviewer can follow the architectural-progression
    audit trail without grep)."""
    judges = (_REPO_ROOT / "JUDGES.md").read_text()
    missing = []
    for prefix in _HISTORICAL_BUNDLE_PREFIXES:
        if prefix not in judges:
            missing.append(prefix)
    assert not missing, (
        f"JUDGES.md missing historical bundle short prefixes: "
        f"{missing}. The architectural-progression narrative depends "
        f"on JUDGES citing every preserved bundle."
    )
