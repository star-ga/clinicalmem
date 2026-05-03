"""Pin the BitNet repro_hash for the warfarin + ibuprofen demo case.

The hero stamp on the dashboard displays a 16-char `repro_hash`
prefix specifically for the {warfarin, ibuprofen} drug pair, the
canonical demo example. The "bit-identical across CPU/GPU/NPU" +
"FDA-replayable decades later" claims are only credible if the
displayed hash is exactly what the on-disk classifier produces.

Two checks:

  1. The live classifier's output for `classify("warfarin",
     "ibuprofen", weights)` matches the pinned full SHA-256.
  2. The 16-char short form displayed in `docs/demo.html` matches
     the live full hash's prefix.

Drift triggers (any of which break this test):
  - BitNet weights bundle rotated.
  - `_encode_drug_token` / `_q16_dot_ternary` / `_q16_relu`
    implementation changed.
  - `engine/bitnet_weights.json` modified.
  - Drug-pair canonicalization (lex sort) changed.
"""
from __future__ import annotations

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# Pinned canonical hash for warfarin + ibuprofen at iteration 26.
# When this changes, the BitNet bundle was rotated or the inference
# code changed; bump this constant + the doc reference in
# docs/demo.html in the same commit.
_EXPECTED_REPRO_HASH = (
    "bdaf385a5caf7c9a77ad71ac1bf8d821da9eba6ca7d350f5a6c8e4307a9d99d5"
)


def _live_classify():
    import sys
    sys.path.insert(0, str(_REPO_ROOT))
    from engine.bitnet_classifier import classify, load_weights

    weights = load_weights()
    return classify("warfarin", "ibuprofen", weights)


def test_warfarin_ibuprofen_repro_hash_matches_pin():
    result = _live_classify()
    assert result.repro_hash == _EXPECTED_REPRO_HASH, (
        f"BitNet repro_hash drift detected.\n"
        f"  Pinned: {_EXPECTED_REPRO_HASH}\n"
        f"  Live:   {result.repro_hash}\n"
        f"Update _EXPECTED_REPRO_HASH in this test AND the displayed "
        f"short form in docs/demo.html (line near 'repro_hash<br>') in "
        f"the same commit."
    )


def test_dashboard_displays_pinned_short_form():
    """Dashboard's 16-char repro_hash short form matches the live hash."""
    short = _EXPECTED_REPRO_HASH[:16]
    demo_html = (_REPO_ROOT / "docs" / "demo.html").read_text()
    assert short in demo_html, (
        f"docs/demo.html is missing the pinned repro_hash short form {short!r}"
    )


def test_no_stale_short_repro_hash_remains():
    """Old short forms must be removed after rotation."""
    historical = (
        "15ccdd3f23d4e04b",  # pre-iteration-26 short form
    )
    demo_html = (_REPO_ROOT / "docs" / "demo.html").read_text()
    for old in historical:
        assert old not in demo_html, (
            f"Stale repro_hash short form {old!r} still in docs/demo.html — "
            f"replace with the current pinned value."
        )
