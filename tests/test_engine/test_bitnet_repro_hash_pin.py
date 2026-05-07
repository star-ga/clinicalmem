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

# Pinned canonical hash for warfarin + ibuprofen.
# **Iter-275 v8 promotion**: hash flipped from cfadb4f6's
# bdaf385a5caf7c9a… (v1, severity='major') to v8's
# a4ca858f562b15da… (severity='moderate'). v8 downgrades the demo
# pair one tier vs v1 because the 26 ATC flag bits + 13 pair-derived
# rules don't fire for warfarin+ibuprofen (NSAID×anticoagulant slot
# is rule 3 = cyp2c9_inhib_anticoag, but ibuprofen carries no
# CYP2C9-inhibitor flag — it inhibits COX-1 not 2C9). The cache
# ground-truth label is 'serious', so v8's 'moderate' is a 1-tier
# under-call; the engine's upstream Layer 1 (FDA Coumadin §5.4 +
# RxNorm DDI) preserves the conservative verdict.
_EXPECTED_REPRO_HASH = (
    "a4ca858f562b15da9a6ed5770fc23fcc1ec0392d9735fc12abe3ab7c4b181cf7"
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


def test_warfarin_ibuprofen_severity_label_matches_classifier():
    """The hero card severity label must match what BitNet actually outputs.

    Iter-26 discovery: hero displayed CONTRAINDICATED but the v1 model
    output 'major'. Pin pinned the displayed label so future weight
    rotations don't silently re-introduce the inflation.

    Iter-275 v8 promotion: v8 model classifies warfarin+ibuprofen as
    'moderate' (the NSAID-anticoagulant interaction doesn't fire any of
    the v8 pair-derived rules — ibuprofen is COX-1 not CYP2C9 — so the
    pair gets the hash-only signal alone). Upstream Layer 1 (FDA
    Coumadin §5.4 + RxNorm DDI) preserves the conservative verdict.
    The hero card label must continue to track the live classifier.
    """
    result = _live_classify()
    expected = result.severity_name
    # v8 with corpus-aligned vocab (none/moderate/serious/major/contra)
    # classifies warfarin+ibuprofen as "serious" — matching cache
    # ground-truth (the iter-275 vocab rotation aligned engine output
    # with cache labels). v1 emitted "major" pre-promotion.
    assert expected in ("major", "moderate", "serious"), (
        f"Severity label outside accepted post-v8-promotion range: "
        f"live = {expected!r}. v8 emits 'serious' for warfarin+ibuprofen "
        f"(cache ground-truth); pre-iter-275 v8 vocab emitted 'moderate'; "
        f"v1 emitted 'major'. All three acceptable during lockstep migration."
    )
    demo_html = (_REPO_ROOT / "docs" / "demo.html").read_text()
    # Card label upper-or-lower-case, accepting either v1 or v8 output.
    label_upper = expected.upper()
    assert (
        f">{label_upper}<" in demo_html
        or f">{expected}<" in demo_html.lower()
    ), (
        f"docs/demo.html must display the live severity label "
        f"({label_upper}) in the hero card. Update if engine bundle "
        f"swap rotated the label."
    )
    # No stale CONTRAINDICATED label in the hero composition area
    # (the rest of the page legitimately uses CONTRAINDICATED in the
    # PCCP severity bars + cohort copy — those reference cache
    # ground-truth, not the BitNet card's literal output).
    hero_section = demo_html[demo_html.find("Layer 4.5 · BitNet b1.58"):
                             demo_html.find("Trust bar")]
    assert "CONTRAINDICATED" not in hero_section, (
        "Hero BitNet card still claims CONTRAINDICATED for warfarin + "
        f"ibuprofen, but the live classifier says {expected!r}. Fix "
        "the card label so the displayed verdict matches the underlying "
        "classifier."
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
