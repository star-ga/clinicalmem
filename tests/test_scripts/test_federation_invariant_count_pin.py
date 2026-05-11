"""Pin the federation invariant counts.

Two interlocking claims appear across docs/demo.html, JUDGES.md,
docs/architecture.md, and the federation mock demo:

  Spec  : `JointMemoryFederation.flow.mind` declares 21 typed
          runtime invariants (the full federation contract).
  Demo  : `scripts/federation_mock_demo.py` exercises ALL 21
          end-to-end. Iter-2026-05-11 wired the 5 X25519 sealing
          invariants (`SealedEnvelope` + `_x25519_seal` /
          `_x25519_open` round-trip in-process) so the mock demo
          mirrors the wire-format cryptographic envelope of the
          v4 federation HTTP wire transport (mind-mem `main`
          16a3e25, pending v4.0.x PyPI tag).

This test pins both numbers so any future contract edit that adds
or removes invariants — or any test/demo change that rebases the
exercised count — fails the gate until the docs + the contract +
the demo are updated together.
"""
from __future__ import annotations

import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_FLOW = _REPO_ROOT / "flows" / "JointMemoryFederation.flow.mind"
_DEMO_HTML = _REPO_ROOT / "docs" / "demo.html"
_JUDGES = _REPO_ROOT / "JUDGES.md"
_ARCH = _REPO_ROOT / "docs" / "architecture.md"

# Iter-135 (T1 round 27): the same drift class iter-132 caught for
# the v3.9 wire-transport pin lived here too. Three additional
# user-facing docs lied about the flow's invariant count (claiming
# "16 invariants" when the live count is 21):
#
#   * docs/clinical_validation.md   (multi-site validation summary)
#   * docs/fda_q_sub_draft.md       (regulatory submission draft)
#   * docs/federated_memory.md      (federation architecture status)
#
# These were OUTSIDE the iter-22 pin scope (demo / JUDGES / arch),
# so the same conflation survived undetected. iter-135 expands the
# scope to the full user-facing federation surface.
_USER_FACING_FED_DOCS = (
    _REPO_ROOT / "docs" / "demo.html",
    _REPO_ROOT / "JUDGES.md",
    _REPO_ROOT / "docs" / "architecture.md",
    _REPO_ROOT / "docs" / "clinical_validation.md",
    _REPO_ROOT / "docs" / "fda_q_sub_draft.md",
    _REPO_ROOT / "docs" / "federated_memory.md",
)

# Pinned constants. When an `invariant` is added to or removed from
# the flow contract, update the matching number AND every doc that
# names it in the same commit.
_EXPECTED_SPEC_INVARIANT_COUNT = 21    # full typed contract
_EXPECTED_DEMO_INVARIANT_COUNT = 21    # exercised by federation_mock_demo (X25519 seal/open landed 2026-05-11)


def _count_invariants_in_flow() -> int:
    """Count `invariant <expr>` declarations in the .flow.mind contract."""
    text = _FLOW.read_text()
    return sum(
        1 for line in text.splitlines() if re.match(r"\s*invariant\s+", line)
    )


def _count_demo_invariants() -> int:
    """Count entries in the demo's INVARIANT_DESCRIPTIONS dict."""
    import sys
    sys.path.insert(0, str(_REPO_ROOT))
    from scripts.federation_mock_demo import INVARIANT_DESCRIPTIONS  # noqa

    return len(INVARIANT_DESCRIPTIONS)


def test_flow_contract_declares_pinned_invariant_count():
    n = _count_invariants_in_flow()
    assert n == _EXPECTED_SPEC_INVARIANT_COUNT, (
        f"Flow contract invariant count drifted: live={n}, "
        f"pinned={_EXPECTED_SPEC_INVARIANT_COUNT}. "
        f"Update _EXPECTED_SPEC_INVARIANT_COUNT + every doc reference "
        f"(docs/demo.html, JUDGES.md, docs/architecture.md) in the "
        f"same commit."
    )


def test_demo_exercises_pinned_invariant_count():
    n = _count_demo_invariants()
    assert n == _EXPECTED_DEMO_INVARIANT_COUNT, (
        f"Demo INVARIANT_DESCRIPTIONS count drifted: live={n}, "
        f"pinned={_EXPECTED_DEMO_INVARIANT_COUNT}"
    )


def test_dashboard_claims_match_pinned_counts():
    """Every user-facing federation doc must cite the pinned spec count.

    Iter-22 scope: demo.html + JUDGES.md + architecture.md.
    Iter-135 scope expansion: clinical_validation.md, fda_q_sub_draft.md,
    federated_memory.md — three user-facing docs that lied about the
    flow's invariant count (claimed "16" when the live count is 21).
    Same drift class as the iter-132 v3.9 wire-transport pin extension.
    """
    spec_str = f"{_EXPECTED_SPEC_INVARIANT_COUNT} typed"  # "21 typed invariants"
    demo_html = _DEMO_HTML.read_text()
    demo_str = (
        f"{_EXPECTED_DEMO_INVARIANT_COUNT} / "
        f"{_EXPECTED_DEMO_INVARIANT_COUNT} invariants"
    )  # "16 / 16 invariants PASS"

    for path in _USER_FACING_FED_DOCS:
        if not path.exists():
            continue
        text = path.read_text()
        rel = path.relative_to(_REPO_ROOT)
        assert spec_str in text, (
            f"{rel} must contain {spec_str!r} (the live flow declares "
            f"{_EXPECTED_SPEC_INVARIANT_COUNT} typed invariants; this doc "
            f"is on the user-facing federation surface and a stale count "
            f"would be silent fabrication of the federation contract)."
        )
    assert demo_str in demo_html, f"docs/demo.html must contain {demo_str!r}"


# Phrases that historically appeared in user-facing docs claiming the
# flow has only 16 invariants — that was true at an earlier iter but
# the count has since grown to 21. A bare "16 invariants" / "16 typed"
# in any of the user-facing docs is a regression unless paired with
# explicit "21 typed" disambiguation in the same doc.
_FORBIDDEN_BARE_PHRASES = (
    "16 invariants",          # bare "16 invariants" with no 21-disambiguator
    "16 typed invariants",
    "16 typed runtime invariants",
)


def test_no_unscoped_16_invariant_claim_in_user_facing_docs():
    """Any user-facing federation doc that mentions "16 invariants"
    must also mention "21 typed" in the same file (the disambiguation
    that prevents the bare count from looking like the full contract).

    The demo's `INVARIANT_DESCRIPTIONS` dict in scripts/federation_mock_demo.py
    legitimately has 16 entries (the live-exercised subset); that's a
    Python file, not in the user-facing scope. Likewise `arch_mind_federation_audit.md`
    measurement context says "16 of 21" which is fine.
    """
    for path in _USER_FACING_FED_DOCS:
        if not path.exists():
            continue
        text = path.read_text()
        rel = path.relative_to(_REPO_ROOT)
        for phrase in _FORBIDDEN_BARE_PHRASES:
            if phrase in text and f"{_EXPECTED_SPEC_INVARIANT_COUNT} typed" not in text:
                raise AssertionError(
                    f"{rel} contains bare {phrase!r} without the "
                    f"required '21 typed' disambiguation. Either rephrase "
                    f"the sentence to call the flow's full count "
                    f"({_EXPECTED_SPEC_INVARIANT_COUNT} typed invariants) "
                    f"or pair the 16 with explicit 'of 21' / '21 typed' "
                    f"context (the iter-22 disambiguation: 21 declared, "
                    f"16 exercised by the mock demo, 5 X25519-sealing "
                    f"invariants await MIC@2)."
                )


def test_demo_cites_this_pin_file():
    """Iter-136 surfacing extension: demo.html must cite this pin file
    near the federation-invariants surface, with a locality anchor that
    names the 6-doc cross-doc scope (the iter-135 scope expansion).

    Same `demo cites pin file` pattern as iter-110 (BitNet alone),
    iter-115 (BitNet design-class), iter-121 (negative-control cohort),
    iter-126 (Synthea cohort), iter-131 (BitNet bundle integrity). The
    pattern's purpose: surface the mechanical guard right where the
    user-facing claim lives so a judge reading the federation card sees
    that the 21-count is gated by a test, not just doc copy.
    """
    demo = _DEMO_HTML.read_text()
    pin_basename = "test_federation_invariant_count_pin.py"
    assert pin_basename in demo, (
        f"docs/demo.html must cite the pin filename {pin_basename!r} "
        f"near the JointMemoryFederation card so the cross-doc count "
        f"integrity surface has a mechanical guard pointer."
    )
    # Locality anchor — at least one of these phrases must appear in
    # demo.html so the citation is anchored to the iter-135 surface,
    # not just dropped randomly.
    locality_anchors = (
        "Cross-doc invariant-count integrity",
        "all 6 user-facing federation docs",
        "iter-135 scope-expansion",
    )
    assert any(a in demo for a in locality_anchors), (
        f"demo.html cites the pin file but is missing the iter-135 "
        f"scope-expansion locality anchor. One of these phrases must "
        f"appear near the citation: {locality_anchors!r}"
    )


def test_demo_to_spec_gap_closed():
    """As of 2026-05-11 the 5-invariant gap is CLOSED — the mock demo
    exercises all 21 invariants in-process via SealedEnvelope +
    `_x25519_seal` / `_x25519_open` round-trip. The demo dashboard must
    cite the closure (no stale "5 X25519 await wire" disclosure)."""
    demo_html = _DEMO_HTML.read_text()
    gap = _EXPECTED_SPEC_INVARIANT_COUNT - _EXPECTED_DEMO_INVARIANT_COUNT  # 0
    assert gap == 0, (
        f"Demo-to-spec gap is {gap}; should be 0 after the iter-2026-05-11 "
        f"X25519 in-process round-trip landed."
    )
    # The stale "5 X25519 sealing invariants are" disclosure must not
    # remain on the dashboard — it would now be a misclaim.
    assert "5 X25519 sealing invariants are" not in demo_html, (
        "docs/demo.html still contains the stale 5-invariant 'await wire' "
        "disclosure from before the iter-2026-05-11 X25519 in-process "
        "round-trip landed. Remove the disclosure (the gap is closed)."
    )
