"""Pin the federation invariant counts.

Two interlocking claims appear across docs/demo.html, JUDGES.md,
docs/architecture.md, and the federation mock demo:

  Spec  : `JointMemoryFederation.flow.mind` declares 21 typed
          runtime invariants (the full federation contract).
  Demo  : `scripts/federation_mock_demo.py` exercises 16 of those
          21 (the 5 X25519 sealing invariants 17-21 are *declared*
          but await mind-mem v3.9 wire transport before the demo
          can verify them end-to-end).

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

# Pinned constants. When an `invariant` is added to or removed from
# the flow contract, update the matching number AND every doc that
# names it in the same commit.
_EXPECTED_SPEC_INVARIANT_COUNT = 21    # full typed contract
_EXPECTED_DEMO_INVARIANT_COUNT = 16    # exercised by federation_mock_demo


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
    """Dashboard text must cite the pinned spec + demo numbers."""
    demo_html = _DEMO_HTML.read_text()
    judges = _JUDGES.read_text()
    arch = _ARCH.read_text()

    spec_str = f"{_EXPECTED_SPEC_INVARIANT_COUNT} typed"  # "21 typed invariants"
    demo_str = (
        f"{_EXPECTED_DEMO_INVARIANT_COUNT} / "
        f"{_EXPECTED_DEMO_INVARIANT_COUNT} invariants"
    )  # "16 / 16 invariants PASS"

    assert spec_str in demo_html, f"docs/demo.html must contain {spec_str!r}"
    assert spec_str in judges, f"JUDGES.md must contain {spec_str!r}"
    assert spec_str in arch, f"docs/architecture.md must contain {spec_str!r}"
    assert demo_str in demo_html, f"docs/demo.html must contain {demo_str!r}"


def test_demo_to_spec_gap_explanation_present():
    """The 5-invariant gap (16 demo vs 21 spec) must be explained on the dashboard.

    Iteration 22 added explicit copy under the Phase B governance card
    that calls out which invariants are *declared* vs *exercised*.
    Without that disclosure, the 16 vs 21 split would look like a
    silent overclaim. Make sure the disclosure stays present.
    """
    demo_html = _DEMO_HTML.read_text()
    gap = _EXPECTED_SPEC_INVARIANT_COUNT - _EXPECTED_DEMO_INVARIANT_COUNT  # 5
    assert (
        f"the {gap} X25519 sealing invariants" in demo_html
        or f"{gap} X25519 sealing invariants are" in demo_html
    ), (
        "The 16-of-21 gap explanation must remain on docs/demo.html "
        "(governance Phase B card)."
    )
