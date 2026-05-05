"""Pin: docs must NEVER claim v3.9.0 ships federation transport.

Iteration 122 forensic. iter-54 caught the same drift class at the
v3.9.0 milestone — the JUDGES.md "Honest limitations" #3 was
disambiguated to: "v3.9.0 added an http_transport.py module but it
is a single-workspace REST adapter for non-MCP clients (Slack bots /
Streamlit / etc.), NOT a peer-to-peer federation transport — the
dedicated MIC@2 federation transport adapter targets mind-mem v3.10".

iter-122 caught a SURVIVING claim in `docs/demo.html` arch-mind callout
(L977) — "5 X25519 sealing invariants ... await mind-mem v3.9 wire
transport". v3.9 IS shipped; the X25519 invariants await v3.10's MIC@2
transport, NOT v3.9.

Why this matters:
  A judge reading the demo's arch-mind section sees "await v3.9" and
  cross-checks the live mind-mem version (3.9.0) and concludes the
  federation IS LIVE. That contradicts the JUDGES "Honest limitations"
  disclosure that explicitly says federation transport is mocked
  pending v3.10. Silent contradiction across two judge-visible docs
  erodes trust faster than any other failure mode.

Same drift class as:
  iter-89  - SVG row comments
  iter-91  - FHIR genesis block
  iter-101 - BitNet body recall
  iter-102 - README "85.7% precision on contraindicated" mislabel
  iter-107 - JUDGES manifest description (4 gates → 5)
  iter-112 - README BitNet contra recall stale 8 iters
  iter-117 - BitNet contra TP floor `at_least_six` ratchet

Three invariants enforced by this pin:

  1. NO doc may say "await mind-mem v3.9" / "v3.9 wire transport" /
     "v3.9 federation transport" — those phrases are factually wrong
     since v3.9.0 ships only `http_transport.py` (single-workspace
     REST adapter).
  2. The demo's arch-mind callout MUST contain "v3.10" or "MIC@2"
     somewhere alongside the X25519 sealing invariant disclosure
     (so a judge sees the correct upstream target).
  3. JUDGES.md "Honest limitations" #3 must continue to disclose the
     v3.9 single-workspace REST vs v3.10 federation distinction.
"""
from __future__ import annotations

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_DEMO = _REPO_ROOT / "docs" / "demo.html"
_JUDGES = _REPO_ROOT / "JUDGES.md"
_README = _REPO_ROOT / "README.md"
_ARCH_DOC = _REPO_ROOT / "docs" / "architecture.md"


_FORBIDDEN_PHRASES = (
    # Direct conflations of v3.9 with federation transport.
    "await mind-mem v3.9 wire transport",
    "await mind-mem v3.9.0 wire transport",
    "v3.9 federation transport",
    "v3.9.0 federation transport",
    "v3.9 wire transport",
    "v3.9.0 wire transport",
    # Indirect "v3.9 ships federation" claims.
    "mind-mem v3.9 ships federation",
    "mind-mem v3.9.0 ships federation",
    "mind-mem v3.9 ships the federation transport",
    "mind-mem v3.9.0 ships the federation transport",
)

_USER_FACING = (_DEMO, _JUDGES, _README, _ARCH_DOC)


def test_no_doc_claims_v39_ships_federation_transport():
    """No user-facing doc may conflate v3.9.0 with federation transport.

    v3.9.0 added `http_transport.py` (single-workspace REST adapter for
    non-MCP clients like Slack bots / Streamlit dashboards). The
    dedicated MIC@2 peer-to-peer federation transport targets v3.10.
    Any phrase that implies otherwise is misleading.
    """
    offenders = []
    for path in _USER_FACING:
        if not path.exists():
            continue
        text = path.read_text()
        for phrase in _FORBIDDEN_PHRASES:
            if phrase in text:
                offenders.append((path.name, phrase))
    assert not offenders, (
        f"User-facing doc(s) contain a forbidden v3.9-ships-federation "
        f"phrase:\n"
        + "\n".join(f"  {name}: {phrase!r}" for name, phrase in offenders)
        + "\n\nv3.9.0 ships `http_transport.py` (single-workspace REST adapter), "
          "NOT federation transport. The dedicated MIC@2 federation transport "
          "adapter targets mind-mem v3.10. Update the offending doc(s) to "
          "use the correct upstream target."
    )


def test_demo_archmind_callout_names_correct_target():
    """Demo's arch-mind X25519-sealing-invariant callout names v3.10 / MIC@2.

    The arch-mind 6-healthcare-invariants callout discloses that 5 of
    21 federation invariants (the X25519 sealing ones) are 'declared
    but await' the federation transport adapter. The 'await' target
    must be v3.10 (or the MIC@2 adapter), NOT v3.9 (which is shipped
    and pinned).
    """
    text = _DEMO.read_text()
    # Find the X25519 sealing disclosure region and check the upstream
    # target nearby.
    sealing_anchor = "X25519 sealing invariants"
    if sealing_anchor not in text:
        # If the demo refactors away from this exact phrase, this pin
        # becomes a no-op until the new phrasing stabilizes.
        return
    idx = text.index(sealing_anchor)
    # Look in a 600-char window around the anchor for an upstream-target
    # phrase. v3.10 OR MIC@2 OR "federation-transport adapter" are all OK.
    window = text[max(0, idx - 100) : idx + 600]
    has_v310 = "v3.10" in window
    has_mic2 = "MIC@2" in window
    has_adapter = "federation-transport adapter" in window or "federation transport adapter" in window
    assert has_v310 or has_mic2 or has_adapter, (
        f"Demo's X25519-sealing-invariant disclosure must name the correct "
        f"upstream target. Looked for 'v3.10' or 'MIC@2' or "
        f"'federation-transport adapter' in a 600-char window around "
        f"'X25519 sealing invariants' — none found. The 'await' target "
        f"is the dedicated MIC@2 federation-transport adapter (mind-mem "
        f"v3.10), NOT mind-mem v3.9.0 which ships only a single-workspace "
        f"REST adapter."
    )


def test_judges_honest_limitations_preserves_v39_v310_distinction():
    """JUDGES.md 'Honest limitations' must keep the v3.9-vs-v3.10 distinction.

    iter-54 added the disambiguation (v3.9.0 ships single-workspace REST,
    NOT federation transport). A future copy edit could silently
    collapse the two — judges + auditors would no longer know which
    upstream component is live vs deferred.
    """
    text = _JUDGES.read_text()
    must_have_signals = (
        "single-workspace REST",
        "MIC@2",
    )
    for signal in must_have_signals:
        assert signal in text, (
            f"JUDGES.md must mention {signal!r} in the 'Honest "
            f"limitations' federation entry. Without the "
            f"v3.9-single-workspace-vs-v3.10-MIC@2 distinction, the "
            f"federation-transport claim collapses into ambiguity. "
            f"Same drift class as iter-54 milestone disambiguation."
        )
