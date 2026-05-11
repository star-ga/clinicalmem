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
_ARCH_FEDERATION_AUDIT = _REPO_ROOT / "docs" / "arch_mind_federation_audit.md"


_FORBIDDEN_PHRASES = (
    # Direct conflations of v3.9 with federation transport.
    "await mind-mem v3.9 wire transport",
    "await mind-mem v3.9.0 wire transport",
    "v3.9 federation transport",
    "v3.9.0 federation transport",
    "v3.9 wire transport",
    "v3.9.0 wire transport",
    # iter-132 additions: caught in arch_mind_federation_audit.md
    "v3.9 wire-transport",
    "v3.9.0 wire-transport",
    "v3.9 transport not yet shipped",
    "v3.9.0 transport not yet shipped",
    # Indirect "v3.9 ships federation" claims.
    "mind-mem v3.9 ships federation",
    "mind-mem v3.9.0 ships federation",
    "mind-mem v3.9 ships the federation transport",
    "mind-mem v3.9.0 ships the federation transport",
)

# iter-132: extended _USER_FACING from 4 → 5 docs to include
# arch_mind_federation_audit.md, which had survived iter-122's
# initial sweep with two stale "v3.9 transport not shipped" /
# "v3.9 wire-transport" claims.
_USER_FACING = (_DEMO, _JUDGES, _README, _ARCH_DOC, _ARCH_FEDERATION_AUDIT)


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
    """OBSOLETED at iter-2026-05-11.

    The X25519 sealing invariants are no longer 'awaiting' a future
    transport — mind-mem v4.0.1 shipped wire transport (4 federation
    endpoints + FederationClient stdlib), and the mock demo now
    exercises all 21 invariants in-process via X25519/HKDF/ChaCha20-
    Poly1305. The demo's arch-mind callout was rewritten to reflect
    21/21 PASS, dropping the obsolete 'await v3.10 / MIC@2' disclosure.

    Pin retained so the file shape stays stable for git history /
    cross-pin lineage. Body is a no-op.
    """
    return


def test_judges_honest_limitations_preserves_v39_v310_distinction():
    """OBSOLETED at iter-2026-05-11.

    JUDGES.md 'Honest limitations' no longer carries the v3.9-vs-v3.10
    distinction because the upstream split it described (v3.9 ships
    single-workspace REST; v3.10 will ship MIC@2 federation transport)
    was made obsolete by mind-mem v4.0.1 shipping the wire transport
    directly. Limitations entry was rewritten to reflect that wire
    transport is shipped and that the in-process X25519 round-trip
    closes the previously-pending 5 invariants.

    Pin retained as no-op for cross-pin lineage stability.
    """
    return
