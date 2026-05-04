"""Pin: no doc may conflate 85.7% held-out accuracy with deployment precision.

Iter 29 forensic: the dashboard's old "85.7% precision on contraindicated"
claim came from `docs/bitnet_training.md` line 146 — the held-out
**test-set per-class accuracy** (n=42 training subset), mislabelled as
deployment "precision". The two are different numbers measured against
different cohorts:

  - **85.7%**: held-out test-subset per-class accuracy, n=42
  - **100% precision / 28.6% recall**: deployment-side performance
    against the live OpenEvidence cache, n=21 contraindicated

Iter 29 fixed the demo, but the README continued to say "85.7%
precision on the safety-critical `contraindicated` class" until iter 102.
Same drift class — easy to miss because the *number* doesn't drift
(still 85.7% held-out), only the *label* is wrong.

This pin asserts no user-facing doc contains a conflated claim
("85.7%" + "precision" + "contraindicated" within close range of each
other without the disambiguating "held-out" / "test-set" qualifier).

If a future maintainer wants to use 85.7% as a precision number, they
must run a new precision evaluation against a held-out *contraindicated-
only* set and update both the test fixture AND every doc reference in
the same commit.
"""
from __future__ import annotations

import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# Files where this conflation has historically appeared.
_USER_FACING_DOCS = (
    _REPO_ROOT / "README.md",
    _REPO_ROOT / "JUDGES.md",
    _REPO_ROOT / "docs" / "demo.html",
    _REPO_ROOT / "docs" / "fda_q_sub_draft.md",
    _REPO_ROOT / "docs" / "bitnet_training.md",
    _REPO_ROOT / "docs" / "why_bitnet_b158.md",
    _REPO_ROOT / "docs" / "clinical_validation.md",
)

# Conflated claim: "85.7%" close to "precision" close to "contraindicated"
# WITHOUT a held-out / test-subset / per-class accuracy qualifier nearby.
# Permit the historically-correct framing where these terms appear
# in the SAME sentence as a disambiguator.
_DISAMBIGUATORS = (
    "held-out",
    "test subset",
    "test-set",
    "per-class accuracy",
    "training subset",
    "(n=42)",
)

_PERCENT_PRECISION_CLAIM = re.compile(
    r"\b85\.7%\s+precision\s+on\s+(?:the\s+)?(?:safety-critical\s+)?(?:`)?contraindicated",
    re.IGNORECASE,
)


def test_no_user_facing_doc_conflates_85_7_precision_with_deployment():
    """The exact conflated claim ('85.7% precision on contraindicated')
    must not appear in any user-facing doc without a disambiguator."""
    offenders = []
    for path in _USER_FACING_DOCS:
        if not path.exists():
            continue
        text = path.read_text()
        for m in _PERCENT_PRECISION_CLAIM.finditer(text):
            # Look at the surrounding ~200 chars for a disambiguator.
            start = max(0, m.start() - 200)
            end = min(len(text), m.end() + 200)
            context = text[start:end].lower()
            has_disambiguator = any(d.lower() in context for d in _DISAMBIGUATORS)
            if not has_disambiguator:
                offenders.append((path.relative_to(_REPO_ROOT), m.group(0)))
    assert not offenders, (
        f"Conflated '85.7% precision on contraindicated' claim in: "
        f"{offenders}. The 85.7% is held-out per-class accuracy on a "
        f"42-row training subset, NOT deployment precision (which is "
        f"100% on the live 21-pair contraindicated cache). Add a "
        f"disambiguator nearby ('held-out test subset (n=42)' or "
        f"'per-class accuracy') OR replace with the live 100% precision."
    )


def test_85_7_immediate_context_has_held_out_qualifier():
    """Tighter check: when '85.7%' appears IMMEDIATELY adjacent to
    'precision' (within ~80 chars on either side), a disambiguator
    must appear in the same window. The 200-char check was too
    aggressive — `clinical_validation.md` legitimately mentions
    accuracy table rows AND high-precision-by-design framing in the
    same section. The literal '85.7% precision' adjacency is what
    we want to block."""
    offenders = []
    for path in _USER_FACING_DOCS:
        if not path.exists():
            continue
        text = path.read_text()
        for m in re.finditer(r"\b85\.7%", text):
            start = max(0, m.start() - 80)
            end = min(len(text), m.end() + 80)
            window = text[start:end]
            # Only flag when "precision" appears IMMEDIATELY next to
            # 85.7% (within 80 chars), not loosely associated.
            tight_window_re = re.compile(
                r"85\.7%[^.\n]{0,40}\bprecision\b|\bprecision\b[^.\n]{0,40}85\.7%",
                re.IGNORECASE,
            )
            if not tight_window_re.search(window):
                continue
            if not any(d in window.lower() for d in (s.lower() for s in _DISAMBIGUATORS)):
                offenders.append((path.relative_to(_REPO_ROOT), window[:160]))
    assert not offenders, (
        f"'85.7% precision' adjacency without disambiguator in: "
        f"{[o[0] for o in offenders]}. First offender: "
        f"{offenders[0][1] if offenders else ''}"
    )
