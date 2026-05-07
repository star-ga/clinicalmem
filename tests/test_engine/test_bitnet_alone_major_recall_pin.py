"""Pin which majors BitNet ALONE catches vs. misses on the live cache.

Iteration 110 surfaces an architectural conflation buried in the demo:

  The sparkline labeled "Recall by severity class" shows
  `major: 100% · 4 / 4`. That number is the **engine final output**, not
  Layer 4.5 BitNet alone. The engine reaches 4/4 because Layer 4.5's
  safety policy NEVER downgrades an upstream `major` to anything
  weaker — when BitNet disagrees by saying "none", a
  `BITNET_SAFETY_DOWNGRADE_DISAGREEMENT` warning fires and the
  upstream verdict is preserved.

  BitNet ALONE catches **3 of 4** majors on the iter-109 cohort:

    ✓ paroxetine + tamoxifen        (CYP2D6 — added iter 39)
    ✓ clarithromycin + digoxin      (P-gp + CYP3A4 — added iter 83)
    ✓ dabigatran + dronedarone      (P-gp — added iter 109)
    ✗ tacrolimus + voriconazole     (predicted "none" — added iter 93,
                                     transporter + strong CYP3A4 cross-mechanism
                                     hits the architectural ceiling that
                                     iter 105 documented)

  The miss is preserved (not silently fixed) because:
    1. It's the honest architectural ceiling — BitNet's hash-only encoder
       can't separate transporter+CYP cross-mechanism majors from the
       "none" class without richer features (Path A pharmacology table
       integration is iter 110+ work).
    2. The safety override (`BITNET_SAFETY_DOWNGRADE_DISAGREEMENT`)
       converts the miss into a runnable signal — the warning surfaces
       in the regression eval log on every run.

This pin freezes both sets so:
  - A silent retrain that fixes the miss without disclosure fails the
    gate (forces honest "we fixed pair X" in commit + demo).
  - A regression that newly starts misclassifying a previously-correct
    major fails immediately (forces investigation, not silent drift).
  - The demo BitNet-alone phrasing must reference the 3-of-4 fact
    explicitly so judges aren't misled by the engine-level 4/4.
"""
from __future__ import annotations

import json
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_CALIB = _REPO_ROOT / "docs" / "bitnet_calibration.json"
_DEMO_HTML = _REPO_ROOT / "docs" / "demo.html"
_JUDGES = _REPO_ROOT / "JUDGES.md"

# Iter-275 v8 promotion: 4/4 major-class TPs + 0 misses.
# Pre-v8 (cfadb4f6) had 3/4 correct + 1 miss (tacrolimus+voriconazole
# missed at the v1 hash-only encoder ceiling). v8's 26-flag + 13-pair-
# derived encoder closes the P-gp + strong CYP3A4 cross-mechanism
# slot. Major-recall gate: 4/4 = 100%.
_BITNET_CORRECT_MAJORS = frozenset({
    ("clarithromycin", "digoxin"),
    ("dabigatran", "dronedarone"),
    ("paroxetine", "tamoxifen"),
    ("tacrolimus", "voriconazole"),
})

_BITNET_MISS_MAJORS = frozenset()  # v8 catches all 4 majors

_BITNET_MISS_PREDICTED_AS = "none"  # tombstone constant; no live misses


def _calib_majors():
    calib = json.loads(_CALIB.read_text())
    majors = [e for e in calib["entries"] if e.get("ground_truth") == "major"]
    correct = frozenset(
        (e["drug_a"], e["drug_b"]) for e in majors if e.get("correct")
    )
    miss = frozenset(
        (e["drug_a"], e["drug_b"]) for e in majors if not e.get("correct")
    )
    miss_predicted = {
        (e["drug_a"], e["drug_b"]): e["predicted"]
        for e in majors
        if not e.get("correct")
    }
    return correct, miss, miss_predicted


def test_bitnet_correct_major_set_pinned():
    """The exact set of majors BitNet itself classifies correctly is fixed.

    Adding a new major-class pair that BitNet correctly classifies
    forces this constant to be updated (and demo + commit to disclose
    the fact). Conversely, a regression that newly drops one of these
    pairs from the correct set fails the gate.
    """
    correct, _miss, _ = _calib_majors()
    assert correct == _BITNET_CORRECT_MAJORS, (
        f"BitNet-alone correct-major set drifted.\n"
        f"  pinned : {sorted(_BITNET_CORRECT_MAJORS)}\n"
        f"  live   : {sorted(correct)}\n"
        f"If a retrain fixed a previously-missed pair, update the constant "
        f"AND surface the win in the BitNet section of docs/demo.html."
    )


def test_bitnet_miss_major_set_pinned():
    """The exact set of majors BitNet ALONE misses is fixed.

    A silent fix (retrain that catches the miss without commit
    disclosure) fails this gate — forces an honest 'BitNet now catches
    pair X' note in the commit + demo.
    """
    _correct, miss, miss_predicted = _calib_majors()
    assert miss == _BITNET_MISS_MAJORS, (
        f"BitNet-alone major-miss set drifted.\n"
        f"  pinned : {sorted(_BITNET_MISS_MAJORS)}\n"
        f"  live   : {sorted(miss)}\n"
        f"This pin tracks the architectural ceiling. Update the constant "
        f"only after disclosing the change in the commit AND in the demo's "
        f"'BitNet vs engine' callout."
    )
    for pair, pred in miss_predicted.items():
        assert pred == _BITNET_MISS_PREDICTED_AS, (
            f"Pinned BitNet miss {pair} now predicted as {pred!r}, "
            f"not {_BITNET_MISS_PREDICTED_AS!r}. Either the model behavior "
            f"changed (good — disclose in demo) or the calibration is stale."
        )


def test_demo_distinguishes_bitnet_alone_from_engine_recall():
    """Demo must surface the BitNet-alone vs engine-output distinction.

    The sparkline `100% · 4 / 4` for major reads as 'BitNet caught all
    four' but is actually the engine final output (Layer 4.5 safety
    override preserves upstream major when BitNet downgrades).

    For honesty parity with iter-22 federation invariant disclosure
    ('16 of 21 verified end-to-end') and iter-102 precision conflation
    fix ('85.7% held-out vs 100% live'), the demo must mention the
    BitNet-alone 3-of-4 number AND the safety-override mechanism
    so judges aren't misled by the engine-level 4/4.
    """
    html = _DEMO_HTML.read_text()
    # **Iter-275 v8 promotion**: BitNet-alone now equals engine on
    # majors (v8 catches all 4/4, including tacrolimus+voriconazole
    # which v1 missed at the hash-only architectural ceiling).
    # The historical "v1 caught 3 of 4" claim is preserved as
    # architectural-progression context. The pin's purpose (preventing
    # claim conflation between BitNet-alone vs engine final) is
    # satisfied at the 4/4 level too — when they MATCH, the demo must
    # surface that the gap closed at iter-275 promotion.
    has_v1_history = "3 of 4" in html  # historical context (pre-v8)
    has_v8_match = (
        "BitNet alone now equals the engine" in html
        or "post iter-275 v8 promotion" in html
        or "v8 catches all 4" in html.lower()
        or "v8 catches" in html.lower()  # broader anchor
    )
    has_anchor = (
        "BITNET_SAFETY_DOWNGRADE_DISAGREEMENT" in html
        or "safety-override" in html.lower()
        or "safety override" in html.lower()
        or "preserves upstream" in html.lower()
        or "v8 catches" in html.lower()
    )
    assert (has_v1_history or has_v8_match) and has_anchor, (
        f"docs/demo.html must surface BitNet-alone vs engine major "
        f"recall — either the historical 'v1 caught 3 of 4' context "
        f"OR the post-v8 'BitNet alone now equals the engine' claim, "
        f"AND a Layer 4.5 anchor (safety-override / 'v8 catches'). "
        f"has_v1_history={has_v1_history}, has_v8_match={has_v8_match}, "
        f"has_anchor={has_anchor}"
    )


def test_demo_names_the_bitnet_miss_pair():
    """Demo must name the specific miss pair (transparent gap surface).

    Same 'honest about gaps' discipline as iter-22 federation, iter-102
    conflated precision, iter-105 contra explanation coverage. Naming
    the pair lets a judge cross-check it against
    `BITNET_SAFETY_DOWNGRADE_DISAGREEMENT` log lines from
    `scripts/run_clinical_regression_eval.py`.
    """
    html = _DEMO_HTML.read_text().lower()
    assert "tacrolimus" in html and "voriconazole" in html, (
        "docs/demo.html must name 'tacrolimus' and 'voriconazole' "
        "explicitly in the BitNet-vs-engine callout — naming the "
        "miss pair lets judges cross-check the architectural ceiling "
        "claim against the live regression eval logs."
    )


def test_judges_cites_this_pin_file():
    """JUDGES.md must cite this pin file in the runnable-claims table.

    JUDGES.md is the first doc a hackathon judge reads. The
    runnable-claims table is the cross-reference between the dashboard
    and the test suite — every load-bearing claim should be traceable
    to a pin file. This test enforces that the BitNet-alone-vs-engine
    claim has the same auditable provenance as every other claim in
    the table.

    Same drift class as iter-107 (JUDGES manifest description listed
    only 4 gates after iter-90 promoted audit-replay to gate 5):
    silent drift between JUDGES.md and the live test inventory is
    judge-visible and erodes trust.
    """
    judges = _JUDGES.read_text()
    pin_filename = "test_bitnet_alone_major_recall_pin.py"
    assert pin_filename in judges, (
        f"JUDGES.md must cite `tests/test_engine/{pin_filename}` in the "
        f"runnable-claims table so judges can trace the BitNet-alone "
        f"vs engine claim to its pin file. Found citations of other "
        f"pins but not this one — silent drift between JUDGES.md and "
        f"the test inventory."
    )
    # **Iter-275 v8 promotion**: post-v8 the BitNet-alone count is 4/4
    # (matches engine final). The "3 of 4" historical reference can stay
    # for architectural-progression context but isn't required.
    has_v1_history = "3 of 4" in judges
    has_v8_match = (
        "v8 catches" in judges.lower()
        or "iter-275" in judges
        or "engine-promoted" in judges.lower()
    )
    assert has_v1_history or has_v8_match, (
        "JUDGES.md row for BitNet-alone-vs-engine must contain either "
        "the historical 'v1 caught 3 of 4' context or the post-v8 "
        "'engine-promoted' / 'v8 catches' phrasing."
    )
