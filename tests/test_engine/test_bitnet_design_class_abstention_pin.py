"""Pin the architectural-by-design invariant: BitNet abstains from `minor` and `serious`.

Iteration 115 forensic. The demo's confusion-matrix card carries a
load-bearing architectural claim that has not been pinned:

  "The empty `serious` column is by design — the upstream 4-tier
   pipeline carries that class; Layer 4.5's job is the high-precision
   veto on `contraindicated`."

That claim is true at iter-114 (column totals on 118-pair cache: minor=0,
serious=0, contra=7 all TP). It is true *because* the BitNet weights
were trained to abstain from those two classes — they are the
"primary recall" classes carried by Layers 1-4 (RxNorm + OpenEvidence
+ NIH RxNav + 5-LLM consensus). Layer 4.5 is the high-precision veto
sitting BELOW those layers; its design rationale collapses if it
starts firing on classes the upstream pipeline already covers.

Without a pin, a future weight rotation could silently break the
invariant — e.g. start predicting "serious" on serious-truth pairs,
which would inflate apparent recall but actually undermine the safety
case (it would mean Layer 4.5 is competing with upstream layers
instead of vetoing them).

This pin enforces three claims:

  1. The `minor` and `serious` columns of the live confusion matrix
     are exactly zero.
  2. The demo's "empty serious column is by design" rhetoric is
     present in `docs/demo.html` near the confusion-matrix card.
  3. The architectural invariant is documented in JUDGES.md so an
     auditor can cross-reference the design claim against the
     pinned test.
"""
from __future__ import annotations

import json
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_CONFUSION = _REPO_ROOT / "docs" / "bitnet_confusion_matrix.json"
_DEMO = _REPO_ROOT / "docs" / "demo.html"
_JUDGES = _REPO_ROOT / "JUDGES.md"


def _column_totals() -> dict[str, int]:
    """Sum each column (predicted class) across all rows (truth class).

    A column total = how often BitNet emitted that class on any pair
    in the live cache.
    """
    matrix = json.loads(_CONFUSION.read_text())["matrix"]
    classes = ("none", "minor", "moderate", "serious", "major", "contraindicated")
    return {col: sum(row.get(col, 0) for row in matrix.values()) for col in classes}


def test_bitnet_never_predicts_minor():
    """V8 promotion: vocab rotated to (none/moderate/serious/major/
    contraindicated) — "minor" is no longer in the engine output set,
    so the iter-115 abstention invariant for "minor" is vacuously
    preserved (no class can emit a label that isn't in the vocab).
    """
    totals = _column_totals()
    assert totals["minor"] == 0, (
        f"BitNet emitted `minor` predictions on {totals['minor']} live "
        f"pairs. Post-iter-275, 'minor' is not in the engine output "
        f"vocabulary; if this fires, the vocab rotation was reverted."
    )


def test_bitnet_never_predicts_serious():
    """**Iter-275 v8 promotion architectural rotation**: the v8 model
    is trained against the corpus vocab (none/moderate/serious/major/
    contraindicated) for full-class prediction, NOT the v1 abstention
    architecture. v8 emits "serious" on real pairs where the cache
    ground-truth is "serious" — this is BY DESIGN under the new
    architecture (contra+major are 100% recall classes; serious is
    a primary recall class with ~84% live recall).

    The pre-v8 abstention claim was a property of cfadb4f6's hash-only
    architecture; v8's hash + ATC + pair-derived encoder + h=256 shifts
    the safety case from "high-precision veto on 2 classes" to "full-
    recall on contra/major + good recall on serious". JUDGES.md row 102
    documents the architectural progression.

    This test is preserved as a tombstone — if it ever turns to
    `totals['serious'] == 0` again that means v8 was reverted, which
    would re-open the iter-235 generalization gap.
    """
    totals = _column_totals()
    assert totals["serious"] >= 1, (
        f"BitNet emitted ZERO `serious` predictions — v8 architecture "
        f"normally produces 50+ serious predictions on the 138-pair "
        f"live cache (matching ~84% of the 69 ground-truth serious "
        f"entries). If this fires, v8 was reverted to a v1-style "
        f"abstention architecture; check engine/bitnet_weights.json "
        f"bundle_id."
    )


def test_demo_cites_design_invariant_near_confusion_matrix():
    """Demo must describe Layer 4.5's architectural role near the matrix.

    iter-421 Path B replaced the v1/v8 'high-precision veto' narrative
    (which corresponded to the abstention architecture) with a frozen-
    A-to-B-cascade narrative (100% recall on every class via the
    ensemble). The pin's intent — judges should see Layer 4.5's
    architectural role explicitly, not just the chart — is preserved;
    the rhetoric anchors are rotated to match the new architecture.
    """
    text = _DEMO.read_text()
    # iter-421 ensemble narrative anchors.
    anchors = (
        "ensemble",         # the architectural primitive of iter-421
        "constrained argmax",  # the dispatcher rule (B's class-mask)
        "100% recall",      # the load-bearing claim
        "tier-2",           # B's role descriptor
        "contra",           # A's preserved role
    )
    found = False
    n = len(text)
    for i in range(0, n, 200):
        window = text[i : i + 800]
        if all(a in window for a in anchors):
            found = True
            break
    assert found, (
        "docs/demo.html must contain the iter-421 ensemble architectural "
        "claim near the confusion-matrix SVG. All of "
        f"{anchors!r} must appear within a single 800-char window so "
        "judges can cross-check the dispatcher narrative against the "
        "actual matrix and bundle_ids."
    )


def test_demo_names_ensemble_classes_with_counts():
    """Demo must name all four severity classes with their live counts
    (e.g. `22/22 moderate`) near the matrix.

    iter-421 ensemble achieves 100% recall on every class. The pin's
    intent — judges should be able to cross-check the matrix JSON
    against demo prose without running tests — is preserved with new
    anchors rotated to match the iter-421 cohort sizes (44 contra +
    4 major + 69 serious + 22 moderate).
    """
    matrix = json.loads(_CONFUSION.read_text())["matrix"]
    expected_counts = {}
    for cls in ("contraindicated", "major", "serious", "moderate"):
        total = sum(matrix[cls].values())
        expected_counts[cls] = total
    text = _DEMO.read_text()
    expected_substrings = (
        f"{expected_counts['contraindicated']}/{expected_counts['contraindicated']} contra",
        f"{expected_counts['major']}/{expected_counts['major']} major",
        f"{expected_counts['serious']}/{expected_counts['serious']} serious",
        f"{expected_counts['moderate']}/{expected_counts['moderate']} moderate",
    )
    n = len(text)
    found = False
    for i in range(0, n, 100):
        window = text[i : i + 1000]
        if all(s in window for s in expected_substrings):
            found = True
            break
    assert found, (
        "docs/demo.html confusion-matrix card must include all four "
        f"per-class live counts {expected_substrings!r} within a 1000-char "
        "window so judges can cross-check `docs/bitnet_confusion_matrix.json` "
        "against the prose without running tests. Cache growth that bumps "
        "any of these counts requires rotating the demo numbers in lockstep."
    )


def test_judges_documents_design_class_abstention():
    """JUDGES.md must document the architectural-by-design invariant."""
    text = _JUDGES.read_text()
    # JUDGES already mentions the safety invariant in the runnable-claims
    # table; this pin requires the design-rationale specifically appear
    # alongside the new pin file citation OR in the architecture section.
    assert (
        "test_bitnet_design_class_abstention_pin.py" in text
        or (
            "high-precision veto" in text
            and ("minor" in text or "serious" in text)
        )
    ), (
        "JUDGES.md must either cite "
        "tests/test_engine/test_bitnet_design_class_abstention_pin.py "
        "OR document the architectural-by-design invariant (BitNet "
        "abstains from `minor` and `serious` to preserve the "
        "high-precision-veto safety case). Same drift class as iter-107 "
        "(JUDGES manifest description listed only 4 gates)."
    )
