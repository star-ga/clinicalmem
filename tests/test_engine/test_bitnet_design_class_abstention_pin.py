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
    """BitNet must never emit `minor` on the live cache.

    `minor` is upstream-domain: the 5-LLM consensus + NIH RxNav layer
    handles non-clinically-significant interactions (e.g. caffeine +
    aspirin pharmacokinetic interactions). Layer 4.5 emitting `minor`
    would mean it's doing primary classification, not high-precision
    veto. A weight rotation that lifts this column off zero must
    update the demo + JUDGES design-rationale rhetoric explicitly.
    """
    totals = _column_totals()
    assert totals["minor"] == 0, (
        f"BitNet emitted `minor` predictions on {totals['minor']} live "
        f"pairs. Architectural-by-design invariant violated. Either: "
        f"(a) revert the weight rotation, or (b) update the demo's "
        f"'empty serious column is by design' rhetoric AND JUDGES.md "
        f"design-rationale row to reflect the new behavior. The pin "
        f"text in tests/test_engine/test_bitnet_design_class_abstention_pin.py "
        f"explains the safety-case rationale."
    )


def test_bitnet_never_predicts_serious():
    """BitNet must never emit `serious` on the live cache.

    `serious` is the largest cohort class (69 of 118 pairs at iter-114)
    and is fully carried by upstream Layers 1-4. Layer 4.5 emitting
    `serious` would inflate apparent recall while undermining the
    "high-precision veto on contraindicated" design rationale.
    """
    totals = _column_totals()
    assert totals["serious"] == 0, (
        f"BitNet emitted `serious` predictions on {totals['serious']} "
        f"live pairs. Architectural-by-design invariant violated. The "
        f"demo confusion-matrix card explicitly claims the empty "
        f"serious column is by design — that claim now contradicts "
        f"the live data. Update the demo + JUDGES design-rationale "
        f"rhetoric AND inspect the weight rotation that broke the "
        f"invariant."
    )


def test_demo_cites_design_invariant_near_confusion_matrix():
    """Demo must say 'serious column is by design' near the confusion-matrix."""
    text = _DEMO.read_text()
    # Specific phrasing that's been in the demo since the confusion-matrix
    # SVG was added — pinning the rhetoric prevents silent removal during
    # a future redesign that drops the design-rationale claim.
    anchor = "empty"  # leading word
    design = "serious"  # column class
    by_design = "by design"  # rationale
    veto = "veto on"  # safety-case rationale
    contra = "contraindicated"  # the role Layer 4.5 actually fills
    # Locality: all five tokens must appear within a single 600-character
    # window (heuristic — covers the existing single-paragraph claim).
    found = False
    n = len(text)
    for i in range(0, n, 200):
        window = text[i : i + 600]
        if (
            anchor in window and design in window and by_design in window
            and veto in window and contra in window
        ):
            found = True
            break
    assert found, (
        "docs/demo.html must contain the 'empty serious column is by "
        "design — ... veto on contraindicated' design-rationale claim "
        "near the confusion-matrix SVG. The architectural invariant "
        "(BitNet abstains from minor/serious) is the load-bearing "
        "safety-case justification for the layer's existence; the demo "
        "must state it explicitly so judges can cross-check it against "
        "the pinned column-totals."
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
