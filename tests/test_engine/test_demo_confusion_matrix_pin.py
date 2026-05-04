"""Pin: the confusion-matrix display in demo.html matches the JSON artifact.

Iter 51 added an inline-SVG 4×6 heatmap of the live BitNet confusion
matrix to docs/demo.html. The cell values are hand-coded into the SVG
(no client-side JSON fetch — the page must be deterministic for offline
demo evaluators). This test guards against silent drift between the
JSON artifact and the SVG cells.

If a future iteration regenerates the artifact and the cells change,
this test fires; either update the SVG or accept that the heatmap is
no longer current.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_DEMO = _REPO_ROOT / "docs" / "demo.html"
_ARTIFACT = _REPO_ROOT / "docs" / "bitnet_confusion_matrix.json"


def test_demo_confusion_cells_match_artifact():
    """Each numeric cell in the demo's 4×6 heatmap exists in the JSON artifact."""
    artifact = json.loads(_ARTIFACT.read_text())
    matrix = artifact["matrix"]

    # The 4 active GT classes shipped to demo (none / minor have no
    # ground-truth examples in the live cache).
    visible_gts = ["moderate", "serious", "major", "contraindicated"]
    cols = ["none", "minor", "moderate", "serious", "major", "contraindicated"]

    expected = []
    for gt in visible_gts:
        for col in cols:
            expected.append(matrix[gt][col])
    # 4 rows × 6 cols = 24 cells.
    assert len(expected) == 24

    demo_text = _DEMO.read_text()

    # The heatmap occupies the section between the
    # `Layer 4.5 BitNet confusion matrix` heading and the recall=6/18 footer.
    start = demo_text.find("Layer 4.5 BitNet confusion matrix &mdash; live deployment")
    assert start != -1, "Confusion matrix card not found in demo.html"
    end = demo_text.find("tp_contraindicated_at_least_seven", start)
    assert end != -1, "Confusion matrix card footer not found"
    section = demo_text[start:end]

    # Walk the section in display order: each <text ... text-anchor="middle">N</text>
    # cell text is one matrix value. Header column-labels (none / minor / etc.)
    # are excluded by their non-numeric body.
    cell_pattern = re.compile(
        r'<text[^>]*?text-anchor="middle"[^>]*?>(\d+)</text>',
        re.DOTALL,
    )
    cells = [int(m.group(1)) for m in cell_pattern.finditer(section)]

    # The 24 matrix cells appear after the 6 column labels (which are
    # words, not digits, so they don't match `\d+`). The footer caption
    # contains digit-bearing text ("6 / 6", "6 / 18") that DOES match —
    # strip those by truncating to the first 24 hits in document order.
    assert len(cells) >= 24, (
        f"Expected ≥ 24 numeric cells in heatmap, found {len(cells)}"
    )
    assert cells[:24] == expected, (
        f"Demo heatmap cells drifted from {_ARTIFACT.relative_to(_REPO_ROOT)}.\n"
        f"  expected (row-major, 4 GT × 6 pred): {expected}\n"
        f"  live cells (first 24):              {cells[:24]}\n"
        f"Re-run `python3 scripts/build_bitnet_confusion_matrix.py` then "
        f"hand-edit the SVG cells in docs/demo.html to match."
    )


def test_demo_card_references_pin_test_path():
    """The heatmap caption must point at the test that pins safety invariants."""
    text = _DEMO.read_text()
    assert "tests/test_scripts/test_bitnet_confusion_matrix.py" in text, (
        "Demo confusion-matrix card must reference the safety-invariant pin "
        "test so a judge can audit the pin."
    )
    assert "fp_contraindicated_is_zero" in text, (
        "The fp_contraindicated_is_zero safety invariant must be named "
        "explicitly in the demo."
    )
