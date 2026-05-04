"""Pin per-class size comments inside ``docs/demo.html`` SVGs.

The dashboard ships two SVGs that document per-class cohort sizes via
HTML comments adjacent to each row:

* The "Recall by severity class" sparkline (4 bars).
* The "Layer 4.5 BitNet confusion matrix" 5x4 grid (4 rows).

These comments steer human readers ("the contraindicated class has 20
ground-truth pairs in the live cache; this row's `100% · 20 / 20` label
is correct"). They are NOT user-visible — but they ARE load-bearing
documentation that the visible labels were derived from a real cache
count, not fabricated.

Iter-89 audit caught a multi-iter drift where:

* Sparkline comments lagged 4-class growth: ``Contraindicated (n=17)``
  was last accurate at iter-50, ``Major (n=1)`` was accurate iter-39
  through iter-82, ``Serious (n=64)`` was accurate at iter-19, and
  ``Moderate (n=20)`` was accurate at iter-9.
* Confusion-matrix row comments lagged: ``Row 2: serious (n=67)`` was
  accurate at iter-78 but missed iter-83 (pt-021 second major) and
  iter-88 (pt-022 second serious cohort growth) and ``Row 3: major
  (n=1)`` missed iter-83.

Both visible labels (e.g. ``100% · 20 / 20``) and per-class counts in
``docs/openevidence_cache.json`` were rotated correctly each iter — only
the developer-facing comments drifted.

This test pins the comments to the live cache so the next cohort growth
fails the gate until the comments rotate too.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_DEMO_HTML = _REPO_ROOT / "docs" / "demo.html"
_CACHE = _REPO_ROOT / "docs" / "openevidence_cache.json"
_CONFUSION_MATRIX = _REPO_ROOT / "docs" / "bitnet_confusion_matrix.json"

_SPARKLINE_PATTERN = re.compile(
    r"<!--\s*(?P<klass>Contraindicated|Major|Serious|Moderate)\s*"
    r"\(n=(?P<n>\d+)[^)]*\)\s*-->"
)
_MATRIX_ROW_PATTERN = re.compile(
    r"<!--\s*Row\s*\d+:\s*(?P<klass>contraindicated|major|serious|moderate)\s*"
    r"\(n=(?P<n>\d+)\)[^-]*-->"
)


def _live_per_class_counts() -> dict[str, int]:
    cache = json.loads(_CACHE.read_text())
    counts: dict[str, int] = {}
    for it in cache:
        sev = (it.get("severity") or "").lower()
        if sev:
            counts[sev] = counts.get(sev, 0) + 1
    return counts


def test_sparkline_comments_match_live_cache():
    html = _DEMO_HTML.read_text()
    counts = _live_per_class_counts()
    matches = list(_SPARKLINE_PATTERN.finditer(html))
    seen = {m.group("klass").lower() for m in matches}
    expected = {"contraindicated", "major", "serious", "moderate"}
    assert expected.issubset(seen), (
        f"Sparkline missing class comments: expected {expected}, "
        f"found {seen}. Each <text>...100% · N / N</text> bar must "
        f"have a sibling <!-- Class (n=N) --> comment."
    )
    for m in matches:
        klass = m.group("klass").lower()
        n = int(m.group("n"))
        live = counts.get(klass, 0)
        assert n == live, (
            f"Sparkline comment '<!-- {m.group('klass')} (n={n}) -->' "
            f"is stale: live cache count for {klass!r} is {live}. "
            f"Update the comment AND the visible '100% · {live} / {live}' "
            f"label, then refresh docs/bitnet_confusion_matrix.json."
        )


def test_confusion_matrix_row_comments_match_live_cache():
    html = _DEMO_HTML.read_text()
    counts = _live_per_class_counts()
    matches = list(_MATRIX_ROW_PATTERN.finditer(html))
    assert len(matches) >= 4, (
        f"Confusion matrix should have ≥4 row comments "
        f"(<!-- Row N: classname (n=K) -->); found {len(matches)}."
    )
    for m in matches:
        klass = m.group("klass").lower()
        n = int(m.group("n"))
        live = counts.get(klass, 0)
        assert n == live, (
            f"Confusion-matrix row comment '<!-- Row N: {klass} (n={n}) -->' "
            f"is stale: live cache count for {klass!r} is {live}. "
            f"Cohort growth must rotate the row label comment AND the "
            f"corresponding cells in docs/bitnet_confusion_matrix.json."
        )


def test_confusion_matrix_per_class_totals_match_cache():
    """Belt-and-braces: matrix per_class.ground_truth_total == cache count."""
    matrix = json.loads(_CONFUSION_MATRIX.read_text())
    counts = _live_per_class_counts()
    pc = matrix.get("per_class", {})
    for klass in ("contraindicated", "major", "serious", "moderate"):
        live = counts.get(klass, 0)
        recorded = pc.get(klass, {}).get("ground_truth_total", -1)
        assert recorded == live, (
            f"docs/bitnet_confusion_matrix.json per_class.{klass}."
            f"ground_truth_total = {recorded} drifted from live cache "
            f"count {live}. Re-run scripts/build_bitnet_confusion_matrix.py."
        )
