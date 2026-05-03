"""Pin the BitNet parameter counts.

The dashboard, JUDGES.md, README, fda_q_sub_draft.md, and
docs/bitnet_training.md all reference specific parameter counts.
The "8,517" figure that appeared in those docs was a *typo* for
the correct **8,581** (last two digits transposed). Architecture:

    in_features:    128
    hidden:          64   →  hidden_w (128 × 64 = 8192) + hidden_b (64)
    out_features:    5    →  output_w (64 × 5 = 320)    + output_b (5)
    Ternary weights: 8192 + 320 = 8,512
    Q16.16 biases:     64 +   5 =    69
    TOTAL params   : 8,512 + 69 = 8,581

This test pins those counts so a future weight-shape change can't
silently re-introduce drift.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_BUNDLE = _REPO_ROOT / "engine" / "bitnet_weights.json"

_EXPECTED_TERNARY = 8512
_EXPECTED_BIASES = 69
_EXPECTED_TOTAL = _EXPECTED_TERNARY + _EXPECTED_BIASES  # 8581


def _flat(x):
    if isinstance(x, list):
        for el in x:
            yield from _flat(el)
    else:
        yield x


def _counts() -> dict[str, int]:
    payload = json.loads(_BUNDLE.read_text())
    counts = {
        "hidden_w": sum(1 for _ in _flat(payload["hidden_w"])),
        "hidden_b": sum(1 for _ in _flat(payload["hidden_b"])),
        "output_w": sum(1 for _ in _flat(payload["output_w"])),
        "output_b": sum(1 for _ in _flat(payload["output_b"])),
    }
    counts["ternary"] = counts["hidden_w"] + counts["output_w"]
    counts["biases"] = counts["hidden_b"] + counts["output_b"]
    counts["total"] = counts["ternary"] + counts["biases"]
    return counts


def test_bundle_param_counts_match_pinned():
    c = _counts()
    assert c["ternary"] == _EXPECTED_TERNARY, (
        f"ternary weight count drifted: live={c['ternary']}, pinned={_EXPECTED_TERNARY}"
    )
    assert c["biases"] == _EXPECTED_BIASES, (
        f"bias count drifted: live={c['biases']}, pinned={_EXPECTED_BIASES}"
    )
    assert c["total"] == _EXPECTED_TOTAL, (
        f"total param count drifted: live={c['total']}, pinned={_EXPECTED_TOTAL}"
    )


def test_no_stale_8517_remains_in_user_facing_docs():
    """The typo 8,517 must not appear in any user-facing doc or the dashboard.

    The correct number is 8,581. This test catches half-completed
    rotations that fix one mention and miss others.
    """
    files = (
        _REPO_ROOT / "README.md",
        _REPO_ROOT / "JUDGES.md",
        _REPO_ROOT / "docs" / "demo.html",
        _REPO_ROOT / "docs" / "bitnet_training.md",
        _REPO_ROOT / "docs" / "fda_q_sub_draft.md",
    )
    for p in files:
        if not p.exists():
            continue
        text = p.read_text()
        assert "8,517" not in text and "8517" not in text, (
            f"Stale param count '8,517' (typo for 8,581) still in {p.relative_to(_REPO_ROOT)}"
        )


def test_correct_count_appears_in_judges_doc():
    """JUDGES.md must reference the live counts."""
    text = (_REPO_ROOT / "JUDGES.md").read_text()
    # Either the breakdown or the total should appear
    assert ("8,512" in text) or ("8,581" in text), (
        "JUDGES.md must reference the live BitNet parameter counts (8,512 / 8,581)"
    )
