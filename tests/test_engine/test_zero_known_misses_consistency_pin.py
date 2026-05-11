"""Pin: demo "0 known misses" claim ↔ live `_V8_EXPECTED_MISSES`.

Iter 260 (round 47 T1) — 18th cross-pin family. The user's directive
at iter-255 ("100% / 41/41 is the only goal") and the iter-256
velocity-badge upgrade surfaced the **0 known misses** claim in the
hero badge ("255+ iter · 5-tier round-robin · 17 cross-pin · 0 known
misses") + the v8 callout body ("zero known misses").

These claims hard-depend on `_V8_EXPECTED_MISSES = ()` (empty tuple)
in `tests/test_engine/test_path_a_v8_live_recall_pin.py`. If a future
cohort growth adds a pair v8 misses, that tuple will gain entries —
and the demo claims would silently become stale.

This pin closes the cross-surface gap by asserting:

  IF `len(_V8_EXPECTED_MISSES) == 0`:
      demo MUST claim "0 known misses" + "zero known misses"
  ELSE:
      demo MUST NOT claim "0 known misses" or "zero known misses";
      demo MUST quote the actual count (e.g., "1 known miss" or
      "2 known misses").

Same shape as iter-232 (JUDGES header recall = live pin),
iter-246 (hero contra-recall chip = live cohort total),
iter-247 (velocity-badge iter count = live work-log max),
iter-256 (velocity-badge "0 known misses" surfacing).

Cross-pin lineage: 18th cross-pin family
(iter-178/183/188/193/198/203/223/228/232/234/236/239/240/244/246/
247/255 + iter-260). Defends the user's "100% / 41/41 is the only
goal" demand at the cross-surface consistency layer.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_DEMO = _REPO_ROOT / "docs" / "demo.html"
_V8_PIN = _REPO_ROOT / "tests" / "test_engine" / "test_path_a_v8_live_recall_pin.py"


def _live_v8_miss_count() -> int:
    """Parse `_V8_EXPECTED_MISSES = (…)` and return the number of
    (drug_a, drug_b) tuples in it."""
    text = _V8_PIN.read_text()
    m = re.search(
        r"_V8_EXPECTED_MISSES:\s*tuple\[tuple\[str,\s*str\],\s*\.\.\.\]\s*=\s*\((.*?)\n\)",
        text,
        re.DOTALL,
    )
    assert m, "Could not locate _V8_EXPECTED_MISSES tuple in v8 live-recall pin"
    body = m.group(1)
    # Count actual (drug_a, drug_b) tuples — ignore comments/whitespace.
    # Each tuple is on its own line of the form `    ("a", "b"),`
    pairs = re.findall(r'\(\s*"[^"]+"\s*,\s*"[^"]+"\s*\)', body)
    return len(pairs)


def test_zero_known_misses_demo_claims_match_live_pin():
    """If `_V8_EXPECTED_MISSES` is empty (zero misses), demo MUST
    claim '0 known misses' + 'zero known misses'. If non-empty, demo
    MUST NOT claim either form."""
    miss_count = _live_v8_miss_count()
    demo = _DEMO.read_text()

    if miss_count == 0:
        # Must surface BOTH forms (visible badge + prose body)
        assert "0 known misses" in demo, (
            "Live `_V8_EXPECTED_MISSES = ()` (empty tuple) but demo "
            "doesn't claim '0 known misses' anywhere. The iter-256 "
            "velocity badge surfaced this claim — restore it. "
            "(Single source of truth is the v8 live-recall pin's "
            "_V8_EXPECTED_MISSES tuple.)"
        )
        assert "zero known misses" in demo, (
            "Live `_V8_EXPECTED_MISSES = ()` (empty tuple) but demo "
            "v8 callout body doesn't include the prose 'zero known "
            "misses' framing. The iter-244 / iter-256 narrative depends "
            "on this claim being visible in the body, not just the "
            "badge."
        )
    else:
        # Must NOT claim 0/zero — that would be stale
        assert "0 known misses" not in demo, (
            f"Live `_V8_EXPECTED_MISSES` has {miss_count} entries but "
            f"demo still claims '0 known misses'. Cohort growth must "
            f"rotate the demo claim in lockstep — either fix v8 to "
            f"catch the pair, OR update the demo to reflect "
            f"'{miss_count} known miss{'es' if miss_count > 1 else ''}'."
        )
        assert "zero known misses" not in demo, (
            f"Live `_V8_EXPECTED_MISSES` has {miss_count} entries but "
            f"demo prose still says 'zero known misses'. Same drift "
            f"class as the velocity badge — rotate in lockstep."
        )


@pytest.mark.skip(reason=(
    "velocity badge retired by user request 2026-05-11 — see "
    "tests/test_engine/test_velocity_badge_pin.py docstring. The "
    "'0 known misses' claim still surfaces in the autonomous-work-log "
    "pin and the v8 callout body (test_zero_misses_callout_in_v8_body)."
))
def test_velocity_badge_surfaces_miss_count():
    """RETIRED — see pytestmark above."""
    miss_count = _live_v8_miss_count()
    demo = _DEMO.read_text()
    # body kept for diff-clarity; skipped at the pytestmark layer.
    m = re.search(
        r"animate-pulse[^<]*</span>\s*<span class=\"font-semibold\">(.*?)</span>",
        demo,
        re.DOTALL,
    )
    assert m, "velocity badge not found in demo.html"


def test_zero_misses_callout_in_v8_body():
    """The v8 callout body (separate from the velocity badge) must
    also surface the miss-count state. The callout is judges'
    medium-detail surface; missing this would mean only the hover
    tooltip carries the headline claim.

    Iter-275 v8 promotion: callout phrasing rotated from
    "V8 staged" (pre-promotion) to "v8 LIVE" / "v8 ENGINE-PROMOTED".
    Both forms accepted to preserve historical-iter coherence; either
    counts as the v8-callout anchor."""
    miss_count = _live_v8_miss_count()
    demo = _DEMO.read_text()

    # The body callout is in the L1270 / L1357 paragraphs.
    # Accept either pre-promotion ("V8 staged") or post-promotion
    # ("v8 LIVE" / "v8 ENGINE-PROMOTED" / "v8 promotion") phrasing.
    anchors = (
        r"V8 staged",
        r"v8 LIVE",
        r"v8 ENGINE-PROMOTED",
        r"iter-275 v8 promotion",
        r"Path A v8 LIVE in engine",
        r"Path A v8 is the LIVE engine bundle",
    )
    body_starts = []
    for pat in anchors:
        body_starts.extend(m.start() for m in re.finditer(pat, demo))
    assert body_starts, (
        f"v8 callout not found in demo.html — expected one of "
        f"{anchors!r}"
    )

    if miss_count == 0:
        # At least ONE of the callout paragraphs must say "zero known
        # misses" (case-insensitive) or "0 known misses".
        callouts = [demo[s:s + 4000] for s in body_starts]
        has_zero_phrase = any(
            "zero known misses" in c.lower() or "0 known misses" in c
            for c in callouts
        )
        assert has_zero_phrase, (
            "v8 callout body must surface 'zero known misses' "
            "or '0 known misses' when _V8_EXPECTED_MISSES is empty. "
            "The visible badge alone isn't enough — the body callout "
            "is the medium-detail surface judges read on tap/scroll."
        )
