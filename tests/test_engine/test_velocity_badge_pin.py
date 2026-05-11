"""Pin: the active-improvement velocity badge in docs/demo.html must
NOT carry a hardcoded `iter N` count that will drift on every cron
fire.

RETIRED 2026-05-11: the velocity badge was intentionally removed from
the demo as part of the compaction pass — the user explicitly asked
that the loop-cadence surface not be visible in the hero. The drift
class this pin defended (hardcoded iter-N going stale at the 15-min
cron cadence) is moot when the surface itself is gone. All three tests
in this module pytest.skip with a pointer to this docstring; remove
the entire file once the next cleanup sweep retires the file.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_DEMO = _REPO_ROOT / "docs" / "demo.html"

_RETIRED_REASON = (
    "velocity badge retired by user request 2026-05-11 — drift class moot"
)
pytestmark = pytest.mark.skip(reason=_RETIRED_REASON)


def _badge_text() -> str:
    """Extract the velocity-badge inner text from docs/demo.html."""
    text = _DEMO.read_text()
    # The badge has an animate-pulse green dot adjacent to a font-semibold span
    # naming the round-robin. Match the surrounding container.
    m = re.search(
        r"animate-pulse[^<]*</span>\s*<span class=\"font-semibold\">(.*?)</span>",
        text,
        re.S,
    )
    assert m, "velocity badge not found in docs/demo.html — did the structure change?"
    return m.group(1)


def test_badge_uses_durable_plus_form_not_hardcoded_iter():
    """The badge must use `<N>+ iter` form, not `iter N`.

    Iter-159 fix: hardcoded `iter 158` is OK on the day it ships, but
    by iter-160 (~ 4 hours later at 15-min cron fires) it's already
    stale. The `+` suffix lets the surface stay honest until the next
    explicit surface sweep.
    """
    text = _badge_text()
    # Forbid the bare `iter \d+ \W` form (no `+` after the number)
    bad = re.search(r"\biter\s+(\d+)\s*[·.]", text)
    assert bad is None, (
        f"velocity badge has hardcoded iter number {bad.group(1)!r} that "
        f"will drift as the cron loop fires every 15 min. Use the "
        f"`<N>+ iter` form (e.g. `160+ iter`) to make the badge durable."
    )
    # Require the `+` form
    assert re.search(r"\b\d+\+\s*iter\b", text), (
        f"velocity badge text {text!r} must contain a `<N>+ iter` "
        f"durable count (e.g. `160+ iter`)."
    )


def test_badge_names_round_robin():
    """The badge must surface the 5-tier round-robin so a judge sees
    that improvement is structured, not random."""
    text = _badge_text()
    assert re.search(r"5[- ]tier\s+round[- ]robin", text), (
        f"velocity badge must name the 5-tier round-robin "
        f"(found: {text!r})"
    )


def test_badge_links_to_work_log():
    """The badge must be a clickable anchor to AUTONOMOUS_WORK_LOG.md
    so a curious judge can audit every commit's iter row."""
    text = _DEMO.read_text()
    # Find any anchor with the work-log href
    assert re.search(
        r'<a href="https://github\.com/star-ga/clinicalmem/blob/main/AUTONOMOUS_WORK_LOG\.md"',
        text,
    ), (
        "velocity badge must wrap an anchor to AUTONOMOUS_WORK_LOG.md so "
        "judges can audit the per-iter cadence."
    )
