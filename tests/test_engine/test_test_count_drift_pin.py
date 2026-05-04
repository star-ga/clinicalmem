"""Pin: no stale test-count claims may linger in user-facing docs.

Iteration 36 audit caught five stale "771 tests" mentions across
README.md (badge, capabilities table, CI comment, project structure
caption, "Why ClinicalMem" bullet), plus an "817 tests" mention that
predated the iter-32 cohort growth. Each stale claim is a small but
real promise that no longer matches reality.

This test sweeps a fixed list of user-facing docs for known
historical test counts and fails if any of them reappear. It also
verifies the live count is at least as large as the floor reported
in the live test count (this catches accidental down-bumps).

Pinned floor: 857 (iter 47 standard scope = test_engine + test_scripts).
The floor is "≥ 857" rather than "== 857" so cohort growth or new
pin tests can land without re-bumping multiple docs each time.
"""
from __future__ import annotations

import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent

_USER_FACING_DOCS = (
    _REPO_ROOT / "README.md",
    _REPO_ROOT / "JUDGES.md",
    _REPO_ROOT / "docs" / "demo.html",
    _REPO_ROOT / "docs" / "architecture.md",
)

# Historical test counts that have appeared in user-facing docs and
# are no longer accurate. As the count grows, append the previous
# floor here when bumping; this list catches half-completed rotations.
_HISTORICAL_COUNTS = (
    "429",
    "623",
    "771",
    "802",
    "811",
    "814",
    "817",
    "820",
    "822",
    "826",
    "843",
    "846",
    "848",
    "854",
    "861",
    "867",
    "869",
    "877",
    "885",
    "888",
    "893",
    "896",
    "898",
    "904",
    "910",
    "913",
    "917",
    "921",
    "932",
    "940",
    "942",
    "948",
    "954",
    "959",
    "960",
    "964",
    "970",
    "972",
    "973",
    "977",
    "980",
    "984",
    "989",
    "996",
    "1000",
    "1001",
    "1005",
    "1011",
    "1014",
    "1016",
    "1019",
    "1022",
    "1025",
    "1029",
    "1033",
    "1034",
    "1038",
    "1042",
    "1046",
    "1047",
    "1050",
)

# The "100% line coverage" claim was unverified (the loop's standard
# scope doesn't measure coverage). It must not reappear in marketing
# copy until a coverage-gate CI step exists and the number is real.
_FORBIDDEN_COVERAGE_CLAIMS = (
    "100% coverage",
    "100% line coverage",
    "100%25%20coverage",  # URL-encoded form (in shields.io badges)
)

# Pinned floor — the loop's standard scope (engine + scripts) must
# stay at or above this many tests. Bump when adding new pins.
_TEST_COUNT_FLOOR = 1056


def test_no_stale_test_counts_in_docs():
    for path in _USER_FACING_DOCS:
        if not path.exists():
            continue
        text = path.read_text()
        for stale in _HISTORICAL_COUNTS:
            # Look for the count followed by "tests" or " passed" or "%20passed"
            for pattern in (
                rf"\b{stale}\s+tests?\b",
                rf"\b{stale}\s+passed\b",
                rf"-{stale}%20passed",
                rf"-{stale}%20tests",
            ):
                m = re.search(pattern, text)
                assert m is None, (
                    f"Stale test count {stale!r} still in "
                    f"{path.relative_to(_REPO_ROOT)} "
                    f"(matched {pattern!r}). Update to the live count."
                )


def test_no_unverified_coverage_claims():
    for path in _USER_FACING_DOCS:
        if not path.exists():
            continue
        text = path.read_text()
        for claim in _FORBIDDEN_COVERAGE_CLAIMS:
            assert claim not in text, (
                f"Unverified coverage claim {claim!r} in "
                f"{path.relative_to(_REPO_ROOT)}. Either drop the claim "
                f"or wire `pytest --cov` into CI and report the real number."
            )


def test_live_test_count_at_or_above_floor():
    """The live engine + scripts count must stay at or above the floor."""
    import subprocess
    import sys

    cp = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            str(_REPO_ROOT / "tests" / "test_engine"),
            str(_REPO_ROOT / "tests" / "test_scripts"),
            "--collect-only",
            "-q",
        ],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=str(_REPO_ROOT),
    )
    assert cp.returncode == 0, f"pytest --collect-only failed:\n{cp.stderr}"
    # Last numeric line of stdout has the count, e.g. "843 tests collected in 0.4s"
    m = re.search(r"(\d+)\s+tests?\s+collected", cp.stdout)
    assert m is not None, f"Could not parse test count from:\n{cp.stdout[-300:]}"
    live = int(m.group(1))
    assert live >= _TEST_COUNT_FLOOR, (
        f"Live test count regressed: {live} < floor {_TEST_COUNT_FLOOR}. "
        f"If this is intentional (test removal), lower the floor in this file."
    )
