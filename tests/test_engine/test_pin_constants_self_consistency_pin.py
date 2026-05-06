"""Pin: constants across cross-pin files must self-consistent.

Iter 236 (round 49 T1) — closes a drift class where a cohort-growth
ripple bumps one constant but forgets a sibling. The recurring shape:

  • iter-225 attempt: bumped _V6_CONTRA_TOTAL but forgot to compute
    _V6_CONTRA_HITS correctly → had to roll back
  • iter-235: bumped _TEST_COUNT_FLOOR but had to manually update
    README badge in lockstep (no pin caught this)

This pin asserts:

  1. **v6 constants self-consistency**: `_V6_CONTRA_HITS +
     len(_V6_EXPECTED_MISSES) == _V6_CONTRA_TOTAL` — if any one of the
     three drifts without the other two updating in lockstep, this fires.

  2. **v6 ↔ cache cohort coherence**: `_V6_CONTRA_TOTAL` must equal
     `len([entries where severity == 'contraindicated'])` in the live
     cache. Catches the case where pin constants drift from the
     openevidence_cache.json that's regenerated independently.

  3. **README test-count consistency**: the README badge (`tests-NNNN%20`)
     and the L390-area prose ("NNNN tests") must both match
     `_TEST_COUNT_FLOOR`. Catches the iter-235-style drift where
     I bumped the floor but had to remember the badge separately.

  4. **README L390 prose ↔ badge consistency**: even within README, the
     two surfaces must agree. Catches half-completed rotations.

10th cross-pin family in the discipline lineage (after iter-178/183/188
/193/198/203/223/228/232/234).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_V6_PIN = _REPO_ROOT / "tests" / "test_engine" / "test_path_a_v6_live_recall_pin.py"
_DRIFT_PIN = _REPO_ROOT / "tests" / "test_engine" / "test_test_count_drift_pin.py"
_README = _REPO_ROOT / "README.md"
_CACHE = _REPO_ROOT / "docs" / "openevidence_cache.json"


def _read_int_constant(pin_path: Path, var_name: str) -> int:
    text = pin_path.read_text()
    m = re.search(rf"^{var_name}\s*=\s*(\d+)", text, re.MULTILINE)
    assert m is not None, (
        f"Could not find {var_name} in {pin_path.relative_to(_REPO_ROOT)}. "
        f"This pin assumes {var_name} is a top-level int constant."
    )
    return int(m.group(1))


def _v6_expected_misses_count() -> int:
    """Parse the _V6_EXPECTED_MISSES tuple length without importing
    (since import would re-run the whole pin file)."""
    text = _V6_PIN.read_text()
    # Find the _V6_EXPECTED_MISSES assignment + count tuples inside.
    m = re.search(
        r"_V6_EXPECTED_MISSES.*?=\s*\((.*?)^\)",
        text,
        re.DOTALL | re.MULTILINE,
    )
    assert m is not None, (
        "Could not find _V6_EXPECTED_MISSES tuple in v6 live-recall pin. "
        "This pin assumes the tuple is a multi-line top-level constant."
    )
    body = m.group(1)
    # Each miss is a tuple literal like `("ketoconazole", "lurasidone"),`
    # — count parenthesized 2-string tuples.
    misses = re.findall(r"\(\s*\"[^\"]+\"\s*,\s*\"[^\"]+\"\s*\)", body)
    return len(misses)


def test_v6_constants_self_consistency():
    """`_V6_CONTRA_HITS + len(_V6_EXPECTED_MISSES) == _V6_CONTRA_TOTAL`.

    If a cohort-growth iter bumps any one of the three without the
    other two updating in lockstep, this fires at commit time.
    """
    hits = _read_int_constant(_V6_PIN, "_V6_CONTRA_HITS")
    total = _read_int_constant(_V6_PIN, "_V6_CONTRA_TOTAL")
    misses = _v6_expected_misses_count()

    assert hits + misses == total, (
        f"v6 constants drift detected:\n"
        f"  _V6_CONTRA_HITS = {hits}\n"
        f"  len(_V6_EXPECTED_MISSES) = {misses}\n"
        f"  _V6_CONTRA_TOTAL = {total}\n"
        f"  hits + misses = {hits + misses} != total {total}\n"
        f"Cohort growth must update HITS / MISSES / TOTAL in lockstep:\n"
        f"  • If v6 catches the new pair: HITS += 1, TOTAL += 1\n"
        f"  • If v6 misses the new pair: TOTAL += 1, MISSES += 1 entry\n"
        f"Anything else is silent drift between the three constants."
    )


def test_v6_total_matches_live_cache_cohort():
    """`_V6_CONTRA_TOTAL` must equal the live count of contraindicated
    entries in `docs/openevidence_cache.json`. Catches the case where
    pin constants drift from the regenerated cache."""
    total = _read_int_constant(_V6_PIN, "_V6_CONTRA_TOTAL")
    cache = json.loads(_CACHE.read_text())
    live_total = sum(1 for e in cache if e.get("severity") == "contraindicated")
    assert total == live_total, (
        f"v6 cohort drift: _V6_CONTRA_TOTAL={total}, but the live cache "
        f"has {live_total} contraindicated entries. The pin constant must "
        f"update in lockstep with cache regen on every cohort-growth event."
    )


def test_readme_badge_matches_test_count_floor():
    """The README badge `tests-NNNN%20passed` integer must match
    `_TEST_COUNT_FLOOR` from test_test_count_drift_pin.py. Catches
    iter-235-style drift where the floor is bumped but the README badge
    is missed (or vice versa)."""
    floor = _read_int_constant(_DRIFT_PIN, "_TEST_COUNT_FLOOR")
    readme = _README.read_text()
    # Badge format: `tests-NNNN%20passed` (URL-encoded)
    m = re.search(r"tests-(\d+)%20passed", readme)
    assert m is not None, (
        f"Could not find `tests-NNNN%20passed` badge in README.md. "
        f"This pin assumes the shields.io badge is present."
    )
    badge_count = int(m.group(1))
    assert badge_count == floor, (
        f"README badge / test-count floor drift:\n"
        f"  README.md badge: tests-{badge_count}%20passed\n"
        f"  _TEST_COUNT_FLOOR: {floor}\n"
        f"Update both in lockstep — bumping the floor without bumping the "
        f"badge means the README claims fewer tests than the live count."
    )


def test_readme_prose_test_count_matches_badge():
    """The README L390-area prose ("NNNN tests across ...") must match
    the badge integer. Catches half-completed rotations within the same
    file."""
    floor = _read_int_constant(_DRIFT_PIN, "_TEST_COUNT_FLOOR")
    readme = _README.read_text()
    # Prose format: "**NNNN tests** across `tests/test_engine/`..."
    m = re.search(
        r"\*\*(\d+)\s+tests\*\*\s+across\s+`tests/test_engine/`",
        readme,
    )
    assert m is not None, (
        "Could not find the README L390 prose `**NNNN tests** across "
        "`tests/test_engine/``. This pin assumes that template phrasing."
    )
    prose_count = int(m.group(1))
    assert prose_count == floor, (
        f"README badge ↔ prose drift:\n"
        f"  README.md L390 prose: {prose_count} tests\n"
        f"  _TEST_COUNT_FLOOR: {floor}\n"
        f"Update both in lockstep — half-completed rotations leave the "
        f"prose stale even after the badge is bumped."
    )
