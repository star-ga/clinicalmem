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
    "1056",
    "1057",
    "1060",
    "1065",
    "1073",
    "1074",
    "1077",
    "1082",
    "1090",
    "1091",
    "1098",
    "1099",
    "1100",
    "1101",
    "1107",
    "1108",
    "1111",
    "1112",
    "1117",
    "1123",
    "1127",
    "1131",
    "1134",
    "1139",
    "1147",
    "1151",
    "1156",
    "1162",
    "1167",
    "1173",
    "1177",
    "1181",
    "1182",
    "1186",
    "1196",
    "1200",
    "1205",
    "1209",
    "1213",
    "1216",
    "1220",
    # NOTE: "1224" + "1228" intentionally absent — they are recurring
    # live counts (1224 at iter-208/216, 1228 at iter-209/219 after the
    # iter-215 pin cleanup wave brought the count back). Stale-count
    # pin would false-positive on the live README badge otherwise.
    "1234",
    "1242",
    # iter-220 NVIDIA Nemotron Ultra 253B addition: live count went
    # 1226 → 1227 (net +1: replaced test_five_model_consensus with
    # test_six_model_consensus + added test_nvidia_nemotron_included).
    # 1226 is now the previous floor, so it goes here.
    "1226",
    # iter-222 T4 round-46 ratchet on engine/snomed_client.py: live
    # count 1227 → 1231 (+4 — 3 event-presence tests + 1 floor pin).
    # 1227 is the previous floor.
    "1227",
    # iter-223 T1 round-46: structural pin on the 6-LLM consensus
    # provider set (test_consensus_provider_set_pin.py, 6 tests
    # locking labels + env keys + model IDs + NIM endpoint +
    # available_providers tuple shape). Live 1231 → 1237.
    "1231",
    # iter-226 T4 round-47: ratchet engine/what_if.py logger density
    # 24.9 → 35.9/kloc (+4 silent paths closed: simulate_add critical
    # + monitored recommendation_path tracking, simulate_remove
    # resolved + no_change recommendation_path tracking). Live
    # 1237 → 1241.
    "1237",
    # iter-228 T1 round-47: structural pin self-locking the iter-227
    # drift class — test_user_doc_test_path_existence_pin.py (8th
    # cross-pin family): every pin-file path in user-facing docs
    # MUST exist on disk + sanity floor + JUDGES v6 pin citation +
    # self-pin. Live 1241 → 1245.
    "1241",
    # iter-232 T1 round-48: structural pin self-locking the recurring
    # header-vs-body drift class (caught 5x: iter-195/213/218/227/231).
    # test_judges_v6_header_recall_consistency_pin.py (9th cross-pin
    # family) — JUDGES row 102 header recall fraction MUST equal live
    # _V6_CONTRA_HITS/_V6_CONTRA_TOTAL + 4/4 major + 0 FP + Q16.16
    # framing. Live 1245 → 1249.
    "1245",
    # iter-234 T4 round-48: ratchet engine/rxnorm_client.py — rewrite 5
    # old-style %s positional logger calls (which passed drug names
    # directly as record.args, a PHI-leak risk) to structured extra={}
    # form + 4 silent-path closures. New pin file
    # test_rxnorm_logging_pin.py (8 tests with PHI sentinel scrubs +
    # source-level positional-pattern guard). Density 26.9 → 35.6/kloc.
    # Live 1249 → 1257.
    "1249",
    # iter-236 T1 round-49: structural pin family for pin-constants
    # self-consistency across files (11th cross-pin family).
    # test_pin_constants_self_consistency_pin.py (4 tests):
    # _V6_CONTRA_HITS + len(_V6_EXPECTED_MISSES) == _V6_CONTRA_TOTAL,
    # _V6_CONTRA_TOTAL == live cache contra count, README badge ==
    # _TEST_COUNT_FLOOR, README L390 prose == badge. Live 1257 → 1261.
    "1257",
    # iter-239 T4 round-49: ratchet engine/llm_synthesizer.py — 8 PHI-
    # risky positional %s exception logger calls rewritten to structured
    # extra={} form (mirror of iter-234 rxnorm_client refactor) + 2 new
    # aggregate cascade-failure events. New pin file
    # test_llm_synthesizer_logging_pin.py (6 tests, 12th cross-pin
    # family) with source-level positional-pattern + extras-PHI-scrub
    # guards. Live 1261 → 1267.
    "1261",
    # iter-240 T1 round-50: generalized PHI-discipline pin scanning
    # ALL engine/*.py for the iter-234/iter-239 positional-%s pattern.
    # New pin family test_engine_phi_discipline_pin.py (3 tests, 13th
    # cross-pin family) caught 2 more sites on first run: clinical_scoring.py
    # (BITNET_WEIGHTS_TAMPER %s, exc) and openevidence_cache.py (cache
    # malformed: %s, exc). Both rewritten. Live 1267 → 1270.
    "1267",
    # iter-244 T1 round-51: structural pin asserting staged bundle
    # files exist on disk (14th cross-pin family, mirror of iter-228
    # for binary artifacts). test_staged_bundle_existence_pin.py
    # (4 tests): every retrain_runpod/bitnet_weights_*.json path
    # mentioned in user-facing docs / pin files must exist + v6 staged
    # bundle present + shipped engine bundle present + v3+v5 historical
    # bundles preserved. Live 1270 → 1274.
    "1270",
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
_TEST_COUNT_FLOOR = 1274


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
