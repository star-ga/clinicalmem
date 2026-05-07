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
    # iter-245 v6 → v8 swap (h=128 → h=256 architectural double broke
    # v7 BOOST_KEYS @200x ceiling; sweep seed=71 hit 41/41 contra +
    # 4/4 major + 0 FP at bundle 1f0f8859…). Surface rotated:
    # test_path_a_v6_* (-2 files, 12 tests) replaced by
    # test_path_a_v8_* (+2 files, 14 tests = 6 live-recall + 8 q16
    # determinism); test_staged_bundle_existence_pin grew 4 → 5
    # (added v6-historical-bundle assertion since v6 is no longer
    # the active staged surface but stays on disk for FDA SaMD
    # audit-trail rigor). Net +1 test. Live 1274 → 1275.
    "1274",
    # iter-246 T2 round-52: demo hero stat-chip + trust-bar drift
    # caught at iter-246 audit — chip 1 still read "35 / 35 · NTI
    # cohort" (frozen iter-202, 80+ iters behind), trust-bar still
    # read "1161 / 1161 engine tests" (frozen iter-148-era). Same
    # iter-232 header-drift class but at a higher-visibility surface
    # (judges see hero in 5 sec). New pin family
    # test_demo_hero_stat_consistency_pin.py (4 tests, **15th cross-
    # pin family**): hero contra-recall fraction reads from live
    # _V8_CONTRA_TOTAL; hero MUST mention v8 (surface the iter-244
    # breakthrough); trust-bar test count == _TEST_COUNT_FLOOR;
    # forbidden pre-iter-235 cohort sizes (35-40) blocked from
    # chip subtitle. Live 1275 → 1279.
    "1275",
    # iter-247 T3 round-52: velocity-badge drift (45+ iters stale)
    # — visible badge said "200+ iter" while live work-log was at
    # iter-246; tooltip even more stale at "160+ autonomous
    # improvement cycles". Plus a half-completed v6→v8 rotation in
    # demo.html L1270 — the v8 pin description still said
    # "aggregate: bundle_id + 40/41 + 4/4 + 0 FP" when v8 actually
    # hit 41/41 (this is the FULL-recall breakthrough — calling it
    # 40/41 in the description undersells the architectural double).
    # Both fixed in lockstep: badge → "245+ iter", description →
    # "41/41". New pin family test_velocity_badge_iter_count_pin.py
    # (3 tests, **16th cross-pin family**): badge claims must not
    # lag live work-log highest iter by >50; claims must round to
    # multiples of 5; claims must not exceed live max. Single source
    # of truth = AUTONOMOUS_WORK_LOG.md highest "| <N> |" row. Same
    # shape as iter-232/246 (single source → derived surface).
    # Live 1279 → 1282.
    "1279",
    # iter-255 T1 round-54: v8 staged bundle integrity gap. Shipped
    # cfadb4f6 has 9-test integrity pin (size band, sparsity floor,
    # key set, _meta provenance, self-referenced bundle_id) but the
    # iter-244 staged 1f0f8859 v8 bundle had no equivalent. Future
    # corruption that doesn't change bundle_id (stray top-level key,
    # _meta provenance field drop, JSON pretty-print drift, sparsity
    # collapse) would silently break the 0%-known-misses promise.
    # New pin family test_v8_bundle_integrity_pin.py (9 tests, **17th
    # cross-pin family**): bundle_id == pinned 1f0f88591…; file size
    # 110-130 KB band + 200 KB hard ceiling; ternary sparsity 40-60%;
    # top-level keys exactly 5-element canonical set; _meta carries
    # all 17 FDA SaMD provenance fields; _meta.bundle_id self-
    # consistent (SHA-256 of canonical-form weight payload); contra_fp
    # = 0; contra_recall = 1.0; architecture dims (in=193, hidden=256,
    # out=5) match weight matrix shapes. Mirror of iter-72 cfadb4f6
    # pin applied to v8 staged surface. Defends user's "100% is the
    # only goal" demand at the bundle-content layer. Live 1282 → 1291.
    "1282",
    # iter-260 T1 round-55: cross-surface "0 known misses" claim
    # consistency. iter-256 surfaced "0 known misses" in the velocity
    # badge + body callout, but no pin tied that claim back to the
    # live `_V8_EXPECTED_MISSES = ()` empty-tuple invariant. If a
    # future cohort growth adds a v8-miss pair, _V8_EXPECTED_MISSES
    # gains entries but the demo claim could silently stay at "0
    # known misses". New pin family test_zero_known_misses_consistency_pin.py
    # (3 tests, **18th cross-pin family**): demo "0 known misses" /
    # "zero known misses" claims iff len(_V8_EXPECTED_MISSES) == 0;
    # if non-zero, demo MUST quote the actual count. Same shape as
    # iter-232/iter-246/iter-247 (single source → derived surface).
    # Live 1291 → 1294.
    "1291",
    # iter-265 T1 round-56: extended iter-255 v8 bundle integrity pin
    # with EXACT VALUE assertions on safety-critical _meta provenance
    # fields (was just existence checks). New test added to existing
    # pin file: schema == "bitnet_classifier_v3_atc_flags",
    # training_iter == "iter-242-path-a-v8-h256", weight_dtype ==
    # "ternary", bias_dtype == "q16.16", flag_keys_count == 26,
    # pair_derived_rule_count == 13. Catches drift where someone
    # hand-edits _meta values without rotating bundle_id (which is
    # hashed only over weight matrices, not _meta). Same iter-117
    # ratchet pattern (tighten existing pin once headroom exists).
    # Live 1294 → 1295.
    "1294",
    # iter-268 T4 round-56: forward-protective `no print() in engine`
    # pin extending iter-240 PHI-discipline file (4 tests now). Engine
    # is clean (0 violations); pin is pre-emptive — a future debug
    # `print(f"...")` breadcrumb in engine code would bypass log
    # filters/aggregators/level-controls + the iter-240 PHI extras-
    # key scrub. Live 1295 → 1296.
    "1295",
    # iter-269 T5 round-56: cohort-defense ratchet — `source` field
    # VALUE vocabulary lock added to iter-94 cache_shape pin (was
    # only existence-checked, 7 → 8 tests). Live 138/138 = "CACHED";
    # _VALID_SOURCES = frozenset({"CACHED"}). Future debug values
    # ("API_LIVE", "STAGING") would silently skew downstream metrics
    # without this lock. Live 1296 → 1297.
    "1296",
    # iter-270 T1 round-57: orphan-flag-drug allowlist bound on
    # pharmacology_flags.json (10 tests now). 126 flag entries, 124
    # cache-referenced; 2 intentional orphans (ketorolac NSAID pre-
    # staging, tranylcypromine MAOI pre-staging at iter-264) locked
    # to _ALLOWED_ORPHAN_DRUGS whitelist; soft floor ≤5 total
    # orphans signals cohort-cadence vs flag-table lag. Live 1297 → 1298.
    "1297",
    # iter-273 T4 round-57: extended iter-268 print-prohibition pin
    # to also catch sys.stdout.write / sys.stderr.write / warnings.warn
    # (broader emission-bypass surface). Engine clean (0 violations
    # across all 3 patterns); pin is pre-emptive — these 3 paths
    # bypass structured logging identically to print() but escape
    # the iter-268 regex. 4 → 5 tests in test_engine_phi_discipline_pin.
    # Live 1298 → 1299.
    "1298",
    # iter-274 T5 round-57: clinical_summary upper bound (≤ 2000 chars)
    # complements iter-259 lower bound (≥ 400). Live max=1347, p99=1257,
    # 0/138 entries > 1500. test_cache_shape_invariants_pin.py 8 → 9
    # tests. Live 1299 → 1300.
    "1299",
    # iter-279 T4 round-58: 19th cross-pin family — bitnet_features_v8
    # logging pin (5 tests). Engine module purity preserved via one-shot
    # latch on first encode_pair_v8 call (instead of module-top-level
    # debug). Live 1300 → 1305.
    "1300",
    # iter-281 T1 round-59: 20th cross-pin family — bundle_id cross-file
    # consistency pin (5 tests). Forward-protects against the iter-275
    # cascade drift class (iter-278 caught 6 stale v1 references on
    # judge-visible surfaces post-promotion). Live 1305 → 1310.
    "1305",
    # iter-284 T4 round-59: openevidence_cache.py logger ratchet
    # 18.2/kloc → 41/kloc (the lowest-density engine module). 4 new
    # success-path events (cache load + hit + miss + invalidate) + PHI-
    # safe _hash_pair helper for drug-name hashing. NEW pin family
    # test_openevidence_cache_logging_pin.py (6 tests). Live 1310 → 1316.
    "1310",
    # iter-285 T5 round-59: 21st cross-pin family — every contra cache
    # entry MUST cite ≥ 1 URL from the authoritative whitelist (FDA /
    # EMA / ACR / PubMed / AHA Journals / BMJ / Oxford Academic / AGS
    # Beers / JAAD / AAN). Catches cohort-growth events that introduce
    # a contra backed only by secondary review sources. 4 tests:
    # per-entry coverage; whitelist breadth ≥ 10 hosts; every host in
    # whitelist is live-used (or in _ALLOWED_UNUSED_HOSTS); JUDGES
    # cites the pin. Live 1316 → 1320.
    "1316",
    # iter-286 T1 round-60: 22nd cross-pin family — v1 backup bundle
    # replayability. iter-281 pinned the v1 backup file EXISTS;
    # iter-286 pins it's FUNCTIONAL via the live engine loader: schema
    # = bitnet_classifier_v1, dims=128/64/5, bundle_id starts with
    # cfadb4f6, warfarin+ibuprofen → severity='major' + repro_hash
    # starts with bdaf385a (iter-26 historical anchor), v1 vs v8
    # repro_hashes differ (catches accidental v8-overwrite-of-v1),
    # forward pass deterministic, _meta.bundle_id self-reference
    # matches canonical SHA. 6 tests. Live 1320 → 1326.
    "1320",
    # iter-289 T4 round-60: structured logging on
    # engine.clinical_scoring.confidence_gate (the abstention gate that
    # decides answer-vs-refuse). 3 new events: INFO no_records, WARNING
    # abstained, DEBUG pass. PHI-safe: scalar metrics only. 5 tests
    # (one per branch + PHI scrub + source-level ≥3 logger.* guard).
    # Live 1326 → 1331.
    "1326",
    # iter-295 T5 round-61: cohort-defense ratchet — every contra cache
    # entry's clinical_summary MUST contain ≥ 1 citation marker (FDA
    # label / PMID-DOI / major journal / guideline body / mechanism
    # token). 3 tests: per-entry coverage; ≥ 3 marker classes used
    # collectively; ≥ 80% cite FDA OR primary peer-reviewed source.
    # Same iter-285 shape applied to summary text instead of URL
    # layer. Live 1331 → 1334.
    "1331",
    # iter-296 T1 round-62: runtime regression test for the iter-291
    # PHI fix in BITNET_SAFETY_DOWNGRADE_DISAGREEMENT. iter-240 source
    # scan + iter-291 regex extension catch the leaking PATTERN; this
    # pin tests the LIVE log record shape via a mocked classifier
    # (forces the disagreement path which v8 100%-recall makes
    # otherwise unreachable). 3 tests: structured-extras-no-raw-names,
    # canonical lex-sorted pair hash, source-level no-legacy-%s guard.
    # Live 1334 → 1337.
    "1334",
    # iter-301 T1 round-63: 25th cross-pin family —
    # test_demo_pin_description_recall_consistency_pin.py (4 tests).
    # Same iter-232/iter-298 single-source-of-truth → derived-surface
    # drift class as the JUDGES header pin, but applied to inline pin-
    # description text inside docs/demo.html. Caught at iter-301 audit:
    # 3 stale 43/43 + 1 stale "42 contraindicated" survived the
    # iter-280 cohort growth from 43 → 44. Pin locks the aggregate
    # phrase ("aggregate: bundle_id + N/N + 4/4 + 0 FP +
    # strictly_supersedes"), the iter-275 promotion phrase ("bundle_id
    # + N/N contra + 4/4 major + 0 FP + meta-block"), the cohort
    # breakdown ("139-pair recall cohort (X contraindicated · Y
    # major · Z moderate · W serious)"), and forbids pre-iter-280 4[23]
    # contra fractions from re-appearing. Live 1337 → 1341.
    "1337",
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
_TEST_COUNT_FLOOR = 1341


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
