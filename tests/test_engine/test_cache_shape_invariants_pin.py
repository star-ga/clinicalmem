"""Pin the load-bearing shape invariants of `docs/openevidence_cache.json`.

Existing test suites cover **engine API behaviour** on top of the cache
(canonical-pair lookup, fallback semantics) and **evidence-URL hygiene**
(HTTPS, host+path, density, authoritative whitelist). What they do NOT
pin is the cache file's own structural integrity — the kind of thing a
typo or a half-finished merge can silently break:

  - severity ∈ a closed vocabulary (none / minor / moderate / serious /
    major / contraindicated). A bare typo like ``"sereious"`` would slip
    past every existing test, but break ``run_clinical_regression_eval``
    silently.
  - drug_pair_canonical == sorted([drug_a, drug_b]). Required for stable
    cache lookup; a manually-edited entry that violates this would cause
    cache misses on the very pair it was supposed to add.
  - No duplicate (drug_a, drug_b) pairs. A duplicate would inflate the
    per-class count and skew every dashboard claim.
  - clinical_summary length ≥ 400 chars (ratcheted from 100 at iter-259
    once the cohort matured to 138 entries with min length 470, p10
    545, median 621 — the original floor was set when the cache was
    young; tightening it now forces any new stub-length entry to fail
    the gate visibly). A one-line stub is not a citation; the
    dashboard's "evidence-backed" pitch breaks.
  - retrieved_at is ISO-8601 ``YYYY-MM-DD``. A free-form date breaks
    the ``audit_replay_pins`` provenance pin.
  - All 8 required fields present on every entry.

T1 round 19 (iter 94, LLM-eval infra still blocked since iter 5):
substitute work — instead of running an external LLM eval, lock the
**shape** of the data the eval depends on. Future cache growth that
violates any invariant fails the gate immediately.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_CACHE = _REPO_ROOT / "docs" / "openevidence_cache.json"

_VALID_SEVERITIES = frozenset({
    "none", "minor", "moderate", "serious", "major", "contraindicated",
})
_REQUIRED_FIELDS = (
    "drug_a", "drug_b", "drug_pair_canonical", "severity",
    "clinical_summary", "evidence_urls", "retrieved_at", "source",
)
_MIN_SUMMARY_LEN = 400  # iter-259 ratchet — live min is 470 (azathioprine+allopurinol),
                          # p10 = 545, median = 621. The original 100-char floor was set
                          # at iter-66 when the cache was small; with 138 cohort entries
                          # the discipline has matured well past stub-length. Ratcheting
                          # to 400 leaves ~70 char headroom for future entries while
                          # forcing any new entry that approaches stub-length to fail
                          # the gate visibly (the iter-117 ratchet pattern: tighten
                          # invariants once enough headroom exists).
_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _cache():
    return json.loads(_CACHE.read_text())


def test_every_entry_has_required_fields():
    """Every cache entry must have the 8 canonical fields."""
    missing = []
    for i, it in enumerate(_cache()):
        miss = [f for f in _REQUIRED_FIELDS if f not in it]
        if miss:
            pair = (it.get("drug_a", "?"), it.get("drug_b", "?"))
            missing.append((i, pair, miss))
    assert not missing, (
        f"{len(missing)} cache entries missing required fields. "
        f"First: index={missing[0][0]} pair={missing[0][1]} "
        f"missing={missing[0][2]}"
    )


def test_every_severity_is_in_canonical_vocabulary():
    """A typo (e.g. 'sereious') would be silently accepted by the
    classifier comparison until someone audits the cache by hand. Pin
    the closed set."""
    bad = []
    for i, it in enumerate(_cache()):
        sev = it.get("severity")
        if sev not in _VALID_SEVERITIES:
            bad.append((i, (it.get("drug_a"), it.get("drug_b")), sev))
    assert not bad, (
        f"{len(bad)} cache entries have non-canonical severity. "
        f"Allowed: {sorted(_VALID_SEVERITIES)}. "
        f"First offender: index={bad[0][0]} pair={bad[0][1]} severity={bad[0][2]!r}"
    )


def test_drug_pair_canonical_is_alphabetical_sort_of_drug_pair():
    """drug_pair_canonical is the lookup key. It MUST equal the
    alphabetical sort of [drug_a, drug_b] — otherwise the engine's
    canonical lookup misses the entry it just added."""
    bad = []
    for i, it in enumerate(_cache()):
        da, db = it.get("drug_a", ""), it.get("drug_b", "")
        canon = it.get("drug_pair_canonical", [])
        expected = sorted([da, db])
        if list(canon) != expected:
            bad.append((i, (da, db), canon, expected))
    assert not bad, (
        f"{len(bad)} cache entries have drug_pair_canonical that "
        f"doesn't match sorted([drug_a, drug_b]). "
        f"First offender: index={bad[0][0]} pair={bad[0][1]} "
        f"canonical={bad[0][2]} expected={bad[0][3]}"
    )


def test_no_duplicate_drug_pairs():
    """Two entries for the same (drug_a, drug_b) — even with different
    severities — would inflate per-class counts and skew every dashboard
    claim."""
    counts: Counter = Counter()
    for it in _cache():
        key = tuple(sorted([it.get("drug_a", ""), it.get("drug_b", "")]))
        counts[key] += 1
    dups = {k: c for k, c in counts.items() if c > 1}
    assert not dups, (
        f"{len(dups)} duplicate drug pairs in cache. "
        f"First duplicate: {next(iter(dups.items()))}"
    )


def test_every_clinical_summary_is_at_least_100_chars():
    """The dashboard pitches 'every drug pair has an evidence-backed
    clinical summary'. A one-line stub doesn't qualify."""
    short = []
    for i, it in enumerate(_cache()):
        summary = it.get("clinical_summary", "")
        if len(summary) < _MIN_SUMMARY_LEN:
            short.append((i, (it.get("drug_a"), it.get("drug_b")), len(summary)))
    assert not short, (
        f"{len(short)} cache entries have clinical_summary < {_MIN_SUMMARY_LEN} "
        f"chars. First: index={short[0][0]} pair={short[0][1]} "
        f"length={short[0][2]}"
    )


def test_every_retrieved_at_is_iso_8601_date():
    """retrieved_at must be ISO 8601 (YYYY-MM-DD). The audit-replay
    provenance pin parses this; a free-form date breaks it."""
    bad = []
    for i, it in enumerate(_cache()):
        ts = it.get("retrieved_at", "")
        if not _ISO_DATE_RE.match(ts):
            bad.append((i, (it.get("drug_a"), it.get("drug_b")), ts))
    assert not bad, (
        f"{len(bad)} cache entries have non-ISO-8601 retrieved_at. "
        f"First: index={bad[0][0]} pair={bad[0][1]} retrieved_at={bad[0][2]!r}"
    )


def test_per_class_count_floor():
    """Soft floor: cache must always have at least 1 entry per safety
    class we publicly claim coverage on. If a future restructure drops
    the moderate class to 0, the dashboard's recall claim becomes
    vacuous (a 0/0 short-circuit) and this test fires before that
    reaches a judge."""
    counts = Counter(it.get("severity") for it in _cache())
    for cls in ("contraindicated", "major", "serious", "moderate"):
        assert counts.get(cls, 0) >= 1, (
            f"Safety class {cls!r} has 0 entries in cache. "
            f"Dashboard claim '100% recall on {cls}' becomes vacuous."
        )
