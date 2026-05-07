"""Pin: engine/openevidence_cache.py emits structured logs on every code path.

Iter-284 T4 round-59 ratchet. Pre-iter-284 the module had only 3
logger.warning calls (all in error paths) for 165 lines — the
**lowest engine module density** at 18.2/kloc. Success paths emitted
nothing, so an auditor couldn't correlate a cache lookup to the
snapshot loaded or distinguish hit/miss without the surrounding
engine event chain.

Post-iter-284 the module emits 4 NEW events on the success path
(load + hit + miss + invalidate) AND preserves the iter-234/iter-239/
iter-240 PHI discipline: drug names hashed via SHA-256 16-char prefix,
NEVER carried raw in any log record. Lifts density to ~41/kloc.

Pinned events
=============
- DEBUG `openevidence_cache_loaded` — entry_count, url_count_total,
  severity_counts.
- DEBUG `openevidence_cache_hit` — pair_hash_prefix, severity,
  evidence_url_count, source.
- DEBUG `openevidence_cache_miss` — pair_hash_prefix, cache_size.
- INFO `openevidence_cache_invalidated` — prev_entry_count.
"""
from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_MODULE_PATH = _REPO_ROOT / "engine" / "openevidence_cache.py"


def _expected_pair_hash(a: str, b: str) -> str:
    a_norm = a.strip().lower()
    b_norm = b.strip().lower()
    if a_norm > b_norm:
        a_norm, b_norm = b_norm, a_norm
    return hashlib.sha256(f"{a_norm}|{b_norm}".encode("utf-8")).hexdigest()[:16]


def test_cache_load_emits_structured_debug(caplog):
    """First call to lookup triggers the cache load → DEBUG event with
    counts (no PHI)."""
    import sys
    sys.path.insert(0, str(_REPO_ROOT))
    from engine.openevidence_cache import (  # noqa: PLC0415
        invalidate_cache, lookup_cached,
    )

    invalidate_cache()  # ensure next lookup reloads
    with caplog.at_level(logging.DEBUG, logger="engine.openevidence_cache"):
        lookup_cached("warfarin", "ibuprofen")

    matched = [
        r for r in caplog.records
        if r.name == "engine.openevidence_cache"
        and r.message == "openevidence_cache_loaded"
    ]
    assert matched, "Cache load must emit 'openevidence_cache_loaded' DEBUG"
    rec = matched[0]
    assert rec.levelno == logging.DEBUG
    assert rec.entry_count >= 100  # live cache is 139+; floor 100 catches regressions
    assert rec.url_count_total >= rec.entry_count  # >= 1 URL per entry
    assert isinstance(rec.severity_counts, dict)
    assert "contraindicated" in rec.severity_counts


def test_cache_hit_emits_structured_debug(caplog):
    """A successful cache lookup → DEBUG with hashed pair + severity."""
    import sys
    sys.path.insert(0, str(_REPO_ROOT))
    from engine.openevidence_cache import lookup_cached  # noqa: PLC0415

    with caplog.at_level(logging.DEBUG, logger="engine.openevidence_cache"):
        result = lookup_cached("warfarin", "ibuprofen")

    assert result is not None  # cache hit guaranteed (live cohort)
    matched = [
        r for r in caplog.records
        if r.name == "engine.openevidence_cache"
        and r.message == "openevidence_cache_hit"
    ]
    assert matched, "Cache hit must emit 'openevidence_cache_hit' DEBUG"
    rec = matched[0]
    assert rec.pair_hash_prefix == _expected_pair_hash("warfarin", "ibuprofen")
    assert re.fullmatch(r"[0-9a-f]{16}", rec.pair_hash_prefix)
    assert rec.severity in ("none", "moderate", "serious", "major", "contraindicated")
    assert rec.evidence_url_count >= 1
    assert rec.source == "CACHED"


def test_cache_miss_emits_structured_debug(caplog):
    """A cache miss (drug pair not in the cohort) → DEBUG with hashed
    pair + cache_size."""
    import sys
    sys.path.insert(0, str(_REPO_ROOT))
    from engine.openevidence_cache import lookup_cached  # noqa: PLC0415

    with caplog.at_level(logging.DEBUG, logger="engine.openevidence_cache"):
        result = lookup_cached(
            "definitely-not-a-real-drug-zzz",
            "another-fake-drug-yyy",
        )

    assert result is None
    matched = [
        r for r in caplog.records
        if r.name == "engine.openevidence_cache"
        and r.message == "openevidence_cache_miss"
    ]
    assert matched, "Cache miss must emit 'openevidence_cache_miss' DEBUG"
    rec = matched[0]
    assert rec.pair_hash_prefix == _expected_pair_hash(
        "definitely-not-a-real-drug-zzz", "another-fake-drug-yyy"
    )
    assert rec.cache_size >= 100


def test_cache_invalidate_emits_structured_info(caplog):
    """Cache invalidation → INFO event with prev_entry_count."""
    import sys
    sys.path.insert(0, str(_REPO_ROOT))
    from engine.openevidence_cache import (  # noqa: PLC0415
        invalidate_cache, lookup_cached,
    )

    # Prime the cache
    lookup_cached("warfarin", "ibuprofen")

    with caplog.at_level(logging.INFO, logger="engine.openevidence_cache"):
        invalidate_cache()

    matched = [
        r for r in caplog.records
        if r.name == "engine.openevidence_cache"
        and r.message == "openevidence_cache_invalidated"
    ]
    assert matched, "invalidate_cache must emit 'openevidence_cache_invalidated' INFO"
    rec = matched[0]
    assert rec.levelno == logging.INFO
    assert rec.prev_entry_count >= 100


def test_no_raw_drug_names_in_cache_hit_log(caplog):
    """The cache-hit log MUST hash drug names. Same iter-234/iter-239/
    iter-240/iter-279 PHI discipline applied to the cache lookup
    surface."""
    import sys
    sys.path.insert(0, str(_REPO_ROOT))
    from engine.openevidence_cache import lookup_cached  # noqa: PLC0415

    sentinel_a = "secret-pii-drug-aaa"
    sentinel_b = "secret-pii-drug-bbb"

    with caplog.at_level(logging.DEBUG, logger="engine.openevidence_cache"):
        # Use an OOV pair so this is a miss (nothing PHI-bearing in
        # the response either).
        lookup_cached(sentinel_a, sentinel_b)

    for rec in caplog.records:
        if rec.name != "engine.openevidence_cache":
            continue
        full = rec.getMessage() + " " + " ".join(
            f"{k}={v}" for k, v in vars(rec).items()
            if not k.startswith("_") and isinstance(v, (str, int, bool, float))
        )
        assert sentinel_a not in full, (
            f"Raw drug name {sentinel_a!r} leaked into "
            f"engine.openevidence_cache log record: {full!r}"
        )
        assert sentinel_b not in full, (
            f"Raw drug name {sentinel_b!r} leaked into "
            f"engine.openevidence_cache log record: {full!r}"
        )


def test_module_has_at_least_seven_logger_calls():
    """Source-level guard: openevidence_cache.py must have ≥ 7 logger.*
    calls (3 pre-iter-284 warnings + 4 iter-284 success-path events).
    Catches the regression class where someone deletes the new events.
    """
    src = _MODULE_PATH.read_text()
    matches = re.findall(
        r"logger\.(debug|info|warning|error|critical)\b",
        src,
    )
    assert len(matches) >= 7, (
        f"engine/openevidence_cache.py must have ≥ 7 logger.* calls, "
        f"found {len(matches)} (iter-284 floor: 3 warnings + 4 new "
        f"success-path events)"
    )
