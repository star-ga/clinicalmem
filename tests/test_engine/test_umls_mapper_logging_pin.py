# Copyright 2026 STARGA Inc. — Apache-2.0
"""Pin: structured logging in `engine.umls_mapper` is PHI-/secret-safe.

Iter 186 T4 round-36 ratchet on `engine/umls_mapper.py` — the
lowest-density engine module post-iter-181 audit (195 LOC, 3 logs
= 15.4/kloc).

Two PHI/secret-discipline issues caught by audit
=================================================
The pre-iter-186 logger calls used unstructured `logger.debug(
"... %s, e", ...)` form which calls `str(e)` on the exception. For
httpx errors that exception body carries the FULL request URL
including the `apiKey=<UMLS_API_KEY>` query parameter — a secret
leak in any production log aggregator. The fix migrates all 3
existing calls to structured `extra={"error_type": ..., "source":
..., ...}` form so the API key cannot leak via str(e).

Three previously-silent paths closed
====================================
  (1) crosswalk HTTP error -> INFO `umls_crosswalk_http_error`
      Operators auditing cross-vocabulary coverage need visibility
      into hit rate per source/target pair.
  (2) crosswalk success -> DEBUG `umls_crosswalk_complete`
      Captures match-count distribution per source/target.
  (3) find_concept HTTP error -> INFO `umls_search_http_error`
      Same rationale as (1) for the search endpoint.
  (4) _get_cui exception -> DEBUG `umls_get_cui_exception`
      Was completely silent — operators debugging are_same_concept
      failures need the error_type signal.

PHI / secret discipline:
  * NEVER log the exception body (str(e)) for httpx errors — apiKey
    leaks via the URL query parameter.
  * NEVER log search terms — clinical narrative.
  * source/target/code are public vocabulary identifiers — safe.
  * status_code, error_type, match_count are cardinal/categorical —
    safe.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_MODULE_PATH = _REPO_ROOT / "engine" / "umls_mapper.py"


_EXPECTED_LOGGER_FLOOR_ITER186 = 7  # was 3 pre-iter-186


def test_umls_mapper_logger_floor_iter186():
    """Floor bump 3 -> 7 after iter-186 closed 4 silent paths and
    rewrote the 3 existing %s,e calls to PHI-safe structured form.
    Density 15.4 -> 26.2/kloc."""
    src = _MODULE_PATH.read_text()
    calls = re.findall(
        r"\blogger\.(debug|info|warning|error|critical)\(",
        src,
    )
    assert len(calls) >= _EXPECTED_LOGGER_FLOOR_ITER186, (
        f"engine/umls_mapper.py logger-call count regressed: "
        f"{len(calls)} < floor {_EXPECTED_LOGGER_FLOOR_ITER186}. "
        f"A structured event from iter-186 was silently removed."
    )


def test_no_str_e_format_in_logger_calls_iter186():
    """No `logger.X("...%s...", e)` patterns may leak exception body
    (and thus apiKey query parameter for httpx errors). All exception
    paths must use structured `extra={"error_type": type(e).__name__}`
    form."""
    src = _MODULE_PATH.read_text()
    # Forbid the bare `, e)` end-of-args pattern
    forbidden = re.findall(
        r'logger\.\w+\([^)]*"%s"[^)]*,\s*e\s*\)',
        src,
    )
    assert not forbidden, (
        f"engine/umls_mapper.py has unstructured exception-format "
        f"logger calls that can leak the apiKey via str(e): "
        f"{forbidden}.  Migrate to "
        f"extra={{'error_type': type(e).__name__}} form."
    )

    # Also forbid string-formatting the exception with %s
    forbidden_format = re.findall(
        r'logger\.\w+\([^)]*str\(\s*e\s*\)',
        src,
    )
    assert not forbidden_format, (
        f"engine/umls_mapper.py has str(e) in logger args: "
        f"{forbidden_format}.  Use error_type only."
    )


def test_crosswalk_http_error_emits_info_event_iter186(caplog):
    """`crosswalk` emits `umls_crosswalk_http_error` INFO when the
    UMLS API returns a non-200 status. PHI-safe: structural fields
    only, NEVER the exception body or search term."""
    from unittest.mock import patch

    sentinel_secret = "ZZZ_API_KEY_SENTINEL_must_not_leak_iter186"
    # Force UMLS_API_KEY non-empty so the crosswalk attempt actually
    # makes the HTTP call instead of early-returning.
    with patch("engine.umls_mapper.UMLS_API_KEY", sentinel_secret):
        from engine.umls_mapper import crosswalk
        crosswalk.cache_clear()

        class _MockResp:
            status_code = 503
            def json(self): return {}
            text = "Service Unavailable"

        with patch("httpx.get", return_value=_MockResp()):
            with caplog.at_level(logging.INFO, logger="engine.umls_mapper"):
                result = crosswalk("ICD10CM", "E11.9", "SNOMEDCT_US")

    assert result == []
    matches = [r for r in caplog.records
               if r.message == "umls_crosswalk_http_error"]
    assert matches, "umls_crosswalk_http_error event missing"
    rec = matches[0]
    assert rec.levelno == logging.INFO
    assert rec.source == "ICD10CM"
    assert rec.target == "SNOMEDCT_US"
    assert rec.status_code == 503

    # API key sentinel must NOT appear in any logged record (defends
    # against a future regression that adds str(e) or URL logging)
    for r in caplog.records:
        for value in vars(r).values():
            if isinstance(value, str):
                assert sentinel_secret not in value, (
                    "API key sentinel leaked into log record"
                )


def test_crosswalk_exception_emits_debug_event_iter186(caplog):
    """`crosswalk` emits `umls_crosswalk_exception` DEBUG when httpx
    raises. error_type field captures the exception class without
    leaking the URL+apiKey via str(e)."""
    from unittest.mock import patch

    sentinel_url = "ZZZ_URL_SENTINEL_apiKey=secret_must_not_leak"
    with patch("engine.umls_mapper.UMLS_API_KEY", "fake-key"):
        from engine.umls_mapper import crosswalk
        crosswalk.cache_clear()

        def _raise(*a, **kw):
            raise httpx.ConnectError(
                f"Failed to connect to {sentinel_url}"
            )

        import httpx
        with patch("httpx.get", side_effect=_raise):
            with caplog.at_level(logging.DEBUG, logger="engine.umls_mapper"):
                result = crosswalk("ICD10CM", "X", "SNOMEDCT_US")

    assert result == []
    matches = [r for r in caplog.records
               if r.message == "umls_crosswalk_exception"]
    assert matches, "umls_crosswalk_exception event missing"
    rec = matches[0]
    assert rec.levelno == logging.DEBUG
    assert rec.error_type == "ConnectError"
    assert rec.source == "ICD10CM"
    assert rec.target == "SNOMEDCT_US"

    # The exception's URL sentinel (apiKey) must NEVER appear in any
    # logged record — that's the secret-leak the structured form prevents.
    for r in caplog.records:
        for value in vars(r).values():
            if isinstance(value, str):
                assert sentinel_url not in value, (
                    "Exception URL sentinel (apiKey) leaked into log"
                )
