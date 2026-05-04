"""Pin: structured logging in `engine.consensus_engine` is PHI-safe.

Iter 18 + 23 + 38 + 43 + 45 + 48 + 53 + 58 + 64 + 71 + 77 + 82 + 87 +
92 + 98 ratchet trajectory: each round closes the worst evidence-
chain gap in the engine.

Iter 98 audit: `engine/consensus_engine.py` was the **next-worst**
module post-iter-92 — 387 LOC, 4 logger calls, 49 decision points
(10.3 log-calls per kloc, lowest in the engine). Two specific gaps:

* `_parse_verdict` JSON parse failure silently fell back to keyword
  detection. Operators auditing how often each provider returns
  non-JSON had no signal.
* `_call_google` returning 200 with empty candidates (likely Gemini
  safety-filter false-positive on clinical content) was indistinguish-
  able from a non-200 response by log alone.

This iter adds 2 structured calls covering both gaps and pins them
with this test.

PHI / secret discipline:
  - model name + provider label only — never the prompt or response body
  - `parse_error_type` not str(e) (response body may quote prompt)
  - `response_length` is cardinal — fine to log
  - `status_code` + `reason` (categorical) only on no-candidates path
  - NEVER the API key (which is in the request `Authorization` header,
    not in any log path — but defence-in-depth)
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent

from engine.consensus_engine import _parse_verdict


def _record(caplog, event_name):
    for r in caplog.records:
        if r.name == "engine.consensus_engine" and r.message == event_name:
            return r
    return None


def test_parse_verdict_json_failure_emits_debug_log(caplog):
    """When the LLM returns non-JSON, `_parse_verdict` falls back to
    keyword detection — but it MUST log the parse failure at DEBUG so
    operators can audit the rate without flooding production INFO."""
    caplog.set_level(logging.DEBUG, logger="engine.consensus_engine")

    sentinel = "ZZZ_LLM_RAW_RESPONSE_PROMPT_LEAK"
    # Non-JSON LLM response that contains the sentinel — verifies the
    # parse-failure log doesn't accidentally include the response body.
    fallback_text = f"Yes I agree with this finding because {sentinel}"
    verdict = _parse_verdict(fallback_text, "Test-Model")

    assert verdict.agrees is True  # keyword fallback caught "agree"
    rec = _record(caplog, "consensus_verdict_unparseable")
    assert rec is not None, "consensus_verdict_unparseable event missing"
    assert rec.levelno == logging.DEBUG
    assert rec.model == "Test-Model"
    assert isinstance(rec.parse_error_type, str)
    assert rec.parse_error_type in {"JSONDecodeError", "ValueError", "TypeError"}
    assert isinstance(rec.response_length, int)
    assert rec.response_length == len(fallback_text)

    # Sentinel-leak scrub: the raw response body must not be in any
    # logged record's structured fields.
    for value in vars(rec).values():
        assert sentinel not in repr(value), (
            f"consensus_verdict_unparseable leaked response body sentinel"
        )


def test_parse_verdict_valid_json_emits_no_unparseable_log(caplog):
    """Happy path: JSON-valid response should NOT fire the
    unparseable event."""
    caplog.set_level(logging.DEBUG, logger="engine.consensus_engine")
    verdict = _parse_verdict(
        '{"agrees": true, "confidence": 0.9, "reasoning": "looks correct"}',
        "Test-Model",
    )
    assert verdict.agrees is True
    rec = _record(caplog, "consensus_verdict_unparseable")
    assert rec is None, "Valid JSON must not fire unparseable log"


_LOGGER_PATTERN = re.compile(
    r"\b(?:logger|log_fn)\.(?:debug|info|warning|error|exception|critical)\("
    r"|^\s*log_fn\(",
    re.M,
)


def test_consensus_engine_has_at_least_six_structured_logs():
    """Iter-98 floor: consensus_engine.py grew from 4 to 6 direct
    logger calls (5 → 7 if you count the indirect log_fn call site).
    Future evidence-chain regressions can't quietly slip below the
    floor without firing this gate."""
    src = (_REPO_ROOT / "engine" / "consensus_engine.py").read_text()
    direct_pattern = re.compile(
        r"\blogger\.(?:debug|info|warning|error|exception|critical)\("
    )
    count = len(direct_pattern.findall(src))
    assert count >= 6, (
        f"engine/consensus_engine.py has {count} direct logger calls; "
        f"floor is 6 (iter-98 baseline)."
    )


def test_consensus_engine_no_url_with_api_key_in_logs():
    """Defence-in-depth: even though API keys go via Authorization
    header (not URL), no log call may log a full URL string. A future
    maintainer who logs `resp.url` would leak the path; logging only
    domains is acceptable. This test asserts no `resp.url` or
    `request.url` appears as a logged field."""
    src = (_REPO_ROOT / "engine" / "consensus_engine.py").read_text()
    forbidden_patterns = (
        r"logger\.\w+\([^)]*resp\.url",
        r"logger\.\w+\([^)]*request\.url",
        r'logger\.\w+\([^)]*"https://',  # full URL literal in a log call
    )
    for pat in forbidden_patterns:
        m = re.search(pat, src)
        assert m is None, (
            f"engine/consensus_engine.py logs a URL or full URL literal "
            f"(matched {pat!r}) — this could leak query strings or path "
            f"params. Log domain + path-prefix only."
        )
