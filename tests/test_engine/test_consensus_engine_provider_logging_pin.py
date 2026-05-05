"""Pin engine/consensus_engine.py provider HTTP-error logging discipline.

Iter 128 (round 25 T4 evidence-chain ratchet).

`engine/consensus_engine.py` orchestrates the 5-LLM consensus surface
(OpenAI GPT-5.5, Gemini 3.1 Pro, xAI Grok 4.3, Anthropic Claude Opus
4.7, Perplexity Sonar Pro). Pre-iter-128 it had 6 logger calls / 496
LOC (12.1/kloc — lowest density of all engine modules with > 200 LOC).
While the per-call provider raises were caught by an asyncio.gather
return_exceptions handler that emitted a `consensus_provider_error`
WARNING, **the actual HTTP status code was lost in transit** —
operators reading the logs saw only `error_type=RuntimeError` with
no way to distinguish 401 (auth fail) from 429 (rate limit) from
503 (provider outage).

Six paths hardened:

  Five status-code raises:
    _call_openai          (L158): status != 200 -> RuntimeError
    _call_google          (L175): status != 200 -> RuntimeError
    _call_openai_compatible (L222): status != 200 -> RuntimeError (xAI)
    _call_perplexity      (L244): status != 200 -> RuntimeError
    _call_anthropic       (L268): status != 200 -> RuntimeError

  All five now pre-log structured WARNING:
    logger.warning("consensus_provider_http_error",
                   extra={"provider": "<label>", "status_code": <int>})

  Plus the PHI guard ImportError fallback (L297) — was silent;
  now emits WARNING `consensus_phi_guard_unavailable` with
  error_type + fallback="prompt_constructed_without_redaction".

Bundle/provider labels are PUBLIC configuration; status codes are
the integrity signal an operator needs to triage. NEVER log
response body or exception messages — third-party SDKs sometimes
echo the prompt back in error paths, and we redact PHI upstream
but defense-in-depth.

Net engine/consensus_engine.py: 6 -> 12 logger calls (+6 events).
"""
from __future__ import annotations

import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_MODULE = _REPO_ROOT / "engine" / "consensus_engine.py"


def _module_text() -> str:
    return _MODULE.read_text()


def test_consensus_engine_logger_floor():
    """Pin a logger-call floor (>= 12) so silent-removal regressions fail."""
    text = _module_text()
    calls = re.findall(r"\blogger\.(debug|info|warning|error|critical)\(", text)
    assert len(calls) >= 12, (
        f"engine/consensus_engine.py logger-call count regressed: "
        f"{len(calls)} < floor 12. A structured event was silently removed."
    )


def test_no_bare_str_e_logger_calls_in_consensus_engine():
    """No bare `logger.X(..., e)` patterns — PHI/secret discipline.

    Provider response bodies can include rate-limit / auth messages
    that quote the prompt back. error_type + status_code only.
    """
    text = _module_text()
    offenders: list[tuple[int, str]] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        if not re.search(r"\blogger\.(debug|info|warning|error|critical)\(", line):
            continue
        if "%s" in line and re.search(r",\s*(e|exc)\s*[)]", line):
            offenders.append((lineno, line.strip()))
    assert not offenders, (
        "engine/consensus_engine.py contains bare `logger.X(\"... %s\", e)` "
        "patterns that can leak provider response bodies / prompts. "
        "Convert to structured form with error_type + provider labels.\n"
        "Offenders:\n" + "\n".join(f"  L{n}: {l}" for n, l in offenders)
    )


def test_every_provider_status_raise_pre_logs_structured_event():
    """Every `raise RuntimeError(f"... returned {resp.status_code}")`
    must be preceded by a structured WARNING with provider + status_code.

    Source-level pin: scans the module for `RuntimeError` raises
    matching the provider-status pattern and asserts each is
    co-located (within ~400 chars) with a
    `consensus_provider_http_error` event.
    """
    text = _module_text()
    # Find every `raise RuntimeError(f"... returned {resp.status_code}")`
    # — this is the canonical provider-fail pattern across all 5 callers.
    raise_pattern = re.compile(
        r'raise RuntimeError\(f["\'][^"\']*returned \{resp\.status_code\}["\']\)'
    )
    raise_matches = list(raise_pattern.finditer(text))
    assert len(raise_matches) >= 5, (
        f"Expected >= 5 provider-status raises (OpenAI, Google, xAI, "
        f"Perplexity, Anthropic); found {len(raise_matches)}. The 5-LLM "
        f"consensus design depends on having all 5 providers wired."
    )
    for match in raise_matches:
        # Walk backward up to 400 chars to find the structured pre-raise log.
        window_start = max(0, match.start() - 400)
        window = text[window_start : match.start()]
        assert '"consensus_provider_http_error"' in window, (
            f"Provider-status raise at offset {match.start()} is NOT "
            f"preceded (within ~400 chars) by a "
            f"`logger.warning(\"consensus_provider_http_error\", ...)` "
            f"call. Each of the 5 provider callers must pre-log the "
            f"HTTP status before the RuntimeError so SaMD operators can "
            f"distinguish 401/429/503 in audit-replay. Match text: "
            f"{match.group()!r}"
        )
        assert '"provider"' in window, (
            f"`consensus_provider_http_error` event near offset "
            f"{match.start()} is missing the `provider` field in "
            f"extra dict — operators need the label to triage."
        )
        assert '"status_code"' in window, (
            f"`consensus_provider_http_error` event near offset "
            f"{match.start()} is missing the `status_code` field in "
            f"extra dict — that's the whole point of the pre-raise log."
        )


def test_phi_guard_import_failure_logs_fallback():
    """PHI guard ImportError fallback emits structured WARNING.

    Pre-iter-128 the `except ImportError: prompt = ...` was silent —
    operators couldn't tell whether the prompt was redacted or not
    (operationally significant for SaMD audit). iter-128 added a
    `consensus_phi_guard_unavailable` WARNING with error_type +
    fallback marker.
    """
    text = _module_text()
    assert '"consensus_phi_guard_unavailable"' in text, (
        "PHI guard ImportError fallback must emit "
        "logger.warning('consensus_phi_guard_unavailable', extra={...}). "
        "Without this signal, an SaMD operator cannot tell whether the "
        "prompt sent to LLM providers was redacted or contained PHI."
    )
    # Verify the fallback marker is present so reviewers know what
    # the system did when the guard was unavailable.
    idx = text.find('"consensus_phi_guard_unavailable"')
    window = text[idx : idx + 400]
    assert '"fallback"' in window, (
        "consensus_phi_guard_unavailable event must include `fallback` "
        "in extra dict so operators see what the system did when the "
        "PHI guard was unavailable."
    )
    assert "prompt_constructed_without_redaction" in window, (
        "consensus_phi_guard_unavailable event must mark the fallback "
        "as 'prompt_constructed_without_redaction' so operators are "
        "explicit about the absence of redaction."
    )


def test_no_response_body_logged_in_consensus_engine():
    """Source-level pin: the 5 provider callers' status-code logs
    must NOT include `resp.text` / `resp.json()` / `resp.content`
    in their structured `extra` dicts.

    Same paranoia as iter-103 / iter-108 / iter-118 sentinel-leak
    scrubs: defensively block patterns that would echo provider
    response bodies into structured logs (where the SaMD audit
    chain reads them).
    """
    text = _module_text()
    # Find every consensus_provider_http_error block + check the
    # extra dict for forbidden fields.
    for match in re.finditer(
        r'"consensus_provider_http_error".*?\}',
        text,
        re.DOTALL,
    ):
        block = match.group()
        forbidden = ("resp.text", "resp.json()", "resp.content", "response_body")
        for token in forbidden:
            assert token not in block, (
                f"consensus_provider_http_error event contains forbidden "
                f"token {token!r} — would leak provider response body "
                f"into the SaMD audit chain. Status code + provider "
                f"label only."
            )
