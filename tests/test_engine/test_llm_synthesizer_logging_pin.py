"""Pin: structured logging in `engine.llm_synthesizer` is PHI-safe.

Iter 239 (T4 round 49) — closes the same PHI-leak class iter-234 caught
in `engine/rxnorm_client.py`. Pre-iter-239 audit found 8 old-style
positional `%s` logger calls in the medical-LLM cascade (`_call_medical_llm_async`
+ `_call_medical_llm_sync`) passing **exception objects directly as
record.args**:

    logger.info("OpenAI failed: %s, trying next", e)
    logger.info("%s failed: %s, trying next", model_label, e)
    ...

The medical-LLM cascade sends PROMPTS containing patient context to
OpenAI / Gemini. httpx exceptions can carry the request URL+body in
their `__str__()` representation; `%s` on an exception calls that
serialization. Same PHI-risk pattern as iter-234 rxnorm fix. Drug
names, condition codes, observation values reach `record.args` if
the request fails partway.

**Iter-239 rewrites all 8 calls to structured `extra={}` form** with
`error_type` + `status_code` + `provider` only — no exception body.
Plus 2 new aggregate cascade-failure events (async + sync) so ops
dashboards see when both providers fail and downstream synthesis
falls back to deterministic templates.

Pinned event surface
====================
  • synth_openai_non_200 (WARNING) — async cascade, status_code + provider
  • synth_openai_error (WARNING) — async cascade, error_type + provider
  • synth_gemini_non_200 (WARNING) — async, same shape
  • synth_gemini_error (WARNING) — async, same shape
  • synth_async_cascade_failed (WARNING) — both async providers exhausted
  • synth_openai_non_200_sync (WARNING) — sync cascade
  • synth_openai_error_sync (WARNING) — sync cascade
  • synth_gemini_non_200_sync (WARNING) — sync cascade
  • synth_gemini_error_sync (WARNING) — sync cascade
  • synth_sync_cascade_failed (WARNING) — both sync providers exhausted

PHI safety contract
===================
  • Exception messages NEVER appear — only `error_type`. httpx
    exceptions can carry the request URL with prompt content; PHI
    sentinel scan asserts none reach record.args.
  • Provider labels ARE logged (public configuration).
  • Status codes ARE logged (operational signal, not PHI).
  • Prompt content NEVER appears in extras (no `prompt`, `system`,
    `text`, `content` fields in any `extra={}` dict).

12th cross-pin family in the discipline lineage (after iter-178/183/
188/193/198/203/223/228/232/234/236).
"""
from __future__ import annotations

import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_MODULE = _REPO_ROOT / "engine" / "llm_synthesizer.py"

_EXPECTED_LOGGER_FLOOR_ITER239 = 15  # was 13 pre-iter-239 (8 refactored + 2 new aggregates)


def _module_text() -> str:
    return _MODULE.read_text()


def test_llm_synthesizer_logger_floor_iter239():
    """Source-level pin: engine/llm_synthesizer.py must keep >= 15 logger
    calls (was 13 pre-iter-239; iter-239 added 2 new structured aggregates
    while refactoring all 8 positional %s calls)."""
    src = _module_text()
    n_logs = len(re.findall(r"logger\.\w+\(", src))
    assert n_logs >= _EXPECTED_LOGGER_FLOOR_ITER239, (
        f"engine/llm_synthesizer.py logger-call count regressed to {n_logs}; "
        f"iter-239 floor is {_EXPECTED_LOGGER_FLOOR_ITER239}."
    )


def test_llm_synthesizer_no_positional_exception_logs_iter239():
    """Source-level PHI-discipline pin: NO `logger.X("...%s...", e)` or
    similar pattern with exception args. iter-239 audit caught 4 such
    sites (sync + async × OpenAI + Gemini) plus 4 status-code positional
    calls. Catches the regression class where a maintainer reverts to
    `logger.info("OpenAI failed: %s", e)`.

    Same shape as iter-234 rxnorm_client PHI-discipline pin.
    """
    src = _module_text()
    forbidden_patterns = (
        # logger.X("...%s...", <exception_var>) — primary PHI risk
        r'logger\.\w+\([^)]*"[^"]*%s[^"]*",\s*e\b',
        r'logger\.\w+\([^)]*"[^"]*%s[^"]*",\s*exc\b',
        r'logger\.\w+\([^)]*"[^"]*%s[^"]*",\s*err\b',
        # logger.X("OpenAI %s", model_label, e) — second-arg exception
        r'logger\.\w+\([^)]*"[^"]*",\s*\w+,\s*e\)',
    )
    for pat in forbidden_patterns:
        m = re.search(pat, src)
        assert m is None, (
            f"PHI regression: engine/llm_synthesizer.py contains a "
            f"`logger.X(\"...%s...\", <exception>)` pattern: "
            f"{m.group()!r}. iter-239 rewrote all such sites to "
            f"structured extra={{error_type: type(e).__name__}}. "
            f"Exception messages must NEVER reach record.args (httpx "
            f"errors can carry the prompt body in the URL/payload)."
        )


def test_llm_synthesizer_extras_have_no_prompt_fields_iter239():
    """Source-level PHI scrub: no `extra={}` dict in llm_synthesizer.py
    may include a `prompt` / `system` / `text` / `content` / `message`
    field. These would directly leak the patient-context-derived prompt
    to logs. The discipline must be LITERAL: status_code, error_type,
    provider, and aggregate counts only."""
    src = _module_text()
    # Find every extra={...} block
    forbidden_keys = ("prompt", "system_msg", "text", "content", "message_body")
    for match in re.finditer(r'extra=\{[^}]*\}', src):
        block = match.group()
        for key in forbidden_keys:
            forbidden_patterns = (
                f'"{key}"',
                f"'{key}'",
            )
            for pat in forbidden_patterns:
                assert pat not in block, (
                    f"PHI regression: llm_synthesizer.py extra={{}} block "
                    f"contains forbidden field {pat!r} that would leak "
                    f"prompt/patient-context content to logs:\n  {match.group()}"
                )


def test_llm_synthesizer_async_cascade_aggregate_present_iter239():
    """Source-level pin: the async cascade must emit
    `synth_async_cascade_failed` when both providers fail. Aggregate
    failure signal so ops dashboards can measure cascade-exhaustion
    rate."""
    src = _module_text()
    assert '"synth_async_cascade_failed"' in src, (
        "engine/llm_synthesizer.py must emit `synth_async_cascade_failed` "
        "WARNING when both OpenAI and Gemini async paths fail. iter-239 "
        "added this aggregate so deterministic-template-fallback rate "
        "can be measured."
    )


def test_llm_synthesizer_sync_cascade_aggregate_present_iter239():
    """Source-level pin: the sync cascade must emit
    `synth_sync_cascade_failed` when both providers fail."""
    src = _module_text()
    assert '"synth_sync_cascade_failed"' in src, (
        "engine/llm_synthesizer.py must emit `synth_sync_cascade_failed` "
        "WARNING when both OpenAI and Gemini sync paths fail."
    )


def test_llm_synthesizer_provider_status_events_pinned_iter239():
    """Source-level pin: the 8 refactored events must all be present
    in source. Catches the regression where a maintainer removes one
    and the silent-path returns."""
    src = _module_text()
    required_events = (
        '"synth_openai_non_200"',
        '"synth_openai_error"',
        '"synth_gemini_non_200"',
        '"synth_gemini_error"',
        '"synth_openai_non_200_sync"',
        '"synth_openai_error_sync"',
        '"synth_gemini_non_200_sync"',
        '"synth_gemini_error_sync"',
    )
    missing = [e for e in required_events if e not in src]
    assert not missing, (
        f"engine/llm_synthesizer.py is missing iter-239 structured event "
        f"names: {missing}. The 8 cascade-error events (4 async + 4 sync) "
        f"must all be preserved — each closes a specific silent path or "
        f"replaces a positional-%s PHI-leak site."
    )
