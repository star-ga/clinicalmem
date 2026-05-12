"""Multi-LLM Consensus Verification — independent verification of clinical findings.

Instead of trusting a single LLM, this module runs up to 6 LLMs independently
on the same clinical finding and calculates an agreement score. Only findings
with ≥2/3 agreement are reported as verified.

Models (used when API keys are available — all US-based):
- OpenAI GPT-5.5 (clinical validation, 260 physicians, HIPAA BAA)
- Google Gemini 3.1 Pro (flagship reasoning, 1M context)
- xAI Grok 4.3 (fast reasoning, 2M context)
- Anthropic Claude Opus 4.7 (deepest reasoning, safety-focused)
- Perplexity Sonar Reasoning Pro (web-grounded clinical search)
- Meta Llama 4 Maverick (400B MoE, 17B active / 128 experts, via NVIDIA NIM)

All 6 providers are US-headquartered, ensuring HIPAA-compatible data residency.
Diversity of model architectures reduces correlated hallucination risk.
"""
import asyncio
import json
import logging
import os
from dataclasses import dataclass
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_VERIFICATION_PROMPT = """You are a clinical pharmacist verifying a safety finding.

CLINICAL FINDING:
{finding}

SUPPORTING EVIDENCE:
{evidence}

PATIENT CONTEXT:
{patient_context}

Based on the evidence provided, do you agree this is a genuine clinical safety concern?
Respond with ONLY valid JSON:
{{"agrees": true/false, "confidence": 0.0-1.0, "reasoning": "1-2 sentence explanation"}}"""


@dataclass(frozen=True)
class LLMVerdict:
    """A single LLM's verdict on a clinical finding."""

    model: str
    agrees: bool
    confidence: float
    reasoning: str


@dataclass(frozen=True)
class ConsensusResult:
    """Multi-LLM consensus verification result."""

    finding: str
    verdicts: tuple[LLMVerdict, ...]
    agreement_count: int
    total_models: int
    consensus_level: str  # HIGH, MEDIUM, LOW, NONE, LIMITED
    confidence_score: float
    should_report: bool
    reasoning_summary: str


def _build_prompt(
    finding: str,
    evidence: list[dict[str, Any]],
    patient_context: dict[str, Any],
) -> str:
    """Build the verification prompt."""
    evidence_text = "\n".join(
        f"- [{e.get('block_id', '?')}] {e.get('title', '')}: {e.get('content', '')}"
        for e in evidence[:10]
    )
    ctx_text = json.dumps(
        {
            k: v
            for k, v in patient_context.items()
            if k in ("medications", "conditions", "allergies", "patient_id")
        },
        default=str,
    )[:500]

    return _VERIFICATION_PROMPT.format(
        finding=finding,
        evidence=evidence_text or "No evidence blocks provided.",
        patient_context=ctx_text,
    )


def _parse_verdict(text: str, model: str) -> LLMVerdict:
    """Parse an LLM response into a verdict."""
    text = text.strip()
    # Handle markdown code blocks
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        text = text.rsplit("```", 1)[0].strip()

    try:
        data = json.loads(text)
        return LLMVerdict(
            model=model,
            agrees=bool(data.get("agrees", False)),
            confidence=max(0.0, min(1.0, float(data.get("confidence", 0.5)))),
            reasoning=str(data.get("reasoning", ""))[:300],
        )
    except (json.JSONDecodeError, ValueError, TypeError) as e:
        # If we can't parse JSON, check for agreement keywords. Log at
        # DEBUG so SaMD operators can audit how often each provider's
        # output is non-JSON without raw text leaking into production
        # INFO logs. PHI / secret discipline: model + parse_error_type
        # + response_length only — NEVER the response body itself
        # (the LLM may quote the prompt verbatim including any drug
        # names that passed through the redaction layer).
        logger.debug(
            "consensus_verdict_unparseable",
            extra={
                "model": model,
                "parse_error_type": type(e).__name__,
                "response_length": len(text),
            },
        )
        lower = text.lower()
        agrees = any(w in lower for w in ["agree", "yes", "genuine", "real concern", "confirmed"])
        return LLMVerdict(
            model=model,
            agrees=agrees,
            confidence=0.5,
            reasoning=text[:300],
        )


# --- Provider callers (each isolated, all async) ---

_SYSTEM_MSG = "You are a clinical safety verification assistant. Respond ONLY with JSON."


async def _call_openai(prompt: str, api_key: str) -> LLMVerdict:
    """Call OpenAI GPT-5.5 for verification."""
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": "gpt-5.5",
                "messages": [
                    {"role": "system", "content": _SYSTEM_MSG},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.1,
                "max_completion_tokens": 256,
            },
        )
        if resp.status_code != 200:
            # Pre-raise structured WARNING — provider + status_code only
            # (no body, which can include rate-limit messages that quote
            # the prompt back). The downstream consensus_provider_error
            # handler sees only RuntimeError type, which loses the HTTP
            # status. This event preserves it for SaMD audit-replay.
            logger.warning(
                "consensus_provider_http_error",
                extra={
                    "provider": "OpenAI-GPT-5.5",
                    "status_code": resp.status_code,
                },
            )
            raise RuntimeError(f"OpenAI returned {resp.status_code}")
        text = resp.json()["choices"][0]["message"]["content"]
        return _parse_verdict(text, "OpenAI-GPT-5.5")


async def _call_google(prompt: str, api_key: str, model_id: str, label: str) -> LLMVerdict:
    """Call a Google model for verification."""
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{model_id}:generateContent",
            headers={"x-goog-api-key": api_key},
            json={
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": 0.1, "maxOutputTokens": 256},
            },
        )
        if resp.status_code != 200:
            logger.warning(
                "consensus_provider_http_error",
                extra={
                    "provider": label,
                    "status_code": resp.status_code,
                },
            )
            raise RuntimeError(f"{label} returned {resp.status_code}")
        candidates = resp.json().get("candidates", [])
        if not candidates:
            # Google returns 200 with empty candidates when the prompt
            # is filtered by Gemini's safety classifiers (e.g. clinical
            # content flagged as medical advice). Distinct failure mode
            # from a non-200 response — log it separately so operators
            # can distinguish API outages from safety-filter false-
            # positives. Provider label only; never the prompt or the
            # response body (Google echoes the prompt in some error
            # paths and we redact PHI upstream, but defense-in-depth).
            logger.warning(
                "consensus_provider_no_candidates",
                extra={
                    "provider": label,
                    "status_code": resp.status_code,
                    "reason": "empty_candidates_likely_safety_filter",
                },
            )
            raise RuntimeError(f"{label} returned no candidates")
        text = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "")
        return _parse_verdict(text, label)


async def _call_openai_compatible(
    prompt: str,
    api_key: str,
    base_url: str,
    model: str,
    label: str,
) -> LLMVerdict:
    """Call any OpenAI-compatible API (xAI, Perplexity)."""
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.post(
            f"{base_url}/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": _SYSTEM_MSG},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.1,
                "max_tokens": 256,
            },
        )
        if resp.status_code != 200:
            logger.warning(
                "consensus_provider_http_error",
                extra={
                    "provider": label,
                    "status_code": resp.status_code,
                },
            )
            raise RuntimeError(f"{label} returned {resp.status_code}")
        text = resp.json()["choices"][0]["message"]["content"]
        return _parse_verdict(text, label)


async def _call_perplexity(prompt: str, api_key: str) -> LLMVerdict:
    """Call Perplexity Sonar Reasoning Pro for verification."""
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.post(
            "https://api.perplexity.ai/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": "sonar-reasoning-pro",
                "messages": [
                    {"role": "system", "content": _SYSTEM_MSG},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.1,
                "max_tokens": 256,
            },
        )
        if resp.status_code != 200:
            logger.warning(
                "consensus_provider_http_error",
                extra={
                    "provider": "Perplexity-Sonar-Pro",
                    "status_code": resp.status_code,
                },
            )
            raise RuntimeError(f"Perplexity returned {resp.status_code}")
        text = resp.json()["choices"][0]["message"]["content"]
        return _parse_verdict(text, "Perplexity-Sonar-Pro")


async def _call_anthropic(prompt: str, api_key: str) -> LLMVerdict:
    """Call Anthropic Claude Opus 4.7 for verification."""
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": "claude-opus-4-7",
                "max_tokens": 256,
                "temperature": 0.1,
                "system": _SYSTEM_MSG,
                "messages": [{"role": "user", "content": prompt}],
            },
        )
        if resp.status_code != 200:
            logger.warning(
                "consensus_provider_http_error",
                extra={
                    "provider": "Anthropic-Claude-Opus-4.7",
                    "status_code": resp.status_code,
                },
            )
            raise RuntimeError(f"Anthropic returned {resp.status_code}")
        content = resp.json().get("content", [])
        text = content[0].get("text", "") if content else ""
        return _parse_verdict(text, "Anthropic-Claude-Opus-4.7")


async def verify_finding_consensus(
    finding: str,
    evidence: list[dict[str, Any]],
    patient_context: dict[str, Any],
) -> ConsensusResult:
    """Run multi-LLM consensus verification on a clinical finding.

    Fires all available LLMs in parallel (up to 5 US-based). Calculates agreement score.
    ≥90% = HIGH, ≥67% = MEDIUM, ≥1 = LOW, 0 = NONE.
    Findings with < 2 models available get consensus_level "LIMITED".

    Args:
        finding: The clinical finding to verify.
        evidence: Supporting evidence blocks from patient memory.
        patient_context: Patient summary dict.

    Returns:
        Immutable ConsensusResult with verdicts and agreement level.
    """
    # PHI guard
    try:
        from engine.phi_detector import redact_phi
        prompt, _ = redact_phi(_build_prompt(finding, evidence, patient_context))
    except ImportError as exc:
        # WARNING level — PHI guard absence is operationally significant.
        # The downstream prompt construction proceeds without redaction,
        # which is acceptable when phi_detector is unavailable (e.g.
        # development/test environments) but operators must see the
        # absence in production logs. error_type only — no exception
        # body which can include filesystem paths.
        logger.warning(
            "consensus_phi_guard_unavailable",
            extra={
                "error_type": type(exc).__name__,
                "fallback": "prompt_constructed_without_redaction",
            },
        )
        prompt = _build_prompt(finding, evidence, patient_context)

    openai_key = os.environ.get("OPENAI_API_KEY")
    google_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    xai_key = os.environ.get("XAI_API_KEY")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    perplexity_key = os.environ.get("PERPLEXITY_API_KEY")
    nvidia_key = os.environ.get("NVIDIA_API_KEY")

    # PHI-safe entry log: counts only — no finding text, no evidence body,
    # no patient context. Provider availability is public configuration.
    available_providers = [
        label for key, label in (
            (openai_key, "OpenAI-GPT-5.5"),
            (google_key, "Gemini-3.1-Pro"),
            (xai_key, "xAI-Grok-4.3"),
            (anthropic_key, "Anthropic-Claude-Opus-4.7"),
            (perplexity_key, "Perplexity-Sonar-Pro"),
            (nvidia_key, "Meta-Llama-4-Maverick-400B"),
        )
        if key
    ]
    logger.debug(
        "consensus_dispatch",
        extra={
            "providers_available": len(available_providers),
            "providers": available_providers,
            "evidence_block_count": len(evidence),
            "prompt_length": len(prompt),
        },
    )

    tasks: list[tuple[str, Any]] = []

    # Tier 1: Clinical-validated models
    if openai_key:
        tasks.append(("OpenAI-GPT-5.5", _call_openai(prompt, openai_key)))
    if google_key:
        tasks.append(("Gemini-3.1-Pro", _call_google(
            prompt, google_key, "gemini-3.1-pro-preview", "Gemini-3.1-Pro",
        )))

    # Tier 2: Diverse US-based models (reduce correlated errors)
    if xai_key:
        tasks.append(("xAI-Grok-4.3", _call_openai_compatible(
            prompt, xai_key, "https://api.x.ai", "grok-4.3", "xAI-Grok-4.3",
        )))
    if anthropic_key:
        tasks.append(("Anthropic-Claude-Opus-4.7", _call_anthropic(prompt, anthropic_key)))
    if perplexity_key:
        tasks.append(("Perplexity-Sonar-Pro", _call_perplexity(
            prompt, perplexity_key,
        )))
    if nvidia_key:
        tasks.append(("Meta-Llama-4-Maverick-400B", _call_openai_compatible(
            prompt,
            nvidia_key,
            "https://integrate.api.nvidia.com",
            "meta/llama-4-maverick-17b-128e-instruct",
            "Meta-Llama-4-Maverick-400B",
        )))

    if not tasks:
        # WARNING level — operators must see when consensus can't run at
        # all. The abstention gate downstream will catch this, but the
        # signal needs to surface in production logs.
        logger.warning(
            "consensus_no_providers",
            extra={
                "providers_available": 0,
                "level": "NONE",
                "should_report": False,
            },
        )
        return ConsensusResult(
            finding=finding,
            verdicts=(),
            agreement_count=0,
            total_models=0,
            consensus_level="NONE",
            confidence_score=0.0,
            should_report=False,
            reasoning_summary="No LLM API keys available for consensus verification.",
        )

    # Fire all in parallel
    verdicts: list[LLMVerdict] = []
    results = await asyncio.gather(
        *(t[1] for t in tasks), return_exceptions=True
    )

    for (label, _), result in zip(tasks, results):
        if isinstance(result, Exception):
            # Per-provider error — type only (not message body, which can
            # leak request/response bodies in third-party SDKs).
            logger.warning(
                "consensus_provider_error",
                extra={
                    "provider": label,
                    "error_type": type(result).__name__,
                },
            )
            verdicts.append(
                LLMVerdict(
                    model=label,
                    agrees=False,
                    confidence=0.0,
                    reasoning=f"API call failed: {result}",
                )
            )
        else:
            # Per-provider success — verdict bool + confidence bucket only,
            # never the reasoning string (model output may quote PHI even
            # though the prompt was redacted).
            logger.debug(
                "consensus_provider_verdict",
                extra={
                    "provider": label,
                    "agrees": result.agrees,
                    "confidence_bucket": (
                        "high" if result.confidence >= 0.75
                        else "medium" if result.confidence >= 0.5
                        else "low"
                    ),
                },
            )
            verdicts.append(result)

    agreement_count = sum(1 for v in verdicts if v.agrees)
    total = len(verdicts)

    if total < 2:
        level = "LIMITED"
    elif agreement_count == total:
        level = "HIGH"
    elif agreement_count >= total * 2 / 3:
        level = "MEDIUM"
    elif agreement_count >= 1:
        level = "LOW"
    else:
        level = "NONE"

    # Weighted confidence: models that agree with higher confidence count more
    if total > 0:
        weighted_sum = sum(
            v.confidence if v.agrees else (1.0 - v.confidence)
            for v in verdicts
        )
        confidence = weighted_sum / total
    else:  # pragma: no cover — early return handles empty tasks
        confidence = 0.0

    reasoning_parts = [
        f"{v.model}: {'agrees' if v.agrees else 'disagrees'} ({v.confidence:.1f}) — {v.reasoning}"
        for v in verdicts
    ]

    # Aggregate-result log: categorical level + counts only. INFO when the
    # finding will be reported; WARNING when consensus dropped to LOW /
    # NONE / LIMITED — those are the abstention triggers downstream.
    should_report = level in ("HIGH", "MEDIUM")
    log_fn = logger.info if should_report else logger.warning
    log_fn(
        "consensus_aggregated",
        extra={
            "level": level,
            "agreement_count": agreement_count,
            "total_models": total,
            "should_report": should_report,
            "confidence_score": round(confidence, 2),
        },
    )

    return ConsensusResult(
        finding=finding,
        verdicts=tuple(verdicts),
        agreement_count=agreement_count,
        total_models=total,
        consensus_level=level,
        confidence_score=round(confidence, 2),
        should_report=should_report,
        reasoning_summary=" | ".join(reasoning_parts),
    )


def verify_finding_consensus_sync(
    finding: str,
    evidence: list[dict[str, Any]],
    patient_context: dict[str, Any],
) -> ConsensusResult:
    """Synchronous wrapper for verify_finding_consensus."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # Already in an async context — create a new thread
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(
                asyncio.run,
                verify_finding_consensus(finding, evidence, patient_context),
            )
            return future.result(timeout=30)
    else:
        return asyncio.run(
            verify_finding_consensus(finding, evidence, patient_context)
        )
