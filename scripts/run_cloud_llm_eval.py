"""Cloud-API submission re-eval (task #122 — round 7).

Runs frontier cloud LLMs (xAI Grok, Mistral Large, DeepSeek-V4-Pro,
Perplexity Sonar Pro, NVIDIA Nemotron Ultra) as judges on the current
ClinicalMem submission package. Same 10-category rubric as round-5.

Reuses the orchestrator at
~/.claude/plugins/marketplaces/claude-code-ultimate/multi-llm-orchestrator
+ API keys at ~/.claude-ultimate/.env.

Usage:
  python3 scripts/run_cloud_llm_eval.py [--out PATH]
  python3 scripts/run_cloud_llm_eval.py --models xai,mistral
"""
from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import os
import re
import sys
import time
from pathlib import Path

# Wire the orchestrator + .env
ORCH_ROOT = Path(
    "~/.claude/plugins/marketplaces/claude-code-ultimate/multi-llm-orchestrator"
)
sys.path.insert(0, str(ORCH_ROOT))

ENV_PATH = Path("~/.claude-ultimate/.env")
if ENV_PATH.exists():
    for line in ENV_PATH.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip("'").strip('"'))

REPO = Path(__file__).resolve().parent.parent
CATEGORIES = [
    "technical_depth",
    "innovation_differentiation",
    "clinical_credibility",
    "open_source_ecosystem_fit",
    "reproducibility_fda_defensibility",
    "demo_quality_judge_wow",
    "documentation_polish",
    "license_ip_clarity",
    "test_coverage_evidence",
    "ui_visual_design",
]


def _load_submission_package() -> str:
    parts = []
    for path, head_chars in [
        (REPO / "JUDGES.md", 12000),
        (REPO / "DEVPOST.md", 6000),
        (REPO / "README.md", 4000),
    ]:
        if path.exists():
            text = path.read_text()[:head_chars]
            parts.append(f"=== {path.name} ===\n{text}\n")
    return "\n".join(parts)


PROMPT_TEMPLATE = """You are a hackathon judge for the Agents Assemble: Healthcare AI Endgame hackathon ($25K prize, deadline 2026-05-11).

Score this submission on these 10 categories from 0 to 10 (integer scores ONLY). Be rigorous — 10 means a serious clinical-software-grade product, not "nice hackathon."

Categories:
1. technical_depth (architecture, novel CS contributions)
2. innovation_differentiation (vs Epic Sense, Hippocratic AI, OpenEvidence)
3. clinical_credibility (advisor review, FHIR coverage, NPI / 21 CFR Part 11 audit)
4. open_source_ecosystem_fit (Apache-2.0 compliance, contribution path)
5. reproducibility_fda_defensibility (Q16.16 bit-identical replay, audit-chain, manifest pinning)
6. demo_quality_judge_wow (5-second comprehension, hero, call-to-action)
7. documentation_polish (README depth, JUDGES.md, FDA Q-Sub draft)
8. license_ip_clarity (Apache-2.0 + patent grant, no buried licensing)
9. test_coverage_evidence (1371 engine + scripts tests; CI green)
10. ui_visual_design (information density, hierarchy, brand consistency)

Output ONLY a single JSON object. NO prose, NO markdown, NO code fences (```), NO commentary before or after. Start your response with `{` and end with `}`.

{
  "scores": {
    "technical_depth": 0,
    "innovation_differentiation": 0,
    "clinical_credibility": 0,
    "open_source_ecosystem_fit": 0,
    "reproducibility_fda_defensibility": 0,
    "demo_quality_judge_wow": 0,
    "documentation_polish": 0,
    "license_ip_clarity": 0,
    "test_coverage_evidence": 0,
    "ui_visual_design": 0
  },
  "verdict": "REJECT | WEAK | STRONG_CONTENDER | TOP_TIER | WINNER",
  "blocking_gaps_for_10_10": [
    {"category": "<one of the 10>", "gap": "<one sentence>"}
  ],
  "strengths": ["<bullet>", "<bullet>"]
}

SUBMISSION PACKAGE:

{PACKAGE}
"""


def _strip_thinking_and_fences(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = re.sub(r"```", "", text)
    return text


def _parse_response(text: str) -> dict | None:
    cleaned = _strip_thinking_and_fences(text)
    starts = [i for i, ch in enumerate(cleaned) if ch == "{"]
    for s in starts:
        depth, in_str, esc = 0, False, False
        for i in range(s, len(cleaned)):
            ch = cleaned[i]
            if esc:
                esc = False
                continue
            if ch == "\\":
                esc = True
                continue
            if ch == '"':
                in_str = not in_str
                continue
            if in_str:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    snippet = cleaned[s:i + 1]
                    try:
                        obj = json.loads(snippet)
                    except json.JSONDecodeError:
                        break
                    if isinstance(obj, dict) and isinstance(
                        obj.get("scores"), dict
                    ):
                        return obj
                    break
    try:
        obj = json.loads(cleaned)
        if isinstance(obj, dict) and isinstance(obj.get("scores"), dict):
            return obj
    except json.JSONDecodeError:
        pass
    return None


# Map provider key -> (provider_class, model, env_var)
PROVIDERS = {
    "xai": ("XAIProvider", "grok-4-1-fast-reasoning", "XAI_API_KEY"),
    "mistral": (
        "MistralProvider",
        "mistral-large-latest",
        "MISTRAL_API_KEY",
    ),
    "deepseek": (
        "DeepSeekProvider",
        "deepseek-chat",
        "DEEPSEEK_API_KEY",
    ),
    "perplexity": (
        "PerplexityProvider",
        "sonar-pro",
        "PERPLEXITY_API_KEY",
    ),
    "nvidia": (
        "NvidiaProvider",
        "nvidia/llama-3.1-nemotron-ultra-253b-v1",
        "NVIDIA_API_KEY",
    ),
}


async def _score_one(provider_key: str, prompt: str) -> dict:
    klass_name, model, env_var = PROVIDERS[provider_key]
    api_key = os.environ.get(env_var, "")
    if not api_key:
        return {
            "provider": provider_key,
            "model": model,
            "error": f"missing env {env_var}",
            "valid": False,
            "latency_ms": 0,
        }
    from providers import (  # type: ignore
        XAIProvider,
        MistralProvider,
        DeepSeekProvider,
        PerplexityProvider,
        NvidiaProvider,
        RateLimitConfig,
    )
    klass = {
        "XAIProvider": XAIProvider,
        "MistralProvider": MistralProvider,
        "DeepSeekProvider": DeepSeekProvider,
        "PerplexityProvider": PerplexityProvider,
        "NvidiaProvider": NvidiaProvider,
    }[klass_name]
    rate = RateLimitConfig()
    provider = klass(api_key=api_key, model=model, rate_config=rate)
    print(f"[{provider_key}/{model}] generating...", file=sys.stderr, flush=True)
    t0 = time.time()
    try:
        result = await provider.request(prompt)
    except Exception as e:
        return {
            "provider": provider_key,
            "model": model,
            "error": f"{type(e).__name__}: {e}",
            "valid": False,
            "latency_ms": int((time.time() - t0) * 1000),
        }
    latency_ms = int((time.time() - t0) * 1000)
    if result.error:
        return {
            "provider": provider_key,
            "model": model,
            "error": str(result.error),
            "raw_response_head": (result.content or "")[:300],
            "valid": False,
            "latency_ms": latency_ms,
        }
    parsed = _parse_response(result.content or "")
    if parsed is None:
        return {
            "provider": provider_key,
            "model": model,
            "error": "could not parse scores from response",
            "raw_response_head": (result.content or "")[:300],
            "valid": False,
            "latency_ms": latency_ms,
        }
    scores = parsed["scores"]
    clean_scores = {}
    for cat in CATEGORIES:
        v = scores.get(cat)
        try:
            clean_scores[cat] = int(round(float(v))) if v is not None else None
        except (TypeError, ValueError):
            clean_scores[cat] = None
    valid = all(v is not None for v in clean_scores.values())
    composite = (
        round(sum(clean_scores.values()) / 10, 2) if valid else None
    )
    return {
        "provider": provider_key,
        "model": model,
        "scores": clean_scores,
        "composite_arithmetic_mean": composite,
        "verdict": parsed.get("verdict", "UNKNOWN"),
        "blocking_gaps_for_10_10": parsed.get("blocking_gaps_for_10_10", []),
        "strengths": parsed.get("strengths", []),
        "valid": valid,
        "latency_ms": latency_ms,
    }


async def main_async(args) -> None:
    keys = [k.strip() for k in args.models.split(",") if k.strip()]
    package = _load_submission_package()
    print(
        f"submission package: {len(package):,} chars; {len(keys)} providers",
        file=sys.stderr,
    )

    # Run providers concurrently for speed
    prompt = PROMPT_TEMPLATE.replace("{PACKAGE}", package)
    tasks = [_score_one(k, prompt) for k in keys]
    per_model = await asyncio.gather(*tasks, return_exceptions=False)

    for m in per_model:
        if m.get("valid"):
            print(
                f"  {m['provider']:12s} {m['model']:38s}  "
                f"composite={m['composite_arithmetic_mean']}  "
                f"latency={m['latency_ms']/1000:.1f}s",
                file=sys.stderr,
            )
        else:
            print(
                f"  {m['provider']:12s} {m.get('model','?'):38s}  "
                f"INVALID  ({m.get('error','?')[:60]})",
                file=sys.stderr,
            )

    valid = [m for m in per_model if m.get("valid")]
    n_valid = len(valid)
    if n_valid:
        avg_composite = round(
            sum(m["composite_arithmetic_mean"] for m in valid) / n_valid, 3
        )
        avg_per_cat = {
            c: round(sum(m["scores"][c] for m in valid) / n_valid, 3)
            for c in CATEGORIES
        }
    else:
        avg_composite = None
        avg_per_cat = {}

    out = {
        "round": "round_7_cloud_llm_post_v8",
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "n_models": len(per_model),
        "n_valid": n_valid,
        "context": (
            "Cloud-API frontier-judge re-eval (task #122 round 7) post "
            "iter-275 v8 engine promotion + iter-336 PHI source-scan "
            "lock-in. Companion to round-6 local-LLM run (composite "
            "7.56 on n=5/6 Ollama Q4_K_M judges). Round-7 uses xAI "
            "Grok 4.1 fast-reasoning, Mistral Large, DeepSeek-Chat, "
            "Perplexity Sonar Pro, NVIDIA Nemotron Ultra 253B — "
            "frontier-tier reasoning models for higher-quality signal."
        ),
        "average_composite": avg_composite,
        "average_per_category": avg_per_cat,
        "per_model": per_model,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {out_path}", file=sys.stderr)
    print(
        f"avg composite ({n_valid}/{len(per_model)}): {avg_composite}",
        file=sys.stderr,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default=str(REPO / "docs" / "eval_runs" / "round_7_cloud_llm_post_v8.json"),
    )
    parser.add_argument(
        "--models",
        default=",".join(PROVIDERS.keys()),
        help="Comma-separated provider keys: " + ",".join(PROVIDERS.keys()),
    )
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
