"""Local-LLM submission re-eval (task #122 — local-only variant).

Runs each available Ollama-served local LLM as a judge on the current
ClinicalMem submission package (JUDGES.md + DEVPOST.md + README.md
top section + demo.html chip text). Produces eval_round_6 artifact in
the same schema as docs/eval_runs/round_5_post_iter140.json.

Why local: Runpod balance exhausted 2026-05-08; the original
multi-LLM round-5 used cloud API calls (Gemini, Grok, Perplexity,
Mistral). This local variant produces a fresh post-iter-275 v8
promotion + iter-336 PHI lock-in eval signal using the local Ollama
service — air-gapped, reproducible, no cloud cost.

Models scored: top-tier local Ollama models that fit on a 10GB GPU
(qwen3.5 9.7B Q4_K_M, deepseek-r1:8b, qwen3:8b, qwen2.5:7b, glm4:9b,
mind-mem:4b). Same 10-category rubric as round-5.

Usage:
  python3 scripts/run_local_llm_eval.py [--out PATH] [--models M1,M2,...]

Env: requires Ollama running at http://localhost:11434 (default).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
import time
from pathlib import Path

import httpx

REPO = Path(__file__).resolve().parent.parent
OLLAMA = "http://localhost:11434"

DEFAULT_MODELS = [
    "qwen3.5:latest",   # 9.7B, the largest dense local model
    "qwen3:8b",
    "deepseek-r1:8b",
    "qwen2.5:7b",
    "glm4:9b",
    "mind-mem:4b",      # the local fine-tune
]

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
    """Concatenate the judge-facing parts of the submission. Capped to
    keep prompt under typical 8k-context windows."""
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

Output ONLY a single JSON object (no prose, no code fences, no markdown):

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


def _call_ollama(model: str, prompt: str, timeout: float = 600.0) -> tuple[str, float]:
    """Returns (response_text, latency_seconds)."""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_ctx": 8192,
            "num_predict": 1500,
        },
    }
    t0 = time.time()
    with httpx.Client(timeout=timeout) as cli:
        r = cli.post(f"{OLLAMA}/api/generate", json=payload)
        r.raise_for_status()
        data = r.json()
    return data.get("response", ""), time.time() - t0


def _parse_response(text: str) -> dict | None:
    """Extract the first JSON object that has a 'scores' key."""
    # Strip <think> blocks (deepseek-r1 etc.)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    # Find balanced JSON
    candidates = re.findall(r"\{[^{]*\"scores\".*?\}", text, re.S)
    for c in candidates:
        depth = 0
        end = -1
        for i, ch in enumerate(c):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        if end < 0:
            continue
        try:
            obj = json.loads(c[:end])
            if isinstance(obj.get("scores"), dict):
                return obj
        except json.JSONDecodeError:
            continue
    # Fallback: try the whole text
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _score_one(model: str, package: str) -> dict:
    prompt = PROMPT_TEMPLATE.replace("{PACKAGE}", package)
    print(f"[{model}] generating...", file=sys.stderr, flush=True)
    try:
        resp, latency = _call_ollama(model, prompt)
    except Exception as e:
        return {
            "model": model,
            "error": f"{type(e).__name__}: {e}",
            "valid": False,
            "latency_ms": 0,
        }
    parsed = _parse_response(resp)
    if parsed is None or not isinstance(parsed.get("scores"), dict):
        return {
            "model": model,
            "error": "could not parse scores from response",
            "raw_response_head": resp[:300],
            "valid": False,
            "latency_ms": int(latency * 1000),
        }
    scores = parsed["scores"]
    # Coerce to int + only the canonical 10 categories
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
        "model": model,
        "scores": clean_scores,
        "composite_arithmetic_mean": composite,
        "verdict": parsed.get("verdict", "UNKNOWN"),
        "blocking_gaps_for_10_10": parsed.get("blocking_gaps_for_10_10", []),
        "strengths": parsed.get("strengths", []),
        "valid": valid,
        "latency_ms": int(latency * 1000),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default=str(REPO / "docs" / "eval_runs" / "round_6_local_llm_post_v8.json"),
    )
    parser.add_argument(
        "--models",
        default=",".join(DEFAULT_MODELS),
        help="Comma-separated Ollama model tags",
    )
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    package = _load_submission_package()
    print(
        f"submission package: {len(package):,} chars; "
        f"{len(models)} models",
        file=sys.stderr,
    )

    per_model = []
    for m in models:
        result = _score_one(m, package)
        per_model.append(result)
        if result.get("valid"):
            print(
                f"  {m:24s}  composite={result['composite_arithmetic_mean']}  "
                f"latency={result['latency_ms']/1000:.1f}s",
                file=sys.stderr,
            )
        else:
            print(
                f"  {m:24s}  INVALID  ({result.get('error','?')})",
                file=sys.stderr,
            )

    valid_models = [m for m in per_model if m.get("valid")]
    n_valid = len(valid_models)
    if n_valid:
        avg_composite = round(
            sum(m["composite_arithmetic_mean"] for m in valid_models) / n_valid,
            3,
        )
        avg_per_category = {}
        for cat in CATEGORIES:
            xs = [m["scores"][cat] for m in valid_models]
            avg_per_category[cat] = round(sum(xs) / len(xs), 3)
    else:
        avg_composite = None
        avg_per_category = {}

    out = {
        "round": "round_6_local_llm_post_v8",
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "n_models": len(per_model),
        "n_valid": n_valid,
        "context": (
            "Local-LLM re-eval (task #122 local variant) post iter-275 v8 "
            "engine promotion + iter-336 PHI source-scan lock-in. Runpod "
            "balance exhausted 2026-05-08, so this round uses local Ollama "
            "models in place of the round-5 cloud APIs. Same 10-category "
            "rubric, integer scores 0-10, composite is arithmetic mean."
        ),
        "average_composite": avg_composite,
        "average_per_category": avg_per_category,
        "per_model": per_model,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {out_path}", file=sys.stderr)
    print(
        f"avg composite ({n_valid}/{len(per_model)} models): {avg_composite}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
