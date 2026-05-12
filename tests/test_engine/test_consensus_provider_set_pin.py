"""Pin: the Layer-4 consensus surface has exactly 6 US-based providers.

Iter 223 (round 46 T1) — structural pin that codifies the iter-220
Meta Llama 4 Maverick (400B MoE) addition as an architectural invariant on
`engine/consensus_engine.py`. Same drift-prevention shape as the
iter-178 BOOST_KEYS coverage cross-pin, iter-183 Q16.16 canonical-pin
coverage cross-pin, iter-188 encode_pair encoder contract pin, and
iter-193 audit-replay structural-integrity pin.

What the existing pin catches and what this one adds
----------------------------------------------------

`tests/test_engine/test_consensus_engine_provider_logging_pin.py` (iter
128) pins the *PHI-safe HTTP-error log discipline* — every provider
HTTP non-200 raise must be co-located with a `consensus_provider_http_error`
WARNING that includes provider label + status_code, never response body.
But it asserts only `>= 5 raise` statements (the 6 providers share 5
distinct `_call_*` codepaths because xAI Grok and NVIDIA Nemotron both
use `_call_openai_compatible`). If a future change silently removed
`NVIDIA Nemotron` from the dispatch list while leaving `_call_openai_compatible`
intact, the existing pin would still pass.

This pin closes that gap by asserting the *six labels are present* in
the source — labels are public configuration, not PHI, so it's safe
to lock them at the source-text level.

Locked surface
--------------
1. The 6 canonical labels appear verbatim in `consensus_engine.py`:
     OpenAI-GPT-5.5
     Gemini-3.1-Pro
     xAI-Grok-4.3
     Anthropic-Claude-Opus-4.7
     Perplexity-Sonar-Pro
     Meta-Llama-4-Maverick-400B

2. The 6 environment variable names appear verbatim:
     OPENAI_API_KEY
     GOOGLE_API_KEY
     XAI_API_KEY
     ANTHROPIC_API_KEY
     PERPLEXITY_API_KEY
     NVIDIA_API_KEY

3. The 6 canonical model identifiers appear verbatim:
     gpt-5.5
     gemini-3.1-pro-preview
     grok-4.3
     claude-opus-4-7
     sonar-reasoning-pro
     meta/llama-4-maverick-17b-128e-instruct

4. NVIDIA Nemotron uses NIM at integrate.api.nvidia.com (sharing
   _call_openai_compatible with xAI Grok, which is a deliberate
   architectural choice — both expose OpenAI-compatible /v1/chat
   surfaces).

5. The available_providers tuple in `verify_finding_consensus`
   has exactly 6 (label, key) pairs — guards against "added 7th
   provider but forgot to extend the available_providers list".

A 7th provider is fine — but anyone adding it must extend this pin
explicitly so the surface count stays a deliberate decision, not a
silent drift.
"""
from __future__ import annotations

import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_MODULE = _REPO_ROOT / "engine" / "consensus_engine.py"


def _module_text() -> str:
    return _MODULE.read_text()


_CANONICAL_LABELS = (
    "OpenAI-GPT-5.5",
    "Gemini-3.1-Pro",
    "xAI-Grok-4.3",
    "Anthropic-Claude-Opus-4.7",
    "Perplexity-Sonar-Pro",
    "Meta-Llama-4-Maverick-400B",
)


_CANONICAL_ENV_KEYS = (
    "OPENAI_API_KEY",
    "GOOGLE_API_KEY",
    "XAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "PERPLEXITY_API_KEY",
    "NVIDIA_API_KEY",
)


_CANONICAL_MODEL_IDS = (
    "gpt-5.5",
    "gemini-3.1-pro-preview",
    "grok-4.3",
    "claude-opus-4-7",
    "sonar-reasoning-pro",
    "meta/llama-4-maverick-17b-128e-instruct",
)


def test_six_canonical_provider_labels_present():
    """All 6 US-based provider labels must appear in source.

    Labels are public configuration; safe to lock at source-text level.
    """
    text = _module_text()
    missing = [label for label in _CANONICAL_LABELS if label not in text]
    assert not missing, (
        f"Missing canonical provider labels in engine/consensus_engine.py: "
        f"{missing}. The Layer-4 consensus surface is pinned at exactly 6 "
        f"US-based providers (iter-220 added Meta-Llama-4-Maverick-400B as "
        f"the 6th). Removing any of these silently downgrades the consensus "
        f"surface — an architectural change that must be deliberate."
    )


def test_six_canonical_env_keys_present():
    """All 6 environment-variable names must appear in source."""
    text = _module_text()
    missing = [key for key in _CANONICAL_ENV_KEYS if key not in text]
    assert not missing, (
        f"Missing canonical env-key references in engine/consensus_engine.py: "
        f"{missing}. Each provider's API key is read via os.environ.get(...) "
        f"and the var name appears verbatim in source."
    )


def test_six_canonical_model_ids_present():
    """All 6 wire-format model identifiers must appear in source.

    These are the exact strings sent to provider APIs — drift between
    documented model names (e.g. demo chip tooltips) and wire-format
    IDs would mean the demo claims one model but the engine talks to
    another. Locking the IDs at source level keeps the chain honest.
    """
    text = _module_text()
    missing = [m for m in _CANONICAL_MODEL_IDS if m not in text]
    assert not missing, (
        f"Missing canonical model IDs in engine/consensus_engine.py: "
        f"{missing}. The wire-format identifiers must match the demo's "
        f"chip tooltips and the docs (DEVPOST.md, README.md, "
        f"why_bitnet_b158.md, why_mind_mem_v3.md). A model-ID change here "
        f"without doc updates is exactly the iter-220 drift class."
    )


def test_nvidia_nemotron_uses_nim_endpoint():
    """NVIDIA Nemotron must use the NIM endpoint at integrate.api.nvidia.com.

    Iter-220 chose NIM (vs. building.nvidia.com or other gateways)
    because (a) NIM exposes an OpenAI-compatible chat-completions
    surface so we reuse `_call_openai_compatible`, and (b) NIM hosting
    is in US datacenters consistent with the HIPAA-compatible
    data-residency claim that gates the entire Layer-4 surface.
    """
    text = _module_text()
    assert "integrate.api.nvidia.com" in text, (
        "Meta Llama 4 Maverick (400B MoE) must call the NIM endpoint at "
        "https://integrate.api.nvidia.com — this URL is what the demo's "
        "chip tooltip and the DEVPOST.md L58 wire-format claim both "
        "promise. A change to a different gateway (e.g. build.nvidia.com) "
        "would break the architectural-residency claim."
    )


def test_available_providers_has_six_entries():
    """The `available_providers` tuple in verify_finding_consensus
    must have exactly 6 (key, label) pairs.

    This guards against "added a 7th provider but forgot to extend
    the available_providers list, so the dispatch-log count is wrong"
    — a real shape of bug from iter-128 era.
    """
    text = _module_text()
    # Find the available_providers list-comprehension block. The pattern
    # we lock is the 6-tuple literal that feeds the comprehension.
    match = re.search(
        r"available_providers\s*=\s*\[[^\]]+for\s+key,\s+label\s+in\s+\((.*?)\)\s+if\s+key\s*\]",
        text,
        re.DOTALL,
    )
    assert match is not None, (
        "Could not locate the available_providers list-comprehension in "
        "engine/consensus_engine.py. The block was added in iter-128 and "
        "should look like:\n"
        "    available_providers = [\n"
        "        label for key, label in (\n"
        "            (openai_key, 'OpenAI-GPT-5.5'),\n"
        "            ... (5 more pairs) ...\n"
        "        )\n"
        "        if key\n"
        "    ]"
    )
    body = match.group(1)
    # Each (key, label) pair contains exactly one comma-separated tuple.
    # Count parenthesized tuples — one per provider.
    pair_count = len(re.findall(r"\([^()]*,\s*\"[A-Za-z0-9.\-]+\"\s*\)", body))
    assert pair_count == 6, (
        f"available_providers tuple has {pair_count} (key, label) pairs; "
        f"iter-220 pinned 6. If you're adding a 7th provider, extend this "
        f"pin's expected count AND the canonical-labels / env-keys / "
        f"model-ids tuples above so the architectural decision is "
        f"deliberate, not silent drift."
    )


def test_consensus_engine_module_loads():
    """Sanity: the module imports cleanly. Catches a syntax error
    that would otherwise only show up in the integration tests.
    """
    import importlib

    try:
        mod = importlib.import_module("engine.consensus_engine")
    except Exception as exc:  # pragma: no cover - defensive
        raise AssertionError(
            f"engine.consensus_engine failed to import: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    # The module exposes verify_finding_consensus + its sync wrapper.
    assert hasattr(mod, "verify_finding_consensus")
    assert hasattr(mod, "verify_finding_consensus_sync")
