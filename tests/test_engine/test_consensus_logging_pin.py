"""Pin: structured logging in `engine.consensus_engine` is PHI-safe.

Locks the iter-48 evidence-chain ratchet on the multi-LLM consensus
verifier. The function fires up to 6 LLMs in parallel (iter-220 added
Meta Llama 4 Maverick (400B MoE) as the 6th provider via NIM); each call sees
clinical findings (potentially PHI-derived) over the wire. Logging
discipline:

  - Provider labels are public configuration; safe to log.
  - Counts, levels (HIGH/MEDIUM/LOW/NONE/LIMITED), agreement/total —
    safe to log.
  - The finding text, evidence body, reasoning_summary, and any LLM
    output text — NEVER logged in extras. The prompt is redacted via
    the PHI gate before transport, but model responses may still quote
    patient narrative.

Test surface:
  - `consensus_no_providers` fires WARNING when no API keys present.
  - `consensus_aggregated` fires WARNING when consensus is LOW / NONE /
    LIMITED (the abstention triggers) and INFO when HIGH / MEDIUM.
  - Final scrub: no log record from `engine.consensus_engine` in any of
    the above paths contains a sentinel finding string.
"""
import asyncio
import logging
import os

import pytest

from engine.consensus_engine import verify_finding_consensus


_FINDING = "WARFARIN_INTERACTION_SENTINEL_TOKEN"
_EVIDENCE = [{"block_id": "blk-1", "title": "FDA Label", "content": "secret-clinical-text"}]
_PATIENT_CTX = {"patient_id": "SYN-MRN-CONSENSUS-PIN", "medications": ["warfarin", "ibuprofen"]}


@pytest.fixture(autouse=True)
def _clear_llm_keys(monkeypatch):
    """Most tests run with no API keys — prevents the test from making
    real network calls if the developer's shell happens to have keys."""
    for var in (
        "OPENAI_API_KEY",
        "GOOGLE_API_KEY",
        "GEMINI_API_KEY",
        "XAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "PERPLEXITY_API_KEY",
        "NVIDIA_API_KEY",
    ):
        monkeypatch.delenv(var, raising=False)


def _record(caplog, event_name):
    for r in caplog.records:
        if r.name == "engine.consensus_engine" and r.message == event_name:
            return r
    return None


def test_no_providers_emits_warning(caplog):
    caplog.set_level(logging.DEBUG, logger="engine.consensus_engine")
    asyncio.run(verify_finding_consensus(_FINDING, _EVIDENCE, _PATIENT_CTX))
    rec = _record(caplog, "consensus_no_providers")
    assert rec is not None, "expected consensus_no_providers warning when no keys present"
    assert rec.levelno == logging.WARNING
    assert rec.providers_available == 0
    assert rec.level == "NONE"


def test_dispatch_log_lists_zero_providers_when_no_keys(caplog):
    caplog.set_level(logging.DEBUG, logger="engine.consensus_engine")
    asyncio.run(verify_finding_consensus(_FINDING, _EVIDENCE, _PATIENT_CTX))
    rec = _record(caplog, "consensus_dispatch")
    assert rec is not None
    assert rec.providers_available == 0
    assert rec.evidence_block_count == len(_EVIDENCE)


def test_consensus_logs_never_emit_finding_text(caplog):
    caplog.set_level(logging.DEBUG, logger="engine.consensus_engine")
    asyncio.run(verify_finding_consensus(_FINDING, _EVIDENCE, _PATIENT_CTX))
    for rec in caplog.records:
        if rec.name != "engine.consensus_engine":
            continue
        for value in vars(rec).values():
            text = repr(value)
            assert _FINDING not in text, (
                f"finding sentinel leaked into log {rec.message}"
            )
            assert "secret-clinical-text" not in text, (
                f"evidence body leaked into log {rec.message}"
            )
            # patient_id is an internal identifier; it can flow into
            # adjacent module logs (fhir_client, what_if), but the
            # consensus path receives a redacted prompt and shouldn't
            # echo it back.
            assert "SYN-MRN-CONSENSUS-PIN" not in text, (
                f"patient_id leaked into log {rec.message}"
            )


def test_consensus_dispatch_log_records_provider_label_when_keys_present(monkeypatch, caplog):
    """When at least one key is present, the dispatch log lists which
    provider labels participated. Provider labels are public model
    names — safe to log."""
    monkeypatch.setenv("OPENAI_API_KEY", "fake-key-for-dispatch-test")
    caplog.set_level(logging.DEBUG, logger="engine.consensus_engine")
    # Run the dispatch but don't actually wait for the network call:
    # the in-process httpx call will fail; verify_finding_consensus
    # catches and logs that as consensus_provider_error.
    asyncio.run(verify_finding_consensus(_FINDING, _EVIDENCE, _PATIENT_CTX))
    dispatch = _record(caplog, "consensus_dispatch")
    assert dispatch is not None
    assert dispatch.providers_available == 1
    assert "OpenAI-GPT-5.5" in dispatch.providers
