"""Pin: structured logging in `engine.flow_runner` is PHI-safe and complete.

The flow runner is the audit-replayable executor — every clinical
decision flows through it and the audit chain records the resulting
plan_hash + inputs_hash + output_hash. Logging discipline:

  - Hashes are non-reversible; safe to log (prefixed to 16 chars).
  - flow_name + profile + kernel are public configuration.
  - input_keys are dict KEYS (not values); safe.
  - NEVER log: input dict values (medications, patient_id raw,
    observation values), node expression text, parsed flow source.

Pinned event surface:
  - flow_plan_hash_computed (DEBUG)
  - flow_contract_parsed (DEBUG)
  - flow_replay_verified (INFO when matches=True)
  - flow_replay_mismatch (WARNING)
  - flow_execute_start (INFO)
  - flow_execute_complete (INFO when failed=0, WARNING when failed>0)
"""
import logging
from pathlib import Path

import pytest

from engine.flow_runner import (
    compute_plan_hash,
    parse_flow_contract,
    verify_replay,
    execute,
    list_flow_names,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_FLOWS_DIR = _REPO_ROOT / "flows"


@pytest.fixture(scope="module")
def first_flow_name() -> str:
    names = list_flow_names(_FLOWS_DIR)
    assert names, "expected at least one .flow.mind file in flows/"
    return names[0]


def _record(caplog, event_name):
    for r in caplog.records:
        if r.name == "engine.flow_runner" and r.message == event_name:
            return r
    return None


def test_compute_plan_hash_emits_debug_log(caplog, first_flow_name):
    caplog.set_level(logging.DEBUG, logger="engine.flow_runner")
    digest = compute_plan_hash(first_flow_name, flows_dir=_FLOWS_DIR)
    rec = _record(caplog, "flow_plan_hash_computed")
    assert rec is not None
    assert rec.flow_name == first_flow_name
    assert rec.plan_hash_prefix == digest[:16]
    assert isinstance(rec.canonical_size_bytes, int)
    assert rec.canonical_size_bytes > 0


def test_parse_flow_contract_emits_debug_log(caplog, first_flow_name):
    caplog.set_level(logging.DEBUG, logger="engine.flow_runner")
    contract = parse_flow_contract(first_flow_name, flows_dir=_FLOWS_DIR)
    rec = _record(caplog, "flow_contract_parsed")
    assert rec is not None
    assert rec.flow_name == contract.name
    assert rec.node_count == len(contract.nodes)
    assert rec.invariant_count == len(contract.invariants)


def test_verify_replay_match_emits_info(caplog, first_flow_name):
    caplog.set_level(logging.DEBUG, logger="engine.flow_runner")
    expected = compute_plan_hash(first_flow_name, flows_dir=_FLOWS_DIR)
    caplog.clear()
    result = verify_replay(first_flow_name, expected, flows_dir=_FLOWS_DIR)
    assert result.matches
    rec = _record(caplog, "flow_replay_verified")
    assert rec is not None
    assert rec.levelno == logging.INFO
    assert rec.matches is True


def test_verify_replay_mismatch_emits_warning(caplog, first_flow_name):
    caplog.set_level(logging.DEBUG, logger="engine.flow_runner")
    bad_hash = "00" * 32  # deliberately wrong
    result = verify_replay(first_flow_name, bad_hash, flows_dir=_FLOWS_DIR)
    assert not result.matches
    rec = _record(caplog, "flow_replay_mismatch")
    assert rec is not None
    assert rec.levelno == logging.WARNING
    assert rec.matches is False


def test_execute_emits_start_and_complete(caplog, first_flow_name):
    caplog.set_level(logging.DEBUG, logger="engine.flow_runner")
    SECRET_DRUG = "ZZZ_SECRET_FLOW_RUNNER_DRUG"
    SECRET_PATIENT = "ZZZ_SECRET_PATIENT_ID"
    result = execute(
        first_flow_name,
        inputs={
            "medications": [SECRET_DRUG],
            "patient_id": SECRET_PATIENT,
        },
        flows_dir=_FLOWS_DIR,
    )
    start = _record(caplog, "flow_execute_start")
    complete = _record(caplog, "flow_execute_complete")
    assert start is not None
    assert complete is not None
    assert start.flow_name == result.flow_name

    # PHI scrub: SECRET strings must NEVER appear in any flow_runner log.
    for rec in caplog.records:
        if rec.name != "engine.flow_runner":
            continue
        for value in vars(rec).values():
            text = repr(value)
            assert SECRET_DRUG not in text, (
                f"secret drug name leaked in log {rec.message}"
            )
            assert SECRET_PATIENT not in text, (
                f"secret patient_id leaked in log {rec.message}"
            )

    # The start log must record input KEYS (not values) so auditors can
    # see what was passed without re-leaking PHI.
    assert "medications" in start.input_keys
    assert "patient_id" in start.input_keys


def test_flow_runner_logs_record_only_hash_prefixes(caplog, first_flow_name):
    """No log record should ever contain a full SHA-256 hex digest.

    All hashes are prefixed to 16 chars. A full 64-char hex digest in a
    log record means a future maintainer logged the raw hash field —
    correct behavior would be `digest[:16]`.
    """
    caplog.set_level(logging.DEBUG, logger="engine.flow_runner")
    compute_plan_hash(first_flow_name, flows_dir=_FLOWS_DIR)
    parse_flow_contract(first_flow_name, flows_dir=_FLOWS_DIR)

    import re
    hex64_re = re.compile(r"[0-9a-f]{64}")
    for rec in caplog.records:
        if rec.name != "engine.flow_runner":
            continue
        for value in vars(rec).values():
            text = repr(value)
            assert hex64_re.search(text) is None, (
                f"full SHA-256 hex digest in log record {rec.message}: "
                f"prefix to 16 chars instead."
            )
