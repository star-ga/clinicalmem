"""Pin: trials_client.py emits structured logs at every decision point,
with PHI discipline (no patient-condition strings leaking into log records).

`engine/trials_client.py` queries ClinicalTrials.gov v2 to match patient
conditions to active recruiting trials. Patient conditions are clinical
diagnoses — NOT PHI per HIPAA Safe Harbor in isolation, but combined with
cohort size + geography they are quasi-identifiers. Log length / count /
categorical metadata only; never the literal condition string in the
local audit trail. (The string DOES go on the wire to a public CT.gov
endpoint; the audit log stays clean.)

Iter-71 ratchet pattern:
  * trials_search_start fires on every search_trials() call
  * trials_search_non_200 fires (WARNING) on non-200 HTTP responses
  * trials_search_oversize_response fires (WARNING) above 200 studies
  * trials_search_complete fires after parse
  * trials_search_failed fires (WARNING) on httpx exception with
    error_type only -- never str(e) (response body fragments leak).
  * trials_match_start / trials_match_complete (INFO) on the
    higher-level patient-match entry point with condition/med counts only.
"""
from __future__ import annotations

import logging

import pytest


_SECRET_CONDITION_TOKEN = "ZZZ_SECRET_PATIENT_CONDITION_TOKEN"


@pytest.fixture
def caplog_at_debug(caplog):
    caplog.set_level(logging.DEBUG, logger="engine.trials_client")
    return caplog


def _records(caplog) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.name == "engine.trials_client"]


def test_match_start_and_complete_emit_structured_logs(caplog_at_debug, monkeypatch):
    """match_patient_to_trials emits start + complete INFO with counts only."""
    from engine import trials_client

    # Stub search_trials so the test doesn't hit the network.
    def _fake_search(condition, status="RECRUITING", max_results=10):
        return ()

    monkeypatch.setattr(trials_client, "search_trials", _fake_search)

    trials_client.match_patient_to_trials(
        conditions=[_SECRET_CONDITION_TOKEN, "diabetes mellitus type 2"],
        medications=["warfarin"],
        max_per_condition=3,
    )

    records = _records(caplog_at_debug)
    msgs = {r.message for r in records}
    assert "trials_match_start" in msgs, (
        "match_patient_to_trials must emit 'trials_match_start' INFO"
    )
    assert "trials_match_complete" in msgs, (
        "match_patient_to_trials must emit 'trials_match_complete' INFO"
    )

    # PHI scrub: condition strings must NOT appear in any log record.
    for r in records:
        record_dump = str(r.__dict__)
        assert _SECRET_CONDITION_TOKEN not in record_dump, (
            "patient condition string leaked into trials log record — "
            "PHI risk; only count/length should be logged"
        )


def test_search_failure_uses_error_type_not_str_exception(caplog_at_debug, monkeypatch):
    """When httpx raises, log error_type only — not str(e). httpx exception
    messages can carry response-body fragments (server error pages echo
    request params), which is a PHI/secret leak vector."""
    from engine import trials_client

    class _StubException(Exception):
        pass

    _BODY_LEAK_TOKEN = "ZZZ_RESPONSE_BODY_LEAK_TOKEN"

    def _raise_stub(*args, **kwargs):
        raise _StubException(f"500 Internal Error -- {_BODY_LEAK_TOKEN}")

    monkeypatch.setattr(trials_client.httpx, "get", _raise_stub)
    # Bypass lru_cache by using a fresh condition string each test.
    trials_client.search_trials.cache_clear()

    result = trials_client.search_trials(_SECRET_CONDITION_TOKEN)
    assert result == ()  # graceful failure

    records = _records(caplog_at_debug)
    failed = [r for r in records if r.message == "trials_search_failed"]
    assert failed, (
        "trials_client must emit 'trials_search_failed' WARNING on httpx "
        "exception"
    )
    assert failed[0].levelname == "WARNING"
    # error_type recorded but body content NOT leaked
    assert getattr(failed[0], "error_type", None) == "_StubException"
    for r in records:
        record_dump = str(r.__dict__)
        assert _BODY_LEAK_TOKEN not in record_dump, (
            "httpx exception body leaked into log record via str(e)"
        )
        assert _SECRET_CONDITION_TOKEN not in record_dump, (
            "patient condition leaked into log record"
        )


def test_search_start_fires_with_safe_metadata(caplog_at_debug, monkeypatch):
    """trials_search_start must fire with cond_length + status + max_results
    -- never the condition string."""
    from engine import trials_client

    class _Stub200:
        status_code = 200
        def json(self):
            return {"studies": []}

    monkeypatch.setattr(trials_client.httpx, "get", lambda *a, **kw: _Stub200())
    trials_client.search_trials.cache_clear()

    trials_client.search_trials(
        f"{_SECRET_CONDITION_TOKEN}-condition-name", status="RECRUITING", max_results=5
    )

    records = _records(caplog_at_debug)
    start_records = [r for r in records if r.message == "trials_search_start"]
    assert start_records, "trials_search_start must fire on every search"
    s = start_records[0]
    # safe fields recorded
    assert getattr(s, "status_filter", None) == "RECRUITING"
    assert getattr(s, "max_results", None) == 5
    # condition string NOT in any field
    for r in records:
        assert _SECRET_CONDITION_TOKEN not in str(r.__dict__), (
            "search start log leaked condition string"
        )


def test_non_200_warns_with_status_code(caplog_at_debug, monkeypatch):
    from engine import trials_client

    class _Stub429:
        status_code = 429  # rate limited
        def json(self):
            return {}

    monkeypatch.setattr(trials_client.httpx, "get", lambda *a, **kw: _Stub429())
    trials_client.search_trials.cache_clear()

    result = trials_client.search_trials("hypertension")
    assert result == ()

    records = _records(caplog_at_debug)
    non200 = [r for r in records if r.message == "trials_search_non_200"]
    assert non200, "non-200 response must emit 'trials_search_non_200' WARNING"
    assert non200[0].levelname == "WARNING"
    assert getattr(non200[0], "status_code", None) == 429
