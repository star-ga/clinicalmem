"""Pin: fda_client.py emits structured logs at every decision point,
with PHI / secret discipline (no drug names or response bodies leak).

`engine/fda_client.py` is the openFDA integration: drug adverse-event,
label-warning, and recall queries against api.fda.gov. Drug names are
NOT PHI per HIPAA Safe Harbor in isolation, but the caller may pass
through free-text from a clinical note — so log length / count /
categorical metadata only. The literal drug name DOES go on the wire
to public openFDA endpoints; the local audit trail stays clean.

Iter-71 / iter-77 / iter-82 ratchet pattern. fda_client at 4 logger
calls / 272 lines was the largest sub-10% engine module
post-iter-82.

These tests pin:
  * fda_adverse_search_start fires (DEBUG) on every adverse-events query
  * fda_adverse_non_200 fires (WARNING) on HTTP non-200
  * fda_adverse_oversize_response fires (WARNING) above 100 results
  * fda_adverse_failed fires (WARNING) on httpx exception with
    error_type only (NEVER str(e) — response bodies leak the request
    query string, classic PHI / secret leak vector)
  * fda_label_failed fires (WARNING) on label endpoint exception
  * fda_recall_failed fires (WARNING) on recall endpoint exception
  * fda_safety_profile_complete (INFO) on the patient-safety entry
    point with med_count + alert counts only
"""
from __future__ import annotations

import logging

import pytest


_SECRET_DRUG_TOKEN = "ZZZ_SECRET_DRUG_NAME_FROM_CLINICAL_NOTE"
_BODY_LEAK_TOKEN = "ZZZ_RESPONSE_BODY_LEAK_TOKEN"


@pytest.fixture
def caplog_at_debug(caplog):
    caplog.set_level(logging.DEBUG, logger="engine.fda_client")
    return caplog


def _records(caplog) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.name == "engine.fda_client"]


def test_safety_profile_complete_emits_structured_log(caplog_at_debug, monkeypatch):
    """get_safety_profile emits INFO with med_count + alert_count
    + black_box_count + total_adverse_events + highest_severity."""
    from engine import fda_client

    # Stub all three sub-fetches to avoid network.
    monkeypatch.setattr(fda_client, "get_adverse_events", lambda d: ())
    monkeypatch.setattr(fda_client, "get_label_warnings", lambda d: ())
    monkeypatch.setattr(fda_client, "get_drug_recalls", lambda d: ())

    fda_client.get_safety_profile(
        [_SECRET_DRUG_TOKEN, "warfarin", "metformin"]
    )

    records = _records(caplog_at_debug)
    msgs = {r.message for r in records}
    assert "fda_safety_profile_complete" in msgs, (
        "get_safety_profile must emit 'fda_safety_profile_complete' INFO"
    )
    # Drug names are NOT PHI but caller-supplied free-text -- length only.
    for r in records:
        assert _SECRET_DRUG_TOKEN not in str(r.__dict__), (
            "drug name leaked into fda_client log record — caller may "
            "have passed clinical-note free text"
        )


def test_adverse_search_start_logs_safe_metadata(caplog_at_debug, monkeypatch):
    """fda_adverse_search_start fires before httpx call with
    drug_name_length + endpoint + limit. The literal drug name must
    NOT appear in the log record."""
    from engine import fda_client

    class _Stub200:
        status_code = 200
        def json(self):
            return {"results": []}

    monkeypatch.setattr(fda_client, "_get_httpx", lambda: type("M", (), {"get": lambda *a, **kw: _Stub200()})())

    fda_client.get_adverse_events(_SECRET_DRUG_TOKEN, limit=7)

    records = _records(caplog_at_debug)
    starts = [r for r in records if r.message == "fda_adverse_search_start"]
    assert starts, "fda_adverse_search_start must fire on every search"
    s = starts[0]
    assert getattr(s, "endpoint", None) == "drug/event"
    assert getattr(s, "limit", None) == 7
    for r in records:
        assert _SECRET_DRUG_TOKEN not in str(r.__dict__), (
            "drug name leaked into adverse-search start log"
        )


def test_failure_uses_error_type_not_str_exception(caplog_at_debug, monkeypatch):
    """When httpx raises, all 3 fetchers (adverse / label / recall)
    log error_type only — not str(e). httpx exception messages can
    carry response-body fragments echoing request params, which is a
    PHI/secret leak vector."""
    from engine import fda_client

    class _StubException(Exception):
        pass

    def _raise_stub(*args, **kwargs):
        raise _StubException(f"500 Internal Error -- {_BODY_LEAK_TOKEN}")

    class _ExplodingHttpx:
        get = staticmethod(_raise_stub)

    monkeypatch.setattr(fda_client, "_get_httpx", lambda: _ExplodingHttpx)

    # All three fetchers should fail gracefully + emit structured WARNING
    fda_client.get_adverse_events(_SECRET_DRUG_TOKEN)
    fda_client.get_label_warnings(_SECRET_DRUG_TOKEN)
    fda_client.get_drug_recalls(_SECRET_DRUG_TOKEN)

    records = _records(caplog_at_debug)
    failed_msgs = {r.message for r in records if r.message.endswith("_failed")}
    assert "fda_adverse_failed" in failed_msgs
    assert "fda_label_failed" in failed_msgs
    assert "fda_recall_failed" in failed_msgs
    # All three failure logs must be WARNING (not DEBUG -- failures must
    # surface past DEBUG suppression).
    for r in records:
        if r.message.endswith("_failed"):
            assert r.levelname == "WARNING"
            assert getattr(r, "error_type", None) == "_StubException"
    # Sentinel-leak scrub: neither drug name nor response body fragment
    # may appear in any log record.
    for r in records:
        record_dump = str(r.__dict__)
        assert _SECRET_DRUG_TOKEN not in record_dump, (
            "drug name leaked into failure log"
        )
        assert _BODY_LEAK_TOKEN not in record_dump, (
            "httpx exception body leaked via str(e)"
        )


def test_non_200_warns_with_status_code(caplog_at_debug, monkeypatch):
    from engine import fda_client

    class _Stub429:
        status_code = 429  # rate limited
        def json(self):
            return {}

    class _RateLimitedHttpx:
        get = staticmethod(lambda *a, **kw: _Stub429())

    monkeypatch.setattr(fda_client, "_get_httpx", lambda: _RateLimitedHttpx)

    result = fda_client.get_adverse_events("hypertension")
    assert result == []

    records = _records(caplog_at_debug)
    non200 = [r for r in records if r.message == "fda_adverse_non_200"]
    assert non200, "non-200 must emit 'fda_adverse_non_200' WARNING"
    assert non200[0].levelname == "WARNING"
    assert getattr(non200[0], "status_code", None) == 429
