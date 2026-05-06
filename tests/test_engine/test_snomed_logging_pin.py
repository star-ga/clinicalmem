"""Pin: structured logging in `engine.snomed_client` is PHI-safe.

Iter 71 closes the evidence-chain gap on the SNOMED CT lookup path
(was 28 dec / 2 ev = 7.1%, the lowest ratio after audit_export_part11
in iter 64). SNOMED queries pass drug names + allergens to public APIs
on the wire, but local logs use length / class metadata to keep
records categorical.

Iter 222 (T4 round 46) ratchet — closes 3 silent paths in the
lowest-density engine module (23.9 → 38.3/kloc):
  • snomed_snowstorm_non_200 (WARNING) — Snowstorm primary lookup
    fails non-200; operators need status code to triage 401/429/503
    that silently demoted the surface to UMLS / empty results.
  • snomed_umls_non_200 (WARNING) — same shape for UMLS fallback;
    when both return non-200 the cross-reactivity surface is silently
    empty.
  • snomed_within_class_cross_reactive (WARNING) — within-class
    cross-reactivity hit (was silent; mirror-shape of the
    beta_lactam cross-class WARNING already pinned). Categorical
    drug_class only, never the allergen / med strings.

Pinned event surface:
  • snomed_search_start (DEBUG) — term length + semantic_tag (FHIR
    standard) + UMLS-fallback availability flag.
  • snomed_search_complete (DEBUG) — result_count + source
    ("snowstorm" / "umls" / "none").
  • snomed_snowstorm_error (WARNING) — error_type only — exception
    messages from httpx can carry response-body fragments that may
    include adjacent narrative.
  • snomed_snowstorm_non_200 (WARNING, iter-222) — status_code +
    endpoint + term_length only; never term text or response body.
  • snomed_umls_error (WARNING) — same discipline.
  • snomed_umls_non_200 (WARNING, iter-222) — status_code + endpoint
    + term_length only; never the API key (apiKey is in params but
    must not leak).
  • snomed_beta_lactam_cross_reactive (WARNING) — categorical class
    pair only ("penicillin", "cephalosporin"), never the literal
    allergen / medication strings (clinical input may carry adjacent
    free-text narrative).
  • snomed_within_class_cross_reactive (WARNING, iter-222) — drug_class
    only; mirror-shape of the cross-class beta-lactam pin.
"""
import logging

import pytest

from engine.snomed_client import (
    is_allergy_cross_reactive,
    search_snomed,
)


def _record(caplog, event_name):
    for r in caplog.records:
        if r.name == "engine.snomed_client" and r.message == event_name:
            return r
    return None


def test_search_snomed_emits_start_and_complete_logs(caplog, monkeypatch):
    """search_snomed() emits structured start + complete logs."""
    caplog.set_level(logging.DEBUG, logger="engine.snomed_client")

    # Mock the HTTP path so we don't depend on the public Snowstorm API.
    import engine.snomed_client as mod

    def fake_snowstorm(term, tag):
        return [
            mod.SnomedConcept(concept_id="123", term="warfarin", semantic_tag="substance"),
        ]

    monkeypatch.setattr(mod, "_search_snowstorm", fake_snowstorm)

    results = search_snomed("warfarin")
    assert len(results) == 1

    start = _record(caplog, "snomed_search_start")
    complete = _record(caplog, "snomed_search_complete")
    assert start is not None
    assert complete is not None
    assert start.term_length == len("warfarin")
    assert complete.result_count == 1
    assert complete.source in {"snowstorm", "umls", "none"}


def test_search_snomed_complete_records_none_source_when_empty(caplog, monkeypatch):
    """When neither Snowstorm nor UMLS returns hits, source is 'none'."""
    caplog.set_level(logging.DEBUG, logger="engine.snomed_client")
    import engine.snomed_client as mod

    monkeypatch.setattr(mod, "_search_snowstorm", lambda *a, **k: [])
    monkeypatch.setattr(mod, "UMLS_API_KEY", "")  # disable UMLS fallback

    search_snomed("zzz_nonexistent_drug")
    complete = _record(caplog, "snomed_search_complete")
    assert complete is not None
    assert complete.source == "none"
    assert complete.result_count == 0


def test_beta_lactam_cross_reactive_emits_warning(caplog):
    """penicillin allergy + cephalosporin must fire the WARNING-level
    cross-reactive log with categorical class pair only."""
    caplog.set_level(logging.DEBUG, logger="engine.snomed_client")
    is_allergy_cross_reactive("penicillin", "cefazolin")
    rec = _record(caplog, "snomed_beta_lactam_cross_reactive")
    assert rec is not None
    assert rec.levelno == logging.WARNING
    assert rec.allergy_class == "penicillin"
    assert rec.med_class == "cephalosporin"


def test_snomed_logs_never_emit_drug_name_string(caplog, monkeypatch):
    """Sentinel-leak scrub: a sentinel drug name passed in must NEVER
    appear in any engine.snomed_client log record."""
    caplog.set_level(logging.DEBUG, logger="engine.snomed_client")
    import engine.snomed_client as mod
    monkeypatch.setattr(mod, "_search_snowstorm", lambda *a, **k: [])
    monkeypatch.setattr(mod, "UMLS_API_KEY", "")

    sentinel = "ZZZ_SECRET_DRUG_NAME_TOKEN"
    search_snomed(sentinel)
    is_allergy_cross_reactive("penicillin", "cefazolin")  # also check post-hit logs

    for rec in caplog.records:
        if rec.name != "engine.snomed_client":
            continue
        for value in vars(rec).values():
            text = repr(value)
            assert sentinel not in text, (
                f"drug name leaked into log {rec.message}"
            )


# -------------------------------------------------------------------- #
# iter-222 T4 round-46 ratchet — 3 new event-presence + PHI tests for
# the silent paths just closed in engine/snomed_client.py.
# -------------------------------------------------------------------- #


def test_snowstorm_non_200_emits_warning_iter222(caplog, monkeypatch):
    """Non-200 from Snowstorm fires WARNING with status_code + endpoint
    + term_length — never term text or response body."""
    caplog.set_level(logging.DEBUG, logger="engine.snomed_client")

    class FakeResp:
        status_code = 503

        @staticmethod
        def json():
            raise AssertionError("non-200 path must not parse body")

    import engine.snomed_client as mod

    def fake_get(url, params=None, timeout=None):
        return FakeResp()

    monkeypatch.setattr(mod.httpx, "get", fake_get)

    # Bypass the LRU cache so this test is independent.
    mod.search_snomed.cache_clear()
    monkeypatch.setattr(mod, "UMLS_API_KEY", "")  # disable UMLS so we don't double-warn

    SENTINEL = "ZZZ_PHI_TERM_TOKEN"
    results = mod._search_snowstorm(SENTINEL)
    assert results == []

    rec = _record(caplog, "snomed_snowstorm_non_200")
    assert rec is not None, "expected snomed_snowstorm_non_200 WARNING when Snowstorm returns 503"
    assert rec.levelno == logging.WARNING
    assert rec.status_code == 503
    assert rec.endpoint == "snowstorm/MAIN/concepts"
    assert rec.term_length == len(SENTINEL)
    # PHI scrub: sentinel term must NEVER reach this WARNING record
    for value in vars(rec).values():
        assert SENTINEL not in repr(value), (
            f"PHI term leaked into snomed_snowstorm_non_200 record"
        )


def test_umls_non_200_emits_warning_iter222(caplog, monkeypatch):
    """Non-200 from UMLS fires WARNING with status_code + endpoint +
    term_length — never the API key, never the term."""
    caplog.set_level(logging.DEBUG, logger="engine.snomed_client")

    class FakeResp:
        status_code = 401

        @staticmethod
        def json():
            raise AssertionError("non-200 path must not parse body")

    import engine.snomed_client as mod

    def fake_get(url, params=None, timeout=None):
        return FakeResp()

    monkeypatch.setattr(mod.httpx, "get", fake_get)
    monkeypatch.setattr(mod, "UMLS_API_KEY", "fake-umls-key-for-test")

    SENTINEL = "ZZZ_PHI_UMLS_TOKEN"
    results = mod._search_umls_snomed(SENTINEL)
    assert results == []

    rec = _record(caplog, "snomed_umls_non_200")
    assert rec is not None, "expected snomed_umls_non_200 WARNING when UMLS returns 401"
    assert rec.levelno == logging.WARNING
    assert rec.status_code == 401
    assert rec.endpoint == "uts-ws/search/current"
    assert rec.term_length == len(SENTINEL)
    # PHI scrub: term + API key must NEVER reach the WARNING record
    for value in vars(rec).values():
        text = repr(value)
        assert SENTINEL not in text, (
            f"PHI term leaked into snomed_umls_non_200 record"
        )
        assert "fake-umls-key-for-test" not in text, (
            f"UMLS API key leaked into snomed_umls_non_200 record"
        )


def test_within_class_cross_reactive_emits_warning_iter222(caplog):
    """Within-class cross-reactivity (e.g. amoxicillin allergy + ampicillin
    prescription) fires WARNING with drug_class — never allergen / med
    strings (mirror-shape of beta_lactam cross-class WARNING)."""
    caplog.set_level(logging.DEBUG, logger="engine.snomed_client")
    SENTINEL_ALLERGY = "AMOXICILLIN_PHI_TOKEN"  # noqa: not real PHI; sentinel for scrub
    SENTINEL_MED = "AMPICILLIN_PHI_TOKEN"

    # The match logic uses substring containment, so we have to use real
    # class-member tokens; the sentinel scrub below targets the actual
    # allergen / medication strings we passed in.
    is_allergy_cross_reactive("amoxicillin", "ampicillin")

    rec = _record(caplog, "snomed_within_class_cross_reactive")
    assert rec is not None, (
        "expected snomed_within_class_cross_reactive WARNING when an "
        "allergy and medication share a drug class"
    )
    assert rec.levelno == logging.WARNING
    assert rec.drug_class == "penicillin"  # both amoxicillin + ampicillin in penicillin

    # PHI scrub: actual allergen / med strings must NOT appear in the record
    for value in vars(rec).values():
        text = repr(value)
        assert "amoxicillin" not in text.lower(), (
            f"allergen string leaked into snomed_within_class_cross_reactive"
        )
        assert "ampicillin" not in text.lower(), (
            f"medication string leaked into snomed_within_class_cross_reactive"
        )


def test_snomed_logger_floor_iter222():
    """Ratchet floor: engine/snomed_client.py must keep >= 8 logger
    calls (was 5 pre-iter-222). This is the source-level density pin
    that prevents silent regressions of the iter-222 ratchet."""
    import re
    from pathlib import Path

    src = (Path(__file__).resolve().parent.parent.parent / "engine" / "snomed_client.py").read_text()
    n_logs = len(re.findall(r"logger\.\w+\(", src))
    assert n_logs >= 8, (
        f"engine/snomed_client.py logger-call count regressed to {n_logs}; "
        f"iter-222 ratchet pinned floor at 8 (3 silent paths closed: "
        f"snowstorm_non_200, umls_non_200, within_class_cross_reactive)."
    )
