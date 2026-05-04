"""Pin: structured logging in `engine.snomed_client` is PHI-safe.

Iter 71 closes the evidence-chain gap on the SNOMED CT lookup path
(was 28 dec / 2 ev = 7.1%, the lowest ratio after audit_export_part11
in iter 64). SNOMED queries pass drug names + allergens to public APIs
on the wire, but local logs use length / class metadata to keep
records categorical.

Pinned event surface:
  • snomed_search_start (DEBUG) — term length + semantic_tag (FHIR
    standard) + UMLS-fallback availability flag.
  • snomed_search_complete (DEBUG) — result_count + source
    ("snowstorm" / "umls" / "none").
  • snomed_snowstorm_error (WARNING) — error_type only — exception
    messages from httpx can carry response-body fragments that may
    include adjacent narrative.
  • snomed_umls_error (WARNING) — same discipline.
  • snomed_beta_lactam_cross_reactive (WARNING) — categorical class
    pair only ("penicillin", "cephalosporin"), never the literal
    allergen / medication strings (clinical input may carry adjacent
    free-text narrative).
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
