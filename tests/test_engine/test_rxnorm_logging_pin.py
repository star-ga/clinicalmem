"""Pin: structured logging in `engine.rxnorm_client` is PHI-safe.

Iter 234 (T4 round 48) — closes the evidence-chain gap on the RxNorm
client. Pre-iter-234 the module had 5 old-style positional %s logger
calls passing **drug names directly as positional args**:

    logger.debug("RxNorm exact lookup failed for %s: %s", clean, e)
    logger.debug("RxNorm approximate lookup failed for %s: %s", clean, e)
    logger.warning("NIH interaction API failed: %s", e)
    logger.debug("Resolved %s → RxCUI %s (%s)", med, concept.rxcui, concept.name)
    logger.warning("Could not resolve medication: %s", med)

Drug names from clinical context can be PHI-adjacent (especially the
`med` string which may carry dosage + brand name + clinician narrative).
Old-style positional %s args land in `record.args` and get rendered to
the final log line — same exposure as a structured `extra` field, but
without the discipline of a categorical contract. **Iter-234 rewrites
all 5 calls to structured `extra={}` form** + adds 4 silent-path
closures, so the module emits 9 structured events with PHI-safe
metadata only (name_length, error_type, status_code, rxcui, tty,
batch counts).

Pinned event surface
====================
  • rxnorm_exact_lookup_error (DEBUG) — error_type + name_length
  • rxnorm_exact_non_200 (WARNING) — status_code + endpoint + name_length
  • rxnorm_approx_lookup_error (DEBUG) — error_type + name_length
  • rxnorm_approx_non_200 (WARNING) — status_code + endpoint + name_length
  • rxnorm_interaction_non_200 (WARNING) — status_code + endpoint + rxcui_count
  • rxnorm_interaction_error (WARNING) — error_type + rxcui_count
  • rxnorm_resolved (DEBUG) — name_length + rxcui (public NIH ID) + tty
  • rxnorm_unresolved (WARNING) — name_length only
  • rxnorm_normalize_batch_complete (INFO) — total + resolved + unresolved counts

PHI safety contract
===================
  • Drug names NEVER appear in any `record.args` or `record.<field>` —
    only `name_length` is logged.
  • Exception messages NEVER appear — only `error_type`. httpx
    exceptions can carry the request URL which contains drug names
    in `name=` / `term=` query params; PHI-sentinel scrub catches.
  • RxCUIs are public NIH IDs and ARE logged for operational triage.

Pin family lineage: 10th cross-pin family (after iter-178/183/188/193/
198/203/223/228/232).
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

import httpx

import engine.rxnorm_client as mod

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent

_EXPECTED_LOGGER_FLOOR_ITER234 = 9
_PHI_SENTINEL_DRUG = "ZZZ_PHI_DRUG_TOKEN_iter234"


def _record(caplog, event_name):
    """Find the first record on engine.rxnorm_client matching event_name."""
    for r in caplog.records:
        if r.name == "engine.rxnorm_client" and r.message == event_name:
            return r
    return None


def test_rxnorm_logger_floor_iter234():
    """Source-level pin: engine/rxnorm_client.py must keep >= 9 logger
    calls (was 5 pre-iter-234; iter-234 added 4 new structured events
    while refactoring all 5 existing calls to structured form)."""
    src = (_REPO_ROOT / "engine" / "rxnorm_client.py").read_text()
    n_logs = len(re.findall(r"logger\.\w+\(", src))
    assert n_logs >= _EXPECTED_LOGGER_FLOOR_ITER234, (
        f"engine/rxnorm_client.py logger-call count regressed to {n_logs}; "
        f"iter-234 ratchet pinned floor at {_EXPECTED_LOGGER_FLOOR_ITER234} "
        f"(5 refactored from old %s positional form + 4 silent paths "
        f"closed: exact_non_200, approx_non_200, interaction_non_200, "
        f"normalize_batch_complete)."
    )


def test_rxnorm_no_old_style_positional_drug_logs_iter234():
    """Source-level PHI-discipline pin: NO `logger.X("...%s...", med, ...)`
    pattern with drug-name args. iter-234 audit caught 5 such sites; this
    pin ensures none come back. Catches the regression class where a
    maintainer reverts to `logger.warning("Could not resolve %s", med)`."""
    src = (_REPO_ROOT / "engine" / "rxnorm_client.py").read_text()
    # Old-style format-string call with drug-related positional arg names.
    # Specifically catch the iter-234 anti-patterns.
    forbidden_patterns = (
        r'logger\.\w+\([^)]*"[^"]*%s[^"]*",\s*clean\b',
        r'logger\.\w+\([^)]*"[^"]*%s[^"]*",\s*med\b',
        r'logger\.\w+\([^)]*"[^"]*%s[^"]*",\s*drug_name\b',
    )
    for pat in forbidden_patterns:
        m = re.search(pat, src)
        assert m is None, (
            f"PHI regression: engine/rxnorm_client.py contains a "
            f"`logger.X(\"...%s...\", <drug_name_var>)` pattern: "
            f"{m.group()!r}. iter-234 rewrote all such sites to "
            f"structured extra={{name_length: len(name)}}. Drug names "
            f"must NEVER reach record.args."
        )


def test_resolve_rxcui_exact_non_200_emits_warning_iter234(caplog, monkeypatch):
    """Non-200 from rxnav exact lookup fires WARNING with status_code
    + endpoint + name_length only. PHI-safe: drug name never reaches
    the record (length-only)."""
    class FakeResp:
        status_code = 503

        @staticmethod
        def json():
            raise AssertionError("non-200 path must not parse body")

    def fake_get(url, params=None, timeout=None):
        # Both exact AND approx hit fake_get; both should emit non_200.
        return FakeResp()

    monkeypatch.setattr(mod.httpx, "get", fake_get)
    mod.resolve_rxcui.cache_clear()

    with caplog.at_level(logging.DEBUG, logger="engine.rxnorm_client"):
        result = mod.resolve_rxcui(_PHI_SENTINEL_DRUG)
    assert result is None

    rec = _record(caplog, "rxnorm_exact_non_200")
    assert rec is not None, (
        "expected rxnorm_exact_non_200 WARNING when rxnav returns 503"
    )
    assert rec.levelno == logging.WARNING
    assert rec.status_code == 503
    assert rec.endpoint == "rxnav/rxcui"
    assert rec.name_length == len(_PHI_SENTINEL_DRUG)

    # PHI scrub: drug name must NEVER reach the record
    for value in vars(rec).values():
        assert _PHI_SENTINEL_DRUG not in repr(value), (
            f"drug-name PHI sentinel leaked into rxnorm_exact_non_200 "
            f"record: {rec.message}"
        )


def test_resolve_rxcui_approx_non_200_emits_warning_iter234(caplog, monkeypatch):
    """Non-200 from rxnav approximate fires WARNING — fires when both
    paths fail (exact already non-200; approx also non-200)."""
    class FakeResp:
        status_code = 429

        @staticmethod
        def json():
            raise AssertionError("non-200 path must not parse body")

    def fake_get(url, params=None, timeout=None):
        return FakeResp()

    monkeypatch.setattr(mod.httpx, "get", fake_get)
    mod.resolve_rxcui.cache_clear()

    with caplog.at_level(logging.DEBUG, logger="engine.rxnorm_client"):
        mod.resolve_rxcui(_PHI_SENTINEL_DRUG)

    rec = _record(caplog, "rxnorm_approx_non_200")
    assert rec is not None, (
        "expected rxnorm_approx_non_200 WARNING when rxnav approxTerm "
        "returns 429"
    )
    assert rec.levelno == logging.WARNING
    assert rec.status_code == 429
    assert rec.endpoint == "rxnav/approximateTerm"
    assert rec.name_length == len(_PHI_SENTINEL_DRUG)

    for value in vars(rec).values():
        assert _PHI_SENTINEL_DRUG not in repr(value), (
            "drug-name PHI sentinel leaked into rxnorm_approx_non_200"
        )


def test_get_interactions_for_list_non_200_emits_warning_iter234(caplog, monkeypatch):
    """Non-200 from NIH Drug Interaction API fires WARNING with
    status_code + endpoint + rxcui_count. Layer-3 detection is
    load-bearing for the safety pipeline; operators MUST see the
    silent demote-to-empty-list."""
    class FakeResp:
        status_code = 502

        @staticmethod
        def json():
            raise AssertionError("non-200 path must not parse body")

    def fake_get(url, params=None, timeout=None):
        return FakeResp()

    monkeypatch.setattr(mod.httpx, "get", fake_get)

    rxcuis = ["12345", "67890"]
    with caplog.at_level(logging.DEBUG, logger="engine.rxnorm_client"):
        result = mod.get_interactions_for_list(rxcuis)
    assert result == []

    rec = _record(caplog, "rxnorm_interaction_non_200")
    assert rec is not None, (
        "expected rxnorm_interaction_non_200 WARNING when NIH "
        "interaction API returns 502"
    )
    assert rec.levelno == logging.WARNING
    assert rec.status_code == 502
    assert rec.endpoint == "rxnav/interaction/list"
    assert rec.rxcui_count == 2


def test_normalize_medication_list_emits_batch_complete_iter234(caplog, monkeypatch):
    """`normalize_medication_list` emits an INFO `rxnorm_normalize_batch_complete`
    with total + resolved + unresolved counts after each batch — useful
    for ops dashboards measuring resolution rate over time."""
    # Make all resolutions fail so we get deterministic counts.
    monkeypatch.setattr(mod, "resolve_rxcui", lambda x: None)

    sentinel_a = "ZZZ_DRUG_A_iter234"
    sentinel_b = "ZZZ_DRUG_B_iter234"
    with caplog.at_level(logging.DEBUG, logger="engine.rxnorm_client"):
        mod.normalize_medication_list([sentinel_a, sentinel_b])

    rec = _record(caplog, "rxnorm_normalize_batch_complete")
    assert rec is not None, (
        "expected rxnorm_normalize_batch_complete INFO after batch"
    )
    assert rec.levelno == logging.INFO
    assert rec.total_count == 2
    assert rec.resolved_count == 0
    assert rec.unresolved_count == 2

    # PHI scrub: neither drug name reaches the batch-complete record
    for value in vars(rec).values():
        text = repr(value)
        assert sentinel_a not in text, "drug A leaked into batch_complete"
        assert sentinel_b not in text, "drug B leaked into batch_complete"


def test_rxnorm_unresolved_emits_warning_with_phi_scrub_iter234(caplog, monkeypatch):
    """`rxnorm_unresolved` WARNING carries name_length only; the drug
    name itself must NEVER appear in the record."""
    monkeypatch.setattr(mod, "resolve_rxcui", lambda x: None)

    with caplog.at_level(logging.DEBUG, logger="engine.rxnorm_client"):
        mod.normalize_medication_list([_PHI_SENTINEL_DRUG])

    rec = _record(caplog, "rxnorm_unresolved")
    assert rec is not None, "expected rxnorm_unresolved WARNING"
    assert rec.levelno == logging.WARNING
    assert rec.name_length == len(_PHI_SENTINEL_DRUG)

    for value in vars(rec).values():
        assert _PHI_SENTINEL_DRUG not in repr(value), (
            f"drug-name PHI sentinel leaked into rxnorm_unresolved record"
        )


def test_rxnorm_resolved_emits_debug_with_phi_scrub_iter234(caplog, monkeypatch):
    """`rxnorm_resolved` DEBUG carries name_length + RxCUI (public NIH
    ID, OK to log) + tty (public taxonomy). NEVER the drug name."""
    fake_concept = mod.RxConcept(rxcui="12345", name="acetaminophen", tty="IN")
    monkeypatch.setattr(mod, "resolve_rxcui", lambda x: fake_concept)

    with caplog.at_level(logging.DEBUG, logger="engine.rxnorm_client"):
        mod.normalize_medication_list([_PHI_SENTINEL_DRUG])

    rec = _record(caplog, "rxnorm_resolved")
    assert rec is not None, "expected rxnorm_resolved DEBUG"
    assert rec.levelno == logging.DEBUG
    assert rec.name_length == len(_PHI_SENTINEL_DRUG)
    assert rec.rxcui == "12345"
    assert rec.tty == "IN"

    for value in vars(rec).values():
        assert _PHI_SENTINEL_DRUG not in repr(value), (
            "drug-name PHI sentinel leaked into rxnorm_resolved record"
        )
