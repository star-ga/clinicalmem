"""Pin: structured logging in `engine.clinical_memory` is PHI-safe.

Iter 18 + 23 + 38 + 43 + 45 + 48 + 53 + 58 + 64 + 71 + 77 + 82 + 87 +
92 + 98 ratchet trajectory. Iter 103 audited `engine/clinical_memory.py`
(899 LOC, 10 logger calls, 11.1/kloc — second-worst module after
bitnet_classifier). Found 6 logger calls leaking `e` (the exception
object) directly via `%s`-formatting — PHI / secret discipline
violations:

  - mind-mem init failure (config paths, DB connection strings, API URLs)
  - medication ingest exception (drug names, dosage strings, patient IDs)
  - condition ingest exception (ICD codes, problem-list text)
  - allergy ingest exception (allergen names, reaction text)
  - observation ingest exception (lab values, biometric numbers)
  - hybrid-search exception (query text — may quote prompt + cohort drug names)

Fix: convert all 6 to structured `extra={"error_type": type(e).__name__}`
form. The pin file enforces:

  1. No bare `logger.X("...: %s", e)` patterns survive in this module.
  2. Each new structured event has a stable name + error_type field.

Future maintainers re-introducing `%s, e` patterns fail the gate.

iter 138 (T4 round 27) extension:
The module had grown to 1081 LOC at 9.3 calls/kloc — LOWEST density
of any module in the engine. Five safety-critical query methods went
silent (medication_safety_check, detect_contradictions,
verify_audit_chain success path, patient_summary, clinical_handoff).
Net 10 -> 18 logger calls (+8 events, density 9.3 -> 15.1/kloc).
PHI-safe metadata: patient_id (synthetic Synthea), counts only —
NEVER medication names, allergens, condition names, observation
values, or audit-chain entry content.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_MODULE_PATH = _REPO_ROOT / "engine" / "clinical_memory.py"


def test_no_bare_str_e_logger_calls():
    """No logger.X(..., e) pattern may leak the exception object's
    str representation. The structured form is logger.X(name,
    extra={"error_type": type(e).__name__})."""
    src = _MODULE_PATH.read_text()
    # Match: logger.{level}("...something %s ... ", ..., e[,)])
    # Catches both "logger.warning('...: %s', e)" and similar.
    bare_pattern = re.compile(
        r'\blogger\.(?:debug|info|warning|error|exception|critical)\('
        r'[^)]*%s[^)]*,\s*e[,\s)]',
        re.DOTALL,
    )
    matches = bare_pattern.findall(src)
    assert not matches, (
        f"engine/clinical_memory.py contains {len(matches)} "
        f"logger.X(..., e) call(s) that leak str(e) directly. "
        f"Convert to structured form: "
        f"logger.X('event_name', extra={{'error_type': type(e).__name__}}). "
        f"First match: {matches[0][:120]}"
    )


def test_structured_event_names_use_module_prefix():
    """Iter-103 convention: each structured event in clinical_memory
    is prefixed `clinical_memory_*` so log aggregators can filter
    by module without parsing the source. Catches future logs that
    introduce a non-prefixed event name."""
    src = _MODULE_PATH.read_text()
    # Find all logger.X("event_name", extra=...) patterns
    event_pattern = re.compile(
        r'logger\.(?:debug|info|warning|error|exception|critical)\(\s*'
        r'"([a-z_]+(?:_[a-z]+)*)"\s*,\s*extra\s*=',
    )
    events = event_pattern.findall(src)
    if not events:
        # If no structured events with extra=, this test is informational
        return
    bad = [e for e in events if not e.startswith("clinical_memory_")]
    assert not bad, (
        f"clinical_memory.py has structured events without the "
        f"`clinical_memory_*` prefix: {bad}. Module convention "
        f"(iter 103) is to prefix all structured events."
    )


def test_ingest_failure_logs_error_type_only(caplog, tmp_path):
    """Wire-up test: trigger a real ingest failure and assert no
    str(e) content leaks into the structured log record."""
    caplog.set_level(logging.ERROR, logger="engine.clinical_memory")

    sentinel = "ZZZ_PATIENT_DOSAGE_LEAK_TOKEN"

    # Inject a class that raises with the sentinel as its message.
    class FakeFhir:
        def get_medications(self):
            raise RuntimeError(sentinel)
        def get_conditions(self): return []
        def get_allergies(self): return []
        def get_observations(self): return []

    # Spin up a minimal ClinicalMemEngine on a tmp data dir.
    from engine.clinical_memory import ClinicalMemEngine
    engine = ClinicalMemEngine(data_dir=str(tmp_path))
    pid = "synthea-pt-test-iter103"

    # Ingest: medications path will raise; the structured log should
    # log the event with error_type but NOT the sentinel.
    try:
        engine.ingest_fhir_resources(pid, FakeFhir())
    except Exception:
        pass  # the engine catches medication errors internally; if
              # the whole ingest re-raises, we still verify caplog

    medication_records = [
        r for r in caplog.records
        if r.name == "engine.clinical_memory"
        and r.message == "clinical_memory_medications_ingest_failed"
    ]
    if medication_records:
        rec = medication_records[0]
        assert rec.error_type == "RuntimeError", (
            f"medications_ingest_failed must log error_type=RuntimeError; "
            f"got {getattr(rec, 'error_type', 'MISSING')}"
        )
        # Sentinel-leak scrub
        for value in vars(rec).values():
            assert sentinel not in repr(value), (
                f"clinical_memory_medications_ingest_failed leaked "
                f"str(e) sentinel into structured log"
            )


# ─── iter-138 (T4 round 27) ratchet — five new structured events ─────────

_EXPECTED_LOGGER_FLOOR_ITER138 = 18  # was 10 pre-iter-138


def test_clinical_memory_logger_floor_iter138():
    """Pin a logger-call floor (>= 18) so silent-removal regressions of
    the iter-138 events fail the gate.

    Iter-138 added 5 structured events on previously-silent paths
    (med_safety_check, contradictions_check, audit_chain_verification_ok,
    patient_summary, handoff_generated) plus 3 restructured WARNING events
    on verify_audit_chain failure paths. Net 10 -> 18 logger calls.
    """
    src = _MODULE_PATH.read_text()
    calls = re.findall(
        r"\blogger\.(debug|info|warning|error|critical)\(",
        src,
    )
    assert len(calls) >= _EXPECTED_LOGGER_FLOOR_ITER138, (
        f"engine/clinical_memory.py logger-call count regressed: "
        f"{len(calls)} < floor {_EXPECTED_LOGGER_FLOOR_ITER138}. "
        f"A structured event from iter-138 was silently removed."
    )


def test_med_safety_check_emits_info_event_iter138(caplog, tmp_path):
    """`medication_safety_check` emits `clinical_memory_med_safety_check`
    INFO. Pre-iter-138 the load-bearing safety query was silent."""
    from engine.clinical_memory import ClinicalMemEngine

    engine = ClinicalMemEngine(data_dir=str(tmp_path / "iter138_a"))
    with caplog.at_level(logging.INFO, logger="engine.clinical_memory"):
        engine.medication_safety_check("pt-test-iter138")
    matched = [
        r for r in caplog.records
        if r.levelno == logging.INFO
        and "clinical_memory_med_safety_check" in r.getMessage()
    ]
    assert matched, (
        "medication_safety_check emitted no "
        "`clinical_memory_med_safety_check` INFO event."
    )
    rec = matched[0]
    assert getattr(rec, "patient_id", None) == "pt-test-iter138"
    assert getattr(rec, "med_count", "MISSING") == 0
    assert getattr(rec, "interaction_count", "MISSING") == 0


def test_detect_contradictions_emits_info_event_iter138(caplog, tmp_path):
    """`detect_contradictions` emits
    `clinical_memory_contradictions_check` INFO."""
    from engine.clinical_memory import ClinicalMemEngine

    engine = ClinicalMemEngine(data_dir=str(tmp_path / "iter138_b"))
    with caplog.at_level(logging.INFO, logger="engine.clinical_memory"):
        engine.detect_contradictions("pt-test-iter138-b")
    matched = [
        r for r in caplog.records
        if r.levelno == logging.INFO
        and "clinical_memory_contradictions_check" in r.getMessage()
    ]
    assert matched, (
        "detect_contradictions emitted no "
        "`clinical_memory_contradictions_check` INFO event."
    )
    rec = matched[0]
    assert getattr(rec, "patient_id", None) == "pt-test-iter138-b"
    assert getattr(rec, "contradiction_count", "MISSING") == 0
    # Iter-294: type_counts dict added so auditors can see the breakdown
    # without re-querying. For empty cohort it's an empty dict.
    type_counts = getattr(rec, "type_counts", "MISSING")
    assert isinstance(type_counts, dict), (
        f"detect_contradictions log must carry `type_counts` dict "
        f"(iter-294); got {type_counts!r}"
    )
    assert type_counts == {}, (
        f"empty patient detect_contradictions must report empty "
        f"type_counts dict, got {type_counts!r}"
    )


def test_verify_audit_chain_success_emits_debug_event_iter138(caplog, tmp_path):
    """`verify_audit_chain` emits
    `clinical_memory_audit_chain_verification_ok` DEBUG on the success
    path. Pre-iter-138 only failures logged, success was silent."""
    from engine.clinical_memory import ClinicalMemEngine

    engine = ClinicalMemEngine(data_dir=str(tmp_path / "iter138_c"))
    with caplog.at_level(logging.DEBUG, logger="engine.clinical_memory"):
        engine.verify_audit_chain()
    matched = [
        r for r in caplog.records
        if r.levelno == logging.DEBUG
        and "clinical_memory_audit_chain_verification_ok" in r.getMessage()
    ]
    assert matched, (
        "verify_audit_chain success path emitted no "
        "`clinical_memory_audit_chain_verification_ok` DEBUG event."
    )


def test_patient_summary_log_contains_no_phi_content_iter138(caplog, tmp_path):
    """`clinical_memory_patient_summary` must NEVER include medication
    names, allergens, condition names, or observation values in any
    field of the log record.

    Same paranoia as iter-103/108/113/118/123/128/133 sentinel-leak
    scrubs.
    """
    from engine.clinical_memory import ClinicalBlock, ClinicalMemEngine

    sentinel_med = "ZZZ_LEAK_SENTINEL_MED_iter138"
    sentinel_allergen = "ZZZ_LEAK_SENTINEL_ALLERGEN_iter138"
    sentinel_condition = "ZZZ_LEAK_SENTINEL_COND_iter138"

    engine = ClinicalMemEngine(data_dir=str(tmp_path / "iter138_d"))
    pid = "pt-test-iter138-d"
    engine._patient_blocks[pid] = [
        ClinicalBlock(
            block_id="b1", patient_id=pid,
            resource_type="MedicationRequest",
            title="med", content="med content",
            metadata={"medication_name": sentinel_med},
            timestamp="2026-01-01", source="test",
        ),
        ClinicalBlock(
            block_id="b2", patient_id=pid,
            resource_type="AllergyIntolerance",
            title="allergy", content="allergy content",
            metadata={"allergen": sentinel_allergen},
            timestamp="2026-01-01", source="test",
        ),
        ClinicalBlock(
            block_id="b3", patient_id=pid,
            resource_type="Condition",
            title="condition", content="condition content",
            metadata={"condition_name": sentinel_condition},
            timestamp="2026-01-01", source="test",
        ),
    ]

    with caplog.at_level(logging.DEBUG, logger="engine.clinical_memory"):
        engine.patient_summary(pid)

    sentinels = (sentinel_med, sentinel_allergen, sentinel_condition)
    for rec in caplog.records:
        for s in sentinels:
            assert s not in rec.getMessage(), (
                f"PHI sentinel leaked into log message"
            )
        for value in vars(rec).values():
            if isinstance(value, str):
                for s in sentinels:
                    assert s not in value, (
                        f"PHI sentinel leaked into record attribute"
                    )
            elif isinstance(value, (list, tuple)):
                for item in value:
                    if isinstance(item, str):
                        for s in sentinels:
                            assert s not in item, (
                                f"PHI sentinel leaked into record list"
                            )


def test_module_exports_remain_stable_iter138():
    """Source-level pin: the public class + key methods that downstream
    callers (run_clinical_regression_eval.py, federation_mock_demo.py)
    import by name must remain stable."""
    text = _MODULE_PATH.read_text()
    for required in (
        "class ClinicalMemEngine",
        "def medication_safety_check(",
        "def detect_contradictions(",
        "def verify_audit_chain(",
        "def patient_summary(",
        "def clinical_handoff(",
    ):
        assert required in text, (
            f"engine/clinical_memory.py removed or renamed {required!r} — "
            f"this is part of the public API the SaMD audit-replay "
            f"harness depends on."
        )


# ─── iter-171 (T4 round 33) ratchet — 5 new structured events ───────────────

_EXPECTED_LOGGER_FLOOR_ITER171 = 23  # was 18 pre-iter-171


def test_clinical_memory_logger_floor_iter171():
    """Floor bump 18 -> 23 after iter-171 closed 4 silent paths:
    bundle ingest entry, empty-patient recall abstention, audit-trail
    read (mind-mem + fallback both logged), explain-conflict abstention.
    Net 18 -> 23 logger calls (density 15.1 -> 18.2/kloc)."""
    src = _MODULE_PATH.read_text()
    calls = re.findall(
        r"\blogger\.(debug|info|warning|error|critical)\(",
        src,
    )
    assert len(calls) >= _EXPECTED_LOGGER_FLOOR_ITER171, (
        f"engine/clinical_memory.py logger-call count regressed: "
        f"{len(calls)} < floor {_EXPECTED_LOGGER_FLOOR_ITER171}. "
        f"A structured event from iter-171 was silently removed."
    )


def test_bundle_ingest_emits_debug_event_iter171(caplog):
    """`ingest_from_bundle` emits `clinical_memory_bundle_ingest_start`
    DEBUG with per-resource-type counts but never resource bodies."""
    from engine.clinical_memory import ClinicalMemEngine
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        engine = ClinicalMemEngine(data_dir=tmp)
        bundle = {
            "entry": [
                {"resource": {"resourceType": "Patient", "id": "p1"}},
                {"resource": {"resourceType": "Condition", "id": "c1",
                              "code": {"text": "ZZZ_PHI_SENTINEL_CONDITION"}}},
                {"resource": {"resourceType": "Condition", "id": "c2",
                              "code": {"text": "ZZZ_PHI_SENTINEL_CONDITION_2"}}},
            ],
        }
        with caplog.at_level(logging.DEBUG, logger="engine.clinical_memory"):
            engine.ingest_from_bundle(bundle, "p1")

    matches = [r for r in caplog.records
               if r.message == "clinical_memory_bundle_ingest_start"]
    assert matches, "bundle_ingest_start event missing"
    rec = matches[0]
    assert rec.levelno == logging.DEBUG
    assert rec.patient_id == "p1"
    assert rec.total_entries == 3
    assert rec.resource_type_counts.get("Condition") == 2

    # PHI sentinel must NOT leak into any record attribute
    for r in caplog.records:
        for value in vars(r).values():
            if isinstance(value, str):
                assert "ZZZ_PHI_SENTINEL_CONDITION" not in value, (
                    "PHI sentinel leaked into log record"
                )


def test_recall_empty_patient_emits_info_event_iter171(caplog):
    """`recall` on a patient with zero blocks emits
    `clinical_memory_recall_empty_patient` INFO. The query string must
    NOT appear in any log record (queries can carry clinician narrative)."""
    from engine.clinical_memory import ClinicalMemEngine
    import tempfile

    sentinel_query = "ZZZ_PHI_SENTINEL_QUERY_drug_allergy_for_John"
    with tempfile.TemporaryDirectory() as tmp:
        engine = ClinicalMemEngine(data_dir=tmp)
        with caplog.at_level(logging.INFO, logger="engine.clinical_memory"):
            result = engine.recall("never-ingested-patient", sentinel_query)

    assert result.blocks == []
    matches = [r for r in caplog.records
               if r.message == "clinical_memory_recall_empty_patient"]
    assert matches, "recall_empty_patient event missing"
    rec = matches[0]
    assert rec.levelno == logging.INFO
    assert rec.patient_id == "never-ingested-patient"
    assert rec.block_count == 0
    assert rec.results == 0

    # Query string MUST NOT appear in any log record (downstream PHI)
    for r in caplog.records:
        for value in vars(r).values():
            if isinstance(value, str):
                assert sentinel_query not in value, (
                    "PHI sentinel query leaked into log record"
                )


def test_audit_trail_read_emits_debug_event_iter171(caplog):
    """`get_audit_trail` emits `clinical_memory_audit_trail_read` DEBUG
    on both mind-mem-backed and fallback backends."""
    from engine.clinical_memory import ClinicalMemEngine
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        engine = ClinicalMemEngine(data_dir=tmp)
        with caplog.at_level(logging.DEBUG, logger="engine.clinical_memory"):
            entries = engine.get_audit_trail(limit=5)

    matches = [r for r in caplog.records
               if r.message == "clinical_memory_audit_trail_read"]
    assert matches, "audit_trail_read event missing"
    rec = matches[0]
    assert rec.levelno == logging.DEBUG
    assert rec.limit == 5
    assert rec.backend in ("mind_mem", "fallback")
    assert isinstance(rec.returned_count, int)
    assert rec.returned_count == len(entries)


def test_explain_conflict_abstained_emits_info_event_iter171(caplog):
    """`explain_clinical_conflict` emits
    `clinical_memory_explain_conflict_abstained` INFO when the
    abstention gate fires. The reason field is a structural string,
    not patient narrative."""
    from engine.clinical_memory import ClinicalMemEngine
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        engine = ClinicalMemEngine(data_dir=tmp)
        with caplog.at_level(logging.INFO, logger="engine.clinical_memory"):
            narrative = engine.explain_clinical_conflict(
                "no-such-patient", conflict_index=0
            )

    assert narrative.abstained is True
    matches = [r for r in caplog.records
               if r.message == "clinical_memory_explain_conflict_abstained"]
    assert matches, "explain_conflict_abstained event missing"
    rec = matches[0]
    assert rec.levelno == logging.INFO
    assert rec.patient_id == "no-such-patient"
    assert rec.reason in ("no_conflicts", "conflict_index_out_of_range")
    assert rec.contradiction_count == 0


# ────────────────────────────────────────────────────────────────────
# Iter-201 T4 round-39 ratchet — four previously-silent paths closed
# in engine/clinical_memory.py: hybrid-empty fallback, mindmem complete,
# fallback complete, negation-query scoring.
# Density 18.2 -> 20.4/kloc.
# ────────────────────────────────────────────────────────────────────


_EXPECTED_LOGGER_FLOOR_ITER201 = 27  # was 23 pre-iter-201 (iter-138 added
                                       # 5; iter-176 added 5 more; iter-201
                                       # adds 4 = 27)


def test_clinical_memory_logger_floor_iter201():
    """Pin a logger-call floor (>= 27) so silent-removal regressions of
    the iter-201 events fail the gate. Iter-201 added 4 DEBUG events on
    silent recall paths (mindmem_empty_falling_back, recall_mindmem_complete,
    recall_fallback_complete, negation_query_scored)."""
    src = _MODULE_PATH.read_text()
    calls = re.findall(
        r"\blogger\.(debug|info|warning|error|critical)\(",
        src,
    )
    assert len(calls) >= _EXPECTED_LOGGER_FLOOR_ITER201, (
        f"engine/clinical_memory.py logger-call count regressed: "
        f"{len(calls)} < floor {_EXPECTED_LOGGER_FLOOR_ITER201}. "
        f"A structured event from iter-201 was silently removed."
    )


def test_recall_fallback_complete_emits_debug_event_iter201(caplog, tmp_path):
    """`_recall_fallback` emits `clinical_memory_recall_fallback_complete`
    DEBUG when recall completes via the approximate BM25 path. Pre-iter-201
    the success path was silent — operators couldn't distinguish hybrid-
    backend recalls from approximate BM25 recalls in audit-aggregate
    statistics. PHI-safe: scalars only, no query/result content."""
    from engine.clinical_memory import ClinicalMemEngine, ClinicalBlock

    engine = ClinicalMemEngine(data_dir=str(tmp_path / "iter201_fallback"))
    # Pre-populate with a synthetic block so recall has something to score
    block = ClinicalBlock(
        block_id="iter201-blk-1",
        patient_id="pt-iter201",
        resource_type="MedicationRequest",
        title="Synthetic test block for iter-201 ratchet",
        content="patient on lisinopril 10mg daily; no allergies",
        metadata={"medication_name": "lisinopril"},
        timestamp="2026-05-06T09:30:00Z",
        source="iter201-test",
    )
    engine._patient_blocks["pt-iter201"] = [block]
    # Force fallback path (no hybrid backend in test env)
    engine._hybrid_backend = None

    with caplog.at_level(logging.DEBUG, logger="engine.clinical_memory"):
        engine.recall("pt-iter201", "hypertension medication")

    matched = [
        r for r in caplog.records
        if r.levelno == logging.DEBUG
        and "clinical_memory_recall_fallback_complete" in r.getMessage()
    ]
    assert matched, (
        "_recall_fallback emitted no "
        "`clinical_memory_recall_fallback_complete` DEBUG event."
    )
    rec = matched[0]
    assert rec.patient_id == "pt-iter201"
    assert rec.block_count == 1
    # PHI sentinel scan — recall log records must not carry the block content.
    for r in caplog.records:
        rd = r.__dict__
        assert "lisinopril 10mg" not in str(rd), (
            f"block content leaked into log record: {rd}"
        )


def test_negation_query_scored_emits_debug_event_iter201(caplog, tmp_path):
    """`_recall_fallback` emits `clinical_memory_negation_query_scored`
    DEBUG when the query is detected as a negation query (gets 0.5x score
    multiplier on blocks without negation markers). The query string itself
    is NOT logged — only counts + length — to keep PHI safe."""
    from engine.clinical_memory import ClinicalMemEngine, ClinicalBlock

    engine = ClinicalMemEngine(data_dir=str(tmp_path / "iter201_neg"))
    block = ClinicalBlock(
        block_id="iter201-neg-blk-1",
        patient_id="pt-iter201-neg",
        resource_type="AllergyIntolerance",
        title="Allergy record",
        content="no known drug allergies",
        metadata={"allergen": "none"},
        timestamp="2026-05-06T09:30:00Z",
        source="iter201-test",
    )
    engine._patient_blocks["pt-iter201-neg"] = [block]
    engine._hybrid_backend = None

    sentinel_query = "ZZZ_SECRET_QUERY_TOKEN does the patient have no allergies"
    with caplog.at_level(logging.DEBUG, logger="engine.clinical_memory"):
        engine.recall("pt-iter201-neg", sentinel_query)

    matched = [
        r for r in caplog.records
        if r.levelno == logging.DEBUG
        and "clinical_memory_negation_query_scored" in r.getMessage()
    ]
    assert matched, (
        "_recall_fallback negation branch emitted no "
        "`clinical_memory_negation_query_scored` DEBUG event."
    )
    # Sentinel scan — query string must NEVER appear in any log record.
    for r in caplog.records:
        assert "ZZZ_SECRET_QUERY_TOKEN" not in str(r.__dict__), (
            f"negation-query log record leaked the query string: {r.__dict__}"
        )
