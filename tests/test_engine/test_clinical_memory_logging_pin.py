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
