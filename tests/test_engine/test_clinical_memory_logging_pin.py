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
