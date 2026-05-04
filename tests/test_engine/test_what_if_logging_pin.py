"""Pin: structured logging in `engine.what_if` is PHI-safe and complete.

Locks the iter-45 evidence-chain ratchet: every `simulate_*` function emits
a `what_if_*` structured log; logged extras contain only counts/categorical
metadata (never drug names, allergen text, or descriptions).
"""
import logging

from engine.what_if import (
    simulate_add_medication,
    simulate_remove_medication,
    simulate_swap_medication,
)


# Inputs chosen to produce a deterministic critical signal via the allergy
# path (penicillin + amoxicillin is the canonical cross-reactive pair) without
# needing the offline interaction table to carry a "contraindicated" entry.
_PATIENT_ID = "SYN-MRN-WHATIF-PIN"
_BASELINE_MEDS = ["lisinopril 10 mg"]
_CRITICAL_ALLERGY = ["penicillin"]
_CRITICAL_NEW_MED = "amoxicillin 500 mg"


def _record(caplog, event_name):
    """Return the first record whose `message` matches the event name."""
    for r in caplog.records:
        if r.message == event_name:
            return r
    return None


def test_simulate_add_emits_entry_log(caplog):
    caplog.set_level(logging.DEBUG, logger="engine.what_if")
    simulate_add_medication(
        patient_id=_PATIENT_ID,
        new_medication="amoxicillin 500 mg",
        current_medications=_BASELINE_MEDS,
        allergies=[],
        observations=[],
    )
    rec = _record(caplog, "what_if_add_entry")
    assert rec is not None, "expected what_if_add_entry log"
    assert rec.scenario == "add"
    assert rec.current_med_count == len(_BASELINE_MEDS)
    assert rec.allergy_count == 0
    assert rec.observation_count == 0


def test_simulate_add_critical_emits_warning(caplog):
    caplog.set_level(logging.DEBUG, logger="engine.what_if")
    simulate_add_medication(
        patient_id=_PATIENT_ID,
        new_medication=_CRITICAL_NEW_MED,
        current_medications=_BASELINE_MEDS,
        allergies=_CRITICAL_ALLERGY,
        observations=[],
    )
    rec = _record(caplog, "what_if_add_critical_risk")
    assert rec is not None, "expected critical-risk warning when adding amoxicillin under penicillin allergy"
    assert rec.levelno == logging.WARNING
    assert rec.safe_to_proceed is False


def test_simulate_add_safe_emits_info_outcome(caplog):
    caplog.set_level(logging.DEBUG, logger="engine.what_if")
    simulate_add_medication(
        patient_id=_PATIENT_ID,
        new_medication="amoxicillin 500 mg",
        current_medications=_BASELINE_MEDS,
        allergies=[],
        observations=[],
    )
    rec = _record(caplog, "what_if_add_outcome")
    assert rec is not None, "expected outcome log on the safe path"
    assert rec.levelno == logging.INFO
    assert rec.safe_to_proceed is True


def test_simulate_remove_emits_outcome_log(caplog):
    caplog.set_level(logging.DEBUG, logger="engine.what_if")
    simulate_remove_medication(
        patient_id=_PATIENT_ID,
        remove_medication="ibuprofen 800 mg",
        current_medications=_BASELINE_MEDS + ["ibuprofen 800 mg"],
        allergies=[],
        observations=[],
    )
    entry = _record(caplog, "what_if_remove_entry")
    outcome = _record(caplog, "what_if_remove_outcome")
    assert entry is not None
    assert outcome is not None
    assert outcome.levelno == logging.INFO
    assert isinstance(outcome.resolved_risk_count, int)


def test_simulate_swap_emits_outcome_log(caplog):
    caplog.set_level(logging.DEBUG, logger="engine.what_if")
    simulate_swap_medication(
        patient_id=_PATIENT_ID,
        remove_medication="ibuprofen 800 mg",
        add_medication="acetaminophen 500 mg",
        current_medications=_BASELINE_MEDS + ["ibuprofen 800 mg"],
        allergies=[],
        observations=[],
    )
    entry = _record(caplog, "what_if_swap_entry")
    outcome = _record(caplog, "what_if_swap_outcome")
    assert entry is not None
    assert outcome is not None
    assert outcome.scenario == "swap"
    assert outcome.direction in {"safer", "neutral", "riskier"}


def test_what_if_logs_never_emit_drug_names(caplog):
    """PHI-safety: arity counts and categorical fields only.

    Guards against future regressions where a maintainer adds drug names,
    allergen text, or description strings into the log extras.
    """
    caplog.set_level(logging.DEBUG, logger="engine.what_if")

    secret_drug = "WARFARIN_SECRET_5MG"
    secret_allergen = "PENICILLIN_SECRET"

    simulate_add_medication(
        patient_id=_PATIENT_ID,
        new_medication=secret_drug,
        current_medications=[secret_drug],
        allergies=[secret_allergen],
        observations=[],
    )
    simulate_remove_medication(
        patient_id=_PATIENT_ID,
        remove_medication=secret_drug,
        current_medications=[secret_drug],
        allergies=[secret_allergen],
        observations=[],
    )
    simulate_swap_medication(
        patient_id=_PATIENT_ID,
        remove_medication=secret_drug,
        add_medication=secret_drug,
        current_medications=[secret_drug],
        allergies=[secret_allergen],
        observations=[],
    )

    for rec in caplog.records:
        if rec.name != "engine.what_if":
            continue
        for value in vars(rec).values():
            text = repr(value)
            assert secret_drug not in text, f"drug name leaked in log: {rec.message}"
            assert secret_allergen not in text, f"allergen leaked in log: {rec.message}"
