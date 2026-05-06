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


# ─── iter-181 (T4 round 35) ratchet — 3 new structured events ───────────────

from pathlib import Path
import re

_EXPECTED_LOGGER_FLOOR_ITER181 = 9  # was 6 pre-iter-181


def test_what_if_logger_floor_iter181():
    """Floor bump 6 -> 9 after iter-181 closed 3 silent paths:
    drug-not-in-list during remove (WARNING), swap-target-not-found
    (WARNING), recommendation-branch path (DEBUG). Density 15.0 ->
    20.4/kloc."""
    src_path = (
        Path(__file__).resolve().parent.parent.parent
        / "engine" / "what_if.py"
    )
    src = src_path.read_text()
    calls = re.findall(
        r"\blogger\.(debug|info|warning|error|critical)\(",
        src,
    )
    assert len(calls) >= _EXPECTED_LOGGER_FLOOR_ITER181, (
        f"engine/what_if.py logger-call count regressed: "
        f"{len(calls)} < floor {_EXPECTED_LOGGER_FLOOR_ITER181}. "
        f"A structured event from iter-181 was silently removed."
    )


def test_simulate_remove_emits_warning_when_drug_not_in_list_iter181(caplog):
    """`simulate_remove_medication` emits
    `what_if_remove_drug_not_in_list` WARNING when the caller asks
    to remove a medication that isn't in the current list (clinical-
    input-validation signal)."""
    import logging
    from engine.what_if import simulate_remove_medication

    sentinel_drug_not_in_list = "ZZZ_PHI_SENTINEL_drug_NOT_in_list_unique"
    with caplog.at_level(logging.WARNING, logger="engine.what_if"):
        result = simulate_remove_medication(
            patient_id="p-test",
            remove_medication=sentinel_drug_not_in_list,
            current_medications=["warfarin", "ibuprofen"],
            allergies=[],
            observations=[],
        )

    assert result.removed_risks == []
    matches = [r for r in caplog.records
               if r.message == "what_if_remove_drug_not_in_list"]
    assert matches, "what_if_remove_drug_not_in_list WARNING missing"
    rec = matches[0]
    assert rec.levelno == logging.WARNING
    assert rec.patient_id == "p-test"
    assert rec.scenario == "remove"
    assert rec.current_med_count == 2

    # PHI sentinel drug name must NOT leak into any log record
    for r in caplog.records:
        for value in vars(r).values():
            if isinstance(value, str):
                assert sentinel_drug_not_in_list not in value, (
                    "PHI sentinel drug name leaked into log record"
                )


def test_simulate_swap_emits_warning_when_remove_target_not_found_iter181(caplog):
    """`simulate_swap_medication` emits
    `what_if_swap_remove_target_not_found` WARNING when the swap-from
    medication isn't in the current list."""
    import logging
    from engine.what_if import simulate_swap_medication

    sentinel_drug = "ZZZ_PHI_SENTINEL_swap_target_unique"
    with caplog.at_level(logging.WARNING, logger="engine.what_if"):
        simulate_swap_medication(
            patient_id="p-test",
            remove_medication=sentinel_drug,
            add_medication="aspirin",
            current_medications=["warfarin", "ibuprofen"],
            allergies=[],
            observations=[],
        )

    matches = [r for r in caplog.records
               if r.message == "what_if_swap_remove_target_not_found"]
    assert matches, "what_if_swap_remove_target_not_found WARNING missing"
    rec = matches[0]
    assert rec.levelno == logging.WARNING
    assert rec.patient_id == "p-test"
    assert rec.scenario == "swap"
    assert rec.current_med_count == 2

    for r in caplog.records:
        for value in vars(r).values():
            if isinstance(value, str):
                assert sentinel_drug not in value, (
                    "PHI sentinel drug name leaked into log record"
                )


def test_simulate_add_emits_recommendation_path_debug_iter181(caplog):
    """`simulate_add_medication` emits `what_if_recommendation_path`
    DEBUG with the branch label ('no_new_risks' / 'critical' /
    'monitored'). The no-new-risks branch is the simplest to trigger
    deterministically (use a no-known-interaction safe medication)."""
    import logging
    from engine.what_if import simulate_add_medication

    with caplog.at_level(logging.DEBUG, logger="engine.what_if"):
        result = simulate_add_medication(
            patient_id="p-test",
            new_medication="acetaminophen",
            current_medications=["lisinopril"],
            allergies=[],
            observations=[],
        )

    matches = [r for r in caplog.records
               if r.message == "what_if_recommendation_path"]
    assert matches, "what_if_recommendation_path DEBUG missing"
    rec = matches[0]
    assert rec.levelno == logging.DEBUG
    assert rec.scenario == "add"
    assert rec.branch in ("no_new_risks", "critical", "monitored")
    assert isinstance(rec.new_risk_count, int)
