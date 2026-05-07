"""Pin: `engine.clinical_memory.ClinicalMemory.explain_clinical_conflict`
emits a structured log event on BOTH the abstention and success paths.

Iter-304 T4 round-64 (26th cross-pin family). Pre-iter-304 the function
fired the abstention event (`clinical_memory_explain_conflict_abstained`,
INFO) but the success path (when a narrative IS generated) was silent —
operators couldn't compute the explanation rate
(success / (success + abstain)) or distinguish "no invocations" from
"all abstained". This was the same iter-289 observability-gap class as
the pre-iter-289 confidence_gate (which had ZERO events at all).

Surfaces locked
===============
1. SUCCESS path emits `clinical_memory_explain_conflict_generated` (INFO)
   with structured extras (conflict_type token, evidence_block_count,
   citation_count, abstained, model_used, conflict_index, patient_id).
2. ABSTENTION path emits `clinical_memory_explain_conflict_abstained`
   (INFO) — pre-existing iter-289-era event preserved.
3. Source-level guard: both event_name strings appear in
   `engine/clinical_memory.py`, plus a logger.info call inside the
   `explain_clinical_conflict` method body.

PHI safety: extras carry only structural identifiers (synthetic
patient_id, conflict_type token like "provider_attribution_disagreement",
conflict_index, evidence/citation counts, abstention flag, model_used
identifier). No drug names, allergens, free-text clinical narrative,
or other patient detail reaches the log record.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_MODULE_PATH = _REPO_ROOT / "engine" / "clinical_memory.py"


def test_success_event_name_in_source():
    """The success-path event_name string MUST appear in the source.
    iter-304 added it; future maintainers cannot silently remove it
    without re-introducing the iter-289 observability-gap class."""
    src = _MODULE_PATH.read_text()
    assert '"clinical_memory_explain_conflict_generated"' in src, (
        "engine/clinical_memory.py must emit "
        "'clinical_memory_explain_conflict_generated' on the "
        "explain_clinical_conflict success path. iter-304 added this "
        "to close the silent-success-path gap (the function previously "
        "only logged abstention via 'clinical_memory_explain_conflict_"
        "abstained'). Restore the logger.info call if it was removed."
    )


def test_abstention_event_name_in_source():
    """The pre-existing abstention event must remain. Iter-289 era."""
    src = _MODULE_PATH.read_text()
    assert '"clinical_memory_explain_conflict_abstained"' in src, (
        "engine/clinical_memory.py must continue to emit "
        "'clinical_memory_explain_conflict_abstained' on the abstain "
        "path. This event is iter-289 era (the confidence-gate "
        "observability discipline applied to the explain-conflict "
        "abstention gate)."
    )


def test_explain_method_has_at_least_two_logger_calls():
    """The `explain_clinical_conflict` method body must contain at
    least 2 logger.X calls (one abstention + one success). Source-
    level lower-bound guard against accidental deletion."""
    src = _MODULE_PATH.read_text()
    # Slice from "def explain_clinical_conflict" to the next top-level "def "
    m = re.search(
        r"(    def explain_clinical_conflict\(.*?)(?=\n    def |\Z)",
        src,
        re.DOTALL,
    )
    assert m is not None, (
        "Could not find `explain_clinical_conflict` method body in "
        "engine/clinical_memory.py. If renamed, update this pin."
    )
    body = m.group(1)
    n = len(re.findall(r"\blogger\.(info|warning|error|debug)\b", body))
    assert n >= 2, (
        f"explain_clinical_conflict body has only {n} logger.X call(s); "
        f"expected ≥ 2 (abstention + success). Adding observability "
        f"events is one-way: removing them re-introduces the iter-289 "
        f"silent-path class."
    )


def test_success_event_carries_no_phi_extras():
    """The success event_name and the extras keys it uses must be the
    PHI-safe structural set (no drug_a/drug_b/medication/allergen/
    raw narrative content). Source-level scan."""
    src = _MODULE_PATH.read_text()
    # Find the logger.info("clinical_memory_explain_conflict_generated", extra={...}) call.
    m = re.search(
        r'logger\.info\(\s*\n\s*"clinical_memory_explain_conflict_generated"'
        r',\s*\n\s*extra=\{(.*?)\}',
        src,
        re.DOTALL,
    )
    assert m is not None, (
        "Could not find logger.info('clinical_memory_explain_conflict_"
        "generated', extra={...}) call in engine/clinical_memory.py. "
        "If reformatted, update this pin to track the new shape."
    )
    extras_block = m.group(1)
    # Forbidden field-name substrings that would leak PHI through `extra=`.
    forbidden = (
        '"drug_a"', '"drug_b"', '"medication"', '"allergen"',
        '"narrative"', '"narrative_text"', '"clinical_summary"',
        '"description"', '"medications"', '"allergies"',
        '"content"', '"raw_text"',
    )
    leaked = [k for k in forbidden if k in extras_block]
    assert not leaked, (
        f"clinical_memory_explain_conflict_generated extras leak PHI-"
        f"risky keys: {leaked!r}. Use only structural counts + "
        f"identifiers (patient_id, conflict_type, conflict_index, "
        f"counts, abstained, model_used)."
    )
