"""Pin: `engine.clinical_scoring.confidence_gate` emits structured logs.

Iter-289 T4 round-60. The confidence gate is the abstention gate that
decides whether to ANSWER a clinical query or REFUSE for safety.
Pre-iter-289 it emitted ZERO log events — an auditor reading the
clinical_memory chain couldn't tell:

  - WHY the system abstained (no records vs low confidence)
  - WHAT the confidence value was when it abstained
  - HOW many records contributed to the score

Post-iter-289 the gate emits one structured event per decision branch:

  - INFO `confidence_gate_no_records` — empty bm25_scores → forced abstain.
  - WARNING `confidence_gate_abstained` — confidence < threshold → abstain.
  - DEBUG `confidence_gate_pass` — confidence ≥ threshold → answer.

PHI safety: only scalar floats + integers reach the log record (BM25
scores are pre-aggregated averages, no record content). No drug names,
no patient identifiers, no clinical narrative text.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_MODULE_PATH = _REPO_ROOT / "engine" / "clinical_scoring.py"


def test_no_records_emits_structured_info(caplog):
    """Empty bm25_scores → INFO event with should_abstain=True +
    threshold context."""
    import sys
    sys.path.insert(0, str(_REPO_ROOT))
    from engine.clinical_scoring import confidence_gate  # noqa: PLC0415

    with caplog.at_level(logging.INFO, logger="engine.clinical_scoring"):
        result = confidence_gate(bm25_scores=[], entity_overlaps=[])

    assert result.should_abstain is True
    matched = [
        r for r in caplog.records
        if r.name == "engine.clinical_scoring"
        and r.message == "confidence_gate_no_records"
    ]
    assert matched, "Empty bm25_scores must emit 'confidence_gate_no_records' INFO"
    rec = matched[0]
    assert rec.levelno == logging.INFO
    assert rec.should_abstain is True
    assert rec.abstention_threshold == 0.3
    assert rec.score_weight == 0.6
    assert rec.overlap_weight == 0.4


def test_below_threshold_emits_structured_warning(caplog):
    """confidence < threshold → WARNING with all scalar metrics for
    audit correlation."""
    import sys
    sys.path.insert(0, str(_REPO_ROOT))
    from engine.clinical_scoring import confidence_gate  # noqa: PLC0415

    with caplog.at_level(logging.WARNING, logger="engine.clinical_scoring"):
        result = confidence_gate(
            bm25_scores=[0.1, 0.2],
            entity_overlaps=[0.0, 0.1],
            abstention_threshold=0.5,
        )

    assert result.should_abstain is True
    matched = [
        r for r in caplog.records
        if r.name == "engine.clinical_scoring"
        and r.message == "confidence_gate_abstained"
    ]
    assert matched, "Below-threshold confidence must emit 'confidence_gate_abstained' WARNING"
    rec = matched[0]
    assert rec.levelno == logging.WARNING
    assert rec.abstention_threshold == 0.5
    assert rec.n_records == 2
    assert isinstance(rec.confidence, float)
    assert isinstance(rec.avg_bm25, float)
    assert isinstance(rec.avg_entity_overlap, float)


def test_above_threshold_emits_structured_debug(caplog):
    """confidence >= threshold → DEBUG pass event with same metric
    shape as the WARNING path."""
    import sys
    sys.path.insert(0, str(_REPO_ROOT))
    from engine.clinical_scoring import confidence_gate  # noqa: PLC0415

    with caplog.at_level(logging.DEBUG, logger="engine.clinical_scoring"):
        result = confidence_gate(
            bm25_scores=[0.9, 0.8, 0.85],
            entity_overlaps=[0.7, 0.8, 0.75],
            abstention_threshold=0.3,
        )

    assert result.should_abstain is False
    matched = [
        r for r in caplog.records
        if r.name == "engine.clinical_scoring"
        and r.message == "confidence_gate_pass"
    ]
    assert matched, "Above-threshold confidence must emit 'confidence_gate_pass' DEBUG"
    rec = matched[0]
    assert rec.levelno == logging.DEBUG
    assert rec.confidence >= 0.3
    assert rec.n_records == 3


def test_no_record_content_leaks_into_log(caplog):
    """Confidence-gate logs must NEVER carry record content, drug
    names, or patient identifiers — only scalar floats + ints. Same
    iter-234/iter-239/iter-240/iter-279/iter-284 PHI discipline."""
    import sys
    sys.path.insert(0, str(_REPO_ROOT))
    from engine.clinical_scoring import confidence_gate  # noqa: PLC0415

    with caplog.at_level(logging.DEBUG, logger="engine.clinical_scoring"):
        confidence_gate(
            bm25_scores=[0.5, 0.6],
            entity_overlaps=[0.5, 0.5],
        )

    for rec in caplog.records:
        if rec.name != "engine.clinical_scoring":
            continue
        for k, v in vars(rec).items():
            if k.startswith("_"):
                continue
            # Allow strings only for the message + standard logging
            # fields; reject any string-valued extras (would suggest
            # record content leaking through).
            if k in {"message", "msg", "levelname", "name", "pathname",
                     "filename", "module", "funcName", "exc_text",
                     "stack_info", "should_abstain"}:
                continue
            if k in {"confidence", "abstention_threshold", "n_records",
                     "avg_bm25", "avg_entity_overlap", "score_weight",
                     "overlap_weight"}:
                # Allowed scalar extras
                assert isinstance(v, (int, float, bool)), (
                    f"Confidence-gate log extra {k!r} = {v!r} must be "
                    f"scalar; got {type(v).__name__}"
                )


def test_module_has_at_least_three_confidence_gate_logs():
    """Source-level guard: confidence_gate function must contain
    ≥ 3 logger.* calls (one per decision branch). Catches regression
    where someone deletes the WARNING/DEBUG events."""
    src = _MODULE_PATH.read_text()
    # Slice from `def confidence_gate(` to the next `def `
    m = re.search(r"def confidence_gate\(", src)
    assert m is not None
    after = src[m.start():]
    next_def = re.search(r"\ndef \w+\(", after)
    body = after[:next_def.start()] if next_def else after
    matches = re.findall(r"logger\.(debug|info|warning|error|critical)\b", body)
    assert len(matches) >= 3, (
        f"confidence_gate body must have ≥ 3 logger.* calls "
        f"(no_records / abstained / pass branches), found {len(matches)}"
    )
