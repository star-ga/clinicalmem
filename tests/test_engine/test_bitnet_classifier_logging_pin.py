"""Pin engine/bitnet_classifier.py logger discipline.

Iteration 123 (round 24 T4 ratchet). The BitNet classifier is the
SHA-256 repro_hash producer that the entire FDA-grade audit-replay
guarantee rides on. Pre-iter-123 it had 4 logger calls / 461 LOC
(8.7/kloc) but ZERO observability on its 6 release-blocking raises.

Six paths hardened:
  1. hidden_w row count != 64
  2. hidden_w col count != 128
  3. hidden_b length != 64
  4. output_w row count != 5
  5. output_w col count != 64
  6. output_b length != 5
  + WeightsTamperError pre-raise (CRITICAL, the most release-blocking
    SaMD integrity event in the system).

Bundle IDs (SHA-256 prefixes of the canonical-JSON weights file) are
safe to log — they ARE the integrity primitive, not secret material.

Net engine/bitnet_classifier.py: 4 → 11 logger calls (+7 events).
"""
from __future__ import annotations

import json
import logging
import re
import tempfile
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_MODULE = _REPO_ROOT / "engine" / "bitnet_classifier.py"


def _module_text() -> str:
    return _MODULE.read_text()


def test_bitnet_classifier_logger_floor():
    """Pin a logger-call floor (>= 11) so silent-removal regressions fail."""
    text = _module_text()
    calls = re.findall(r"\blogger\.(debug|info|warning|error|critical)\(", text)
    assert len(calls) >= 11, (
        f"engine/bitnet_classifier.py logger-call count regressed: "
        f"{len(calls)} < floor 11. A structured event was silently removed."
    )


def test_no_bare_str_e_logger_calls_in_classifier():
    """No bare `logger.X(..., e)` patterns — PHI/secret discipline.

    Bundle IDs are safe to log; raw exception text on the load path
    can include filesystem paths / partial JSON content. Same regex
    block as test_clinical_scoring_logging_pin (iter 113).
    """
    text = _module_text()
    offenders: list[tuple[int, str]] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        if not re.search(r"\blogger\.(debug|info|warning|error|critical)\(", line):
            continue
        if "%s" in line and re.search(r",\s*(e|exc)\s*[)]", line):
            offenders.append((lineno, line.strip()))
    assert not offenders, (
        "engine/bitnet_classifier.py contains bare `logger.X(\"... %s\", e)` "
        "patterns. Convert to structured form with error_type only.\n"
        "Offenders:\n" + "\n".join(f"  L{n}: {l}" for n, l in offenders)
    )


def _make_weights_payload(*, hidden_w_rows=64, hidden_w_cols=128,
                           hidden_b_len=64, output_w_rows=5,
                           output_w_cols=64, output_b_len=5):
    return {
        "hidden_w": [[0] * hidden_w_cols for _ in range(hidden_w_rows)],
        "hidden_b": [0] * hidden_b_len,
        "output_w": [[0] * output_w_cols for _ in range(output_w_rows)],
        "output_b": [0] * output_b_len,
    }


def test_load_weights_shape_mismatch_emits_structured_error(caplog):
    """A malformed weights bundle must surface a structured ERROR
    BEFORE the ValueError propagates. Audit reviewers need to see
    what shape constraint failed without parsing exception messages.
    """
    import sys as _sys
    _sys.path.insert(0, str(_REPO_ROOT))
    from engine.bitnet_classifier import load_weights

    bad_payload = _make_weights_payload(hidden_w_rows=63)  # missing one row

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as f:
        json.dump(bad_payload, f)
        bad_path = f.name

    try:
        with caplog.at_level(logging.ERROR, logger="engine.bitnet_classifier"):
            try:
                load_weights(Path(bad_path))
            except ValueError:
                pass
            else:
                raise AssertionError("expected ValueError on malformed weights")

        matched = [
            r for r in caplog.records
            if r.levelno == logging.ERROR
            and "bitnet_weights_shape_mismatch" in r.getMessage()
        ]
        assert matched, (
            f"load_weights() shape-mismatch path emitted no structured "
            f"ERROR log. Records seen: "
            f"{[(r.levelname, r.getMessage()) for r in caplog.records]}"
        )
        rec = matched[0]
        assert getattr(rec, "field", None) == "hidden_w"
        assert getattr(rec, "expected_rows", None) == 64
        assert getattr(rec, "actual_rows", None) == 63
    finally:
        Path(bad_path).unlink(missing_ok=True)


def test_tamper_detected_event_critical_with_safe_metadata():
    """Source-level pin: the WeightsTamperError pre-raise emits CRITICAL
    with `pinned_bundle_id` + `on_disk_bundle_id` (16-char prefixes
    only — full IDs are integrity primitives, NOT secret).

    A functional tamper test would require swapping the weights file
    mid-process; that's already covered by
    `tests/test_engine/test_bitnet_repro_hash_pin.py`. This pin is a
    source-level guarantee that the structured event remains in place.
    """
    text = _module_text()
    # Locate the WeightsTamperError raise site and verify a logger.critical
    # appears with the expected event name in the same enclosing function.
    assert 'logger.critical(' in text
    assert '"bitnet_weights_tamper_detected"' in text, (
        "WeightsTamperError pre-raise must emit "
        "logger.critical('bitnet_weights_tamper_detected', extra=...)."
    )
    # The raise + log must be co-located so reading sequence is:
    #   if mismatch:
    #       logger.critical(...)
    #       raise WeightsTamperError(...)
    crit_idx = text.find('"bitnet_weights_tamper_detected"')
    raise_idx = text.find("raise WeightsTamperError(", crit_idx)
    assert 0 < raise_idx - crit_idx < 800, (
        f"logger.critical(bitnet_weights_tamper_detected) must immediately "
        f"precede the WeightsTamperError raise (within ~800 chars). "
        f"crit_idx={crit_idx}, raise_idx={raise_idx}"
    )
    # Pinned-bundle-id + on-disk-bundle-id must be in the extra dict.
    extra_window = text[crit_idx : crit_idx + 800]
    assert '"pinned_bundle_id"' in extra_window, (
        "tamper-detected event must carry pinned_bundle_id metadata"
    )
    assert '"on_disk_bundle_id"' in extra_window, (
        "tamper-detected event must carry on_disk_bundle_id metadata"
    )


def test_non_ternary_weight_emits_structured_error_event():
    """Source-level pin: non-ternary weight values trigger a structured
    `bitnet_weights_non_ternary` ERROR before raising.

    This event already existed pre-iter-123; the pin freezes the event
    name + metadata fields so a future refactor can't silently rename
    or drop the metadata an SaMD reviewer relies on.
    """
    text = _module_text()
    assert '"bitnet_weights_non_ternary"' in text, (
        "Non-ternary-weight raise path must emit "
        "logger.error('bitnet_weights_non_ternary', extra={'matrix', 'row', 'col', 'value'})"
    )
    # Verify the metadata fields a reviewer expects
    idx = text.find('"bitnet_weights_non_ternary"')
    window = text[idx : idx + 400]
    for required_field in ('"matrix"', '"row"', '"col"', '"value"'):
        assert required_field in window, (
            f"bitnet_weights_non_ternary event missing required field "
            f"{required_field} in extra dict — SaMD reviewer needs all 4 "
            f"to localize the offending weight."
        )
