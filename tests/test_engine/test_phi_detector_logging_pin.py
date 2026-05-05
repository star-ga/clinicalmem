"""Pin engine/phi_detector.py PHI-redaction logger discipline.

Iter 133 (round 26 T4 evidence-chain ratchet).

`engine/phi_detector.py` is the load-bearing PHI-redaction module.
Pre-iter-133 it had 3 logger calls / 192 LOC (15.6/kloc):
  - `phi_redact_clean` (DEBUG) — clean text
  - `phi_redact_match` (INFO) — PHI categories hit (count + categories,
    never the matched values themselves)
  - `phi_scan_phi_detected` (WARNING) — scan-time PHI alert

Three observability gaps closed:

  1. **Empty-input path was silent.** `detect_phi("")` returned [] with
     no log. Empty/None input is usually an upstream bug (caller
     forgot to pass the text); silent silence means SaMD operators
     can't correlate empty-input events with caller-side fixes.
     Fix: `phi_detect_empty_input` (DEBUG) with text_length.

  2. **Input-size anomaly was silent.** A multi-MB text reaching the
     PHI regex panel is a DoS vector (catastrophic backtracking)
     AND a sign of upstream PHI-bundle accumulation that shouldn't
     have reached the detector. Fix: `phi_input_size_anomaly`
     (WARNING) at >500 KB threshold. PHI-safe: LENGTH only,
     never any slice of content.

  3. **Clean-scan path was silent.** `scan_phi(text)` with no PHI
     matches returned a clean PHIReport with no log. Without this
     signal an operator can't compute the PHI-rate (matches per
     scan) or distinguish "no scans" from "scans with no PHI" in
     the audit log. Fix: `phi_scan_clean` (DEBUG) with text_length.

Net engine/phi_detector.py: 3 -> 6 logger calls (+3 events).

PHI / secret discipline:
  Bundle/pattern category names ARE structural metadata (not PHI).
  Match counts ARE structural metadata. Text LENGTH is structural
  metadata. Anything else (substring of input, matched values,
  categories with content) is PHI and must NEVER be logged.
"""
from __future__ import annotations

import logging
import re
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from engine.phi_detector import detect_phi, redact_phi, scan_phi

_MODULE = _REPO_ROOT / "engine" / "phi_detector.py"


def _module_text() -> str:
    return _MODULE.read_text()


def test_phi_detector_logger_floor():
    """Pin a logger-call floor (>= 6) so silent-removal regressions fail."""
    text = _module_text()
    calls = re.findall(r"\blogger\.(debug|info|warning|error|critical)\(", text)
    assert len(calls) >= 6, (
        f"engine/phi_detector.py logger-call count regressed: "
        f"{len(calls)} < floor 6. A structured event was silently removed."
    )


def test_no_bare_str_e_logger_calls_in_phi_detector():
    """No bare `logger.X(..., e)` patterns — PHI/secret discipline.

    Same regex block as iter-103/108/113/118/123/128 PHI hardening.
    PHI detector is doubly load-bearing: logging exception text on
    a regex-failure path could echo the very PHI the detector was
    asked to redact.
    """
    text = _module_text()
    offenders: list[tuple[int, str]] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        if not re.search(r"\blogger\.(debug|info|warning|error|critical)\(", line):
            continue
        if "%s" in line and re.search(r",\s*(e|exc)\s*[)]", line):
            offenders.append((lineno, line.strip()))
    assert not offenders, (
        "engine/phi_detector.py contains bare `logger.X(\"... %s\", e)` "
        "patterns. Convert to structured form with error_type only.\n"
        "Offenders:\n" + "\n".join(f"  L{n}: {l}" for n, l in offenders)
    )


def test_empty_input_emits_debug_event(caplog):
    """`detect_phi("")` emits `phi_detect_empty_input` DEBUG event.

    Pre-iter-133 the empty-input path was silent. Silent silence
    means an upstream bug (caller forgot to pass text) is invisible
    in the SaMD audit log. iter-133 added structured DEBUG event
    so empty-input traffic shows up in observability.
    """
    with caplog.at_level(logging.DEBUG, logger="engine.phi_detector"):
        result = detect_phi("")
    assert result == []
    matched = [
        r for r in caplog.records
        if r.levelno == logging.DEBUG
        and "phi_detect_empty_input" in r.getMessage()
    ]
    assert matched, (
        f"detect_phi('') emitted no `phi_detect_empty_input` event. "
        f"Records seen: {[(r.levelname, r.getMessage()) for r in caplog.records]}"
    )
    rec = matched[0]
    assert getattr(rec, "text_length", "MISSING") == 0


def test_input_size_anomaly_emits_warning(caplog):
    """Large input (>500 KB) emits `phi_input_size_anomaly` WARNING.

    iter-133 hardening: the PHI regex panel is a DoS surface for
    catastrophic backtracking. A multi-MB input is a sign of
    upstream PHI-bundle accumulation OR a malicious caller. Either
    way operators must see it at default log level.

    PHI-safe: only the LENGTH and the threshold are logged, never
    any slice of the input.
    """
    # 600 KB input — exceeds the 500 KB iter-133 threshold.
    big_text = "a" * 600_000
    with caplog.at_level(logging.WARNING, logger="engine.phi_detector"):
        detect_phi(big_text)
    matched = [
        r for r in caplog.records
        if r.levelno == logging.WARNING
        and "phi_input_size_anomaly" in r.getMessage()
    ]
    assert matched, (
        f"600 KB input to detect_phi() emitted no "
        f"`phi_input_size_anomaly` WARNING. The DoS-class signal "
        f"would be lost in production."
    )
    rec = matched[0]
    assert getattr(rec, "text_length", 0) == 600_000
    assert getattr(rec, "anomaly_threshold_bytes", 0) == 500_000


def test_clean_scan_emits_debug_event(caplog):
    """`scan_phi(clean_text)` emits `phi_scan_clean` DEBUG event.

    Without this signal an operator can't compute the PHI-rate
    (matches per scan) or distinguish "no scans" from "scans with
    no PHI" in the audit log.
    """
    clean_text = "The drug interaction was reviewed by the pharmacist."
    with caplog.at_level(logging.DEBUG, logger="engine.phi_detector"):
        report = scan_phi(clean_text)
    assert report.is_safe is True
    assert report.phi_count == 0
    matched = [
        r for r in caplog.records
        if r.levelno == logging.DEBUG
        and "phi_scan_clean" in r.getMessage()
    ]
    assert matched, (
        f"scan_phi(clean_text) with is_safe=True emitted no "
        f"`phi_scan_clean` DEBUG event. PHI-rate computation in "
        f"the audit log requires this baseline event."
    )
    rec = matched[0]
    assert getattr(rec, "text_length", 0) == len(clean_text)
    assert getattr(rec, "is_safe", None) is True


def test_phi_match_log_contains_no_match_values(caplog):
    """`phi_redact_match` event must NEVER include matched PHI values
    in any field of the log record.

    Same paranoia as iter-103/108/113/118/123/128 sentinel-leak
    scrubs: defensively block any path that would echo the PHI
    the detector just identified into the audit chain.
    """
    sentinel_email = "ZZZ_LEAK_SENTINEL_EMAIL_iter133@example.test"
    sentinel_phone = "555-867-5309"
    text = (
        f"Contact patient at {sentinel_email} or call {sentinel_phone}. "
        f"MRN-99887766."
    )
    with caplog.at_level(logging.DEBUG, logger="engine.phi_detector"):
        redacted, matches = redact_phi(text)
    # PHI was found, so phi_redact_match should fire
    assert matches, "Test setup: redact_phi should have found PHI"
    # No log record may contain the sentinel values anywhere — message,
    # extras, or the formatted output.
    for rec in caplog.records:
        msg = rec.getMessage()
        assert sentinel_email not in msg, (
            f"Email sentinel leaked into log message: {msg!r}"
        )
        assert sentinel_phone not in msg, (
            f"Phone sentinel leaked into log message: {msg!r}"
        )
        # Also scan record __dict__ for sentinel in any string attribute
        for value in vars(rec).values():
            if isinstance(value, str):
                assert sentinel_email not in value, (
                    f"Email sentinel leaked into record attribute"
                )
                assert sentinel_phone not in value, (
                    f"Phone sentinel leaked into record attribute"
                )
            elif isinstance(value, (list, tuple)):
                for item in value:
                    if isinstance(item, str):
                        assert sentinel_email not in item, (
                            f"Email sentinel leaked into record list/tuple"
                        )
                        assert sentinel_phone not in item, (
                            f"Phone sentinel leaked into record list/tuple"
                        )


def test_module_exports_remain_stable():
    """Source-level pin: the 3 public functions phi_detector exports
    must remain stable.

    Downstream callers (engine/consensus_engine.py, engine/clinical_memory.py,
    engine/llm_synthesizer.py) import `redact_phi` / `detect_phi` /
    `scan_phi` by name. A rename / removal would silently break the
    PHI-redaction guard chain.
    """
    text = _module_text()
    for required in ("def detect_phi(", "def redact_phi(", "def scan_phi("):
        assert required in text, (
            f"engine/phi_detector.py removed or renamed `{required}` — "
            f"this is the public API the consensus engine + clinical "
            f"memory + LLM synthesizer all import. Renaming silently "
            f"breaks the PHI-redaction guard chain."
        )
