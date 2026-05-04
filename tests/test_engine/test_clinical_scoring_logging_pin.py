"""Pin clinical_scoring.py logger discipline.

iter 112 audit ranked clinical_scoring.py at 14.5/kloc logger density
(1106 LOC, 16 logger calls). It is the safety-decision module —
iter 110 cited line 329-339 as the BITNET_SAFETY_DOWNGRADE_DISAGREEMENT
site. Two classes of gaps were closed in iter 113:

1. Two SILENT paths (no logger before return / continue):
   - `_rxnorm_check_interactions()` ImportError on `engine.rxnorm_client` —
     the entire Layer 3 silently dropped with zero observability.
   - `_rxnorm_check_interactions()` `len(rxcuis) < 2` pre-flight check —
     zero log when RxNorm fails to normalize ≥ 2 medications to
     RxCUIs (could mean uncovered drug name OR API throttled).

2. Three PHI-leak-risk paths (`%s` of `e` formatting):
   - `_check_openevidence_layer2()` API call failure
   - `_call_openai_json()` API call failure
   - `_call_google_json()` API call failure
   All converted to structured form with `error_type` only — never
   `str(e)`. httpx exceptions can include URL fragments / request
   bodies in their string form; safer to log type only.

Net engine/clinical_scoring.py: 16 → 18 logger calls.

This pin enforces the convention so future regressions can't sneak
back in.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from unittest.mock import patch

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_MODULE = _REPO_ROOT / "engine" / "clinical_scoring.py"


def _module_text() -> str:
    return _MODULE.read_text()


def test_no_bare_str_e_logger_calls():
    """Block `logger.X(..., e)` and `%s ... str(e)` patterns.

    iter 113 PHI-leak hardening: three call sites used `logger.X("...
    failed: %s", e)` which can leak URL fragments + request bodies on
    httpx errors. Convert all to structured `error_type` only form.
    """
    text = _module_text()

    # Line-by-line scan so the assertion message points at offenders.
    offenders: list[tuple[int, str]] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        # Skip the bitnet warning at L313 which uses `%s` for the
        # WeightsTamperError exception object — it's an explicit
        # exception type whose str representation is the safety
        # invariant message itself, not a leaky payload.
        if "BITNET_WEIGHTS_TAMPER" in line:
            continue
        # Match `logger.<level>(...)` containing `%s` followed
        # somewhere later by a bare `, e)` or `, exc)` argument.
        if not re.search(r"\blogger\.(debug|info|warning|error|critical)\(", line):
            continue
        if "%s" in line and re.search(r",\s*(e|exc)\s*[)]", line):
            offenders.append((lineno, line.strip()))

    assert not offenders, (
        "engine/clinical_scoring.py contains bare `logger.X(\"... %s\", e)` "
        "patterns that can leak httpx URL/body fragments. Convert to:\n"
        "  logger.X(\"event_name\", extra={\"error_type\": type(e).__name__})\n"
        "Offenders:\n" + "\n".join(f"  L{n}: {l}" for n, l in offenders)
    )


def test_clinical_scoring_logger_floor():
    """Pin a logger-call floor so regressions can't drop visibility.

    iter 113 ratchet brought the count to 18. Future audits may add
    more (would raise this floor). A regression that drops below 18
    means an event got silently removed.
    """
    text = _module_text()
    # Count logger.<level>( occurrences (not just `logger.`).
    calls = re.findall(r"\blogger\.(debug|info|warning|error|critical)\(", text)
    assert len(calls) >= 18, (
        f"engine/clinical_scoring.py logger-call count regressed: "
        f"{len(calls)} < floor 18. Some structured event was silently "
        f"removed."
    )


def test_rxnorm_import_failure_logged(caplog):
    """Functional test: when `engine.rxnorm_client` import fails,
    a structured ERROR log MUST be emitted before returning [].

    Same drift class as iter-103 clinical_memory pre-raise events —
    silent drops of an entire safety layer are the worst kind of
    observability bug for an FDA SaMD audit.

    We avoid `importlib.reload` here — reloading clinical_scoring
    breaks the `DrugInteraction` class identity that other tests
    import (they isinstance-check against the old class object).
    Instead we monkey-patch sys.modules so the function-local import
    inside `_rxnorm_check_interactions` raises, then restore.
    """
    import sys

    sys.path.insert(0, str(_REPO_ROOT))
    from engine.clinical_scoring import _rxnorm_check_interactions

    original = sys.modules.pop("engine.rxnorm_client", None)
    try:
        # Prevent re-import: set the module to None so the import
        # raises ImportError/ModuleNotFoundError in the try/except.
        sys.modules["engine.rxnorm_client"] = None  # type: ignore[assignment]
        with caplog.at_level(logging.ERROR, logger="engine.clinical_scoring"):
            result = _rxnorm_check_interactions(["warfarin", "ibuprofen"], set())
        assert result == []
        # Assert: at least one ERROR-level event with "rxnorm_client_import_failed"
        matched = [
            r for r in caplog.records
            if r.levelno == logging.ERROR
            and "rxnorm_client_import_failed" in r.getMessage()
        ]
        assert matched, (
            f"_rxnorm_check_interactions() ImportError path emitted no structured "
            f"ERROR log. Records seen: "
            f"{[(r.levelname, r.getMessage()) for r in caplog.records]}"
        )
        # And: error_type metadata present, no str(e) leak.
        rec = matched[0]
        # ModuleNotFoundError is a subclass of ImportError; either is fine.
        assert getattr(rec, "error_type", None) in (
            "ImportError",
            "ModuleNotFoundError",
        ), (
            f"Structured event must carry error_type='ImportError' or "
            f"'ModuleNotFoundError', got "
            f"{getattr(rec, 'error_type', 'MISSING')!r}"
        )
    finally:
        if original is not None:
            sys.modules["engine.rxnorm_client"] = original
        else:
            sys.modules.pop("engine.rxnorm_client", None)


def test_no_phi_sentinel_leak_in_clinical_scoring_logs(caplog):
    """Inject a sentinel-bearing payload through the OpenAI fail path
    and assert the sentinel never appears in the log records.

    Same pattern as iter-103 clinical_memory + iter-108 fhir_adapter
    sentinel-leak tests.

    No reload here — would break other tests' `DrugInteraction` class
    identity.
    """
    import sys

    sys.path.insert(0, str(_REPO_ROOT))
    from engine.clinical_scoring import _call_openai_json

    sentinel = "ZZZ_OPENAI_RAW_PAYLOAD_LEAK_TOKEN_iter113"

    class _Boom(RuntimeError):
        def __str__(self) -> str:
            return f"Status 500 — payload echo: {sentinel}"

    with patch("httpx.post", side_effect=_Boom("boom")):
        with caplog.at_level(logging.INFO, logger="engine.clinical_scoring"):
            result = _call_openai_json("dummy prompt with " + sentinel, "fake_key")

    assert result is None
    for rec in caplog.records:
        msg = rec.getMessage()
        assert sentinel not in msg, (
            f"PHI-leak sentinel {sentinel!r} appeared in caplog message: "
            f"{msg!r}"
        )
        # Also check structured `extra` payload — record __dict__ would
        # carry it through if `extra={'error_type': str(e)}` regression.
        for value in vars(rec).values():
            if isinstance(value, str):
                assert sentinel not in value, (
                    f"PHI-leak sentinel {sentinel!r} appeared in caplog "
                    f"record attribute (record.__dict__ scan)"
                )
