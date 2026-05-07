"""Pin: BITNET_SAFETY_DOWNGRADE_DISAGREEMENT log record is PHI-safe.

**Iter-296 T1 round-62 — regression test for the iter-291 PHI fix.**

Iter-291 caught a real PHI leak: the BITNET_SAFETY_DOWNGRADE_DISAGREEMENT
WARNING was emitting raw drug names via positional `%s` formatting
across an adjacent-string-literal concatenation that escaped the
iter-240 PHI-discipline regex. Iter-291 fixed both the leak (replaced
with structured `extra={}` form using a SHA-256 pair hash) and the
regex (extended to handle multi-line adjacent-string concat).

This pin is the **runtime regression test** for the iter-291 fix —
the iter-240 pin scans source code for the leaking pattern; this
pin invokes the disagreement code path with a stubbed classifier
and asserts the live log record's actual shape.

Without this pin, a future refactor could:
- Revert the structured-extras form to positional %s (caught by
  iter-240 source scan AND by this runtime test)
- Add a new field that DOES carry raw drug names (caught only by
  this runtime test — source scan is structurally unaware)
- Stop emitting the disagreement event entirely (caught only by
  this runtime test)

Live state: v8 ENGINE achieves 100% contra recall, so the
disagreement path almost never fires in production. This pin uses
a mock classifier to force the path deterministically.
"""
from __future__ import annotations

import logging
import re
import sys
from pathlib import Path
from unittest.mock import patch

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))


class _MockBitNetResult:
    """Minimal fixture mimicking engine.bitnet_classifier.BitNetResult.

    The iter-291 code path reads ``severity_name``, ``feature_hash``,
    ``repro_hash``, and ``weights_id`` from the result.
    """

    def __init__(self):
        self.severity_name = "none"  # forces disagreement vs upstream
        self.feature_hash = "f" * 64
        self.repro_hash = "a" * 64
        self.weights_id = "1f0f88591c05af57c62d844b667639b29c7d1f0eb1b213073d158101611f76e6"


def _make_upstream_dangerous_interaction(drug_a: str, drug_b: str):
    from engine.clinical_scoring import DrugInteraction  # noqa: PLC0415
    return DrugInteraction(
        drug_a=drug_a,
        drug_b=drug_b,
        severity="contraindicated",  # upstream says dangerous
        description="test fixture interaction",
        score=0.99,
    )


def test_disagreement_log_emits_structured_extras_no_raw_names(caplog):
    """When upstream says 'contraindicated' and the BitNet classifier
    returns 'none' (a downgrade disagreement), the WARNING log MUST
    carry a SHA-256 pair_hash_prefix + drug_*_length scalars + audit-
    grade hashes — but NEVER the raw drug names.
    """
    from engine.clinical_scoring import _attach_bitnet_repro_hashes  # noqa: PLC0415

    # Use sentinel drug names so we can scrub for PHI leakage.
    sentinel_a = "secret-patient-drug-aaa-marker"
    sentinel_b = "secret-patient-drug-bbb-marker"
    interaction = _make_upstream_dangerous_interaction(sentinel_a, sentinel_b)

    # Mock the classifier to return a 'none' verdict (forces disagreement).
    mock_result = _MockBitNetResult()
    with patch(
        "engine.bitnet_classifier.classifier_layer",
        return_value=mock_result,
    ):
        with caplog.at_level(logging.WARNING, logger="engine.clinical_scoring"):
            _attach_bitnet_repro_hashes([interaction])

    matched = [
        r for r in caplog.records
        if r.name == "engine.clinical_scoring"
        and r.message == "BITNET_SAFETY_DOWNGRADE_DISAGREEMENT"
    ]
    assert matched, (
        "Disagreement path must emit 'BITNET_SAFETY_DOWNGRADE_DISAGREEMENT' "
        "WARNING when upstream says contraindicated and BitNet says none."
    )
    rec = matched[0]
    # Required PHI-safe fields
    assert rec.levelno == logging.WARNING
    assert hasattr(rec, "pair_hash_prefix"), (
        "iter-291 fix: log MUST carry pair_hash_prefix (16-char SHA-256)"
    )
    assert re.fullmatch(r"[0-9a-f]{16}", rec.pair_hash_prefix), (
        f"pair_hash_prefix must be 16-char lowercase hex; got "
        f"{rec.pair_hash_prefix!r}"
    )
    assert isinstance(rec.drug_a_length, int)
    assert isinstance(rec.drug_b_length, int)
    assert rec.drug_a_length == len(sentinel_a)
    assert rec.drug_b_length == len(sentinel_b)
    assert rec.upstream_severity == "contraindicated"
    assert rec.bitnet_severity == "none"
    assert rec.feature_hash == "f" * 64
    assert rec.repro_hash == "a" * 64
    assert rec.weights_id.startswith("1f0f8859")
    assert isinstance(rec.advisory, str)

    # PHI scrub: raw sentinels must NEVER appear anywhere in the
    # record (message, args, or any extra field).
    full_text = rec.getMessage() + " " + " ".join(
        f"{k}={v}" for k, v in vars(rec).items()
        if not k.startswith("_") and isinstance(v, (str, int, bool, float))
    )
    assert sentinel_a not in full_text, (
        f"PHI LEAK: raw drug name {sentinel_a!r} found in disagreement "
        f"log record. iter-291 fix reverted? Record: {full_text!r}"
    )
    assert sentinel_b not in full_text, (
        f"PHI LEAK: raw drug name {sentinel_b!r} found in disagreement "
        f"log record. iter-291 fix reverted? Record: {full_text!r}"
    )


def test_disagreement_log_pair_hash_is_canonical_lex_sorted():
    """The pair_hash_prefix MUST be computed from the lex-sorted
    canonical pair (so swap(a, b) produces the same hash)."""
    import hashlib  # noqa: PLC0415
    from engine.clinical_scoring import _attach_bitnet_repro_hashes  # noqa: PLC0415

    drug_x = "zzz-drug-late"
    drug_y = "aaa-drug-early"

    expected_canon = f"{min(drug_x, drug_y)}|{max(drug_x, drug_y)}"
    expected_hash = hashlib.sha256(
        expected_canon.encode("utf-8")
    ).hexdigest()[:16]

    mock_result = _MockBitNetResult()
    for a, b in [(drug_x, drug_y), (drug_y, drug_x)]:
        interaction = _make_upstream_dangerous_interaction(a, b)
        with patch(
            "engine.bitnet_classifier.classifier_layer",
            return_value=mock_result,
        ):
            import logging as _logging
            with __import__("contextlib").nullcontext():
                # Use caplog-equivalent via root handler
                handler = _logging.StreamHandler()
                rec_box: list = []

                class _Cap(_logging.Handler):
                    def emit(self, r):
                        rec_box.append(r)
                cap = _Cap()
                cap.setLevel(_logging.WARNING)
                lg = _logging.getLogger("engine.clinical_scoring")
                lg.addHandler(cap)
                try:
                    _attach_bitnet_repro_hashes([interaction])
                finally:
                    lg.removeHandler(cap)

                disagreements = [
                    r for r in rec_box
                    if r.message == "BITNET_SAFETY_DOWNGRADE_DISAGREEMENT"
                ]
                assert disagreements
                assert disagreements[0].pair_hash_prefix == expected_hash, (
                    f"Pair-hash should be lex-canonical: "
                    f"order ({a!r}, {b!r}) → expected {expected_hash!r}, "
                    f"got {disagreements[0].pair_hash_prefix!r}"
                )


def test_no_legacy_positional_format_string_in_disagreement_call():
    """Source-level guard: the disagreement logger.warning call site
    in engine/clinical_scoring.py MUST NOT contain the legacy
    `pair=%s+%s` positional pattern. Catches the regression class
    where someone reverts the iter-291 structured-extras fix to
    the old positional %s form.
    """
    src = (_REPO_ROOT / "engine" / "clinical_scoring.py").read_text()
    # Locate the disagreement WARNING call body
    disagreement_idx = src.find('"BITNET_SAFETY_DOWNGRADE_DISAGREEMENT')
    assert disagreement_idx >= 0, (
        "Could not locate BITNET_SAFETY_DOWNGRADE_DISAGREEMENT call site "
        "in engine/clinical_scoring.py"
    )
    # Slice the surrounding 800 chars (the call + its body)
    window = src[disagreement_idx:disagreement_idx + 800]
    assert "pair=%s+%s" not in window, (
        "Legacy positional pattern 'pair=%s+%s' found at the iter-291 "
        "disagreement log site. The iter-291 fix replaced positional %s "
        "with structured extras={pair_hash_prefix=...}. Reverting to %s "
        "would re-introduce the PHI leak via record.args."
    )
    assert "extra=" in window, (
        "Disagreement log MUST use structured `extra={...}` form, "
        "not positional %s args."
    )
    assert "pair_hash_prefix" in window, (
        "Disagreement log MUST carry pair_hash_prefix per iter-291 "
        "PHI-safe discipline."
    )
