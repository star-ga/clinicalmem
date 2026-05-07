"""Pin: engine/bitnet_features_v8.py emits structured logs for OOV +
load events.

Iter-279 T4 round-58 ratchet. The v8 feature encoder shipped at
iter-275 with **zero** logger calls (0/kloc density), making it the
only engine module without observability. Two events are load-bearing
enough to pin:

1. Module-load DEBUG `bitnet_features_v8_loaded` fires once per
   process when the encoder imports. Auditors use this to correlate
   every BitNet decision to the flag-table snapshot.

2. WARNING `bitnet_v8_oov_drug` fires when ``encode_pair_v8`` is
   called with EITHER drug unknown to the flag table. This is the OOV
   signal — the model falls back to hash-only encoding for that drug,
   which is safety-relevant because the cohort-aggregate recall claim
   `43/43` only covers in-distribution drugs.

PHI safety: drug names are hashed (SHA-256 canonical, first 16 hex)
before logging. Raw names never reach the log record.
"""
from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_FEATURES = _REPO_ROOT / "engine" / "bitnet_features_v8.py"


def test_oov_drug_emits_structured_warning(caplog):
    """Both-unknown drug pair → structured WARNING with hash prefixes,
    not raw names."""
    import sys
    sys.path.insert(0, str(_REPO_ROOT))
    from engine.bitnet_features_v8 import encode_pair_v8  # noqa: PLC0415

    with caplog.at_level(logging.WARNING, logger="engine.bitnet_features_v8"):
        encode_pair_v8("totally-unknown-drug-aaa", "totally-unknown-drug-bbb")

    matched = [
        r for r in caplog.records
        if r.name == "engine.bitnet_features_v8"
        and r.message == "bitnet_v8_oov_drug"
    ]
    assert matched, (
        "encode_pair_v8 with both-unknown drugs must emit "
        "'bitnet_v8_oov_drug' WARNING"
    )
    rec = matched[0]
    assert rec.levelno == logging.WARNING
    assert rec.drug_a_known is False
    assert rec.drug_b_known is False
    assert rec.fallback == "hash_only_encoding"
    assert rec.feat_dim == 193
    # 16-char SHA-256 prefix shape
    assert re.fullmatch(r"[0-9a-f]{16}", rec.drug_a_hash_prefix)
    assert re.fullmatch(r"[0-9a-f]{16}", rec.drug_b_hash_prefix)
    # Verify hash is the canonical SHA-256 of the lowercased name
    expected_a_prefix = hashlib.sha256(
        "totally-unknown-drug-aaa".encode("utf-8")
    ).hexdigest()[:16]
    assert rec.drug_a_hash_prefix == expected_a_prefix


def test_oov_warning_does_not_leak_raw_drug_names(caplog):
    """The structured log MUST NOT carry raw drug names — PHI-safe
    discipline (mirror of iter-234/iter-239/iter-240 across the
    engine; v8 features are the new module that joins the pin).
    """
    import sys
    sys.path.insert(0, str(_REPO_ROOT))
    from engine.bitnet_features_v8 import encode_pair_v8  # noqa: PLC0415

    sentinel_a = "secret-patient-drug-name-aaa"
    sentinel_b = "secret-patient-drug-name-bbb"

    with caplog.at_level(logging.WARNING, logger="engine.bitnet_features_v8"):
        encode_pair_v8(sentinel_a, sentinel_b)

    for rec in caplog.records:
        if rec.name != "engine.bitnet_features_v8":
            continue
        # Raw sentinel must not appear in the message OR any extras
        full_text = rec.getMessage() + " " + " ".join(
            f"{k}={v}" for k, v in vars(rec).items()
            if not k.startswith("_") and isinstance(v, (str, int, bool, float))
        )
        assert sentinel_a not in full_text, (
            f"Raw drug name {sentinel_a!r} leaked into "
            f"engine.bitnet_features_v8 log record: {full_text!r}"
        )
        assert sentinel_b not in full_text, (
            f"Raw drug name {sentinel_b!r} leaked into "
            f"engine.bitnet_features_v8 log record: {full_text!r}"
        )


def test_known_pair_emits_no_oov_warning(caplog):
    """In-distribution pair → no OOV warning. Verifies the warning
    fires only when at least one drug is unknown."""
    import sys
    sys.path.insert(0, str(_REPO_ROOT))
    from engine.bitnet_features_v8 import encode_pair_v8  # noqa: PLC0415

    with caplog.at_level(logging.WARNING, logger="engine.bitnet_features_v8"):
        # Both drugs are in the live flag table (sildenafil + nitroglycerin
        # is the canonical pde5_nitrate pair)
        encode_pair_v8("sildenafil", "nitroglycerin")

    oov_records = [
        r for r in caplog.records
        if r.name == "engine.bitnet_features_v8"
        and r.message == "bitnet_v8_oov_drug"
    ]
    assert not oov_records, (
        f"In-distribution pair must NOT emit bitnet_v8_oov_drug, "
        f"got {len(oov_records)} records"
    )


def test_one_unknown_drug_still_emits_warning(caplog):
    """Even ONE unknown drug triggers the OOV warning — both-fields
    discipline (drug_a_known + drug_b_known both surfaced)."""
    import sys
    sys.path.insert(0, str(_REPO_ROOT))
    from engine.bitnet_features_v8 import encode_pair_v8  # noqa: PLC0415

    with caplog.at_level(logging.WARNING, logger="engine.bitnet_features_v8"):
        encode_pair_v8("warfarin", "novel-research-compound-xyz")

    matched = [
        r for r in caplog.records
        if r.name == "engine.bitnet_features_v8"
        and r.message == "bitnet_v8_oov_drug"
    ]
    assert matched, "One-unknown-drug pair must still emit OOV warning"
    rec = matched[0]
    # Lex-sort: 'novel-research-compound-xyz' sorts before 'warfarin'
    # so drug_a is the unknown one, drug_b is warfarin.
    assert rec.drug_a_known is False
    assert rec.drug_b_known is True


def test_module_has_logger_calls():
    """Source-level guard: bitnet_features_v8.py must have at least
    3 logger.* calls (the iter-279 ratchet floor: load + warning +
    error-on-dim-mismatch). Catches the regression class where someone
    deletes the logger.warning + the test still passes by accident.
    """
    src = _FEATURES.read_text()
    matches = re.findall(
        r"logger\.(debug|info|warning|error|critical)\b",
        src,
    )
    assert len(matches) >= 3, (
        f"engine/bitnet_features_v8.py must have ≥ 3 logger.* calls, "
        f"found {len(matches)} (iter-279 floor)"
    )
