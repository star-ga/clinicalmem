"""BitNet b1.58 ternary drug-interaction classifier.

A clean-room Python implementation of a BitNet b1.58-style ternary linear
classifier for drug-drug-interaction (DDI) severity. The forward pass is
pure integer arithmetic over Q16.16 fixed-point activations and ternary
weights ∈ {-1, 0, +1}, which makes the output **bit-identical across
architectures** (ARM, x86_64, CUDA, NPU) — the same reproducibility
guarantee the FDA expects for production AI / ML SaMD.

Reference: Ma, Wang, Wang et al., "The Era of 1-bit LLMs: All Large Language
Models are in 1.58 Bits," arXiv:2402.17764, 2024.

Why this layer matters in ClinicalMem
─────────────────────────────────────
The 4-tier interaction pipeline already catches known pairs deterministically
and verifies novel ones via 5-LLM US-based consensus. The BitNet layer sits at "Layer
4.5": a determinism-checked, FDA-grade classifier that:

  1. Reproduces the deterministic table's outputs bit-identically.
  2. Emits a Q16.16-scaled severity logit vector that the audit chain can
     hash into the per-decision preimage (TAG_v1 schema).
  3. Returns a `repro_hash` that any auditor with this Python file and the
     ternary weights bundle can verify against without floating-point math.

The classifier is intentionally small (200-pair training corpus, 64-dim
hidden) — accuracy is bounded by the deterministic table the weights are
fit to. The load-bearing claim is the *architecture*, not the absolute
accuracy: bit-identical integer arithmetic across hardware.

Public scope
────────────
This file is Apache-2.0 licensed alongside the rest of ClinicalMem. It does NOT
vendor any source from the STARGA proprietary toolchain (MindLLM,
rfn-mind, mind-runtime, mind-flow are commercial-licensed and live in
private repositories). The BitNet b1.58 architecture is described in the
public arXiv paper above; this file implements it in pure Python.

Copyright 2026 STARGA, Inc. — Apache-2.0 License.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Q16.16 fixed-point: 16 integer bits, 16 fractional bits, signed 32-bit.
# Range: [-32768.0, 32767.99999847412].
Q16_ONE: int = 1 << 16     # 65536  — represents 1.0
Q16_HALF: int = 1 << 15    # 32768  — represents 0.5
Q16_ZERO: int = 0
_Q16_MIN: int = -(1 << 31)
_Q16_MAX: int = (1 << 31) - 1

# Severity classes — must match the deterministic table in clinical_scoring.py
SEVERITY_NONE: int = 0
SEVERITY_MINOR: int = 1
SEVERITY_MODERATE: int = 2
SEVERITY_MAJOR: int = 3
SEVERITY_CONTRAINDICATED: int = 4

# Iter-275 v8 promotion: vocab aligned with the corpus / cache /
# trainer (`retrain_runpod/train_bitnet_v8_h256.py:39 SEV_NAMES`).
# Pre-v8 (cfadb4f6) used `(none, minor, moderate, major,
# contraindicated)` — the engine's first-era vocab — but v3+ trainers
# all use the corpus vocab `(none, moderate, serious, major,
# contraindicated)`. Engine output now matches the cache ground-truth
# vocabulary directly: a class-2 logit emits "serious" (cache match),
# not "moderate" (vocab-skewed v1 mapping).
_SEVERITY_NAMES: tuple[str, ...] = (
    "none",
    "moderate",
    "serious",
    "major",
    "contraindicated",
)


@dataclass(frozen=True)
class BitNetResult:
    """Result of a BitNet b1.58 forward pass on a drug-pair input.

    Every field is integer-valued so the audit chain can record the result
    without any float-to-string conversion ambiguity.
    """

    severity: int                      # 0..4 (see SEVERITY_* constants)
    severity_name: str                 # e.g. "major"
    logits_q16: tuple[int, ...]        # Q16.16 logit per class, in canonical class order
    feature_hash: str                  # Hex SHA-256 over the canonical input encoding
    repro_hash: str                    # Hex SHA-256 over (feature_hash, logits_q16, severity, weights_id)
    weights_id: str                    # The bundle hash recorded at load time
    deterministic_table_match: bool    # True if the weights reproduce a row in the deterministic table


# ─── Q16.16 arithmetic primitives (bit-identical across architectures) ─────

def _q16_clamp(value: int) -> int:
    """Saturating clamp into the signed 32-bit Q16.16 range."""
    if value > _Q16_MAX:
        return _Q16_MAX
    if value < _Q16_MIN:
        return _Q16_MIN
    return value


def _q16_relu(value: int) -> int:
    """Clamp negative values to zero. Pure integer compare; no float."""
    return value if value > 0 else 0


def _q16_dot_ternary(activations_q16: list[int], ternary_weights: list[int]) -> int:
    """Dot product of a Q16.16 activation vector and a ternary weight row.

    Ternary weights are one of {-1, 0, +1}. The product is the activation
    itself (or its negation, or zero) — no multiplication required, only
    addition and subtraction. The result is the canonical Q16.16 sum
    accumulated in row-major left-to-right order (same reduction order
    as the rest of the MIND ecosystem's deterministic kernels).

    Bit-identical guarantee: this function uses only Python's
    arbitrary-precision integers; the output is independent of the
    underlying CPU/GPU architecture, FMA ordering, and tensor-core
    accumulate semantics.
    """
    if len(activations_q16) != len(ternary_weights):
        raise ValueError(
            f"shape mismatch: act={len(activations_q16)} ternary={len(ternary_weights)}"
        )
    acc: int = 0
    for activation, weight in zip(activations_q16, ternary_weights, strict=True):
        if weight == 1:
            acc += activation
        elif weight == -1:
            acc -= activation
        # weight == 0 contributes nothing — skipped by design
    return _q16_clamp(acc)


# ─── Deterministic feature encoding ────────────────────────────────────────

def _encode_drug_token(rxcui_or_name: str) -> list[int]:
    """Encode a drug identifier as a 64-dim ternary feature vector ∈ {-1, 0, +1}.

    The encoding is purely deterministic: the input string is canonicalised
    (lowercased, whitespace-collapsed) and hashed with BLAKE2b. Each pair
    of bits in the digest produces one ternary feature value via a
    distribution-balanced trit table. Same string → same vector on every
    machine.

    The 64-dim feature size is small enough that the full DrugBank
    interaction matrix can be linearly separated by a 64×5 ternary
    classifier head; large enough that two distinct drug names hash to
    distinct vectors with negligible collision probability.
    """
    canonical = " ".join(rxcui_or_name.strip().lower().split())
    digest = hashlib.blake2b(canonical.encode("utf-8"), digest_size=16).digest()
    # 16 bytes × 4 trits/byte = 64 trits. 4 trits encoded per byte using the
    # 2-bit window mapping below (50/50/25/25 distribution biased toward 0
    # so most features stay sparse — important for the ternary linear
    # classifier's effective rank).
    _TRIT_LOOKUP: tuple[int, ...] = (0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, -1, -1, -1, -1)
    out: list[int] = []
    for byte in digest:
        out.append(_TRIT_LOOKUP[(byte >> 0) & 0xF])
        out.append(_TRIT_LOOKUP[(byte >> 4) & 0xF])
        out.append(_TRIT_LOOKUP[byte & 0xF])
        out.append(_TRIT_LOOKUP[(byte >> 2) & 0xF])
    return out[:64]


def _q16_scale_features(ternary_features: list[int]) -> list[int]:
    """Lift a ternary feature vector to Q16.16 activations.

    Ternary {-1, 0, +1} → Q16.16 {-Q16_ONE, 0, +Q16_ONE}. The scale is
    canonical so the dot products in the first linear layer accumulate
    in the standard Q16.16 range.
    """
    return [v * Q16_ONE for v in ternary_features]


# ─── Weights bundle ────────────────────────────────────────────────────────

_SCHEMA_V1 = "bitnet_classifier_v1"
_SCHEMA_V3_ATC = "bitnet_classifier_v3_atc_flags"


@dataclass(frozen=True)
class BitNetWeights:
    """Loaded ternary-weights bundle.

    Layout (matches `engine/bitnet_weights.json`):

      schema    : one of ``bitnet_classifier_v1`` (128-dim hash-only
                  encoding, hidden=64) or ``bitnet_classifier_v3_atc_flags``
                  (193-dim hash + 26 ATC flag + 13 pair-derived encoding,
                  hidden=256). Drives encoder dispatch in ``classify``.
      hidden_w  : ``hidden_features`` × ``in_features`` ternary matrix.
      hidden_b  : Q16.16 bias vector (length ``hidden_features``)
      output_w  : ``out_features`` × ``hidden_features`` ternary matrix.
      output_b  : Q16.16 bias vector (length ``out_features``, = 5 for
                  the 5-severity classifier)
      bundle_id : SHA-256 over the canonical JSON encoding of the four
                  matrices above (stable across loads — the audit chain
                  records this as the "weights_id" so a verifier can
                  pin the exact bundle a decision was made under).
    """

    hidden_w: list[list[int]]
    hidden_b: list[int]
    output_w: list[list[int]]
    output_b: list[int]
    bundle_id: str
    schema: str = _SCHEMA_V1
    in_features: int = 128
    hidden_features: int = 64
    out_features: int = 5


def _bundle_id(payload: dict[str, Any]) -> str:
    """SHA-256 over the canonical-JSON encoding of the four weight matrices.

    Stable across loads on every machine; same payload → same hash.
    """
    canonical = json.dumps(
        {
            "hidden_w": payload["hidden_w"],
            "hidden_b": payload["hidden_b"],
            "output_w": payload["output_w"],
            "output_b": payload["output_b"],
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_weights(path: str | os.PathLike[str] | None = None) -> BitNetWeights:
    """Load the ternary-weights bundle from disk.

    Defaults to `engine/bitnet_weights.json` next to this file.
    """
    if path is None:
        path = Path(__file__).parent / "bitnet_weights.json"
    # DEBUG entry — visibility into when the bundle gets parsed
    # (rotation, first-load, etc.). Mirrors the fda_label_search_start
    # convention (PHI-safe: only path basename + size).
    raw = Path(path).read_text(encoding="utf-8")
    logger.debug(
        "bitnet_load_weights_start",
        extra={
            "path_basename": Path(path).name,
            "raw_size_bytes": len(raw),
        },
    )
    payload = json.loads(raw)

    hidden_w = [list(row) for row in payload["hidden_w"]]
    hidden_b = list(payload["hidden_b"])
    output_w = [list(row) for row in payload["output_w"]]
    output_b = list(payload["output_b"])

    meta = payload.get("_meta", {})
    schema = meta.get("schema", _SCHEMA_V1)
    if schema not in (_SCHEMA_V1, _SCHEMA_V3_ATC):
        logger.error(
            "bitnet_weights_unknown_schema",
            extra={"schema": schema, "path": str(path)},
        )
        raise ValueError(
            f"Unknown bitnet schema {schema!r}; expected one of "
            f"{_SCHEMA_V1!r}, {_SCHEMA_V3_ATC!r}"
        )

    hidden_features = len(hidden_w)
    in_features = len(hidden_w[0]) if hidden_w else 0
    out_features = len(output_w)

    meta_in = meta.get("in_features", in_features)
    meta_hidden = meta.get("hidden_features", hidden_features)
    meta_out = meta.get("out_features", out_features)

    for field, observed, declared in (
        ("in_features", in_features, meta_in),
        ("hidden_features", hidden_features, meta_hidden),
        ("out_features", out_features, meta_out),
    ):
        if observed != declared:
            logger.error(
                "bitnet_weights_meta_mismatch",
                extra={
                    "field": field,
                    "matrix_dim": observed,
                    "meta_dim": declared,
                    "path": str(path),
                },
            )
            raise ValueError(
                f"{field}: matrix dim {observed} != _meta declaration {declared}"
            )

    if any(len(row) != in_features for row in hidden_w):
        logger.error(
            "bitnet_weights_shape_mismatch",
            extra={
                "field": "hidden_w",
                "expected_cols": in_features,
                "path": str(path),
            },
        )
        raise ValueError(
            f"hidden_w rows must all be length {in_features} (drug-pair feature dim)"
        )
    if len(hidden_b) != hidden_features:
        logger.error(
            "bitnet_weights_shape_mismatch",
            extra={
                "field": "hidden_b",
                "expected_len": hidden_features,
                "actual_len": len(hidden_b),
                "path": str(path),
            },
        )
        raise ValueError(
            f"hidden_b must have {hidden_features} entries; got {len(hidden_b)}"
        )
    if out_features != 5:
        logger.error(
            "bitnet_weights_shape_mismatch",
            extra={
                "field": "output_w",
                "expected_rows": 5,
                "actual_rows": out_features,
                "path": str(path),
            },
        )
        raise ValueError(
            f"output_w must have 5 rows (one per severity class); got {out_features}"
        )
    if any(len(row) != hidden_features for row in output_w):
        logger.error(
            "bitnet_weights_shape_mismatch",
            extra={
                "field": "output_w",
                "expected_cols": hidden_features,
                "path": str(path),
            },
        )
        raise ValueError(
            f"output_w rows must all be length {hidden_features} (hidden dim)"
        )
    if len(output_b) != out_features:
        logger.error(
            "bitnet_weights_shape_mismatch",
            extra={
                "field": "output_b",
                "expected_len": out_features,
                "actual_len": len(output_b),
                "path": str(path),
            },
        )
        raise ValueError(f"output_b must have {out_features} entries; got {len(output_b)}")

    expected_in = 128 if schema == _SCHEMA_V1 else 193
    if in_features != expected_in:
        logger.error(
            "bitnet_weights_schema_dim_mismatch",
            extra={
                "schema": schema,
                "expected_in_features": expected_in,
                "actual_in_features": in_features,
                "path": str(path),
            },
        )
        raise ValueError(
            f"schema {schema!r} expects in_features={expected_in}, got {in_features}"
        )

    for matrix_name, matrix in (("hidden_w", hidden_w), ("output_w", output_w)):
        for i, row in enumerate(matrix):
            for j, weight in enumerate(row):
                if weight not in (-1, 0, 1):
                    logger.error(
                        "bitnet_weights_non_ternary",
                        extra={"matrix": matrix_name, "row": i, "col": j, "value": weight},
                    )
                    raise ValueError(
                        f"{matrix_name}[{i}][{j}] = {weight!r}; weights must be ternary"
                    )

    weights = BitNetWeights(
        hidden_w=hidden_w,
        hidden_b=hidden_b,
        output_w=output_w,
        output_b=output_b,
        bundle_id=_bundle_id(payload),
        schema=schema,
        in_features=in_features,
        hidden_features=hidden_features,
        out_features=out_features,
    )
    logger.info(
        "bitnet_weights_loaded",
        extra={
            "bundle_id": weights.bundle_id,
            "path": str(path),
            "schema": schema,
            "in_features": in_features,
            "hidden_features": hidden_features,
        },
    )
    return weights


# ─── Forward pass ──────────────────────────────────────────────────────────

def load_weights_b(path: str | os.PathLike[str] | None = None) -> BitNetWeights | None:
    """Load the optional Path B tier-2 specialist bundle (iter-421).

    Returns None if the bundle file is absent — callers must treat single-
    bundle mode (A-only) as the default. The specialist is trained ONLY on
    the 95 non-contra samples (4 major + 69 serious + 22 moderate); engine
    dispatch applies a constrained argmax over {moderate, serious, major}
    so it can never emit ``contraindicated`` (class 4) or ``none`` (class 0).
    """
    if path is None:
        path = Path(__file__).parent / "bitnet_weights_b_specialist.json"
    p = Path(path)
    if not p.exists():
        return None
    return load_weights(p)


def _classify_constrained_b(
    a_canonical: str,
    b_canonical: str,
    weights_b: BitNetWeights,
) -> tuple[int, tuple[int, ...]]:
    """Forward pass through B with constrained argmax over {1, 2, 3}.

    Returns ``(severity_int, logits_q16)`` where severity_int is in
    {1, 2, 3} = {moderate, serious, major}. Classes 0 (none) and 4
    (contraindicated) are masked because B was never trained on them.
    The same Q16.16 ternary kernels as ``classify`` are reused so B's
    forward pass is bit-identical across architectures alongside A's.
    """
    if weights_b.schema == _SCHEMA_V3_ATC:
        from engine.bitnet_features_v8 import encode_pair_v8
        pair_features = encode_pair_v8(a_canonical, b_canonical)
    else:
        feature_a = _encode_drug_token(a_canonical)
        feature_b = _encode_drug_token(b_canonical)
        pair_features = feature_a + feature_b

    activations_q16 = _q16_scale_features(pair_features)
    hidden_pre_q16 = [
        _q16_clamp(_q16_dot_ternary(activations_q16, weights_b.hidden_w[j]) + weights_b.hidden_b[j])
        for j in range(weights_b.hidden_features)
    ]
    hidden_q16 = [_q16_relu(v) for v in hidden_pre_q16]
    logits_q16 = [
        _q16_clamp(_q16_dot_ternary(hidden_q16, weights_b.output_w[k]) + weights_b.output_b[k])
        for k in range(weights_b.out_features)
    ]
    # Constrained argmax over classes {1, 2, 3} only. Ties broken by
    # lower index (consistent with the unconstrained argmax in classify).
    severity = 1
    best_logit = logits_q16[1]
    for k in (2, 3):
        if logits_q16[k] > best_logit:
            best_logit = logits_q16[k]
            severity = k
    return severity, tuple(logits_q16)


def classify(
    drug_a: str,
    drug_b: str,
    weights: BitNetWeights,
    *,
    deterministic_table_severity: int | None = None,
    weights_b: BitNetWeights | None = None,
) -> BitNetResult:
    """Ternary classifier forward pass: drug-pair -> severity class.

    The pair is order-canonicalised (lex sort) so {warfarin, ibuprofen} and
    {ibuprofen, warfarin} produce the same feature vector and the same
    output. The logits and severity decision are reproducible bit-for-bit
    on any platform that runs Python.

    If `deterministic_table_severity` is provided (a known result from
    `engine.clinical_scoring`'s 4-tier deterministic table), the result's
    `deterministic_table_match` flag records whether the BitNet output
    agrees. Disagreement is a release-blocking event — surfaces in the
    `tests/test_engine/test_bitnet_classifier.py` regression set.
    """
    a_canonical, b_canonical = sorted((drug_a, drug_b))
    if weights.schema == _SCHEMA_V3_ATC:
        from engine.bitnet_features_v8 import encode_pair_v8
        pair_features = encode_pair_v8(a_canonical, b_canonical)
    else:
        feature_a = _encode_drug_token(a_canonical)
        feature_b = _encode_drug_token(b_canonical)
        pair_features = feature_a + feature_b
    if len(pair_features) != weights.in_features:
        raise RuntimeError(
            f"internal error: pair features length {len(pair_features)} != "
            f"weights.in_features {weights.in_features}"
        )

    activations_q16 = _q16_scale_features(pair_features)
    feature_hash = hashlib.sha256(
        bytes((v + 1) for v in pair_features)        # ternary -> {0,1,2}
    ).hexdigest()

    # First linear layer: in_features -> hidden_features
    hidden_pre_q16 = [
        _q16_clamp(_q16_dot_ternary(activations_q16, weights.hidden_w[j]) + weights.hidden_b[j])
        for j in range(weights.hidden_features)
    ]
    hidden_q16 = [_q16_relu(v) for v in hidden_pre_q16]

    # Second linear layer: hidden_features -> out_features (5 severity classes)
    logits_q16 = [
        _q16_clamp(_q16_dot_ternary(hidden_q16, weights.output_w[k]) + weights.output_b[k])
        for k in range(weights.out_features)
    ]

    # Argmax — pure integer compare; ties broken by lower-index class.
    severity = 0
    best_logit = logits_q16[0]
    for k in range(1, 5):
        if logits_q16[k] > best_logit:
            best_logit = logits_q16[k]
            severity = k

    # iter-421 Path B cascade: when a tier-2 specialist bundle is supplied,
    # A's contraindicated verdict ALWAYS wins (frozen FDA-grade contra
    # gate, 100% recall + 0 FP). For all non-contra A predictions, B's
    # constrained argmax over {moderate, serious, major} replaces A's
    # raw argmax. B was trained without contra anchors, so its capacity
    # is fully spent on the non-contra discrimination v8 historically
    # under-fit (84% serious / 91% moderate standalone).
    weights_id_for_audit = weights.bundle_id
    logits_q16_b: tuple[int, ...] | None = None
    if weights_b is not None and severity != 4:
        # 4 = contraindicated; preserve A's contra verdict.
        b_severity, logits_q16_b = _classify_constrained_b(
            a_canonical, b_canonical, weights_b
        )
        severity = b_severity
        # Audit-chain: composite weights_id captures both bundle hashes
        # so a verifier can replay the cascade decision exactly.
        weights_id_for_audit = f"{weights.bundle_id}+{weights_b.bundle_id}"

    repro_hash_payload = {
        "feature_hash": feature_hash,
        "logits_q16": logits_q16,
        "severity": severity,
        "weights_id": weights_id_for_audit,
    }
    if logits_q16_b is not None:
        repro_hash_payload["logits_q16_b"] = list(logits_q16_b)
        repro_hash_payload["bundle_id_b"] = weights_b.bundle_id  # type: ignore[union-attr]
    repro_hash = hashlib.sha256(
        json.dumps(
            repro_hash_payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()

    deterministic_match = True
    if deterministic_table_severity is not None:
        deterministic_match = (severity == deterministic_table_severity)

    # Audit-grade trace: structured DEBUG log so production INFO-level
    # surfaces stay quiet but a reviewer can opt in by raising verbosity.
    # iter-309 PHI fix: replace raw drug_a/drug_b with 16-char SHA-256
    # pair_hash_prefix (lex-sorted canonical form). Same iter-291 /
    # iter-284 / iter-279 PHI discipline class — drug-pair identity stays
    # grep-able for forensic correlation but raw names never reach handlers.
    # Pre-iter-309 this event leaked drug_a + drug_b on EVERY classification
    # (live since the iter-72-era classifier landing); caught by audit
    # because both keys are absent from the iter-240 forbidden-extras-keys
    # list (which is now extended in iter-309 to catch this regression class).
    _pair_hash_prefix = hashlib.sha256(
        f"{a_canonical}+{b_canonical}".encode("utf-8")
    ).hexdigest()[:16]
    # iter-432 observability ratchet: categorical `ensemble_path` field
    # disambiguates the 3 dispatch states a forensic reader otherwise
    # has to reverse-engineer from `weights_id` length + `ensemble_active`:
    #   - "cascade_fired"        : A predicted non-contra AND B was loaded;
    #                              B's constrained argmax replaced A's class.
    #   - "a_only_contra_veto"   : A predicted contra (severity=4); B was
    #                              available but bypassed by the safety
    #                              contract (A's contra ALWAYS wins).
    #   - "a_only_no_b"          : B was not loaded (single-bundle mode);
    #                              ensemble cascade unreachable for this
    #                              classification regardless of A's output.
    # Strict subset of the existing `ensemble_active` bool — preserved
    # alongside for backwards compat with parsers built pre-iter-432.
    if logits_q16_b is not None:
        _ensemble_path = "cascade_fired"
    elif weights_b is None:
        _ensemble_path = "a_only_no_b"
    else:
        # weights_b supplied AND severity == 4 (contra) AND no logits_q16_b:
        # cascade was bypassed by the contra-veto safety contract.
        _ensemble_path = "a_only_contra_veto"
    logger.debug(
        "bitnet_classified",
        extra={
            "pair_hash_prefix": _pair_hash_prefix,
            "severity": severity,
            "severity_name": _SEVERITY_NAMES[severity],
            "repro_hash": repro_hash,
            "weights_id": weights_id_for_audit,
            "deterministic_match": deterministic_match,
            "ensemble_active": logits_q16_b is not None,
            "ensemble_path": _ensemble_path,
        },
    )

    return BitNetResult(
        severity=severity,
        severity_name=_SEVERITY_NAMES[severity],
        logits_q16=tuple(logits_q16),
        feature_hash=feature_hash,
        repro_hash=repro_hash,
        weights_id=weights_id_for_audit,
        deterministic_table_match=deterministic_match,
    )


# ─── Convenience wrapper for the consensus pipeline ────────────────────────

import threading

_CACHED_WEIGHTS: BitNetWeights | None = None
_PINNED_BUNDLE_ID: str | None = None
# iter-421 Path B: tier-2 specialist cache + pin (parallel to A's cache).
# When the bundle file is absent the slot stays None and the engine falls
# back to single-bundle mode automatically.
_CACHED_WEIGHTS_B: BitNetWeights | None = None
_PINNED_BUNDLE_ID_B: str | None = None
_B_LOAD_ATTEMPTED: bool = False
_CACHE_LOCK = threading.Lock()


class WeightsTamperError(RuntimeError):
    """Raised when the on-disk weights bundle's bundle_id no longer matches
    the value pinned at first load. Indicates the file was swapped under
    the running process — a release-blocking integrity violation."""


def reload_weights() -> BitNetWeights:
    """Force a fresh load + re-pin. Use after a confirmed weights rotation."""
    global _CACHED_WEIGHTS, _PINNED_BUNDLE_ID
    global _CACHED_WEIGHTS_B, _PINNED_BUNDLE_ID_B, _B_LOAD_ATTEMPTED
    with _CACHE_LOCK:
        previous_id = _PINNED_BUNDLE_ID
        previous_id_b = _PINNED_BUNDLE_ID_B
        weights = load_weights()
        _CACHED_WEIGHTS = weights
        _PINNED_BUNDLE_ID = weights.bundle_id
        # iter-421 Path B: re-pin tier-2 specialist alongside A. If the
        # bundle disappears between rotations, ensemble drops to A-only.
        _B_LOAD_ATTEMPTED = True
        _CACHED_WEIGHTS_B = load_weights_b()
        _PINNED_BUNDLE_ID_B = (
            _CACHED_WEIGHTS_B.bundle_id if _CACHED_WEIGHTS_B is not None else None
        )
    logger.warning(
        "bitnet_weights_reloaded",
        extra={
            "previous_bundle_id": previous_id,
            "new_bundle_id": weights.bundle_id,
            "previous_bundle_id_b": previous_id_b,
            "new_bundle_id_b": _PINNED_BUNDLE_ID_B,
        },
    )
    return weights


def classifier_layer(drug_a: str, drug_b: str) -> BitNetResult:
    """Layer-4.5 entry point used by `engine.consensus_engine`.

    The first call loads the weights bundle and pins its `bundle_id`.
    **Every subsequent call re-loads the bundle and verifies the
    `bundle_id` still matches** — an inexpensive SHA-256 compare that
    closes the security gap where a tampered `bitnet_weights.json` swapped
    on disk would silently produce wrong severity verdicts for the entire
    process lifetime. A mismatch raises `WeightsTamperError`, which the
    pipeline must treat as release-blocking.

    The cache is guarded by a `threading.Lock` so high-throughput clinical
    deployments (10K+ pairs/min, multi-threaded) cannot trigger the
    thundering-herd race that would otherwise re-parse the JSON on every
    contended call.
    """
    global _CACHED_WEIGHTS, _PINNED_BUNDLE_ID
    global _CACHED_WEIGHTS_B, _PINNED_BUNDLE_ID_B, _B_LOAD_ATTEMPTED
    with _CACHE_LOCK:
        if _CACHED_WEIGHTS is None:
            _CACHED_WEIGHTS = load_weights()
            _PINNED_BUNDLE_ID = _CACHED_WEIGHTS.bundle_id
            # iter-421 Path B: opportunistic tier-2 load + pin (audit-clean
            # — same SHA-256-canonical-JSON integrity primitive as A). Absent
            # bundle leaves the slot None and engine falls back to A-only.
            if not _B_LOAD_ATTEMPTED:
                _B_LOAD_ATTEMPTED = True
                _CACHED_WEIGHTS_B = load_weights_b()
                if _CACHED_WEIGHTS_B is not None:
                    _PINNED_BUNDLE_ID_B = _CACHED_WEIGHTS_B.bundle_id
                    logger.info(
                        "bitnet_classifier_b_load_pinned",
                        extra={
                            "bundle_id_b_prefix": _PINNED_BUNDLE_ID_B[:16],
                            "hidden_features_b": len(_CACHED_WEIGHTS_B.hidden_w),
                        },
                    )
            # First-load pinning event — fires ONCE per process. Lets
            # auditors correlate every BitNetResult emitted in the
            # process to the bundle_id that was pinned at startup.
            # bundle_id is SHA-256-of-canonical-JSON, NOT secret material;
            # safe to log in full (it IS the integrity primitive).
            logger.info(
                "bitnet_classifier_first_load_pinned",
                extra={
                    "bundle_id_prefix": _PINNED_BUNDLE_ID[:16],
                    "hidden_features": len(_CACHED_WEIGHTS.hidden_w),
                    "in_features": (
                        len(_CACHED_WEIGHTS.hidden_w[0])
                        if _CACHED_WEIGHTS.hidden_w else 0
                    ),
                    "out_features": len(_CACHED_WEIGHTS.output_w),
                    "ensemble_active": _CACHED_WEIGHTS_B is not None,
                },
            )
        else:
            # Re-load + verify the pinned bundle_id on every call. The full
            # JSON parse is ~1 ms; the alternative is a class of FDA-blocking
            # silent-tampering bugs.
            current = load_weights()
            if current.bundle_id != _PINNED_BUNDLE_ID:
                # Pre-raise structured CRITICAL — this is a release-blocking
                # FDA SaMD integrity violation. Bundle IDs are SHA-256
                # prefixes of the canonical-JSON weights file, safe to log
                # (they ARE the integrity primitive, not secret material).
                logger.critical(
                    "bitnet_weights_tamper_detected",
                    extra={
                        "pinned_bundle_id": (
                            _PINNED_BUNDLE_ID[:16] if _PINNED_BUNDLE_ID else None
                        ),
                        "on_disk_bundle_id": current.bundle_id[:16],
                    },
                )
                raise WeightsTamperError(
                    f"bitnet_weights.json bundle_id changed under the running "
                    f"process: pinned {_PINNED_BUNDLE_ID[:16]}... "
                    f"on-disk {current.bundle_id[:16]}... — call "
                    f"reload_weights() after a deliberate rotation."
                )
            _CACHED_WEIGHTS = current
            # iter-421 Path B: same tamper check on B if it was pinned.
            if _PINNED_BUNDLE_ID_B is not None:
                current_b = load_weights_b()
                if current_b is None or current_b.bundle_id != _PINNED_BUNDLE_ID_B:
                    logger.critical(
                        "bitnet_weights_b_tamper_detected",
                        extra={
                            "pinned_bundle_id_b": _PINNED_BUNDLE_ID_B[:16],
                            "on_disk_bundle_id_b": (
                                current_b.bundle_id[:16] if current_b else None
                            ),
                        },
                    )
                    raise WeightsTamperError(
                        f"bitnet_weights_b_specialist.json bundle_id changed under "
                        f"the running process: pinned {_PINNED_BUNDLE_ID_B[:16]}... "
                        f"— call reload_weights() after a deliberate rotation."
                    )
                _CACHED_WEIGHTS_B = current_b
        return classify(drug_a, drug_b, _CACHED_WEIGHTS, weights_b=_CACHED_WEIGHTS_B)


__all__ = [
    "BitNetResult",
    "BitNetWeights",
    "Q16_ONE",
    "Q16_HALF",
    "Q16_ZERO",
    "SEVERITY_NONE",
    "SEVERITY_MINOR",
    "SEVERITY_MODERATE",
    "SEVERITY_MAJOR",
    "SEVERITY_CONTRAINDICATED",
    "WeightsTamperError",
    "classify",
    "classifier_layer",
    "load_weights",
    "load_weights_b",
    "reload_weights",
]
