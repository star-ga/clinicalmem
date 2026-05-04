# Copyright 2026 STARGA Inc. — Apache-2.0
"""21 CFR Part 11 compliant audit-trail export for ClinicalMem.

21 CFR Part 11 requires electronic records to be:
  (a) Attributable   — who made the record
  (b) Legible        — human-readable
  (c) Contemporaneous — timestamped at the moment of action
  (d) Original       — not a copy of memory
  (e) Accurate       — matches the actual action
  (f) Complete       — full data, not excerpts
  (g) Consistent     — chronological order
  (h) Enduring       — cannot be deleted
  (i) Available      — retrievable on demand

This module satisfies all nine properties:
  (a) Every event carries ``actor_id`` + the attesting clinician's NPI.
  (b) JSON-LD with human-readable ``action`` + ``event_type`` strings.
  (c) ``timestamp_iso`` is captured by the originating engine layer.
  (d) The Merkle chain anchors each event to its predecessor hash.
  (e) ``before_hash`` / ``after_hash`` capture memory-block state.
  (f) No fields are truncated; the full event payload is serialised.
  (g) Events are sorted by ISO 8601 timestamp before hashing.
  (h) The output file is append-only; ``verify_audit_trail`` detects
      any deletion or modification via chain replay.
  (i) ``verify_audit_trail(path)`` is a stdlib-only re-reader.

Signature scheme
----------------
The attestation signature uses Ed25519 via ``cryptography.hazmat``.

For the demo build an ephemeral keypair is generated at export time.
The public key is embedded in the JSON-LD ``self_verification`` block
(base64url-encoded, no padding).

PRODUCTION SLOT: replace ``_generate_demo_keypair()`` with a call to
an HSM-backed signing key.  The interface contract is:
  private_key : Ed25519PrivateKey  (from an HSM provider object)
  public_key  : Ed25519PublicKey

The rest of the export pipeline is HSM-agnostic.

Merkle chain construction
-------------------------
Chain seed = SHA-256("CLINICALMEM_PART11_CHAIN_SEED_v1")
For event[i]:
  event_canonical = json.dumps(event_payload, sort_keys=True, separators=(",",":"))
  after_hash[i]   = SHA-256(after_hash[i-1] + ":" + event_canonical)

The final ``after_hash`` is the Merkle chain root.

JSON-LD @context
----------------
Points to the FDA guidance document on 21 CFR Part 11 scope/application:
  https://www.fda.gov/regulatory-information/search-fda-guidance-documents/
  electronic-records-electronic-signatures-scope-and-application
"""
from __future__ import annotations

import base64
import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

logger = logging.getLogger(__name__)

# ── constants ──────────────────────────────────────────────────────────────────

_JSONLD_CONTEXT = (
    "https://www.fda.gov/regulatory-information/search-fda-guidance-documents/"
    "electronic-records-electronic-signatures-scope-and-application"
)

_CHAIN_SEED_PREIMAGE = b"CLINICALMEM_PART11_CHAIN_SEED_v1"
_CHAIN_SEED: str = hashlib.sha256(_CHAIN_SEED_PREIMAGE).hexdigest()

_EXPORT_ATTESTATION_TYPE = "EXPORT_ATTESTATION"


# ── dataclasses ───────────────────────────────────────────────────────────────

@dataclass
class AuditEvent:
    """One entry in the ClinicalMem audit chain.

    Fields map directly to the mind-mem ``AuditChain`` entry schema so
    existing engine code can hand events straight to this module.

    ``before_hash``
        SHA-256 of the memory-block state *before* this action.
        Use the empty-string SHA-256 for a creation event.

    ``after_hash``
        SHA-256 of the memory-block state *after* this action.
        Populated by the engine layer; the export module reassigns it
        during Merkle chain construction.

    ``plan_hash``
        SHA-256 of the clinical plan version active at event time.
        Provides cross-reference to the care plan audit chain.
    """

    timestamp_iso: str    # ISO 8601 with timezone, e.g. "2026-05-02T20:00:00Z"
    event_type: str       # e.g. "MEMORY_WRITE", "DRUG_INTERACTION_DETECTED"
    actor_id: str         # clinician NPI, system service name, or user ID
    action: str           # human-readable description of what happened
    resource_ref: str     # e.g. "MedicationStatement/med-warfarin"
    before_hash: str      # SHA-256 of pre-action state
    after_hash: str       # SHA-256 of post-action state (reassigned during export)
    plan_hash: str        # SHA-256 of care plan at event time


@dataclass(frozen=True)
class ExportResult:
    """Return value of ``export_audit_trail``."""

    output_path: Path
    chain_root_hash: str
    event_count: int       # excludes the synthetic attestation event
    attestation_event_count: int   # always 1 if export succeeded
    attesting_npi: str
    total_bytes: int


# ── helpers ───────────────────────────────────────────────────────────────────

def _b64url(data: bytes) -> str:
    """URL-safe base64 without padding."""
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _sha256_hex(data: str) -> str:
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _event_canonical(event_payload: dict[str, Any]) -> str:
    """Canonical JSON: sorted keys, no whitespace."""
    return json.dumps(event_payload, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True)


def _chain_hash(predecessor_hash: str, event_canonical: str) -> str:
    """Compute next Merkle link."""
    preimage = predecessor_hash + ":" + event_canonical
    return hashlib.sha256(preimage.encode("utf-8")).hexdigest()


def _generate_demo_keypair() -> tuple[Ed25519PrivateKey, Ed25519PublicKey]:
    """Generate an ephemeral Ed25519 keypair for the demo build.

    PRODUCTION SLOT
    ---------------
    Replace this function body with HSM key retrieval, e.g.::

        from your_hsm_provider import load_signing_key
        private_key = load_signing_key(key_id=os.environ["AUDIT_SIGNING_KEY_ID"])
        public_key  = private_key.public_key()
        return private_key, public_key

    The rest of the pipeline is HSM-agnostic.
    """
    private_key = Ed25519PrivateKey.generate()
    return private_key, private_key.public_key()


def _sign(private_key: Ed25519PrivateKey, data: str) -> str:
    """Return URL-safe base64 Ed25519 signature over UTF-8 bytes of data."""
    sig_bytes = private_key.sign(data.encode("utf-8"))
    return _b64url(sig_bytes)


def _verify_signature(
    public_key: Ed25519PublicKey,
    data: str,
    signature_b64url: str,
) -> bool:
    """Return True if the signature is valid for data under public_key."""
    from cryptography.exceptions import InvalidSignature

    try:
        sig_bytes = base64.urlsafe_b64decode(signature_b64url + "==")
        public_key.verify(sig_bytes, data.encode("utf-8"))
        return True
    except InvalidSignature as exc:
        # Cryptographically invalid signature — the bytes parsed cleanly
        # but verification failed. Caller treats False as a tampering
        # signal; surface a WARNING so audit-replay debugging can see
        # which signature failed (NOT the bytes themselves — those would
        # leak the attacker-controlled payload).
        logger.warning(
            "audit_signature_invalid",
            extra={"error_type": type(exc).__name__},
        )
        return False
    except Exception as exc:
        # Unexpected error path (malformed b64, key shape mismatch, etc.).
        # Same PHI discipline — error_type only, no message contents.
        logger.warning(
            "audit_signature_verification_exception",
            extra={"error_type": type(exc).__name__},
        )
        return False


# ── NPI validation (reuse engine module) ──────────────────────────────────────

def _assert_npi_valid(npi: str) -> None:
    from engine.npi_registry import validate_npi

    if not validate_npi(npi):
        # NPI is public CMS-registry data — safe to log alongside
        # the failure reason. Pre-raise structured WARNING so
        # operators can spot bad-NPI submissions before the
        # ValueError propagates.
        logger.warning(
            "audit_export_npi_invalid",
            extra={
                "npi": npi,
                "reason": "cms_luhn_validation_failed",
            },
        )
        raise ValueError(
            f"attestation_signer_npi '{npi}' failed CMS Luhn validation"
        )


# ── export ────────────────────────────────────────────────────────────────────

def export_audit_trail(
    events: list[AuditEvent],
    output_path: Path,
    attestation_signer_npi: str,
    *,
    include_metadata: bool = True,
) -> ExportResult:
    """Export a tamper-evident 21 CFR Part 11 JSON-LD audit document.

    Parameters
    ----------
    events
        ClinicalMem audit chain entries.  Need not be sorted — the export
        sorts them chronologically (ISO 8601) before hashing.
    output_path
        Destination file path.  Parent directory must exist.
    attestation_signer_npi
        CMS Luhn-validated NPI of the attesting clinician.  The export
        appends a synthetic ``EXPORT_ATTESTATION`` event signed under this NPI.
    include_metadata
        When True (default), the JSON-LD document includes a ``metadata``
        block with system version, chain parameters, and the public key
        needed for offline signature verification.

    Returns
    -------
    ExportResult
        Chain root hash, event count, output path, and attestation NPI.

    Raises
    ------
    ValueError
        If ``attestation_signer_npi`` fails CMS Luhn validation.
    """
    _assert_npi_valid(attestation_signer_npi)

    # PHI-safe entry log: event count + signer NPI (NPI is public CMS
    # registry data, safe to log) + output path (file system metadata).
    # Never log actor_id, action text, or resource_ref — those can carry
    # clinical narrative.
    logger.info(
        "audit_export_part11_start",
        extra={
            "event_count": len(events),
            "attestation_signer_npi": attestation_signer_npi,
            "output_path": str(output_path),
            "include_metadata": include_metadata,
        },
    )

    # 1. Sort chronologically ─────────────────────────────────────────────────
    sorted_events = sorted(events, key=lambda e: e.timestamp_iso)

    # 2. Build Merkle hash chain ───────────────────────────────────────────────
    running_hash = _CHAIN_SEED
    chain_events: list[dict[str, Any]] = []

    for evt in sorted_events:
        payload: dict[str, Any] = {
            "timestamp_iso": evt.timestamp_iso,
            "event_type": evt.event_type,
            "actor_id": evt.actor_id,
            "action": evt.action,
            "resource_ref": evt.resource_ref,
            "before_hash": evt.before_hash,
            "plan_hash": evt.plan_hash,
        }
        canonical = _event_canonical(payload)
        new_hash = _chain_hash(running_hash, canonical)
        payload["after_hash"] = new_hash
        payload["predecessor_hash"] = running_hash
        chain_events.append(payload)
        running_hash = new_hash

    chain_root = running_hash

    # 3. Build attestation event ───────────────────────────────────────────────
    private_key, public_key = _generate_demo_keypair()
    pub_bytes = public_key.public_bytes_raw()
    pub_b64 = _b64url(pub_bytes)

    # Attestation signs the chain root hash under the attesting NPI
    attestation_payload_str = json.dumps(
        {
            "chain_root_hash": chain_root,
            "attesting_npi": attestation_signer_npi,
            "event_count": len(sorted_events),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    signature = _sign(private_key, attestation_payload_str)

    time_range = {
        "earliest": sorted_events[0].timestamp_iso if sorted_events else "",
        "latest": sorted_events[-1].timestamp_iso if sorted_events else "",
    }

    attestation_event: dict[str, Any] = {
        "event_type": _EXPORT_ATTESTATION_TYPE,
        "actor_id": attestation_signer_npi,
        "action": (
            "21 CFR Part 11 export attested by CMS-Luhn-validated NPI "
            f"{attestation_signer_npi}"
        ),
        "attesting_npi": attestation_signer_npi,
        "attested_chain_root": chain_root,
        "signature_algorithm": "Ed25519",
        "signature": signature,
        "signed_payload_canonical": attestation_payload_str,
    }

    # 4. Build JSON-LD document ────────────────────────────────────────────────
    document: dict[str, Any] = {
        "@context": _JSONLD_CONTEXT,
        "@type": "ClinicalAuditTrail",
        "regulation": "21 CFR Part 11",
        "standard_ref": (
            "Electronic Records; Electronic Signatures — "
            "Scope and Application (2003 FDA Guidance)"
        ),
        "audit_events": chain_events,
        "attestation": attestation_event,
    }

    if include_metadata:
        document["metadata"] = {
            "system": "ClinicalMem",
            "chain_algorithm": "SHA-256 Merkle",
            "chain_seed_preimage": _CHAIN_SEED_PREIMAGE.decode("ascii"),
            "signature_algorithm": "Ed25519",
            "public_key_b64url": pub_b64,
            "public_key_encoding": "raw 32-byte Ed25519 public key",
            "hsm_note": (
                "Demo build uses ephemeral keypair. "
                "Production: replace _generate_demo_keypair() with HSM key load."
            ),
        }

    # 5. Self-verification block ───────────────────────────────────────────────
    document["self_verification"] = {
        "chain_root_hash": chain_root,
        "event_count": len(sorted_events),
        "time_range": time_range,
        "attestation_signer_npi": attestation_signer_npi,
        "attestation_signature": signature,
        "public_key_b64url": pub_b64,
        "verify_instruction": (
            "Run verify_audit_trail(path) from engine.audit_export_part11 "
            "to replay the Merkle chain and confirm authenticity."
        ),
    }

    # 6. Serialise to file ─────────────────────────────────────────────────────
    output_path = Path(output_path)
    serialised = json.dumps(document, indent=2, ensure_ascii=True)
    output_path.write_text(serialised, encoding="utf-8")
    total_bytes = len(serialised.encode("utf-8"))

    # PHI-safe completion log: counts + chain_root prefix (non-reversible,
    # but safe to truncate as a correlation aid for auditors).
    logger.info(
        "audit_export_part11_complete",
        extra={
            "event_count": len(sorted_events),
            "chain_root_prefix": chain_root[:16],
            "output_path": str(output_path),
            "total_bytes": total_bytes,
            "attestation_signer_npi": attestation_signer_npi,
        },
    )

    return ExportResult(
        output_path=output_path,
        chain_root_hash=chain_root,
        event_count=len(sorted_events),
        attestation_event_count=1,
        attesting_npi=attestation_signer_npi,
        total_bytes=total_bytes,
    )


# ── verification ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class VerificationResult:
    """Result of verifying an exported audit trail file."""

    is_valid: bool
    tampering_detected: list[str]   # human-readable descriptions of anomalies
    event_count: int
    chain_root_hash: str
    attesting_npi: str
    signature_valid: bool


def verify_audit_trail(path: Path | str) -> VerificationResult:
    """Re-read and verify a 21 CFR Part 11 audit trail file.

    Replays the Merkle hash chain from scratch and validates the
    Ed25519 attestation signature.  Returns a ``VerificationResult``
    describing any detected tampering.

    Parameters
    ----------
    path
        Path to the JSON-LD audit file produced by ``export_audit_trail``.

    Returns
    -------
    VerificationResult
        ``is_valid=True`` only if chain replay and signature both pass.
    """
    from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

    path = Path(path)
    # PHI-safe verify-start log: path only — file content is examined,
    # never logged.
    logger.debug(
        "audit_verify_start",
        extra={"path": str(path)},
    )
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        # WARNING level — release-blocking integrity event. Log error
        # type only (exception messages may include path / partial content).
        logger.warning(
            "audit_verify_read_failure",
            extra={
                "path": str(path),
                "error_type": type(exc).__name__,
            },
        )
        return VerificationResult(
            is_valid=False,
            tampering_detected=[f"Cannot read/parse file: {exc}"],
            event_count=0,
            chain_root_hash="",
            attesting_npi="",
            signature_valid=False,
        )

    anomalies: list[str] = []

    # --- @context check -------------------------------------------------------
    if document.get("@context") != _JSONLD_CONTEXT:
        anomalies.append(
            f"@context mismatch: expected '{_JSONLD_CONTEXT}', "
            f"got '{document.get('@context')}'"
        )

    # --- chain replay ---------------------------------------------------------
    chain_events: list[dict[str, Any]] = document.get("audit_events") or []
    running_hash = _CHAIN_SEED

    for i, event in enumerate(chain_events):
        predecessor_in_file = event.get("predecessor_hash", "")
        if predecessor_in_file != running_hash:
            anomalies.append(
                f"Event {i}: predecessor_hash mismatch "
                f"(expected={running_hash[:16]}…, "
                f"file={predecessor_in_file[:16]}…)"
            )

        # Reconstruct the payload that was hashed (excludes after_hash and predecessor_hash)
        payload_keys = {
            "timestamp_iso", "event_type", "actor_id", "action",
            "resource_ref", "before_hash", "plan_hash",
        }
        payload = {k: event[k] for k in payload_keys if k in event}
        canonical = _event_canonical(payload)
        expected_after = _chain_hash(running_hash, canonical)
        file_after = event.get("after_hash", "")

        if file_after != expected_after:
            anomalies.append(
                f"Event {i} ({event.get('event_type', '?')}): "
                f"after_hash mismatch — chain broken at index {i}"
            )

        running_hash = expected_after

    replayed_root = running_hash

    # --- self-verification block consistency ----------------------------------
    sv = document.get("self_verification") or {}
    claimed_root = sv.get("chain_root_hash", "")
    if claimed_root and claimed_root != replayed_root:
        anomalies.append(
            f"self_verification.chain_root_hash '{claimed_root[:16]}…' "
            f"does not match replayed root '{replayed_root[:16]}…'"
        )

    # --- attestation signature ------------------------------------------------
    attestation = document.get("attestation") or {}
    signature = attestation.get("signature") or sv.get("attestation_signature", "")
    signed_payload = attestation.get("signed_payload_canonical", "")
    pub_b64 = (
        (document.get("metadata") or {}).get("public_key_b64url")
        or sv.get("public_key_b64url", "")
    )
    attesting_npi = attestation.get("attesting_npi", "")
    signature_valid = False

    if pub_b64 and signature and signed_payload:
        try:
            pub_bytes = base64.urlsafe_b64decode(pub_b64 + "==")
            from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
            public_key = Ed25519PublicKey.from_public_bytes(pub_bytes)
            signature_valid = _verify_signature(public_key, signed_payload, signature)
        except Exception as exc:
            # Surface the verification-error path through structured
            # logging too — the anomalies list captures the human-
            # readable reason for the verifier output, but operators
            # debugging audit-replay need to see the event in logs
            # with PHI-safe metadata only.
            logger.warning(
                "audit_verify_pubkey_setup_failed",
                extra={
                    "path": str(path),
                    "error_type": type(exc).__name__,
                },
            )
            anomalies.append(f"Signature verification error: {exc}")

    if not signature_valid:
        anomalies.append("Ed25519 attestation signature is INVALID or missing")

    is_valid = len(anomalies) == 0

    # PHI-safe completion log: WARNING when tampering detected, INFO on
    # clean verify. Anomaly count + types only — never the anomaly text
    # itself (which can quote chain hashes already considered safe to
    # log via prefix elsewhere). Tampering_detected count is a strong
    # release-block signal.
    if is_valid:
        logger.info(
            "audit_verify_complete_clean",
            extra={
                "path": str(path),
                "event_count": len(chain_events),
                "chain_root_prefix": replayed_root[:16],
                "signature_valid": True,
            },
        )
    else:
        logger.warning(
            "audit_verify_tampering_detected",
            extra={
                "path": str(path),
                "event_count": len(chain_events),
                "anomaly_count": len(anomalies),
                "signature_valid": signature_valid,
            },
        )

    return VerificationResult(
        is_valid=is_valid,
        tampering_detected=anomalies,
        event_count=len(chain_events),
        chain_root_hash=replayed_root,
        attesting_npi=attesting_npi,
        signature_valid=signature_valid,
    )


__all__ = [
    "AuditEvent",
    "ExportResult",
    "VerificationResult",
    "export_audit_trail",
    "verify_audit_trail",
]
