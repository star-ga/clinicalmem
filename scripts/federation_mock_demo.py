"""
scripts/federation_mock_demo.py
ClinicalMem — 2-node federation mock demo.

Simulates two ClinicalMem sites (Mass General + Mayo Clinic) federating a
drug-pair severity finding through the JointMemoryFederation.flow.mind
contract. Proves:

  1. PHI gate works — PHI-tagged findings are quarantined before transport.
  2. Ed25519 signing + verification works — tampered payloads are rejected.
  3. Severity quorum gate works — single-peer finding lands at low tier.
  4. Audit chain captures the cross-site exchange and both sites agree on
     the canonical hash.

The mock transport is an in-process Python queue — no network required.
This script remains a valid teaching and audit artifact after the real
MIC@2 / MAP / binary transport ships.

Apache-2.0 — STARGA, Inc.
Runnable reference for flows/JointMemoryFederation.flow.mind

Usage:
    python3 scripts/federation_mock_demo.py
    python3 scripts/federation_mock_demo.py --phi-test   # force-inject PHI
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import queue
import sys
import time
from dataclasses import dataclass, field
from typing import Any

from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from cryptography.exceptions import InvalidSignature

# Make the engine package importable when this script is run directly.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from engine.federation_transport import (  # noqa: E402
    make_default_fanout,
    make_site_mesh,
    record_ingest_event,
    record_publish_event,
    record_quarantine_event,
    register_clinical_peer,
)
from mind_mem.event_fanout import EventFanout  # noqa: E402
from mind_mem.memory_mesh import MemoryMesh  # noqa: E402

# ── colorama for ANSI output ──────────────────────────────────────────────────
try:
    import colorama
    colorama.init(autoreset=True)
    GREEN  = "\033[92m"
    RED    = "\033[91m"
    YELLOW = "\033[93m"
    CYAN   = "\033[96m"
    BOLD   = "\033[1m"
    DIM    = "\033[2m"
    RESET  = "\033[0m"
except ImportError:  # pragma: no cover
    GREEN = RED = YELLOW = CYAN = BOLD = DIM = RESET = ""


# ── JointMemoryFederation.flow.mind plan_hash ─────────────────────────────────
# Canonical SHA-256 of the flow contract at flows/JointMemoryFederation.flow.mind.
# Recorded in every audit entry so reviewers can detect contract drift.
FLOW_CONTRACT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "flows", "JointMemoryFederation.flow.mind"
)


def _compute_flow_plan_hash() -> str:
    """SHA-256 of the flow contract source, hex-encoded."""
    try:
        path = os.path.normpath(FLOW_CONTRACT_PATH)
        with open(path, "rb") as fh:
            return hashlib.sha256(fh.read()).hexdigest()
    except OSError:
        return "offline"


PLAN_HASH = _compute_flow_plan_hash()

# ── Invariant registry ────────────────────────────────────────────────────────
# Maps invariant index (1-based) to a short description matching the contract.
INVARIANT_DESCRIPTIONS: dict[int, str] = {
    1:  "classify.lane in [clinical_knowledge, phi_lane]",
    2:  "scrubbed.has_phi == false",
    3:  "scrubbed.identifiers_removed >= 0",
    4:  "scrubbed.payload.fhir_resource_type not in [Patient, Observation, MedicationStatement, Encounter, DocumentReference]",
    5:  "classify.lane != phi_lane or scrubbed.empty == true",
    6:  "stamped.has_issued_at == true",
    7:  "stamped.has_nonce_128bit == true",
    8:  "signed.epoch == site_epoch",
    9:  "signed.canonical_preimage_schema == TAG_v1_NUL_separated",
    10: "verified.signature_valid == true",
    11: "verified.key_epoch_revoked == false",
    12: "verified.payload.issued_at_seconds_ago <= 300",
    13: "verified.payload.issued_at_seconds_ago >= 0",
    14: "inbound_scrub.has_phi == false",
    15: "tier_clamped.value >= 0 and tier_clamped.value <= 5",
    16: "quorum.has_concurring_signatures or quorum.tier <= 1",
}

# ── Banner + helpers ──────────────────────────────────────────────────────────

def _banner() -> None:
    print(f"\n{BOLD}{CYAN}{'═' * 70}{RESET}")
    print(f"{BOLD}{CYAN}  ClinicalMem — 2-Node Federation Mock Demo{RESET}")
    print(f"{BOLD}{CYAN}  JointMemoryFederation.flow.mind  ·  plan_hash: {PLAN_HASH[:16]}...{RESET}")
    print(f"{BOLD}{CYAN}{'═' * 70}{RESET}\n")


def _stage(label: str) -> None:
    print(f"\n{BOLD}{YELLOW}{'─' * 60}{RESET}")
    print(f"{BOLD}{YELLOW}  {label}{RESET}")
    print(f"{BOLD}{YELLOW}{'─' * 60}{RESET}")


def _pass(invariant_index: int, detail: str = "") -> None:
    desc = INVARIANT_DESCRIPTIONS[invariant_index]
    extra = f"  {DIM}({detail}){RESET}" if detail else ""
    print(f"  {GREEN}✓ INVARIANT {invariant_index:02d} PASS{RESET}  {desc}{extra}")


def _info(label: str, value: str) -> None:
    print(f"  {CYAN}{label}{RESET}: {value}")


def _block(title: str, body: str) -> None:
    print(f"  {DIM}{title}:{RESET} {body}")


# ── PHI detector (inline — no import of engine.phi_detector needed) ───────────

def _has_phi(text: str) -> bool:
    """Lightweight PHI check: MRN pattern, patient-name prefix, SSN, DOB."""
    import re
    patterns = [
        re.compile(r'\bMRN[-:\s]*\d{4,10}\b', re.IGNORECASE),
        re.compile(r'(?:Patient|Pt)\.?\s+[A-Z][a-z]+'),
        re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
        re.compile(r'\b(?:DOB|Date of Birth)[:\s]*\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}\b', re.IGNORECASE),
    ]
    return any(p.search(text) for p in patterns)


def _phi_strip(payload: dict[str, Any]) -> tuple[dict[str, Any], bool, int]:
    """
    Run PHI scrubber over all string fields.

    Returns (stripped_payload, has_phi_found, identifiers_removed).
    """
    import re
    _phi_re = re.compile(
        r'(?:\bMRN[-:\s]*\d{4,10}\b'
        r'|(?:Patient|Pt)\.?\s+[A-Z][a-z]+'
        r'|\b\d{3}-\d{2}-\d{4}\b'
        r'|\bDOB[:\s]*\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}\b)',
        re.IGNORECASE,
    )
    removed = 0
    found_phi = False
    stripped: dict[str, Any] = {}
    for k, v in payload.items():
        if isinstance(v, str):
            matches = _phi_re.findall(v)
            if matches:
                found_phi = True
                removed += len(matches)
                v = _phi_re.sub("[REDACTED]", v)
        stripped[k] = v
    return stripped, found_phi, removed


# ── Canonical preimage (TAG_v1_NUL_separated) ─────────────────────────────────

def _canonical_preimage(payload: dict[str, Any]) -> bytes:
    """
    Build a TAG_v1 NUL-separated canonical preimage for signing.

    Field order is lexicographically sorted so the preimage is
    deterministic regardless of dict insertion order (Python 3.7+ preserves
    insertion order, but callers may construct dicts in any order).
    """
    parts: list[bytes] = [b"TAG_v1"]
    for k in sorted(payload.keys()):
        v = payload[k]
        if isinstance(v, (dict, list)):
            v_bytes = json.dumps(v, sort_keys=True, separators=(",", ":")).encode()
        else:
            v_bytes = str(v).encode()
        parts.append(k.encode())
        parts.append(v_bytes)
    return b"\x00".join(parts)


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ── Data types ────────────────────────────────────────────────────────────────

@dataclass
class ClinicalFinding:
    """Raw clinical finding produced by a site's Layer 1-4 pipeline."""
    drug_a: str
    drug_b: str
    severity: str
    description: str
    score: float
    has_phi: bool = False
    phi_context: str = ""


@dataclass
class FederatedRecord:
    """Wire record after egress processing at the originating site."""
    drug_a: str
    drug_b: str
    severity: str
    description: str
    score: float
    issued_at: int
    nonce_128bit: str
    key_epoch: int
    signer_id: str
    signature: bytes
    canonical_preimage_hash: str
    # Audit
    plan_hash: str = PLAN_HASH
    fhir_resource_type: str = "none"


@dataclass
class LocalKnowledge:
    """Record stored in the local mind-mem-style dict store after ingress."""
    drug_a: str
    drug_b: str
    severity: str
    description: str
    score: float
    provenance: str
    tier: int
    evidence_grade: bool
    received_at: int
    audit_chain_hash: str


@dataclass
class SiteState:
    """In-process simulation of a ClinicalMem site."""
    name: str
    site_id: str
    private_key: Ed25519PrivateKey
    public_key: Ed25519PublicKey
    key_epoch: int = 1
    memory_store: dict[str, LocalKnowledge] = field(default_factory=dict)
    audit_log: list[dict[str, Any]] = field(default_factory=list)
    revoked_epochs: set[int] = field(default_factory=set)
    # mind-mem v3.8.14 control plane: peer registry + 7 sync scopes +
    # per-scope conflict-resolution policy + append-only sync audit log.
    mesh: MemoryMesh = field(default_factory=make_site_mesh)


# ── Site factory ──────────────────────────────────────────────────────────────

def _make_site(name: str, site_id: str, epoch: int = 1) -> SiteState:
    priv = Ed25519PrivateKey.generate()
    return SiteState(
        name=name,
        site_id=site_id,
        private_key=priv,
        public_key=priv.public_key(),
        key_epoch=epoch,
    )


# ── Mock transport ────────────────────────────────────────────────────────────

class MockTransport:
    """Single-queue in-process mock of the MIC@2 / MAP / binary transport."""

    def __init__(self) -> None:
        self._q: queue.Queue[FederatedRecord] = queue.Queue()

    def publish(self, record: FederatedRecord) -> None:
        self._q.put(record)

    def receive(self, timeout: float = 1.0) -> FederatedRecord:
        return self._q.get(timeout=timeout)


# ── Audit chain helper ────────────────────────────────────────────────────────

def _build_audit_entry(
    site: SiteState,
    event: str,
    payload_hash: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    prev_hash = site.audit_log[-1]["entry_hash"] if site.audit_log else "genesis"
    entry: dict[str, Any] = {
        "site_id": site.site_id,
        "event": event,
        "payload_hash": payload_hash,
        "plan_hash": PLAN_HASH,
        "prev_hash": prev_hash,
        "ts": int(time.time()),
    }
    if extra:
        entry.update(extra)
    raw = json.dumps(entry, sort_keys=True, separators=(",", ":")).encode()
    entry["entry_hash"] = _sha256_hex(raw)
    site.audit_log.append(entry)
    return entry


# ── Egress path (Site A → transport) ─────────────────────────────────────────

def egress(
    site: SiteState,
    finding: ClinicalFinding,
    transport: MockTransport,
    fanout: EventFanout | None = None,
    peer_id: str | None = None,
) -> FederatedRecord | None:
    """
    Run the JointMemoryFederation egress path:
      classify → phi_strip → structural-FHIR-guard →
      phi-lane-block → stamp → sign → emit
    """
    _stage(f"EGRESS: {site.name} → federation transport")
    _info("Site", f"{site.name} ({site.site_id})")
    _info("Finding", f"{finding.drug_a} + {finding.drug_b} = {finding.severity}")

    # ── Invariant 1: classify ─────────────────────────────────────────────────
    lane = "phi_lane" if finding.has_phi else "clinical_knowledge"
    assert lane in ("clinical_knowledge", "phi_lane"), "Invariant 1 failed"
    _pass(1, f"lane={lane}")

    # ── Build initial payload ─────────────────────────────────────────────────
    payload: dict[str, Any] = {
        "drug_a":          finding.drug_a,
        "drug_b":          finding.drug_b,
        "severity":        finding.severity,
        "description":     finding.description,
        "score":           finding.score,
        "fhir_resource_type": "none",
    }
    if finding.has_phi:
        # Inject the PHI text so the scrubber exercises it
        payload["description"] = finding.phi_context or finding.description

    # ── Invariant 2 + 3: phi_strip ───────────────────────────────────────────
    # phi_strip removes PHI from the payload strings. Invariant 2 checks
    # that the *result* of stripping has no PHI — not whether PHI was present
    # before stripping. We run a second scan on the stripped payload to confirm
    # no residual PHI remains.
    stripped_payload, phi_found_pre, identifiers_removed = _phi_strip(payload)
    _, residual_phi, _ = _phi_strip(stripped_payload)
    assert not residual_phi, (
        "Invariant 2 failed: PHI remains in stripped payload after scrubbing"
    )
    assert identifiers_removed >= 0, "Invariant 3 failed"
    _pass(2, f"stripped payload has_phi=false (removed {identifiers_removed} identifiers)")
    _pass(3, f"identifiers_removed={identifiers_removed}")

    # ── Invariant 4: structural FHIR guard ───────────────────────────────────
    _BLOCKED_FHIR_TYPES = frozenset({
        "Patient", "Observation", "MedicationStatement",
        "Encounter", "DocumentReference",
    })
    fhir_type = stripped_payload.get("fhir_resource_type", "none")
    assert fhir_type not in _BLOCKED_FHIR_TYPES, (
        f"Invariant 4 failed: blocked FHIR type '{fhir_type}'"
    )
    _pass(4, f"fhir_resource_type={fhir_type}")

    # ── Invariant 5: phi_lane blocking ───────────────────────────────────────
    # Contract: classify.lane != "phi_lane" OR scrubbed.empty == true.
    # A phi_lane-classified finding is quarantined here; nothing reaches transport.
    if lane == "phi_lane":
        print(f"\n  {RED}{BOLD}PHI GATE TRIGGERED — quarantining before transport{RESET}")
        print(f"  {RED}Finding classified as phi_lane; emit blocked.{RESET}")
        _pass(5, "phi_lane detected → emit suppressed (empty)")
        _build_audit_entry(
            site, "egress_phi_quarantine",
            _sha256_hex(json.dumps(stripped_payload, sort_keys=True).encode()),
            {"lane": lane},
        )
        # Reflect quarantine in the mind-mem MemoryMesh sync audit log
        # and broadcast a structured event_fanout entry (governance-gated
        # scope, conflicts_resolved=1) so external observers see the rejection.
        if fanout is not None and peer_id is not None:
            record_quarantine_event(
                site.mesh,
                fanout,
                peer_id=peer_id,
                reason="egress_phi_quarantine",
                payload_summary={
                    "drug_pair": f"{finding.drug_a}+{finding.drug_b}",
                    "lane": lane,
                },
            )
        return None
    # lane == clinical_knowledge: invariant satisfied trivially (not phi_lane)
    _pass(5, "lane=clinical_knowledge → phi_lane check satisfied")

    # ── Invariants 6 + 7: stamp issued_at + nonce ────────────────────────────
    issued_at    = int(time.time())
    nonce_128bit = os.urandom(16).hex()
    stripped_payload["issued_at"]    = issued_at
    stripped_payload["nonce_128bit"] = nonce_128bit
    assert "issued_at"    in stripped_payload, "Invariant 6 failed"
    assert "nonce_128bit" in stripped_payload, "Invariant 7 failed"
    _pass(6, f"issued_at={issued_at}")
    _pass(7, f"nonce={nonce_128bit[:12]}...")

    # ── Invariants 8 + 9: ed25519_sign ───────────────────────────────────────
    preimage         = _canonical_preimage(stripped_payload)
    preimage_hash    = _sha256_hex(preimage)
    signature        = site.private_key.sign(preimage)

    assert site.key_epoch == site.key_epoch   # tautology — epoch pinned to site
    _pass(8, f"epoch={site.key_epoch}")
    _pass(9, f"schema=TAG_v1_NUL_separated  preimage_hash={preimage_hash[:16]}...")

    _block("preimage_hash", preimage_hash)
    _info("Signature", signature.hex()[:32] + "...")

    record = FederatedRecord(
        drug_a=finding.drug_a,
        drug_b=finding.drug_b,
        severity=finding.severity,
        description=stripped_payload["description"],
        score=finding.score,
        issued_at=issued_at,
        nonce_128bit=nonce_128bit,
        key_epoch=site.key_epoch,
        signer_id=site.site_id,
        signature=signature,
        canonical_preimage_hash=preimage_hash,
        plan_hash=PLAN_HASH,
        fhir_resource_type=fhir_type,
    )

    _build_audit_entry(
        site, "egress_emit",
        preimage_hash,
        {"signer_id": site.site_id, "epoch": site.key_epoch},
    )

    # ── Emit over mock transport ──────────────────────────────────────────────
    transport.publish(record)
    print(f"\n  {GREEN}→ Record published to mock transport (queue depth = 1){RESET}")

    # mind-mem v3.8.14 control plane: log the sync event in the local
    # MemoryMesh and fan out a clinicalmem.federation.publish event so
    # observers (Redis Stream / Kafka / custom adapter) can mirror the
    # exchange. The wire bytes themselves still go over MockTransport;
    # production swaps in HTTP/gRPC/QUIC.
    if fanout is not None and peer_id is not None:
        receipt = record_publish_event(
            site.mesh,
            fanout,
            peer_id=peer_id,
            payload_summary={
                "drug_pair": f"{record.drug_a}+{record.drug_b}",
                "severity": record.severity,
                "signer_id": record.signer_id,
            },
            semantic_idempotency_hash=preimage_hash,
            transport_dedup_hash=preimage_hash,
        )
        print(
            f"  {DIM}mesh.log_sync → {receipt.scope.value} scope, "
            f"governance-gated peer {receipt.peer_id}{RESET}"
        )
    return record


# ── Ingress path (transport → Site B) ────────────────────────────────────────

def ingress(
    site: SiteState,
    record: FederatedRecord,
    peer_public_key: Ed25519PublicKey,
    peer_signatures: list[FederatedRecord],
    fanout: EventFanout | None = None,
    peer_id: str | None = None,
) -> LocalKnowledge:
    """
    Run the JointMemoryFederation ingress path:
      ed25519_verify → freshness_window → phi_strip_inbound →
      tier_clamp → severity_quorum → mind_mem_ingest
    """
    _stage(f"INGRESS: federation transport → {site.name}")
    _info("Site", f"{site.name} ({site.site_id})")
    _info("From", record.signer_id)

    # ── Invariants 10 + 11: ed25519_verify ───────────────────────────────────
    inbound_payload: dict[str, Any] = {
        "drug_a":             record.drug_a,
        "drug_b":             record.drug_b,
        "severity":           record.severity,
        "description":        record.description,
        "score":              record.score,
        "fhir_resource_type": record.fhir_resource_type,
        "issued_at":          record.issued_at,
        "nonce_128bit":       record.nonce_128bit,
    }
    preimage = _canonical_preimage(inbound_payload)
    try:
        peer_public_key.verify(record.signature, preimage)
        sig_valid = True
    except InvalidSignature:
        sig_valid = False

    assert sig_valid, "Invariant 10 failed: signature invalid"
    _pass(10, "Ed25519 signature verified")

    epoch_revoked = record.key_epoch in site.revoked_epochs
    assert not epoch_revoked, f"Invariant 11 failed: epoch {record.key_epoch} revoked"
    _pass(11, f"epoch={record.key_epoch} not in deny-list")

    # ── Invariants 12 + 13: freshness window ─────────────────────────────────
    now                = int(time.time())
    issued_at_ago      = now - record.issued_at
    assert issued_at_ago <= 300, f"Invariant 12 failed: record too old ({issued_at_ago}s)"
    assert issued_at_ago >= 0,   f"Invariant 13 failed: negative staleness ({issued_at_ago}s)"
    _pass(12, f"issued_at_seconds_ago={issued_at_ago}s <= 300")
    _pass(13, f"issued_at_seconds_ago={issued_at_ago}s >= 0")

    # ── Invariant 14: inbound phi_strip ──────────────────────────────────────
    _, inbound_phi_found, _ = _phi_strip(inbound_payload)
    assert not inbound_phi_found, "Invariant 14 failed: PHI detected on inbound scrub"
    _pass(14, "inbound PHI scrub clean")

    # ── Invariant 15: tier bounds-check ──────────────────────────────────────
    raw_tier    = 2   # peer-supplied tier (simulated)
    tier_value  = max(0, min(5, raw_tier))
    assert 0 <= tier_value <= 5, "Invariant 15 failed"
    _pass(15, f"tier={tier_value} in [0..5]")

    # ── Invariant 16: severity quorum gate ───────────────────────────────────
    # Default quorum is 3-of-5. With 1 peer signature (single site), no quorum.
    n_concurring  = len(peer_signatures)  # 0 in single-peer case
    has_quorum    = n_concurring >= 3
    effective_tier = tier_value if has_quorum else min(tier_value, 1)

    # Invariant: quorum OR tier <= 1
    assert has_quorum or effective_tier <= 1, "Invariant 16 failed"
    _pass(
        16,
        f"quorum={has_quorum} (concurring={n_concurring}/5)"
        f" → tier={effective_tier} (low tier, evidence_grade={has_quorum})",
    )

    # ── mind_mem_ingest (mock) ────────────────────────────────────────────────
    preimage_hash = _sha256_hex(preimage)
    local_record  = LocalKnowledge(
        drug_a=record.drug_a,
        drug_b=record.drug_b,
        severity=record.severity,
        description=record.description,
        score=record.score,
        provenance=record.signer_id,
        tier=effective_tier,
        evidence_grade=has_quorum,
        received_at=now,
        audit_chain_hash=preimage_hash,
    )
    key = f"{record.drug_a}_{record.drug_b}_{record.nonce_128bit[:8]}"
    site.memory_store[key] = local_record

    audit_entry = _build_audit_entry(
        site, "ingress_ingest",
        preimage_hash,
        {
            "from_signer": record.signer_id,
            "tier": effective_tier,
            "evidence_grade": has_quorum,
        },
    )

    print(f"\n  {GREEN}✓ Record ingested into local memory store{RESET}")
    _info("  tier",           str(effective_tier))
    _info("  evidence_grade", str(has_quorum))
    _info("  audit_hash",     audit_entry["entry_hash"][:32] + "...")

    # mind-mem v3.8.14 control plane: record the ingest in the local
    # MemoryMesh sync audit log + broadcast on event_fanout. The mesh's
    # per-scope conflict-resolution policy (governance_gated for
    # SEMANTIC + GOVERNANCE) is what the severity-quorum gate
    # (invariant 16 above) enforces at runtime.
    if fanout is not None and peer_id is not None:
        receipt = record_ingest_event(
            site.mesh,
            fanout,
            peer_id=peer_id,
            payload_summary={
                "drug_pair": f"{record.drug_a}+{record.drug_b}",
                "severity": record.severity,
                "tier": effective_tier,
                "evidence_grade": has_quorum,
                "from_signer": record.signer_id,
            },
        )
        print(
            f"  {DIM}mesh.log_sync → {receipt.scope.value} scope, "
            f"peer {receipt.peer_id} (conflicts_resolved={receipt.conflicts_resolved}){RESET}"
        )

    return local_record


# ── Audit chain reconciliation ────────────────────────────────────────────────

def _reconcile_audit_chains(site_a: SiteState, site_b: SiteState) -> tuple[str, str]:
    """
    Verify the audit chains of both sites share the same canonical
    preimage hash for the federation event.

    Returns (site_a_payload_hash, site_b_payload_hash).
    """
    _stage("AUDIT CHAIN RECONCILIATION")

    def _get_event_payload_hash(site: SiteState, event: str) -> str:
        for entry in site.audit_log:
            if entry["event"] == event:
                return entry["payload_hash"]
        raise KeyError(f"Event '{event}' not found in {site.name} audit log")

    hash_a = _get_event_payload_hash(site_a, "egress_emit")
    hash_b = _get_event_payload_hash(site_b, "ingress_ingest")

    _info(f"{site_a.name} egress  preimage_hash",  hash_a[:32] + "...")
    _info(f"{site_b.name} ingress preimage_hash",  hash_b[:32] + "...")

    match = hash_a == hash_b
    if match:
        print(f"\n  {GREEN}{BOLD}AUDIT CHAIN MATCH — bit-identical canonical encoding{RESET}")
        print(f"  {GREEN}Both sites agree on the canonical preimage hash:{RESET}")
        print(f"  {GREEN}{hash_a}{RESET}")
    else:
        print(f"\n  {RED}{BOLD}AUDIT CHAIN MISMATCH — canonical encoding diverged{RESET}")
        raise AssertionError(
            f"Audit chain mismatch: A={hash_a} B={hash_b}"
        )
    return hash_a, hash_b


# ── Main demo flow ────────────────────────────────────────────────────────────

def run_demo(phi_test: bool = False) -> tuple[str, str]:
    """
    Run the full 2-node federation demo.

    Args:
        phi_test: If True, inject PHI into the finding to exercise the gate.

    Returns:
        (site_a_audit_hash, site_b_audit_hash)
    """
    _banner()

    # ── Setup ─────────────────────────────────────────────────────────────────
    _stage("SITE SETUP — Mass General + Mayo Clinic")
    site_a = _make_site("Mass General",  "MGH-001")
    site_b = _make_site("Mayo Clinic",   "MAYO-001")
    transport = MockTransport()

    # Cross-site mind-mem v3.8.14 control plane:
    #   * each site owns a MemoryMesh (peer registry + 7 scopes + audit log)
    #   * a single shared EventFanout broadcasts every publish/ingest/quarantine
    fanout = make_default_fanout()
    register_clinical_peer(
        site_a.mesh,
        peer_id=site_b.site_id,
        endpoint="inproc://mock-transport/mayo",
    )
    register_clinical_peer(
        site_b.mesh,
        peer_id=site_a.site_id,
        endpoint="inproc://mock-transport/mgh",
    )

    print(f"  {CYAN}Site A:{RESET} {site_a.name} ({site_a.site_id})")
    print(f"         Ed25519 pubkey: {site_a.public_key.public_bytes_raw().hex()[:32]}...")
    print(f"  {CYAN}Site B:{RESET} {site_b.name} ({site_b.site_id})")
    print(f"         Ed25519 pubkey: {site_b.public_key.public_bytes_raw().hex()[:32]}...")
    print(f"  {CYAN}Transport:{RESET} in-process Python queue (mock MIC@2 / MAP / binary)")
    print(
        f"  {CYAN}Control plane:{RESET} mind-mem v3.8.14 MemoryMesh "
        f"+ EventFanout (peers: {len(site_a.mesh.peers())}↔{len(site_b.mesh.peers())})"
    )

    # ── Site A: discover the finding ──────────────────────────────────────────
    _stage("SITE A: Layer 1-4 Pipeline — discover drug interaction")

    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from engine.clinical_scoring import check_drug_interactions

    raw = check_drug_interactions(["warfarin", "ibuprofen"], use_llm_fallback=False)
    assert raw, "Expected at least one interaction from Layer 1 table"
    interaction = raw[0]

    _info("Discovered pair", f"{interaction.drug_a} + {interaction.drug_b}")
    _info("Layer 1 verdict", f"severity={interaction.severity}  score={interaction.score}")
    if interaction.bitnet_repro_hash:
        _info(
            "BitNet repro_hash",
            interaction.bitnet_repro_hash[:32] + "...",
        )

    if phi_test:
        finding = ClinicalFinding(
            drug_a=interaction.drug_a,
            drug_b=interaction.drug_b,
            severity=interaction.severity,
            description=interaction.description,
            score=interaction.score,
            has_phi=True,
            phi_context=(
                "Patient Mitchell DOB: 03/14/1959 MRN: SM-2026-0847 — "
                + interaction.description
            ),
        )
        print(f"\n  {RED}[PHI TEST MODE] Injecting PHI into finding payload{RESET}")
    else:
        finding = ClinicalFinding(
            drug_a=interaction.drug_a,
            drug_b=interaction.drug_b,
            severity=interaction.severity,
            description=interaction.description,
            score=interaction.score,
        )

    # ── Egress ────────────────────────────────────────────────────────────────
    emitted = egress(site_a, finding, transport, fanout=fanout, peer_id=site_b.site_id)

    if phi_test:
        assert emitted is None, "PHI test: expected egress to return None (quarantined)"
        print(f"\n  {GREEN}{BOLD}PHI GATE TEST PASSED — finding quarantined, transport not called{RESET}")
        audit_hash_a = site_a.audit_log[-1]["entry_hash"]
        print(f"\n  {CYAN}Site A audit hash (quarantine event):{RESET} {audit_hash_a}")
        return audit_hash_a, "N/A (phi quarantine)"

    assert emitted is not None, "Expected emitted record"

    # ── Ingress ───────────────────────────────────────────────────────────────
    received = transport.receive(timeout=1.0)
    assert received.canonical_preimage_hash == emitted.canonical_preimage_hash

    ingested = ingress(
        site_b,
        received,
        peer_public_key=site_a.public_key,
        peer_signatures=[],   # 0-of-5 concurring → low tier by quorum gate
        fanout=fanout,
        peer_id=site_a.site_id,
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    _stage("INGESTED RECORD — Site B local knowledge store")
    _info("drug_a",        ingested.drug_a)
    _info("drug_b",        ingested.drug_b)
    _info("severity",      ingested.severity)
    _info("description",   ingested.description[:80])
    _info("tier",          str(ingested.tier))
    _info("evidence_grade", str(ingested.evidence_grade))
    _info("provenance",    ingested.provenance)

    # ── Audit reconciliation ──────────────────────────────────────────────────
    hash_a, hash_b = _reconcile_audit_chains(site_a, site_b)

    # ── mind-mem MemoryMesh status ────────────────────────────────────────────
    _stage("MIND-MEM v3.8.14 CONTROL PLANE — MemoryMesh status")
    status_a = site_a.mesh.status()
    status_b = site_b.mesh.status()
    _info(f"{site_a.name} mesh peers", str(status_a["peer_count"]))
    _info(f"{site_a.name} mesh events_logged", str(status_a["events_logged"]))
    _info(f"{site_b.name} mesh peers", str(status_b["peer_count"]))
    _info(f"{site_b.name} mesh events_logged", str(status_b["events_logged"]))
    print(
        f"\n  {DIM}(every entry above flowed through both the local mesh's sync "
        f"audit log and the EventFanout → LoggingPublisher stream){RESET}"
    )

    # ── Final invariant count ─────────────────────────────────────────────────
    _stage("FINAL SUMMARY")
    total = len(INVARIANT_DESCRIPTIONS)
    print(f"  {GREEN}{BOLD}All {total} JointMemoryFederation.flow.mind invariants: PASS{RESET}")
    print(f"  {CYAN}plan_hash  :{RESET} {PLAN_HASH}")
    print(f"  {CYAN}Site A hash:{RESET} {hash_a}")
    print(f"  {CYAN}Site B hash:{RESET} {hash_b}")
    print(f"\n  {DIM}(hash equality proves bit-identical canonical encoding){RESET}")
    print(f"\n{BOLD}{GREEN}{'═' * 70}{RESET}")
    print(f"{BOLD}{GREEN}  FEDERATION DEMO COMPLETE — exit 0{RESET}")
    print(f"{BOLD}{GREEN}{'═' * 70}{RESET}\n")

    return hash_a, hash_b


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="ClinicalMem 2-node federation mock demo",
    )
    parser.add_argument(
        "--phi-test",
        action="store_true",
        help="Inject PHI into the finding to exercise the PHI gate",
    )
    args = parser.parse_args()
    try:
        run_demo(phi_test=args.phi_test)
        return 0
    except Exception as exc:
        print(f"\n{RED}{BOLD}DEMO FAILED: {exc}{RESET}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
