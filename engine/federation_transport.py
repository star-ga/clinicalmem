"""ClinicalMem federation transport bridge.

Wires the JointMemoryFederation.flow.mind control-plane invariants to the
real mind-mem v3.8.14 surface:

* :class:`mind_mem.memory_mesh.MemoryMesh` for peer registry, the seven
  sync scopes (memories / actions / semantic / procedural / relations /
  graph / governance), per-scope conflict-resolution policy
  (last-write-wins vs. governance-gated), and the append-only sync
  audit log.
* :class:`mind_mem.event_fanout.EventFanout` for the governance pub/sub
  stream. Every successful publish / ingest emits a structured event;
  downstream observers (Kafka, Redis Streams, custom webhook publisher)
  plug in by adding their :class:`Publisher` to the fanout instance.

Cross-machine wire transport (HTTP / gRPC / QUIC) is intentionally not
provided here — mind-mem v3.8.14 explicitly defers it (see
``mind_mem.memory_mesh`` module docstring: "the transport layer slots
in around this core"). ClinicalMem ships a single-process queue
implementation in :mod:`scripts.federation_mock_demo` for the
hackathon demo and the regression suite; production deployments
provide their own transport adapter that calls
:func:`record_publish_event` and :func:`record_ingest_event` on
delivery success.

Apache-2.0 — STARGA, Inc.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from mind_mem import memory_mesh
from mind_mem.event_fanout import (
    Event,
    EventFanout,
    LoggingPublisher,
    Publisher,
)


# Default sync scopes the federation control plane registers for every
# clinical-knowledge peer. SEMANTIC + GOVERNANCE are governance-gated
# (no last-write-wins) by mind-mem's default policy — exactly what the
# severity-quorum gate in flows/JointMemoryFederation.flow.mind requires.
CLINICAL_PEER_SCOPES: tuple[memory_mesh.SyncScope, ...] = (
    memory_mesh.SyncScope.SEMANTIC,
    memory_mesh.SyncScope.GOVERNANCE,
)


# Custom event kinds emitted on the federation lane. mind-mem's
# event_fanout accepts any string and only logs a debug warning for
# non-canonical kinds — these live alongside the eight canonical
# governance events.
EVENT_FEDERATION_PUBLISH = "clinicalmem.federation.publish"
EVENT_FEDERATION_INGEST = "clinicalmem.federation.ingest"
EVENT_FEDERATION_QUARANTINE = "clinicalmem.federation.quarantine"


@dataclass(frozen=True)
class PublishReceipt:
    """Returned by :func:`record_publish_event` after a successful emit.

    Mirrors the shape the flow contract's ``mind_mem_publish`` node
    yields: a record handle the caller can store and an event-fanout
    timestamp for observability tracing.
    """

    peer_id: str
    scope: memory_mesh.SyncScope
    blocks_transferred: int
    semantic_idempotency_hash: str
    transport_dedup_hash: str
    issued_at: str


@dataclass(frozen=True)
class IngestReceipt:
    """Returned by :func:`record_ingest_event` after a successful ingest."""

    peer_id: str
    scope: memory_mesh.SyncScope
    blocks_transferred: int
    conflicts_resolved: int
    received_at: str


def make_default_fanout(
    extra_publishers: tuple[Publisher, ...] = (),
) -> EventFanout:
    """Build an EventFanout with the zero-dep LoggingPublisher.

    Adds any ``extra_publishers`` (Redis, Kafka, custom webhook
    adapters) the caller passes. Production deployments swap in
    :class:`mind_mem.event_fanout.RedisStreamPublisher` for cross-worker
    fan-out; the hackathon demo and integration tests use the logging
    publisher because it has no external dependencies.
    """
    return EventFanout([LoggingPublisher(), *extra_publishers])


def register_clinical_peer(
    mesh: memory_mesh.MemoryMesh,
    *,
    peer_id: str,
    endpoint: str,
    scopes: tuple[memory_mesh.SyncScope, ...] = CLINICAL_PEER_SCOPES,
) -> memory_mesh.Peer:
    """Register a peer site in the local MemoryMesh.

    The peer is added with the canonical clinical-knowledge scope set
    (semantic + governance) by default. Both default to
    ``ConflictResolution.GOVERNANCE_GATED`` in mind-mem's policy table,
    which is what the severity-quorum gate (invariant 16 in the flow
    contract) enforces at runtime.
    """
    return mesh.add_peer(peer_id=peer_id, endpoint=endpoint, scopes=scopes)


def record_publish_event(
    mesh: memory_mesh.MemoryMesh,
    fanout: EventFanout,
    *,
    peer_id: str,
    payload_summary: dict[str, Any],
    semantic_idempotency_hash: str,
    transport_dedup_hash: str,
    blocks_transferred: int = 1,
    scope: memory_mesh.SyncScope = memory_mesh.SyncScope.SEMANTIC,
) -> PublishReceipt:
    """Record a successful federated publish in the local mesh + fanout.

    Called by the transport adapter after the wire bytes have been
    handed to the underlying delivery mechanism (queue in the mock
    demo; HTTP / gRPC in a production deployment).
    """
    issued_at = datetime.now(timezone.utc).isoformat()
    mesh.log_sync(
        peer_id=peer_id,
        scope=scope,
        blocks_transferred=blocks_transferred,
        conflicts_resolved=0,
    )
    fanout.publish(
        Event(
            kind=EVENT_FEDERATION_PUBLISH,
            payload={
                "peer_id": peer_id,
                "scope": scope.value,
                "blocks_transferred": blocks_transferred,
                "semantic_idempotency_hash": semantic_idempotency_hash,
                "transport_dedup_hash": transport_dedup_hash,
                **payload_summary,
            },
        )
    )
    return PublishReceipt(
        peer_id=peer_id,
        scope=scope,
        blocks_transferred=blocks_transferred,
        semantic_idempotency_hash=semantic_idempotency_hash,
        transport_dedup_hash=transport_dedup_hash,
        issued_at=issued_at,
    )


def record_ingest_event(
    mesh: memory_mesh.MemoryMesh,
    fanout: EventFanout,
    *,
    peer_id: str,
    payload_summary: dict[str, Any],
    blocks_transferred: int = 1,
    conflicts_resolved: int = 0,
    scope: memory_mesh.SyncScope = memory_mesh.SyncScope.SEMANTIC,
) -> IngestReceipt:
    """Record a successful federated ingest in the local mesh + fanout.

    Mirrors :func:`record_publish_event` for the receive path. Per-scope
    conflict-resolution policy is owned by the mesh
    (governance-gated for SEMANTIC / GOVERNANCE / etc. by default), so
    callers don't pass it on the per-event path.
    """
    received_at = datetime.now(timezone.utc).isoformat()
    mesh.log_sync(
        peer_id=peer_id,
        scope=scope,
        blocks_transferred=blocks_transferred,
        conflicts_resolved=conflicts_resolved,
    )
    fanout.publish(
        Event(
            kind=EVENT_FEDERATION_INGEST,
            payload={
                "peer_id": peer_id,
                "scope": scope.value,
                "blocks_transferred": blocks_transferred,
                "conflicts_resolved": conflicts_resolved,
                **payload_summary,
            },
        )
    )
    return IngestReceipt(
        peer_id=peer_id,
        scope=scope,
        blocks_transferred=blocks_transferred,
        conflicts_resolved=conflicts_resolved,
        received_at=received_at,
    )


def record_quarantine_event(
    mesh: memory_mesh.MemoryMesh,
    fanout: EventFanout,
    *,
    peer_id: str,
    reason: str,
    payload_summary: dict[str, Any] | None = None,
) -> None:
    """Emit a quarantine event when the PHI gate or signature check fails.

    Quarantine writes a sync event with ``conflicts_resolved=1`` so the
    sync audit log reflects the rejection. Operators reading the
    fanout stream can correlate the event back to the originating peer
    + reason.
    """
    mesh.log_sync(
        peer_id=peer_id,
        scope=memory_mesh.SyncScope.GOVERNANCE,
        blocks_transferred=0,
        conflicts_resolved=1,
    )
    fanout.publish(
        Event(
            kind=EVENT_FEDERATION_QUARANTINE,
            payload={
                "peer_id": peer_id,
                "reason": reason,
                **(payload_summary or {}),
            },
        )
    )


def make_site_mesh() -> memory_mesh.MemoryMesh:
    """Construct a MemoryMesh with ClinicalMem's default policy.

    Returned mesh is empty (no peers); callers add peers via
    :func:`register_clinical_peer`. The default per-scope policy is
    inherited from mind-mem (LWW for hot tiers, governance-gated for
    semantic / procedural / relations / graph / governance) which is
    exactly the policy the severity-quorum gate requires.
    """
    return memory_mesh.MemoryMesh()
