"""Unit tests for engine/federation_transport.py.

Verifies the bridge into mind-mem v3.8.14's MemoryMesh + EventFanout:

* :func:`make_site_mesh` returns a MemoryMesh with the documented
  default per-scope conflict-resolution policy.
* :func:`register_clinical_peer` registers with semantic + governance
  scopes — the two scopes the severity-quorum gate (invariant 16 in
  flows/JointMemoryFederation.flow.mind) needs.
* :func:`record_publish_event` writes a sync entry + emits an event.
* :func:`record_ingest_event` mirrors the same shape.
* :func:`record_quarantine_event` writes ``conflicts_resolved=1`` so
  the rejection is visible in the sync audit log.
"""
from __future__ import annotations

import pytest

from mind_mem.event_fanout import Event, Publisher
from mind_mem.memory_mesh import ConflictResolution, SyncScope

from engine.federation_transport import (
    CLINICAL_PEER_SCOPES,
    EVENT_FEDERATION_INGEST,
    EVENT_FEDERATION_PUBLISH,
    EVENT_FEDERATION_QUARANTINE,
    make_default_fanout,
    make_site_mesh,
    record_ingest_event,
    record_publish_event,
    record_quarantine_event,
    register_clinical_peer,
)


class _CapturingPublisher:
    """Test publisher that records every event it receives."""

    name = "capturing"

    def __init__(self) -> None:
        self.events: list[Event] = []

    def publish(self, event: Event) -> None:
        self.events.append(event)

    def close(self) -> None:
        return None


@pytest.fixture()
def capture() -> _CapturingPublisher:
    return _CapturingPublisher()


@pytest.fixture()
def fanout(capture: _CapturingPublisher):
    # Add the capturing publisher in addition to the LoggingPublisher
    # so we can assert event payloads without parsing log lines.
    return make_default_fanout(extra_publishers=(capture,))


# ── make_site_mesh ────────────────────────────────────────────────────────────


def test_make_site_mesh_starts_empty() -> None:
    mesh = make_site_mesh()
    status = mesh.status()
    assert status["peer_count"] == 0
    assert status["events_logged"] == 0


def test_make_site_mesh_has_governance_gated_semantic_policy() -> None:
    """semantic + governance scopes must be governance_gated by default."""
    mesh = make_site_mesh()
    status = mesh.status()
    assert status["policy"]["semantic"] == "governance_gated"
    assert status["policy"]["governance"] == "governance_gated"
    # Hot tier scopes stay last_write_wins
    assert status["policy"]["memories"] == "last_write_wins"
    assert status["policy"]["actions"] == "last_write_wins"


# ── register_clinical_peer ────────────────────────────────────────────────────


def test_register_clinical_peer_uses_clinical_scopes() -> None:
    mesh = make_site_mesh()
    peer = register_clinical_peer(
        mesh,
        peer_id="MGH-001",
        endpoint="https://mgh.example/federation",
    )
    assert peer.peer_id == "MGH-001"
    assert tuple(peer.scopes) == CLINICAL_PEER_SCOPES
    assert SyncScope.SEMANTIC in peer.scopes
    assert SyncScope.GOVERNANCE in peer.scopes


def test_register_clinical_peer_appears_in_status() -> None:
    mesh = make_site_mesh()
    register_clinical_peer(
        mesh,
        peer_id="MAYO-001",
        endpoint="https://mayo.example/federation",
    )
    status = mesh.status()
    assert status["peer_count"] == 1
    assert status["peers"][0]["peer_id"] == "MAYO-001"
    assert "semantic" in status["peers"][0]["scopes"]
    assert "governance" in status["peers"][0]["scopes"]


# ── record_publish_event ──────────────────────────────────────────────────────


def test_record_publish_event_logs_sync_and_emits_event(
    fanout, capture: _CapturingPublisher
) -> None:
    mesh = make_site_mesh()
    register_clinical_peer(
        mesh, peer_id="MAYO-001", endpoint="inproc://mock/mayo"
    )

    receipt = record_publish_event(
        mesh,
        fanout,
        peer_id="MAYO-001",
        payload_summary={"drug_pair": "warfarin+ibuprofen", "severity": "major"},
        semantic_idempotency_hash="deadbeef",
        transport_dedup_hash="cafef00d",
    )

    # Receipt mirrors the input
    assert receipt.peer_id == "MAYO-001"
    assert receipt.scope == SyncScope.SEMANTIC
    assert receipt.semantic_idempotency_hash == "deadbeef"
    assert receipt.transport_dedup_hash == "cafef00d"

    # Mesh recorded the sync
    status = mesh.status()
    assert status["events_logged"] == 1
    audit = mesh.audit_log()
    assert len(audit) == 1
    assert audit[0]["peer_id"] == "MAYO-001"
    assert audit[0]["scope"] == "semantic"
    assert audit[0]["resolution"] == "governance_gated"
    assert audit[0]["blocks_transferred"] == 1
    assert audit[0]["conflicts_resolved"] == 0

    # Fanout emitted exactly one event with the expected kind + payload
    assert len(capture.events) == 1
    event = capture.events[0]
    assert event.kind == EVENT_FEDERATION_PUBLISH
    assert event.payload["peer_id"] == "MAYO-001"
    assert event.payload["semantic_idempotency_hash"] == "deadbeef"
    assert event.payload["transport_dedup_hash"] == "cafef00d"
    assert event.payload["drug_pair"] == "warfarin+ibuprofen"
    assert event.payload["severity"] == "major"


# ── record_ingest_event ───────────────────────────────────────────────────────


def test_record_ingest_event_logs_sync_and_emits_event(
    fanout, capture: _CapturingPublisher
) -> None:
    mesh = make_site_mesh()
    register_clinical_peer(
        mesh, peer_id="MGH-001", endpoint="inproc://mock/mgh"
    )

    receipt = record_ingest_event(
        mesh,
        fanout,
        peer_id="MGH-001",
        payload_summary={
            "drug_pair": "warfarin+ibuprofen",
            "tier": 1,
            "evidence_grade": False,
        },
    )

    assert receipt.peer_id == "MGH-001"
    assert receipt.scope == SyncScope.SEMANTIC
    assert receipt.conflicts_resolved == 0

    audit = mesh.audit_log()
    assert len(audit) == 1
    assert audit[0]["resolution"] == "governance_gated"

    assert len(capture.events) == 1
    event = capture.events[0]
    assert event.kind == EVENT_FEDERATION_INGEST
    assert event.payload["peer_id"] == "MGH-001"
    assert event.payload["tier"] == 1
    assert event.payload["evidence_grade"] is False


# ── record_quarantine_event ───────────────────────────────────────────────────


def test_record_quarantine_event_marks_conflict_resolved(
    fanout, capture: _CapturingPublisher
) -> None:
    """Quarantine events flag conflicts_resolved=1 so operators can spot rejections."""
    mesh = make_site_mesh()
    register_clinical_peer(
        mesh, peer_id="MAYO-001", endpoint="inproc://mock/mayo"
    )

    record_quarantine_event(
        mesh,
        fanout,
        peer_id="MAYO-001",
        reason="egress_phi_quarantine",
        payload_summary={"drug_pair": "warfarin+ibuprofen", "lane": "phi_lane"},
    )

    audit = mesh.audit_log()
    assert len(audit) == 1
    assert audit[0]["scope"] == "governance"
    assert audit[0]["conflicts_resolved"] == 1
    assert audit[0]["blocks_transferred"] == 0

    assert len(capture.events) == 1
    event = capture.events[0]
    assert event.kind == EVENT_FEDERATION_QUARANTINE
    assert event.payload["peer_id"] == "MAYO-001"
    assert event.payload["reason"] == "egress_phi_quarantine"
    assert event.payload["lane"] == "phi_lane"


# ── End-to-end integration ────────────────────────────────────────────────────


def test_publish_then_ingest_round_trip(
    fanout, capture: _CapturingPublisher
) -> None:
    """Two sites round-trip a finding through the bridge: publish + ingest each emit one event."""
    mesh_a = make_site_mesh()
    mesh_b = make_site_mesh()
    register_clinical_peer(mesh_a, peer_id="MAYO-001", endpoint="inproc://mayo")
    register_clinical_peer(mesh_b, peer_id="MGH-001", endpoint="inproc://mgh")

    record_publish_event(
        mesh_a,
        fanout,
        peer_id="MAYO-001",
        payload_summary={"drug_pair": "warfarin+ibuprofen"},
        semantic_idempotency_hash="abc",
        transport_dedup_hash="def",
    )
    record_ingest_event(
        mesh_b,
        fanout,
        peer_id="MGH-001",
        payload_summary={"drug_pair": "warfarin+ibuprofen"},
    )

    # Each side recorded one sync; fanout saw two events.
    assert mesh_a.status()["events_logged"] == 1
    assert mesh_b.status()["events_logged"] == 1
    assert len(capture.events) == 2
    assert capture.events[0].kind == EVENT_FEDERATION_PUBLISH
    assert capture.events[1].kind == EVENT_FEDERATION_INGEST


def test_capturing_publisher_satisfies_publisher_protocol(
    capture: _CapturingPublisher,
) -> None:
    """The test fixture must satisfy mind-mem's Publisher protocol."""
    assert isinstance(capture, Publisher)
