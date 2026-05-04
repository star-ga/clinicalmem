"""Pin: federation_transport.py emits structured logs at every decision
point, with PHI discipline (no payload bytes leaking into log records).

`engine/federation_transport.py` is the federation control plane bridge
between ClinicalMem and the cross-site MemoryMesh + EventFanout. It is
load-bearing for the JointMemoryFederation flow's 21 typed runtime
invariants — every publish / ingest / quarantine decision must emit a
structured trace that the audit chain can correlate to a peer + scope
without leaking PHI.

Iter-71 ratchet pattern: structured `extra={...}` calls on every
decision point; counts/lengths/categorical metadata only, never
free-form caller-supplied content. PHI Safe-Harbor surface for this
module:

  - `peer_id` is an operator-chosen site identifier; NOT PHI.
  - `scope.value` is a fixed-vocab enum value; NOT PHI.
  - `payload_summary` is caller-supplied dict; MAY carry anything;
    log only `len(payload_summary)` (field count), never values.
  - `endpoint` URL may carry tenant tokens; log only `len(endpoint)`.
  - `reason` for quarantine is a short categorical string; safe.
  - hashes are SHA-256 prefixes; safe.

These tests pin the contract:
  * federation_peer_registered fires on register_clinical_peer
  * federation_publish fires on record_publish_event
  * federation_publish_oversize_batch fires above the 250-block threshold
  * federation_ingest fires on record_ingest_event
  * federation_ingest_conflicts fires when conflicts_resolved > 0
  * federation_quarantine fires (WARNING) on record_quarantine_event
  * No log record leaks payload_summary values, raw endpoints, or
    full hash strings (sentinel-leak scrub).
"""
from __future__ import annotations

import logging

import pytest

# pylint: disable=import-outside-toplevel


_SECRET_PHI_TOKEN = "ZZZ_SECRET_PHI_TOKEN_FROM_PAYLOAD"
_SECRET_ENDPOINT_TOKEN = "ZZZ_SECRET_ENDPOINT_TOKEN"


@pytest.fixture
def caplog_at_debug(caplog):
    caplog.set_level(logging.DEBUG, logger="engine.federation_transport")
    return caplog


def _records(caplog) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.name == "engine.federation_transport"]


def test_register_emits_structured_log(caplog_at_debug):
    from engine.federation_transport import (
        make_site_mesh,
        register_clinical_peer,
    )
    mesh = make_site_mesh()
    register_clinical_peer(
        mesh,
        peer_id="huron-rural-fqhc",
        endpoint=f"https://example.test/peer/{_SECRET_ENDPOINT_TOKEN}/sync",
    )
    records = _records(caplog_at_debug)
    assert any(r.message == "federation_peer_registered" for r in records), (
        "register_clinical_peer must emit 'federation_peer_registered' "
        "structured log"
    )
    # Endpoint token must NOT appear in any log record (length only).
    assert not any(_SECRET_ENDPOINT_TOKEN in str(r.__dict__) for r in records), (
        "endpoint URL leaked into log record — PHI/tenant-token risk"
    )


def test_publish_emits_structured_log_and_no_payload_leak(caplog_at_debug):
    from engine.federation_transport import (
        make_default_fanout,
        make_site_mesh,
        record_publish_event,
        register_clinical_peer,
    )
    mesh = make_site_mesh()
    fanout = make_default_fanout()
    register_clinical_peer(
        mesh, peer_id="boston-academic", endpoint="https://example.test/peer"
    )
    record_publish_event(
        mesh, fanout,
        peer_id="boston-academic",
        payload_summary={"phi_field": _SECRET_PHI_TOKEN, "drug": "warfarin"},
        semantic_idempotency_hash="a" * 64,
        transport_dedup_hash="b" * 64,
        blocks_transferred=12,
    )
    records = _records(caplog_at_debug)
    assert any(r.message == "federation_publish" for r in records), (
        "record_publish_event must emit 'federation_publish' log"
    )
    assert not any(_SECRET_PHI_TOKEN in str(r.__dict__) for r in records), (
        "payload_summary value leaked into log record — PHI risk"
    )
    # Full 64-char hash should NOT appear; only the 16-char prefix.
    assert not any(("a" * 64) in str(r.__dict__) for r in records), (
        "full 64-char idempotency hash leaked; pin uses 16-char prefix"
    )


def test_publish_oversize_batch_warns(caplog_at_debug):
    from engine.federation_transport import (
        make_default_fanout,
        make_site_mesh,
        record_publish_event,
        register_clinical_peer,
    )
    mesh = make_site_mesh()
    fanout = make_default_fanout()
    register_clinical_peer(
        mesh, peer_id="bursting-peer", endpoint="https://example.test/peer"
    )
    record_publish_event(
        mesh, fanout,
        peer_id="bursting-peer",
        payload_summary={},
        semantic_idempotency_hash="c" * 64,
        transport_dedup_hash="d" * 64,
        blocks_transferred=500,  # well above the 250-block threshold
    )
    records = _records(caplog_at_debug)
    oversize = [r for r in records if r.message == "federation_publish_oversize_batch"]
    assert oversize, (
        "publishing > 250 blocks must emit 'federation_publish_oversize_batch' "
        "WARNING — replay-attempt detection signal"
    )
    assert oversize[0].levelname == "WARNING", (
        "oversize-batch log must be WARNING level so it surfaces past "
        "DEBUG suppression"
    )


def test_ingest_emits_structured_log(caplog_at_debug):
    from engine.federation_transport import (
        make_default_fanout,
        make_site_mesh,
        record_ingest_event,
        register_clinical_peer,
    )
    mesh = make_site_mesh()
    fanout = make_default_fanout()
    register_clinical_peer(
        mesh, peer_id="palo-alto-med", endpoint="https://example.test/peer"
    )
    record_ingest_event(
        mesh, fanout,
        peer_id="palo-alto-med",
        payload_summary={"phi_field": _SECRET_PHI_TOKEN},
        blocks_transferred=4,
        conflicts_resolved=0,
    )
    records = _records(caplog_at_debug)
    assert any(r.message == "federation_ingest" for r in records), (
        "record_ingest_event must emit 'federation_ingest' log"
    )
    assert not any(_SECRET_PHI_TOKEN in str(r.__dict__) for r in records), (
        "payload_summary value leaked into ingest log record — PHI risk"
    )


def test_ingest_conflicts_warns(caplog_at_debug):
    from engine.federation_transport import (
        make_default_fanout,
        make_site_mesh,
        record_ingest_event,
        register_clinical_peer,
    )
    mesh = make_site_mesh()
    fanout = make_default_fanout()
    register_clinical_peer(
        mesh, peer_id="conflict-peer", endpoint="https://example.test/peer"
    )
    record_ingest_event(
        mesh, fanout,
        peer_id="conflict-peer",
        payload_summary={},
        blocks_transferred=10,
        conflicts_resolved=2,
    )
    records = _records(caplog_at_debug)
    conflicts = [r for r in records if r.message == "federation_ingest_conflicts"]
    assert conflicts, (
        "conflicts_resolved > 0 must emit 'federation_ingest_conflicts' WARNING"
    )
    assert conflicts[0].levelname == "WARNING"


def test_quarantine_warns_with_categorical_reason_only(caplog_at_debug):
    from engine.federation_transport import (
        make_default_fanout,
        make_site_mesh,
        record_quarantine_event,
        register_clinical_peer,
    )
    mesh = make_site_mesh()
    fanout = make_default_fanout()
    register_clinical_peer(
        mesh, peer_id="rogue-peer", endpoint="https://example.test/peer"
    )
    record_quarantine_event(
        mesh, fanout,
        peer_id="rogue-peer",
        reason="phi_gate_failed",
        payload_summary={"phi_field": _SECRET_PHI_TOKEN},
    )
    records = _records(caplog_at_debug)
    quarantine = [r for r in records if r.message == "federation_quarantine"]
    assert quarantine, (
        "record_quarantine_event must emit 'federation_quarantine' WARNING"
    )
    assert quarantine[0].levelname == "WARNING", (
        "quarantine is a security-grade event; must be WARNING level"
    )
    # Reason is a short categorical string, OK to log.
    assert quarantine[0].__dict__.get("reason") == "phi_gate_failed"
    # PHI in payload_summary must NOT leak.
    assert not any(_SECRET_PHI_TOKEN in str(r.__dict__) for r in records), (
        "quarantine event must not leak payload_summary values"
    )
