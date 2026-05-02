"""Tests for engine.flow_runner.

Covers the four load-bearing claims the flow-runner verifier adds:

  1. **Catalogue completeness** — every `.flow.mind` source in `flows/`
     is discoverable, parseable, and has a stable plan_hash.
  2. **Plan-hash determinism** — the same source bytes always produce
     the same plan_hash on every machine.
  3. **Replay tamper detection** — `verify_replay` returns
     `matches=False` when the recorded plan_hash diverges from the
     current source — the load-bearing FDA SaMD audit-chain integrity
     primitive.
  4. **Contract surface** — every shipped flow has at least one input,
     one output, one node, and one invariant. Empty flows are an
     architectural regression.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_HERE))

from engine.flow_runner import (  # noqa: E402
    FlowContract,
    compute_plan_hash,
    list_flow_names,
    list_flows,
    parse_flow_contract,
    verify_replay,
)

# All six flows ClinicalMem ships today.
SHIPPED_FLOWS: tuple[str, ...] = (
    "MedicationSafetyReview",
    "AllergyCrossReactivity",
    "LabContraindication",
    "ProviderDisagreement",
    "WhatIfSimulation",
    "ClinicalTrialMatch",
)


# ─── catalogue completeness ────────────────────────────────────────────────

def test_all_shipped_flows_discoverable() -> None:
    """`list_flow_names` returns every shipped flow."""
    discovered = set(list_flow_names())
    for name in SHIPPED_FLOWS:
        assert name in discovered, f"flow {name} missing from flows/"


def test_list_flows_returns_contracts() -> None:
    """`list_flows` returns a parsed FlowContract for every shipped flow."""
    contracts = list_flows()
    assert len(contracts) >= len(SHIPPED_FLOWS)
    for c in contracts:
        assert isinstance(c, FlowContract)
        assert c.plan_hash and len(c.plan_hash) == 64


@pytest.mark.parametrize("flow_name", SHIPPED_FLOWS)
def test_each_flow_parses(flow_name: str) -> None:
    """Every shipped flow parses into a valid FlowContract."""
    contract = parse_flow_contract(flow_name)
    assert contract.name  # `flow Name { ... }` extracted
    assert contract.profile  # @profile is required by convention
    assert contract.flow_path.exists()


# ─── plan-hash determinism ────────────────────────────────────────────────

@pytest.mark.parametrize("flow_name", SHIPPED_FLOWS)
def test_plan_hash_stable_across_calls(flow_name: str) -> None:
    """Same flow source -> same plan_hash, every call."""
    a = compute_plan_hash(flow_name)
    b = compute_plan_hash(flow_name)
    assert a == b
    assert len(a) == 64  # SHA-256 hex digest


def test_plan_hashes_differ_across_flows() -> None:
    """Different flows have different plan_hashes."""
    hashes = {name: compute_plan_hash(name) for name in SHIPPED_FLOWS}
    assert len(set(hashes.values())) == len(SHIPPED_FLOWS), (
        "two flows collide on plan_hash — content-addressing is broken"
    )


# ─── replay verifier (audit-chain integrity) ──────────────────────────────

@pytest.mark.parametrize("flow_name", SHIPPED_FLOWS)
def test_replay_matches_current_hash(flow_name: str) -> None:
    """Replay against the current hash returns `matches=True`."""
    current_hash = compute_plan_hash(flow_name)
    result = verify_replay(flow_name, expected_hash=current_hash)
    assert result.matches is True
    assert result.actual_hash == current_hash


def test_replay_detects_tampering() -> None:
    """Replay against a wrong hash returns `matches=False`."""
    result = verify_replay(
        "MedicationSafetyReview",
        expected_hash="0" * 64,
    )
    assert result.matches is False
    assert result.actual_hash != "0" * 64


def test_replay_returns_contract() -> None:
    """The replay result includes the parsed contract for re-rendering."""
    result = verify_replay(
        "MedicationSafetyReview",
        expected_hash=compute_plan_hash("MedicationSafetyReview"),
    )
    assert result.contract.name == "MedicationSafetyReview"
    assert any(p.name == "medications" for p in result.contract.inputs)


# ─── contract surface (architectural regression detection) ────────────────

@pytest.mark.parametrize("flow_name", SHIPPED_FLOWS)
def test_flow_has_at_least_one_input(flow_name: str) -> None:
    contract = parse_flow_contract(flow_name)
    assert len(contract.inputs) >= 1, f"{flow_name} has no inputs"


@pytest.mark.parametrize("flow_name", SHIPPED_FLOWS)
def test_flow_has_at_least_one_output(flow_name: str) -> None:
    contract = parse_flow_contract(flow_name)
    assert len(contract.outputs) >= 1, f"{flow_name} has no outputs"


@pytest.mark.parametrize("flow_name", SHIPPED_FLOWS)
def test_flow_has_at_least_one_node(flow_name: str) -> None:
    contract = parse_flow_contract(flow_name)
    assert len(contract.nodes) >= 1, f"{flow_name} has no nodes"


@pytest.mark.parametrize("flow_name", SHIPPED_FLOWS)
def test_flow_has_at_least_one_invariant(flow_name: str) -> None:
    """Every flow ClinicalMem ships must declare at least one runtime
    invariant; an invariant-free flow is an architectural regression
    that bypasses the safety contract."""
    contract = parse_flow_contract(flow_name)
    assert len(contract.invariants) >= 1, (
        f"{flow_name} has no `invariant` declarations"
    )


# ─── medication-safety-review-specific architectural assertions ────────────

def test_medication_safety_review_has_phi_node() -> None:
    """The flagship safety review must scan PHI before any external call."""
    contract = parse_flow_contract("MedicationSafetyReview")
    assert any(n.name == "phi" for n in contract.nodes)


def test_medication_safety_review_has_bitnet_layer() -> None:
    """Layer 4.5 must be present in the flagship flow."""
    contract = parse_flow_contract("MedicationSafetyReview")
    assert any("bitnet" in n.name for n in contract.nodes)


def test_medication_safety_review_has_consensus_node() -> None:
    """Layer 4 (six-LLM consensus) must be present."""
    contract = parse_flow_contract("MedicationSafetyReview")
    assert any(n.name == "consensus" for n in contract.nodes)


def test_medication_safety_review_demands_consensus_majority_or_abstain() -> None:
    """The abstention contract must be a typed invariant, not a comment."""
    contract = parse_flow_contract("MedicationSafetyReview")
    assert any(
        "majority_or_abstain" in inv.predicate for inv in contract.invariants
    ), "MedicationSafetyReview lost its abstention invariant"
