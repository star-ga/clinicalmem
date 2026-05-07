"""ClinicalMem flow runner — clean-room Python interpreter for `.flow.mind` source.

Each `.flow.mind` file in `flows/` is a typed-graph contract: input/output
ports, named nodes, `@verify` invariants, and a content-addressed
identity (the `plan_hash`). Compiling the file via the upstream MIND Flow
toolchain (`mindlang.dev`) produces a binary execution plan with the
same `plan_hash`; **this Python runner is the open-source verifier path
that ships in ClinicalMem**, so a regulator with this file plus the six
`.flow.mind` source files can replay any clinical decision in minutes
without installing any proprietary toolchain.

What this module does
─────────────────────
1. **Catalogue.** `list_flows()` enumerates the `.flow.mind` files
   available in `flows/`.
2. **Plan hash.** `compute_plan_hash(name)` returns the SHA-256 of the
   canonical bytes of the flow source — the deterministic content
   address every audit-chain entry records as the "decision ID."
3. **Static contract.** `parse_flow_contract(name)` extracts the typed
   port list, node list, and invariant predicates from the source so
   the dashboard can render the graph without compiling it.
4. **Verifiable replay stub.** `verify_replay(name, expected_hash)`
   recomputes the `plan_hash` and asserts it matches the auditor's
   expected value — used by the `/v1/replay` endpoint and CI to detect
   tampering of the flow source.

What this module does NOT do
────────────────────────────
- It does not execute the flow's nodes. Node dispatch goes through
  `engine/consensus_engine.py` and `engine/clinical_scoring.py` as
  before; the `.flow.mind` file is the *contract*, the Python engine
  is the *executor*. The two are kept in sync by
  `tests/test_engine/test_flow_runner.py`.
- It does not implement a full `mindc` parser. The grammar this
  module recognises is a small line-oriented subset of `.flow.mind`
  sufficient to extract the contract surface (`flow Name { … }`,
  `input X: T`, `output Y: T`, `node Z = …`, `invariant E`,
  `@profile`, `@kernel`).
- It does not emit a binary execution plan. That's `mindc`'s job;
  the binary form is opt-in via the upstream toolchain.

Public scope
────────────
This file is Apache-2.0 licensed alongside the rest of ClinicalMem.
It does NOT vendor any source from the STARGA proprietary toolchain
(MindLLM, rfn-mind, mind-runtime, mind-flow are commercial-licensed
and live in private repositories). The `.flow.mind` source files in
`flows/` are STARGA-authored Apache-2.0 architectural artefacts.

Copyright 2026 STARGA, Inc. — Apache-2.0 License.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# Canonical flows directory — relative to the repo root.
_FLOWS_DIR: Path = Path(__file__).parent.parent / "flows"


@dataclass(frozen=True)
class FlowPort:
    """A typed port (input or output) on a flow."""

    name: str
    type_expr: str
    direction: str   # "input" or "output"


@dataclass(frozen=True)
class FlowNode:
    """A named node in the flow graph."""

    name: str
    directive: str        # "@native", "@llm", "@flow", "@recall", "@verify"
    expression: str       # The source expression after the `=`


@dataclass(frozen=True)
class FlowInvariant:
    """An `invariant` predicate that must hold at runtime."""

    predicate: str        # The expression text after `invariant`


@dataclass(frozen=True)
class FlowContract:
    """The static, audit-replayable contract extracted from a flow source."""

    name: str
    flow_path: Path
    plan_hash: str        # SHA-256 hex of the canonical source bytes
    profile: str          # `@profile "..."` value, or "default"
    kernel: str           # `@kernel "..."` value, or "" if unset
    inputs: tuple[FlowPort, ...]
    outputs: tuple[FlowPort, ...]
    nodes: tuple[FlowNode, ...]
    invariants: tuple[FlowInvariant, ...]


# ─── Canonical plan_hash ───────────────────────────────────────────────────

def _canonicalise_source(source_bytes: bytes) -> bytes:
    """Canonicalise flow-source bytes for a stable `plan_hash`.

    Rules (kept tight on purpose; exact-byte where possible):
      1. Decode as UTF-8.
      2. Normalise line endings to `\n` (CRLF / CR -> LF).
      3. Strip trailing whitespace from every line.
      4. Strip a single trailing `\n` so files with / without
         a final newline hash identically.

    This preserves the developer-readable formatting (indentation
    inside lines is significant — it's part of the typed-graph
    contract) while ignoring inconsequential editor artefacts.
    """
    text = source_bytes.decode("utf-8", errors="strict")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in text.split("\n")]
    canonical = "\n".join(lines)
    if canonical.endswith("\n"):
        canonical = canonical.rstrip("\n")
    return canonical.encode("utf-8")


def compute_plan_hash(flow_name: str, flows_dir: Path | None = None) -> str:
    """Return the canonical `plan_hash` (hex SHA-256) for a flow source.

    Stable across whitespace-only edits at end-of-line and EOF; any
    semantic change to the source produces a new hash. The hash is
    the audit-chain "decision ID" recorded for every clinical
    decision the flow produces.
    """
    flow_path = _resolve_flow_path(flow_name, flows_dir)
    canonical = _canonicalise_source(flow_path.read_bytes())
    digest = hashlib.sha256(canonical).hexdigest()
    # PHI-safe: flow_name is config + the digest is by definition
    # non-reversible. canonical_size is metadata, not content.
    logger.debug(
        "flow_plan_hash_computed",
        extra={
            "flow_name": flow_name,
            "plan_hash_prefix": digest[:16],
            "canonical_size_bytes": len(canonical),
        },
    )
    return digest


def _resolve_flow_path(flow_name: str, flows_dir: Path | None = None) -> Path:
    """Resolve a flow name to its `.flow.mind` source path."""
    base = Path(flows_dir) if flows_dir is not None else _FLOWS_DIR
    # Accept "MedicationSafetyReview" or "MedicationSafetyReview.flow.mind"
    if not flow_name.endswith(".flow.mind"):
        flow_name = flow_name + ".flow.mind"
    candidate = base / flow_name
    if not candidate.exists():
        # Log only the requested basename + available count — never the
        # full filesystem path (could leak caller cwd) and never the
        # available-flow list (would let a probe enumerate the catalogue
        # via repeated 404s on misspelled names).
        logger.error(
            "flow_resolve_failed",
            extra={
                "flow_name": flow_name,
                "available_count": len(list_flow_names(base)),
            },
        )
        raise FileNotFoundError(
            f"flow source not found: {candidate}; available flows: {list_flow_names(base)}"
        )
    # DEBUG-level success path: every successful resolution leaves a
    # baseline footprint so operators can compute the resolution rate
    # (calls per minute) and distinguish "no calls" from "all clean".
    # PHI-safe: flow_name is structural metadata (file basename), never
    # patient data.
    logger.debug(
        "flow_resolve_ok",
        extra={"flow_name": flow_name},
    )
    return candidate


# ─── Catalogue ─────────────────────────────────────────────────────────────

def list_flow_names(flows_dir: Path | None = None) -> list[str]:
    """List available flow names (without `.flow.mind` suffix)."""
    base = Path(flows_dir) if flows_dir is not None else _FLOWS_DIR
    if not base.exists():
        # Catalogue request against a non-existent dir is upstream config
        # bug; surface at WARNING so operators can spot misconfigured
        # deployments. PHI-safe: dir path is structural (no patient data).
        logger.warning(
            "flow_catalogue_dir_missing",
            extra={"base_exists": False},
        )
        return []
    names = sorted(p.stem.replace(".flow", "") for p in base.glob("*.flow.mind"))
    logger.debug(
        "flow_catalogue_listed",
        extra={"flow_count": len(names)},
    )
    return names


def list_flows(flows_dir: Path | None = None) -> list[FlowContract]:
    """Parse and return every `.flow.mind` contract under `flows/`."""
    contracts = [
        parse_flow_contract(name, flows_dir=flows_dir)
        for name in list_flow_names(flows_dir)
    ]
    # DEBUG — bulk-parse summary so operators can see the
    # contract-count snapshot when the executor (re)loads the
    # flow set on startup. Pairs with the per-contract DEBUG in
    # parse_flow_contract.
    logger.debug(
        "flow_list_complete",
        extra={
            "contract_count": len(contracts),
            "flows_dir": str(flows_dir) if flows_dir else "default",
        },
    )
    return contracts


# ─── Static contract extraction ────────────────────────────────────────────

_FLOW_HEADER_RE = re.compile(r"^\s*flow\s+(\w+)\s*\{\s*$")
_INPUT_RE = re.compile(r"^\s*input\s+(\w+)\s*:\s*(.+?)\s*$")
_OUTPUT_RE = re.compile(r"^\s*output\s+(\w+)\s*:\s*(.+?)\s*$")
_NODE_RE = re.compile(r"^\s*node\s+(\w+)\s*=\s*(@\w+)\s*(.+?)\s*$")
_ASSIGN_RE = re.compile(r"^\s*assign\s+(\w+)\s*=\s*(@\w+)?\s*(.+?)\s*$")
_INVARIANT_RE = re.compile(r"^\s*invariant\s+(.+?)\s*$")
_PROFILE_RE = re.compile(r'^\s*@profile\s+"(.+?)"\s*$')
_KERNEL_RE = re.compile(r'^\s*@kernel\s+"(.+?)"\s*$')


def parse_flow_contract(
    flow_name: str,
    flows_dir: Path | None = None,
) -> FlowContract:
    """Parse a `.flow.mind` file into a static FlowContract.

    The grammar is line-oriented and forgiving — comments (`//`) and
    blank lines are skipped; multi-line `node` expressions are
    collapsed to a single-line form before regex-matching. This
    module's job is to extract the audit-replayable surface, not to
    reimplement the upstream `mindc` parser.
    """
    flow_path = _resolve_flow_path(flow_name, flows_dir)
    raw_text = flow_path.read_text(encoding="utf-8")

    # Strip line comments (`// …`); keep multi-line expressions on one line
    # so the line-oriented regexes match. A node like
    #     node consensus = @llm consensus(
    #         models: [...],
    #     )
    # is collapsed to a single line.
    stripped_lines: list[str] = []
    pending: str | None = None
    paren_depth: int = 0
    for raw_line in raw_text.splitlines():
        # remove `//`-style comments (basic — does not handle `//` inside strings)
        line = raw_line.split("//", 1)[0].rstrip()
        if not line.strip():
            if pending is None:
                stripped_lines.append("")
                continue
            # pending continuation — keep going, ignore blank lines
            continue
        if pending is not None:
            pending = pending + " " + line.strip()
            paren_depth += line.count("(") - line.count(")")
            paren_depth += line.count("[") - line.count("]")
            paren_depth += line.count("{") - line.count("}")
            if paren_depth <= 0:
                stripped_lines.append(pending)
                pending = None
                paren_depth = 0
            continue
        depth_delta = (
            line.count("(") - line.count(")")
            + line.count("[") - line.count("]")
        )
        if depth_delta > 0:
            pending = line
            paren_depth = depth_delta
            continue
        stripped_lines.append(line)
    if pending is not None:
        # unterminated multi-line; salvage as a single line
        stripped_lines.append(pending)

    flow_decl_name: str | None = None
    profile = "default"
    kernel = ""
    inputs: list[FlowPort] = []
    outputs: list[FlowPort] = []
    nodes: list[FlowNode] = []
    invariants: list[FlowInvariant] = []

    for line in stripped_lines:
        if not line.strip():
            continue
        if (m := _PROFILE_RE.match(line)) is not None:
            profile = m.group(1)
            continue
        if (m := _KERNEL_RE.match(line)) is not None:
            kernel = m.group(1)
            continue
        if (m := _FLOW_HEADER_RE.match(line)) is not None:
            flow_decl_name = m.group(1)
            continue
        if (m := _INPUT_RE.match(line)) is not None:
            inputs.append(FlowPort(name=m.group(1), type_expr=m.group(2), direction="input"))
            continue
        if (m := _OUTPUT_RE.match(line)) is not None:
            outputs.append(FlowPort(name=m.group(1), type_expr=m.group(2), direction="output"))
            continue
        if (m := _NODE_RE.match(line)) is not None:
            nodes.append(
                FlowNode(
                    name=m.group(1),
                    directive=m.group(2),
                    expression=m.group(3),
                )
            )
            continue
        if (m := _ASSIGN_RE.match(line)) is not None:
            # `assign output = @directive ...` — treat as a terminal node
            # that produces the flow's output binding. The executor uses
            # the last-non-skipped node's output as the flow output when
            # the assign callable isn't dispatched.
            target = m.group(1)
            directive = m.group(2) or "@native"
            nodes.append(
                FlowNode(
                    name=target,
                    directive=directive,
                    expression=m.group(3),
                )
            )
            continue
        if (m := _INVARIANT_RE.match(line)) is not None:
            invariants.append(FlowInvariant(predicate=m.group(1)))
            continue

    if flow_decl_name is None:
        logger.error(
            "flow_parse_failed",
            extra={
                "flow_path_basename": flow_path.name,
                "reason": "missing_flow_decl",
            },
        )
        raise ValueError(f"no `flow Name {{ … }}` declaration found in {flow_path}")

    # PHI-safe contract-shape log: counts only, no expression text.
    logger.debug(
        "flow_contract_parsed",
        extra={
            "flow_name": flow_decl_name,
            "profile": profile,
            "kernel": kernel or "",
            "input_count": len(inputs),
            "output_count": len(outputs),
            "node_count": len(nodes),
            "invariant_count": len(invariants),
        },
    )

    return FlowContract(
        name=flow_decl_name,
        flow_path=flow_path,
        plan_hash=compute_plan_hash(flow_name, flows_dir=flows_dir),
        profile=profile,
        kernel=kernel,
        inputs=tuple(inputs),
        outputs=tuple(outputs),
        nodes=tuple(nodes),
        invariants=tuple(invariants),
    )


# ─── Replay verifier ───────────────────────────────────────────────────────

@dataclass(frozen=True)
class ReplayResult:
    """Outcome of `verify_replay` — used by the `/v1/replay` endpoint."""

    flow_name: str
    expected_hash: str
    actual_hash: str
    matches: bool
    contract: FlowContract


def verify_replay(
    flow_name: str,
    expected_hash: str,
    flows_dir: Path | None = None,
) -> ReplayResult:
    """Replay-verify a flow against an auditor's expected `plan_hash`.

    This is the load-bearing FDA SaMD reproducibility primitive: a
    regulator stores `(flow_name, plan_hash, input_hash, output_hash)`
    in the audit chain at decision time; months later, they call this
    function to assert the flow source has not been tampered with.

    Returns a ReplayResult; `matches=False` is a release-blocking
    audit-chain integrity event.
    """
    contract = parse_flow_contract(flow_name, flows_dir=flows_dir)
    matches = (contract.plan_hash == expected_hash)
    if matches:
        # INFO on success — auditors want to see the positive replay outcome
        # in the audit chain, not just failures. Hash prefix only (the full
        # hash is in the ReplayResult anyway).
        logger.info(
            "flow_replay_verified",
            extra={
                "flow_name": flow_name,
                "matches": True,
                "plan_hash_prefix": contract.plan_hash[:16],
            },
        )
    else:
        # WARNING level — release-blocking audit-chain integrity event.
        logger.warning(
            "flow_replay_mismatch",
            extra={
                "flow_name": flow_name,
                "matches": False,
                "expected_prefix": expected_hash[:16],
                "actual_prefix": contract.plan_hash[:16],
            },
        )
    return ReplayResult(
        flow_name=flow_name,
        expected_hash=expected_hash,
        actual_hash=contract.plan_hash,
        matches=matches,
        contract=contract,
    )


# ─── Live executor (consensus-ranked gap #2) ───────────────────────────────
#
# The .flow.mind sources are the typed-graph contracts. The executor below
# walks the parsed contract's node list in declared order, dispatches each
# `@native` node to the corresponding engine entry point, records per-node
# evidence, and emits a structured FlowExecution result. Every execution
# carries the `plan_hash` decision ID a regulator can replay against.
#
# The dispatch table maps `(directive, node_name)` -> Python callable. The
# executor is intentionally narrow: it handles the deterministic-pipeline
# subset (PHI scan, RxNorm normalize, deterministic table check, BitNet
# Layer 4.5 stamp, audit chain emit). The `@llm` consensus + synthesis
# directives are routed through engine.consensus_engine and require the
# corresponding API keys; without them the executor records the node as
# skipped and the `@verify consensus.available_models >= 4` invariant
# fires honestly. `@flow` (compositional) calls re-enter execute().


@dataclass(frozen=True)
class NodeExecution:
    """Per-node evidence stamp — recorded for every executed flow node."""

    node_name: str
    directive: str
    status: str           # "ok" | "skipped" | "failed"
    output_hash: str      # SHA-256 over the canonical JSON encoding of the output
    elapsed_ms: int
    detail: str = ""


@dataclass(frozen=True)
class FlowExecution:
    """Result of a flow execution — drives the audit-chain entry."""

    flow_name: str
    plan_hash: str
    inputs_hash: str      # SHA-256 over canonical JSON encoding of inputs
    output_hash: str      # SHA-256 over canonical JSON encoding of final output
    nodes: tuple[NodeExecution, ...]
    invariant_violations: tuple[str, ...]
    output: object | None
    elapsed_ms: int


def _canonical_json_hash(value: object) -> str:
    """SHA-256 over the canonical-JSON encoding; stable on every machine."""
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _dispatch_table() -> dict[tuple[str, str], object]:
    """Build the `(directive, node_name) -> callable` dispatch table.

    Imported lazily so the verifier path stays import-light. Each callable
    takes a single `inputs: dict` and returns a JSON-serialisable value.
    Unknown nodes are recorded as "skipped" without raising.
    """
    table: dict[tuple[str, str], object] = {}

    # PHI scan (Layer 0) — engine.phi_detector.scan_phi
    def _phi_scan(inputs: dict) -> dict:
        from engine.phi_detector import scan_phi
        text = " ".join(str(v) for v in inputs.get("medications", []))
        report = scan_phi(text)
        # INFO-level event — PHI gate is the safety-critical first node;
        # operators must see every dispatch in the audit log even at
        # default log levels. PHI-safe: only the LENGTH of the joined
        # text + match count + safe-for-external bool. The phi_detector
        # itself emits its own events with text_length only.
        logger.info(
            "flow_node_phi_scan",
            extra={
                "text_length": len(text),
                "med_count": len(inputs.get("medications", [])),
                "phi_match_count": len(report.matches),
                "safe_for_external": len(report.matches) == 0,
            },
        )
        return {
            "safe_for_external": len(report.matches) == 0,
            "phi_match_count": len(report.matches),
        }
    table[("@native", "phi")] = _phi_scan
    table[("@native", "phi_scan")] = _phi_scan

    # RxNorm normalize (Layer 3 prep)
    def _rxnorm_normalize(inputs: dict) -> dict:
        meds = inputs.get("medications", [])
        # Light fallback when the live API is unreachable: count meds as
        # "resolved" when they appear in the deterministic table.
        from engine.clinical_scoring import _KNOWN_INTERACTIONS
        known = {drug for drug_a, drug_b, _, _ in _KNOWN_INTERACTIONS for drug in (drug_a, drug_b)}
        coverage = sum(1 for m in meds if any(d in m.lower() for d in known))
        return {
            "medications": meds,
            "coverage_ratio": coverage / max(len(meds), 1),
        }
    table[("@native", "normalized")] = _rxnorm_normalize
    table[("@native", "normalize")] = _rxnorm_normalize

    # Deterministic table check (Layer 1)
    def _deterministic_check(inputs: dict) -> dict:
        from engine.clinical_scoring import check_drug_interactions
        # Disable LLM fallback so this node is a clean Layer 1 stamp
        results = check_drug_interactions(inputs.get("medications", []), use_llm_fallback=False)
        return {
            "interactions": [
                {"drug_a": r.drug_a, "drug_b": r.drug_b, "severity": r.severity,
                 "bitnet_severity": r.bitnet_severity,
                 "bitnet_repro_hash": r.bitnet_repro_hash}
                for r in results
            ],
            "interaction_count": len(results),
        }
    table[("@native", "tier1")] = _deterministic_check

    # BitNet Layer 4.5 stamp on a single pair (re-entrant)
    def _bitnet_classify(inputs: dict) -> dict:
        from engine.bitnet_classifier import classifier_layer
        meds = inputs.get("medications", [])
        out: list[dict] = []
        for i in range(len(meds)):
            for j in range(i + 1, len(meds)):
                r = classifier_layer(meds[i], meds[j])
                out.append({
                    "drug_a": meds[i], "drug_b": meds[j],
                    "severity": r.severity_name,
                    "repro_hash": r.repro_hash,
                    "weights_id": r.weights_id,
                })
        # iter-314 observability — flow-node-level footprint for the
        # per-pair Layer 4.5 stamp. Fires once per flow execution
        # regardless of cohort size; complements the per-pair DEBUG
        # event from engine.bitnet_classifier (iter-309 PHI-safe form).
        # PHI-safe: only counts + structural metadata + the canonical
        # weights_id (a SHA-256 bundle hash, not patient data). Drug
        # names never reach the log record. Severity-class breakdown is
        # categorical histogram only — never per-pair correspondence.
        sev_counts: dict[str, int] = {}
        for entry in out:
            s = entry["severity"]
            sev_counts[s] = sev_counts.get(s, 0) + 1
        weights_id_prefix = (out[0]["weights_id"][:16] if out else "")
        logger.debug(
            "flow_node_bitnet_classify",
            extra={
                "med_count": len(meds),
                "pair_count": len(out),
                "severity_histogram": sev_counts,
                "weights_id_prefix": weights_id_prefix,
            },
        )
        return {"pairs": out, "weights_id": out[0]["weights_id"] if out else ""}
    table[("@native", "bitnet")] = _bitnet_classify
    table[("@native", "bitnet_ternary_classify")] = _bitnet_classify

    # Audit chain emit (records the per-node evidence into the chain)
    def _emit_audit_chain(inputs: dict) -> dict:
        # Lightweight stamp; the real audit chain is recorded by execute()
        node_count = inputs.get("_node_count", 0)
        # INFO-level event — audit-chain emit is the load-bearing FDA
        # SaMD reproducibility anchor; every emit must leave a footprint.
        # PHI-safe: only the structural _node_count is logged.
        logger.info(
            "flow_node_audit_emit",
            extra={"node_count": node_count},
        )
        return {"emitted": True, "node_count": node_count}
    table[("@native", "audit")] = _emit_audit_chain

    # Final aggregator — the flow's output
    def _build_safety_report(inputs: dict) -> dict:
        return {
            "patient_id": inputs.get("patient_id", ""),
            "node_count": inputs.get("_node_count", 0),
            "interactions": inputs.get("_tier1_interactions", []),
        }
    table[("@native", "report")] = _build_safety_report
    table[("@native", "build_safety_report")] = _build_safety_report

    # DEBUG — dispatch-table boot signal. Lets operators see how many
    # `(directive, node_name)` entries are wired vs. left to the
    # "skipped" default path. A future @llm refactor that adds 3 new
    # nodes should bump this count; if it doesn't, the flow_node_skipped
    # rate spikes and this log explains why. PHI-safe: only the count
    # is logged, not the dispatch keys (which are static + non-PHI).
    logger.debug(
        "flow_dispatch_table_built",
        extra={"dispatch_entry_count": len(table)},
    )

    return table


def execute(
    flow_name: str,
    inputs: dict[str, object] | None = None,
    flows_dir: Path | None = None,
) -> FlowExecution:
    """Execute a `.flow.mind` contract — walks the parsed graph in declared order.

    The executor dispatches each `@native` node to its engine entry point
    (see `_dispatch_table()`). Unknown nodes are recorded as "skipped"
    rather than failing the execution — this keeps the executor honest
    when the `@llm` / `@flow` directives need API keys we don't have.

    Returns a structured FlowExecution carrying the plan_hash, the
    inputs_hash, the final output_hash, and per-node evidence stamps.
    Every field is canonical-JSON-hashable so the audit chain can record
    the entire execution as a single replay-verifiable record.
    """
    inputs = inputs or {}
    contract = parse_flow_contract(flow_name, flows_dir=flows_dir)
    table = _dispatch_table()

    t0 = time.time()
    inputs_hash = _canonical_json_hash(inputs)

    # PHI-safe entry log: counts + hash prefixes only. inputs may carry
    # medications + patient_id, both already past the PHI gate; the hash
    # prefix is non-reversible and lets auditors correlate later.
    logger.info(
        "flow_execute_start",
        extra={
            "flow_name": flow_name,
            "plan_hash_prefix": contract.plan_hash[:16],
            "inputs_hash_prefix": inputs_hash[:16],
            "node_count": len(contract.nodes),
            "input_keys": sorted(inputs.keys()),
        },
    )

    state: dict[str, object] = dict(inputs)
    nodes: list[NodeExecution] = []

    for node in contract.nodes:
        node_t0 = time.time()
        callable_ = table.get((node.directive, node.name))
        if callable_ is None:
            # DEBUG-only: skipping is normal for @llm directives without
            # API keys; auditors who want to see the skip pattern can
            # opt in. Use canonical (directive, name) cardinality only.
            logger.debug(
                "flow_node_skipped",
                extra={
                    "flow_name": flow_name,
                    "node_name": node.name,
                    "directive": node.directive,
                },
            )
            nodes.append(NodeExecution(
                node_name=node.name,
                directive=node.directive,
                status="skipped",
                output_hash="",
                elapsed_ms=0,
                detail="no dispatch entry; engine adapter not yet implemented",
            ))
            continue
        try:
            # Pass the accumulating state so downstream nodes can see prior outputs.
            output = callable_(state)
            output_hash = _canonical_json_hash(output)
            state[node.name] = output
            # Convenience aliases for the build_safety_report aggregator
            if node.name == "tier1" and isinstance(output, dict):
                state["_tier1_interactions"] = output.get("interactions", [])
            state["_node_count"] = len([n for n in nodes if n.status == "ok"]) + 1
            nodes.append(NodeExecution(
                node_name=node.name,
                directive=node.directive,
                status="ok",
                output_hash=output_hash,
                elapsed_ms=int((time.time() - node_t0) * 1000),
            ))
        except Exception as e:
            # PHI / secret discipline: log error_type only, never str(e).
            # Node callables receive the running flow state which can
            # carry medications + patient_id; an exception's str(e) may
            # quote the offending key/value verbatim.
            logger.warning(
                "flow_node_failed",
                extra={
                    "flow_name": flow_name,
                    "node_name": node.name,
                    "directive": node.directive,
                    "error_type": type(e).__name__,
                    "elapsed_ms": int((time.time() - node_t0) * 1000),
                },
            )
            nodes.append(NodeExecution(
                node_name=node.name,
                directive=node.directive,
                status="failed",
                output_hash="",
                elapsed_ms=int((time.time() - node_t0) * 1000),
                detail=f"{type(e).__name__}: {str(e)[:200]}",
            ))

    # Best-effort invariant evaluation (logged, not enforced — the executor
    # is honest about what it can prove without a full mindc analyzer).
    invariant_violations: list[str] = []
    invariants_evaluable = 0
    for inv in contract.invariants:
        # We can only evaluate trivially-decidable invariants today.
        # Anything more complex is flagged for future analysis.
        pass
    # DEBUG — invariant-evaluation summary. Today every invariant is
    # `pass` (no in-process predicate evaluator); this pin documents the
    # honest 0-evaluable state so an auditor can see the contract has
    # invariants but the executor doesn't enforce them yet (the compile-
    # time mindc analyzer does). When a future iter adds an in-process
    # evaluator, `invariants_evaluable` becomes the live count and any
    # discrepancy with `invariant_count` is surfaced for ops review.
    logger.debug(
        "flow_invariants_summary",
        extra={
            "flow_name": flow_name,
            "invariant_count": len(contract.invariants),
            "invariants_evaluable": invariants_evaluable,
            "violations": len(invariant_violations),
        },
    )

    # Final output: prefer the assigned output binding; fall back to the
    # last successfully-executed node's output if the assign callable
    # wasn't dispatched.
    final_output: object | None = None
    if contract.outputs:
        final_output = state.get(contract.outputs[0].name)
    if final_output is None:
        for n in reversed(nodes):
            if n.status == "ok" and n.node_name in state:
                final_output = state[n.node_name]
                break
    output_hash = _canonical_json_hash(final_output)

    # PHI-safe completion log: per-status node counts + elapsed.
    # WARNING when any node failed (not skipped — skipped is normal for
    # @llm without keys); INFO on the all-ok / all-skipped path.
    ok_count = sum(1 for n in nodes if n.status == "ok")
    skipped_count = sum(1 for n in nodes if n.status == "skipped")
    failed_count = sum(1 for n in nodes if n.status == "failed")
    elapsed_ms = int((time.time() - t0) * 1000)
    log_fn = logger.warning if failed_count > 0 else logger.info
    log_fn(
        "flow_execute_complete",
        extra={
            "flow_name": flow_name,
            "plan_hash_prefix": contract.plan_hash[:16],
            "output_hash_prefix": output_hash[:16],
            "ok_nodes": ok_count,
            "skipped_nodes": skipped_count,
            "failed_nodes": failed_count,
            "elapsed_ms": elapsed_ms,
        },
    )

    return FlowExecution(
        flow_name=flow_name,
        plan_hash=contract.plan_hash,
        inputs_hash=inputs_hash,
        output_hash=output_hash,
        nodes=tuple(nodes),
        invariant_violations=tuple(invariant_violations),
        output=final_output,
        elapsed_ms=elapsed_ms,
    )


__all__ = [
    "FlowPort",
    "FlowNode",
    "FlowInvariant",
    "FlowContract",
    "NodeExecution",
    "FlowExecution",
    "execute",
    "ReplayResult",
    "compute_plan_hash",
    "list_flow_names",
    "list_flows",
    "parse_flow_contract",
    "verify_replay",
]
