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
import logging
import os
import re
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
    return hashlib.sha256(canonical).hexdigest()


def _resolve_flow_path(flow_name: str, flows_dir: Path | None = None) -> Path:
    """Resolve a flow name to its `.flow.mind` source path."""
    base = Path(flows_dir) if flows_dir is not None else _FLOWS_DIR
    # Accept "MedicationSafetyReview" or "MedicationSafetyReview.flow.mind"
    if not flow_name.endswith(".flow.mind"):
        flow_name = flow_name + ".flow.mind"
    candidate = base / flow_name
    if not candidate.exists():
        raise FileNotFoundError(
            f"flow source not found: {candidate}; available flows: {list_flow_names(base)}"
        )
    return candidate


# ─── Catalogue ─────────────────────────────────────────────────────────────

def list_flow_names(flows_dir: Path | None = None) -> list[str]:
    """List available flow names (without `.flow.mind` suffix)."""
    base = Path(flows_dir) if flows_dir is not None else _FLOWS_DIR
    if not base.exists():
        return []
    return sorted(p.stem.replace(".flow", "") for p in base.glob("*.flow.mind"))


def list_flows(flows_dir: Path | None = None) -> list[FlowContract]:
    """Parse and return every `.flow.mind` contract under `flows/`."""
    return [
        parse_flow_contract(name, flows_dir=flows_dir)
        for name in list_flow_names(flows_dir)
    ]


# ─── Static contract extraction ────────────────────────────────────────────

_FLOW_HEADER_RE = re.compile(r"^\s*flow\s+(\w+)\s*\{\s*$")
_INPUT_RE = re.compile(r"^\s*input\s+(\w+)\s*:\s*(.+?)\s*$")
_OUTPUT_RE = re.compile(r"^\s*output\s+(\w+)\s*:\s*(.+?)\s*$")
_NODE_RE = re.compile(r"^\s*node\s+(\w+)\s*=\s*(@\w+)\s*(.+?)\s*$")
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
        if (m := _INVARIANT_RE.match(line)) is not None:
            invariants.append(FlowInvariant(predicate=m.group(1)))
            continue

    if flow_decl_name is None:
        raise ValueError(f"no `flow Name {{ … }}` declaration found in {flow_path}")

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
    if not matches:
        logger.warning(
            "replay mismatch for flow %s: expected=%s actual=%s",
            flow_name,
            expected_hash,
            contract.plan_hash,
        )
    return ReplayResult(
        flow_name=flow_name,
        expected_hash=expected_hash,
        actual_hash=contract.plan_hash,
        matches=matches,
        contract=contract,
    )


__all__ = [
    "FlowPort",
    "FlowNode",
    "FlowInvariant",
    "FlowContract",
    "ReplayResult",
    "compute_plan_hash",
    "list_flow_names",
    "list_flows",
    "parse_flow_contract",
    "verify_replay",
]
