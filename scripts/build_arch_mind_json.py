#!/usr/bin/env python3
"""Regenerate docs/arch_mind.json from .arch-mind/{rules,last_summary}.

Builds the data file the dashboard's "Architectural Governance" tile
consumes. Each rule is rendered as a row with current value, floor,
operator, and pass/fail status. The scan-hash is the SHA-256 of the
canonical scan-summary bytes — judges (and regulators) re-verify
client-side via the same Web Crypto API path the flow Verify Replay
buttons use.

Usage:
    python3 scripts/build_arch_mind_json.py        # default paths
    python3 scripts/build_arch_mind_json.py --check # CI sync gate

Apache-2.0 © 2026 STARGA, Inc.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RULES_PATH = REPO_ROOT / ".arch-mind" / "rules.mind"
SUMMARY_PATH = REPO_ROOT / ".arch-mind" / "last_summary.json"
OUTPUT_PATH = REPO_ROOT / "docs" / "arch_mind.json"

# Path to the arch-mind binary; falls back to "arch-mind" on PATH.
ARCH_MIND_BIN = Path("~/arch-mind/bin/arch-mind")


_RULE_RE = re.compile(
    r"\[arch_rule\((\w+),\s*(\w+)\)\]\s*\n\s*const\s+(\w+):\s*i32\s*=\s*(-?\d+)"
)

# Friendly per-metric labels + descriptions (mirror docs/why_*.md framing)
_METRIC_LABELS: dict[str, dict[str, str]] = {
    "acyclicity_q16": {
        "label": "Acyclicity",
        "blurb": "Engine→consensus→synthesis must remain a DAG; a cycle would break the audit-chain replay contract.",
    },
    "redundancy_q16": {
        "label": "Redundancy",
        "blurb": "Bounded duplicate-pattern density across the engine surface.",
    },
    "q16_determinism_purity": {
        "label": "Determinism Purity",
        "blurb": "Q16.16 fixed-point share of the engine surface; load-bearing for the BitNet b1.58 reproducibility layer.",
    },
    "equality_q16": {
        "label": "Equality",
        "blurb": "Symbol-count balance across modules; flags concentration drift.",
    },
    "depth_q16": {
        "label": "Depth",
        "blurb": "Pipeline depth bounded; no over-deep dispatch chains.",
    },
    "modularity_q16": {
        "label": "Modularity",
        "blurb": "engine/ + mcp_server/ + a2a_agent/ inter-package separation.",
    },
    "governance_kernel_coverage": {
        "label": "Governance Coverage",
        "blurb": "[protection]/[invariant] decoration count (MIND-side; Python repo today).",
    },
    "mcp_tool_isolation": {
        "label": "MCP Tool Isolation",
        "blurb": "One-tool-per-public-surface contract across 18 SHARP-on-MCP + 13 A2A skills.",
    },
    "evidence_chain_density": {
        "label": "Evidence Chain Density",
        "blurb": "Audit-chain emission density per module; current source: TAG_v1 in engine/clinical_memory.py.",
    },
}


def _parse_rules(rules_path: Path) -> list[dict[str, object]]:
    text = rules_path.read_text(encoding="utf-8")
    rules: list[dict[str, object]] = []
    for match in _RULE_RE.finditer(text):
        metric, op, const_name, floor = match.groups()
        rules.append({
            "metric": metric,
            "operator": op,             # "eq" or "ge"
            "const_name": const_name,
            "floor": int(floor),
        })
    return rules


def _scan_metrics_for(summary_path: Path) -> dict[str, int]:
    """Run `arch-mind scan --fixture <summary>` and return the q16 metrics."""
    if not ARCH_MIND_BIN.exists():
        # Fallback: scan binary not on disk; emit empty metrics
        return {}
    result = subprocess.run(
        [str(ARCH_MIND_BIN), "scan", "--fixture", str(summary_path)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        sys.stderr.write(f"warning: arch-mind scan failed: {result.stderr}\n")
        return {}
    payload = json.loads(result.stdout)
    return {k: v for k, v in payload.items() if isinstance(v, int) and not k.startswith("_")}


def _evaluate_rule(rule: dict[str, object], current_q16: int) -> tuple[bool, int]:
    """Return (passes, current_raw)."""
    raw = current_q16 // 65536  # Q16.16 -> integer
    if rule["operator"] == "eq":
        passes = (raw == rule["floor"])
    else:  # ge
        passes = (raw >= rule["floor"])
    return passes, raw


def build_payload() -> dict[str, object]:
    rules = _parse_rules(RULES_PATH)
    metrics = _scan_metrics_for(SUMMARY_PATH)

    rows: list[dict[str, object]] = []
    pass_count = 0
    for rule in rules:
        metric = str(rule["metric"])
        current_q16 = metrics.get(metric, 0)
        passes, current_raw = _evaluate_rule(rule, current_q16)
        if passes:
            pass_count += 1
        rows.append({
            **rule,
            "current_q16": current_q16,
            "current_raw": current_raw,
            "passes": passes,
            "label": _METRIC_LABELS.get(metric, {}).get("label", metric),
            "blurb": _METRIC_LABELS.get(metric, {}).get("blurb", ""),
        })

    summary_bytes = SUMMARY_PATH.read_bytes() if SUMMARY_PATH.exists() else b""
    rules_bytes = RULES_PATH.read_bytes() if RULES_PATH.exists() else b""

    return {
        "rules": rows,
        "rule_count": len(rows),
        "pass_count": pass_count,
        "all_pass": pass_count == len(rows) and len(rows) > 0,
        "summary_sha256": hashlib.sha256(summary_bytes).hexdigest(),
        "rules_sha256": hashlib.sha256(rules_bytes).hexdigest(),
        "summary_path": str(SUMMARY_PATH.relative_to(REPO_ROOT)),
        "rules_path": str(RULES_PATH.relative_to(REPO_ROOT)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit non-zero if docs/arch_mind.json is out of sync",
    )
    args = parser.parse_args()

    fresh = build_payload()
    fresh_text = json.dumps(fresh, indent=2) + "\n"

    if args.check:
        if not OUTPUT_PATH.exists():
            print(f"error: {OUTPUT_PATH} missing; run scripts/build_arch_mind_json.py", file=sys.stderr)
            return 1
        current = OUTPUT_PATH.read_text(encoding="utf-8")
        if current != fresh_text:
            print(f"error: {OUTPUT_PATH} out of sync; run scripts/build_arch_mind_json.py", file=sys.stderr)
            return 1
        print(f"ok: {OUTPUT_PATH} in sync ({fresh['pass_count']}/{fresh['rule_count']} rules pass)")
        return 0

    OUTPUT_PATH.write_text(fresh_text, encoding="utf-8")
    print(f"wrote {OUTPUT_PATH} ({fresh['pass_count']}/{fresh['rule_count']} rules pass)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
