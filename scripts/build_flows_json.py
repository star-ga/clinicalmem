#!/usr/bin/env python3
"""Regenerate docs/flows.json from the .flow.mind sources in flows/.

The dashboard at clinicalmem-demo.pages.dev consumes docs/flows.json to
render the Verifiable Clinical AI section. Run this script after editing
any flow source so the dashboard's plan_hash / Verify Replay tiles stay
in sync with the canonical contracts.

Usage:
    python3 scripts/build_flows_json.py        # default paths
    python3 scripts/build_flows_json.py --check # CI mode: assert in-sync

Apache-2.0 © 2026 STARGA, Inc.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from engine.flow_runner import list_flows  # noqa: E402

OUTPUT_PATH = REPO_ROOT / "docs" / "flows.json"


def build_payload() -> list[dict[str, object]]:
    payload: list[dict[str, object]] = []
    for contract in list_flows():
        source = contract.flow_path.read_text(encoding="utf-8")
        payload.append({
            "name":            contract.name,
            "plan_hash":       contract.plan_hash,
            "profile":         contract.profile,
            "kernel":          contract.kernel,
            "inputs":          [{"name": p.name, "type": p.type_expr} for p in contract.inputs],
            "outputs":         [{"name": p.name, "type": p.type_expr} for p in contract.outputs],
            "nodes":           [{"name": n.name, "directive": n.directive} for n in contract.nodes],
            "invariant_count": len(contract.invariants),
            "invariants":      [inv.predicate for inv in contract.invariants],
            "source_path":     str(contract.flow_path.relative_to(REPO_ROOT)),
            "source":          source,
        })
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit non-zero if docs/flows.json is out of sync with flows/*.flow.mind",
    )
    args = parser.parse_args()

    fresh = build_payload()
    fresh_text = json.dumps(fresh, indent=2) + "\n"

    if args.check:
        if not OUTPUT_PATH.exists():
            print(f"error: {OUTPUT_PATH} missing; run scripts/build_flows_json.py", file=sys.stderr)
            return 1
        current = OUTPUT_PATH.read_text(encoding="utf-8")
        if current != fresh_text:
            print(
                f"error: {OUTPUT_PATH} is out of sync with flows/*.flow.mind; "
                "run scripts/build_flows_json.py and commit the diff",
                file=sys.stderr,
            )
            return 1
        print(f"ok: {OUTPUT_PATH} in sync ({len(fresh)} flows)")
        return 0

    OUTPUT_PATH.write_text(fresh_text, encoding="utf-8")
    print(f"wrote {OUTPUT_PATH} ({len(fresh)} flows, {len(fresh_text)} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
