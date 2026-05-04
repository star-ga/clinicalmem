"""All-gates eval driver — runs every deterministic verification in one shot.

The five gates collapsed into one command:

  1. PCCP recall gate          — scripts/run_clinical_regression_eval.py
  2. Negative-control precision — scripts/run_negative_control_eval.py
  3. Federation 16-invariant   — scripts/federation_mock_demo.py
  4. arch-mind L1 governance   — scripts/run_arch_mind_gate.py
  5. Audit-replay verifier     — scripts/verify_audit_replay.py --check

Each gate is invoked as a subprocess so the all-gates result is exactly
what a reviewer sees if they run the five commands by hand. Skipped
gates (arch-mind binary absent) are reported as `skipped`, not `fail`.

Exit code:
  0  -- every present gate passed
  1  -- at least one gate failed
  2  -- all gates skipped (no audit was actually run)

Usage:
    python3 scripts/run_all_gates.py
    python3 scripts/run_all_gates.py --json     # machine-readable
    python3 scripts/run_all_gates.py --skip-federation  # skip the demo

Apache-2.0 — STARGA, Inc.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS = _REPO_ROOT / "scripts"


@dataclass
class GateResult:
    name: str
    status: str  # "pass" | "fail" | "skipped"
    duration_ms: float
    summary: str  # one-line excerpt from the gate's stdout


def _run(name: str, cmd: list[str], skip_reason: str | None = None) -> GateResult:
    if skip_reason is not None:
        return GateResult(name=name, status="skipped", duration_ms=0.0, summary=skip_reason)

    t0 = time.perf_counter()
    cp = subprocess.run(cmd, capture_output=True, text=True, timeout=120, cwd=str(_REPO_ROOT))
    dt = (time.perf_counter() - t0) * 1000.0

    # Pull the most informative one-liner from each gate's stdout.
    stdout_lines = (cp.stdout or "").strip().splitlines()
    summary = ""
    for line in reversed(stdout_lines):
        line = line.strip()
        if not line:
            continue
        if any(
            tok in line
            for tok in (
                "PCCP GATE", "PRECISION GATE", "FEDERATION DEMO COMPLETE",
                "OK: every rule passed", "DEMO FAILED", "FAIL",
                "AUDIT REPLAY", "byte-for-byte", "audit-replay",
            )
        ):
            summary = line
            break
    if not summary and stdout_lines:
        summary = stdout_lines[-1].strip()
    if not summary:
        summary = "(no output)"

    return GateResult(
        name=name,
        status="pass" if cp.returncode == 0 else "fail",
        duration_ms=round(dt, 1),
        summary=summary,
    )


def _arch_mind_available() -> bool:
    if shutil.which("arch-mind"):
        return True
    return Path("~/arch-mind/bin/arch-mind").exists()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    ap.add_argument("--skip-federation", action="store_true", help="skip the federation mock demo")
    ap.add_argument("--skip-arch-mind", action="store_true", help="skip the arch-mind gate")
    args = ap.parse_args()

    gates: list[GateResult] = []

    gates.append(
        _run(
            "PCCP recall",
            [sys.executable, str(_SCRIPTS / "run_clinical_regression_eval.py")],
        )
    )
    gates.append(
        _run(
            "Negative-control precision",
            [sys.executable, str(_SCRIPTS / "run_negative_control_eval.py")],
        )
    )
    gates.append(
        _run(
            "Federation 16-invariant demo",
            [sys.executable, str(_SCRIPTS / "federation_mock_demo.py")],
            skip_reason="--skip-federation passed" if args.skip_federation else None,
        )
    )
    gates.append(
        _run(
            "arch-mind L1 governance",
            [sys.executable, str(_SCRIPTS / "run_arch_mind_gate.py")],
            skip_reason=(
                "--skip-arch-mind passed" if args.skip_arch_mind else
                "arch-mind binary not installed" if not _arch_mind_available() else None
            ),
        )
    )
    gates.append(
        _run(
            "Audit-replay verifier",
            [sys.executable, str(_SCRIPTS / "verify_audit_replay.py"), "--check"],
        )
    )

    pass_count = sum(1 for g in gates if g.status == "pass")
    fail_count = sum(1 for g in gates if g.status == "fail")
    skip_count = sum(1 for g in gates if g.status == "skipped")
    total_dt = sum(g.duration_ms for g in gates)

    if args.json:
        report = {
            "ok": fail_count == 0 and pass_count > 0,
            "pass": pass_count,
            "fail": fail_count,
            "skipped": skip_count,
            "total_duration_ms": round(total_dt, 1),
            "gates": [g.__dict__ for g in gates],
        }
        print(json.dumps(report, indent=2))
    else:
        print("=" * 78)
        print("ClinicalMem — All Gates")
        print("=" * 78)
        for g in gates:
            badge = {"pass": "  PASS  ", "fail": "  FAIL  ", "skipped": "  skip  "}[g.status]
            print(f"  [{badge}] {g.name:32}  {g.duration_ms:8.1f} ms  {g.summary}")
        print("-" * 78)
        print(
            f"  {pass_count} pass · {fail_count} fail · {skip_count} skipped  "
            f"(total {total_dt/1000:.1f} s)"
        )
        if fail_count == 0 and pass_count > 0:
            print("=" * 78)
            print("  ALL-GATES RESULT: PASS")
            print("=" * 78)
        elif fail_count > 0:
            print("=" * 78)
            print(f"  ALL-GATES RESULT: FAIL ({fail_count} gate(s) failed)")
            print("=" * 78)
        else:
            print("=" * 78)
            print("  ALL-GATES RESULT: skipped (no gate executed)")
            print("=" * 78)

    if fail_count > 0:
        return 1
    if pass_count == 0:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
