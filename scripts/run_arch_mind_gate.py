"""arch-mind L1 governance gate (CI-runnable).

Computes the live clinicalmem fixture summary by walking ``engine/*.py``
with the ``ast`` module, runs the ``arch-mind scan`` kernels against
it, and applies the rules profile at
``docs/arch_mind/clinicalmem_rules.mind``.

Exit code:
  0  -- every enforced rule passed
  1  -- at least one rule failed (or the arch-mind binary was not
        found on $PATH / the project default location)

This is the executable counterpart to the manual audit document at
``docs/arch_mind_federation_audit.md`` — running this script reproduces
the per-kernel scan numbers reported there.

Usage:
    python3 scripts/run_arch_mind_gate.py
    python3 scripts/run_arch_mind_gate.py --json
    python3 scripts/run_arch_mind_gate.py --fixture-only   # write fixture, skip scan/rules
    python3 scripts/run_arch_mind_gate.py --skip-scan      # use whatever's already in clinicalmem.scan.json

Apache-2.0 — STARGA, Inc.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_ARCH_MIND_DIR = _REPO_ROOT / "docs" / "arch_mind"
_FIXTURE = _ARCH_MIND_DIR / "clinicalmem.fixture.json"
_SCAN = _ARCH_MIND_DIR / "clinicalmem.scan.json"
_RULES = _ARCH_MIND_DIR / "clinicalmem_rules.mind"
_ENGINE = _REPO_ROOT / "engine"
_MCP_SERVER = _REPO_ROOT / "mcp_server" / "server.py"

# Locations to look for the arch-mind binary.
_ARCH_MIND_CANDIDATES = (
    Path("~/arch-mind/bin/arch-mind"),
    Path.home() / "arch-mind" / "bin" / "arch-mind",
)


# ---------------------------------------------------------------------------
# Fixture computation
# ---------------------------------------------------------------------------

# Anything matching one of these substrings inside an unparsed call's
# func source counts as an "evidence chain" call (audit log / sync /
# fanout / structured logger). Rough but stable.
_EVIDENCE_PATTERNS = (
    "log_sync",
    "fanout.publish",
    "record_publish_event",
    "record_ingest_event",
    "record_quarantine_event",
    "audit_log.append",
    "_build_audit_entry",
    "logger.",
    "log.warning",
    "log.info",
    "log.error",
)


def _compute_engine_summary() -> dict[str, int]:
    """Walk engine/*.py with ast and produce the Phase A summary fields."""
    py_files = sorted(p for p in _ENGINE.rglob("*.py") if "__pycache__" not in str(p))
    n = len(py_files)

    sum_defs = 0
    sum_classes = 0
    sum_decision_points = 0
    sum_evidence_calls = 0
    pure_modules = 0
    per_module_symbols: list[int] = []

    edges: list[tuple[str, str]] = []  # internal package edges
    for src_path in py_files:
        rel = src_path.relative_to(_REPO_ROOT)
        text = src_path.read_text()
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue

        # Top-level side-effect detection
        has_side_effects = False
        for node in tree.body:
            if isinstance(
                node,
                (
                    ast.Import,
                    ast.ImportFrom,
                    ast.FunctionDef,
                    ast.AsyncFunctionDef,
                    ast.ClassDef,
                    ast.Assign,
                    ast.AnnAssign,
                ),
            ):
                continue
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                continue
            has_side_effects = True
        if not has_side_effects:
            pure_modules += 1

        module_symbols = 0
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                sum_defs += 1
                module_symbols += 1
            elif isinstance(node, ast.ClassDef):
                sum_classes += 1
                module_symbols += 1
            elif isinstance(node, (ast.If, ast.For, ast.While, ast.AsyncFor, ast.Try)):
                sum_decision_points += 1

            if isinstance(node, ast.Call):
                try:
                    func_src = ast.unparse(node.func)
                except Exception:
                    func_src = ""
                if any(p in func_src for p in _EVIDENCE_PATTERNS):
                    sum_evidence_calls += 1

            if isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                if mod.startswith(("engine", "tests")):
                    edges.append((str(rel), mod))
            elif isinstance(node, ast.Import):
                for a in node.names:
                    if a.name.startswith(("engine", "tests")):
                        edges.append((str(rel), a.name))
        per_module_symbols.append(module_symbols)

    sum_symbols = sum(per_module_symbols)
    mean = (sum_symbols / n) if n else 0
    sum_abs_symbol_diffs = int(sum(abs(s - mean) for s in per_module_symbols))

    intra_package_edges = sum(
        1
        for s, t in edges
        if t.split(".")[0] == s.split(os.sep)[0]
    )

    # Cycle / longest path: with intra-package only and the engine being
    # very flat in v0.0.1, both are constant. Recompute with DFS instead
    # of trusting last-known constants.
    from collections import defaultdict

    g: dict[str, set[str]] = defaultdict(set)
    for s, t in edges:
        g[s].add(t)
    color: dict[str, int] = defaultdict(int)
    cyclic = 0

    def _dfs(u: str) -> None:
        nonlocal cyclic
        if color[u] == 1:
            cyclic += 1
            return
        if color[u] == 2:
            return
        color[u] = 1
        for v in g.get(u, ()):
            _dfs(v)
        color[u] = 2

    for u in list(g):
        if color[u] == 0:
            _dfs(u)

    memo: dict[str, int] = {}

    def _lp(u: str) -> int:
        if u in memo:
            return memo[u]
        if not g.get(u):
            memo[u] = 0
            return 0
        best = max((_lp(v) for v in g[u]), default=0) + 1
        memo[u] = best
        return best

    longest_path = max((_lp(u) for u in g), default=0)

    # MCP tool count (if mcp_server/server.py exists)
    mcp_tools = 0
    if _MCP_SERVER.exists():
        mcp_tools = _MCP_SERVER.read_text().count("@mcp.tool()")

    return {
        "module_count": n,
        "total_edges": len(edges),
        "total_mcp_tools": mcp_tools,
        "intra_package_edges": intra_package_edges,
        "cyclic_edges": cyclic,
        "longest_path": longest_path,
        "sum_symbols": sum_symbols,
        "sum_abs_symbol_diffs": sum_abs_symbol_diffs,
        "redundancy_excess": 0,
        "pure_modules": pure_modules,
        "sum_decision_points": sum_decision_points,
        "sum_evidence_calls": sum_evidence_calls,
        "max_mcp_tool_overlap": 0,
        "sum_protected_decls": 0,
    }


def _write_fixture(summary: dict[str, int]) -> None:
    payload = {
        "_comment": (
            "Auto-generated by scripts/run_arch_mind_gate.py from a fresh "
            "ast walk of engine/*.py. Do not hand-edit; regenerate via the "
            "script. Phase D parser, when shipped, will replace this "
            "computation with a richer per-file rollup."
        ),
        "_aggregated_for_phase_a": summary,
        "nodes": [],
        "edges": [],
    }
    _ARCH_MIND_DIR.mkdir(parents=True, exist_ok=True)
    _FIXTURE.write_text(json.dumps(payload, indent=2) + "\n")


# ---------------------------------------------------------------------------
# arch-mind invocation
# ---------------------------------------------------------------------------


def _resolve_arch_mind() -> Path | None:
    on_path = shutil.which("arch-mind")
    if on_path:
        return Path(on_path)
    for cand in _ARCH_MIND_CANDIDATES:
        if cand.exists() and os.access(cand, os.X_OK):
            return cand
    return None


def _run_scan(arch_mind: Path) -> dict | None:
    cp = subprocess.run(
        [str(arch_mind), "scan", "--fixture", str(_FIXTURE), "--out", str(_SCAN)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if cp.returncode != 0:
        sys.stderr.write(cp.stderr)
        return None
    return json.loads(_SCAN.read_text())


def _run_rules(arch_mind: Path) -> tuple[bool, str]:
    cp = subprocess.run(
        [
            str(arch_mind),
            "rules",
            "--rules",
            str(_RULES),
            "--scan",
            str(_SCAN),
            "--mode",
            "enforce",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    return cp.returncode == 0, (cp.stdout + cp.stderr).strip()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    ap.add_argument(
        "--fixture-only",
        action="store_true",
        help="write the fixture, skip scan/rules",
    )
    ap.add_argument(
        "--skip-scan",
        action="store_true",
        help="reuse existing scan output (skip fixture regen + scan)",
    )
    args = ap.parse_args()

    if not args.skip_scan:
        summary = _compute_engine_summary()
        _write_fixture(summary)
        if args.fixture_only:
            if args.json:
                print(json.dumps({"fixture_written": str(_FIXTURE), "summary": summary}, indent=2))
            else:
                print(f"wrote {_FIXTURE}")
                for k, v in summary.items():
                    print(f"  {k:24} {v}")
            return 0

    arch_mind = _resolve_arch_mind()
    if arch_mind is None:
        msg = "arch-mind binary not found on $PATH or default candidates"
        if args.json:
            print(json.dumps({"ok": False, "error": msg}))
        else:
            sys.stderr.write(f"ERROR: {msg}\n")
        return 1

    scan = None
    if not args.skip_scan:
        scan = _run_scan(arch_mind)
        if scan is None:
            return 1

    ok, output = _run_rules(arch_mind)

    if args.json:
        report = {
            "ok": ok,
            "scan": scan or json.loads(_SCAN.read_text()) if _SCAN.exists() else None,
            "rules_output": output,
        }
        print(json.dumps(report, indent=2))
    else:
        print(output)
        print()
        print("PASS" if ok else "FAIL")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
