"""Pin: arch-mind L1 governance gate has exactly 9 rules and ALL pass.

The arch-mind gate (`scripts/run_arch_mind_gate.py` + the rules at
`docs/arch_mind/clinicalmem_rules.mind`) is the load-bearing
architectural-governance check that the demo, JUDGES.md, and the
reproducibility manifest all reference as "9 / 9 rules pass". A
silent disable (e.g., `arch_rule(...)` decorator removed; floor lowered
to 0) would silently drop coverage without breaking any other test.

These tests pin:
  • exactly 9 rules in docs/arch_mind.json
  • all 9 currently passing
  • the canonical rule metric set (catches deletion or accidental
    rename to a near-miss like `acyclicity_q16` → `acyclicity_q32`)
  • `build_arch_mind_json.py` exits 0 and reports the same "9/9 pass"
    state on a fresh run (catches stale arch_mind.json)
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_ARCH_JSON = _REPO_ROOT / "docs" / "arch_mind.json"
_BUILD_SCRIPT = _REPO_ROOT / "scripts" / "build_arch_mind_json.py"
_GATE_SCRIPT = _REPO_ROOT / "scripts" / "run_arch_mind_gate.py"

# Canonical 9-rule set. Every rule MUST appear in arch_mind.json. A new
# rule being added means deliberate scope expansion -- bump this list
# explicitly (forces the rule author to update the pin AND the docs that
# claim "9 rules pass").
_CANONICAL_RULES = frozenset({
    "acyclicity_q16",
    "redundancy_q16",
    "q16_determinism_purity",
    "equality_q16",
    "depth_q16",
    "modularity_q16",
    "governance_kernel_coverage",
    "mcp_tool_isolation",
    "evidence_chain_density",
})


@pytest.fixture(scope="module")
def arch_mind() -> dict:
    assert _ARCH_JSON.exists(), (
        f"docs/arch_mind.json missing — run "
        f"`python3 {_BUILD_SCRIPT.relative_to(_REPO_ROOT)}` to regenerate"
    )
    return json.loads(_ARCH_JSON.read_text())


def test_rule_count_is_nine(arch_mind):
    assert arch_mind["rule_count"] == 9, (
        f"Demo + JUDGES claim '9 / 9 rules pass'; live rule_count is "
        f"{arch_mind['rule_count']}. Either rule was added without "
        f"updating docs, or a rule was removed silently."
    )
    assert len(arch_mind["rules"]) == 9


def test_all_rules_pass(arch_mind):
    """Demo + JUDGES + reproducibility manifest all assert the gate is
    PASS. A failing rule would be a release-blocking governance signal."""
    assert arch_mind["pass_count"] == 9, (
        f"arch-mind L1 gate degraded: pass_count={arch_mind['pass_count']} "
        f"(must be 9). Failing rule(s):\n" +
        "\n".join(
            f"  - {r['metric']} (floor={r['floor']}, "
            f"current_raw={r.get('current_raw', '?')})"
            for r in arch_mind["rules"]
            if not r.get("passes")
        )
    )
    assert arch_mind["all_pass"] is True


def test_canonical_rule_set_present(arch_mind):
    """Pins exactly which 9 metrics MUST be measured. Adding a new rule
    requires explicit pin update; renaming a rule (e.g., to handle a
    refactor) without updating this set is caught."""
    live_metrics = {r["metric"] for r in arch_mind["rules"]}
    missing = _CANONICAL_RULES - live_metrics
    extra = live_metrics - _CANONICAL_RULES
    assert not missing, (
        f"Canonical arch-mind rules missing from live gate: {sorted(missing)}. "
        f"A rule was deleted or renamed."
    )
    assert not extra, (
        f"Live gate has rules not in the canonical pin set: {sorted(extra)}. "
        f"Update _CANONICAL_RULES in this test if the new rule is "
        f"deliberate, AND update demo/JUDGES claims of 'N rules pass'."
    )


def _arch_mind_binary_present() -> bool:
    """Whether the arch-mind static analyzer binary is reachable.

    arch-mind is a STARGA-internal toolchain artifact that is not
    available in public CI runners. When absent, the gate script
    cannot produce a fresh '9/9 rules pass' verdict — but the
    on-disk arch_mind.json (frozen by the developer who DID have the
    binary) is still valid and consumed by the dashboard.
    """
    import shutil
    if shutil.which("arch-mind"):
        return True
    home_bin = Path.home() / "arch-mind" / "bin" / "arch-mind"
    if home_bin.exists():
        return True
    return False


def test_arch_mind_summary_round_trips_via_build_script():
    """Re-running `build_arch_mind_json.py` against the live state must
    produce the same 9/9 PASS verdict. Catches the case where a
    developer modifies the rules file without re-building the summary.

    Skipped in environments without the arch-mind binary
    (public CI). The on-disk summary still gets independently checked
    by `test_rule_count_is_nine` / `test_all_rules_pass` against
    docs/arch_mind.json which IS shipped.
    """
    import pytest
    if not _arch_mind_binary_present():
        pytest.skip("arch-mind binary not available (STARGA-internal toolchain); on-disk summary still pinned by sibling tests")
    cp = subprocess.run(
        [sys.executable, str(_BUILD_SCRIPT)],
        capture_output=True, text=True, timeout=60, cwd=str(_REPO_ROOT),
    )
    assert cp.returncode == 0, (
        f"build_arch_mind_json.py exited {cp.returncode}.\n"
        f"stdout: {cp.stdout}\nstderr: {cp.stderr}"
    )
    assert "9/9 rules pass" in cp.stdout, (
        f"Expected '9/9 rules pass' in build output; got:\n{cp.stdout}"
    )


def test_run_arch_mind_gate_exits_zero():
    """The gate script `run_arch_mind_gate.py` is referenced as 'PASS'
    in the reproducibility manifest. If it ever exits non-zero, the
    gate is degraded and the manifest claim is misleading.

    Skipped in environments without the arch-mind binary
    (public CI). The on-disk arch_mind.json reflects the last good
    run from a developer environment that DID have the binary.
    """
    import pytest
    if not _arch_mind_binary_present():
        pytest.skip("arch-mind binary not available (STARGA-internal toolchain)")
    cp = subprocess.run(
        [sys.executable, str(_GATE_SCRIPT)],
        capture_output=True, text=True, timeout=60, cwd=str(_REPO_ROOT),
    )
    assert cp.returncode == 0, (
        f"`scripts/run_arch_mind_gate.py` exited {cp.returncode} — "
        f"the demo + JUDGES claim 'arch-mind L1 9/9 PASS' is now "
        f"misleading until the failure is fixed.\n"
        f"stdout:\n{cp.stdout}\nstderr:\n{cp.stderr}"
    )


def test_canonical_rule_count_in_judges_demo_claim():
    """Pin the '9 / 9' claim in JUDGES.md and demo so a future rule
    addition (rule_count → 10) forces both pins + user-facing copy
    to update together. Same drift class as the cohort-count pin."""
    judges = (_REPO_ROOT / "JUDGES.md").read_text()
    demo = (_REPO_ROOT / "docs" / "demo.html").read_text()
    # JUDGES uses "arch-mind 9 / 9 rules" form; demo uses "arch-mind 9 / 9"
    assert "9 / 9" in judges or "9/9" in judges, (
        "JUDGES.md must reference the 9/9 rule pass count somewhere"
    )
    # Iter-149: trust-bar chip relabelled (user removed `arch-mind 9/9 rules`
    # marketing pill; the live count stays in the governance-section H2 body
    # `9/9 pass` which is the authoritative claim). Accept any of the live
    # surface forms so the pin still mechanically enforces "live count must
    # appear on the demo".
    assert (
        "arch-mind 9 / 9" in demo
        or "arch-mind 9/9" in demo
        or '"arch-pass-count"' in demo  # the JS anchor that fills in 9/9 live
    ), (
        "docs/demo.html must reference the arch-mind 9/9 rule-pass count "
        "somewhere — either as a trust-bar chip ('arch-mind 9/9 rules'), "
        "as a JS-filled span (id='arch-pass-count'), or as an inline body "
        "claim. If rule count changes, update demo + JUDGES + this pin "
        "together."
    )
