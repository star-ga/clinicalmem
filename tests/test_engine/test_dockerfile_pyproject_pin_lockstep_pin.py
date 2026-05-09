"""iter-446 40th cross-pin family — Dockerfile ↔ pyproject deployment-pin lockstep.

Background
==========
iter-438 fix: `Dockerfile.mcp` had `mind-mem>=1.9.0` while `pyproject.toml`
declared `mind-mem>=3.12.0`. The Docker build was silently producing
images compatible with the floor of an obsolete `mind-mem` API surface
(v1.9.x — pre-v2 era) while local dev + the published wheel resolved
`>=3.12.0`. Docker silently shipped a binary stuck 9 minor versions
behind. iter-438 fixed `Dockerfile.mcp`; iter-439 then caught a related
drift in `Dockerfile.a2a` where `a2a-sdk>=0.2` would resolve to 1.0.x
which dropped the `In` symbol our `a2a_agent/app.py:25` imports — the
fix pinned `a2a-sdk==0.3.24` in `Dockerfile.a2a` until upstream
`google-adk` migrates to `a2a-sdk` 1.0+ (still not done as of
iter-446 — `google-adk 1.33.0` imports `from a2a.server.apps import
A2AStarletteApplication` which doesn't exist in `a2a-sdk` 1.0+).

This pin file is the **structural lock** that catches the next
iteration of this drift class **at commit time** rather than at
deployment time. The check:

  For every dependency in `pyproject.toml` that is ALSO referenced in
  either `Dockerfile.mcp` or `Dockerfile.a2a`:
    * The Dockerfile constraint MUST match the pyproject constraint
    * UNLESS the (package, dockerfile) pair is in `_KNOWN_DIVERGENCES`
      with a documented reason

The known-divergence allowlist forces every intentional drift to be
named, dated, and reasoned — preventing silent drift from looking the
same as documented divergence.

Same iter-228 / iter-244 / iter-260 / iter-321 / iter-345 pattern
(structural pin lock with explicit-allowlist for known exceptions).
"""

from __future__ import annotations

import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent

_PYPROJECT = _REPO_ROOT / "pyproject.toml"
_DOCKERFILE_MCP = _REPO_ROOT / "Dockerfile.mcp"
_DOCKERFILE_A2A = _REPO_ROOT / "Dockerfile.a2a"

# Documented intentional divergences. Every entry MUST cite the
# upstream blocker + iter that introduced the divergence + when
# the divergence is expected to resolve.
#
# Tuple shape: (package_name, dockerfile_short, dockerfile_constraint,
#               pyproject_constraint, reason)
_KNOWN_DIVERGENCES = (
    (
        "a2a-sdk",
        "Dockerfile.a2a",
        "==0.3.24",
        ">=0.2",
        # iter-439 fix: a2a-sdk 1.0.x dropped the `In` symbol that
        # `a2a_agent/app.py:25` imports for APIKeySecurityScheme(in_=...).
        # google-adk 1.33.0 still imports `from a2a.server.apps import
        # A2AStarletteApplication` which doesn't exist in a2a-sdk 1.0+,
        # so the upstream blocker is google-adk, not us. The Dockerfile
        # pins ==0.3.24 to keep deployment working; pyproject keeps the
        # >=0.2 floor so dev + CI install whatever resolves (currently
        # 0.3.24). When google-adk migrates to a2a-sdk 1.0+, both pins
        # bump together and this entry deletes.
        "google-adk 1.33.0 not yet migrated to a2a-sdk 1.0+; "
        "a2a-sdk 1.0 dropped the `In` symbol our app.py imports",
    ),
)

_PIP_INSTALL_LINE_RE = re.compile(
    r'^\s*"([^"=<>\s]+)([^"]*)"\s*\\?\s*$'
)


def _parse_pyproject_deps() -> dict[str, str]:
    """Return {package_name: constraint_string} for the [project]
    dependencies block. Constraint includes the operator
    (e.g. `>=3.12.0`). Comments are stripped."""
    text = _PYPROJECT.read_text()
    deps: dict[str, str] = {}
    in_deps = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped == "dependencies = [":
            in_deps = True
            continue
        if in_deps and stripped == "]":
            break
        if not in_deps:
            continue
        # Match `    "package>=X.Y.Z",  # comment`
        m = re.match(
            r'^\s*"([^"=<>\s]+)([^"]*)"\s*,?\s*(?:#.*)?$',
            line,
        )
        if m:
            deps[m.group(1)] = m.group(2)
    return deps


def _parse_dockerfile_pip_install(path: Path) -> dict[str, str]:
    """Return {package_name: constraint_string} for the
    `RUN pip install --no-cache-dir \\` block in the Dockerfile.
    Constraint includes the operator."""
    text = path.read_text()
    deps: dict[str, str] = {}
    in_pip_block = False
    for line in text.splitlines():
        if "RUN pip install" in line:
            in_pip_block = True
            continue
        if not in_pip_block:
            continue
        # Empty / next directive ends the block
        stripped_line = line.strip()
        if not stripped_line or stripped_line.upper().startswith(
            ("RUN ", "WORKDIR ", "COPY ", "CMD ", "ENV ", "ARG ", "FROM ", "EXPOSE ", "USER ")
        ):
            in_pip_block = False
            continue
        m = _PIP_INSTALL_LINE_RE.match(line)
        if m:
            deps[m.group(1)] = m.group(2)
    return deps


# ── Test 1: pyproject parser sanity ───────────────────────────────────────────

def test_pyproject_parser_returns_known_packages():
    """The parser must recognise the 7 known top-level dependencies
    in pyproject.toml. If a package vanishes, the deployment-coverage
    pin (test 4) will misreport coverage."""
    deps = _parse_pyproject_deps()
    expected_keys = {
        "mind-mem", "httpx", "fastmcp", "google-adk",
        "a2a-sdk", "uvicorn", "starlette",
    }
    missing = expected_keys - set(deps.keys())
    assert not missing, (
        f"pyproject parser missed expected dependencies: {sorted(missing)}. "
        f"Either the parser regression-broke or pyproject was edited "
        f"to remove a load-bearing top-level dep."
    )


# ── Test 2: Dockerfile.mcp parser sanity ──────────────────────────────────────

def test_dockerfile_mcp_parser_returns_known_packages():
    """`Dockerfile.mcp` must declare its known pip-install set. If a
    package is silently dropped the deployment ships without it."""
    deps = _parse_dockerfile_pip_install(_DOCKERFILE_MCP)
    expected = {"mind-mem", "httpx", "fastmcp", "uvicorn", "starlette"}
    missing = expected - set(deps.keys())
    assert not missing, (
        f"Dockerfile.mcp dropped pinned dependencies: {sorted(missing)}. "
        f"This would silently ship a deployment image missing those "
        f"libraries (manifest-of-truth drift class)."
    )


# ── Test 3: Dockerfile.a2a parser sanity ──────────────────────────────────────

def test_dockerfile_a2a_parser_returns_known_packages():
    """`Dockerfile.a2a` must declare its known pip-install set."""
    deps = _parse_dockerfile_pip_install(_DOCKERFILE_A2A)
    expected = {
        "mind-mem", "httpx", "google-adk", "a2a-sdk",
        "uvicorn", "starlette",
    }
    missing = expected - set(deps.keys())
    assert not missing, (
        f"Dockerfile.a2a dropped pinned dependencies: {sorted(missing)}. "
        f"This would silently ship a deployment image missing those "
        f"libraries (manifest-of-truth drift class)."
    )


# ── Test 4: lockstep on Dockerfile.mcp ↔ pyproject ────────────────────────────

def test_dockerfile_mcp_constraints_match_pyproject():
    """Every package referenced in `Dockerfile.mcp` MUST have the
    EXACT same version constraint as `pyproject.toml`, unless the
    pair is in `_KNOWN_DIVERGENCES`. Catches the iter-438 drift class
    where Dockerfile.mcp had `mind-mem>=1.9.0` silently while pyproject
    declared `>=3.12.0`."""
    pp = _parse_pyproject_deps()
    df = _parse_dockerfile_pip_install(_DOCKERFILE_MCP)
    divergences_allowlist = {
        (pkg, dfile): (df_c, pp_c, reason)
        for pkg, dfile, df_c, pp_c, reason in _KNOWN_DIVERGENCES
    }
    drifts = []
    for pkg, df_constraint in df.items():
        if pkg not in pp:
            drifts.append(
                f"  {pkg!r}: in Dockerfile.mcp ({df_constraint!r}) but "
                f"NOT in pyproject — Dockerfile would install a package "
                f"that isn't declared as a project dependency"
            )
            continue
        pp_constraint = pp[pkg]
        if df_constraint == pp_constraint:
            continue
        # Check known-divergence allowlist
        allowed = divergences_allowlist.get((pkg, "Dockerfile.mcp"))
        if allowed and allowed[0] == df_constraint and allowed[1] == pp_constraint:
            continue
        drifts.append(
            f"  {pkg!r}: Dockerfile.mcp={df_constraint!r}, "
            f"pyproject={pp_constraint!r}"
        )
    assert not drifts, (
        "Dockerfile.mcp ↔ pyproject deployment-pin DRIFT detected:\n"
        + "\n".join(drifts)
        + "\n\nFix: either bump the lagging side, or add a "
        + "_KNOWN_DIVERGENCES entry citing the upstream blocker. "
        + "Same iter-438 single-source-of-truth → derived-surface drift "
        + "class, applied at the deployment-pin layer."
    )


# ── Test 5: lockstep on Dockerfile.a2a ↔ pyproject ────────────────────────────

def test_dockerfile_a2a_constraints_match_pyproject():
    """Every package referenced in `Dockerfile.a2a` MUST match
    pyproject — same shape as test 4 — unless in `_KNOWN_DIVERGENCES`.
    The current known divergence is `a2a-sdk` (==0.3.24 vs >=0.2) per
    iter-439 fix; reason documented in `_KNOWN_DIVERGENCES`."""
    pp = _parse_pyproject_deps()
    df = _parse_dockerfile_pip_install(_DOCKERFILE_A2A)
    divergences_allowlist = {
        (pkg, dfile): (df_c, pp_c, reason)
        for pkg, dfile, df_c, pp_c, reason in _KNOWN_DIVERGENCES
    }
    drifts = []
    for pkg, df_constraint in df.items():
        if pkg not in pp:
            drifts.append(
                f"  {pkg!r}: in Dockerfile.a2a ({df_constraint!r}) but "
                f"NOT in pyproject"
            )
            continue
        pp_constraint = pp[pkg]
        if df_constraint == pp_constraint:
            continue
        allowed = divergences_allowlist.get((pkg, "Dockerfile.a2a"))
        if allowed and allowed[0] == df_constraint and allowed[1] == pp_constraint:
            continue
        drifts.append(
            f"  {pkg!r}: Dockerfile.a2a={df_constraint!r}, "
            f"pyproject={pp_constraint!r}"
        )
    assert not drifts, (
        "Dockerfile.a2a ↔ pyproject deployment-pin DRIFT detected:\n"
        + "\n".join(drifts)
        + "\n\nFix: either bump the lagging side, or add a "
        + "_KNOWN_DIVERGENCES entry citing the upstream blocker."
    )


# ── Test 6: every known divergence has a documented reason ────────────────────

def test_known_divergences_carry_documented_reasons():
    """The divergence allowlist is a load-bearing primitive — each
    entry MUST cite WHY the divergence exists + when it's expected
    to resolve. Empty / placeholder reasons fail the pin."""
    for pkg, dfile, df_c, pp_c, reason in _KNOWN_DIVERGENCES:
        assert pkg, "package name empty"
        assert dfile, f"dockerfile name empty for {pkg}"
        assert df_c, f"dockerfile constraint empty for {pkg}"
        assert pp_c, f"pyproject constraint empty for {pkg}"
        assert reason and len(reason.strip()) >= 30, (
            f"divergence reason for {pkg!r} too short or empty: {reason!r}. "
            f"Every documented divergence must cite the upstream blocker "
            f"+ when it's expected to resolve (≥30 chars enforces real prose)."
        )


# ── Test 7: pyproject coverage of Dockerfile-only packages ────────────────────

def test_every_dockerfile_pkg_appears_in_pyproject():
    """Closes the inverse drift class: a Dockerfile pinning a package
    that pyproject doesn't declare as a project dependency. Such a
    package would only be installed in the Docker image and silently
    missing from `pip install -e .` → tests pass locally but production
    fails at import time."""
    pp = set(_parse_pyproject_deps().keys())
    for label, df_path in (
        ("Dockerfile.mcp", _DOCKERFILE_MCP),
        ("Dockerfile.a2a", _DOCKERFILE_A2A),
    ):
        df = set(_parse_dockerfile_pip_install(df_path).keys())
        missing_from_pyproject = df - pp
        assert not missing_from_pyproject, (
            f"{label} pins packages NOT declared in pyproject "
            f"dependencies: {sorted(missing_from_pyproject)}. "
            f"Either add them to pyproject.dependencies or remove "
            f"them from {label} (deployment-only deps would diverge "
            f"from local dev environment silently)."
        )
