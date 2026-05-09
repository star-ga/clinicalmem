"""Pin: present-tense 'mind-mem v<X.Y.Z> is shipped and pinned' / 'IS
shipped and pinned' / 'ships no new <module>' narrative claims MUST
match the pyproject.toml pin.

Iter 390 (round 22 T1) — 35th cross-pin family.
==============================================

iter-388 caught 6 stale narrative-drift surfaces in 5 user-facing
regulator/judge docs (clinical_validation.md L184, arch_mind_federation_
audit.md L45 + L55, JUDGES.md L219, architecture.md L170, fda_q_sub_
draft.md L421) that referenced `mind-mem v3.10.0 is shipped and pinned`
or `v3.10.0 ships no new federation-transport module` even though the
live pyproject pin had advanced to v3.10.6 (iter-342) and then v3.10.8
(iter-387).

The existing `test_no_stale_mind_mem_dep_version_as_live_claim` pin
(iter-321/339/345 in `test_user_facing_docs_v8_consistency_pin.py`)
covers the same drift class at the literal version-number layer but
its `extended_tokens` allowlist includes `"released 2026-"`, which
was too broad: any prose that mentioned a release date in a 3-line
window was skipped, even when the prose was simultaneously claiming
the version was the *currently shipped + pinned* one.

This pin closes that gap by scanning for SEMANTIC present-tense
shipping claims directly:
  - `mind-mem v<X.Y.Z> is shipped and pinned`
  - `MIND-Mem v<X.Y.Z> IS shipped and pinned`
  - `v<X.Y.Z> ships no new <module> module`
  - `v<X.Y.Z> is the live release`
  - `v<X.Y.Z> is currently shipped`
  - `mind-mem v<X.Y.Z> is the current pin`

These are claims about *the present state of the live pin*, not
historical statements about a past release. Pin extracts the version,
compares to the pyproject pin, and fails if cited < pin. Multi-version
range framings ("the v3.10.x line through v3.10.8") are allowed via a
narrow allowlist token (`through v<patch>`).

Forward protection: when iter-X+1 advances the pin (e.g. to v3.10.9),
this pin catches every `is shipped and pinned` claim still frozen at
v3.10.8, surfacing the iter-388 drift class at commit time instead of
post-cascade audit.

Same iter-232 / iter-298 / iter-301 / iter-303 / iter-306 / iter-308 /
iter-313 / iter-316 / iter-318 / iter-321 / iter-339 / iter-345 / iter-
388 single-source-of-truth -> derived-surface drift class, applied at
the present-tense narrative claim layer (the layer the iter-321/339/345
literal-version pin tolerates via its release-date window).

35th cross-pin family in the discipline lineage (iter-178/183/188/193/
198/203/223/228/232/234/236/239/240/244/246/247/255+265/260/279/281/
285/286/295/296/301/304/306/311/314/319/324/329/366/374/390).
"""
from __future__ import annotations

import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_PYPROJECT = _REPO_ROOT / "pyproject.toml"

# User-facing regulator / judge / pitch surfaces. Same set the iter-321
# dep-version pin scans, plus all docs/*.md and the demo.html.
_USER_FACING_DOCS: tuple[Path, ...] = (
    _REPO_ROOT / "JUDGES.md",
    _REPO_ROOT / "README.md",
    _REPO_ROOT / "DEVPOST.md",
    _REPO_ROOT / "docs" / "demo.html",
    _REPO_ROOT / "docs" / "architecture.md",
    _REPO_ROOT / "docs" / "edge_pi_offline.md",
    _REPO_ROOT / "docs" / "clinical_validation.md",
    _REPO_ROOT / "docs" / "arch_mind_federation_audit.md",
    _REPO_ROOT / "docs" / "fda_q_sub_draft.md",
    _REPO_ROOT / "docs" / "federated_memory.md",
    _REPO_ROOT / "docs" / "why_mind_mem_v3.md",
    _REPO_ROOT / "docs" / "why_bitnet_b158.md",
    _REPO_ROOT / "docs" / "bitnet_training.md",
)

# Patterns that indicate a PRESENT-TENSE claim about *the current*
# mind-mem version. Each pattern's first capture-group sequence MUST
# isolate the version digits (X, Y, Z).
#
# Ordering: more-specific first (so 'IS shipped' isn't caught twice
# by both 'is shipped' and 'IS shipped').
_PRESENT_TENSE_PATTERNS: tuple[re.Pattern[str], ...] = (
    # 'mind-mem v3.10.0 is shipped and pinned'
    # 'MIND-Mem v3.10.0 IS shipped and pinned'
    re.compile(
        r"\b(?:mind-mem|MIND-Mem|MIND-mem|Mind-Mem)\s+v?(\d+)\.(\d+)\.(\d+)\s+(?:is|IS)\s+shipped\s+and\s+pinned",
    ),
    # 'v3.10.0 ships no new federation-transport module'
    # 'v3.10.0 ships no new transport module'
    re.compile(
        r"(?:^|[^.\w])v(\d+)\.(\d+)\.(\d+)\s+ships\s+no\s+new\s+\w+",
    ),
    # 'v3.10.0 shipped 2026-05-07 with no new federation-transport module'
    # — present-tense framing of past-tense fact: when the doc says
    # 'v<X> shipped <date> with no new module', it's implicitly framing
    # v<X> as the relevant shipping-state baseline. If v<X> < pin, the
    # framing is stale; the live shipping baseline is the pin.
    re.compile(
        r"(?:^|[^.\w])v(\d+)\.(\d+)\.(\d+)\s+shipped\s+\d{4}-\d{2}-\d{2}\s+with\s+no",
    ),
    # 'mind-mem v3.10.0 is the current pin'
    # 'mind-mem v3.10.0 is the live release'
    re.compile(
        r"\b(?:mind-mem|MIND-Mem|MIND-mem|Mind-Mem)\s+v?(\d+)\.(\d+)\.(\d+)\s+is\s+the\s+(?:current|live)",
    ),
    # 'mind-mem v3.10.0 is currently shipped'
    re.compile(
        r"\b(?:mind-mem|MIND-Mem|MIND-mem|Mind-Mem)\s+v?(\d+)\.(\d+)\.(\d+)\s+is\s+currently",
    ),
)

# Allowlist tokens. If any of these appears in the SAME LINE as the
# matched present-tense claim, the match is allowed (the doc is
# explicitly framing a multi-version range).
_LINE_ALLOWLIST_TOKENS: tuple[str, ...] = (
    "through v3.10",      # 'the v3.10.x line through v3.10.8'
    "through v3.11",
    "through v4.",
    "v3.10.x line through",
    "v3.x line through",
    "the line through",
    "iter-342",           # 'iter-342 STEP 1 INTEGRATION FIRED v2 — mind-mem v3.10.0 → v3.10.6'
    "iter-331",           # historical iter-331 STEP 1 INTEGRATION lineage prose
    "the upstream",       # 'the upstream commit a42cf45 (mind-mem v3.10.2)'
)


def _live_pin() -> tuple[int, int, int]:
    """Read pyproject.toml and return (X, Y, Z) of the mind-mem pin."""
    text = _PYPROJECT.read_text() if _PYPROJECT.exists() else ""
    m = re.search(r'"mind-mem>=(\d+)\.(\d+)\.(\d+)"', text)
    assert m is not None, (
        "Could not parse mind-mem pin from pyproject.toml. Expected "
        "'mind-mem>=X.Y.Z' form."
    )
    return (int(m.group(1)), int(m.group(2)), int(m.group(3)))


def test_no_stale_present_tense_shipping_claim():
    """No user-facing doc may make a present-tense shipping/pinning
    claim about a mind-mem version older than the pyproject pin.

    Catches the iter-388 drift class: 'v3.10.0 is shipped and pinned'
    when the pyproject pin is v3.10.8. The iter-321/339/345 dep-version
    pin tolerates this via its 'released 2026-' window allowlist; this
    pin closes the gap.
    """
    pin_maj, pin_min, pin_pt = _live_pin()
    pin_str = f"v{pin_maj}.{pin_min}.{pin_pt}"

    violations: list[str] = []
    for doc in _USER_FACING_DOCS:
        if not doc.exists():
            continue
        lines = doc.read_text().splitlines()
        for i, line in enumerate(lines):
            for pat in _PRESENT_TENSE_PATTERNS:
                for m in pat.finditer(line):
                    maj = int(m.group(1))
                    mn = int(m.group(2))
                    pt = int(m.group(3))
                    if (maj, mn, pt) >= (pin_maj, pin_min, pin_pt):
                        continue  # equal-or-future is always OK
                    line_lower = line.lower()
                    if any(tok in line_lower for tok in _LINE_ALLOWLIST_TOKENS):
                        continue  # 'v3.10.x line through v3.10.8' is OK
                    violations.append(
                        f"{doc.relative_to(_REPO_ROOT)}:{i + 1} "
                        f"[v{maj}.{mn}.{pt} < pin {pin_str}]: "
                        f"{line.strip()[:160]!r}"
                    )
    assert not violations, (
        f"User-facing docs make present-tense shipping/pinning claims "
        f"about mind-mem versions older than the pyproject pin "
        f"({pin_str}). The iter-321/339/345 dep-version pin tolerates "
        f"these via its 'released 2026-' window; this 35th cross-pin "
        f"family closes the gap.\n\n"
        f"Violations ({len(violations)}):\n"
        + "\n".join(f"  - {v}" for v in violations[:20])
        + (f"\n  ... +{len(violations) - 20} more" if len(violations) > 20 else "")
        + "\n\nFix: either update the version to match the live pin, "
        "or reframe to a multi-version range "
        "('the v3.10.x line through v" + f"{pin_maj}.{pin_min}.{pin_pt}'), "
        "which the line-allowlist accepts."
    )


def test_pin_extraction_works():
    """Sanity: the pyproject pin parser actually returns the live pin."""
    maj, mn, pt = _live_pin()
    assert (maj, mn, pt) >= (3, 10, 0), (
        f"Live mind-mem pin {maj}.{mn}.{pt} is unexpectedly older "
        f"than v3.10.0; check whether the pyproject regex still "
        f"matches the pin format."
    )


def test_present_tense_pattern_catches_synthetic_drift():
    """Synthetic regression: if a doc said 'mind-mem v3.9.0 is shipped
    and pinned' (which would be ~10 minor versions stale), the pin must
    catch it."""
    pin_maj, pin_min, pin_pt = _live_pin()
    test_lines = [
        "mind-mem v3.9.0 is shipped and pinned",
        "MIND-Mem v3.9.5 IS shipped and pinned",
        "v3.9.0 ships no new federation-transport module",
        "mind-mem v3.10.0 is the current pin",
        "mind-mem v3.10.0 is currently shipped",
    ]
    caught = 0
    for line in test_lines:
        for pat in _PRESENT_TENSE_PATTERNS:
            m = pat.search(line)
            if m:
                maj, mn, pt = (int(g) for g in m.groups())
                if (maj, mn, pt) < (pin_maj, pin_min, pin_pt):
                    line_lower = line.lower()
                    if not any(tok in line_lower for tok in _LINE_ALLOWLIST_TOKENS):
                        caught += 1
                        break
    assert caught == len(test_lines), (
        f"Expected synthetic drift to be caught {len(test_lines)} times, "
        f"got {caught}. Patterns may be under-matching."
    )


def test_allowlist_accepts_multi_version_framing():
    """Synthetic regression: 'the v3.10.x line through v3.10.8' must
    be accepted (it's the iter-388 forward-protected framing)."""
    pin_maj, pin_min, pin_pt = _live_pin()
    test_lines = [
        "the v3.10.x line through v3.10.8 ships no new federation-transport module",
        "the line through v3.10.8 is hook-installer + CLI + docs only",
    ]
    for line in test_lines:
        line_lower = line.lower()
        # Should match the allowlist
        assert any(tok in line_lower for tok in _LINE_ALLOWLIST_TOKENS), (
            f"Multi-version range framing not caught by allowlist: {line!r}. "
            f"iter-388 / iter-390 'line through' framing must be tolerated."
        )
