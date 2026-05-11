"""Pin: every `tests/test_*/test_*.py` path referenced in user-facing
docs must exist on disk.

Iter 228 (round 47 T1) — self-lock for the drift class iter-227 caught.
At iter-215 the pin-discipline cleanup deleted 5 v3/v5 pin files
(-25 tests) but the JUDGES.md rhetoric was never updated — judges
following the file paths in row 102's "Pinned by 6 pin families"
claim would have hit 4 file-not-found errors. iter-227 (T3 round-47)
caught and rewrote rows 102 + 106 to call out the iter-215 retirement
explicitly. **This pin prevents the regression** by mechanically
asserting every pin-file path mentioned in user-facing docs must exist.

Same drift-prevention shape as the iter-178 BOOST_KEYS coverage,
iter-183 Q16.16 canonical-pins coverage, iter-188 encode_pair contract,
iter-193 audit-replay structural integrity, iter-198 per-pair rule-bit,
iter-203 per-rule cohort-coverage, and iter-223 6-LLM consensus
provider-set pins. **8th cross-pin family** in the discipline lineage.

Surface locked
==============
Every `tests/test_<dir>/test_<name>.py` path mentioned in:
  • JUDGES.md (the 60-second audit guide judges scan first)
  • README.md (the GitHub repo landing page)
  • DEVPOST.md (the hackathon submission body)
  • docs/demo.html (the live dashboard's "pinned by" callouts)
  • docs/why_bitnet_b158.md and docs/why_mind_mem_v3.md (linked from JUDGES)

… must exist on disk. If a future cleanup deletes a pin file, this
pin fires before the doc rhetoric goes stale. The judge experience is:
"every tests/... path you read here resolves to a real, runnable file."

Adding a new pin file referenced in docs is fine; this pin only fires
on the deletion-without-doc-update direction.
"""
from __future__ import annotations

import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# Files where pin-test paths are commonly cited. Keep the list explicit
# rather than crawling every .md / .html so a stray temporary file doesn't
# trigger spurious failures.
#
# iter-404 scope extension (T1 round-24 cycle-4): add 5 more docs that
# cite test paths but were not in the iter-228 pin's original scope.
# Audit at iter-404 found 13 test-path citations across these 5 docs:
#   - docs/architecture.md (2 paths)
#   - docs/clinical_validation.md (1 path)
#   - docs/fda_q_sub_draft.md (6 paths)
#   - docs/arch_mind_federation_audit.md (1 path)
#   - docs/bitnet_training.md (3 paths)
# All would silently lie if the referenced test files were renamed/deleted
# without the citing doc being updated. Mirror of iter-371 scope-extension
# pattern at the user-doc-test-path-existence-pin layer.
_USER_DOCS = (
    _REPO_ROOT / "JUDGES.md",
    _REPO_ROOT / "README.md",
    _REPO_ROOT / "DEVPOST.md",
    _REPO_ROOT / "docs" / "demo.html",
    _REPO_ROOT / "docs" / "why_bitnet_b158.md",
    _REPO_ROOT / "docs" / "why_mind_mem_v3.md",
    # iter-404 scope extensions
    _REPO_ROOT / "docs" / "architecture.md",
    _REPO_ROOT / "docs" / "clinical_validation.md",
    _REPO_ROOT / "docs" / "fda_q_sub_draft.md",
    _REPO_ROOT / "docs" / "arch_mind_federation_audit.md",
    _REPO_ROOT / "docs" / "bitnet_training.md",
    # iter-408 scope extension (T5 round-24 cycle-4): clinicalmem_
    # invariants.md cites 3 test paths but was not in iter-404's
    # 11-doc scope. Same iter-371 / iter-404 scope-extension
    # pattern at the user-doc-test-path-existence-pin layer.
    _REPO_ROOT / "docs" / "clinicalmem_invariants.md",
)

# Pin-file paths look like `tests/test_<dir>/test_<name>.py`.
# We intentionally do NOT match other test paths (e.g. helper modules)
# to keep the surface focused on the user-facing pin-file claims.
_PIN_PATH_RE = re.compile(
    r"\btests/test_(?:engine|scripts|a2a|mcp|fhir)/test_[a-z][a-z_0-9]*\.py\b"
)


def _extract_pin_paths(text: str) -> set[str]:
    """All `tests/test_*/test_*.py` substrings in the text."""
    return set(_PIN_PATH_RE.findall(text))


def _surrounding_context(text: str, path: str, radius: int = 80) -> str:
    """Pretty-print 80 chars on each side of the first match for diagnostics."""
    idx = text.find(path)
    if idx < 0:
        return f"<could not relocate {path!r} for diagnostic>"
    start = max(0, idx - radius)
    end = min(len(text), idx + len(path) + radius)
    snippet = text[start:end].replace("\n", " ")
    return f"…{snippet}…"


def test_every_user_doc_pin_path_exists():
    """For every pin-file path mentioned in a user-facing doc, the file
    must exist on disk. Catches iter-215-style deletions that don't
    propagate to doc rhetoric (the iter-227 drift class)."""
    missing: list[tuple[Path, str]] = []
    for doc_path in _USER_DOCS:
        if not doc_path.exists():
            continue
        text = doc_path.read_text()
        for pin_path in sorted(_extract_pin_paths(text)):
            on_disk = _REPO_ROOT / pin_path
            if not on_disk.exists():
                missing.append((doc_path, pin_path))

    if missing:
        lines = []
        for doc_path, pin_path in missing:
            text = doc_path.read_text()
            ctx = _surrounding_context(text, pin_path)
            lines.append(
                f"  • {doc_path.relative_to(_REPO_ROOT)} → {pin_path}\n"
                f"      context: {ctx}"
            )
        raise AssertionError(
            "User-facing docs reference pin files that no longer exist on disk. "
            "Either restore the deleted file OR rewrite the doc rhetoric to "
            "reflect the cleanup (same shape as iter-227 — call out the retirement "
            "explicitly + redirect readers to the surviving pin):\n"
            + "\n".join(lines)
        )


def test_user_doc_pin_path_floor():
    """Sanity floor: at least 20 unique pin-file paths must be cited
    across the user-facing docs. Drops below 20 means someone bulk-
    deleted pin references without updating the surface — same drift
    class but in the opposite direction (silent rhetoric loss).

    Floor pinned at iter-228 from the live count of 23 unique paths
    in JUDGES.md alone. Anchored 3 below for cohort growth/cleanup
    headroom; if the count truly drops below 20 either a Major
    refactor is happening (and this pin should be re-anchored
    deliberately) or significant doc-rhetoric loss occurred."""
    all_paths: set[str] = set()
    for doc_path in _USER_DOCS:
        if not doc_path.exists():
            continue
        all_paths.update(_extract_pin_paths(doc_path.read_text()))

    assert len(all_paths) >= 20, (
        f"User-facing docs cite only {len(all_paths)} unique pin-file paths; "
        f"iter-228 floor is 20. A bulk doc-rhetoric loss appears to have "
        f"happened without lockstep pin-file deletion. Either restore the "
        f"doc rhetoric or anchor this floor lower with a deliberate "
        f"refactor commit."
    )


def test_judges_audit_guide_cites_recent_v8_pins():
    """JUDGES.md is the 60-second audit guide and must specifically cite
    the active v8 pin pair (iter-244 live-recall + iter-244 q16
    determinism). iter-245 retired the v6 pin family (architectural
    double 128 → 256 broke the BOOST_KEYS @200x ceiling); v8 is the
    current source of truth and must surface to judges. This is a
    softer guarantee than the existence pin above — it asserts the
    SHAPE of the audit-trail rhetoric is current."""
    judges = (_REPO_ROOT / "JUDGES.md").read_text()
    assert "test_path_a_v8_live_recall_pin.py" in judges, (
        "JUDGES.md must cite the iter-244 v8 live-recall pin so judges "
        "can find the source-of-truth recall claim under the v8 staged bundle."
    )
    assert "test_path_a_v8_q16_determinism_pin.py" in judges, (
        "JUDGES.md must cite the iter-244 v8 q16 canonical-pin "
        "determinism pin (16 canonical pairs × 4 pinned values × 100 "
        "iterations = 1600 forward-pass stress)."
    )


def test_user_doc_pin_path_existence_self_pin():
    """Cross-pin self-citation lineage (iter-228 onwards).

    iter-2026-05-11: AUTONOMOUS_WORK_LOG.md is now a local-only dev
    artifact (untracked in git, kept as a video-script reference). When
    the file is absent (e.g., in CI or a fresh clone), the cross-pin
    citation check is a no-op. When the file IS present (developer
    working tree), the citation check still runs.
    """
    import pytest

    log_path = _REPO_ROOT / "AUTONOMOUS_WORK_LOG.md"
    if not log_path.exists():
        pytest.skip("AUTONOMOUS_WORK_LOG.md is local-only (untracked); skipping cross-pin citation check")
    log_text = log_path.read_text()
    self_path = "test_user_doc_test_path_existence_pin.py"
    assert self_path in log_text, (
        f"This pin file ({self_path}) must be cited in AUTONOMOUS_WORK_LOG.md "
        f"by the iter-228 row so the cross-pin discipline lineage stays "
        f"auditable. If you're refactoring the file out, also drop the row "
        f"reference; otherwise add it."
    )
