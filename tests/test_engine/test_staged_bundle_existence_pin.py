"""Pin: every staged-bundle path referenced in user-facing docs / pin
files must exist on disk.

Iter 244 (T1 round 51) — extends the iter-228 user-doc test-path
existence pin (which covers `tests/test_*/test_*.py` files) to the
binary artifacts that live in `retrain_runpod/`. The same drift class
caught at iter-227 — staged bundle files referenced in JUDGES + README
+ demo + pin source files MUST exist on disk, or the FDA SaMD audit
chain integrity claim breaks.

Why this matters
================

The iter-215 pin-discipline cleanup retired 5 v3/v5 PIN FILES but
deliberately KEPT the v3+v5 BUNDLE files on disk for audit-trail
rigor. If a future cleanup deletes a bundle file (and the doc
rhetoric still references it), judges following the bundle path
would hit a 404. The iter-228 pin only catches `tests/test_*.py`
references, not bundle paths.

This pin closes the gap by scanning user-facing docs + pin source
files for `retrain_runpod/bitnet_weights_*.json` references and
asserting each exists.

14th cross-pin family in the discipline lineage (after iter-178/183/
188/193/198/203/223/228/232/234/236/239/240).

Surface scanned
================
  • JUDGES.md
  • README.md
  • DEVPOST.md
  • docs/demo.html
  • docs/why_bitnet_b158.md
  • docs/architecture.md
  • Every `tests/test_*/test_*.py` source file (the v6/v7/v8 pin
    files reference bundle paths — must stay in sync)

Allowed
=======
A bundle path may be referenced AS A HISTORICAL ARTIFACT (e.g. v3
bundle on disk for audit-trail rigor even though no active pin uses
it). The pin only fires if the path is mentioned but the file
doesn't exist. Adding a new bundle path to docs requires the file
to land in the same commit, which is the whole point.
"""
from __future__ import annotations

import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_TESTS_DIR = _REPO_ROOT / "tests"

_USER_DOCS = (
    _REPO_ROOT / "JUDGES.md",
    _REPO_ROOT / "README.md",
    _REPO_ROOT / "DEVPOST.md",
    _REPO_ROOT / "docs" / "demo.html",
    _REPO_ROOT / "docs" / "why_bitnet_b158.md",
    _REPO_ROOT / "docs" / "architecture.md",
)

# Bundle path pattern: retrain_runpod/bitnet_weights_<name>.json
# OR engine/bitnet_weights.json (the shipped baseline).
_BUNDLE_PATH_RE = re.compile(
    r"\b(?:retrain_runpod|engine)/bitnet_weights[a-z0-9_]*\.json\b",
    re.IGNORECASE,
)


def _extract_bundle_paths(text: str) -> set[str]:
    """All `retrain_runpod/...json` and `engine/bitnet_weights.json`
    paths in the text."""
    return set(m.group(0) for m in _BUNDLE_PATH_RE.finditer(text))


def _scanned_files() -> list[Path]:
    """User-facing docs + every test pin file (the v6/v7/v8 pins
    reference bundle paths in their _BUNDLE constants)."""
    files = list(_USER_DOCS)
    files.extend(sorted(_TESTS_DIR.rglob("test_*.py")))
    return [f for f in files if f.exists()]


def test_every_referenced_bundle_path_exists():
    """Every `retrain_runpod/bitnet_weights_*.json` or
    `engine/bitnet_weights.json` path mentioned in user-facing docs
    or test pin files must exist on disk. Catches the iter-227-class
    drift extended to binary artifacts."""
    missing: list[tuple[Path, str]] = []

    for f in _scanned_files():
        text = f.read_text()
        for bundle_path in sorted(_extract_bundle_paths(text)):
            on_disk = _REPO_ROOT / bundle_path
            if not on_disk.exists():
                missing.append((f, bundle_path))

    if missing:
        lines = []
        for f, bundle_path in missing:
            rel = f.relative_to(_REPO_ROOT)
            lines.append(f"  • {rel} → {bundle_path}")
        raise AssertionError(
            "User-facing docs / test pin files reference bundle paths "
            "that no longer exist on disk. Same drift-prevention shape "
            "as iter-228 (which covers tests/test_*.py paths) extended "
            "to binary artifacts. The iter-215 pin-discipline cleanup "
            "deliberately KEPT v3+v5 bundles on disk for FDA SaMD audit-"
            "trail integrity; if a future cleanup deletes a bundle file, "
            "the doc rhetoric must update in lockstep:\n"
            + "\n".join(lines)
        )


def test_v8_staged_bundle_present():
    """The v8 staged bundle (the source of truth for the iter-244
    live-recall pin) MUST exist. This is a focused assertion on the
    LOAD-BEARING bundle that's pinned by 2 active v8 pin families;
    deleting it would cascade-break test_path_a_v8_live_recall_pin
    + test_path_a_v8_q16_determinism_pin."""
    v8_bundle = _REPO_ROOT / "retrain_runpod" / "bitnet_weights_v8_h256.json"
    assert v8_bundle.exists(), (
        f"v8 staged bundle missing: {v8_bundle}. The iter-244 v8 pin "
        f"families read this file as the source of truth for v8's 41/41 "
        f"+ 4/4 + 0 FP recall claim. Deleting it cascades into 14 broken "
        f"pin tests + the JUDGES row 102 narrative + the demo dashboard "
        f"V8 staged-ready callout."
    )


def test_v6_historical_bundle_preserved():
    """The v6 staged bundle (h=128) is now HISTORICAL (replaced by v8 at
    iter-245) but must stay on disk for FDA SaMD audit-trail integrity —
    same iter-215 pin-discipline that kept v3+v5 bundles after their
    pin retirement."""
    v6_bundle = _REPO_ROOT / "retrain_runpod" / "bitnet_weights_v6_h128.json"
    assert v6_bundle.exists(), (
        f"v6 historical bundle missing: {v6_bundle}. The iter-245 v6 → v8 "
        f"swap retired the v6 PIN FILES but explicitly KEPT the v6 BUNDLE "
        f"file on disk for audit-trail rigor. The JUDGES row 102 narrative "
        f"cites v6's 40/41 recall as the predecessor staged bundle; "
        f"deleting it breaks the v3 → v5 → v6 → v8 progression."
    )


def test_shipped_engine_bundle_present():
    """The shipped engine BitNet bundle (cfadb4f6 baseline) MUST exist
    at engine/bitnet_weights.json. This is THE classifier the live
    pipeline loads on every classify() call."""
    engine_bundle = _REPO_ROOT / "engine" / "bitnet_weights.json"
    assert engine_bundle.exists(), (
        f"Shipped engine BitNet bundle missing: {engine_bundle}. The "
        f"live `engine/bitnet_classifier.py::load_weights` reads this "
        f"file. Deleting it breaks every Layer 4.5 classification at "
        f"runtime; the entire safety pipeline collapses."
    )


def test_historical_v3_v5_bundles_preserved():
    """Per the iter-215 pin-discipline cleanup (user pivot "why are we
    still doing v5"): we DELETED 5 v5/v3 PIN FILES but KEPT the v3+v5
    BUNDLE files on disk for FDA SaMD audit-chain integrity. This pin
    locks that "kept on disk" promise — without it, a future cleanup
    might silently delete the bundle files since no active test
    references them."""
    historical_bundles = (
        _REPO_ROOT / "retrain_runpod" / "bitnet_weights_v3_full.json",
        _REPO_ROOT / "retrain_runpod" / "bitnet_weights_v5_h128.json",
    )
    missing = [b for b in historical_bundles if not b.exists()]
    assert not missing, (
        f"Historical v3/v5 bundle(s) missing: {missing}. The iter-215 "
        f"pin-discipline cleanup retired the v3+v5 PIN FILES but "
        f"explicitly KEPT the BUNDLE files for audit-trail rigor. The "
        f"JUDGES row 102 narrative cites these bundle hashes (v3 "
        f"`eea0e637…`, v5 `1ff61a6a…`) so judges can replay the "
        f"v3 → v5 → v6 progression. Removing them breaks the FDA SaMD "
        f"audit-trail integrity claim."
    )
