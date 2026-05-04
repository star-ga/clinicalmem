"""Pin: the reproducibility-manifest card in demo.html shows the live SHAs.

Iter 56 added a card under the SHA-256 audit-trail section that exposes
the iter-55 manifest's per-artifact SHA-256 prefixes (16 chars) so a
judge sees the content-addressed audit surface without leaving the demo.

The prefixes are hand-coded into the HTML for offline-deterministic
display; this test guards against drift between the JSON manifest and
the demo card.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEMO = _REPO_ROOT / "docs" / "demo.html"
_MANIFEST = _REPO_ROOT / "docs" / "reproducibility_manifest.json"


def test_demo_manifest_card_present():
    text = _DEMO.read_text()
    # The card heading must include the artifact path.
    assert "docs/reproducibility_manifest.json" in text
    # Must reference the build script + --check command.
    assert "scripts/build_reproducibility_manifest.py" in text
    assert "--check" in text


def test_demo_card_sha_prefixes_match_manifest():
    """Every artifact prefix shown in the demo card must match the
    16-char prefix from the live manifest. If the manifest regenerates
    with new SHAs, this test fires until the demo is updated.
    """
    manifest = json.loads(_MANIFEST.read_text())
    demo = _DEMO.read_text()

    for name, info in manifest["artifacts"].items():
        if name == "flow_plan_hashes":
            continue
        prefix = info["sha256"][:16]
        # The card writes "<prefix>…" (with an ellipsis). Look for the
        # 16-char prefix anywhere in the demo HTML — covers both the
        # mono-text in the card and any inline mentions.
        assert prefix in demo, (
            f"reproducibility-manifest card in demo.html is missing the "
            f"artifact prefix for {name!r}: expected {prefix!r}.\n"
            f"Re-run `python3 scripts/build_reproducibility_manifest.py` "
            f"and update the hand-coded SHAs in the manifest card."
        )


def test_demo_card_announces_all_gates_pass():
    """The card chip must say 'PASS' for the gate verdict, and the
    manifest must back it up. Card now reports 5/5 gates after iter-90
    promoted the audit-replay verifier into run_all_gates.py."""
    text = _DEMO.read_text()
    manifest = json.loads(_MANIFEST.read_text())
    gate_count = len(manifest["gates"])
    expected_chip = f"{gate_count} / {gate_count} gates · PASS"
    assert expected_chip in text or expected_chip.lower() in text.lower(), (
        f"Reproducibility-manifest card must show {expected_chip!r} chip "
        f"(driven by manifest['gates'] which has {gate_count} entries)."
    )
    for verdict in manifest["gates"].values():
        assert verdict == "PASS", (
            f"Manifest reports a non-PASS gate verdict ({verdict}); "
            f"don't claim {gate_count}/{gate_count} PASS until that's fixed."
        )
