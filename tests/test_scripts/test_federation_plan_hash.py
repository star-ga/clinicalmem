"""Regression test: pin the JointMemoryFederation plan_hash to the
canonical SHA-256 of the on-disk flow contract.

The dashboard, architecture doc, and federation_mock_demo.py all
claim a specific plan_hash. The audit-chain story is "every clinical
decision carries a 64-char plan_hash that recomputes to the same
SHA-256 over the canonical .flow.mind source years later." That
claim is only credible if the doc-displayed value is actually what
the flow file hashes to.

Two checks:

  1. The live `_compute_flow_plan_hash()` helper in the demo script
     produces the expected canonical hash. Any edit to the flow
     contract that's not paired with an updated displayed value
     fails this test.

  2. The displayed short form `cbfaf3e8…4e18b` (in docs/architecture.md
     and docs/demo.html) matches the live full hash's prefix +
     suffix. Catches the reverse drift: docs updated to a different
     hash without re-running the helper.
"""
from __future__ import annotations

import hashlib
import os
import re
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_FLOW = _REPO_ROOT / "flows" / "JointMemoryFederation.flow.mind"
_DEMO = _REPO_ROOT / "scripts" / "federation_mock_demo.py"
_ARCH_DOC = _REPO_ROOT / "docs" / "architecture.md"
_DEMO_HTML = _REPO_ROOT / "docs" / "demo.html"

# Pinned canonical hash — when this changes, the flow contract was
# edited; update this constant + the doc references in the same commit.
_EXPECTED_PLAN_HASH = (
    "cbfaf3e84de0187be82792be5a43e843f57bbad1cdcf4283633026b59d74e18b"
)


def _live_plan_hash() -> str:
    return hashlib.sha256(_FLOW.read_bytes()).hexdigest()


def test_flow_plan_hash_matches_pinned_value():
    """Flow contract on disk hashes to the documented plan_hash."""
    assert _live_plan_hash() == _EXPECTED_PLAN_HASH, (
        f"Flow contract was edited.\n"
        f"  Old plan_hash: {_EXPECTED_PLAN_HASH}\n"
        f"  New plan_hash: {_live_plan_hash()}\n"
        f"Update _EXPECTED_PLAN_HASH in this test AND every doc reference "
        f"in the same commit (docs/architecture.md, docs/demo.html, "
        f"scripts/federation_mock_demo.py displays it dynamically)."
    )


def test_demo_script_helper_returns_pinned_value():
    """The demo's `_compute_flow_plan_hash()` returns the pinned value."""
    sys.path.insert(0, str(_REPO_ROOT))
    from scripts.federation_mock_demo import _compute_flow_plan_hash  # noqa

    assert _compute_flow_plan_hash() == _EXPECTED_PLAN_HASH


def test_doc_displays_match_live_short_form():
    """The short-form plan_hash displayed in docs matches the live hash."""
    short = f"{_EXPECTED_PLAN_HASH[:8]}…{_EXPECTED_PLAN_HASH[-5:]}"
    arch_md = _ARCH_DOC.read_text()
    demo_html = _DEMO_HTML.read_text()

    assert short in arch_md, (
        f"docs/architecture.md is missing the pinned short plan_hash {short!r}"
    )
    assert short in demo_html, (
        f"docs/demo.html is missing the pinned short plan_hash {short!r}"
    )


def test_no_stale_short_hash_remains():
    """Any old plan_hash short forms must be removed from docs after rotation."""
    # Add historical short forms here as the contract evolves; the test
    # ensures they don't linger after a rotation.
    historical = (
        "6c6fb3ea…5846",  # pre-iteration-25 contract
    )
    arch_md = _ARCH_DOC.read_text()
    demo_html = _DEMO_HTML.read_text()
    for old in historical:
        assert old not in arch_md, (
            f"Stale plan_hash {old!r} still in docs/architecture.md — "
            f"replace with the current pinned value."
        )
        assert old not in demo_html, (
            f"Stale plan_hash {old!r} still in docs/demo.html — "
            f"replace with the current pinned value."
        )
