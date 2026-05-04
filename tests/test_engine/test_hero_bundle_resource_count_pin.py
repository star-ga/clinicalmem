"""Pin Sarah Mitchell's hero bundle entry count and Practitioner count.

The dashboard makes two visible numerical claims about the hero
patient's FHIR R4 bundle (`docs/sarah_mitchell_fhir.json`):

* The FHIR section header (`<span id="fhir-resource-count">18</span>
  resources covering Patient, 4 Practitioners (with HHS NPI...)`).
* The audit-chain genesis block (`Patient data ingested:
  Sarah Mitchell (18 FHIR R4 resources)`).

Iter-91 audit found those two surfaces had drifted apart: the FHIR
section was correctly reporting 18, but the genesis block still said
**47** — a stale claim from an earlier bundle revision that never got
rotated when the bundle shrank. The genesis block is load-bearing for
the audit-trail story (judges read it first when evaluating
provenance), so a wrong count there undermines the whole "every demo
finding traces back to a typed FHIR R4 resource" pitch.

This test pins both visible counts to the live bundle so future
bundle additions or removals fail the gate until the demo + audit-chain
copy rotate together.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_DEMO_HTML = _REPO_ROOT / "docs" / "demo.html"
_HERO_BUNDLE = _REPO_ROOT / "docs" / "sarah_mitchell_fhir.json"


def _live_bundle_counts() -> tuple[int, int]:
    """Return (entry_count, practitioner_count) for the hero bundle."""
    bundle = json.loads(_HERO_BUNDLE.read_text())
    entries = bundle.get("entry", [])
    types = Counter(e["resource"]["resourceType"] for e in entries)
    return len(entries), types.get("Practitioner", 0)


def test_fhir_section_resource_count_matches_live_bundle():
    html = _DEMO_HTML.read_text()
    entry_count, _ = _live_bundle_counts()
    pattern = re.compile(
        r'id="fhir-resource-count"[^>]*>(\d+)</span>',
        re.S,
    )
    m = pattern.search(html)
    assert m is not None, (
        "demo.html is missing the <span id=\"fhir-resource-count\">N</span> "
        "marker that announces the hero bundle's resource count."
    )
    displayed = int(m.group(1))
    assert displayed == entry_count, (
        f"FHIR section displays {displayed} resources but live "
        f"docs/sarah_mitchell_fhir.json has {entry_count} entries. "
        f"Update the span value AND the audit-chain genesis block."
    )


def test_fhir_section_practitioner_count_matches_live_bundle():
    html = _DEMO_HTML.read_text()
    _, practitioner_count = _live_bundle_counts()
    pattern = re.compile(r"(\d+)\s+Practitioners?\s*\(with HHS NPI")
    m = pattern.search(html)
    assert m is not None, (
        "demo.html is missing the 'N Practitioners (with HHS NPI...)' "
        "claim in the FHIR section header."
    )
    displayed = int(m.group(1))
    assert displayed == practitioner_count, (
        f"FHIR section claims {displayed} Practitioners but live "
        f"docs/sarah_mitchell_fhir.json has {practitioner_count}. "
        f"Update the demo copy AND any pinned Luhn-coverage claims."
    )


def test_genesis_block_resource_count_matches_live_bundle():
    """The audit-chain genesis block must report the live bundle count.

    Format: 'Patient data ingested: Sarah Mitchell (N FHIR R4 resources)'
    """
    html = _DEMO_HTML.read_text()
    entry_count, _ = _live_bundle_counts()
    pattern = re.compile(
        r"Sarah Mitchell\s*\((\d+)\s+FHIR R4\s+resources?\)"
    )
    m = pattern.search(html)
    assert m is not None, (
        "demo.html audit-chain genesis block must say "
        "'Sarah Mitchell (N FHIR R4 resources)'; pattern not found."
    )
    displayed = int(m.group(1))
    assert displayed == entry_count, (
        f"Audit-chain genesis block claims {displayed} resources but "
        f"live docs/sarah_mitchell_fhir.json has {entry_count} entries. "
        f"Drift here breaks the audit-trail provenance story."
    )


def test_no_stale_47_claim():
    """Iter-91 historical: the audit-chain genesis used to say '47 FHIR
    R4 resources' from an earlier bundle. That number must not reappear
    once the bundle has shrunk to 18.
    """
    html = _DEMO_HTML.read_text()
    entry_count, _ = _live_bundle_counts()
    if entry_count == 47:
        # If a future bundle expansion legitimately hits 47, this test
        # becomes a no-op (the genesis-block test above already guards the
        # value). Catching legitimate copy as stale would be a regression.
        return
    forbidden = "(47 FHIR R4 resources)"
    assert forbidden not in html, (
        f"Stale {forbidden!r} still in docs/demo.html. "
        f"Live bundle has {entry_count} entries — update the genesis block."
    )
