"""Pin: JUDGES.md description of the manifest stays in sync with live manifest.

Iter 107 (T3 round 21): JUDGES.md `Single-file reproducibility manifest`
row enumerates the manifest's gate verdicts and SHA-tracked artifacts.
Iter 90 added the audit-replay gate (4 → 5 gates). Iter 97 added the
pharmacology_flags artifact (7 → 8 SHA-tracked). Both changes shipped
with the manifest builder + demo card update + manifest pin tests, but
the JUDGES.md *description* of the manifest was not rotated.

Same silent-drift class as iter-89 caught for SVG row comments, iter-91
caught for the FHIR genesis block, and iter-101 caught for BitNet body
recall. JUDGES.md is the FIRST doc a hackathon judge reads — drift here
directly contradicts the manifest they're about to inspect.

This pin reads the live manifest and asserts:

  1. Every gate name in `manifest["gates"]` appears in JUDGES.md's
     description (case-insensitive substring match on the canonical
     short form).
  2. The "N SHA-tracked artifacts" claim in JUDGES.md matches the
     non-flow_plan_hashes count in `manifest["artifacts"]`.

Future iters that add a new gate or a new SHA-tracked artifact must
also rotate the JUDGES.md description or the gate fires.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_JUDGES = _REPO_ROOT / "JUDGES.md"
_MANIFEST = _REPO_ROOT / "docs" / "reproducibility_manifest.json"


# Map manifest gate keys to their canonical short form as it appears
# in JUDGES.md prose. Update both this map AND the JUDGES.md row when
# a new gate is added.
_GATE_SHORT_FORM = {
    "pccp_recall": "PCCP",
    "negative_control_precision": "negative-control",
    "federation_invariant": "federation",
    "arch_mind_l1": "arch-mind",
    "audit_replay": "audit-replay",
}


def _manifest():
    return json.loads(_MANIFEST.read_text())


def _judges_manifest_row():
    """Return the line of JUDGES.md describing the reproducibility
    manifest (the row containing 'Single-file reproducibility
    manifest')."""
    text = _JUDGES.read_text()
    for line in text.splitlines():
        if "Single-file reproducibility manifest" in line:
            return line
    return ""


def test_every_gate_named_in_judges_manifest_row():
    """Every gate key in `manifest["gates"]` must appear (by short
    form) in the JUDGES.md manifest row."""
    m = _manifest()
    gate_keys = list(m["gates"].keys())
    row = _judges_manifest_row()
    assert row, (
        "JUDGES.md must contain a row describing the reproducibility "
        "manifest with the marker 'Single-file reproducibility manifest'."
    )
    missing = []
    for gk in gate_keys:
        short = _GATE_SHORT_FORM.get(gk)
        if short is None:
            # If a gate isn't in our short-form map, the test is
            # informational — the operator must update _GATE_SHORT_FORM.
            missing.append(f"{gk!r} (not in _GATE_SHORT_FORM)")
            continue
        if short.lower() not in row.lower():
            missing.append(f"{gk!r} (short form {short!r})")
    assert not missing, (
        f"JUDGES.md manifest row missing gate names: {missing}. "
        f"Update the JUDGES row to enumerate all gates from "
        f"reproducibility_manifest['gates'] (currently: {gate_keys})."
    )


def test_judges_manifest_row_states_correct_sha_tracked_count():
    """JUDGES.md says 'N SHA-tracked artifacts'. N must equal the
    non-flow_plan_hashes count in manifest['artifacts']."""
    m = _manifest()
    artifacts = m["artifacts"]
    sha_tracked = [k for k in artifacts.keys() if k != "flow_plan_hashes"]
    expected = len(sha_tracked)
    row = _judges_manifest_row()
    pattern = re.compile(r"(\d+)\s+SHA-tracked\s+artifacts")
    match = pattern.search(row)
    if match is None:
        # Allow the row to omit the count phrase — but if it's there,
        # it must match.
        return
    displayed = int(match.group(1))
    assert displayed == expected, (
        f"JUDGES.md says '{displayed} SHA-tracked artifacts' but "
        f"manifest has {expected} SHA-tracked artifacts ({sha_tracked}). "
        f"Update the JUDGES row to {expected}."
    )


def test_judges_no_stale_4_gate_phrasing():
    """Iter-90 promoted the audit-replay verifier into run_all_gates
    (4 gates → 5). Iter-107 audit found the JUDGES.md manifest row
    still listed only 4 gates. Block the historical 4-gate phrasing
    from re-appearing once we've crossed to 5+."""
    m = _manifest()
    if len(m["gates"]) <= 4:
        return  # If we ever roll back, this stops being historical.
    text = _JUDGES.read_text()
    # The exact iter-89-era stale phrase from the manifest row.
    historical = (
        "gate verdicts (PCCP / negative-control / federation / arch-mind)",
    )
    for snippet in historical:
        assert snippet not in text, (
            f"Stale 4-gate phrasing {snippet!r} still in JUDGES.md. "
            f"Manifest now tracks {len(m['gates'])} gates; rotate the "
            f"description to enumerate all of them."
        )
