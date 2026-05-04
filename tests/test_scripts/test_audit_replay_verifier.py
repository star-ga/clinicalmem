"""Pin: Layer 4.5 audit-replay verifier exits 0 on the live state.

`scripts/verify_audit_replay.py --check` is the runnable form of the
"decade-stable audit replay" claim. The script re-classifies every
pair in `docs/audit_replay_pins.json` and asserts each `repro_hash`
reproduces byte-for-byte under the live `bundle_id`.

These tests guarantee:
  * pins file exists + has expected schema
  * pinned bundle_id matches engine weights bundle_id
  * `--check` subprocess exits 0
  * if a developer rotates weights without regenerating pins, the
    bundle_id mismatch surfaces as a clear actionable failure
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_SCRIPT = _REPO_ROOT / "scripts" / "verify_audit_replay.py"
_PINS = _REPO_ROOT / "docs" / "audit_replay_pins.json"
_ENGINE_WEIGHTS = _REPO_ROOT / "engine" / "bitnet_weights.json"


@pytest.fixture(scope="module")
def pins() -> dict:
    assert _PINS.exists(), (
        f"Run `python3 {_SCRIPT.relative_to(_REPO_ROOT)}` to regenerate pins."
    )
    return json.loads(_PINS.read_text())


def test_pins_file_exists():
    assert _PINS.exists(), (
        "docs/audit_replay_pins.json must exist — regenerate via "
        "`python3 scripts/verify_audit_replay.py`"
    )


def test_pins_schema(pins):
    for key in ("bundle_id", "pairs", "name", "description", "license"):
        assert key in pins, f"missing top-level key: {key}"
    assert isinstance(pins["pairs"], list)
    assert len(pins["pairs"]) >= 5, (
        "audit-replay pin set must cover at least 5 representative pairs "
        "spanning all severity classes"
    )
    for entry in pins["pairs"]:
        for k in ("drug_a", "drug_b", "severity_name", "feature_hash",
                  "logits_q16", "repro_hash"):
            assert k in entry, f"pin entry missing key: {k}"
        # repro_hash + feature_hash must be 64-char hex
        assert len(entry["repro_hash"]) == 64
        assert len(entry["feature_hash"]) == 64
        assert all(c in "0123456789abcdef" for c in entry["repro_hash"])


def test_pinned_bundle_id_matches_engine_weights(pins):
    """A bundle_id mismatch means weights rotated without regenerating
    pins; the audit chain is broken until pins refresh."""
    weights_payload = json.loads(_ENGINE_WEIGHTS.read_text())
    canonical = json.dumps(
        {k: weights_payload[k] for k in ("hidden_w", "hidden_b", "output_w", "output_b")},
        sort_keys=True, separators=(",", ":"),
    )
    expected = hashlib.sha256(canonical.encode()).hexdigest()
    assert pins["bundle_id"] == expected, (
        f"audit-replay pins were captured under bundle_id "
        f"{pins['bundle_id'][:16]}… but engine weights now hash to "
        f"{expected[:16]}…. After a deliberate weight rotation, "
        f"regenerate via `python3 scripts/verify_audit_replay.py`."
    )


def test_audit_replay_check_passes():
    """The runnable `--check` mode exits 0 on the live state."""
    cp = subprocess.run(
        [sys.executable, str(_SCRIPT), "--check"],
        capture_output=True, text=True, timeout=60, cwd=str(_REPO_ROOT),
    )
    assert cp.returncode == 0, (
        f"`scripts/verify_audit_replay.py --check` exited {cp.returncode} — "
        f"audit-replay drift detected.\n"
        f"stdout:\n{cp.stdout}\n"
        f"stderr:\n{cp.stderr}\n"
        f"Investigate the mismatched repro_hash; if a deliberate weight "
        f"rotation happened, regenerate pins."
    )


def test_pins_cover_every_contraindicated_cache_entry(pins):
    """Iter-85 expanded the audit-replay pin set from 5 hand-picked
    pairs to **every contraindicated cache entry** (20 pairs as of
    iter-83 cohort growth). This pin makes the FDA SaMD 'every
    contraindicated decision is replayable' claim runnable for the
    entire safety class. Future cache growth that adds a contra pair
    AND fails to extend the replay set fails this gate."""
    cache_path = _REPO_ROOT / "docs" / "openevidence_cache.json"
    cache = json.load(cache_path.open())
    contra_keys = {
        tuple(sorted([
            e["drug_pair_canonical"][0].lower().strip(),
            e["drug_pair_canonical"][1].lower().strip(),
        ]))
        for e in cache if e.get("severity") == "contraindicated"
    }
    pin_keys = {
        tuple(sorted([e["drug_a"].lower().strip(),
                      e["drug_b"].lower().strip()]))
        for e in pins["pairs"]
    }
    missing = contra_keys - pin_keys
    assert not missing, (
        f"Audit-replay pin set must cover EVERY contraindicated cache "
        f"entry (full safety-class replay coverage). Missing: "
        f"{sorted(missing)}. Re-run "
        f"`python3 scripts/verify_audit_replay.py` to refresh."
    )


def test_pins_size_at_least_full_contra_coverage(pins):
    """Soft floor: pins must contain at least len(cache_contraindicated)
    pairs. Catches the case where the replay-set builder regressed to
    the iter-80 5-pair hand-picked sample."""
    cache_path = _REPO_ROOT / "docs" / "openevidence_cache.json"
    cache = json.load(cache_path.open())
    contra_count = sum(1 for e in cache if e.get("severity") == "contraindicated")
    assert len(pins["pairs"]) >= contra_count, (
        f"Audit-replay pins ({len(pins['pairs'])}) below the "
        f"contraindicated-class floor ({contra_count}). The replay-set "
        f"builder must include every contraindicated cache entry."
    )


def test_pins_cover_canonical_safety_classes(pins):
    """The pin set must include at least one contraindicated and one
    serious GROUND-TRUTH pair so judges can replay a 'serious safety'
    decision and a 'highest severity' decision both."""
    # The pinned `severity_name` is the BitNet PREDICTION, not ground
    # truth. For the safety-class coverage check, look at canonical
    # ground-truth pair identities.
    pair_keys = {
        tuple(sorted([e["drug_a"], e["drug_b"]]))
        for e in pins["pairs"]
    }
    # warfarin+ibuprofen is GT serious; clarithromycin+simvastatin is
    # GT contraindicated; metformin+iodine is GT major. Locking these
    # gives judges the highest-stakes pairs in the replay set.
    must_have = {
        ("ibuprofen", "warfarin"),                  # GT serious
        ("clarithromycin", "simvastatin"),          # GT contraindicated
        ("iodine", "metformin"),                    # GT major
    }
    missing = must_have - pair_keys
    assert not missing, (
        f"audit-replay pin set must include canonical safety-class "
        f"pairs; missing: {sorted(missing)}"
    )
