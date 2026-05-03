"""Re-generate docs/bitnet_pairs.json from the trained ternary weights.

Runs the production inference path (engine.bitnet_classifier.classify) on
the canonical demo-pair list so the dashboard tile and the in-browser
"Verify Replay" button always reflect the currently-deployed weights
bundle.

Run:
    python3 scripts/build_bitnet_pairs_json.py
or, with --check, exit non-zero if docs/bitnet_pairs.json is stale:
    python3 scripts/build_bitnet_pairs_json.py --check
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_HERE))

from engine.bitnet_classifier import classify, load_weights  # noqa: E402

OUT = _HERE / "docs" / "bitnet_pairs.json"
WEIGHTS = _HERE / "engine" / "bitnet_weights.json"

# Canonical demo-pair list. Order is alphabetical-by-sorted-pair so the
# dashboard always renders deterministically. Add a row by adding the
# pair here — the hash and severity are computed by the script.
DEMO_PAIRS: list[tuple[str, str]] = [
    ("warfarin", "ibuprofen"),
    ("warfarin", "aspirin"),
    ("warfarin", "naproxen"),
    ("warfarin", "amiodarone"),
    ("warfarin", "metronidazole"),
    ("warfarin", "fluconazole"),
    ("coumadin", "ibuprofen"),
    ("coumadin", "aspirin"),
    ("warfarin", "alcohol"),
    ("amoxicillin", "penicillin"),
    ("amoxicillin", "cefalexin"),
    ("cefazolin", "penicillin"),
    ("cephalexin", "penicillin"),
    ("metformin", "iodine"),
    ("metformin", "contrast"),
    ("metformin", "alcohol"),
    ("simvastatin", "grapefruit"),
    ("simvastatin", "clarithromycin"),
    ("atorvastatin", "grapefruit"),
    ("lisinopril", "potassium"),
    ("lisinopril", "spironolactone"),
    ("sertraline", "tramadol"),
    ("paroxetine", "tramadol"),
    ("fluoxetine", "tramadol"),
    ("paroxetine", "phenelzine"),
    ("ssri", "maoi"),
    ("digoxin", "furosemide"),
    ("digoxin", "amiodarone"),
    ("metoprolol", "verapamil"),
    ("atenolol", "verapamil"),
    ("benzodiazepine", "opioid"),
    ("alprazolam", "oxycodone"),
    ("diazepam", "morphine"),
    ("omeprazole", "plavix"),
    ("aspirin", "plavix"),
    ("aspirin", "clopidogrel"),
    ("aspirin", "vitaminC"),
    ("calcium", "vitaminD"),
    ("coffee", "water"),
]


def _pair_key(a: str, b: str) -> str:
    return "|".join(sorted([a.lower().strip(), b.lower().strip()]))


def build_payload() -> dict:
    weights = load_weights(WEIGHTS)
    weights_file_sha = hashlib.sha256(WEIGHTS.read_bytes()).hexdigest()

    pairs: dict[str, dict] = {}
    for a, b in DEMO_PAIRS:
        result = classify(a, b, weights)
        key = _pair_key(a, b)
        pairs[key] = {
            "severity_name": result.severity_name,
            "repro_hash": result.repro_hash,
            "feature_hash": result.feature_hash,
            "logits_q16": list(result.logits_q16),
            "weights_id": result.weights_id,
        }

    return {
        "weights_id": weights.bundle_id,
        "weights_file_sha256": weights_file_sha,
        "pair_count": len(pairs),
        "pairs": pairs,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--check", action="store_true",
                   help="Exit 1 if docs/bitnet_pairs.json is stale.")
    args = p.parse_args()

    fresh = build_payload()
    fresh_text = json.dumps(fresh, sort_keys=True, separators=(",", ":"))

    if args.check:
        if not OUT.exists():
            print(f"FAIL: {OUT} does not exist", file=sys.stderr)
            return 1
        on_disk = json.dumps(
            json.loads(OUT.read_text(encoding="utf-8")),
            sort_keys=True, separators=(",", ":"),
        )
        if on_disk != fresh_text:
            print(f"FAIL: {OUT} is stale; re-run scripts/build_bitnet_pairs_json.py",
                  file=sys.stderr)
            return 1
        print(f"OK: {OUT} matches current weights")
        return 0

    OUT.write_text(json.dumps(fresh, sort_keys=True, indent=2), encoding="utf-8")
    print(f"Wrote {len(fresh['pairs'])} pairs -> {OUT}")
    print(f"  weights_id = {fresh['weights_id']}")
    print(f"  weights_file_sha256 = {fresh['weights_file_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
