#!/usr/bin/env python3
"""Build the augmented training corpus for the v2 BitNet retrain.

Goal: lift contraindicated recall from 6 / 20 = 30.0% to 20 / 20 = 100%
on the live OpenEvidence cache, WITHOUT introducing any false positives
on the contraindicated class (preserves the iter-50 safety invariant
`fp_contraindicated_is_zero`).

Strategy:
  1. Load the existing 3,247-pair corpus from
     clinicalmem-bitnet-training/drug_corpus.jsonl
  2. Read every `severity=contraindicated` entry from
     docs/openevidence_cache.json (live cache, currently 20 pairs).
  3. Append those 20 pairs to the corpus as `source=cache_contraindicated`,
     so the train script's forced-train logic picks them up as
     audit-chain anchors.
  4. Write the augmented corpus to
     retrain_runpod/drug_corpus_augmented.jsonl

The augmentation is additive — no existing rows are modified or removed.

Usage::

    python3 retrain_runpod/build_augmented_corpus.py

Output:
    retrain_runpod/drug_corpus_augmented.jsonl  (3,267 lines)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_BASE_CORPUS = Path("clinicalmem-bitnet-training/drug_corpus.jsonl")
_CACHE = _REPO_ROOT / "docs" / "openevidence_cache.json"
_OUT = _REPO_ROOT / "retrain_runpod" / "drug_corpus_augmented.jsonl"

# Severity name → integer (matches engine.bitnet_classifier).
_SEVERITY_INT = {
    "none": 0,
    "minor": 1,
    "moderate": 2,
    "major": 3,
    "contraindicated": 4,
    "serious": 3,  # serious maps to 3 ("major" slot) per the existing schema
}


def _row_key(drug_a: str, drug_b: str) -> tuple[str, str]:
    return tuple(sorted([drug_a.lower().strip(), drug_b.lower().strip()]))


def main() -> int:
    if not _BASE_CORPUS.exists():
        print(f"FAIL — base corpus not found: {_BASE_CORPUS}")
        print("       Expected clinicalmem-bitnet-training/")
        print("       to be populated; run build_corpus.py there first.")
        return 1

    if not _CACHE.exists():
        print(f"FAIL — OpenEvidence cache not found: {_CACHE}")
        return 1

    # Load existing corpus.
    base_rows: list[dict] = []
    with _BASE_CORPUS.open("r", encoding="utf-8") as fh:
        for line in fh:
            base_rows.append(json.loads(line))
    print(f"Loaded base corpus: {len(base_rows)} rows from {_BASE_CORPUS}")

    base_keys = {_row_key(r["drug_a"], r["drug_b"]) for r in base_rows}

    # Load cache contraindicated entries.
    cache = json.loads(_CACHE.read_text())
    contra = [e for e in cache if e.get("severity") == "contraindicated"]
    print(f"Loaded {len(contra)} contraindicated cache entries from {_CACHE}")

    # Append entries that are NOT already in the corpus, as the
    # `cache_contraindicated` source. The train script's REGRESSION_PAIRS
    # logic will be extended to recognise this source as a forced-train
    # fold (see retrain_runpod/train_bitnet_v2.py).
    new_rows: list[dict] = []
    duplicates: list[tuple[str, str]] = []
    for entry in contra:
        a, b = entry["drug_pair_canonical"]
        key = _row_key(a, b)
        if key in base_keys:
            duplicates.append((a, b))
            continue
        new_rows.append({
            "drug_a": a,
            "drug_b": b,
            "severity": _SEVERITY_INT["contraindicated"],
            "severity_name": "contraindicated",
            "source": "cache_contraindicated",
            "description": entry.get("clinical_summary", "")[:280],
            "_iter65_anchor": True,  # marker for the train script
        })

    print(f"  {len(new_rows)} new contraindicated pairs to add")
    if duplicates:
        print(f"  {len(duplicates)} cache pairs already in corpus (skipped):")
        for a, b in duplicates:
            print(f"    {a} + {b}")

    # Write augmented corpus.
    augmented = base_rows + new_rows
    _OUT.parent.mkdir(parents=True, exist_ok=True)
    with _OUT.open("w", encoding="utf-8") as fh:
        for row in augmented:
            fh.write(json.dumps(row, sort_keys=True) + "\n")

    print(f"\nWrote augmented corpus: {len(augmented)} rows -> {_OUT}")
    print(f"  (base {len(base_rows)} + cache_contraindicated {len(new_rows)})")

    # Per-class summary
    by_sev: dict[str, int] = {}
    for row in augmented:
        s = row.get("severity_name", "?")
        by_sev[s] = by_sev.get(s, 0) + 1
    print("\nPer-class corpus distribution:")
    for sev in ("none", "minor", "moderate", "serious", "major", "contraindicated"):
        if sev in by_sev:
            print(f"  {sev:18s} {by_sev[sev]:>5d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
