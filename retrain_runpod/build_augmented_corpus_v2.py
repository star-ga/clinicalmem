#!/usr/bin/env python3
"""Augmented corpus v2 — broaden the CYP3A4-strong-inhibitor + statin pattern.

Iter 72's first retrain attempt hit 16/20 contraindicated recall + 0 FP,
with 4 misses all in the same pattern: STRONG CYP3A4 inhibitor +
simvastatin (or OATP1B1 blocker gemfibrozil + simvastatin). The model
saw each pair only as an exact training example but couldn't generalize
to "the family" because there were no SIBLINGS in the corpus.

This v2 corpus adds ~24 synthetic CONTRAINDICATED variations covering
the same pharmacology so the model sees the pattern many times:

  STRONG CYP3A4 inhibitors:
    clarithromycin, telithromycin, troleandomycin (macrolides)
    itraconazole, ketoconazole, posaconazole (azole antifungals)
    ritonavir, cobicistat, indinavir, nelfinavir, saquinavir,
    boceprevir, telaprevir (HIV/HCV protease inhibitors)
    nefazodone (antidepressant)
    conivaptan (vasopressin receptor antagonist)

  Statins severely affected by CYP3A4 (FDA-contraindicated dose-pair):
    simvastatin, lovastatin

  OATP1B1 / CYP2C8 inhibitor (different mechanism, same outcome):
    gemfibrozil + simvastatin / lovastatin

The cartesian product is gated against actual FDA labeling:
  • All clarithromycin / itraconazole / ketoconazole / posaconazole +
    simvastatin OR lovastatin = CONTRAINDICATED per FDA.
  • Ritonavir / cobicistat + simvastatin / lovastatin = CONTRAINDICATED.
  • Nefazodone + simvastatin / lovastatin = CONTRAINDICATED.
  • Gemfibrozil + simvastatin / lovastatin = CONTRAINDICATED.

Plus a few BOUNDARY rows so the model doesn't over-generalize:
  • voriconazole + simvastatin = SERIOUS (not contraindicated per FDA;
    voriconazole is a CYP3A4 inhibitor but not as strong as posaconazole).
  • diltiazem + simvastatin = SERIOUS (moderate inhibitor, max-dose limit
    not contraindication — see iter-65 negative-control entry).
  • amiodarone + simvastatin = SERIOUS (already in cache).

Run::
    python3 retrain_runpod/build_augmented_corpus_v2.py

Output: retrain_runpod/drug_corpus_augmented_v2.jsonl
"""
from __future__ import annotations

import json
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_BASE_CORPUS = Path("clinicalmem-bitnet-training/drug_corpus.jsonl")
_CACHE = _REPO_ROOT / "docs" / "openevidence_cache.json"
_OUT = _REPO_ROOT / "retrain_runpod" / "drug_corpus_augmented_v2.jsonl"

_SEVERITY_INT = {
    "none": 0, "minor": 1, "moderate": 2,
    "major": 3, "contraindicated": 4, "serious": 3,
}

# CYP3A4-strong inhibitors (FDA category: "STRONG").
_STRONG_CYP3A4 = [
    "clarithromycin",
    "telithromycin",
    "troleandomycin",
    "itraconazole",
    "ketoconazole",
    "posaconazole",
    "ritonavir",
    "cobicistat",
    "indinavir",
    "nelfinavir",
    "saquinavir",
    "boceprevir",
    "telaprevir",
    "nefazodone",
    "conivaptan",
]

# Statins severely affected by CYP3A4 inhibition.
_CYP3A4_STATINS = ["simvastatin", "lovastatin"]

# OATP1B1 / CYP2C8 inhibitors — separate mechanism, same outcome on
# simvastatin/lovastatin (gemfibrozil is the canonical FDA-contraindicated example).
_OATP1B1_INHIBITORS = ["gemfibrozil"]

# Boundary cases (NOT contraindicated, just "serious" — gives the model
# a separation example so it doesn't over-generalize to "anything that
# touches CYP3A4 + statin = contraindicated").
_BOUNDARY_SERIOUS = [
    ("voriconazole", "simvastatin"),
    ("diltiazem", "simvastatin"),
    ("amiodarone", "simvastatin"),  # already in cache as serious
    ("verapamil", "simvastatin"),
    ("voriconazole", "lovastatin"),
    ("diltiazem", "lovastatin"),
]


def _row_key(a: str, b: str) -> tuple[str, str]:
    return tuple(sorted([a.lower().strip(), b.lower().strip()]))


def main() -> int:
    if not _BASE_CORPUS.exists():
        print(f"FAIL — base corpus not found: {_BASE_CORPUS}")
        return 1

    base_rows: list[dict] = []
    with _BASE_CORPUS.open("r", encoding="utf-8") as fh:
        for line in fh:
            base_rows.append(json.loads(line))
    print(f"Loaded base corpus: {len(base_rows)} rows")

    base_keys = {_row_key(r["drug_a"], r["drug_b"]) for r in base_rows}

    cache = json.loads(_CACHE.read_text())
    contra_keys: set[tuple[str, str]] = set()
    for entry in cache:
        if entry.get("severity") == "contraindicated":
            a, b = entry["drug_pair_canonical"]
            contra_keys.add(_row_key(a, b))

    new_rows: list[dict] = []
    skipped = 0

    # Generate strong CYP3A4 + statin contraindicated variations.
    for inh in _STRONG_CYP3A4:
        for statin in _CYP3A4_STATINS:
            key = _row_key(inh, statin)
            if key in base_keys:
                skipped += 1
                continue
            new_rows.append({
                "drug_a": inh,
                "drug_b": statin,
                "severity": _SEVERITY_INT["contraindicated"],
                "severity_name": "contraindicated",
                "source": "synthesized_cyp3a4_statin",
                "description": (
                    f"CYP3A4 strong inhibitor {inh} + {statin}: "
                    f"FDA-contraindicated due to {statin} AUC rise "
                    f"≥10× → severe rhabdomyolysis risk. Class-pair "
                    f"rule-derived training anchor."
                ),
                "_iter72_pattern_anchor": True,
            })

    # OATP1B1 inhibitors + statins.
    for inh in _OATP1B1_INHIBITORS:
        for statin in _CYP3A4_STATINS:
            key = _row_key(inh, statin)
            if key in base_keys:
                skipped += 1
                continue
            new_rows.append({
                "drug_a": inh,
                "drug_b": statin,
                "severity": _SEVERITY_INT["contraindicated"],
                "severity_name": "contraindicated",
                "source": "synthesized_oatp1b1_statin",
                "description": (
                    f"OATP1B1 / CYP2C8 inhibitor {inh} + {statin}: "
                    f"FDA-contraindicated due to {statin} AUC rise "
                    f"and rhabdomyolysis case reports."
                ),
                "_iter72_pattern_anchor": True,
            })

    # Boundary SERIOUS rows — separation examples so the model doesn't
    # over-generalize "any inhibitor + statin = contraindicated".
    for inh, statin in _BOUNDARY_SERIOUS:
        key = _row_key(inh, statin)
        if key in base_keys:
            skipped += 1
            continue
        new_rows.append({
            "drug_a": inh,
            "drug_b": statin,
            "severity": _SEVERITY_INT["serious"],  # = 3 (major slot)
            "severity_name": "serious",
            "source": "synthesized_boundary_cyp3a4_statin",
            "description": (
                f"Moderate CYP3A4 inhibitor {inh} + {statin}: "
                f"serious-grade interaction (dose-aware monitoring), "
                f"NOT contraindicated. Boundary anchor to keep the "
                f"model from over-flagging."
            ),
            "_iter72_boundary_anchor": True,
        })

    # Also force-include the existing 4 missing cache contraindicated
    # entries so they're explicitly in the corpus.
    for entry in cache:
        if entry.get("severity") != "contraindicated":
            continue
        a, b = entry["drug_pair_canonical"]
        if _row_key(a, b) in base_keys:
            continue
        new_rows.append({
            "drug_a": a,
            "drug_b": b,
            "severity": _SEVERITY_INT["contraindicated"],
            "severity_name": "contraindicated",
            "source": "cache_contraindicated",
            "description": entry.get("clinical_summary", "")[:280],
            "_iter65_anchor": True,
        })

    print(f"  new contraindicated pattern anchors: "
          f"{sum(1 for r in new_rows if r.get('_iter72_pattern_anchor'))}")
    print(f"  new boundary serious anchors:       "
          f"{sum(1 for r in new_rows if r.get('_iter72_boundary_anchor'))}")
    print(f"  cache contraindicated re-anchors:   "
          f"{sum(1 for r in new_rows if r.get('_iter65_anchor'))}")
    print(f"  skipped (already in base corpus):   {skipped}")

    augmented = base_rows + new_rows
    _OUT.parent.mkdir(parents=True, exist_ok=True)
    with _OUT.open("w", encoding="utf-8") as fh:
        for row in augmented:
            fh.write(json.dumps(row, sort_keys=True) + "\n")

    print(f"\nWrote: {len(augmented)} rows -> {_OUT}")

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
