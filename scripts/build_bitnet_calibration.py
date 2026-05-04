#!/usr/bin/env python3
"""Build the BitNet calibration / margin-diagnostic artifact.

Why this exists
───────────────
The Layer 4.5 confusion matrix (`docs/bitnet_confusion_matrix.json`)
tells you "how often is the classifier right." It does NOT tell you
"when the classifier is wrong, is it confidently wrong, or just
uncertain."

For an FDA SaMD reviewer, the second question is at least as
important. A model that is *uncertain* on its misses (small
top-1-vs-top-2 logit margin) is a model that knows what it
doesn't know — those cases can be safely deferred to upstream
layers (RxNorm / OpenEvidence / 5-LLM consensus).

The 4 iter-72 misses (clarithromycin / itraconazole / ketoconazole
/ gemfibrozil + simvastatin) all classify as `major` instead of
`contraindicated`. If their `major - contraindicated` logit margin
is small, the model is essentially saying "I'm 51% sure this is
major, 49% contraindicated" — and the upstream pipeline already
flags any of these as contraindicated via the deterministic FDA
table. The miss is *recoverable*, not silent.

This script computes that margin for every cache entry and writes:

  docs/bitnet_calibration.json

The artifact contains, per-pair:
  • drug_a, drug_b, ground_truth, predicted, correct
  • logits_q16 (all 5 class logits in canonical order)
  • top1_class, top2_class
  • margin_q16 (top1 - top2 in Q16.16; integer, not float)

And per-class aggregates:
  • count, correct_count, wrong_count
  • mean_margin_correct, mean_margin_wrong
  • worst_margin_wrong (smallest margin among wrong predictions)

Run::
    python3 scripts/build_bitnet_calibration.py

Output: `docs/bitnet_calibration.json`
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from engine.bitnet_classifier import (  # noqa: E402
    _SEVERITY_NAMES,
    classify,
    load_weights,
)

_CACHE = _REPO_ROOT / "docs" / "openevidence_cache.json"
_OUT = _REPO_ROOT / "docs" / "bitnet_calibration.json"


def _argsort_desc(values: list[int]) -> list[int]:
    return sorted(range(len(values)), key=lambda i: values[i], reverse=True)


def main() -> int:
    cache = json.loads(_CACHE.read_text())
    weights = load_weights()

    entries: list[dict] = []
    by_class: dict[str, dict] = {
        name: {
            "count": 0,
            "correct": 0,
            "wrong": 0,
            "_correct_margins": [],
            "_wrong_margins": [],
        }
        for name in _SEVERITY_NAMES
    }

    for entry in cache:
        a, b = entry["drug_pair_canonical"]
        gt = entry.get("severity")
        result = classify(a, b, weights)
        logits = list(result.logits_q16)
        order = _argsort_desc(logits)
        top1, top2 = order[0], order[1]
        margin = logits[top1] - logits[top2]
        pred = result.severity_name
        correct = (pred == gt)

        entries.append({
            "drug_a": a,
            "drug_b": b,
            "ground_truth": gt,
            "predicted": pred,
            "correct": correct,
            "logits_q16": logits,
            "top1_class": _SEVERITY_NAMES[top1],
            "top2_class": _SEVERITY_NAMES[top2],
            "margin_q16": margin,
        })

        if gt in by_class:
            bucket = by_class[gt]
            bucket["count"] += 1
            if correct:
                bucket["correct"] += 1
                bucket["_correct_margins"].append(margin)
            else:
                bucket["wrong"] += 1
                bucket["_wrong_margins"].append(margin)

    # Aggregate per-class.
    summary: dict[str, dict] = {}
    for sev, bucket in by_class.items():
        if bucket["count"] == 0:
            continue
        cm = bucket["_correct_margins"]
        wm = bucket["_wrong_margins"]
        summary[sev] = {
            "count": bucket["count"],
            "correct_count": bucket["correct"],
            "wrong_count": bucket["wrong"],
            "recall": bucket["correct"] / bucket["count"],
            "mean_margin_correct_q16": (sum(cm) // len(cm)) if cm else None,
            "mean_margin_wrong_q16": (sum(wm) // len(wm)) if wm else None,
            "min_margin_wrong_q16": min(wm) if wm else None,
            "max_margin_wrong_q16": max(wm) if wm else None,
        }

    # Top-K worst-margin wrong predictions (smallest margin = most uncertain
    # close-calls; a high margin on a wrong prediction is the bad case
    # because the model is "confidently wrong").
    wrong_entries = [e for e in entries if not e["correct"]]
    by_margin_asc = sorted(wrong_entries, key=lambda e: e["margin_q16"])
    by_margin_desc = sorted(wrong_entries, key=lambda e: -e["margin_q16"])

    payload = {
        "@context": "https://schema.org",
        "@type": "Dataset",
        "name": "ClinicalMem BitNet Layer 4.5 Calibration / Margin Diagnostic",
        "version": "1.0.0",
        "license": "Apache-2.0",
        "description": (
            "Per-pair top-1-vs-top-2 logit margin (Q16.16) for the live "
            "OpenEvidence cache. Surfaces whether the BitNet classifier "
            "is confidently right, uncertainly right, uncertainly wrong, "
            "or confidently wrong. The 'uncertainly wrong' bucket is the "
            "deferral-friendly case — small margin → upstream layer "
            "(RxNorm / OpenEvidence / 5-LLM consensus) decides."
        ),
        "weights_id": weights.bundle_id,
        "total_pairs": len(entries),
        "by_class": summary,
        "worst_close_calls": [
            {
                "drug_a": e["drug_a"],
                "drug_b": e["drug_b"],
                "ground_truth": e["ground_truth"],
                "predicted": e["predicted"],
                "margin_q16": e["margin_q16"],
            }
            for e in by_margin_asc[:10]
        ],
        "confidently_wrong": [
            {
                "drug_a": e["drug_a"],
                "drug_b": e["drug_b"],
                "ground_truth": e["ground_truth"],
                "predicted": e["predicted"],
                "margin_q16": e["margin_q16"],
            }
            for e in by_margin_desc[:10]
        ],
        "entries": entries,
    }

    _OUT.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(f"Wrote {_OUT.relative_to(_REPO_ROOT)} ({len(entries)} pairs)")

    # Concise console summary.
    print("\nPer-class calibration:")
    for sev in ("contraindicated", "major", "serious", "moderate", "minor", "none"):
        if sev not in summary:
            continue
        s = summary[sev]
        cm = s["mean_margin_correct_q16"]
        wm = s["mean_margin_wrong_q16"]
        cm_str = f"{cm:>10}" if cm is not None else "       n/a"
        wm_str = f"{wm:>10}" if wm is not None else "       n/a"
        print(
            f"  {sev:18s}  recall={s['recall']:.3f}  "
            f"mean_margin_correct={cm_str}  mean_margin_wrong={wm_str}"
        )

    print("\nWorst close-calls (smallest margin among wrong predictions):")
    for e in by_margin_asc[:5]:
        print(
            f"  margin_q16={e['margin_q16']:>10}  "
            f"{e['drug_a']} + {e['drug_b']}  "
            f"gt={e['ground_truth']:<18s}  pred={e['predicted']}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
