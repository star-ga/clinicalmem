#!/usr/bin/env python3
"""Compute the full Layer 4.5 BitNet confusion matrix on the live cache.

Produces ``docs/bitnet_confusion_matrix.json``: a JSON artifact that maps
every (ground_truth_severity, predicted_severity) pair to a count, plus
per-class precision / recall / TP / FP / FN.

The artifact is the audit-grade companion to ``test_bitnet_live_precision_pin.py``.
That test pins **only the contraindicated** class. This script gives auditors
a one-look full picture of where BitNet is precise (contraindicated: precision
1.000) and where the upstream 4-tier pipeline carries the load (serious:
BitNet rarely predicts this class — by design; the upstream consensus
catches it instead).

Run with::

    python3 scripts/build_bitnet_confusion_matrix.py

Use ``--check`` to verify the on-disk artifact matches the live computation
(for CI parity).
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_CACHE = _REPO_ROOT / "docs" / "openevidence_cache.json"
_OUT = _REPO_ROOT / "docs" / "bitnet_confusion_matrix.json"


_CLASSES: tuple[str, ...] = (
    "none",
    "minor",
    "moderate",
    "serious",
    "major",
    "contraindicated",
)


def _compute_matrix() -> dict:
    sys.path.insert(0, str(_REPO_ROOT))
    # iter-421 Path B: use classifier_layer (the live engine entry point)
    # so the matrix reflects the production cascade. classifier_layer
    # auto-loads the tier-2 specialist bundle when present and runs the
    # frozen-A-then-constrained-B dispatch the consensus pipeline uses.
    # Falls back transparently to A-only when bundle B is absent.
    from engine.bitnet_classifier import classifier_layer, load_weights, load_weights_b  # noqa: PLC0415

    weights = load_weights()
    weights_b = load_weights_b()
    cache = json.loads(_CACHE.read_text())

    matrix: dict[str, dict[str, int]] = {
        gt: {pred: 0 for pred in _CLASSES} for gt in _CLASSES
    }

    for entry in cache:
        gt = entry["severity"]
        if gt not in matrix:
            # Unknown ground-truth class — skip with a clear marker.
            continue
        drug_a, drug_b = entry["drug_pair_canonical"]
        pred = classifier_layer(drug_a, drug_b).severity_name
        if pred not in matrix[gt]:
            # New class emitted by classifier — extend matrix.
            for row in matrix.values():
                row[pred] = row.get(pred, 0)
        matrix[gt][pred] += 1

    per_class: dict[str, dict[str, float]] = {}
    for cls in _CLASSES:
        tp = matrix[cls][cls]
        fp = sum(matrix[gt][cls] for gt in _CLASSES if gt != cls)
        fn = sum(matrix[cls][p] for p in _CLASSES if p != cls)
        total = tp + fn
        if total == 0:
            continue
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / total
        per_class[cls] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "ground_truth_total": total,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
        }

    weights_id_a = weights.bundle_id
    weights_id_b = weights_b.bundle_id if weights_b is not None else None
    ensemble_active = weights_b is not None

    description = (
        "Live deployment-side confusion matrix of the Q16.16 ternary "
        "BitNet classifier ensemble on the OpenEvidence ground-truth cache. "
        "Layer 4.5 dispatches a frozen tier-1 contra/major gate (bundle A, v8) "
        "to a tier-2 serious/moderate specialist (bundle B, iter-421) under "
        "constrained argmax — A wins on contraindicated, B wins on every "
        "other class. Both forward passes are bit-identical Q16.16 ternary."
    ) if ensemble_active else (
        "Live deployment-side confusion matrix of the Q16.16 ternary "
        "BitNet classifier on the OpenEvidence ground-truth cache. "
        "Single-bundle (A-only) mode — tier-2 specialist absent."
    )

    return {
        "@context": "https://schema.org",
        "@type": "Dataset",
        "name": "ClinicalMem Layer 4.5 BitNet Confusion Matrix",
        "version": "2.0.0" if ensemble_active else "1.0.0",
        "dateCreated": datetime.now(timezone.utc).isoformat(),
        "license": "Apache-2.0",
        "description": description,
        "weights_id": weights_id_a,
        "weights_id_a": weights_id_a,
        "weights_id_b": weights_id_b,
        "ensemble_active": ensemble_active,
        "cache_pairs_total": sum(
            sum(row.values()) for row in matrix.values()
        ),
        "matrix": matrix,
        "per_class": per_class,
        "safety_invariants": {
            "fp_contraindicated_is_zero": (
                sum(
                    matrix[gt]["contraindicated"]
                    for gt in _CLASSES
                    if gt != "contraindicated"
                )
                == 0
            ),
            "tp_contraindicated_at_least_seven": (
                matrix["contraindicated"]["contraindicated"] >= 7
            ),
        },
    }


def _diff(live: dict, on_disk: dict) -> list[str]:
    """Return non-trivial differences between live + on-disk artifacts.

    Ignores the dateCreated timestamp.
    """
    live_copy = {k: v for k, v in live.items() if k != "dateCreated"}
    on_disk_copy = {k: v for k, v in on_disk.items() if k != "dateCreated"}
    if live_copy == on_disk_copy:
        return []
    diffs: list[str] = []
    for k in set(live_copy) | set(on_disk_copy):
        if live_copy.get(k) != on_disk_copy.get(k):
            diffs.append(f"{k}: live={live_copy.get(k)!r} vs on_disk={on_disk_copy.get(k)!r}")
    return diffs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify the on-disk artifact matches the live computation; exit 1 on diff",
    )
    args = parser.parse_args()

    live = _compute_matrix()

    if args.check:
        if not _OUT.exists():
            print(f"FAIL — {_OUT} does not exist")
            return 1
        on_disk = json.loads(_OUT.read_text())
        diffs = _diff(live, on_disk)
        if diffs:
            print("FAIL — on-disk artifact differs from live computation:")
            for d in diffs:
                print(f"  • {d}")
            return 1
        print(f"PASS — {_OUT} matches live computation")
        return 0

    _OUT.write_text(json.dumps(live, indent=2) + "\n")
    print(f"wrote {_OUT}")

    # Print a human-readable summary
    print("\nGround-truth → Predicted")
    print(f"{'GT':>16s} | " + " ".join(f"{c[:5]:>6s}" for c in _CLASSES) + " | total")
    for gt in _CLASSES:
        total = sum(live["matrix"][gt].values())
        if total == 0:
            continue
        cells = " ".join(f"{live['matrix'][gt][p]:>6d}" for p in _CLASSES)
        print(f"{gt:>16s} | {cells} | {total:>5d}")

    print("\nPer-class precision / recall:")
    for cls, m in live["per_class"].items():
        print(
            f"  {cls:>16s}: precision={m['precision']:.3f} "
            f"recall={m['recall']:.3f} (tp={m['tp']} fp={m['fp']} fn={m['fn']})"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
