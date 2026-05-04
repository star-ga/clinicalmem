#!/usr/bin/env python3
"""Determinism stress test for Layer 4.5 BitNet's Q16.16 forward pass.

Runs the classifier on a fixed set of drug pairs many times in the same
process and asserts every iteration produces **bit-identical** output:
the SHA-256 ``repro_hash``, the ``severity_name``, and the Q16.16 logit
tuple all match exactly across iterations.

This is the headline FDA-SaMD claim ("an FDA auditor can replay the
Q16.16 forward pass decades later, on a $15 Pi Zero or an A100 GPU,
and obtain the same answer to the bit") tested at the same-process
level. Cross-machine determinism is implied by the Q16.16 fixed-point
math (no floating-point ops), but cannot be exercised on a single
host; this script proves at minimum that **the same machine** never
drifts.

Run::

    python3 scripts/run_bitnet_determinism_stress.py

Use ``--iterations N`` to set the per-pair iteration count (default 100,
total run = 100 × 12 pairs = 1200 classifier invocations, takes ~3 s
on a CPython 3.12 laptop).

Exit codes:
  0 — all iterations identical, no drift
  1 — at least one iteration produced a different output (regression)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent


# Stress fixture — 12 representative drug pairs covering every severity
# class plus a few edge cases (case differences, whitespace, ordering).
_STRESS_PAIRS: tuple[tuple[str, str], ...] = (
    ("warfarin", "ibuprofen"),
    ("clarithromycin", "simvastatin"),
    ("ciprofloxacin", "tizanidine"),
    ("methotrexate", "trimethoprim"),
    ("doxycycline", "isotretinoin"),
    ("clarithromycin", "ergotamine"),
    ("lisinopril", "sacubitril"),
    ("metformin", "lisinopril"),
    ("acetaminophen", "lisinopril"),       # negative control
    ("amlodipine", "atorvastatin"),         # CYP3A4 boundary, negative
    ("Warfarin", "IBUPROFEN"),              # case-insensitivity check
    ("  warfarin ", "ibuprofen "),          # whitespace tolerance
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="iterations per pair (default 100)",
    )
    args = parser.parse_args()

    sys.path.insert(0, str(_REPO_ROOT))
    from engine.bitnet_classifier import classify, load_weights  # noqa: PLC0415

    weights = load_weights()
    iterations = max(1, args.iterations)

    total_calls = 0
    drift_events: list[tuple[tuple[str, str], int, str, str]] = []

    print(f"BitNet Q16.16 determinism stress — {len(_STRESS_PAIRS)} pairs × {iterations} iterations")
    print("=" * 72)
    print(f"{'pair':40s} {'severity':16s} {'repro_hash':16s}")
    print("-" * 72)

    for pair in _STRESS_PAIRS:
        baseline = classify(pair[0], pair[1], weights)
        baseline_repr = (baseline.severity_name, baseline.repro_hash, baseline.logits_q16)
        total_calls += 1

        for i in range(1, iterations):
            result = classify(pair[0], pair[1], weights)
            this_repr = (result.severity_name, result.repro_hash, result.logits_q16)
            total_calls += 1
            if this_repr != baseline_repr:
                drift_events.append(
                    (pair, i, baseline.repro_hash[:16], result.repro_hash[:16]),
                )

        pair_label = f"{pair[0]} + {pair[1]}"[:38]
        marker = "✓" if not any(d[0] == pair for d in drift_events) else "✗"
        print(
            f"  {marker} {pair_label:38s} {baseline.severity_name:16s} "
            f"{baseline.repro_hash[:16]}…",
        )

    print("=" * 72)
    print(f"Total calls: {total_calls}")
    print(f"Drift events: {len(drift_events)}")

    if drift_events:
        print("\nDRIFT DETAIL:")
        for pair, iteration_index, expected, actual in drift_events:
            print(
                f"  {pair[0]} + {pair[1]} @ iter {iteration_index}: "
                f"expected={expected}… got={actual}…",
            )
        print("\nFAIL — Q16.16 forward pass produced non-identical output")
        print("       in the same process. The deterministic-replay")
        print("       claim is invalidated. Investigate weight loading,")
        print("       class state, or any code path that mutates a")
        print("       module-level dict / list during classify().")
        return 1

    print("\nPASS — every classify() call produced bit-identical output")
    print(f"       (severity_name + repro_hash + logits_q16 all match across {iterations} iterations)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
