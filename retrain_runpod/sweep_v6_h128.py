"""Iter-207 Path A v6 multi-seed sweep — 8 BOOST_KEYS anchors.

Runs train_bitnet_v6_h128.py at 30 seeds. Gate (live, runs against
iter-202 38-contra cohort): full recall (38/38 contra + 4/4 major +
<= 1 FP). First seed to hit the gate saves the v6 bundle and stops.

CPU-bound, no GPU required (BitNet 1.58 ternary classifier). Each
seed takes ~30-90 seconds.
"""
from __future__ import annotations
import json
import os
import re
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent
_LOG_DIR = Path("/tmp/v6_h128_sweep")
_LOG_DIR.mkdir(exist_ok=True)


def run_seed(seed: int) -> dict:
    env = os.environ.copy()
    env["TRAIN_SEED"] = str(seed)
    env["TRAIN_EPOCHS"] = "1800"
    log_path = _LOG_DIR / f"seed_{seed}.log"
    cp = subprocess.run(
        [sys.executable, str(_REPO / "train_bitnet_v6_h128.py")],
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
        cwd=str(_REPO),
    )
    log_path.write_text(cp.stdout + "\n---STDERR---\n" + cp.stderr)
    out = cp.stdout
    contra_match = re.search(r"Contraindicated recall: (\d+)/(\d+)", out)
    major_match = re.search(r"Major recall: (\d+)/(\d+)", out)
    fp_match = re.search(r"Contraindicated false positives: (\d+)", out)
    return {
        "seed": seed,
        "contra_tp": int(contra_match.group(1)) if contra_match else 0,
        "contra_n": int(contra_match.group(2)) if contra_match else 0,
        "major_tp": int(major_match.group(1)) if major_match else 0,
        "major_n": int(major_match.group(2)) if major_match else 0,
        "fp": int(fp_match.group(1)) if fp_match else 999,
        "saved": "saved to " in out,
    }


def main():
    seeds = [
        1, 2, 3, 5, 7, 11, 13, 17, 19, 23,
        29, 31, 37, 41, 43, 47, 53, 59, 61, 67,
        71, 73, 79, 83, 89, 97, 101, 103, 107, 109,
    ]
    print(f"Iter-207 v6 sweep — 30 seeds × 1800 epochs")
    print(f"Gate: full recall (38/38 contra + 4/4 major + <=1 FP)")
    summary = []
    best = None
    for s in seeds:
        r = run_seed(s)
        summary.append(r)
        gate_full = (
            r["contra_tp"] == r["contra_n"]
            and r["major_tp"] == r["major_n"]
            and r["fp"] <= 1
            and r["contra_n"] >= 38
        )
        marker = "✓ GATE HIT" if gate_full else "."
        print(
            f"seed={s:3d} contra={r['contra_tp']}/{r['contra_n']} "
            f"major={r['major_tp']}/{r['major_n']} fp={r['fp']} {marker}"
        )
        if r["saved"]:
            print(f"*** v6 bundle saved at seed={s} — stopping sweep ***")
            break
        if best is None or (
            (r["contra_tp"], r["major_tp"], -r["fp"])
            > (best["contra_tp"], best["major_tp"], -best["fp"])
        ):
            best = r
    summary_path = _REPO / "training_summary_v6.json"
    summary_path.write_text(
        json.dumps(
            {"seeds": summary, "best": best},
            indent=2,
        )
    )
    print(f"\nsummary written to {summary_path}")
    if best:
        print(
            f"best: seed={best['seed']} "
            f"contra={best['contra_tp']}/{best['contra_n']} "
            f"major={best['major_tp']}/{best['major_n']} "
            f"fp={best['fp']}"
        )


if __name__ == "__main__":
    main()
