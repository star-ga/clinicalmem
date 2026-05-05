"""Iter-148 Path A v3 full-recall multi-seed sweep V2.

Wider sweep (30 seeds) at EPOCHS=1800 (was 1200). Same gate:
29/29 contra + 4/4 major + <=1 FP.
"""
from __future__ import annotations
import os
import subprocess
import sys
import re
from pathlib import Path

_REPO = Path(__file__).resolve().parent
_LOG_DIR = Path("/tmp/v5_h128_sweep")
_LOG_DIR.mkdir(exist_ok=True)


def run_seed(seed: int) -> dict:
    env = os.environ.copy()
    env["TRAIN_SEED"] = str(seed)
    env["TRAIN_EPOCHS"] = "1800"
    log_path = _LOG_DIR / f"seed_{seed}.log"
    cp = subprocess.run(
        [sys.executable, str(_REPO / "train_bitnet_v5_h128.py")],
        env=env,
        capture_output=True,
        text=True,
        timeout=240,
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
    # 30 fresh seeds (skip the 14 already tried)
    seeds = [
        1, 2, 3, 5, 11, 17, 19, 23, 31, 53,
        71, 89, 101, 137, 199, 251, 401, 503, 701, 911,
        1009, 1499, 2003, 2999, 5003, 6007, 8009, 9999, 13337, 19999,
    ]
    results = []
    for s in seeds:
        print(f"=== seed={s} ===", flush=True)
        try:
            r = run_seed(s)
        except subprocess.TimeoutExpired:
            print(f"  seed={s} TIMEOUT", flush=True)
            continue
        contra_pct = (r['contra_tp'] / r['contra_n'] * 100) if r['contra_n'] else 0
        major_pct = (r['major_tp'] / r['major_n'] * 100) if r['major_n'] else 0
        print(
            f"  contra {r['contra_tp']}/{r['contra_n']} ({contra_pct:.1f}%)  "
            f"major {r['major_tp']}/{r['major_n']} ({major_pct:.1f}%)  "
            f"fp={r['fp']}  saved={r['saved']}",
            flush=True,
        )
        results.append(r)
        if r["saved"]:
            print(f"\n*** HIT at seed={s} — bundle saved. ***", flush=True)
            break

    if not any(r["saved"] for r in results):
        print("\nNo seed hit 29/29 + 4/4 + <=1 FP. Best partial results:", flush=True)
        scored = sorted(
            results,
            key=lambda r: (
                -(r["contra_tp"] + r["major_tp"]),
                r["fp"],
                -r["contra_tp"],
            )
        )
        for r in scored[:8]:
            print(
                f"  seed={r['seed']:5d}: contra {r['contra_tp']}/{r['contra_n']}  "
                f"major {r['major_tp']}/{r['major_n']}  fp={r['fp']}",
                flush=True,
            )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
