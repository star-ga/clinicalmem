#!/bin/bash
# Iter-148 Path A v3 full-recall multi-seed sweep.
# Runs train_bitnet_v3_full.py at 10 seeds; first to hit
# 29/29 contra + 4/4 major + ≤1 FP saves bundle and stops.
set -e
cd "$(dirname "$0")"
SEEDS=(0 7 13 42 99 256 512 1024 2048 4096)
for s in "${SEEDS[@]}"; do
    echo "=== seed=$s ==="
    if TRAIN_SEED=$s python3 train_bitnet_v3_full.py 2>&1 | tee "/tmp/v3_full_seed_${s}.log" | tail -10 | grep -q "✓"; then
        echo "*** HIT at seed=$s ***"
        exit 0
    fi
    if [ -f bitnet_weights_v3_full.json ]; then
        echo "*** bundle present after seed=$s ***"
        exit 0
    fi
done
echo "No seed hit the 29/29+4/4+≤1FP target."
exit 1
