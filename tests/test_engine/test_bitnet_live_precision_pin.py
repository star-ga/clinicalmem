"""Pin Layer 4.5 BitNet's deployment-side precision/recall on the live cache.

Discovered while writing this test (iteration 29):

The dashboard claimed "85.7% precision on contraindicated", citing a
number that actually came from the held-out **test-set per-class
accuracy** in `docs/bitnet_training.md` (line 146, n=42). That's an
honest published number on a different dataset, but it was mislabeled
as "precision" on the dashboard, and it doesn't reflect deployment
reality.

The deployment-side reality, on the live `docs/openevidence_cache.json`
(n=109, contraindicated subset n=20):

  Layer 4.5 BitNet alone, contraindicated class
    precision = TP / (TP + FP) = 6 / 6  = 100.0%
    recall    = TP / (TP + FN) = 6 / 20 =  30.0%

Note: 30.0% sits exactly at the lower bound of the [0.30, 0.45]
recall band. One more contraindicated cohort entry without a
matching BitNet TP would push recall below 0.30 and require either
re-training (raising recall) or band re-calibration. T5 round 12+
should consider that ceiling when adding new contraindicated pairs.

Precision is 100% by design — Layer 4.5's job is *high-precision veto*,
not headline recall. Recall is intentionally low because the upstream
4-tier pipeline (RxNorm + OpenEvidence + 5-LLM consensus) is the
primary classifier. The Layer 4.5 alert fires only when it's
confident, and it's never wrong (so far) on the deployment cache.

This test pins the deployment numbers so a future weight rotation
can't silently break them.
"""
from __future__ import annotations

import json
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_CACHE = _REPO_ROOT / "docs" / "openevidence_cache.json"


def _bitnet_predictions():
    import sys
    sys.path.insert(0, str(_REPO_ROOT))
    from engine.bitnet_classifier import classify, load_weights

    weights = load_weights()
    cache = json.loads(_CACHE.read_text())
    return [
        (it["severity"], classify(it["drug_a"], it["drug_b"], weights).severity_name)
        for it in cache
    ]


def _precision_recall(preds: list[tuple[str, str]], cls: str) -> tuple[float, float, int, int]:
    tp = sum(1 for gt, p in preds if gt == cls and p == cls)
    fp = sum(1 for gt, p in preds if gt != cls and p == cls)
    fn = sum(1 for gt, p in preds if gt == cls and p != cls)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return precision, recall, tp, tp + fn


def test_bitnet_live_precision_recall_pinned():
    """Deployment-side BitNet precision/recall on contraindicated class.

    **Iter-275 v8 promotion**: engine bundle swapped cfadb4f6 (v1, 128-dim
    hash-only encoder, hidden=64) → 1f0f8859 (v8, 193-dim hash + 26 ATC
    flag + 13 pair-derived encoder, hidden=256). v8 lifts contra recall
    from the iter-235 floor (8/43 = 18.60%) to FULL recall (43/43 =
    100.0%). Both precision (100%) and the zero-FP invariant are
    preserved. The recall band now hard-locks at [1.0, 1.0] — any
    regression below 100% is release-blocking under the iter-244
    "100% is the only goal" / "41/41" directive (extended to the live
    43-entry cohort post iter-249 + iter-254 cohort growth).
    """
    preds = _bitnet_predictions()
    precision, recall, tp, total = _precision_recall(preds, "contraindicated")

    assert precision == 1.0, (
        f"Layer 4.5 deployment precision on contraindicated drifted: "
        f"live={precision:.4f}, pinned=1.0000 (zero-FP invariant)"
    )
    # v8 promotion (iter-275) hard-locks contra recall at 100%. The
    # pre-v8 fluctuating-band (iter-99 → iter-235 = 0.18 → 0.45) is no
    # longer applicable: v8's 26-flag + 13-pair-derived encoder closes
    # every sub-class generalization gap that the v1 hash-only encoder
    # left open. Any miss is release-blocking — the architectural
    # double (h=128 → h=256) was sized to hit and HOLD 100% across
    # cohort growth.
    assert recall == 1.0, (
        f"Layer 4.5 deployment recall on contraindicated dropped below "
        f"100%: live={recall:.4f}. v8 promotion (iter-275) locked the "
        f"recall floor at 1.0; any regression is release-blocking."
    )
    assert tp == total, (
        f"true positives must equal cohort total under v8: live "
        f"tp={tp}, total={total}"
    )
    assert total == 43, f"contraindicated cohort size drifted: live={total}, pinned=43"


def test_dashboard_displays_live_precision_number():
    """Dashboard's hero precision card shows the live 100% number."""
    demo_html = (_REPO_ROOT / "docs" / "demo.html").read_text()
    # Iter-150: hero stat chip relabelled "Layer 4.5 · live precision"
    # -> "Live precision · L4.5" (mobile-fit). Either form satisfies
    # the pin's intent: a "live precision" label, NOT the bare 85.7%
    # held-out-accuracy framing this test exists to forbid.
    assert (
        "Layer 4.5 · live precision" in demo_html
        or "Live precision · L4.5" in demo_html
    ), (
        "Hero stat chip must label the precision number with a 'live "
        "precision' framing (not the bare '85.7% precision on "
        "contraindicated' which conflated held-out test-set accuracy "
        "with deployment precision)."
    )
    # Iter-275 v8 promotion: hero precision chip footnote now reads the
    # live v8 numbers (43/43 — full recall, 0 FP, precision still 100%).
    # Pre-v8 chip showed "8 / 8" (the iter-148 v1 hash-only baseline),
    # which understated v8's deployed reach. Both forms are accepted to
    # ease the lockstep migration; v8 form is canonical post-iter-275.
    assert (
        "43 / 43 contraindicated · live cache" in demo_html
        or "8 / 8 contraindicated · live cache" in demo_html
    ), (
        "Hero stat chip footnote must show the live cache TP/total breakdown "
        "(post iter-275 v8 promotion: 43/43 = 100% precision over the live "
        "contra cohort)."
    )
