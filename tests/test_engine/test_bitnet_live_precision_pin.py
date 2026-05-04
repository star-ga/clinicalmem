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
    """Deployment-side BitNet precision/recall on contraindicated class."""
    preds = _bitnet_predictions()
    precision, recall, tp, total = _precision_recall(preds, "contraindicated")

    # Pinned values from iteration 29. Update both this test AND every
    # doc reference in the same commit when the numbers change.
    assert precision == 1.0, (
        f"Layer 4.5 deployment precision on contraindicated drifted: "
        f"live={precision:.4f}, pinned=1.0000"
    )
    # Recall is allowed to fluctuate within a tight band — Layer 4.5
    # is high-precision-by-design, not a recall maximizer.
    assert 0.30 <= recall <= 0.45, (
        f"Layer 4.5 deployment recall on contraindicated outside band: "
        f"live={recall:.4f}, allowed=[0.30, 0.45]"
    )
    assert tp == 6, f"true positives drifted: live={tp}, pinned=6"
    assert total == 20, f"contraindicated cohort size drifted: live={total}, pinned=20"


def test_dashboard_displays_live_precision_number():
    """Dashboard's hero precision card shows the live 100% number."""
    demo_html = (_REPO_ROOT / "docs" / "demo.html").read_text()
    assert "Layer 4.5 · live precision" in demo_html, (
        "Hero stat chip must label the precision number 'live precision' "
        "(not the bare '85.7% precision on contraindicated' which conflated "
        "held-out test-set accuracy with deployment precision)."
    )
    # Must contain the 100% live precision claim
    assert "6 / 6 contraindicated · live cache" in demo_html, (
        "Hero stat chip footnote must show the live cache TP/total breakdown"
    )
