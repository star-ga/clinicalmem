"""Pin: `bitnet_classified` DEBUG event carries `ensemble_path` categorical.

iter-432 (round T4) — observability ratchet for the iter-421 cascade.
Pre-iter-432 the `bitnet_classified` event had only an `ensemble_active`
boolean which conflated 3 distinct dispatch states:

  | weights_b loaded? | A's prediction | ensemble_active (pre-432) | path category |
  |-------------------|----------------|---------------------------|---------------|
  | yes               | non-contra     | true                      | cascade_fired |
  | yes               | contra (sev=4) | false                     | a_only_contra_veto |
  | no                | any            | false                     | a_only_no_b |

Forensic readers had to reverse-engineer state (b) vs (c) from
`weights_id` length + presence of B in the on-disk bundle dir. This
pin locks the new categorical field so the 3 states are explicit
in the structured log stream.

Same shape as iter-311 runtime PHI scrub pin (LogRecord extras
inspection on the live classifier) but for observability completeness
rather than PHI discipline.

Cross-pin lineage: 38th cross-pin family
(iter-178/183/188/193/198/203/223/228/232/234/236/239/240/244/246/247
 + iter-255 + iter-260 + iter-279 + iter-281 + iter-285 + iter-286 +
 iter-295 + iter-296 + iter-301 + iter-304 + iter-306 + iter-311 +
 iter-314 + iter-319 + iter-324 + iter-329 + iter-366 + iter-374 +
 iter-390 + iter-425 + iter-428 + iter-432).
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from engine.bitnet_classifier import classify, load_weights, load_weights_b  # noqa: E402

_CONTRA_PAIR = ("clarithromycin", "simvastatin")  # contra (v8 BOOST anchor)
_SERIOUS_PAIR = ("warfarin", "aspirin")           # serious (B closes via cascade)


def _capture_classified_record(weights, weights_b, drug_a, drug_b):
    """Run classify() and return the LogRecord for the `bitnet_classified`
    event so test cases can inspect extras structurally."""

    captured: list[logging.LogRecord] = []

    class _Handler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            # record.msg is the raw event token ("bitnet_classified");
            # record.message is only set after getMessage() runs (which
            # the default emit path does — but in our capture handler
            # we never call format()/getMessage(), so use .msg).
            if record.msg == "bitnet_classified":
                captured.append(record)

    logger = logging.getLogger("engine.bitnet_classifier")
    handler = _Handler(level=logging.DEBUG)
    prior_level = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    try:
        classify(drug_a, drug_b, weights, weights_b=weights_b)
    finally:
        logger.removeHandler(handler)
        logger.setLevel(prior_level)
    assert captured, "bitnet_classified event was not emitted"
    return captured[-1]


def test_ensemble_path_cascade_fired_on_non_contra():
    """When B is loaded AND A predicts non-contra, the structured event
    carries `ensemble_path == 'cascade_fired'`."""
    weights_a = load_weights()
    weights_b = load_weights_b()
    assert weights_b is not None, "Bundle B must be present for this pin"

    record = _capture_classified_record(weights_a, weights_b, *_SERIOUS_PAIR)
    assert getattr(record, "ensemble_path", None) == "cascade_fired", (
        f"Expected ensemble_path=='cascade_fired' on cascade-firing pair "
        f"{_SERIOUS_PAIR}; got {getattr(record, 'ensemble_path', '<missing>')!r}. "
        f"ensemble_active was {getattr(record, 'ensemble_active', None)!r}."
    )
    assert getattr(record, "ensemble_active") is True, (
        "ensemble_active must remain True when cascade fires (backwards "
        "compat with pre-iter-432 parsers)"
    )


def test_ensemble_path_a_only_contra_veto():
    """When B is loaded AND A predicts contra, cascade is BYPASSED by
    the safety contract — `ensemble_path == 'a_only_contra_veto'`.
    `ensemble_active` stays False because B was not consulted."""
    weights_a = load_weights()
    weights_b = load_weights_b()
    assert weights_b is not None

    record = _capture_classified_record(weights_a, weights_b, *_CONTRA_PAIR)
    assert getattr(record, "ensemble_path", None) == "a_only_contra_veto", (
        f"Expected ensemble_path=='a_only_contra_veto' on contra pair "
        f"{_CONTRA_PAIR}; got {getattr(record, 'ensemble_path', '<missing>')!r}"
    )
    assert getattr(record, "ensemble_active") is False, (
        "ensemble_active must remain False on contra-veto path (B is "
        "bypassed; backwards compat with pre-iter-432 parsers)"
    )


def test_ensemble_path_a_only_no_b():
    """When B is NOT loaded (single-bundle mode), `ensemble_path ==
    'a_only_no_b'` regardless of A's prediction. Catches the drift
    class where someone deletes the B bundle and the engine silently
    degrades to A-only without the log stream surfacing it."""
    weights_a = load_weights()

    # Pass weights_b=None explicitly to exercise single-bundle mode
    record_serious = _capture_classified_record(weights_a, None, *_SERIOUS_PAIR)
    record_contra = _capture_classified_record(weights_a, None, *_CONTRA_PAIR)

    for record, pair in ((record_serious, _SERIOUS_PAIR), (record_contra, _CONTRA_PAIR)):
        assert getattr(record, "ensemble_path", None) == "a_only_no_b", (
            f"Expected ensemble_path=='a_only_no_b' under single-bundle mode "
            f"for {pair}; got {getattr(record, 'ensemble_path', '<missing>')!r}"
        )
        assert getattr(record, "ensemble_active") is False, (
            "ensemble_active must be False when B is not loaded"
        )


def test_ensemble_path_field_present_in_extras_source():
    """Source-level guard: the `ensemble_path` key must appear in the
    `bitnet_classified` extras dict in `engine/bitnet_classifier.py`.
    Catches the regression where someone refactors the dispatcher and
    accidentally drops the categorical field."""
    src = (_REPO_ROOT / "engine" / "bitnet_classifier.py").read_text()
    assert '"ensemble_path"' in src, (
        "engine/bitnet_classifier.py must include 'ensemble_path' as an "
        "extras key in the bitnet_classified event."
    )
    # Each of the 3 categorical values must be present in source
    for category in ("cascade_fired", "a_only_contra_veto", "a_only_no_b"):
        assert f'"{category}"' in src, (
            f"engine/bitnet_classifier.py must define the {category!r} "
            f"ensemble_path category."
        )
