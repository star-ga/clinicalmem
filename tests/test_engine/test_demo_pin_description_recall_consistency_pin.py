"""Pin: every recall-fraction phrase inside `docs/demo.html` pin
description text must match the live v8 pin constants.

Iter 301 (round 63 T1) — 25th cross-pin family in the discipline
lineage. Same iter-232 single-source-of-truth → derived-surface drift
class as `test_judges_v8_header_recall_consistency_pin.py`, but applied
one tier deeper: the *inline pin-description text inside demo.html*
that describes the v8 pin's coverage. Caught at iter-301 audit (3
stale 43/43 + 1 stale "42 contraindicated" survived the iter-280
cohort growth from 43 → 44):

  • L1270  "(aggregate: bundle_id + 43/43 + ...)"          ← stale
  • L1357  "bundle_id + 43/43 contra + 4/4 major + ..."    ← stale
  • L1928  "139-pair recall cohort (42 contraindicated ..." ← stale

Same drift class as iter-298 (README L136 + JUDGES row 102 stale
43/43) but at the demo.html surface layer. iter-298 fixed the prose;
this pin prevents regression at the third surface.

Surfaces locked
===============
1. The "(aggregate: ... + N/N + 4/4 + 0 FP + strictly_supersedes invariant)"
   phrase wherever it appears (currently L1270).
2. The "bundle_id + N/N contra + 4/4 major + 0 FP + meta-block invariants"
   phrase wherever it appears (currently L1357 — iter-275 promotion
   pin description).
3. The "139-pair recall cohort (X contraindicated · 4 major · 22
   moderate · 69 serious)" cohort breakdown (L1928) — every count
   must match the live severity distribution in
   `docs/openevidence_cache.json`.

Single source of truth: `_V8_CONTRA_HITS` / `_V8_CONTRA_TOTAL` in
`tests/test_engine/test_path_a_v8_live_recall_pin.py` for #1 and #2;
`docs/openevidence_cache.json` Counter for #3.

If a future cohort-growth iteration updates the pin constants but
forgets the demo.html prose, this pin fires before judges see stale
numbers in the page that decides their 5-second impression.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_DEMO = _REPO_ROOT / "docs" / "demo.html"
_V8_PIN = _REPO_ROOT / "tests" / "test_engine" / "test_path_a_v8_live_recall_pin.py"
_CACHE = _REPO_ROOT / "docs" / "openevidence_cache.json"


def _live_v8_constants() -> dict[str, int]:
    """Extract _V8_CONTRA_HITS / _V8_CONTRA_TOTAL from the v8 pin source.

    Mirror of `test_judges_v8_header_recall_consistency_pin._live_v6_constants`
    pattern — single source of truth at the constants layer."""
    text = _V8_PIN.read_text()
    out: dict[str, int] = {}
    for var in ("_V8_CONTRA_HITS", "_V8_CONTRA_TOTAL"):
        m = re.search(rf"{var}\s*=\s*(\d+)", text)
        assert m is not None, (
            f"Could not find {var} in {_V8_PIN.relative_to(_REPO_ROOT)}. "
            f"This pin assumes _V8_CONTRA_HITS / _V8_CONTRA_TOTAL are "
            f"top-level int constants in the v8 live-recall pin."
        )
        out[var] = int(m.group(1))
    return out


def _live_severity_counts() -> dict[str, int]:
    """Live severity distribution from openevidence_cache.json."""
    data = json.loads(_CACHE.read_text())
    return dict(Counter(e.get("severity", "") for e in data))


def test_demo_aggregate_pin_description_quotes_live_contra_fraction():
    """The "(aggregate: bundle_id + N/N + 4/4 + 0 FP + strictly_supersedes
    invariant)" phrase in demo.html must use the live N/N. Catches
    iter-301-class drift (43/43 stale post iter-280 cohort growth)."""
    consts = _live_v8_constants()
    expected = f"{consts['_V8_CONTRA_HITS']}/{consts['_V8_CONTRA_TOTAL']}"
    text = _DEMO.read_text()
    # Match the aggregate phrase pattern: anything with "bundle_id" +
    # one or more N/N fractions + "+ 0 FP" + "strictly_supersedes" within
    # ~200 chars. Accepts both the legacy "aggregate: bundle_id + N/N + 4/4
    # + 0 FP + strictly_supersedes" form and the cleanup-2026-05-11 form
    # "bundle_id + N/N contra + 4/4 major + 0 FP + meta-block + strictly_supersedes".
    matches = re.findall(
        r"bundle_id\s*\+\s*(\d+/\d+)(?:\s+contra)?\s*\+\s*4/4(?:\s+major)?\s*\+\s*0 FP(?:\s*\+\s*meta-block)?\s*\+\s*strictly_supersedes",
        text,
    )
    assert matches, (
        f"demo.html missing the canonical pin-description aggregate phrase. "
        f"Expected at least one occurrence of "
        f"'bundle_id + {expected} contra + 4/4 major + 0 FP + meta-block + strictly_supersedes' "
        f"(or the legacy 'aggregate: bundle_id + {expected} + 4/4 + 0 FP + strictly_supersedes'). "
        f"If the wording was rephrased again, update this pin to track the new form."
    )
    stale = [m for m in matches if m != expected]
    assert not stale, (
        f"demo.html has stale aggregate-phrase contra fractions: {stale!r}. "
        f"Live v8 pin says {expected!r}.\n"
        f"This is the iter-280/iter-298/iter-301 drift class — cohort "
        f"growth bumped the pin constants but the demo.html prose still "
        f"quotes the OLD fraction. Update every aggregate phrase to "
        f"{expected!r}."
    )


def test_demo_promotion_pin_description_quotes_live_contra_fraction():
    """The "bundle_id + N/N contra + 4/4 major + 0 FP + meta-block invariants"
    phrase (iter-275 v8 promotion pin description) must use the live N/N."""
    consts = _live_v8_constants()
    expected = f"{consts['_V8_CONTRA_HITS']}/{consts['_V8_CONTRA_TOTAL']}"
    text = _DEMO.read_text()
    matches = re.findall(
        r"bundle_id\s*\+\s*(\d+/\d+)\s+contra\s*\+\s*4/4\s+major\s*\+\s*0 FP\s*\+\s*meta-block",
        text,
    )
    assert matches, (
        f"demo.html missing the canonical promotion pin-description phrase. "
        f"Expected at least one occurrence of "
        f"'bundle_id + {expected} contra + 4/4 major + 0 FP + meta-block'. "
        f"If the wording was rephrased, update this pin to track the new form."
    )
    stale = [m for m in matches if m != expected]
    assert not stale, (
        f"demo.html has stale promotion-phrase contra fractions: {stale!r}. "
        f"Live v8 pin says {expected!r}. Update every occurrence."
    )


def test_demo_cohort_breakdown_matches_live_severity_distribution():
    """The "139-pair recall cohort (X contraindicated · Y major ·
    Z moderate · W serious)" line must match the live cache severity
    distribution. Catches iter-301-class stale "42 contraindicated"
    drift after iter-280 cohort growth."""
    live = _live_severity_counts()
    text = _DEMO.read_text()
    # Allow HTML tags (e.g. </strong>) between "N-pair recall cohort" and
    # the parenthesised breakdown. The breakdown itself is plain-text inside
    # one tag; <[^>]*> matches a single HTML tag boundary.
    m = re.search(
        r"(\d+)-pair recall cohort(?:</[^>]+>)?\s*\(\s*(\d+)\s+contraindicated"
        r"\s*·\s*(\d+)\s+major\s*·\s*(\d+)\s+moderate\s*·\s*(\d+)\s+serious"
        r"\s*\)",
        text,
    )
    assert m is not None, (
        "demo.html missing the canonical cohort-breakdown phrase "
        "'<N>-pair recall cohort (<C> contraindicated · <M> major · "
        "<O> moderate · <S> serious)'. If the wording was rephrased, "
        "update this pin to track the new form."
    )
    total, contra, major, moderate, serious = (int(g) for g in m.groups())
    expected_total = sum(live.values())
    assert total == expected_total, (
        f"demo.html cohort-breakdown total {total} != live {expected_total}. "
        f"Update the breakdown phrase."
    )
    expected = (
        live.get("contraindicated", 0),
        live.get("major", 0),
        live.get("moderate", 0),
        live.get("serious", 0),
    )
    actual = (contra, major, moderate, serious)
    assert actual == expected, (
        f"demo.html cohort-breakdown distribution {actual} (contra/major/"
        f"moderate/serious) != live {expected}. Update the phrase to "
        f"'{expected_total}-pair recall cohort ({expected[0]} contraindicated "
        f"· {expected[1]} major · {expected[2]} moderate · {expected[3]} "
        f"serious)'."
    )


def test_demo_no_pre_iter280_contra_count_remains():
    """Forbid pre-iter-280 contra counts (42, 43) from re-appearing in
    demo.html as recall fractions. Catches the half-completed-rotation
    drift class (where some surfaces are updated but others lag)."""
    consts = _live_v8_constants()
    live = consts["_V8_CONTRA_TOTAL"]
    text = _DEMO.read_text()
    # Look for stale fractions in pin-description contexts: "N/N + 4/4"
    # or "N/N contra" patterns where N is 42 or 43 (pre-iter-280).
    stale_aggregate = re.findall(r"\b(4[23])/(4[23])\s*\+\s*4/4", text)
    assert not stale_aggregate, (
        f"demo.html has stale pre-iter-280 contra fractions in aggregate "
        f"phrases: {stale_aggregate!r}. Live v8 contra total is {live}. "
        f"These are the iter-280-cohort-growth fossils that iter-301 swept."
    )
    stale_promotion = re.findall(r"\b(4[23])/(4[23])\s+contra\b", text)
    assert not stale_promotion, (
        f"demo.html has stale pre-iter-280 contra fractions in promotion "
        f"phrases: {stale_promotion!r}. Live v8 contra total is {live}."
    )
