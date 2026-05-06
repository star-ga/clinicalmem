"""Pin: JUDGES.md row 102 header recall fraction must match the live
v6 pin's `_V6_CONTRA_HITS / _V6_CONTRA_TOTAL`.

Iter 232 (round 48 T1) — self-lock for the recurring header-vs-body
drift class. Caught **5 times** in the iteration history:

  • iter-195: header said "31/36" but body narrative stopped at 31/34
    (cohort-growth ripple missed the row header)
  • iter-213: header v6 reframed but body still opened with v5 bundle
    path (header-vs-body bundle-path contradiction)
  • iter-218: 4 stale claims after iter-215 cohort growth + iter-216
    pin extension (recurring catch class)
  • iter-227: 4 deleted v5 pin file references in JUDGES rows 102 + 106
    survived the iter-215 cleanup
  • iter-231: header still claimed "38/38 full-recall breakthrough"
    while body said "39/40" (iter-230 cohort growth ripple missed
    the row header AGAIN)

The `_V6_CONTRA_HITS / _V6_CONTRA_TOTAL` constants in
`tests/test_engine/test_path_a_v6_live_recall_pin.py` are the single
source of truth for v6's recall claim — this pin asserts the JUDGES
row 102 header (the first thing a judge reads on the audit table)
quotes the same fraction, locking header ↔ pin agreement.

Same drift-prevention shape as the iter-178/183/188/193/198/203/223/228
cross-pins (9th cross-pin family in the discipline lineage).

Surface locked
==============

JUDGES.md row 102 header (`Path A v6 — ...`) must contain:
  1. The recall fraction `<HITS>/<TOTAL>` matching live v6 pin constants
  2. The major-class fraction `4/4` (still 100%)
  3. The "0 FP" zero-false-positive invariant claim
  4. The Q16.16 inference framing (since v6 is bit-identical Q16.16)

A future cohort growth that updates the v6 pin constants but forgets
to update the JUDGES header fires this pin. The header IS the
source-of-truth claim a judge reads first; if it lies, the audit
table loses credibility.
"""
from __future__ import annotations

import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_JUDGES = _REPO_ROOT / "JUDGES.md"
_V6_PIN = _REPO_ROOT / "tests" / "test_engine" / "test_path_a_v6_live_recall_pin.py"


def _live_v6_constants() -> dict[str, int]:
    """Extract _V6_CONTRA_HITS / _V6_CONTRA_TOTAL / _V6_FP_COUNT from
    the v6 pin source. Single source of truth: if the pin constants
    drift, this function reads the new values and the assertion below
    fires when JUDGES is stale."""
    text = _V6_PIN.read_text()
    out: dict[str, int] = {}
    for var in ("_V6_CONTRA_HITS", "_V6_CONTRA_TOTAL", "_V6_FP_COUNT",
                "_V6_MAJOR_HITS", "_V6_MAJOR_TOTAL"):
        m = re.search(rf"{var}\s*=\s*(\d+)", text)
        assert m is not None, (
            f"Could not find {var} in {_V6_PIN.relative_to(_REPO_ROOT)}. "
            f"This pin assumes _V6_*_HITS / _V6_*_TOTAL / _V6_FP_COUNT "
            f"are top-level int constants in the v6 live-recall pin."
        )
        out[var] = int(m.group(1))
    return out


def _v6_row_header() -> str:
    """Extract the JUDGES.md row 102 v6 header (first backtick-block
    after the row's pipe-prefix)."""
    text = _JUDGES.read_text()
    # Row format: `| `Path A v6 — ... XX/YY + ... 0 FP ...` | <body> |`
    # The header is in the FIRST backtick-block on the line.
    m = re.search(
        r"^\|\s*`(Path A v6 — [^`]+)`\s*\|",
        text,
        re.MULTILINE,
    )
    assert m is not None, (
        "JUDGES.md row 102 header not found. Expected a row starting "
        "with '| `Path A v6 — ...` |'. If the row was renamed/moved, "
        "update this pin to find the new header."
    )
    return m.group(1)


def test_v6_header_contra_fraction_matches_live_pin():
    """The header's recall fraction must equal the live v6 pin's
    _V6_CONTRA_HITS / _V6_CONTRA_TOTAL — locks the iter-195/213/231
    header-vs-body drift class."""
    consts = _live_v6_constants()
    expected = f"{consts['_V6_CONTRA_HITS']}/{consts['_V6_CONTRA_TOTAL']}"
    header = _v6_row_header()
    # Find ALL N/M fractions in the header. The contra fraction is the
    # one matching live_hits/live_total.
    fractions = re.findall(r"\b(\d+)/(\d+)\b", header)
    matched = [(n, m) for n, m in fractions
               if n == str(consts["_V6_CONTRA_HITS"])
               and m == str(consts["_V6_CONTRA_TOTAL"])]
    assert matched, (
        f"JUDGES.md row 102 header missing live v6 contra fraction "
        f"{expected!r}. Header found: {header!r}\n"
        f"Live constants from {_V6_PIN.relative_to(_REPO_ROOT)}: "
        f"_V6_CONTRA_HITS={consts['_V6_CONTRA_HITS']}, "
        f"_V6_CONTRA_TOTAL={consts['_V6_CONTRA_TOTAL']}.\n"
        f"This is the recurring iter-195/213/231 drift class — "
        f"cohort growth bumped the pin constants but the header still "
        f"quotes the OLD fraction. Update the JUDGES row 102 header "
        f"to match {expected!r}."
    )


def test_v6_header_major_fraction_matches_live_pin():
    """Major-class fraction (4/4) must appear in the header.

    Major recall is still 100% since the iter-93 4-pair major cohort
    landed; v6 catches all 4. If a future major-class miss lands, the
    pin constants update first, then this test fires until JUDGES
    catches up."""
    consts = _live_v6_constants()
    expected = f"{consts['_V6_MAJOR_HITS']}/{consts['_V6_MAJOR_TOTAL']}"
    header = _v6_row_header()
    fractions = re.findall(r"\b(\d+)/(\d+)\b", header)
    matched = [(n, m) for n, m in fractions
               if n == str(consts["_V6_MAJOR_HITS"])
               and m == str(consts["_V6_MAJOR_TOTAL"])]
    assert matched, (
        f"JUDGES.md row 102 header missing live v6 major fraction "
        f"{expected!r}. Header found: {header!r}\n"
        f"Live constants: _V6_MAJOR_HITS={consts['_V6_MAJOR_HITS']}, "
        f"_V6_MAJOR_TOTAL={consts['_V6_MAJOR_TOTAL']}."
    )


def test_v6_header_quotes_zero_fp_invariant():
    """The 0-false-positive invariant ("0 FP") is the load-bearing
    safety claim for v6 — must always appear in the header."""
    header = _v6_row_header()
    assert _live_v6_constants()["_V6_FP_COUNT"] == 0, (
        "v6 pin says zero FPs; if a FP lands the pin constants change "
        "first and this assertion fires until JUDGES updates to match."
    )
    assert re.search(r"0\s*FP\b", header), (
        f"JUDGES.md row 102 header must include the '0 FP' invariant "
        f"(zero false positives is the load-bearing safety claim — the "
        f"reason the engine accepts the high-precision-veto-by-design "
        f"recall band). Header found: {header!r}"
    )


def test_v6_header_quotes_q16_16_framing():
    """The header MUST surface the 'Q16.16' framing — v6's whole
    point is bit-identical Q16.16 inference for FDA SaMD audit
    replay. A header that drops Q16.16 silently turns v6 into a
    "just an accuracy number" claim, losing the FDA-relevance hook
    that distinguishes ClinicalMem from generic ML demos."""
    header = _v6_row_header()
    assert "Q16.16" in header, (
        f"JUDGES.md row 102 header must include 'Q16.16' to keep the "
        f"FDA-SaMD bit-identical-inference framing visible to judges. "
        f"Header found: {header!r}"
    )
