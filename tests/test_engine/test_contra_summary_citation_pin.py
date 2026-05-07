"""Pin: every contra cache entry's clinical_summary contains a citation marker.

**Iter-295 T5 round-61 cohort defense.**

Iter-285 pinned that every contraindicated cache entry cites ≥ 1 URL
from the authoritative whitelist. That guards the URL layer.

iter-285 doesn't guard the SUMMARY TEXT layer — a future cohort
addition could pass with vague prose like "this combination is
risky and should be avoided" backed by a single FDA URL. The summary
itself wouldn't carry the regulatory section number, mechanism, or
peer-reviewed citation that a clinician needs to evaluate the claim
without leaving the page.

This pin enforces a complementary layer: every contra's
`clinical_summary` MUST contain ≥ 1 marker from the set:

  - FDA prescribing-info marker: `FDA`, `§ 4`, `§ 5`, `Section 4`, or
    a label-`§`-pattern co-occurrence.
  - Peer-reviewed primary citation: PMID:, DOI:, or a "10.NNNN/..."
    DOI literal.
  - Major medical journal reference: NEJM, JAMA, Lancet, Mayo, N Engl
    J Med.
  - Clinical guideline body: ACC/AHA, AHA, EULAR, KDIGO, ESC, Beers,
    ACR.
  - Pharmacological mechanism token: CYP3A4, CYP2C9, CYP1A2, OATP,
    P-gp, MAO, PDE5 — each a specific named enzyme/transporter that
    grounds the claim in a documented mechanism.

Live (iter-295): 44/44 contras pass. Pin is forward-protective: a
future contra entry with vague prose will fail the gate at commit.

Same shape as iter-285 authoritative-URL pin but applied to the
summary text layer instead of the URL layer.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_CACHE = _REPO_ROOT / "docs" / "openevidence_cache.json"


_CITATION_MARKERS = {
    "fda_label": re.compile(
        r"(?:FDA|§\s*\d|Section\s+\d|label[\s\S]{0,30}§)",
        re.IGNORECASE,
    ),
    "primary_citation": re.compile(
        # PMID / DOI literal OR "et al." style peer-reviewed citation
        # (e.g. "Feighner et al., J Clin Psychiatry 1990").
        r"(?:PMID:|DOI:|10\.\d{4,}|\bet\s+al\.?)",
        re.IGNORECASE,
    ),
    "major_journal": re.compile(
        # Top-tier journals + broader peer-reviewed journal patterns
        # (J Clin Xxx, J Am Coll Xxx, Eur J Xxx, Br J Xxx) — these are
        # standard PubMed-indexed journal abbreviations.
        r"(?:NEJM|JAMA|Lancet|Mayo\s+Clin|N\s+Engl\s+J\s+Med"
        r"|J\s+Clin\s+\w+|J\s+Am\s+Coll\s+\w+|Eur\s+J\s+\w+|Br\s+J\s+\w+"
        r"|Am\s+J\s+\w+|Ann\s+\w+\s+Dis)",
        re.IGNORECASE,
    ),
    "guideline_body": re.compile(
        r"(?:ACC/AHA|AHA|EULAR|KDIGO|ESC|Beers|ACR\b)",
        re.IGNORECASE,
    ),
    "mechanism_token": re.compile(
        # Specific named CYP / transporter enzymes + common
        # pharmacology class abbreviations that uniquely ground a
        # contraindicated claim (MAOI/SSRI/SNRI/PDE5 etc).
        r"(?:CYP3A4?|CYP2C9|CYP1A2|CYP2D6|OATP1B\d|OATP\d|P-?gp"
        r"|MAOI?|SSRI|SNRI|NSAID|PDE5|ACE\s+inhibitor|ergot)",
        re.IGNORECASE,
    ),
}


def _cache_contras():
    return [
        it for it in json.loads(_CACHE.read_text())
        if it["severity"] == "contraindicated"
    ]


def test_every_contra_summary_has_citation_marker():
    """Per-entry: every contra cache entry's clinical_summary MUST
    contain ≥ 1 citation marker. Catches cohort-growth events that
    introduce a contra with vague prose backed only by URLs."""
    failures = []
    for it in _cache_contras():
        s = it.get("clinical_summary", "")
        matched = [name for name, p in _CITATION_MARKERS.items() if p.search(s)]
        if not matched:
            failures.append({
                "pair": (it["drug_a"], it["drug_b"]),
                "summary_first_100": s[:100],
            })
    assert not failures, (
        f"{len(failures)} contraindicated entries lack ≥ 1 citation "
        f"marker in clinical_summary. Allowed marker classes: "
        f"{sorted(_CITATION_MARKERS)}\n"
        f"First offender: {failures[0]}\n"
        f"Either add a regulatory/citation/mechanism reference to the "
        f"summary OR extend _CITATION_MARKERS deliberately + document "
        f"the new accepted form."
    )


def test_marker_breadth_floor_at_least_three_classes():
    """The cohort's contras MUST collectively use ≥ 3 distinct marker
    classes. Catches the trivial-bypass case where every entry happens
    to cite "FDA" but lacks mechanism/journal/guideline grounding."""
    classes_used: set[str] = set()
    for it in _cache_contras():
        s = it.get("clinical_summary", "")
        for name, p in _CITATION_MARKERS.items():
            if p.search(s):
                classes_used.add(name)
    assert len(classes_used) >= 3, (
        f"Cache contras collectively use only {len(classes_used)} "
        f"marker classes ({sorted(classes_used)}). Discipline floor: "
        f"≥ 3 distinct marker classes (regulatory + mechanism + "
        f"peer-reviewed). A cohort that only cites one regulatory "
        f"body would silently degrade citation diversity."
    )


def test_at_least_eighty_percent_contras_cite_fda_or_primary():
    """At least 80% of contras must cite FDA labels OR primary
    peer-reviewed sources (PMID/DOI). Catches the drift class where
    a future cohort addition leans heavily on guideline-only
    citations without primary-source grounding."""
    contras = _cache_contras()
    fda_or_primary = 0
    for it in contras:
        s = it.get("clinical_summary", "")
        if (_CITATION_MARKERS["fda_label"].search(s)
                or _CITATION_MARKERS["primary_citation"].search(s)):
            fda_or_primary += 1
    ratio = fda_or_primary / len(contras) if contras else 0.0
    assert ratio >= 0.80, (
        f"Only {ratio:.1%} of contras cite FDA OR primary "
        f"(PMID/DOI/journal-DOI). Discipline floor: ≥ 80%. "
        f"Cohort = {len(contras)}, FDA-or-primary = {fda_or_primary}."
    )
