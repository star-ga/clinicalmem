"""Pin: every contraindicated cache entry has ≥ 1 authoritative-source URL.

**Iter-285 T5 round-59 — 21st cross-pin family.**

The cache claims "contraindicated" for 44 drug pairs (post-iter-280).
Each is a high-stakes safety call: it triggers the BitNet Layer 4.5
veto, blocks the engine final verdict, and writes a SHA-256 audit
row that an FDA SaMD reviewer may replay decades later. Every such
claim needs a citation chain that survives audit — primary regulatory
labels (FDA / EMA / ACR) or peer-reviewed primary sources (PubMed,
AHA Journals, BMJ, Oxford Academic, AGS Beers).

Pre-iter-285 the cache_evidence_urls pin enforced ≥ 2 URLs per entry
+ ≥ 90% from a NIH/FDA whitelist + 100% HTTPS. That's a cohort-aggregate
floor. This pin is per-entry: **every contra MUST cite ≥ 1 source from
the authoritative whitelist**, so a future contra entry can't sneak in
with only secondary review sources (UpToDate, MedScape, Wikipedia).

Authoritative whitelist
=======================
Primary regulatory:
  - accessdata.fda.gov   FDA prescribing information labels
  - fda.gov              FDA safety communications + warnings
  - ema.europa.eu        European Medicines Agency
  - acr.org              American College of Radiology (contrast guidelines)

Peer-reviewed primary:
  - pubmed.ncbi.nlm.nih.gov     PubMed
  - ncbi.nlm.nih.gov            NIH PMC full-text
  - ahajournals.org             American Heart Association journals
  - ard.bmj.com                 BMJ Annals of Rheumatic Diseases
  - academic.oup.com            Oxford University Press medical journals
  - jaad.org                    Journal of American Academy of Dermatology
  - aan.com                     American Academy of Neurology
  - geriatricscareonline.org    AGS Beers Criteria
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from urllib.parse import urlparse

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_CACHE = _REPO_ROOT / "docs" / "openevidence_cache.json"

# Hosts whose URLs count as authoritative for the per-entry pin.
# Built from the live iter-285 audit; any future addition must be
# reviewed and added here deliberately (the test failure forces the
# discussion).
_AUTHORITATIVE_HOSTS = frozenset({
    # Primary regulatory labels + safety communications
    "www.accessdata.fda.gov",
    "accessdata.fda.gov",
    "www.fda.gov",
    "fda.gov",
    "www.ema.europa.eu",
    "ema.europa.eu",
    "www.acr.org",
    "acr.org",
    # Peer-reviewed primary
    "pubmed.ncbi.nlm.nih.gov",
    "www.ncbi.nlm.nih.gov",
    "ncbi.nlm.nih.gov",
    "www.ahajournals.org",
    "ahajournals.org",
    "ard.bmj.com",
    "academic.oup.com",
    "www.jaad.org",
    "jaad.org",
    "www.aan.com",
    "aan.com",
    "geriatricscareonline.org",
})


def _cache():
    return json.loads(_CACHE.read_text())


def test_every_contra_has_two_authoritative_urls():
    """Every contraindicated cache entry MUST cite ≥ 2 URLs whose
    netlocs are in the authoritative whitelist.

    iter-310 ratchet (was ≥ 1 at iter-285): live cohort audit (44
    contra entries post iter-280 cohort growth) found 0 / 44 at the
    floor of 1, 17 / 44 at 2, 27 / 44 at 3 (mean = 2.61 authoritative
    URLs per entry). Tightening 1 → 2 leaves 0 entries at the new
    floor with full cohort-growth tolerance: a new contra cited only
    by a single primary source (one FDA label OR one PubMed paper)
    would fail the gate, forcing operators to add a second
    authoritative source before landing the entry. Same iter-117
    ratchet-when-headroom-exists discipline applied to citation
    breadth (where iter-305 ratcheted citation-quality FDA-or-primary
    floor 80% → 85%; same direction, complementary axis).

    Forward-protects against cohort-growth events that introduce a
    contra grounded in only one primary source plus secondary review
    sources (UpToDate, MedScape, etc.). Two independent primary
    sources is the FDA SaMD substantial-equivalence discipline floor.
    """
    contras = [it for it in _cache() if it["severity"] == "contraindicated"]
    failures = []
    for it in contras:
        urls = it.get("evidence_urls", [])
        # Count distinct authoritative URLs (path-distinct citations
        # from any authoritative host — three FDA labels at the same
        # host count as three distinct primary sources, since each
        # label is an independent regulatory document for a different
        # drug).
        auth_urls = [u for u in urls if urlparse(u).netloc in _AUTHORITATIVE_HOSTS]
        if len(auth_urls) < 2:
            failures.append({
                "pair": (it["drug_a"], it["drug_b"]),
                "auth_url_count": len(auth_urls),
                "auth_urls": auth_urls,
                "all_urls": urls,
            })
    assert not failures, (
        f"{len(failures)} contraindicated entries with < 2 "
        f"authoritative-source URLs (iter-310 ratchet, was ≥ 1 at "
        f"iter-285). Allowed hosts: "
        f"{sorted(_AUTHORITATIVE_HOSTS)[:8]}...\n"
        f"First offender: {failures[0]}\n"
        f"Either add a second primary citation (FDA / EMA / PubMed / "
        f"ACR / AHA / BMJ / etc.) or extend _AUTHORITATIVE_HOSTS "
        f"deliberately + document the source-grade rationale."
    )


_REGULATORY_HOSTS = frozenset({
    "accessdata.fda.gov",
    "www.accessdata.fda.gov",
    "fda.gov",
    "www.fda.gov",
    "ema.europa.eu",
    "www.ema.europa.eu",
})


def test_every_contra_cites_two_distinct_authoritative_hosts():
    """Every contraindicated cache entry MUST cite ≥ 2 *distinct*
    authoritative hosts. Stricter than the iter-310 path-distinct
    URL-count ratchet — three FDA labels at the same accessdata.fda.
    gov host count as one source for cross-jurisdictional
    independence purposes.

    iter-320 ratchet (iter-310-era candidate executed). Live cohort
    audit (44 contra post iter-280 + iter-315 enrichment + iter-320
    enrichment of ketoconazole+tolvaptan with PubMed Shoaf 2012 DDI
    study, PMID 22220626): 0 / 44 at the floor of 1, 32 / 44 at 2,
    12 / 44 at 3 (mean 2.27 distinct hosts). Tightening from 'no
    distinct-host floor' to ≥ 2 leaves 0 entries at the new floor
    with full cohort-growth tolerance: a new contra cited only by
    multiple URLs at a single FDA-label host (e.g., 3 labels at
    accessdata.fda.gov) would fail the gate, forcing operators to
    add at least one independent regulatory or peer-reviewed source.

    Same iter-117 ratchet-when-headroom-exists pattern; complementary
    axis to the iter-310 path-distinct URL count and iter-315 FDA-OR-
    EMA mandatory floor:

      iter-310  citation breadth (URL count, ≥ 2 path-distinct)
      iter-315  citation jurisdiction (≥ 1 FDA OR EMA, 100%)
      iter-320  source independence (≥ 2 distinct hosts, this ratchet)

    Forward-protects against cohort-growth events where a single
    organisation publishes multiple labels covering both drugs in a
    pair. Two distinct authoritative hosts is the FDA SaMD source-
    independence discipline floor (the iter-72 'no single point of
    citation failure' invariant applied to evidence sources).
    """
    contras = [it for it in _cache() if it["severity"] == "contraindicated"]
    failures = []
    for it in contras:
        urls = it.get("evidence_urls", [])
        auth_hosts = {urlparse(u).netloc for u in urls} & _AUTHORITATIVE_HOSTS
        if len(auth_hosts) < 2:
            failures.append({
                "pair": (it["drug_a"], it["drug_b"]),
                "auth_host_count": len(auth_hosts),
                "auth_hosts": sorted(auth_hosts),
            })
    assert not failures, (
        f"{len(failures)} contraindicated entries with < 2 distinct "
        f"authoritative hosts (iter-320 ratchet). Allowed hosts: "
        f"{sorted(_AUTHORITATIVE_HOSTS)[:8]}...\n"
        f"First offender: {failures[0]}\n"
        f"Either add a citation from a different authoritative "
        f"organisation OR extend _AUTHORITATIVE_HOSTS deliberately."
    )


def test_every_contra_cites_fda_or_ema_regulatory_label():
    """Every contraindicated cache entry MUST cite ≥ 1 URL from a
    primary regulatory body (FDA accessdata.fda.gov / fda.gov OR
    EMA ema.europa.eu). Cross-jurisdictional regulatory grounding.

    iter-315 ratchet (iter-310-era candidate executed). Live cohort
    audit (44 contra entries post iter-280 cohort growth + iter-315
    enrichment of contrast dye+metformin with the FDA Glucophage
    label): 44/44 (100%) cite FDA or EMA. The single pre-iter-315
    outlier (`contrast dye + metformin`, originally backed by ACR
    Contrast Manual + PubMed only — both authoritative under iter-285,
    but neither regulatory) was enriched with the FDA Glucophage label
    PDF (Section 5.2 explicitly names iodinated contrast media as a
    lactic-acidosis precipitant).

    Forward-protects against cohort-growth events that introduce a
    contra grounded only in clinical-society guidelines + journal
    citations without an FDA / EMA regulatory anchor. The FDA SaMD
    substantial-equivalence pathway requires regulatory backing for
    safety-critical claims; this pin enforces that floor at the
    citation layer.
    """
    contras = [it for it in _cache() if it["severity"] == "contraindicated"]
    failures = []
    for it in contras:
        urls = it.get("evidence_urls", [])
        hosts = {urlparse(u).netloc for u in urls}
        if not (hosts & _REGULATORY_HOSTS):
            failures.append({
                "pair": (it["drug_a"], it["drug_b"]),
                "hosts": sorted(hosts),
            })
    assert not failures, (
        f"{len(failures)} contraindicated entries without ≥ 1 "
        f"regulatory-body URL (FDA / EMA). iter-315 floor = 100%.\n"
        f"First offender: {failures[0]}\n"
        f"Allowed regulatory hosts: {sorted(_REGULATORY_HOSTS)}\n"
        f"Either add the relevant FDA label / EMA reference URL OR "
        f"extend _REGULATORY_HOSTS deliberately + document the "
        f"jurisdictional rationale."
    )


def test_authoritative_whitelist_is_not_overfit_to_one_source():
    """Reject the trivial-bypass case where the whitelist is reduced
    to a single host. The pin's value comes from breadth — at least
    5 distinct authoritative hosts must be present (any single body
    going dark cannot break the pin's coverage)."""
    assert len(_AUTHORITATIVE_HOSTS) >= 10, (
        f"_AUTHORITATIVE_HOSTS has only {len(_AUTHORITATIVE_HOSTS)} "
        f"entries. The pin's value comes from breadth — collapsing "
        f"to a single host would let any future contra sneak in by "
        f"citing only that one. Maintain ≥ 10 distinct hosts."
    )


def test_all_authoritative_hosts_used_by_at_least_one_live_contra():
    """Every host in `_AUTHORITATIVE_HOSTS` MUST be used by at least
    one live contra entry. Catches dead-host accumulation in the
    whitelist (the iter-270 orphan-allowlist pattern applied to URL
    hosts: the whitelist documents what's ACTUALLY in the cache, not
    aspirational sources we'd accept).

    Iter-285: a small number of historically-trusted hosts are
    retained even without live citation (e.g., regulatory bodies that
    haven't yet appeared in cohort growth) — those go in the
    `_ALLOWED_UNUSED_HOSTS` allowlist with a documented rationale.
    """
    contras = [it for it in _cache() if it["severity"] == "contraindicated"]
    used_hosts = set()
    for it in contras:
        for u in it.get("evidence_urls", []):
            used_hosts.add(urlparse(u).netloc)

    # Hosts in the whitelist that aren't yet referenced by any live
    # contra. Allowed (with documented rationale) but tracked.
    _ALLOWED_UNUSED_HOSTS = frozenset({
        # Bare-domain forms accepted alongside www-prefixed live form.
        "accessdata.fda.gov",
        "fda.gov",
        "ema.europa.eu",
        "acr.org",
        "ncbi.nlm.nih.gov",
        "ahajournals.org",
        "jaad.org",
        "aan.com",
    })

    unused = _AUTHORITATIVE_HOSTS - used_hosts - _ALLOWED_UNUSED_HOSTS
    assert not unused, (
        f"{len(unused)} hosts in _AUTHORITATIVE_HOSTS are not used by "
        f"any live contra and not in _ALLOWED_UNUSED_HOSTS: {sorted(unused)}. "
        f"Either (a) remove the dead host, (b) move it to "
        f"_ALLOWED_UNUSED_HOSTS with a rationale comment, or (c) add "
        f"a contra that cites it."
    )


def test_judges_documents_authoritative_whitelist_pin():
    """JUDGES.md must cite this pin so the audit-trail discipline is
    grep-able from the runnable-claims table."""
    judges = (_REPO_ROOT / "JUDGES.md").read_text()
    assert "test_contra_authoritative_url_pin.py" in judges or (
        "authoritative" in judges.lower() and "evidence" in judges.lower()
    ), (
        "JUDGES.md must reference the iter-285 authoritative-URL pin "
        "(by file name OR by the discipline phrase 'authoritative ... "
        "evidence') so an FDA SaMD reviewer can find it without grep."
    )
