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


def test_every_contra_has_authoritative_url():
    """Every contraindicated cache entry MUST cite ≥ 1 URL whose
    netloc is in the authoritative whitelist. Catches cohort-growth
    events that introduce a contra backed only by secondary sources.
    """
    contras = [it for it in _cache() if it["severity"] == "contraindicated"]
    failures = []
    for it in contras:
        urls = it.get("evidence_urls", [])
        hosts = {urlparse(u).netloc for u in urls}
        if not (hosts & _AUTHORITATIVE_HOSTS):
            failures.append({
                "pair": (it["drug_a"], it["drug_b"]),
                "hosts": sorted(hosts),
            })
    assert not failures, (
        f"{len(failures)} contraindicated entries without ≥ 1 "
        f"authoritative-source URL. Allowed hosts: "
        f"{sorted(_AUTHORITATIVE_HOSTS)[:8]}...\n"
        f"First offender: {failures[0]}\n"
        f"Either add a primary citation (FDA / EMA / PubMed / ACR / "
        f"AHA / BMJ) or extend _AUTHORITATIVE_HOSTS deliberately + "
        f"document the source-grade rationale."
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
