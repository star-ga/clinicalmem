"""Pin: every OpenEvidence cache entry has well-formed evidence URLs.

Iter 66 makes the evidence backbone testable: an FDA SaMD reviewer
expects every drug-pair classification to point at primary-literature
or regulatory citations. This test asserts structural well-formedness
without making any network calls (CI-safe).

What's pinned:
  • Every cache entry has ≥ 2 evidence URLs (ratcheted iter-264 from ≥ 1
    once live distribution showed min=2, median=2, max=3 across all
    138 entries — FDA SaMD cross-citation discipline).
  • Every URL is HTTPS (never plain http: — auditors must be able to
    reproduce the citation later, and HTTP-without-TLS is unstable).
  • Every URL has a non-empty host AND a non-empty path (catches
    malformed entries like "https://" or "https://example.com").
  • No duplicate URLs within a single entry.
  • Host distribution is documented (auditor can see the source
    breakdown without re-counting): ≥ 90% of URLs come from the
    NIH/FDA/peer-review whitelist (NCBI, accessdata.fda.gov, PubMed,
    NEJM, AHA, BMJ, JAMA, etc.). Soft floor — if the corpus widens
    beyond this set, update the whitelist; never lower the floor.
"""
import json
from pathlib import Path
from urllib.parse import urlparse

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_CACHE = _REPO_ROOT / "docs" / "openevidence_cache.json"


# Whitelisted source families (authoritative regulatory + peer-review).
# Soft check — if a new authoritative source lands, add it here.
_AUTHORITATIVE_HOSTS = frozenset({
    "www.ncbi.nlm.nih.gov",
    "ncbi.nlm.nih.gov",
    "pubmed.ncbi.nlm.nih.gov",
    "www.accessdata.fda.gov",
    "accessdata.fda.gov",
    "www.fda.gov",
    "fda.gov",
    "www.nejm.org",
    "nejm.org",
    "www.ahajournals.org",
    "ahajournals.org",
    "www.acr.org",
    "acr.org",
    "academic.oup.com",
    "agsjournals.onlinelibrary.wiley.com",
    "ard.bmj.com",
    "bmj.com",
    "www.bmj.com",
    "gut.bmj.com",
    "www.aafp.org",
    "www.jaad.org",
    "www.aan.com",
    "www.cdc.gov",
    "cdc.gov",
    "kdigo.org",
    "www.kdigo.org",
    "ascopubs.org",
    "www.escardio.org",
    "academic.oup.com",
    "jamanetwork.com",
    "www.jamanetwork.com",
})


def _load_cache():
    return json.loads(_CACHE.read_text())


def test_every_entry_has_at_least_two_evidence_urls():
    """Every cache entry MUST have ≥ 2 evidence URLs.

    iter-264 ratchet (was ≥ 1 at iter-66): live distribution shows
    min=2 across all 138 cache entries (median=2, max=3). The original
    1-URL floor was set when the cache was young; tightening to 2
    forces any future stub-evidence entry to fail the gate visibly.
    Same iter-117 / iter-259 pattern (ratchet floors once enough
    headroom exists). FDA SaMD audit-trail rigor benefits from
    cross-citation: a single URL is a single-point-of-failure if the
    citation rots; two independent URLs (typically FDA label + PubMed)
    survive one-link rot.
    """
    cache = _load_cache()
    for entry in cache:
        urls = entry.get("evidence_urls", [])
        pair = f"{entry.get('drug_a', '?')} + {entry.get('drug_b', '?')}"
        assert len(urls) >= 2, (
            f"Cache entry {pair} has only {len(urls)} evidence_urls; "
            f"iter-264 floor is 2 (FDA SaMD cross-citation discipline)"
        )


def test_every_url_is_https():
    cache = _load_cache()
    for entry in cache:
        pair = f"{entry.get('drug_a', '?')} + {entry.get('drug_b', '?')}"
        for url in entry.get("evidence_urls", []):
            assert url.startswith("https://"), (
                f"{pair}: non-HTTPS URL {url!r} — auditors must be able "
                f"to reproduce the citation; HTTP-without-TLS is unstable."
            )


def test_every_url_has_host_and_path():
    cache = _load_cache()
    for entry in cache:
        pair = f"{entry.get('drug_a', '?')} + {entry.get('drug_b', '?')}"
        for url in entry.get("evidence_urls", []):
            parsed = urlparse(url)
            assert parsed.netloc, f"{pair}: URL {url!r} has no host"
            assert parsed.path and parsed.path != "/", (
                f"{pair}: URL {url!r} has no path beyond root — likely truncated"
            )


def test_no_duplicate_urls_within_entry():
    cache = _load_cache()
    for entry in cache:
        pair = f"{entry.get('drug_a', '?')} + {entry.get('drug_b', '?')}"
        urls = entry.get("evidence_urls", [])
        assert len(urls) == len(set(urls)), (
            f"{pair}: duplicate evidence_urls — entry has {len(urls)} but "
            f"only {len(set(urls))} unique"
        )


def test_authoritative_host_floor():
    """At least 90% of URLs must come from the NIH/FDA/peer-review whitelist.

    Catches the case where a future iteration adds a non-authoritative
    source (random blog, vendor whitepaper, archived link). Update
    `_AUTHORITATIVE_HOSTS` if a NEW authoritative source lands — never
    lower the floor.
    """
    cache = _load_cache()
    total = 0
    authoritative = 0
    non_auth_hosts: set[str] = set()
    for entry in cache:
        for url in entry.get("evidence_urls", []):
            host = urlparse(url).netloc
            total += 1
            if host in _AUTHORITATIVE_HOSTS:
                authoritative += 1
            else:
                non_auth_hosts.add(host)
    ratio = authoritative / max(total, 1)
    assert ratio >= 0.90, (
        f"Authoritative-host ratio {ratio:.2%} below 90% floor "
        f"({authoritative} / {total} URLs).\n"
        f"Non-whitelisted hosts (consider adding to _AUTHORITATIVE_HOSTS "
        f"if these are NEW authoritative sources): {sorted(non_auth_hosts)}"
    )


def test_cache_meets_evidence_density_floor():
    """Average ≥ 1.5 evidence URLs per cache entry — every drug-pair
    classification should be backed by AT LEAST a primary citation
    AND a secondary corroboration (e.g. FDA label + peer-review or
    guideline + PubMed)."""
    cache = _load_cache()
    total_urls = sum(len(e.get("evidence_urls", [])) for e in cache)
    avg = total_urls / max(len(cache), 1)
    assert avg >= 1.5, (
        f"Evidence-URL density per cache entry: {avg:.2f} (floor 1.5). "
        f"Add corroborating citations to entries with only one URL."
    )
