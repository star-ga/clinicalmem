"""Pin: pharmacology flag table covers every cache drug + URL hygiene.

Iter 100 (T1 round 20 substituted, eval-rigor): the new
`docs/pharmacology_flags.json` is shipped + manifest-tracked
(iter 96 + iter 97), but no test enforces:

  1. **Coverage**: every drug name appearing in
     `docs/openevidence_cache.json` (drug_a or drug_b) must have a
     matching entry in pharmacology_flags. Future cache growth that
     adds a new drug pair without flagging the new drug would slip
     past every existing test, breaking the curated-table claim
     ("every drug in our cohort traces to FDA pharmacology classes").

  2. **URL hygiene**: every flagged drug must have ≥ 1 evidence URL,
     all HTTPS. Same pattern as `test_cache_evidence_urls.py` but
     against the flag table.

  3. **Schema-version pin**: schema_version present + flag_keys is
     a non-empty list (iter-96 baseline schema).

This is the same drift class the iter-94 cache shape pins catch for
the cache itself, applied to its companion artifact.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_CACHE = _REPO_ROOT / "docs" / "openevidence_cache.json"
_FLAGS = _REPO_ROOT / "docs" / "pharmacology_flags.json"


def _cache_drug_set():
    cache = json.loads(_CACHE.read_text())
    drugs = set()
    for it in cache:
        drugs.add(it["drug_a"].lower().strip())
        drugs.add(it["drug_b"].lower().strip())
    return drugs


def _flags_doc():
    return json.loads(_FLAGS.read_text())


def test_every_cache_drug_has_pharmacology_flag_entry():
    """Every drug name in the cache must have an entry in
    `pharmacology_flags.json::drugs`. Empty-flag entries are fine
    (some drugs genuinely have no pharmacology-class flags), but the
    drug must be CATALOGUED so a future flag addition has a home."""
    cache_drugs = _cache_drug_set()
    flagged = set(_flags_doc()["drugs"].keys())
    missing = cache_drugs - flagged
    assert not missing, (
        f"{len(missing)} cache drugs missing from pharmacology_flags: "
        f"{sorted(missing)[:8]}{'...' if len(missing) > 8 else ''}. "
        f"Add each one to docs/pharmacology_flags.json with at least "
        f"`flags: []` and an FDA-label evidence URL."
    )


def test_every_flagged_drug_has_at_least_one_evidence_url():
    """The curated-table pitch is 'every flag traces to a published
    FDA label or peer-reviewed reference'. An entry with no URLs
    breaks that pitch."""
    flags = _flags_doc()["drugs"]
    no_url = [n for n, e in flags.items() if not e.get("evidence_urls")]
    assert not no_url, (
        f"{len(no_url)} flagged drugs have no evidence URL: "
        f"{no_url[:5]}{'...' if len(no_url) > 5 else ''}. "
        f"Each entry must cite at least one FDA label or peer-review reference."
    )


def test_every_evidence_url_is_https():
    """Plain-HTTP URLs in a regulated-domain audit chain are a
    silent integrity hole. Same pin as test_cache_evidence_urls.py
    applied to the flag table."""
    flags = _flags_doc()["drugs"]
    bad = []
    for name, e in flags.items():
        for u in e.get("evidence_urls", []):
            if not u.startswith("https://"):
                bad.append((name, u))
    assert not bad, (
        f"{len(bad)} pharmacology flag URLs are not HTTPS. "
        f"First: {bad[0]}"
    )


def test_schema_version_and_flag_keys_present():
    """Iter-96 baseline schema: schema_version set + flag_keys non-empty
    list. Removing either would silently break the audit chain."""
    doc = _flags_doc()
    assert doc.get("schema_version"), (
        "pharmacology_flags.json must carry a schema_version field"
    )
    flag_keys = doc.get("flag_keys", [])
    assert isinstance(flag_keys, list) and len(flag_keys) >= 12, (
        f"flag_keys must be a non-empty list of >= 12 entries; "
        f"got {len(flag_keys) if isinstance(flag_keys, list) else type(flag_keys).__name__}"
    )


def test_every_flag_key_is_canonical_snake_case_is_prefix():
    """Every flag in `flag_keys` MUST start with `is_` and be
    snake_case. Violating this breaks downstream encoders that pattern-
    match on the prefix (e.g. `_pair_derived_flags` checks for
    `has_pair("is_cyp3a4_strong_inhibitor", "is_cyp3a4_substrate")`)."""
    flag_keys = _flags_doc()["flag_keys"]
    snake = re.compile(r"^is_[a-z][a-z0-9_]*$")
    bad = [k for k in flag_keys if not snake.match(k)]
    assert not bad, (
        f"non-canonical flag key(s): {bad}. "
        f"All keys must match `is_[a-z][a-z0-9_]*`."
    )


def test_drug_names_in_flag_table_are_canonicalised_lowercase():
    """Drug-name keys must be lowercase + whitespace-collapsed so
    `_flag_bits(drug_name)` lookup is consistent with the cache's
    `drug_a` / `drug_b` canonicalisation."""
    flags = _flags_doc()["drugs"]
    bad = [n for n in flags.keys() if n != " ".join(n.lower().split())]
    assert not bad, (
        f"non-canonical drug-name keys: {bad[:5]}. "
        f"Each key must equal `' '.join(name.lower().split())`."
    )
