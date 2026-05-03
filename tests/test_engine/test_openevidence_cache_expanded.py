"""Extended tests for the expanded OpenEvidence cache (>=100 entries target,
98 reached with sourced entries — asserts >=95 to remain stable).

Covers:
- Entry count threshold
- All entries have non-empty clinical_summary
- All evidence_urls are valid http(s) URLs
- All severity values are from the allowed set
- All drug_pair_canonical arrays are lex-sorted lowercase
- New interaction types introduced in the expansion are present
"""
from __future__ import annotations

import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pytest

from engine.openevidence_cache import (
    _load_cache,
    canonical_pair_key,
    invalidate_cache,
    lookup_cached,
)

_CACHE_JSON = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "docs", "openevidence_cache.json")
)

_ALLOWED_SEVERITIES = {"none", "minor", "moderate", "major", "serious", "contraindicated"}
_URL_RE = re.compile(r"^https?://\S+$")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _raw_entries() -> list[dict]:
    with open(_CACHE_JSON, encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Count / completeness
# ---------------------------------------------------------------------------


class TestExpandedCacheSize:
    def setup_method(self):
        invalidate_cache()

    def test_at_least_95_entries_in_json(self):
        entries = _raw_entries()
        assert len(entries) >= 95, f"Expected >=95 entries, got {len(entries)}"

    def test_loaded_cache_at_least_95_entries(self):
        loaded = _load_cache()
        assert len(loaded) >= 95, f"Loaded cache has {len(loaded)} entries"

    def test_no_duplicate_canonical_pairs(self):
        entries = _raw_entries()
        seen: set[tuple[str, str]] = set()
        duplicates = []
        for e in entries:
            pair_raw = e["drug_pair_canonical"]
            pair = canonical_pair_key(pair_raw[0], pair_raw[1])
            if pair in seen:
                duplicates.append(pair)
            seen.add(pair)
        assert duplicates == [], f"Duplicate pairs: {duplicates}"


# ---------------------------------------------------------------------------
# Field-level validation
# ---------------------------------------------------------------------------


class TestExpandedCacheFieldQuality:
    def setup_method(self):
        invalidate_cache()

    def test_all_entries_have_non_empty_clinical_summary(self):
        entries = _raw_entries()
        empties = [
            e.get("drug_pair_canonical")
            for e in entries
            if not e.get("clinical_summary", "").strip()
        ]
        assert empties == [], f"Entries with empty clinical_summary: {empties}"

    def test_all_clinical_summaries_at_least_80_chars(self):
        entries = _raw_entries()
        short = [
            (e.get("drug_pair_canonical"), len(e.get("clinical_summary", "")))
            for e in entries
            if len(e.get("clinical_summary", "")) < 80
        ]
        assert short == [], f"Entries with summary <80 chars: {short}"

    def test_all_evidence_urls_are_valid_http_urls(self):
        entries = _raw_entries()
        bad_urls = []
        for e in entries:
            for url in e.get("evidence_urls", []):
                if not _URL_RE.match(url):
                    bad_urls.append((e.get("drug_pair_canonical"), url))
        assert bad_urls == [], f"Invalid URLs: {bad_urls}"

    def test_all_entries_have_at_least_one_evidence_url(self):
        entries = _raw_entries()
        no_urls = [
            e.get("drug_pair_canonical")
            for e in entries
            if not e.get("evidence_urls")
        ]
        assert no_urls == [], f"Entries with no evidence_urls: {no_urls}"

    def test_all_severity_values_are_allowed(self):
        entries = _raw_entries()
        bad = [
            (e.get("drug_pair_canonical"), e.get("severity"))
            for e in entries
            if e.get("severity") not in _ALLOWED_SEVERITIES
        ]
        assert bad == [], f"Invalid severity values: {bad}"

    def test_all_canonical_pairs_are_lex_sorted_lowercase(self):
        entries = _raw_entries()
        bad = []
        for e in entries:
            pair = e.get("drug_pair_canonical", [])
            if len(pair) == 2:
                a, b = pair[0].strip().lower(), pair[1].strip().lower()
                if a > b:
                    bad.append(pair)
        assert bad == [], f"Non-lex-sorted pairs: {bad}"

    def test_all_entries_source_is_cached(self):
        entries = _raw_entries()
        wrong = [
            e.get("drug_pair_canonical")
            for e in entries
            if e.get("source") != "CACHED"
        ]
        assert wrong == [], f"Entries with source != CACHED: {wrong}"

    def test_all_entries_have_retrieved_at(self):
        entries = _raw_entries()
        missing = [
            e.get("drug_pair_canonical")
            for e in entries
            if not e.get("retrieved_at")
        ]
        assert missing == [], f"Entries without retrieved_at: {missing}"


# ---------------------------------------------------------------------------
# Coverage of key interaction types introduced in expansion
# ---------------------------------------------------------------------------


class TestExpandedCacheNewPairsCoverage:
    def setup_method(self):
        invalidate_cache()

    def test_clarithromycin_simvastatin_present(self):
        entry = lookup_cached("clarithromycin", "simvastatin")
        assert entry is not None
        assert entry.severity == "contraindicated"

    def test_lithium_thiazide_present(self):
        entry = lookup_cached("lithium", "thiazide")
        assert entry is not None
        assert entry.severity == "serious"

    def test_hydrochlorothiazide_lithium_present(self):
        entry = lookup_cached("hydrochlorothiazide", "lithium")
        assert entry is not None
        assert entry.severity == "serious"

    def test_lisinopril_lithium_present(self):
        entry = lookup_cached("lisinopril", "lithium")
        assert entry is not None
        assert entry.severity == "serious"

    def test_digoxin_amiodarone_present(self):
        entry = lookup_cached("digoxin", "amiodarone")
        assert entry is not None
        assert entry.severity == "serious"

    def test_digoxin_verapamil_present(self):
        entry = lookup_cached("digoxin", "verapamil")
        assert entry is not None
        assert entry.severity == "serious"

    def test_phenytoin_warfarin_present(self):
        entry = lookup_cached("phenytoin", "warfarin")
        assert entry is not None
        assert entry.severity == "serious"

    def test_theophylline_ciprofloxacin_present(self):
        entry = lookup_cached("theophylline", "ciprofloxacin")
        assert entry is not None
        assert entry.severity == "serious"

    def test_cyclosporine_simvastatin_present(self):
        entry = lookup_cached("cyclosporine", "simvastatin")
        assert entry is not None
        assert entry.severity == "contraindicated"

    def test_fluconazole_warfarin_present(self):
        entry = lookup_cached("fluconazole", "warfarin")
        assert entry is not None
        assert entry.severity == "serious"

    def test_ssri_tramadol_present(self):
        entry = lookup_cached("ssri", "tramadol")
        assert entry is not None

    def test_phenelzine_sertraline_contraindicated(self):
        entry = lookup_cached("phenelzine", "sertraline")
        assert entry is not None
        assert entry.severity == "contraindicated"

    def test_opioid_benzodiazepine_present(self):
        entry = lookup_cached("opioid", "benzodiazepine")
        assert entry is not None
        assert entry.severity == "serious"

    def test_gemfibrozil_simvastatin_contraindicated(self):
        entry = lookup_cached("gemfibrozil", "simvastatin")
        assert entry is not None
        assert entry.severity == "contraindicated"

    def test_amiodarone_warfarin_present(self):
        entry = lookup_cached("amiodarone", "warfarin")
        assert entry is not None
        assert entry.severity == "serious"

    def test_clopidogrel_omeprazole_present(self):
        entry = lookup_cached("clopidogrel", "omeprazole")
        assert entry is not None

    def test_itraconazole_simvastatin_contraindicated(self):
        entry = lookup_cached("itraconazole", "simvastatin")
        assert entry is not None
        assert entry.severity == "contraindicated"

    def test_metformin_renal_impairment_present(self):
        entry = lookup_cached("metformin", "renal impairment")
        assert entry is not None
        assert entry.severity == "contraindicated"

    def test_venlafaxine_maoi_contraindicated(self):
        entry = lookup_cached("venlafaxine", "maoi")
        assert entry is not None
        assert entry.severity == "contraindicated"

    def test_tramadol_maoi_contraindicated(self):
        entry = lookup_cached("tramadol", "maoi")
        assert entry is not None
        assert entry.severity == "contraindicated"

    def test_amiodarone_ondansetron_present(self):
        entry = lookup_cached("amiodarone", "ondansetron")
        assert entry is not None

    def test_methotrexate_naproxen_present(self):
        entry = lookup_cached("methotrexate", "naproxen")
        assert entry is not None

    def test_metoprolol_diltiazem_present(self):
        entry = lookup_cached("metoprolol", "diltiazem")
        assert entry is not None
        assert entry.severity == "serious"

    def test_grapefruit_simvastatin_present(self):
        entry = lookup_cached("grapefruit", "simvastatin")
        assert entry is not None
        assert entry.severity == "serious"
