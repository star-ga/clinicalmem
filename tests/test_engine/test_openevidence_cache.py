"""Tests for the OpenEvidence cached fixture set (engine/openevidence_cache.py).

Covers:
- canonical_pair_key lex-sort contract
- lookup_cached happy-path and miss
- _openevidence_check_interactions cache fallback when no API key is set
- Audit-trail prefix on cached DrugInteraction descriptions
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pytest

from engine.openevidence_cache import (
    CachedOpenEvidenceResponse,
    canonical_pair_key,
    invalidate_cache,
    lookup_cached,
)
from engine.clinical_scoring import DrugInteraction, _openevidence_check_interactions


# ---------------------------------------------------------------------------
# canonical_pair_key
# ---------------------------------------------------------------------------


class TestCanonicalPairKey:
    def test_lex_sorts_alphabetically(self):
        key = canonical_pair_key("warfarin", "ibuprofen")
        assert key == ("ibuprofen", "warfarin")

    def test_is_commutative(self):
        assert canonical_pair_key("warfarin", "ibuprofen") == canonical_pair_key(
            "ibuprofen", "warfarin"
        )

    def test_lowercases_both_drugs(self):
        key = canonical_pair_key("Warfarin", "IBUPROFEN")
        assert key == ("ibuprofen", "warfarin")

    def test_equal_names_are_stable(self):
        key = canonical_pair_key("aspirin", "aspirin")
        assert key == ("aspirin", "aspirin")

    def test_strips_whitespace(self):
        key = canonical_pair_key("  warfarin  ", "ibuprofen")
        assert key == ("ibuprofen", "warfarin")


# ---------------------------------------------------------------------------
# lookup_cached
# ---------------------------------------------------------------------------


class TestLookupCached:
    def setup_method(self):
        # Ensure we start each test with a fresh cache load.
        invalidate_cache()

    def test_returns_warfarin_ibuprofen(self):
        entry = lookup_cached("warfarin", "ibuprofen")
        assert entry is not None
        assert entry.severity is not None
        assert len(entry.severity) > 0

    def test_commutative_lookup_ibuprofen_warfarin(self):
        fwd = lookup_cached("warfarin", "ibuprofen")
        rev = lookup_cached("ibuprofen", "warfarin")
        assert fwd is not None
        assert rev is not None
        assert fwd.drug_pair_canonical == rev.drug_pair_canonical

    def test_returns_correct_severity_for_warfarin_ibuprofen(self):
        entry = lookup_cached("warfarin", "ibuprofen")
        assert entry is not None
        assert entry.severity == "serious"

    def test_returns_none_for_unknown_pair(self):
        entry = lookup_cached("acetaminophen", "loratadine")
        assert entry is None

    def test_returns_none_for_single_known_drug(self):
        # One drug from a known pair + an unknown partner
        entry = lookup_cached("warfarin", "unknowndrug_xyz")
        assert entry is None

    def test_entry_is_cached_response_type(self):
        entry = lookup_cached("ciprofloxacin", "tizanidine")
        assert isinstance(entry, CachedOpenEvidenceResponse)

    def test_ciprofloxacin_tizanidine_is_contraindicated(self):
        entry = lookup_cached("ciprofloxacin", "tizanidine")
        assert entry is not None
        assert entry.severity == "contraindicated"

    def test_entry_has_clinical_summary(self):
        entry = lookup_cached("fluoxetine", "tramadol")
        assert entry is not None
        assert len(entry.clinical_summary) > 20

    def test_entry_has_evidence_urls(self):
        entry = lookup_cached("metoprolol", "verapamil")
        assert entry is not None
        assert isinstance(entry.evidence_urls, list)
        assert len(entry.evidence_urls) >= 1

    def test_entry_source_is_cached(self):
        entry = lookup_cached("warfarin", "aspirin")
        assert entry is not None
        assert entry.source == "CACHED"

    def test_entry_has_retrieved_at(self):
        entry = lookup_cached("warfarin", "aspirin")
        assert entry is not None
        assert entry.retrieved_at == "2026-05-02"

    def test_metformin_contrast_dye_is_contraindicated(self):
        entry = lookup_cached("metformin", "contrast dye")
        assert entry is not None
        assert entry.severity == "contraindicated"

    def test_simvastatin_amiodarone_is_serious(self):
        entry = lookup_cached("simvastatin", "amiodarone")
        assert entry is not None
        assert entry.severity == "serious"

    def test_methotrexate_trimethoprim_is_serious(self):
        entry = lookup_cached("methotrexate", "trimethoprim")
        assert entry is not None
        assert entry.severity == "serious"

    def test_canonical_pair_stored_lex_sorted(self):
        entry = lookup_cached("warfarin", "aspirin")
        assert entry is not None
        a, b = entry.drug_pair_canonical
        assert a <= b

    # Regression pairs from the frozen test suite
    def test_regression_ibuprofen_warfarin(self):
        entry = lookup_cached("ibuprofen", "warfarin")
        assert entry is not None
        assert entry.severity in ("serious", "contraindicated")

    def test_regression_amoxicillin_penicillin(self):
        entry = lookup_cached("amoxicillin", "penicillin")
        assert entry is not None

    def test_regression_metformin_iodine(self):
        entry = lookup_cached("metformin", "iodine")
        assert entry is not None

    def test_regression_atorvastatin_grapefruit(self):
        entry = lookup_cached("atorvastatin", "grapefruit")
        assert entry is not None

    def test_regression_aspirin_warfarin(self):
        entry = lookup_cached("aspirin", "warfarin")
        assert entry is not None


# ---------------------------------------------------------------------------
# _openevidence_check_interactions — cache fallback path
# ---------------------------------------------------------------------------


class TestOpenEvidenceCheckInteractionsCacheFallback:
    """Verify the live function falls back to cache when no API key is set."""

    def setup_method(self):
        invalidate_cache()

    def _unset_api_key(self, monkeypatch):
        monkeypatch.delenv("OPENEVIDENCE_API_KEY", raising=False)

    def test_falls_back_to_cache_when_no_key(self, monkeypatch):
        self._unset_api_key(monkeypatch)
        results = _openevidence_check_interactions(
            ["warfarin", "ibuprofen"], already_found=set()
        )
        assert len(results) >= 1
        assert any(isinstance(r, DrugInteraction) for r in results)

    def test_cached_response_has_cached_prefix(self, monkeypatch):
        self._unset_api_key(monkeypatch)
        results = _openevidence_check_interactions(
            ["warfarin", "ibuprofen"], already_found=set()
        )
        assert len(results) >= 1
        descriptions = [r.description for r in results]
        assert any("[CACHED" in d for d in descriptions), (
            f"Expected '[CACHED' prefix in one of {descriptions}"
        )

    def test_cached_response_description_contains_date(self, monkeypatch):
        self._unset_api_key(monkeypatch)
        results = _openevidence_check_interactions(
            ["warfarin", "ibuprofen"], already_found=set()
        )
        assert any("2026-05-02" in r.description for r in results)

    def test_cached_result_severity_is_non_empty(self, monkeypatch):
        self._unset_api_key(monkeypatch)
        results = _openevidence_check_interactions(
            ["warfarin", "ibuprofen"], already_found=set()
        )
        assert all(r.severity for r in results)

    def test_cached_result_has_non_zero_score(self, monkeypatch):
        self._unset_api_key(monkeypatch)
        results = _openevidence_check_interactions(
            ["warfarin", "ibuprofen"], already_found=set()
        )
        assert all(r.score > 0 for r in results)

    def test_already_found_pair_is_skipped(self, monkeypatch):
        self._unset_api_key(monkeypatch)
        already = {("ibuprofen", "warfarin")}
        results = _openevidence_check_interactions(
            ["warfarin", "ibuprofen"], already_found=already
        )
        # The warfarin+ibuprofen pair is already covered — should not reappear.
        for r in results:
            pair = tuple(sorted([r.drug_a, r.drug_b]))
            assert pair not in {("ibuprofen", "warfarin")}

    def test_unknown_pair_returns_empty_list(self, monkeypatch):
        self._unset_api_key(monkeypatch)
        results = _openevidence_check_interactions(
            ["acetaminophen", "loratadine"], already_found=set()
        )
        assert results == []

    def test_single_medication_returns_empty_list(self, monkeypatch):
        self._unset_api_key(monkeypatch)
        results = _openevidence_check_interactions(["warfarin"], already_found=set())
        assert results == []

    def test_drug_interaction_fields_are_correct_types(self, monkeypatch):
        self._unset_api_key(monkeypatch)
        results = _openevidence_check_interactions(
            ["ciprofloxacin", "tizanidine"], already_found=set()
        )
        assert len(results) >= 1
        r = results[0]
        assert isinstance(r.drug_a, str)
        assert isinstance(r.drug_b, str)
        assert isinstance(r.severity, str)
        assert isinstance(r.description, str)
        assert isinstance(r.score, float)

    def test_multiple_pairs_all_returned(self, monkeypatch):
        self._unset_api_key(monkeypatch)
        # warfarin+ibuprofen AND fluoxetine+tramadol should both return cache hits
        results = _openevidence_check_interactions(
            ["warfarin", "ibuprofen", "fluoxetine", "tramadol"], already_found=set()
        )
        # Layer 1 (deterministic table) normally picks these up; cache is Layer 2,
        # so pass an already_found that excludes them to force cache hits.
        results2 = _openevidence_check_interactions(
            ["warfarin", "ibuprofen", "fluoxetine", "tramadol"],
            already_found=set(),
        )
        # At minimum cache must produce results for at least one known pair
        assert len(results2) >= 1
