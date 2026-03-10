"""Extended coverage tests for engine/clinical_memory.py — mind-mem paths, audit, recall."""
import json
import os
import sys
import tempfile
import pytest
from unittest.mock import patch, MagicMock, PropertyMock

from engine.clinical_memory import ClinicalMemEngine, ClinicalBlock


@pytest.fixture
def engine():
    return ClinicalMemEngine()


@pytest.fixture
def loaded_engine():
    eng = ClinicalMemEngine()
    fixture_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "fixtures", "sarah_mitchell_bundle.json",
    )
    with open(fixture_path) as f:
        bundle = json.load(f)
    eng.ingest_from_bundle(bundle, "patient-sarah-mitchell")
    return eng


# ── _patient_dir edge cases ──────────────────────────────────────────────────

class TestPatientDir:
    def test_valid_id(self, engine):
        path = engine._patient_dir("patient-1")
        assert os.path.isdir(path)

    def test_empty_id(self, engine):
        # All special chars get sanitized away, leaving "_"s which is valid
        # Test with truly empty string after strip
        with pytest.raises(ValueError, match="Invalid patient_id"):
            engine._patient_dir("")

    def test_sanitized_id(self, engine):
        path = engine._patient_dir("abc!@#def")
        assert "abc" in os.path.basename(path)


# ── _append_audit_fallback ───────────────────────────────────────────────────

class TestAuditFallback:
    def test_genesis_hash(self, engine):
        engine._mind_mem_available = False
        engine._audit_chain_mm = None
        h = engine._append_audit("test", {"key": "val"})
        assert isinstance(h, str)
        assert len(h) == 64  # SHA-256

    def test_chain_links(self, engine):
        engine._audit_chain_mm = None
        h1 = engine._append_audit("op1", {"a": 1})
        h2 = engine._append_audit("op2", {"b": 2})
        assert h1 != h2
        # Verify chain
        chain = engine._audit_chain_fallback
        assert chain[0]["prev_hash"] == "genesis"
        assert chain[1]["prev_hash"] == h1


# ── verify_audit_chain fallback ──────────────────────────────────────────────

class TestVerifyAuditChainFallback:
    def test_empty_chain_valid(self, engine):
        engine._audit_chain_mm = None
        assert engine.verify_audit_chain() is True

    def test_valid_chain(self, engine):
        engine._audit_chain_mm = None
        engine._append_audit("a", {})
        engine._append_audit("b", {})
        assert engine.verify_audit_chain() is True

    def test_tampered_chain(self, engine):
        engine._audit_chain_mm = None
        engine._append_audit("a", {})
        engine._append_audit("b", {})
        engine._audit_chain_fallback[1]["prev_hash"] = "tampered"
        assert engine.verify_audit_chain() is False

    def test_first_entry_bad_genesis(self, engine):
        engine._audit_chain_mm = None
        engine._append_audit("a", {})
        engine._audit_chain_fallback[0]["prev_hash"] = "not_genesis"
        assert engine.verify_audit_chain() is False


# ── get_audit_trail fallback ─────────────────────────────────────────────────

class TestGetAuditTrailFallback:
    def test_empty(self, engine):
        engine._audit_chain_mm = None
        assert engine.get_audit_trail() == []

    def test_returns_entries(self, engine):
        engine._audit_chain_mm = None
        engine._append_audit("a", {"x": 1})
        trail = engine.get_audit_trail()
        assert len(trail) == 1
        assert trail[0]["action"] == "a"

    def test_limit(self, engine):
        engine._audit_chain_mm = None
        for i in range(10):
            engine._append_audit(f"op{i}", {})
        trail = engine.get_audit_trail(limit=3)
        assert len(trail) == 3


# ── _recall_fallback with negation ───────────────────────────────────────────

class TestRecallFallback:
    def test_negation_query(self, loaded_engine):
        result = loaded_engine.recall("patient-sarah-mitchell", "no allergies denied")
        assert result.query == "no allergies denied"
        assert result.patient_id == "patient-sarah-mitchell"

    def test_empty_patient(self, engine):
        result = engine.recall("nonexistent", "test query")
        assert result.blocks == []
        assert result.confidence.should_abstain is True

    def test_recall_returns_blocks(self, loaded_engine):
        result = loaded_engine.recall("patient-sarah-mitchell", "warfarin medication")
        assert len(result.blocks) > 0
        for b in result.blocks:
            assert "block_id" in b
            assert "score" in b


# ── _recall_mindmem path ────────────────────────────────────────────────────

class TestRecallMindmem:
    def test_mindmem_search_exception_falls_back(self, loaded_engine):
        """When mind-mem search throws, fall back to BM25."""
        loaded_engine._hybrid_backend = MagicMock()
        loaded_engine._hybrid_backend.search.side_effect = Exception("search failed")
        result = loaded_engine.recall("patient-sarah-mitchell", "warfarin")
        assert len(result.blocks) > 0
        assert result.blocks[0]["search_backend"] == "fallback/bm25-approx"

    def test_mindmem_returns_empty_falls_back(self, loaded_engine):
        """When mind-mem returns nothing, fall back."""
        loaded_engine._hybrid_backend = MagicMock()
        loaded_engine._hybrid_backend.search.return_value = []
        result = loaded_engine.recall("patient-sarah-mitchell", "warfarin")
        assert len(result.blocks) > 0

    def test_mindmem_returns_matched_blocks(self, loaded_engine):
        """When mind-mem returns matching blocks, use them."""
        loaded_engine._hybrid_backend = MagicMock()
        blocks = loaded_engine._patient_blocks["patient-sarah-mitchell"]
        loaded_engine._hybrid_backend.search.return_value = [
            {"_id": blocks[0].block_id, "rrf_score": 0.9, "Title": blocks[0].title, "Content": blocks[0].content, "Type": blocks[0].resource_type},
        ]
        result = loaded_engine.recall("patient-sarah-mitchell", "test")
        assert len(result.blocks) == 1
        assert result.blocks[0]["search_backend"] == "mind-mem/hybrid"

    def test_mindmem_unmatched_block(self, loaded_engine):
        """When mind-mem returns a block not in our map, include raw."""
        loaded_engine._hybrid_backend = MagicMock()
        loaded_engine._hybrid_backend.search.return_value = [
            {"_id": "unknown-id", "rrf_score": 0.5, "Title": "Unknown", "Content": "test", "Type": "Unknown"},
        ]
        result = loaded_engine.recall("patient-sarah-mitchell", "test")
        assert len(result.blocks) == 1
        assert result.blocks[0]["block_id"] == "unknown-id"


# ── clinical_handoff ─────────────────────────────────────────────────────────

class TestClinicalHandoff:
    def test_handoff_no_data(self, engine):
        narrative = engine.clinical_handoff("nonexistent")
        assert narrative.abstained is True or "ABSTAIN" in narrative.narrative or "CARE HANDOFF" in narrative.narrative

    def test_handoff_with_data(self, loaded_engine):
        narrative = loaded_engine.clinical_handoff("patient-sarah-mitchell")
        assert narrative.narrative  # Not empty


# ── explain_clinical_conflict ────────────────────────────────────────────────

class TestExplainClinicalConflict:
    def test_no_conflicts(self, engine):
        narrative = engine.explain_clinical_conflict("nonexistent")
        assert narrative.abstained is True

    def test_out_of_range_index(self, loaded_engine):
        narrative = loaded_engine.explain_clinical_conflict("patient-sarah-mitchell", conflict_index=999)
        # Should either abstain or handle gracefully
        assert narrative.abstained is True or narrative.narrative

    def test_with_conflicts(self, loaded_engine):
        narrative = loaded_engine.explain_clinical_conflict("patient-sarah-mitchell", conflict_index=0)
        assert narrative.narrative


# ── _store_block + _write_block_markdown ─────────────────────────────────────

class TestStoreBlock:
    def test_write_markdown_when_mindmem(self, engine):
        engine._mind_mem_available = True
        block = ClinicalBlock(
            block_id="test-1", patient_id="p-1",
            resource_type="Observation", title="Test",
            content="Content", metadata={"key": "value"},
            timestamp="2026-01-01", source="test",
        )
        engine._store_block(block)
        assert "p-1" in engine._patient_blocks
        # Check markdown was written
        pdir = engine._patient_dir("p-1")
        corpus_path = os.path.join(pdir, "corpus", "Observation.md")
        assert os.path.isfile(corpus_path)


# ── mind-mem init error paths (lines 123-126) ─────────────────────────────

class TestInitMindMemErrors:
    """Cover _init_mind_mem ImportError (lines 123-124) and Exception (lines 125-126)."""

    def test_import_error_falls_back(self):
        """Line 123-124: ImportError when importing mind-mem modules."""
        # Force ImportError by making mind_mem modules unimportable
        saved_audit = sys.modules.get("mind_mem.audit_chain")
        saved_hybrid = sys.modules.get("mind_mem.hybrid_recall")
        sys.modules["mind_mem.audit_chain"] = None  # Force ImportError
        sys.modules["mind_mem.hybrid_recall"] = None
        try:
            eng = ClinicalMemEngine()
            assert eng._mind_mem_available is False
            assert eng._audit_chain_mm is None
            assert eng._hybrid_backend is None
        finally:
            if saved_audit is not None:
                sys.modules["mind_mem.audit_chain"] = saved_audit
            else:
                sys.modules.pop("mind_mem.audit_chain", None)
            if saved_hybrid is not None:
                sys.modules["mind_mem.hybrid_recall"] = saved_hybrid
            else:
                sys.modules.pop("mind_mem.hybrid_recall", None)

    def test_non_import_exception_falls_back(self):
        """Line 125-126: Non-ImportError exception during mind-mem init."""
        # Mock the imports to succeed but constructor to raise RuntimeError
        mock_audit_chain_cls = MagicMock(side_effect=RuntimeError("init boom"))
        mock_module_audit = MagicMock()
        mock_module_audit.AuditChain = mock_audit_chain_cls
        mock_module_hybrid = MagicMock()

        saved_audit = sys.modules.get("mind_mem.audit_chain")
        saved_hybrid = sys.modules.get("mind_mem.hybrid_recall")
        sys.modules["mind_mem.audit_chain"] = mock_module_audit
        sys.modules["mind_mem.hybrid_recall"] = mock_module_hybrid
        try:
            eng = ClinicalMemEngine()
            assert eng._mind_mem_available is False
            assert eng._audit_chain_mm is None
            assert eng._hybrid_backend is None
        finally:
            if saved_audit is not None:
                sys.modules["mind_mem.audit_chain"] = saved_audit
            else:
                sys.modules.pop("mind_mem.audit_chain", None)
            if saved_hybrid is not None:
                sys.modules["mind_mem.hybrid_recall"] = saved_hybrid
            else:
                sys.modules.pop("mind_mem.hybrid_recall", None)


# ── Observation ingestion with valueString (line 329) ──────────────────────

class TestIngestValueString:
    """Cover line 329: Observation with valueString instead of valueQuantity."""

    def test_valuestring_observation(self, engine):
        """When an Observation has valueString, it should be used as the value."""
        bundle = {
            "resourceType": "Bundle",
            "type": "collection",
            "entry": [
                {
                    "resource": {
                        "resourceType": "Observation",
                        "id": "obs-vs-1",
                        "status": "final",
                        "category": [
                            {
                                "coding": [
                                    {"system": "http://terminology.hl7.org/CodeSystem/observation-category", "code": "laboratory"}
                                ]
                            }
                        ],
                        "code": {
                            "text": "Urine Color",
                            "coding": [{"display": "Urine Color"}],
                        },
                        "valueString": "Dark Amber",
                        "effectiveDateTime": "2026-01-15",
                    }
                }
            ],
        }
        counts = engine.ingest_from_bundle(bundle, "patient-vs-test")
        assert counts["observations"] >= 1
        blocks = engine._patient_blocks.get("patient-vs-test", [])
        obs_blocks = [b for b in blocks if b.resource_type == "Observation"]
        assert len(obs_blocks) >= 1
        # The value should contain the valueString text
        assert any("Dark Amber" in b.content for b in obs_blocks)


# ── _recall_fallback when _hybrid_backend is None (line 470) ───────────────

class TestRecallFallbackPath:
    """Cover line 470: explicit test that recall goes to _recall_fallback when _hybrid_backend is None."""

    def test_recall_uses_fallback_when_no_hybrid_backend(self, loaded_engine):
        """Line 470: When _hybrid_backend is None, recall routes to _recall_fallback."""
        # Ensure hybrid backend is None
        loaded_engine._hybrid_backend = None
        result = loaded_engine.recall("patient-sarah-mitchell", "warfarin")
        assert result.patient_id == "patient-sarah-mitchell"
        # Should get results from fallback BM25
        assert len(result.blocks) > 0


# ── _get_audit_trail_mindmem when chain file doesn't exist (line 862) ──────

class TestGetAuditTrailMindmemNoFile:
    """Cover line 862: _get_audit_trail_mindmem returns [] when chain file doesn't exist."""

    def test_returns_empty_when_chain_file_missing(self, engine):
        """Line 862: When the audit chain JSONL file does not exist, return empty list."""
        mock_chain = MagicMock()
        mock_chain._chain_path = "/tmp/nonexistent_chain_file_12345.jsonl"
        engine._audit_chain_mm = mock_chain
        trail = engine.get_audit_trail()
        assert trail == []

    def test_returns_entries_when_chain_file_exists(self, engine):
        """Positive case: When file exists with entries, return them."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(json.dumps({"action": "create_block", "ts": 1}) + "\n")
            f.write(json.dumps({"action": "read_block", "ts": 2}) + "\n")
            tmp_path = f.name
        try:
            mock_chain = MagicMock()
            mock_chain._chain_path = tmp_path
            engine._audit_chain_mm = mock_chain
            trail = engine.get_audit_trail(limit=50)
            assert len(trail) == 2
            assert trail[0]["action"] == "create_block"
        finally:
            os.unlink(tmp_path)


# ── verify_audit_chain with mind-mem errors (line 876) ─────────────────────

class TestVerifyAuditChainMindmem:
    """Cover line 876: audit chain verification with errors logged."""

    def test_verify_with_errors_logs_warning(self, engine):
        """Line 876: When audit chain verify() returns errors, log warning."""
        mock_chain = MagicMock()
        mock_chain.verify.return_value = (True, ["hash mismatch at entry 3"])
        engine._audit_chain_mm = mock_chain
        result = engine.verify_audit_chain()
        assert result is True
        mock_chain.verify.assert_called_once()

    def test_verify_invalid_chain(self, engine):
        """verify() returns False with errors."""
        mock_chain = MagicMock()
        mock_chain.verify.return_value = (False, ["tampering detected"])
        engine._audit_chain_mm = mock_chain
        result = engine.verify_audit_chain()
        assert result is False

    def test_verify_valid_no_errors(self, engine):
        """verify() returns True with no errors."""
        mock_chain = MagicMock()
        mock_chain.verify.return_value = (True, [])
        engine._audit_chain_mm = mock_chain
        result = engine.verify_audit_chain()
        assert result is True


# ── Line 135: _patient_dir path traversal guard ─────────────────────────────


class TestPatientDirPathTraversal:
    """Line 135: os.path.realpath resolves to outside data directory (symlink attack)."""

    def test_path_escapes_data_directory(self, engine):
        with patch("os.path.realpath", side_effect=["/evil/outside/test", "/data/safe"]):
            with pytest.raises(ValueError, match="escapes data directory"):
                engine._patient_dir("test-patient")
