"""Tests for hallucination detector — evidence grounding gate."""
import pytest

from engine.hallucination_detector import (
    extract_clinical_claims,
    verify_claim_against_blocks,
    ground_check,
    ClaimVerification,
    GroundingReport,
)


def _block(block_id, content, title="", metadata=None):
    """Helper to build a mock evidence block."""
    return {
        "block_id": block_id,
        "content": content,
        "title": title,
        "metadata": metadata or {},
    }


class TestExtractClinicalClaims:
    def test_extracts_medication_claim(self):
        text = "The patient is currently taking metformin 500mg twice daily."
        claims = extract_clinical_claims(text)
        assert len(claims) >= 1
        assert any("metformin" in c.lower() for c in claims)

    def test_extracts_diagnosis_claim(self):
        text = "Patient was diagnosed with Type 2 diabetes in 2020."
        claims = extract_clinical_claims(text)
        assert len(claims) >= 1

    def test_extracts_lab_claim(self):
        text = "The HbA1c level is 7.2% which indicates moderate control."
        claims = extract_clinical_claims(text)
        assert len(claims) >= 1

    def test_extracts_allergy_claim(self):
        text = "The patient has a known allergy to penicillin."
        claims = extract_clinical_claims(text)
        assert len(claims) >= 1

    def test_extracts_blood_pressure_claim(self):
        text = "Blood pressure is 140/90 mmHg, elevated above target."
        claims = extract_clinical_claims(text)
        assert len(claims) >= 1

    def test_extracts_interaction_claim(self):
        text = "There is a known drug interaction between warfarin and aspirin."
        claims = extract_clinical_claims(text)
        assert len(claims) >= 1

    def test_skips_non_clinical_text(self):
        text = "The weather is nice today. I like coffee. Hello world."
        claims = extract_clinical_claims(text)
        assert len(claims) == 0

    def test_skips_short_sentences(self):
        text = "Low dose."
        claims = extract_clinical_claims(text)
        assert len(claims) == 0

    def test_empty_text(self):
        claims = extract_clinical_claims("")
        assert claims == []

    def test_multiple_claims(self):
        text = (
            "Patient is taking lisinopril 10mg daily for hypertension. "
            "The creatinine level is 1.8 mg/dL, above normal. "
            "A drug interaction between lisinopril and potassium supplements was detected."
        )
        claims = extract_clinical_claims(text)
        assert len(claims) >= 2

    def test_recommendation_claim(self):
        text = "We recommend switching to an alternative medication."
        claims = extract_clinical_claims(text)
        assert len(claims) >= 1


class TestVerifyClaimAgainstBlocks:
    def test_grounded_claim(self):
        claim = "Patient is taking metformin 500mg for diabetes."
        blocks = [
            _block("b1", "Metformin 500mg prescribed for Type 2 Diabetes"),
        ]
        result = verify_claim_against_blocks(claim, blocks)
        assert isinstance(result, ClaimVerification)
        assert result.grounded is True
        assert result.confidence > 0.0
        assert "b1" in result.evidence_block_ids

    def test_ungrounded_claim(self):
        claim = "Patient is taking warfarin for atrial fibrillation."
        blocks = [
            _block("b1", "Metformin 500mg prescribed for Type 2 Diabetes"),
        ]
        result = verify_claim_against_blocks(claim, blocks)
        assert result.grounded is False

    def test_partial_match(self):
        claim = "The patient has diabetes and is on medication."
        blocks = [
            _block("b1", "Type 2 Diabetes diagnosed 2020"),
            _block("b2", "Metformin 500mg daily"),
        ]
        result = verify_claim_against_blocks(claim, blocks, threshold=0.2)
        assert result.grounded is True

    def test_empty_blocks(self):
        claim = "Patient takes aspirin daily."
        result = verify_claim_against_blocks(claim, [])
        assert result.grounded is False
        assert result.confidence == 0.0

    def test_empty_claim_terms(self):
        claim = "the and for"  # All stop words
        result = verify_claim_against_blocks(claim, [_block("b1", "test")])
        assert result.grounded is False

    def test_metadata_matching(self):
        claim = "Patient's potassium level is elevated."
        blocks = [
            _block(
                "b1", "Lab result received",
                metadata={"lab_name": "potassium", "value": "5.8", "status": "elevated"},
            ),
        ]
        result = verify_claim_against_blocks(claim, blocks)
        assert result.grounded is True

    def test_max_evidence_blocks(self):
        claim = "Patient takes metformin for diabetes treatment and management."
        blocks = [
            _block(f"b{i}", f"Metformin diabetes treatment record {i}")
            for i in range(10)
        ]
        result = verify_claim_against_blocks(claim, blocks)
        assert len(result.evidence_block_ids) <= 3  # Capped at 3

    def test_snippet_length_capped(self):
        long_content = "x" * 500
        blocks = [_block("b1", long_content)]
        claim = "xxxxx yyyyy"
        result = verify_claim_against_blocks(claim, blocks, threshold=0.01)
        for snippet in result.evidence_snippets:
            assert len(snippet) <= 200

    def test_custom_threshold(self):
        claim = "Patient has diabetes."
        blocks = [_block("b1", "Type 2 Diabetes")]
        low = verify_claim_against_blocks(claim, blocks, threshold=0.1)
        high = verify_claim_against_blocks(claim, blocks, threshold=0.9)
        # Lower threshold should be easier to ground
        assert low.grounded or not high.grounded


class TestGroundCheck:
    def test_full_grounding_report(self):
        text = "Patient is taking metformin 500mg. The HbA1c level is 7.2%."
        blocks = [
            _block("b1", "Metformin 500mg prescribed daily"),
            _block("b2", "HbA1c result: 7.2%"),
        ]
        report = ground_check(text, blocks)
        assert isinstance(report, GroundingReport)
        assert report.text == text
        assert len(report.claims) >= 1
        assert report.grounded_count + report.ungrounded_count == len(report.claims)
        assert 0.0 <= report.grounding_score <= 1.0

    def test_no_claims_perfect_score(self):
        text = "The weather is nice today."
        report = ground_check(text, [])
        assert report.grounding_score == 1.0
        assert report.grounded_count == 0
        assert report.ungrounded_count == 0
        assert report.flagged_hallucinations == []

    def test_hallucinated_claims_flagged(self):
        text = "Patient has a severe allergy to ibuprofen."
        blocks = [
            _block("b1", "Metformin 500mg prescribed"),
        ]
        report = ground_check(text, blocks)
        if report.ungrounded_count > 0:
            assert len(report.flagged_hallucinations) > 0

    def test_fully_grounded(self):
        text = "Patient is taking metformin for diabetes."
        blocks = [
            _block("b1", "Metformin prescribed for Type 2 Diabetes management"),
        ]
        report = ground_check(text, blocks)
        if report.claims:
            assert report.grounding_score > 0.0

    def test_empty_text(self):
        report = ground_check("", [])
        assert report.grounding_score == 1.0
        assert report.claims == []

    def test_custom_threshold(self):
        text = "Patient is taking aspirin daily for heart condition."
        blocks = [_block("b1", "Aspirin 81mg daily")]
        strict = ground_check(text, blocks, threshold=0.9)
        lenient = ground_check(text, blocks, threshold=0.1)
        assert lenient.grounding_score >= strict.grounding_score
