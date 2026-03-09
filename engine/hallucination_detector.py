"""
Evidence Grounding Gate — FHIR-traced claim verification.

Extension of ClinicalMem's safe abstention philosophy: just as the system
refuses to answer when evidence is insufficient (Layer 6), this module
refuses to trust LLM claims that aren't traceable to audited FHIR evidence
blocks. Every claim in a clinical narrative must cite a verifiable source.
"""
import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ClaimVerification:
    """Result of verifying a single clinical claim."""

    claim: str
    grounded: bool
    evidence_block_ids: list[str]
    evidence_snippets: list[str]
    confidence: float  # 0.0-1.0


@dataclass(frozen=True)
class GroundingReport:
    """Full grounding report for an LLM-generated clinical text."""

    text: str
    claims: list[ClaimVerification]
    grounded_count: int
    ungrounded_count: int
    grounding_score: float  # fraction of claims that are grounded
    flagged_hallucinations: list[str]


def extract_clinical_claims(text: str) -> list[str]:
    """
    Extract individual clinical claims from LLM-generated text.

    Splits on sentence boundaries and filters for statements that
    make clinical assertions (medication, diagnosis, lab value, etc.).
    """
    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())

    clinical_indicators = [
        "prescribed", "taking", "medication", "drug", "dose", "mg",
        "allergy", "allergic", "condition", "diagnosis", "diagnosed",
        "lab", "result", "level", "value", "mmhg", "mg/dl", "ml/min",
        "gfr", "egfr", "inr", "hba1c", "a1c", "potassium", "creatinine",
        "blood pressure", "heart rate", "temperature", "weight",
        "interaction", "conflict", "contraindicated", "risk",
        "recommend", "suggest", "should", "must", "avoid",
        "history of", "presents with", "suffers from",
    ]

    claims = []
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence or len(sentence) < 10:
            continue
        lower = sentence.lower()
        if any(indicator in lower for indicator in clinical_indicators):
            claims.append(sentence)

    return claims


def verify_claim_against_blocks(
    claim: str,
    blocks: list[dict],
    threshold: float = 0.3,
) -> ClaimVerification:
    """
    Verify a single clinical claim against stored patient blocks.

    Uses term overlap + metadata matching to find supporting evidence.
    A claim is grounded if it has sufficient overlap with at least one block.
    """
    claim_lower = claim.lower()
    claim_terms = set(re.findall(r'\b\w{3,}\b', claim_lower))

    # Remove common stop words
    stop_words = {
        "the", "and", "for", "with", "this", "that", "has", "was", "are",
        "been", "being", "have", "had", "does", "did", "will", "would",
        "could", "should", "may", "might", "can", "patient", "currently",
        "also", "from", "into", "about", "their", "which", "when", "where",
    }
    claim_terms -= stop_words

    if not claim_terms:
        return ClaimVerification(
            claim=claim,
            grounded=False,
            evidence_block_ids=[],
            evidence_snippets=[],
            confidence=0.0,
        )

    best_score = 0.0
    evidence_ids = []
    evidence_snippets = []

    for block in blocks:
        block_text = (
            (block.get("content", "") + " " + block.get("title", "")).lower()
        )
        # Include metadata values
        metadata = block.get("metadata", {})
        meta_text = " ".join(str(v) for v in metadata.values()).lower()
        full_text = block_text + " " + meta_text

        block_terms = set(re.findall(r'\b\w{3,}\b', full_text))
        block_terms -= stop_words

        if not block_terms:
            continue

        overlap = len(claim_terms & block_terms)
        score = overlap / len(claim_terms)

        if score >= threshold:
            evidence_ids.append(block.get("block_id", "unknown"))
            # Extract the most relevant snippet
            snippet = block.get("content", "")[:200]
            evidence_snippets.append(snippet)
            best_score = max(best_score, score)

    grounded = best_score >= threshold and len(evidence_ids) > 0

    return ClaimVerification(
        claim=claim,
        grounded=grounded,
        evidence_block_ids=evidence_ids[:3],  # Top 3 evidence blocks
        evidence_snippets=evidence_snippets[:3],
        confidence=round(best_score, 3),
    )


def ground_check(
    text: str,
    patient_blocks: list[dict],
    threshold: float = 0.3,
) -> GroundingReport:
    """
    Full hallucination detection: extract claims, verify each against evidence.

    Returns a GroundingReport with per-claim verification and overall score.
    """
    claims = extract_clinical_claims(text)

    if not claims:
        return GroundingReport(
            text=text,
            claims=[],
            grounded_count=0,
            ungrounded_count=0,
            grounding_score=1.0,  # No claims = nothing to hallucinate
            flagged_hallucinations=[],
        )

    verifications = []
    for claim in claims:
        v = verify_claim_against_blocks(claim, patient_blocks, threshold)
        verifications.append(v)

    grounded = [v for v in verifications if v.grounded]
    ungrounded = [v for v in verifications if not v.grounded]

    score = len(grounded) / len(verifications) if verifications else 0.0

    return GroundingReport(
        text=text,
        claims=verifications,
        grounded_count=len(grounded),
        ungrounded_count=len(ungrounded),
        grounding_score=round(score, 3),
        flagged_hallucinations=[v.claim for v in ungrounded],
    )
