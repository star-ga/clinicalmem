"""
PHI Detector — Protected Health Information detection and redaction.

Detects and redacts HIPAA-defined PHI categories in clinical text:
- Names, dates, phone numbers, emails, SSNs, MRNs
- Addresses, ZIP codes, account numbers
- URLs, IP addresses, device identifiers

Uses regex patterns — no external dependencies, deterministic, auditable.
"""
import re
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PHIMatch:
    """A single PHI detection."""

    category: str  # HIPAA category
    text: str  # The matched text
    start: int
    end: int
    confidence: float  # 0.0-1.0


@dataclass(frozen=True)
class PHIReport:
    """Full PHI scan report."""

    original_text: str
    redacted_text: str
    matches: list[PHIMatch]
    phi_count: int
    categories_found: list[str]
    is_safe: bool  # True if no PHI detected


# Compiled regex patterns for PHI categories
_PHI_PATTERNS: list[tuple[str, re.Pattern, float]] = [
    # SSN: 123-45-6789 or 123456789
    ("SSN", re.compile(r'\b\d{3}-?\d{2}-?\d{4}\b'), 0.9),

    # Phone numbers: (123) 456-7890, 123-456-7890, 1234567890
    ("PHONE", re.compile(
        r'\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b'
    ), 0.8),

    # Email addresses
    ("EMAIL", re.compile(
        r'\b[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}\b'
    ), 0.95),

    # Medical Record Numbers: MRN-123456, MRN: 123456, MRN 123456
    ("MRN", re.compile(
        r'\bMRN[-:\s]*\d{4,10}\b', re.IGNORECASE
    ), 0.95),

    # Dates of birth / specific dates: MM/DD/YYYY, MM-DD-YYYY, YYYY-MM-DD
    ("DATE", re.compile(
        r'\b(?:\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}|\d{4}[/\-]\d{1,2}[/\-]\d{1,2})\b'
    ), 0.7),

    # ZIP codes (5-digit or ZIP+4)
    ("ZIP", re.compile(r'\b\d{5}(?:-\d{4})?\b'), 0.5),

    # IP addresses
    ("IP_ADDRESS", re.compile(
        r'\b(?:\d{1,3}\.){3}\d{1,3}\b'
    ), 0.8),

    # Account/ID numbers (generic): various formats with "ID", "account", "number"
    ("ACCOUNT_NUMBER", re.compile(
        r'\b(?:account|acct|id|number)[-:\s#]*\d{6,}\b', re.IGNORECASE
    ), 0.7),

    # URLs (could contain PHI in query params)
    ("URL", re.compile(
        r'https?://[^\s<>"\']+', re.IGNORECASE
    ), 0.6),

    # Names preceded by common clinical prefixes
    ("NAME", re.compile(
        r'(?:(?:Dr|Mr|Mrs|Ms|Miss|Patient|Pt)\.?\s+)'
        r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})'
    ), 0.7),

    # Street addresses: number + street name
    ("ADDRESS", re.compile(
        r'\b\d{1,5}\s+(?:[A-Z][a-z]+\s+){1,3}'
        r'(?:St|Street|Ave|Avenue|Blvd|Boulevard|Dr|Drive|Rd|Road|Ln|Lane|Way|Ct|Court)\b'
    ), 0.75),
]


def detect_phi(text: str) -> list[PHIMatch]:
    """
    Scan text for all PHI categories.

    Returns a list of PHIMatch objects sorted by position.
    """
    matches = []
    for category, pattern, confidence in _PHI_PATTERNS:
        for m in pattern.finditer(text):
            matches.append(PHIMatch(
                category=category,
                text=m.group(),
                start=m.start(),
                end=m.end(),
                confidence=confidence,
            ))

    # Sort by position, deduplicate overlapping matches (keep higher confidence)
    matches.sort(key=lambda m: (m.start, -m.confidence))
    deduped = []
    last_end = -1
    for m in matches:
        if m.start >= last_end:
            deduped.append(m)
            last_end = m.end

    return deduped


def redact_phi(text: str, replacement: str = "[REDACTED]") -> tuple[str, list[PHIMatch]]:
    """
    Detect and redact all PHI from text.

    Returns (redacted_text, list_of_matches).
    """
    matches = detect_phi(text)
    if not matches:
        return text, []

    # Build redacted text by replacing matched spans
    parts = []
    last_pos = 0
    for m in matches:
        parts.append(text[last_pos:m.start])
        parts.append(f"[{m.category}_{replacement}]")
        last_pos = m.end
    parts.append(text[last_pos:])

    return "".join(parts), matches


def scan_phi(text: str) -> PHIReport:
    """
    Full PHI scan: detect, redact, and report.
    """
    redacted, matches = redact_phi(text)
    categories = sorted(set(m.category for m in matches))

    return PHIReport(
        original_text=text,
        redacted_text=redacted,
        matches=matches,
        phi_count=len(matches),
        categories_found=categories,
        is_safe=len(matches) == 0,
    )
