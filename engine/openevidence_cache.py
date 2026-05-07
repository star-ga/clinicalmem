"""
OpenEvidence cached fixture set for demo reliability when API key is pending.

When OPENEVIDENCE_API_KEY is not yet available the engine falls back to this
module so the dashboard and tests can demonstrate realistic responses with
real public clinical references.  The moment a live key is set the engine
uses the live API and this module is not called.

Provenance: all summaries are derived from publicly documented drug-interaction
mechanisms referenced in peer-reviewed literature and authoritative label
sources (FDA prescribing information, ACR Manual on Contrast Media, NIH/NCBI
PMC articles).  They are NOT proprietary OpenEvidence content — they are
synthesised from long-established clinical pharmacology facts.

Audit trail note: every DrugInteraction object produced from this module
receives a description prefixed with "[CACHED <date>]" and a structured INFO
log line, so reviewers can distinguish cached entries from live API responses.
"""

import hashlib
import json
import logging
import os
from dataclasses import dataclass

logger = logging.getLogger(__name__)


def _hash_pair(drug_a: str, drug_b: str) -> str:
    """16-char SHA-256 prefix of the canonical pair key.

    PHI-safe identifier for log records: stable across calls, traceable
    to the audit-replay row, but never leaks the raw drug names. Same
    discipline as engine.bitnet_features_v8 OOV warnings.
    """
    a = drug_a.strip().lower()
    b = drug_b.strip().lower()
    if a > b:
        a, b = b, a
    canonical = f"{a}|{b}"
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]

# Path resolved relative to this file so the module works regardless of CWD.
_CACHE_JSON_PATH = os.path.join(
    os.path.dirname(__file__), "..", "docs", "openevidence_cache.json"
)


@dataclass(frozen=True)
class CachedOpenEvidenceResponse:
    """Serialisable record for a single cached OpenEvidence drug-pair response.

    Fields mirror the openevidence_cache.json schema so the demo dashboard
    can load the JSON directly and the engine can map to DrugInteraction
    without data loss.
    """

    drug_pair_canonical: tuple[str, str]   # lex-sorted (drug_a, drug_b)
    severity: str                           # "serious" | "contraindicated" | "moderate"
    clinical_summary: str                   # 2-3 sentence clinical summary
    evidence_urls: list[str]                # public literature / label references
    retrieved_at: str                       # ISO date string "YYYY-MM-DD"
    source: str                             # "CACHED" or "LIVE"


def canonical_pair_key(drug_a: str, drug_b: str) -> tuple[str, str]:
    """Return a lex-sorted (lowercase) pair key.

    Ensures lookup(warfarin, ibuprofen) == lookup(ibuprofen, warfarin).
    Mirrors the canonical encoding used by the BitNet classifier feature hash
    and the covered_pairs set in check_drug_interactions.
    """
    a = drug_a.strip().lower()
    b = drug_b.strip().lower()
    return (a, b) if a <= b else (b, a)


def _load_cache() -> list[CachedOpenEvidenceResponse]:
    """Load and parse openevidence_cache.json once per process lifetime.

    Returns an empty list and logs a warning if the file is missing or
    malformed — callers treat None from lookup_cached as "no cached entry"
    rather than an error, so degradation is graceful.
    """
    path = os.path.normpath(_CACHE_JSON_PATH)
    try:
        with open(path, encoding="utf-8") as fh:
            raw: list[dict] = json.load(fh)
    except FileNotFoundError:
        # PHI-safe: log filesystem path category not the literal path
        # (path may include user identifiers in some deployments).
        logger.warning(
            "openevidence_cache_not_found",
            extra={
                "path_basename": str(path).split("/")[-1] if path else "?",
            },
        )
        return []
    except json.JSONDecodeError as exc:
        # PHI-safe: error_type + decode position only — exc message can
        # quote the offending raw cache content. Same iter-234 / iter-239
        # discipline.
        logger.warning(
            "openevidence_cache_malformed",
            extra={
                "error_type": "JSONDecodeError",
                "decode_pos": getattr(exc, "pos", -1),
                "decode_msg_class": getattr(exc, "msg", "")[:40],
            },
        )
        return []

    entries: list[CachedOpenEvidenceResponse] = []
    for item in raw:
        try:
            pair_raw = item["drug_pair_canonical"]
            pair = canonical_pair_key(pair_raw[0], pair_raw[1])
            entries.append(
                CachedOpenEvidenceResponse(
                    drug_pair_canonical=pair,
                    severity=item["severity"],
                    clinical_summary=item["clinical_summary"],
                    evidence_urls=list(item.get("evidence_urls", [])),
                    retrieved_at=item.get("retrieved_at", ""),
                    source=item.get("source", "CACHED"),
                )
            )
        except (KeyError, IndexError, TypeError) as exc:
            # PHI-safe: log only structural metadata about the malformed
            # entry, NEVER the entry itself (`item` is a dict containing
            # drug names + clinical summary + evidence URLs — all PHI-
            # adjacent). Same iter-234 / iter-239 / iter-240 discipline.
            logger.warning(
                "openevidence_cache_skipped_malformed_entry",
                extra={
                    "error_type": type(exc).__name__,
                    "entry_keys": sorted(item.keys()) if isinstance(item, dict) else [],
                    "entry_type": type(item).__name__,
                },
            )

    # Iter-284: success-path DEBUG event so auditors can correlate every
    # cache lookup chain to the snapshot the engine loaded. PHI-safe:
    # entry count + per-severity tally + total URL count only — no drug
    # names, no clinical summaries.
    severity_counts: dict[str, int] = {}
    url_total = 0
    for entry in entries:
        severity_counts[entry.severity] = severity_counts.get(entry.severity, 0) + 1
        url_total += len(entry.evidence_urls)
    logger.debug(
        "openevidence_cache_loaded",
        extra={
            "entry_count": len(entries),
            "url_count_total": url_total,
            "severity_counts": severity_counts,
        },
    )

    return entries


# Module-level cache; loaded once on first call.
_CACHE: list[CachedOpenEvidenceResponse] | None = None


def _get_cache() -> list[CachedOpenEvidenceResponse]:
    global _CACHE
    if _CACHE is None:
        _CACHE = _load_cache()
    return _CACHE


def lookup_cached(drug_a: str, drug_b: str) -> CachedOpenEvidenceResponse | None:
    """Return the cached entry for a drug pair or None if not present.

    Lookup is lex-sorted and case-insensitive so argument order does not matter.

    Args:
        drug_a: First drug name (any case, any order).
        drug_b: Second drug name.

    Returns:
        Matching CachedOpenEvidenceResponse or None.
    """
    key = canonical_pair_key(drug_a, drug_b)
    for entry in _get_cache():
        if entry.drug_pair_canonical == key:
            # Iter-284: PHI-safe cache-hit DEBUG. Drug names hashed (not
            # raw); severity + URL count surfaced for auditor correlation.
            logger.debug(
                "openevidence_cache_hit",
                extra={
                    "pair_hash_prefix": _hash_pair(drug_a, drug_b),
                    "severity": entry.severity,
                    "evidence_url_count": len(entry.evidence_urls),
                    "source": entry.source,
                },
            )
            return entry
    # Iter-284: PHI-safe cache-miss DEBUG. Hashed pair only.
    logger.debug(
        "openevidence_cache_miss",
        extra={
            "pair_hash_prefix": _hash_pair(drug_a, drug_b),
            "cache_size": len(_get_cache()),
        },
    )
    return None


def invalidate_cache() -> None:
    """Force the next lookup to reload openevidence_cache.json from disk.

    Useful in tests that temporarily replace the JSON file.
    """
    global _CACHE
    # Iter-284: cache-invalidation INFO event so auditors see when the
    # in-process snapshot was rotated. Carries pre-invalidation entry
    # count for context.
    prev_count = len(_CACHE) if _CACHE is not None else 0
    logger.info(
        "openevidence_cache_invalidated",
        extra={"prev_entry_count": prev_count},
    )
    _CACHE = None
