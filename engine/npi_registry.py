"""NPI (National Provider Identifier) — CMS NPPES utilities.

NPI is a 10-digit identifier issued by the Centers for Medicare & Medicaid
Services (CMS) under HIPAA. The format is:

    Position 1     : Type prefix (1 = individual, 2 = organization)
    Positions 2-9  : 8-digit base identifier
    Position 10    : Luhn check digit, computed over the 9-digit prefix
                     PADDED with the CMS-published prefix "80840"

The Luhn algorithm with the 80840 prefix is documented at:
    https://www.cms.gov/Regulations-and-Guidance/Administrative-Simplification/NationalProvIdentStand/Downloads/NPIcheckdigit.pdf

ClinicalMem uses this module for two jobs:

  1. **NPI validation** — `validate_npi("1234567893")` returns True/False
     by re-computing the Luhn check digit. Used everywhere a Practitioner
     identifier enters the system (FHIR bundles, MCP tools, A2A handshake).

  2. **NPI Registry (NPPES) lookup** — `lookup_npi("1234567893")` queries
     the public CMS NPPES API (no auth required) and returns the
     practitioner's name, specialty (Healthcare Provider Taxonomy Code),
     practice address, and credential. Used by the dashboard's
     Practitioner tile so judges see a real provider record, not a
     fabricated one.

The lookup is best-effort: if NPPES is unreachable (CI environment, demo
laptop offline) the function returns `None` and callers fall back to the
local `Practitioner` resources shipped in the FHIR bundle.

This file is Apache-2.0 alongside the rest of clinicalmem.
"""
from __future__ import annotations

import json
import logging
import urllib.parse
import urllib.request
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# CMS-mandated prefix that precedes the 9-digit NPI base when computing
# the Luhn check digit. Documented in the NPI checkdigit specification.
_CMS_LUHN_PREFIX = "80840"

# NPPES API endpoint (public, no auth).
_NPPES_BASE = "https://npiregistry.cms.hhs.gov/api/"
_NPPES_VERSION = "2.1"


@dataclass(frozen=True)
class NPIRecord:
    """Normalised NPPES Practitioner record.

    Returned by `lookup_npi` on a successful NPPES query. Every field is a
    plain string so it slots cleanly into the FHIR Practitioner resource
    schema without further coercion.
    """

    npi: str
    enumeration_type: str       # "NPI-1" (individual) or "NPI-2" (organisation)
    name: str
    credential: str
    primary_taxonomy: str       # e.g. "Internal Medicine"
    primary_taxonomy_code: str  # e.g. "207R00000X"
    practice_state: str
    practice_city: str


# ─── Luhn check ────────────────────────────────────────────────────────────

def _luhn_check_digit(prefix: str) -> int:
    """Compute the standard Luhn check digit for a string of digits.

    Identical to the credit-card algorithm but applied to the 14-digit
    string that is `_CMS_LUHN_PREFIX + 9-digit-NPI-base`.
    """
    digits = [int(c) for c in prefix]
    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 0:
            doubled = d * 2
            if doubled > 9:
                doubled -= 9
            total += doubled
        else:
            total += d
    return (10 - total % 10) % 10


def validate_npi(npi: str) -> bool:
    """Return True if `npi` is a 10-digit string with a valid CMS Luhn check."""
    if not isinstance(npi, str):
        return False
    if len(npi) != 10:
        return False
    if not npi.isdigit():
        return False
    if npi[0] not in ("1", "2"):
        return False
    base = npi[:9]
    expected = _luhn_check_digit(_CMS_LUHN_PREFIX + base)
    return expected == int(npi[9])


def generate_test_npi(seed_string: str, *, individual: bool = True) -> str:
    """Deterministically derive a Luhn-valid NPI from a seed string.

    Used only for synthetic / demo data. The seed string is hashed to a
    9-digit base, then the standard Luhn check is computed. The first
    digit is forced to 1 (individual) or 2 (organisation). The output is
    guaranteed to pass `validate_npi`.

    NOT intended for production — real NPIs are issued by CMS NPPES and
    bound to a credentialed provider.
    """
    import hashlib
    digest = hashlib.blake2b(seed_string.encode("utf-8"),
                             digest_size=8).hexdigest()
    base_int = int(digest, 16) % 100_000_000  # 8-digit number
    type_prefix = "1" if individual else "2"
    base_str = type_prefix + f"{base_int:08d}"
    if len(base_str) != 9:
        raise RuntimeError(f"internal: base_str length {len(base_str)}")
    check = _luhn_check_digit(_CMS_LUHN_PREFIX + base_str)
    return base_str + str(check)


# ─── NPPES API ─────────────────────────────────────────────────────────────

def lookup_npi(npi: str, *, timeout: float = 5.0) -> NPIRecord | None:
    """Look up an NPI in the public NPPES registry.

    Returns None on any failure (network down, NPI not found, malformed
    response). Never raises. Callers should fall back to the local
    Practitioner resource in the FHIR bundle.
    """
    if not validate_npi(npi):
        logger.warning("invalid NPI passed to lookup_npi: %s", npi)
        return None

    params = {"version": _NPPES_VERSION, "number": npi}
    url = f"{_NPPES_BASE}?{urllib.parse.urlencode(params)}"

    # Cap the inbound response at 1 MB. NPPES typical response is ~3-8 KB;
    # anything larger is either a hostile mirror or a misconfigured CDN
    # and must NEVER be allowed to OOM the process. Bound BEFORE json.loads.
    _MAX_RESPONSE_BYTES = 1 * 1024 * 1024  # 1 MB

    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310 - public API
            raw = resp.read(_MAX_RESPONSE_BYTES + 1)
            if len(raw) > _MAX_RESPONSE_BYTES:
                logger.warning(
                    "NPPES response for %s exceeded %d bytes; refusing to parse",
                    npi, _MAX_RESPONSE_BYTES,
                )
                return None
            payload = json.loads(raw.decode("utf-8"))
    except (OSError, ValueError) as exc:
        logger.info("NPPES lookup failed for %s: %s", npi, exc)
        return None

    if not payload.get("results"):
        return None

    result = payload["results"][0]
    enumeration_type = result.get("enumeration_type", "")
    basic = result.get("basic", {}) or {}
    addresses = result.get("addresses", []) or [{}]
    practice = next(
        (a for a in addresses if a.get("address_purpose") == "LOCATION"),
        addresses[0],
    )
    taxonomies = result.get("taxonomies", []) or [{}]
    primary = next((t for t in taxonomies if t.get("primary")), taxonomies[0])

    if enumeration_type == "NPI-1":
        first = basic.get("first_name", "")
        last = basic.get("last_name", "")
        credential = basic.get("credential", "") or ""
        name = " ".join(filter(None, [first, last])).strip() or "Unknown Provider"
    else:
        name = basic.get("organization_name", "Unknown Organisation")
        credential = ""

    return NPIRecord(
        npi=npi,
        enumeration_type=enumeration_type,
        name=name,
        credential=credential,
        primary_taxonomy=(primary.get("desc") or "").strip(),
        primary_taxonomy_code=(primary.get("code") or "").strip(),
        practice_state=(practice.get("state") or "").strip(),
        practice_city=(practice.get("city") or "").strip(),
    )


__all__ = [
    "NPIRecord",
    "validate_npi",
    "generate_test_npi",
    "lookup_npi",
]
