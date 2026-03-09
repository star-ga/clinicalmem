"""UMLS Metathesaurus — cross-vocabulary concept mapping.

Maps between ICD-10, SNOMED CT, LOINC, RxNorm, and MeSH via
the NLM UMLS REST API crosswalk endpoint.

Requires UMLS_API_KEY environment variable.
API docs: https://documentation.uts.nlm.nih.gov/rest/home.html
"""
import logging
import os
from dataclasses import dataclass
from functools import lru_cache

import httpx

logger = logging.getLogger(__name__)

UMLS_API_KEY = os.environ.get("UMLS_API_KEY", "")
UMLS_BASE = "https://uts-ws.nlm.nih.gov/rest"

_TIMEOUT = 10

# Source abbreviations
SAB_SNOMEDCT = "SNOMEDCT_US"
SAB_ICD10CM = "ICD10CM"
SAB_LOINC = "LNC"
SAB_RXNORM = "RXNORM"
SAB_MESH = "MSH"
SAB_CPT = "CPT"


@dataclass(frozen=True)
class UMLSConcept:
    """A concept from the UMLS Metathesaurus."""

    cui: str  # Concept Unique Identifier (e.g., C0011849 = Diabetes Mellitus)
    name: str
    source: str  # Which vocabulary (SNOMEDCT_US, ICD10CM, etc.)
    source_code: str  # Code in that vocabulary


@lru_cache(maxsize=256)
def crosswalk(source: str, code: str, target: str) -> list[UMLSConcept]:
    """Map a code from one vocabulary to another via UMLS crosswalk.

    Examples:
        crosswalk("ICD10CM", "E11.9", "SNOMEDCT_US") → T2DM SNOMED concepts
        crosswalk("RXNORM", "6809", "SNOMEDCT_US")   → Metformin SNOMED concepts
        crosswalk("SNOMEDCT_US", "44054006", "ICD10CM") → T2DM ICD-10 codes
    """
    if not UMLS_API_KEY:
        logger.debug("UMLS crosswalk skipped — no API key")
        return []

    try:
        resp = httpx.get(
            f"{UMLS_BASE}/crosswalk/current/source/{source}/{code}",
            params={"targetSource": target, "apiKey": UMLS_API_KEY},
            timeout=_TIMEOUT,
        )
        if resp.status_code != 200:
            return []

        data = resp.json()
        results_list = data.get("result", [])
        if isinstance(results_list, dict):
            results_list = results_list.get("results", [])

        return [
            UMLSConcept(
                cui=r.get("ui", ""),
                name=r.get("name", ""),
                source=target,
                source_code=r.get("ui", ""),
            )
            for r in results_list
            if r.get("ui") != "NONE"
        ]
    except Exception as e:
        logger.debug("UMLS crosswalk failed %s/%s → %s: %s", source, code, target, e)
        return []


@lru_cache(maxsize=256)
def find_concept(term: str, source: str | None = None) -> list[UMLSConcept]:
    """Search UMLS Metathesaurus for a clinical concept.

    If source is specified, restricts to that vocabulary.
    Returns up to 5 matching concepts sorted by relevance.
    """
    if not UMLS_API_KEY:
        return []

    try:
        params: dict[str, str | int] = {
            "string": term,
            "apiKey": UMLS_API_KEY,
            "pageSize": 5,
        }
        if source:
            params["sab"] = source

        resp = httpx.get(
            f"{UMLS_BASE}/search/current",
            params=params,
            timeout=_TIMEOUT,
        )
        if resp.status_code != 200:
            return []

        data = resp.json()
        results_list = data.get("result", {}).get("results", [])
        return [
            UMLSConcept(
                cui=r.get("ui", ""),
                name=r.get("name", ""),
                source=r.get("rootSource", source or ""),
                source_code=r.get("ui", ""),
            )
            for r in results_list
            if r.get("ui") != "NONE"
        ]
    except Exception as e:
        logger.debug("UMLS search failed for %s: %s", term, e)
        return []


def are_same_concept(
    code_a: tuple[str, str], code_b: tuple[str, str]
) -> bool:
    """Check if two coded concepts refer to the same UMLS CUI.

    Enables cross-vocabulary matching:
        are_same_concept(("ICD10CM", "E11.9"), ("SNOMEDCT_US", "44054006")) → True
    """
    if not UMLS_API_KEY:
        return False

    # Get CUI for code_a
    cui_a = _get_cui(code_a[0], code_a[1])
    if not cui_a:
        return False

    # Get CUI for code_b
    cui_b = _get_cui(code_b[0], code_b[1])
    if not cui_b:
        return False

    return cui_a == cui_b


@lru_cache(maxsize=256)
def _get_cui(source: str, code: str) -> str | None:
    """Get UMLS CUI for a source vocabulary code."""
    if not UMLS_API_KEY:
        return None

    try:
        resp = httpx.get(
            f"{UMLS_BASE}/content/current/source/{source}/{code}",
            params={"apiKey": UMLS_API_KEY},
            timeout=_TIMEOUT,
        )
        if resp.status_code != 200:
            return None

        data = resp.json()
        result = data.get("result", {})
        # Extract CUI from the concept URI
        concept_uri = result.get("concept", "")
        if concept_uri and "/" in concept_uri:
            return concept_uri.rsplit("/", 1)[-1]
        return None
    except Exception:
        return None


def enrich_with_crosswalk(
    source: str, code: str, display: str
) -> dict[str, str]:
    """Enrich a coded concept with cross-vocabulary mappings.

    Returns a dict of {vocabulary: code} for all known mappings.
    """
    mappings: dict[str, str] = {source: code}

    targets = [SAB_SNOMEDCT, SAB_ICD10CM, SAB_RXNORM, SAB_LOINC]
    for target in targets:
        if target == source:
            continue
        results = crosswalk(source, code, target)
        if results:
            mappings[target] = results[0].source_code

    return mappings
