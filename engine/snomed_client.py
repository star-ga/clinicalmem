"""SNOMED CT client — coded clinical terminology lookup.

Provides allergy cross-reactivity checking and clinical concept resolution
using the free Snowstorm browser API and UMLS REST API.

Cross-reactivity hierarchy:
- Penicillin allergy (91936005) → descendants include amoxicillin, ampicillin
- Beta-lactam (35631009) → broader class including cephalosporins
"""
import logging
import os
from dataclasses import dataclass
from functools import lru_cache

import httpx

logger = logging.getLogger(__name__)

UMLS_API_KEY = os.environ.get("UMLS_API_KEY", "")
UMLS_BASE = "https://uts-ws.nlm.nih.gov/rest"
SNOMED_BROWSER = "https://browser.ihtsdotools.org/snowstorm/snomed-ct"

_TIMEOUT = 10

# Known SNOMED CT drug class hierarchies for cross-reactivity
_DRUG_CLASS_HIERARCHY: dict[str, list[str]] = {
    "penicillin": [
        "amoxicillin", "ampicillin", "piperacillin", "nafcillin",
        "oxacillin", "dicloxacillin", "ticarcillin",
    ],
    "cephalosporin": [
        "cephalexin", "cefazolin", "ceftriaxone", "cefuroxime",
        "cefdinir", "cefepime", "ceftazidime",
    ],
    "sulfonamide": [
        "sulfamethoxazole", "sulfasalazine", "celecoxib",
        "trimethoprim-sulfamethoxazole",
    ],
    "fluoroquinolone": [
        "ciprofloxacin", "levofloxacin", "moxifloxacin", "ofloxacin",
    ],
    "opioid": [
        "morphine", "codeine", "hydrocodone", "oxycodone",
        "hydromorphone", "fentanyl", "tramadol",
    ],
    "nsaid": [
        "ibuprofen", "naproxen", "aspirin", "ketorolac",
        "indomethacin", "meloxicam", "diclofenac", "piroxicam",
    ],
    "ace inhibitor": [
        "lisinopril", "enalapril", "ramipril", "captopril",
        "benazepril", "fosinopril", "quinapril",
    ],
    "statin": [
        "atorvastatin", "simvastatin", "rosuvastatin", "pravastatin",
        "lovastatin", "fluvastatin", "pitavastatin",
    ],
}


@dataclass(frozen=True)
class SnomedConcept:
    """A SNOMED CT concept."""

    concept_id: str
    term: str
    semantic_tag: str  # "disorder", "substance", "finding", etc.


@lru_cache(maxsize=256)
def search_snomed(
    term: str, semantic_tag: str | None = None
) -> list[SnomedConcept]:
    """Search SNOMED CT for a clinical term.

    Uses the free Snowstorm browser API. Falls back to UMLS if API key is set.
    """
    results = _search_snowstorm(term, semantic_tag)
    if not results and UMLS_API_KEY:
        results = _search_umls_snomed(term)
    return results


def _search_snowstorm(
    term: str, semantic_tag: str | None = None
) -> list[SnomedConcept]:
    """Search via the free SNOMED CT Snowstorm browser."""
    try:
        params: dict[str, str | int] = {"term": term, "limit": 5, "activeFilter": "true"}
        if semantic_tag:
            params["semanticTag"] = semantic_tag

        resp = httpx.get(
            f"{SNOMED_BROWSER}/MAIN/concepts",
            params=params,
            timeout=_TIMEOUT,
        )
        if resp.status_code != 200:
            return []

        data = resp.json()
        items = data.get("items", [])
        return [
            SnomedConcept(
                concept_id=str(item.get("conceptId", "")),
                term=item.get("fsn", {}).get("term", item.get("pt", {}).get("term", "")),
                semantic_tag=item.get("fsn", {}).get("term", "").split("(")[-1].rstrip(")") if "(" in item.get("fsn", {}).get("term", "") else "",
            )
            for item in items
        ]
    except Exception as e:
        logger.debug("Snowstorm search failed for %s: %s", term, e)
        return []


def _search_umls_snomed(term: str) -> list[SnomedConcept]:
    """Search SNOMED via UMLS REST API (requires UMLS_API_KEY)."""
    if not UMLS_API_KEY:
        return []
    try:
        resp = httpx.get(
            f"{UMLS_BASE}/search/current",
            params={
                "string": term,
                "sab": "SNOMEDCT_US",
                "apiKey": UMLS_API_KEY,
                "pageSize": 5,
            },
            timeout=_TIMEOUT,
        )
        if resp.status_code != 200:
            return []
        data = resp.json()
        results_list = data.get("result", {}).get("results", [])
        return [
            SnomedConcept(
                concept_id=r.get("ui", ""),
                term=r.get("name", ""),
                semantic_tag="",
            )
            for r in results_list
            if r.get("ui") != "NONE"
        ]
    except Exception as e:
        logger.debug("UMLS SNOMED search failed for %s: %s", term, e)
        return []


def is_allergy_cross_reactive(allergy: str, medication: str) -> bool:
    """Check if a medication is cross-reactive with a known allergy.

    Uses local drug class hierarchy first (fast, deterministic),
    then falls back to SNOMED CT hierarchy traversal if available.
    """
    allergy_lower = allergy.lower().strip()
    med_lower = medication.lower().strip()

    # Check local hierarchy first (with alias expansion)
    _ALIASES: dict[str, str] = {
        "sulfa": "sulfonamide",
        "penicillin v": "penicillin",
        "pen-vk": "penicillin",
    }
    expanded_allergy = _ALIASES.get(allergy_lower, allergy_lower)
    for class_name, members in _DRUG_CLASS_HIERARCHY.items():
        allergy_match = class_name in expanded_allergy or any(
            m in expanded_allergy for m in members
        )
        if allergy_match and any(m in med_lower for m in members):
            return True

    # Also check cross-class: penicillin allergy + cephalosporin (~2% cross-reactivity)
    if "penicillin" in allergy_lower:
        cephalosporins = _DRUG_CLASS_HIERARCHY.get("cephalosporin", [])
        if any(c in med_lower for c in cephalosporins):
            return True  # Flag for clinical review

    return False


def get_allergy_cross_reactions(allergy: str) -> list[str]:
    """Get all medications that cross-react with a given allergy."""
    allergy_lower = allergy.lower().strip()
    cross_reactive: list[str] = []

    for class_name, members in _DRUG_CLASS_HIERARCHY.items():
        if class_name in allergy_lower or any(m in allergy_lower for m in members):
            cross_reactive.extend(members)

    # Penicillin → cephalosporin cross-reactivity
    if "penicillin" in allergy_lower:
        cross_reactive.extend(_DRUG_CLASS_HIERARCHY.get("cephalosporin", []))

    return list(set(cross_reactive))


def map_fhir_code_to_snomed(coding: dict) -> SnomedConcept | None:
    """Extract SNOMED CT concept from a FHIR CodeableConcept coding entry."""
    system = coding.get("system", "")
    code = coding.get("code", "")
    display = coding.get("display", "")

    if "snomed" in system.lower() and code:
        return SnomedConcept(concept_id=code, term=display, semantic_tag="")

    # If not SNOMED, try to find via search
    if display:
        results = search_snomed(display)
        return results[0] if results else None

    return None
