"""RxNorm API client — medication normalization and interaction lookup.

Uses NIH's free RxNorm REST API (no auth required) for:
- Drug name → RxCUI resolution (exact + fuzzy)
- Pairwise drug interaction checking
- Brand/generic/ingredient relationship traversal

API docs: https://lhncbc.nlm.nih.gov/RxNav/APIs/RxNormAPIs.html
"""
import logging
from dataclasses import dataclass
from functools import lru_cache

import httpx

logger = logging.getLogger(__name__)

RXNORM_BASE = "https://rxnav.nlm.nih.gov/REST"
_TIMEOUT = 8


@dataclass(frozen=True)
class RxConcept:
    """A normalized RxNorm concept."""

    rxcui: str
    name: str
    tty: str  # term type: IN (ingredient), SBD (branded drug), SCD (clinical drug)


@dataclass(frozen=True)
class RxInteraction:
    """A drug-drug interaction from the NIH interaction database."""

    drug_a: str
    drug_b: str
    rxcui_a: str
    rxcui_b: str
    severity: str
    description: str
    source: str  # DrugBank, ONCHigh, etc.


@lru_cache(maxsize=512)
def resolve_rxcui(drug_name: str) -> RxConcept | None:
    """Resolve a drug name to its RxNorm concept (RxCUI).

    Tries exact match first, falls back to approximate term search.
    """
    name = drug_name.strip()
    if not name:
        return None

    # Strip dosage info for better matching
    clean = name.split()[0] if " " in name else name

    # 1. Exact match
    try:
        resp = httpx.get(
            f"{RXNORM_BASE}/rxcui.json",
            params={"name": clean, "search": 2},
            timeout=_TIMEOUT,
        )
        if resp.status_code == 200:
            data = resp.json()
            ids = data.get("idGroup", {}).get("rxnormId", [])
            if ids:
                rxcui = ids[0]
                concept = _get_concept_properties(rxcui)
                if concept:
                    return concept
                return RxConcept(rxcui=rxcui, name=clean, tty="IN")
    except Exception as e:
        logger.debug("RxNorm exact lookup failed for %s: %s", clean, e)

    # 2. Approximate match
    try:
        resp = httpx.get(
            f"{RXNORM_BASE}/approximateTerm.json",
            params={"term": clean, "maxEntries": 3},
            timeout=_TIMEOUT,
        )
        if resp.status_code == 200:
            data = resp.json()
            candidates = data.get("approximateGroup", {}).get("candidate", [])
            if candidates:
                rxcui = candidates[0].get("rxcui", "")
                score = candidates[0].get("score", "0")
                if rxcui and int(score) >= 60:
                    concept = _get_concept_properties(rxcui)
                    if concept:
                        return concept
                    return RxConcept(rxcui=rxcui, name=clean, tty="IN")
    except Exception as e:
        logger.debug("RxNorm approximate lookup failed for %s: %s", clean, e)

    return None


@lru_cache(maxsize=512)
def _get_concept_properties(rxcui: str) -> RxConcept | None:
    """Get display name and term type for an RxCUI."""
    try:
        resp = httpx.get(
            f"{RXNORM_BASE}/rxcui/{rxcui}/properties.json",
            timeout=_TIMEOUT,
        )
        if resp.status_code == 200:
            props = resp.json().get("properties", {})
            return RxConcept(
                rxcui=props.get("rxcui", rxcui),
                name=props.get("name", ""),
                tty=props.get("tty", "IN"),
            )
    except Exception:
        pass
    return None


def get_ingredient_rxcui(rxcui: str) -> str:
    """Follow allrelated to get base ingredient RxCUI (most useful for interactions)."""
    try:
        resp = httpx.get(
            f"{RXNORM_BASE}/rxcui/{rxcui}/allrelated.json",
            timeout=_TIMEOUT,
        )
        if resp.status_code == 200:
            groups = resp.json().get("allRelatedGroup", {}).get("conceptGroup", [])
            for group in groups:
                if group.get("tty") == "IN":  # Ingredient
                    props = group.get("conceptProperties", [])
                    if props:
                        return props[0].get("rxcui", rxcui)
    except Exception:
        pass
    return rxcui


def get_interactions_for_list(rxcuis: list[str]) -> list[RxInteraction]:
    """Check all pairwise interactions for a list of RxCUI codes.

    Uses the NIH Drug Interaction API — same database used by Epic, Cerner,
    and all certified EHR systems.
    """
    if len(rxcuis) < 2:
        return []

    rxcui_str = "+".join(rxcuis)

    try:
        resp = httpx.get(
            f"{RXNORM_BASE}/interaction/list.json",
            params={"rxcuis": rxcui_str},
            timeout=_TIMEOUT,
        )
        if resp.status_code != 200:
            return []
        data = resp.json()
    except Exception as e:
        logger.warning("NIH interaction API failed: %s", e)
        return []

    results = []
    for group in data.get("fullInteractionTypeGroup", []):
        source = group.get("sourceName", "Unknown")
        for itype in group.get("fullInteractionType", []):
            for pair in itype.get("interactionPair", []):
                concepts = pair.get("interactionConcept", [])
                if len(concepts) < 2:
                    continue

                drug_a_info = concepts[0].get("minConceptItem", {})
                drug_b_info = concepts[1].get("minConceptItem", {})
                drug_a = drug_a_info.get("name", "").lower()
                drug_b = drug_b_info.get("name", "").lower()

                if not drug_a or not drug_b:
                    continue

                desc = pair.get("description", "Interaction detected")
                sev_text = pair.get("severity", "").lower()

                if "contraindicated" in sev_text:
                    severity = "contraindicated"
                elif any(
                    w in desc.lower()
                    for w in [
                        "serious", "major", "significant", "bleeding",
                        "serotonin", "qt prolongation", "avoid",
                    ]
                ):
                    severity = "serious"
                else:
                    severity = "moderate"

                results.append(
                    RxInteraction(
                        drug_a=drug_a,
                        drug_b=drug_b,
                        rxcui_a=drug_a_info.get("rxcui", ""),
                        rxcui_b=drug_b_info.get("rxcui", ""),
                        severity=severity,
                        description=desc[:300],
                        source=source,
                    )
                )

    return results


def normalize_medication_list(
    medications: list[str],
) -> dict[str, RxConcept | None]:
    """Normalize a list of medication names to RxCUI codes.

    Returns {original_name: RxConcept or None}.
    """
    resolved: dict[str, RxConcept | None] = {}
    for med in medications:
        concept = resolve_rxcui(med)
        resolved[med] = concept
        if concept:
            logger.debug("Resolved %s → RxCUI %s (%s)", med, concept.rxcui, concept.name)
        else:
            logger.warning("Could not resolve medication: %s", med)
    return resolved
