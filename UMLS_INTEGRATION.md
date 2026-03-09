# UMLS Integration Instructions for ClinicalMem

> STARGA Inc. | UMLS License: Approved (Nikolai Nedovodin)
> UTS API Key required — get from https://uts.nlm.nih.gov/uts/profile

---

## What This Unlocks

The UMLS license gives ClinicalMem access to real medical terminology databases,
replacing hardcoded lists with production-grade clinical vocabulary:

| Resource | Replaces | Impact |
|----------|----------|--------|
| **RxNorm API** | 12 hardcoded drug interaction pairs | Thousands of validated interaction pairs via RxCUI lookup |
| **RxNorm normalization** | Fuzzy substring matching on drug names | Canonical drug name resolution ("Glucophage" → "metformin") |
| **SNOMED CT** | 4 hardcoded allergy cross-reaction groups | Full allergy hierarchy with coded cross-reactivity |
| **UMLS Metathesaurus** | String matching across vocabularies | Cross-vocabulary concept mapping (ICD-10 ↔ SNOMED CT ↔ LOINC) |
| **VSAC value sets** | Manual condition lists | Standard clinical value sets for eCQM and C-CDA compliance |

---

## Implementation Plan (4 modules)

### Module 1: `engine/rxnorm_client.py` — RxNorm API Client

**Purpose:** Normalize medication names to RxCUI codes and check interactions via the NIH RxNorm API.

**APIs (all free, no auth required for basic endpoints):**
- `GET https://rxnav.nlm.nih.gov/REST/rxcui.json?name={drugName}` — name → RxCUI
- `GET https://rxnav.nlm.nih.gov/REST/approximateTerm.json?term={term}` — fuzzy name → RxCUI
- `GET https://rxnav.nlm.nih.gov/REST/interaction/list.json?rxcuis={rxcui1}+{rxcui2}` — pairwise interaction check
- `GET https://rxnav.nlm.nih.gov/REST/interaction/interaction.json?rxcui={rxcui}` — all interactions for a drug
- `GET https://rxnav.nlm.nih.gov/REST/rxcui/{rxcui}/allrelated.json` — related concepts (brand ↔ generic ↔ ingredient)

**Implementation:**

```python
"""RxNorm API client — medication normalization and interaction lookup."""

import httpx
import logging
from dataclasses import dataclass
from functools import lru_cache

logger = logging.getLogger(__name__)

RXNORM_BASE = "https://rxnav.nlm.nih.gov/REST"


@dataclass(frozen=True)
class RxConcept:
    """A normalized RxNorm concept."""
    rxcui: str
    name: str
    tty: str  # term type: IN (ingredient), SBD (branded drug), SCD (clinical drug), etc.


@lru_cache(maxsize=512)
def resolve_rxcui(drug_name: str) -> RxConcept | None:
    """
    Resolve a drug name to its RxNorm concept (RxCUI).

    Tries exact match first, falls back to approximate term search.
    Caches results to avoid repeated API calls.
    """
    # 1. Exact match
    # GET /rxcui.json?name={drug_name}&search=1

    # 2. If no exact match, try approximate
    # GET /approximateTerm.json?term={drug_name}&maxEntries=1

    # 3. For ingredients (most useful for interaction checking),
    #    follow allrelated to get the base ingredient RxCUI
    pass


def get_interactions_for_list(rxcuis: list[str]) -> list[dict]:
    """
    Check all pairwise interactions for a list of RxCUI codes.

    GET /interaction/list.json?rxcuis={rxcui1}+{rxcui2}+...

    Returns interaction pairs with:
    - severity (from interactionPair.severity)
    - description (from interactionPair.description)
    - source (DrugBank, ONCHigh, etc.)
    """
    pass


def normalize_medication_list(medications: list[str]) -> dict[str, RxConcept | None]:
    """
    Normalize a list of medication names to RxCUI codes.

    Returns {original_name: RxConcept or None}.
    Logs warnings for unresolved medications.
    """
    pass
```

**Integration point in `clinical_scoring.py`:**

Replace the current Layer 3 (`_nih_check_interactions`) with a proper RxNorm pipeline:

```python
# In check_drug_interactions(), replace Layer 3:

# Layer 3: RxNorm API (normalized drug names + comprehensive interaction DB)
if use_llm_fallback and len(meds_lower) >= 2:
    from engine.rxnorm_client import normalize_medication_list, get_interactions_for_list

    # Normalize all medication names to RxCUI codes
    resolved = normalize_medication_list(medications)
    rxcuis = [rc.rxcui for rc in resolved.values() if rc is not None]

    if len(rxcuis) >= 2:
        rxnorm_interactions = get_interactions_for_list(rxcuis)
        # Convert to DrugInteraction objects, skip already-covered pairs
        for ri in rxnorm_interactions:
            pair = (ri["drug_a"].lower(), ri["drug_b"].lower())
            if pair not in covered_pairs and tuple(reversed(pair)) not in covered_pairs:
                interactions.append(DrugInteraction(
                    drug_a=ri["drug_a"],
                    drug_b=ri["drug_b"],
                    severity=ri["severity"],
                    description=ri["description"],
                    score=medication_severity_score(ri["severity"]),
                ))
                covered_pairs.add(pair)
```

---

### Module 2: `engine/snomed_client.py` — SNOMED CT Lookup

**Purpose:** Code-based condition and allergy matching instead of string matching.

**API (requires UMLS API key for full access):**
- `GET https://browser.ihtsdotools.org/snowstorm/snomed-ct/MAIN/concepts?term={term}&limit=5` — free SNOMED browser
- UMLS REST API: `GET https://uts-ws.nlm.nih.gov/rest/search/current?string={term}&sab=SNOMEDCT_US&apiKey={key}` — full SNOMED search

**Key SNOMED hierarchies for ClinicalMem:**
- `91936005` — Allergy to penicillin (check descendants for cross-reactivity)
- `416098002` — Drug allergy (parent concept)
- `419511003` — Propensity to adverse reactions to drug
- `281647001` — Adverse reaction (qualifier value)

**Implementation:**

```python
"""SNOMED CT client — coded clinical terminology lookup."""

import os
import httpx
import logging
from dataclasses import dataclass
from functools import lru_cache

logger = logging.getLogger(__name__)

UMLS_API_KEY = os.environ.get("UMLS_API_KEY", "")
UMLS_BASE = "https://uts-ws.nlm.nih.gov/rest"
SNOMED_BROWSER = "https://browser.ihtsdotools.org/snowstorm/snomed-ct"


@dataclass(frozen=True)
class SnomedConcept:
    concept_id: str
    term: str
    semantic_tag: str  # "disorder", "substance", "finding", etc.


@lru_cache(maxsize=256)
def search_snomed(term: str, semantic_tag: str | None = None) -> list[SnomedConcept]:
    """
    Search SNOMED CT for a clinical term.

    Uses free Snowstorm browser API first. Falls back to UMLS API
    if UMLS_API_KEY is set (provides broader coverage).
    """
    pass


def is_allergy_cross_reactive(allergy_code: str, medication_code: str) -> bool:
    """
    Check if a medication is cross-reactive with a known allergy
    using SNOMED CT hierarchy traversal.

    Example: allergy to penicillin (91936005) → check if amoxicillin
    is a descendant of the penicillin drug class.
    """
    pass


def map_fhir_code_to_snomed(coding: dict) -> SnomedConcept | None:
    """
    Extract SNOMED CT concept from a FHIR CodeableConcept coding entry.

    FHIR codes may use system: http://snomed.info/sct
    If the code is ICD-10 or LOINC, use UMLS crosswalk to find SNOMED equivalent.
    """
    pass
```

**Integration point in `clinical_scoring.py`:**

Enhance `check_allergy_conflicts()` to use SNOMED hierarchy:

```python
# After the existing _ALLERGY_CROSS_REACTIONS check, add SNOMED fallback:
if not conflicts and UMLS_API_KEY:
    from engine.snomed_client import is_allergy_cross_reactive
    for allergy in allergies:
        for med in medications:
            if is_allergy_cross_reactive(allergy, med):
                conflicts.append(AllergyConflict(...))
```

---

### Module 3: `engine/umls_mapper.py` — Cross-Vocabulary Concept Mapping

**Purpose:** Map between ICD-10, SNOMED CT, LOINC, RxNorm, and MeSH for unified concept matching.

**API:**
- `GET https://uts-ws.nlm.nih.gov/rest/crosswalk/current/source/{source}/{id}?targetSource={target}&apiKey={key}`
- Example: ICD-10 `E11.9` (T2DM) → SNOMED CT `44054006`

**Implementation:**

```python
"""UMLS Metathesaurus — cross-vocabulary concept mapping."""

import os
import httpx
import logging
from dataclasses import dataclass
from functools import lru_cache

logger = logging.getLogger(__name__)

UMLS_API_KEY = os.environ.get("UMLS_API_KEY", "")
UMLS_BASE = "https://uts-ws.nlm.nih.gov/rest"

# Source abbreviations
SAB_SNOMEDCT = "SNOMEDCT_US"
SAB_ICD10CM = "ICD10CM"
SAB_LOINC = "LNC"
SAB_RXNORM = "RXNORM"
SAB_MESH = "MSH"


@dataclass(frozen=True)
class UMLSConcept:
    cui: str          # Concept Unique Identifier (C0011849 = Diabetes Mellitus)
    name: str
    source: str       # Which vocabulary
    source_code: str  # Code in that vocabulary


@lru_cache(maxsize=256)
def crosswalk(source: str, code: str, target: str) -> list[UMLSConcept]:
    """
    Map a code from one vocabulary to another via UMLS crosswalk.

    Examples:
        crosswalk("ICD10CM", "E11.9", "SNOMEDCT_US") → T2DM SNOMED concepts
        crosswalk("RXNORM", "6809", "SNOMEDCT_US")   → Metformin SNOMED concepts
        crosswalk("SNOMEDCT_US", "44054006", "ICD10CM") → T2DM ICD-10 codes
    """
    pass


@lru_cache(maxsize=256)
def find_concept(term: str, source: str | None = None) -> list[UMLSConcept]:
    """
    Search UMLS Metathesaurus for a clinical concept.

    If source is specified, restricts to that vocabulary.
    Returns up to 5 matching concepts sorted by relevance.
    """
    pass


def are_same_concept(code_a: tuple[str, str], code_b: tuple[str, str]) -> bool:
    """
    Check if two coded concepts (source, code) refer to the same UMLS CUI.

    Enables cross-vocabulary matching:
        are_same_concept(("ICD10CM", "E11.9"), ("SNOMEDCT_US", "44054006")) → True
    """
    pass
```

**Integration point in `clinical_memory.py`:**

Use UMLS mapping when ingesting FHIR resources to enrich blocks with cross-vocabulary codes:

```python
# In _ingest_condition(), after extracting the condition name:
if UMLS_API_KEY:
    from engine.umls_mapper import find_concept, crosswalk
    # Enrich the block with SNOMED CT code if FHIR only has ICD-10
    # This enables better recall matching across vocabularies
```

---

### Module 4: Environment & Config

**Add to `.env` (or Cloud Run env vars):**
```bash
UMLS_API_KEY=your-uts-api-key-here
```

**Get your API key:**
1. Go to https://uts.nlm.nih.gov/uts/profile
2. Log in with your approved UTS account
3. Copy the API key from your profile page

**Add to `pyproject.toml` dependencies:**
```toml
# No new deps needed — httpx is already included
# RxNorm API and UMLS API are REST endpoints, no SDK required
```

**Add to `engine/__init__.py` exports:**
```python
from engine.rxnorm_client import resolve_rxcui, normalize_medication_list, get_interactions_for_list
from engine.snomed_client import search_snomed, is_allergy_cross_reactive
from engine.umls_mapper import crosswalk, find_concept, are_same_concept
```

---

## Integration Priority (build in this order)

### Phase 1: RxNorm Drug Normalization (highest impact, no API key needed)
1. Build `engine/rxnorm_client.py` with `resolve_rxcui()` and `get_interactions_for_list()`
2. Replace Layer 3 in `check_drug_interactions()` with RxNorm pipeline
3. Add tests: `tests/test_engine/test_rxnorm_client.py`
4. **Impact:** 12 hardcoded pairs → thousands of validated interactions

### Phase 2: SNOMED CT Allergy Hierarchy (needs UMLS_API_KEY)
1. Build `engine/snomed_client.py` with `search_snomed()` and `is_allergy_cross_reactive()`
2. Enhance `check_allergy_conflicts()` with SNOMED fallback
3. Add tests: `tests/test_engine/test_snomed_client.py`
4. **Impact:** 4 cross-reaction groups → full coded allergy hierarchy

### Phase 3: UMLS Cross-Vocabulary Mapping (needs UMLS_API_KEY)
1. Build `engine/umls_mapper.py` with `crosswalk()` and `find_concept()`
2. Enrich FHIR ingestion with cross-vocabulary codes
3. Add tests: `tests/test_engine/test_umls_mapper.py`
4. **Impact:** String matching → coded concept matching across ICD-10, SNOMED, LOINC, RxNorm

---

## Testing Strategy

**Unit tests:** Mock the HTTP responses from RxNorm/UMLS APIs. Store fixture responses in `tests/fixtures/`:
- `rxnorm_metformin.json` — RxCUI lookup response for metformin
- `rxnorm_interactions_warfarin_ibuprofen.json` — interaction check response
- `snomed_penicillin_allergy.json` — SNOMED hierarchy response
- `umls_crosswalk_t2dm.json` — ICD-10 → SNOMED crosswalk response

**Integration tests:** Hit real APIs (mark with `@pytest.mark.network`):
```python
@pytest.mark.network
def test_rxnorm_resolves_metformin():
    concept = resolve_rxcui("metformin")
    assert concept is not None
    assert concept.rxcui == "6809"
    assert "metformin" in concept.name.lower()
```

**Run without network (default):**
```bash
python -m pytest tests/ -v -m "not network"
```

**Run with network (CI or manual):**
```bash
python -m pytest tests/ -v -m "network"
```

---

## Demo Enhancement

With UMLS integration, the Sarah Mitchell demo becomes much stronger:

**Before (hardcoded):**
> "Found 2 drug interactions from our database of 12 known pairs"

**After (RxNorm):**
> "Normalized 6 medications via RxNorm (RxCUI codes). Found 3 interactions from the NIH drug interaction database. Cross-referenced allergy to penicillin (SNOMED 91936005) against medication list — flagged amoxicillin as beta-lactam cross-reactive via SNOMED CT hierarchy."

This is what Josh Mandel (the judge — he wrote SMART on FHIR) wants to see: proper coded terminology, not string matching.

---

## Rate Limits

| API | Rate Limit | Notes |
|-----|-----------|-------|
| RxNorm REST | 20 req/sec | No auth needed, generous limits |
| UMLS REST | 20 req/sec per API key | Needs UMLS_API_KEY |
| Snowstorm SNOMED | 100 req/min | Free browser API, no auth |

Use `@lru_cache` on all lookup functions to minimize API calls. A typical patient with 6 medications needs ~10 API calls total (6 RxCUI lookups + 1 interaction list check + 3 SNOMED lookups).
