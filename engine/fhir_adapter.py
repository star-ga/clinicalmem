# Copyright 2026 STARGA Inc. — Apache-2.0
"""SMART-on-FHIR R4 ingress adapter.

Accepts a FHIR R4 Bundle (collection or transaction) and maps every
recognised resource type into the ClinicalMem internal representation.

Supported resource types
------------------------
Patient · Practitioner · Condition · MedicationStatement ·
AllergyIntolerance · Observation

Auth model
----------
The adapter operates at library level.  Transport, token acquisition,
and the ``Authorization: Bearer <token>`` header are the caller's
responsibility (SMART-on-FHIR EHR launch or standalone flow).  The
adapter itself makes no network calls.

PHI scrubbing
-------------
Patient.identifier entries that carry an MRN or SSN system URI are
stripped before any field is surfaced in the result.  The count of
redacted tokens is reported in ``phi_redactions``.

Audit anchor
------------
``bundle_sha256`` is the SHA-256 of the canonical JSON encoding (keys
sorted, no whitespace) of the validated bundle dict.  Any downstream
audit system can re-derive the hash from the raw bundle to prove the
adapter processed exactly the bytes that arrived.
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


def _hash_patient_id(patient_id: str) -> str:
    """PHI-safe 16-char SHA-256 prefix of patient_id.

    iter-333 PHI extras-key migration: hash patient_id before logging
    via `extra={}`. Mirrors `engine/fhir_client.py::_hash_patient_id`
    (iter-332) and the iter-291/iter-284/iter-279/iter-309 PHI
    discipline pattern.
    """
    return hashlib.sha256((patient_id or "").encode("utf-8")).hexdigest()[:16]

# ── constants ──────────────────────────────────────────────────────────────────

_KNOWN_RESOURCE_TYPES: frozenset[str] = frozenset(
    {
        "Patient",
        "Practitioner",
        "Condition",
        "MedicationStatement",
        "AllergyIntolerance",
        "Observation",
        "Bundle",
        # common container types — accepted but not deeply parsed
        "Encounter",
        "Procedure",
        "DiagnosticReport",
        "DocumentReference",
        "Device",
        "Organization",
        "Location",
        "Medication",
        "Immunization",
        "CarePlan",
        "Goal",
        "MedicationRequest",
    }
)

# FHIR system URIs that carry PHI identifiers subject to Safe Harbor removal
_PHI_IDENTIFIER_SYSTEMS: frozenset[str] = frozenset(
    {
        "http://terminology.hl7.org/CodeSystem/v2-0203",  # MRN/SSN type coding
        "urn:oid:2.16.840.1.113883.4.1",                  # SSN OID
        "http://hl7.org/fhir/sid/us-ssn",                 # SSN FHIR URI
        # MRN — any system containing "mrn" (case-insensitive) is stripped
    }
)

_MRN_IDENTIFIER_CODES: frozenset[str] = frozenset({"MR", "MRN", "MRN-ID", "PI", "NH"})

_NPI_SYSTEM = "http://hl7.org/fhir/sid/us-npi"

# FHIR systems for clinical codes
_SNOMED_SYSTEM = "http://snomed.info/sct"
_ICD10_SYSTEM = "http://hl7.org/fhir/sid/icd-10"
_RXNORM_SYSTEM = "http://www.nlm.nih.gov/research/umls/rxnorm"
_LOINC_SYSTEM = "http://loinc.org"


# ── result types ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class NormalizedCondition:
    """A Condition resource reduced to its coding essentials."""

    resource_id: str
    display_text: str
    snomed_code: str | None
    icd10_code: str | None
    clinical_status: str


@dataclass(frozen=True)
class NormalizedMedication:
    """A MedicationStatement resource reduced to its coding essentials."""

    resource_id: str
    display_text: str
    rxnorm_code: str | None
    status: str
    prescriber_reference: str | None  # e.g. "Practitioner/prac-cardio"


@dataclass(frozen=True)
class NormalizedAllergy:
    """An AllergyIntolerance resource reduced to its essentials."""

    resource_id: str
    display_text: str
    criticality: str | None   # "high" | "low" | "unable-to-assess" | None
    allergy_type: str | None  # "allergy" | "intolerance"
    category: list[str]       # e.g. ["medication"]
    snomed_code: str | None


@dataclass(frozen=True)
class NormalizedObservation:
    """An Observation resource reduced to its essentials."""

    resource_id: str
    display_text: str
    loinc_code: str | None
    value: float | str | None    # numeric or string value
    unit: str | None
    effective_datetime: str | None


@dataclass
class ClinicalIngestResult:
    """Result of ingesting one FHIR R4 Bundle into ClinicalMem.

    All list fields are populated from the bundle; empty lists mean no
    resources of that type were found or all were rejected.

    ``rejected_resources``
        Entries the adapter could not map, each as ``(resource_type, reason)``.

    ``phi_redactions``
        Count of PHI tokens stripped from Patient.identifier entries.

    ``bundle_sha256``
        SHA-256 of the canonical JSON encoding of the validated bundle.
        Acts as the audit anchor for the 21 CFR Part 11 export layer.
    """

    patient_id: str
    normalized_medications: list[str]          # display names for check_drug_interactions()
    normalized_conditions: list[NormalizedCondition]
    allergies: list[NormalizedAllergy]
    observations: list[NormalizedObservation]
    practitioner_npis: list[str]
    rejected_resources: list[tuple[str, str]]  # (resource_type, reason)
    phi_redactions: int
    bundle_sha256: str

    # Rich structured views (kept alongside the flat name list for scoring)
    medications: list[NormalizedMedication] = field(default_factory=list)


# ── PHI scrubbing ─────────────────────────────────────────────────────────────

def _is_phi_identifier(identifier: dict) -> bool:
    """Return True if an identifier dict carries an MRN or SSN."""
    system = (identifier.get("system") or "").lower()
    if "ssn" in system or "social-security" in system:
        return True
    if "mrn" in system:
        return True

    # Check type coding codes
    type_block = identifier.get("type") or {}
    for coding in type_block.get("coding", []):
        code = (coding.get("code") or "").upper()
        sys_uri = (coding.get("system") or "").lower()
        if code in _MRN_IDENTIFIER_CODES:
            return True
        if "ssn" in sys_uri or "social-security" in sys_uri:
            return True

    # Fallback: system OID/URI membership
    for phi_sys in _PHI_IDENTIFIER_SYSTEMS:
        if phi_sys.lower() in system:
            return True

    return False


def _scrub_patient_identifiers(
    identifiers: list[dict],
) -> tuple[list[dict], int]:
    """Remove PHI identifiers; return (clean_list, redaction_count)."""
    clean: list[dict] = []
    count = 0
    for ident in identifiers:
        if _is_phi_identifier(ident):
            count += 1
            logger.debug(
                "phi_scrubber: removed identifier system=%s",
                ident.get("system", "<unknown>"),
            )
        else:
            clean.append(ident)
    return clean, count


# ── code extraction helpers ───────────────────────────────────────────────────

def _first_code_by_system(codings: list[dict], system: str) -> str | None:
    for c in codings:
        if (c.get("system") or "").startswith(system.rstrip("/")):
            return c.get("code") or None
    return None


def _display_text(concept: dict) -> str:
    text = concept.get("text")
    if text:
        return text
    for c in concept.get("coding", []):
        if c.get("display"):
            return c["display"]
    return "Unknown"


# ── resource parsers ──────────────────────────────────────────────────────────

def _parse_patient(resource: dict) -> tuple[str, int]:
    """Return (patient_id, phi_redactions) after scrubbing identifiers."""
    patient_id = resource.get("id") or "unknown-patient"
    raw_identifiers = resource.get("identifier") or []
    _, phi_count = _scrub_patient_identifiers(raw_identifiers)
    return patient_id, phi_count


def _parse_condition(resource: dict) -> NormalizedCondition | None:
    code_block = resource.get("code") or {}
    codings = code_block.get("coding") or []
    status_block = resource.get("clinicalStatus") or {}
    status_codings = status_block.get("coding") or []
    clinical_status = (
        status_codings[0].get("code") if status_codings else "unknown"
    )
    return NormalizedCondition(
        resource_id=resource.get("id") or "",
        display_text=_display_text(code_block),
        snomed_code=_first_code_by_system(codings, _SNOMED_SYSTEM),
        icd10_code=_first_code_by_system(codings, _ICD10_SYSTEM),
        clinical_status=clinical_status or "unknown",
    )


def _parse_medication_statement(
    resource: dict,
) -> NormalizedMedication | None:
    concept = resource.get("medicationCodeableConcept") or {}
    codings = concept.get("coding") or []
    info_source = resource.get("informationSource") or {}
    return NormalizedMedication(
        resource_id=resource.get("id") or "",
        display_text=_display_text(concept),
        rxnorm_code=_first_code_by_system(codings, _RXNORM_SYSTEM),
        status=resource.get("status") or "unknown",
        prescriber_reference=info_source.get("reference"),
    )


def _parse_allergy(resource: dict) -> NormalizedAllergy | None:
    code_block = resource.get("code") or {}
    codings = code_block.get("coding") or []
    category_raw = resource.get("category") or []
    return NormalizedAllergy(
        resource_id=resource.get("id") or "",
        display_text=_display_text(code_block),
        criticality=resource.get("criticality"),
        allergy_type=resource.get("type"),
        category=list(category_raw),
        snomed_code=_first_code_by_system(codings, _SNOMED_SYSTEM),
    )


def _parse_observation(resource: dict) -> NormalizedObservation | None:
    code_block = resource.get("code") or {}
    codings = code_block.get("coding") or []

    # Numeric value
    vq = resource.get("valueQuantity") or {}
    if vq:
        value: float | str | None = vq.get("value")
        unit: str | None = vq.get("unit")
    elif "valueString" in resource:
        value = resource["valueString"]
        unit = None
    elif "valueCodeableConcept" in resource:
        concept = resource["valueCodeableConcept"]
        value = _display_text(concept)
        unit = None
    else:
        value = None
        unit = None
        # DEBUG — Observation resource without any of the standard
        # value[x] fields. FHIR R4 R-spec allows this (an observation
        # MAY be value-less when used as a header for sub-component
        # observations), but operators auditing data-completeness
        # need visibility into how often this happens. PHI-safe:
        # resource id (FHIR identifier, synthetic in our cohort)
        # only — NEVER LOINC code or display text (clinical narrative).
        logger.debug(
            "fhir_observation_no_value",
            extra={
                "resource_id": resource.get("id", "?"),
                "loinc_code_present": bool(_first_code_by_system(codings, _LOINC_SYSTEM)),
            },
        )

    return NormalizedObservation(
        resource_id=resource.get("id") or "",
        display_text=_display_text(code_block),
        loinc_code=_first_code_by_system(codings, _LOINC_SYSTEM),
        value=value,
        unit=unit,
        effective_datetime=resource.get("effectiveDateTime"),
    )


def _extract_practitioner_npis(
    resource: dict,
    rejected: list[tuple[str, str]],
) -> list[str]:
    """Extract and Luhn-validate NPI identifiers from a Practitioner resource."""
    from engine.npi_registry import validate_npi

    npis: list[str] = []
    for ident in (resource.get("identifier") or []):
        system = ident.get("system") or ""
        if _NPI_SYSTEM not in system:
            continue
        npi_value = ident.get("value") or ""
        if validate_npi(npi_value):
            npis.append(npi_value)
        else:
            # WARNING — NPI Luhn-validation failure is a Practitioner
            # data-quality signal that operators must see at default
            # log levels. PHI-safe: NPIs are public CMS-registry data,
            # but we log only the first 4 digits (ATC-style organisation
            # prefix is enough for operators to spot a typo / synthetic-
            # collision pattern) plus the resource id (FHIR identifier,
            # synthetic in our cohort). NEVER log the full malformed
            # NPI — partial digits don't help and could leak attempted-
            # forgery patterns in a future real-EHR ingest.
            logger.warning(
                "fhir_practitioner_npi_luhn_failed",
                extra={
                    "npi_prefix4": (npi_value[:4] if isinstance(npi_value, str) else "?"),
                    "npi_len": (len(npi_value) if isinstance(npi_value, str) else 0),
                    "resource_id": resource.get("id", "?"),
                },
            )
            rejected.append(
                (
                    "Practitioner",
                    f"NPI '{npi_value}' failed CMS Luhn validation "
                    f"(resource id={resource.get('id', '?')})",
                )
            )
    return npis


# ── canonical hash ─────────────────────────────────────────────────────────────

def _canonical_sha256(bundle: dict) -> str:
    """SHA-256 of the canonical (sorted-keys, no whitespace) JSON encoding."""
    canonical = json.dumps(bundle, sort_keys=True, separators=(",", ":"),
                           ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ── public API ─────────────────────────────────────────────────────────────────

def ingest_bundle(
    bundle: dict | str,
    *,
    strict_resource_types: bool = False,
) -> ClinicalIngestResult:
    """Ingest a FHIR R4 Bundle into ClinicalMem.

    Parameters
    ----------
    bundle
        A FHIR R4 Bundle as a Python dict or a JSON string.
    strict_resource_types
        When True, resources with an unrecognised ``resourceType`` are added
        to ``rejected_resources``.  When False (default) they are silently
        skipped (tolerant mode for bundles containing profile-specific
        extensions).

    Returns
    -------
    ClinicalIngestResult
        Populated result; see class docstring for field semantics.

    Raises
    ------
    ValueError
        If ``bundle`` is not a valid FHIR Bundle structure (missing ``entry``
        array or wrong ``resourceType``).
    """
    # --- parse JSON string if needed ------------------------------------------
    if isinstance(bundle, str):
        try:
            bundle = json.loads(bundle)
        except json.JSONDecodeError as exc:
            # PHI-safe pre-raise log: position + msg-class only, NEVER
            # the raw input. JSONDecodeError exposes column/line of the
            # parse failure which can quote raw bundle content; exc.msg
            # is the canned phrase ("Expecting value", etc.) — safe.
            logger.error(
                "fhir_bundle_invalid_json",
                extra={
                    "error_type": "JSONDecodeError",
                    "decode_msg": exc.msg,
                    "input_size_bytes": len(bundle) if isinstance(bundle, str) else 0,
                },
            )
            raise ValueError(f"bundle is not valid JSON: {exc}") from exc

    if not isinstance(bundle, dict):
        logger.error(
            "fhir_bundle_wrong_python_type",
            extra={"actual_type": type(bundle).__name__},
        )
        raise ValueError("bundle must be a dict or JSON string")

    # --- structural validation -------------------------------------------------
    resource_type = bundle.get("resourceType")
    if resource_type != "Bundle":
        # resourceType is an FHIR R4 standard string — safe to log
        # categorically. A bundle with a wildly malformed resourceType
        # (e.g. very long, embedded HTML) is logged as length-only.
        logger.error(
            "fhir_bundle_wrong_resource_type",
            extra={
                "resource_type_len": (
                    len(resource_type) if isinstance(resource_type, str) else 0
                ),
                "resource_type_class": type(resource_type).__name__,
            },
        )
        raise ValueError(
            f"Expected resourceType='Bundle', got '{resource_type}'"
        )
    entries = bundle.get("entry")
    if not isinstance(entries, list):
        logger.error(
            "fhir_bundle_entry_not_array",
            extra={"entry_type_class": type(entries).__name__},
        )
        raise ValueError("Bundle.entry must be an array")

    bundle_type = bundle.get("type", "collection")
    if bundle_type not in ("collection", "transaction", "searchset", "batch",
                            "history", "document"):
        logger.error(
            "fhir_bundle_unsupported_type",
            extra={
                "bundle_type_len": (
                    len(bundle_type) if isinstance(bundle_type, str) else 0
                ),
            },
        )
        raise ValueError(f"Unsupported Bundle.type: '{bundle_type}'")

    # --- compute audit anchor before any mutation ------------------------------
    bundle_sha256 = _canonical_sha256(bundle)

    # PHI-safe entry log: structural metadata only — bundle_type is FHIR R4
    # standard, sha256 prefix is non-reversible, entry_count is a count.
    logger.debug(
        "fhir_bundle_ingest_start",
        extra={
            "bundle_type": bundle_type,
            "entry_count": len(entries),
            "bundle_sha256_prefix": bundle_sha256[:16],
            "strict_resource_types": strict_resource_types,
        },
    )

    # --- iterate entries -------------------------------------------------------
    patient_id = "unknown-patient"
    phi_redactions = 0
    conditions: list[NormalizedCondition] = []
    medications: list[NormalizedMedication] = []
    allergies: list[NormalizedAllergy] = []
    observations: list[NormalizedObservation] = []
    practitioner_npis: list[str] = []
    rejected: list[tuple[str, str]] = []

    for entry in entries:
        resource = entry.get("resource") or {}
        rt = resource.get("resourceType") or ""

        if not rt:
            # iter-443 T4 ratchet: closes the entry-without-resourceType
            # silent dispatch path. Operators auditing FHIR-bundle quality
            # need a per-entry signal — the iter-191 aggregate
            # `fhir_bundle_rejection_summary` shows counts by resource_type
            # but groups all "(unknown)" together, masking serial bundle-
            # builder bugs that emit many empty-resourceType entries.
            # PHI-safe: structural metadata only — no resource body, no
            # identifier values, no narrative text.
            logger.debug(
                "fhir_bundle_entry_no_resource_type",
                extra={
                    "entry_keys_present": sorted((entry or {}).keys()),
                    "resource_keys_present": sorted((resource or {}).keys()),
                },
            )
            rejected.append(("(unknown)", "entry.resource has no resourceType"))
            continue

        if rt not in _KNOWN_RESOURCE_TYPES:
            if strict_resource_types:
                # WARNING level when strict mode rejects a resource — operators
                # need to see this surface. resourceType names are FHIR R4
                # standard, public configuration — safe to log.
                logger.warning(
                    "fhir_bundle_unknown_resource_type_rejected",
                    extra={"resource_type": rt, "strict": True},
                )
                rejected.append((rt, f"unrecognised resourceType '{rt}'"))
            else:
                logger.debug(
                    "fhir_bundle_unknown_resource_type_skipped",
                    extra={"resource_type": rt, "strict": False},
                )
            continue

        try:
            if rt == "Patient":
                pid, phi_count = _parse_patient(resource)
                patient_id = pid
                phi_redactions += phi_count

            elif rt == "Condition":
                cond = _parse_condition(resource)
                if cond:
                    conditions.append(cond)
                else:
                    rejected.append((rt, "condition parsing returned None"))

            elif rt == "MedicationStatement":
                med = _parse_medication_statement(resource)
                if med:
                    medications.append(med)
                else:
                    rejected.append((rt, "medication parsing returned None"))

            elif rt == "AllergyIntolerance":
                allergy = _parse_allergy(resource)
                if allergy:
                    allergies.append(allergy)
                else:
                    rejected.append((rt, "allergy parsing returned None"))

            elif rt == "Observation":
                obs = _parse_observation(resource)
                if obs:
                    observations.append(obs)
                else:
                    rejected.append((rt, "observation parsing returned None"))

            elif rt == "Practitioner":
                npis = _extract_practitioner_npis(resource, rejected)
                practitioner_npis.extend(npis)
                # iter-443 T4 ratchet: closes the Practitioner-zero-NPI
                # silent dispatch path. _extract_practitioner_npis logs
                # per-NPI Luhn failures (L370) but a Practitioner whose
                # ALL identifiers are non-NPI systems (e.g. only internal
                # synthetic ids) yields an empty NPI list silently. The
                # Care-team-attestation flow (iter-401) hard-depends on
                # ≥1 valid NPI per Practitioner; operators need a debug-
                # level signal so a future EHR ingest with stripped
                # NPI identifiers fires visibly. PHI-safe: NPIs are
                # public CMS-registry data; only resource id (FHIR
                # identifier, synthetic in our cohort) + identifier-
                # system count surfaces.
                if not npis:
                    logger.debug(
                        "fhir_practitioner_zero_valid_npis",
                        extra={
                            "resource_id": resource.get("id", "?"),
                            "identifier_count": len(
                                resource.get("identifier") or []
                            ),
                            "patient_id_hash_prefix": _hash_patient_id(patient_id),
                        },
                    )

        except Exception as exc:  # noqa: BLE001 — surface per-resource errors as rejections
            # PHI-safe error log: exception TYPE only, never the message
            # body (FHIR exceptions can sometimes embed resource content
            # in their message, which would re-leak data).
            logger.warning(
                "fhir_bundle_parse_failure",
                extra={
                    "resource_type": rt,
                    "error_type": type(exc).__name__,
                },
            )
            rejected.append((rt, f"parse error: {exc}"))

    # Flat medication name list for check_drug_interactions() compatibility
    normalized_medication_names = [m.display_text for m in medications]

    # PHI-safe completion log: counts + sha-prefix + patient_id (post-scrub
    # internal identifier). WARNING when any resource was rejected; INFO
    # on the all-clean path. Drug names + condition codes + observation
    # values stay sealed — only counts surface.

    # iter-191 extension: per-rejected-resource-type DEBUG summary so
    # operators auditing data-quality drift can see WHICH resource
    # type families are dropping resources without parsing the full
    # rejected list. PHI-safe: structural type names only (e.g.
    # 'Practitioner', 'Condition' — FHIR R4 standard strings).
    if rejected:
        rejection_type_counts: dict[str, int] = {}
        for rt_rej, _reason in rejected:
            rejection_type_counts[rt_rej] = rejection_type_counts.get(rt_rej, 0) + 1
        logger.debug(
            "fhir_bundle_rejection_summary",
            extra={
                "patient_id_hash_prefix": _hash_patient_id(patient_id),
                "total_rejected": len(rejected),
                "by_resource_type": rejection_type_counts,
            },
        )

    log_fn = logger.warning if rejected else logger.info
    log_fn(
        "fhir_bundle_ingest_complete",
        extra={
            "patient_id_hash_prefix": _hash_patient_id(patient_id),
            "bundle_sha256_prefix": bundle_sha256[:16],
            "medication_count": len(medications),
            "condition_count": len(conditions),
            "allergy_count": len(allergies),
            "observation_count": len(observations),
            "practitioner_count": len(practitioner_npis),
            "rejected_count": len(rejected),
            "phi_redactions": phi_redactions,
        },
    )

    return ClinicalIngestResult(
        patient_id=patient_id,
        normalized_medications=normalized_medication_names,
        normalized_conditions=conditions,
        allergies=allergies,
        observations=observations,
        practitioner_npis=practitioner_npis,
        rejected_resources=rejected,
        phi_redactions=phi_redactions,
        bundle_sha256=bundle_sha256,
        medications=medications,
    )


__all__ = [
    "ClinicalIngestResult",
    "NormalizedCondition",
    "NormalizedMedication",
    "NormalizedAllergy",
    "NormalizedObservation",
    "ingest_bundle",
]
