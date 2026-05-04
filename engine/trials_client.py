"""ClinicalTrials.gov API v2 client — clinical trial matching.

Matches patient conditions to active recruiting trials using the
ClinicalTrials.gov public API v2 (no auth required).

API docs: https://clinicaltrials.gov/data-api/api
"""
import logging
from dataclasses import dataclass
from functools import lru_cache
from urllib.parse import urlparse

import httpx

logger = logging.getLogger(__name__)

CTGOV_BASE = "https://clinicaltrials.gov/api/v2"
_TIMEOUT = 10
_ALLOWED_HOSTS = frozenset({"clinicaltrials.gov"})

# A response with > 200 studies for a single condition is anomalous;
# ClinicalTrials.gov rarely returns that many for a typical query, and a
# burst is either a misconfigured pageSize or a server-side change worth
# surfacing as WARNING for triage.
_RESPONSE_SIZE_WARN_THRESHOLD = 200


def _ssrf_guard(url: str) -> None:
    """Validate URL hostname against allowlist."""
    parsed = urlparse(url)
    if parsed.hostname not in _ALLOWED_HOSTS:
        raise ValueError(
            f"SSRF blocked: {parsed.hostname} not in ClinicalTrials.gov allowed hosts"
        )


@dataclass(frozen=True)
class ClinicalTrial:
    """A clinical trial from ClinicalTrials.gov."""

    nct_id: str
    title: str
    status: str
    conditions: tuple[str, ...]
    interventions: tuple[str, ...]
    phase: str
    enrollment: int | None
    start_date: str
    locations: tuple[str, ...]
    url: str


@dataclass(frozen=True)
class TrialMatchResult:
    """Result from matching a patient to trials."""

    patient_conditions: tuple[str, ...]
    matched_trials: tuple[ClinicalTrial, ...]
    total_found: int
    search_terms: tuple[str, ...]


@lru_cache(maxsize=128)
def search_trials(
    condition: str,
    status: str = "RECRUITING",
    max_results: int = 10,
) -> tuple[ClinicalTrial, ...]:
    """Search ClinicalTrials.gov for trials matching a condition.

    Args:
        condition: Medical condition to search for.
        status: Trial status filter (RECRUITING, ACTIVE_NOT_RECRUITING, etc.)
        max_results: Maximum trials to return.

    Returns:
        Tuple of ClinicalTrial (immutable).
    """
    cond = condition.strip()
    if not cond:
        return ()

    url = f"{CTGOV_BASE}/studies"
    _ssrf_guard(url)

    # PHI discipline: a free-text condition is clinical context (diagnosis
    # name) but caller-supplied; could carry adjacent narrative. Log length
    # + status filter only -- never the literal condition string in the
    # local audit trail. The string IS sent on the wire to a public API
    # (CT.gov), but the audit log stays clean.
    logger.debug(
        "trials_search_start",
        extra={
            "cond_length": len(cond),
            "status_filter": status,
            "max_results": max_results,
        },
    )

    try:
        resp = httpx.get(
            url,
            params={
                "query.cond": cond,
                "filter.overallStatus": status,
                "pageSize": min(max_results, 20),
                "format": "json",
            },
            timeout=_TIMEOUT,
        )
        if resp.status_code != 200:
            logger.warning(
                "trials_search_non_200",
                extra={
                    "cond_length": len(cond),
                    "status_code": resp.status_code,
                    "status_filter": status,
                },
            )
            return ()

        data = resp.json()
        studies = data.get("studies", [])
        if len(studies) > _RESPONSE_SIZE_WARN_THRESHOLD:
            # Anomalous response size -- worth surfacing for triage.
            logger.warning(
                "trials_search_oversize_response",
                extra={
                    "cond_length": len(cond),
                    "studies_returned": len(studies),
                    "threshold": _RESPONSE_SIZE_WARN_THRESHOLD,
                },
            )
        trials = []

        for study in studies:
            proto = study.get("protocolSection", {})
            ident = proto.get("identificationModule", {})
            status_mod = proto.get("statusModule", {})
            cond_mod = proto.get("conditionsModule", {})
            arms_mod = proto.get("armsInterventionsModule", {})
            design_mod = proto.get("designModule", {})
            contacts_mod = proto.get("contactsLocationsModule", {})

            nct_id = ident.get("nctId", "")
            title = ident.get("briefTitle", ident.get("officialTitle", ""))

            # Extract interventions
            interventions_raw = arms_mod.get("interventions", [])
            interventions = tuple(
                i.get("name", "") for i in interventions_raw if i.get("name")
            )

            # Extract phases
            phases = design_mod.get("phases", [])
            phase = phases[0] if phases else "N/A"

            # Extract enrollment
            enrollment_info = design_mod.get("enrollmentInfo", {})
            enrollment = enrollment_info.get("count")

            # Extract locations (first 3)
            locations_raw = contacts_mod.get("locations", [])
            locations = tuple(
                f"{loc.get('city', '')}, {loc.get('country', '')}"
                for loc in locations_raw[:3]
                if loc.get("city")
            )

            trials.append(
                ClinicalTrial(
                    nct_id=nct_id,
                    title=title[:200],
                    status=status_mod.get("overallStatus", status),
                    conditions=tuple(cond_mod.get("conditions", [])),
                    interventions=interventions,
                    phase=phase,
                    enrollment=enrollment,
                    start_date=status_mod.get("startDateStruct", {}).get(
                        "date", ""
                    ),
                    locations=locations,
                    url=f"https://clinicaltrials.gov/study/{nct_id}",
                )
            )

        logger.debug(
            "trials_search_complete",
            extra={
                "cond_length": len(cond),
                "result_count": len(trials),
                "status_filter": status,
            },
        )
        return tuple(trials)

    except Exception as e:
        # PHI discipline: httpx exception messages can carry response-body
        # fragments (server error pages echo request params). Log only the
        # exception type, never str(e) directly.
        logger.warning(
            "trials_search_failed",
            extra={
                "cond_length": len(cond),
                "error_type": type(e).__name__,
            },
        )
        return ()


def match_patient_to_trials(
    conditions: list[str],
    medications: list[str] | None = None,
    max_per_condition: int = 5,
) -> TrialMatchResult:
    """Match a patient's conditions to active recruiting trials.

    Searches for each condition separately, deduplicates by NCT ID,
    and returns combined results.

    Args:
        conditions: List of patient conditions to search.
        medications: Optional — not used for filtering yet, reserved for future.
        max_per_condition: Max trials per condition search.
    """
    if not conditions:
        return TrialMatchResult(
            patient_conditions=(),
            matched_trials=(),
            total_found=0,
            search_terms=(),
        )

    # Per-patient match start. PHI discipline: count of conditions only,
    # never the conditions themselves -- patient diagnoses can be a
    # quasi-identifier when combined with cohort size + geography.
    logger.info(
        "trials_match_start",
        extra={
            "condition_count": len(conditions),
            "med_count": len(medications) if medications else 0,
            "max_per_condition": max_per_condition,
        },
    )

    seen_nct: set[str] = set()
    duplicate_count = 0
    all_trials: list[ClinicalTrial] = []
    search_terms: list[str] = []

    for condition in conditions:
        cond = condition.strip()
        if not cond:
            continue
        search_terms.append(cond)
        trials = search_trials(cond, max_results=max_per_condition)
        for trial in trials:
            if trial.nct_id in seen_nct:
                duplicate_count += 1
                continue
            seen_nct.add(trial.nct_id)
            all_trials.append(trial)

    logger.info(
        "trials_match_complete",
        extra={
            "condition_count": len(conditions),
            "search_terms_used": len(search_terms),
            "matched_trial_count": len(all_trials),
            "duplicates_dropped": duplicate_count,
        },
    )

    return TrialMatchResult(
        patient_conditions=tuple(conditions),
        matched_trials=tuple(all_trials),
        total_found=len(all_trials),
        search_terms=tuple(search_terms),
    )
