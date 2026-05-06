"""
FDA Client — openFDA drug safety alerts integration.

Queries the openFDA API for:
- Drug adverse events (FAERS database)
- Drug labeling (FDA-approved labels with black box warnings)
- Drug recalls

Free API, no authentication required, rate limited to 240 req/min.
Same data source used by pharmacovigilance teams worldwide.

API docs: https://open.fda.gov/apis/
"""
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# A response with > 100 results for a single drug is anomalous; the
# openFDA endpoints we call are pagesize-bounded by `limit`, so a
# burst above this threshold means either a misconfigured `limit`
# parameter or a server-side change worth surfacing.
_RESPONSE_SIZE_WARN_THRESHOLD = 100


@dataclass(frozen=True)
class FDAAlert:
    """A single FDA safety alert for a drug."""

    drug_name: str
    alert_type: str  # "adverse_event", "label_warning", "recall"
    severity: str  # "critical", "high", "moderate", "low"
    description: str
    source: str  # "openFDA/FAERS", "openFDA/label", "openFDA/recall"
    count: int  # number of reports (for adverse events)


@dataclass(frozen=True)
class FDASafetyProfile:
    """Complete FDA safety profile for a medication list."""

    medications: list[str]
    alerts: list[FDAAlert]
    black_box_warnings: list[str]
    total_adverse_events: int
    highest_severity: str


def _get_httpx():
    """Lazy import httpx."""
    try:
        import httpx
        return httpx
    except ImportError:
        return None


def get_adverse_events(
    drug_name: str, limit: int = 5
) -> list[FDAAlert]:
    """
    Query openFDA FAERS for top adverse events for a drug.

    Uses: https://api.fda.gov/drug/event.json
    """
    httpx = _get_httpx()
    if httpx is None:
        return []

    clean_name = drug_name.strip().lower()
    # PHI discipline: drug name is not PHI per HIPAA Safe Harbor, but the
    # caller may pass through free-text from a clinical note. Log
    # length + endpoint kind only; the literal string DOES go on the
    # wire to a public openFDA endpoint, but the local audit trail
    # stays clean.
    logger.debug(
        "fda_adverse_search_start",
        extra={
            "drug_name_length": len(clean_name),
            "limit": limit,
            "endpoint": "drug/event",
        },
    )
    try:
        resp = httpx.get(
            "https://api.fda.gov/drug/event.json",
            params={
                "search": f'patient.drug.medicinalproduct:"{clean_name}"',
                "count": "patient.reaction.reactionmeddrapt.exact",
                "limit": str(limit),
            },
            timeout=5,
        )
        if resp.status_code != 200:
            logger.warning(
                "fda_adverse_non_200",
                extra={
                    "drug_name_length": len(clean_name),
                    "status_code": resp.status_code,
                    "endpoint": "drug/event",
                },
            )
            return []

        data = resp.json()
        results = data.get("results", [])
        if len(results) > _RESPONSE_SIZE_WARN_THRESHOLD:
            logger.warning(
                "fda_adverse_oversize_response",
                extra={
                    "drug_name_length": len(clean_name),
                    "results_returned": len(results),
                    "threshold": _RESPONSE_SIZE_WARN_THRESHOLD,
                },
            )

        alerts = []
        for r in results:
            reaction = r.get("term", "Unknown reaction")
            count = r.get("count", 0)
            severity = "high" if count > 1000 else "moderate" if count > 100 else "low"
            alerts.append(FDAAlert(
                drug_name=drug_name,
                alert_type="adverse_event",
                severity=severity,
                description=f"{reaction} ({count:,} reports in FAERS)",
                source="openFDA/FAERS",
                count=count,
            ))
        return alerts

    except Exception as e:
        # PHI / secret discipline: httpx exception messages can carry
        # response-body fragments (server error pages echo the request
        # query string). Log error_type only; never str(e).
        logger.warning(
            "fda_adverse_failed",
            extra={
                "drug_name_length": len(clean_name),
                "error_type": type(e).__name__,
                "endpoint": "drug/event",
            },
        )
        return []


def get_label_warnings(drug_name: str) -> list[FDAAlert]:
    """
    Query openFDA drug labeling for black box warnings and warnings.

    Uses: https://api.fda.gov/drug/label.json
    """
    httpx = _get_httpx()
    if httpx is None:
        return []

    clean_name = drug_name.strip().lower()
    # PHI discipline: drug name length + endpoint kind only.
    logger.debug(
        "fda_label_search_start",
        extra={
            "drug_name_length": len(clean_name),
            "endpoint": "drug/label",
        },
    )
    try:
        resp = httpx.get(
            "https://api.fda.gov/drug/label.json",
            params={
                "search": f'openfda.generic_name:"{clean_name}"',
                "limit": "1",
            },
            timeout=5,
        )
        if resp.status_code != 200:
            logger.warning(
                "fda_label_non_200",
                extra={
                    "drug_name_length": len(clean_name),
                    "status_code": resp.status_code,
                    "endpoint": "drug/label",
                },
            )
            return []

        data = resp.json()
        results = data.get("results", [])
        if not results:
            return []

        label = results[0]
        alerts = []

        # Black box warnings (most severe)
        boxed = label.get("boxed_warning", [])
        if boxed:
            warning_text = boxed[0][:500] if boxed else ""
            alerts.append(FDAAlert(
                drug_name=drug_name,
                alert_type="label_warning",
                severity="critical",
                description=f"BLACK BOX WARNING: {warning_text}",
                source="openFDA/label",
                count=0,
            ))

        # Warnings and precautions
        warnings = label.get("warnings_and_precautions", []) or label.get("warnings", [])
        if warnings:
            warning_text = warnings[0][:300] if warnings else ""
            alerts.append(FDAAlert(
                drug_name=drug_name,
                alert_type="label_warning",
                severity="high",
                description=f"FDA Warning: {warning_text}",
                source="openFDA/label",
                count=0,
            ))

        # Contraindications from label
        contras = label.get("contraindications", [])
        if contras:
            contra_text = contras[0][:300] if contras else ""
            alerts.append(FDAAlert(
                drug_name=drug_name,
                alert_type="label_warning",
                severity="high",
                description=f"Contraindications: {contra_text}",
                source="openFDA/label",
                count=0,
            ))

        return alerts

    except Exception as e:
        logger.warning(
            "fda_label_failed",
            extra={
                "drug_name_length": len(clean_name),
                "error_type": type(e).__name__,
                "endpoint": "drug/label",
            },
        )
        return []


def get_drug_recalls(drug_name: str, limit: int = 3) -> list[FDAAlert]:
    """
    Query openFDA for recent drug recalls.

    Uses: https://api.fda.gov/drug/enforcement.json
    """
    httpx = _get_httpx()
    if httpx is None:
        return []

    clean_name = drug_name.strip().lower()
    # PHI discipline: drug name length + limit + endpoint kind only.
    logger.debug(
        "fda_recall_search_start",
        extra={
            "drug_name_length": len(clean_name),
            "limit": limit,
            "endpoint": "drug/enforcement",
        },
    )
    try:
        resp = httpx.get(
            "https://api.fda.gov/drug/enforcement.json",
            params={
                "search": f'openfda.generic_name:"{clean_name}"',
                "limit": str(limit),
                "sort": "report_date:desc",
            },
            timeout=5,
        )
        if resp.status_code != 200:
            logger.warning(
                "fda_recall_non_200",
                extra={
                    "drug_name_length": len(clean_name),
                    "status_code": resp.status_code,
                    "endpoint": "drug/enforcement",
                },
            )
            return []

        data = resp.json()
        results = data.get("results", [])

        alerts = []
        for r in results:
            classification = r.get("classification", "")
            reason = r.get("reason_for_recall", "Unknown reason")
            severity = (
                "critical" if classification == "Class I"
                else "high" if classification == "Class II"
                else "moderate"
            )
            alerts.append(FDAAlert(
                drug_name=drug_name,
                alert_type="recall",
                severity=severity,
                description=f"{classification} Recall: {reason[:300]}",
                source="openFDA/recall",
                count=0,
            ))
        return alerts

    except Exception as e:
        logger.warning(
            "fda_recall_failed",
            extra={
                "drug_name_length": len(clean_name),
                "error_type": type(e).__name__,
                "endpoint": "drug/enforcement",
            },
        )
        return []


def get_safety_profile(medications: list[str]) -> FDASafetyProfile:
    """
    Get comprehensive FDA safety profile for a list of medications.

    Queries adverse events, label warnings, and recalls for each drug.
    """
    all_alerts: list[FDAAlert] = []
    black_box: list[str] = []
    total_adverse = 0

    for med in medications:
        # Get all FDA data for this medication
        adverse = get_adverse_events(med)
        labels = get_label_warnings(med)
        recalls = get_drug_recalls(med)

        all_alerts.extend(adverse)
        all_alerts.extend(labels)
        all_alerts.extend(recalls)

        total_adverse += sum(a.count for a in adverse)

        for alert in labels:
            if "BLACK BOX" in alert.description:
                black_box.append(f"{med}: {alert.description}")

    # Sort by severity
    severity_rank = {"critical": 4, "high": 3, "moderate": 2, "low": 1}
    all_alerts.sort(
        key=lambda a: severity_rank.get(a.severity, 0), reverse=True
    )

    highest = "none"
    if all_alerts:
        highest = all_alerts[0].severity

    # Per-patient FDA safety-profile complete. Log counts only --
    # medication names are caller-supplied; cardinality is the
    # auditor-friendly summary.
    logger.info(
        "fda_safety_profile_complete",
        extra={
            "med_count": len(medications),
            "alert_count": len(all_alerts),
            "black_box_count": len(black_box),
            "total_adverse_events": total_adverse,
            "highest_severity": highest,
        },
    )

    return FDASafetyProfile(
        medications=medications,
        alerts=all_alerts,
        black_box_warnings=black_box,
        total_adverse_events=total_adverse,
        highest_severity=highest,
    )
