"""
Clinical scoring — MIND kernel-inspired scoring for healthcare memory.

Implements the abstention, importance, and adversarial kernels from MIND Lang
as pure Python for hackathon use. These demonstrate MIND Lang's expressiveness
without requiring the commercial runtime.

Kernel sources: https://github.com/star-ga/mind/tree/main/mind/
"""
import logging
import math
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ClinicalConfidence:
    """Result of confidence gating — decides if we should answer or abstain."""

    score: float  # 0.0-1.0
    should_abstain: bool
    reason: str

    @property
    def level(self) -> str:
        if self.score >= 0.8:
            return "high"
        if self.score >= 0.5:
            return "moderate"
        return "low"


def confidence_gate(
    bm25_scores: list[float],
    entity_overlaps: list[float],
    score_weight: float = 0.6,
    overlap_weight: float = 0.4,
    abstention_threshold: float = 0.3,
) -> ClinicalConfidence:
    """
    Confidence gating from abstention.mind kernel.

    Decides whether the system has enough evidence to answer a clinical query.
    In healthcare, abstaining is better than guessing.

    Args:
        bm25_scores: BM25 retrieval scores for top results
        entity_overlaps: Entity overlap ratios between query and results
        score_weight: Weight for retrieval scores
        overlap_weight: Weight for entity overlap
        abstention_threshold: Below this, abstain from answering
    """
    if not bm25_scores:
        return ClinicalConfidence(
            score=0.0,
            should_abstain=True,
            reason="No matching clinical records found",
        )

    n = len(bm25_scores)
    avg_score = sum(bm25_scores) / n
    avg_overlap = sum(entity_overlaps) / n
    confidence = avg_score * score_weight + avg_overlap * overlap_weight

    should_abstain = confidence < abstention_threshold
    if should_abstain:
        reason = (
            f"Confidence {confidence:.2f} below clinical threshold "
            f"{abstention_threshold}. Insufficient evidence for a safe answer."
        )
    else:
        reason = f"Confidence {confidence:.2f} — sufficient clinical evidence."

    return ClinicalConfidence(
        score=confidence,
        should_abstain=should_abstain,
        reason=reason,
    )


def clinical_importance(
    access_count: int,
    days_since_access: float,
    connection_degree: int,
    is_acute: bool = False,
    decay_rate: float = -0.1,
    access_weight: float = 0.3,
    recency_weight: float = 0.5,
    connection_weight: float = 0.2,
) -> float:
    """
    Importance scoring from importance.mind kernel.

    Scores how important a clinical memory block is for current context.
    Acute conditions get a boost. Returns a multiplier in [0.8, 1.5].
    """
    freq = math.log(access_count + 1)
    recency = math.exp(decay_rate * days_since_access)
    conn = math.log(connection_degree + 1)

    raw = access_weight * freq + recency_weight * recency + connection_weight * conn
    max_est = access_weight * 3.0 + recency_weight * 1.0 + connection_weight * 3.0
    normalized = raw / (max_est + 1e-6)
    score = 0.8 + normalized * 0.7

    if is_acute:
        score = min(score * 1.3, 1.5)

    return round(score, 4)


def medication_severity_score(
    interaction_type: str,
    severity: str | None = None,
) -> float:
    """
    Score medication interaction severity for prioritization.

    Returns 0.0-1.0 where higher = more severe.
    """
    base_scores = {
        "contraindicated": 1.0,
        "serious": 0.8,
        "moderate": 0.5,
        "minor": 0.2,
        "unknown": 0.4,
    }
    severity_boost = {
        "high": 0.15,
        "moderate": 0.05,
        "low": 0.0,
    }
    base = base_scores.get(interaction_type.lower(), 0.4)
    boost = severity_boost.get((severity or "").lower(), 0.0)
    return min(base + boost, 1.0)


def is_negation_query(query: str) -> bool:
    """
    Negation detection from adversarial.mind kernel.

    Critical for clinical queries like "NOT allergic to penicillin"
    vs "allergic to penicillin".
    """
    negation_markers = [
        "not ", "no ", "never ", "without ", "absence of ",
        "denies ", "negative for ", "ruled out", "unlikely ",
        "non-", "un-", "n't ",
    ]
    lower = query.lower()
    return any(marker in lower for marker in negation_markers)


# Known drug interaction pairs (subset for demo — production would use RxNorm API)
_KNOWN_INTERACTIONS: list[tuple[str, str, str, str]] = [
    ("warfarin", "aspirin", "serious", "Increased bleeding risk"),
    ("warfarin", "ibuprofen", "serious", "Increased bleeding risk"),
    ("warfarin", "naproxen", "serious", "Increased bleeding risk"),
    ("warfarin", "nsaid", "serious", "Increased bleeding risk"),
    ("metformin", "contrast dye", "contraindicated", "Risk of lactic acidosis"),
    ("lisinopril", "potassium", "moderate", "Risk of hyperkalemia"),
    ("lisinopril", "spironolactone", "moderate", "Risk of hyperkalemia"),
    ("metoprolol", "verapamil", "serious", "Risk of severe bradycardia"),
    ("simvastatin", "amiodarone", "serious", "Increased risk of rhabdomyolysis"),
    ("fluoxetine", "tramadol", "serious", "Serotonin syndrome risk"),
    ("ciprofloxacin", "tizanidine", "contraindicated", "Dangerous hypotension"),
    ("methotrexate", "trimethoprim", "serious", "Increased methotrexate toxicity"),
]

# Known allergy cross-reactions
_ALLERGY_CROSS_REACTIONS: list[tuple[str, list[str], str]] = [
    ("penicillin", ["amoxicillin", "ampicillin", "piperacillin"], "Beta-lactam cross-reactivity"),
    ("sulfa", ["sulfamethoxazole", "sulfasalazine", "celecoxib"], "Sulfonamide cross-reactivity"),
    ("codeine", ["morphine", "hydrocodone", "oxycodone"], "Opioid cross-sensitivity"),
    ("nsaid", ["ibuprofen", "naproxen", "aspirin", "ketorolac"], "NSAID class reaction"),
]


@dataclass(frozen=True)
class DrugInteraction:
    drug_a: str
    drug_b: str
    severity: str
    description: str
    score: float
    # Layer 4.5 (BitNet b1.58 ternary classifier) reproducibility fields.
    # Populated by `check_drug_interactions` for every reported pair —
    # `bitnet_severity` is the classifier's independent severity verdict
    # (`none|minor|moderate|major|contraindicated`), `bitnet_repro_hash`
    # is the SHA-256 over the canonical (feature_hash, logits_q16,
    # severity, weights_id) tuple any auditor can re-verify in <1 ms per
    # pair with engine/bitnet_classifier.py and engine/bitnet_weights.json.
    # `bitnet_weights_id` pins the exact weights bundle the verdict
    # came from. `None` only when the classifier raised — never silently.
    bitnet_severity: str | None = None
    bitnet_repro_hash: str | None = None
    bitnet_weights_id: str | None = None


def check_drug_interactions(
    medications: list[str], use_llm_fallback: bool = True
) -> list[DrugInteraction]:
    """
    Check a medication list for interactions.

    Five-tier detection pipeline (v4.2+):
      1. Deterministic table (12 known pairs) — fast, reliable, auditable
      2. OpenEvidence API — clinically authoritative, purpose-built for medicine
      3. NIH Drug Interaction API (RxNorm) — federal gold standard
      4. Gemini LLM — general-purpose fallback for remaining uncovered pairs
      4.5. BitNet b1.58 ternary classifier — Q16.16 fixed-point forward pass
           with bit-identical output across ARM, x86_64, CUDA, NPU. Runs
           AFTER the upstream detection layers so each reported pair carries
           its own `bitnet_repro_hash` (SHA-256 over the canonical encoding
           of feature_hash, logits_q16, severity, weights_id). The hash is
           the FDA SaMD reproducibility primitive — any auditor with
           engine/bitnet_classifier.py + engine/bitnet_weights.json can
           re-verify any past clinical decision in <1 ms per pair, no
           proprietary toolchain required.

    Each layer only checks pairs not already found by previous layers,
    except the BitNet layer which runs on every reported pair as a
    deterministic verification stamp (Layer 4.5).
    """
    meds_lower = [m.lower().strip() for m in medications]
    interactions = []

    # Layer 1: Deterministic table (microseconds)
    covered_pairs: set[tuple[str, str]] = set()
    for drug_a, drug_b, severity, description in _KNOWN_INTERACTIONS:
        a_match = any(drug_a in m for m in meds_lower)
        b_match = any(drug_b in m for m in meds_lower)
        if a_match and b_match:
            interactions.append(
                DrugInteraction(
                    drug_a=drug_a,
                    drug_b=drug_b,
                    severity=severity,
                    description=description,
                    score=medication_severity_score(severity),
                )
            )
            covered_pairs.add((drug_a, drug_b))

    # Layer 2: OpenEvidence API (clinically authoritative, purpose-built for medicine)
    if use_llm_fallback and len(meds_lower) >= 2:
        oe_interactions = _openevidence_check_interactions(medications, covered_pairs)
        interactions.extend(oe_interactions)
        for i in oe_interactions:
            covered_pairs.add((i.drug_a, i.drug_b))

    # Layer 3: RxNorm API — proper drug normalization + NIH interaction DB
    if use_llm_fallback and len(meds_lower) >= 2:
        rxnorm_interactions = _rxnorm_check_interactions(medications, covered_pairs)
        interactions.extend(rxnorm_interactions)
        for i in rxnorm_interactions:
            covered_pairs.add((i.drug_a, i.drug_b))

    # Layer 4: Gemini fallback (general-purpose LLM for remaining uncovered pairs)
    if use_llm_fallback and len(meds_lower) >= 2:
        llm_interactions = _llm_check_interactions(medications, covered_pairs)
        interactions.extend(llm_interactions)

    # Layer 4.5: BitNet b1.58 deterministic-classifier verification stamp.
    # Runs on every reported pair (no API call, no network, no float math) —
    # produces a Q16.16 logit vector + repro_hash that the audit chain
    # records alongside the upstream-layer evidence. Failure to load the
    # classifier is logged but does not prevent reporting (the upstream
    # layers are still the source of truth for the severity decision).
    interactions = _attach_bitnet_repro_hashes(interactions)

    final = sorted(interactions, key=lambda i: i.score, reverse=True)

    # PHI-safe aggregate exit log — counts per severity only, never
    # drug names. The per-layer loggers above fire on hits/errors but
    # this aggregate captures the full invocation footprint so
    # operators can compute alert rates and per-tier hit attribution
    # across runs without combining multiple log streams.
    sev_counts: dict[str, int] = {}
    for i in final:
        sev_counts[i.severity] = sev_counts.get(i.severity, 0) + 1
    # DEBUG (not INFO) to avoid polluting scripts/run_clinical_regression_eval.py
    # stdout-bound logging.basicConfig and break the negative-control JSON
    # parser. Default-level logs stay quiet; ops can re-enable with -v.
    logger.debug(
        "drug_interactions_check_complete",
        extra={
            "med_count": len(meds_lower),
            "interaction_count": len(final),
            "severity_counts": sev_counts,
            "use_llm_fallback": use_llm_fallback,
        },
    )

    return final


_SAFETY_DOWNGRADE_SEVERITY: set[str] = {"contraindicated", "serious", "major"}


def _attach_bitnet_repro_hashes(
    interactions: list[DrugInteraction],
) -> list[DrugInteraction]:
    """Layer 4.5 — stamp every interaction with a BitNet b1.58 repro_hash.

    The classifier is order-canonicalised (lex sort) so {warfarin, ibuprofen}
    and {ibuprofen, warfarin} produce the same hash. Failure to load the
    classifier (missing weights bundle, etc.) is recorded as `None` and
    never silently dropped — the audit chain reflects the actual state.

    **Safety-downgrade alerts:** if the upstream pipeline has reported a
    pair as `contraindicated`, `serious`, or `major` AND the BitNet
    classifier disagrees by predicting `none` or `minor`, that is a
    release-blocking signal — the docstring's "Disagreement is a
    release-blocking event" promise must hold at runtime, not just in
    tests. We log a structured WARNING with the canonical preimage so
    the alert is grep-able and the audit chain still records both the
    upstream verdict and the contradictory BitNet stamp. The upstream
    severity is never silently downgraded; the safer (more severe)
    verdict wins.
    """
    try:
        from engine.bitnet_classifier import classifier_layer, WeightsTamperError
    except Exception as exc:
        logger.warning(
            "bitnet_classifier_import_failed",
            extra={"error_type": type(exc).__name__, "layer": "4.5"},
        )
        return interactions

    stamped: list[DrugInteraction] = []
    failed_stamps = 0
    disagreement_count = 0
    for it in interactions:
        try:
            result = classifier_layer(it.drug_a, it.drug_b)
        except WeightsTamperError as exc:
            # Tamper is release-blocking. Re-raise so the caller's
            # MedicationSafetyReview flow's invariants surface it.
            # PHI-safe: error_type only — exception messages can carry
            # weights-bundle path or other internal state.
            logger.error(
                "bitnet_weights_tamper",
                extra={
                    "error_type": type(exc).__name__,
                },
            )
            raise
        except Exception as exc:
            # Other classifier failures (missing weights file, malformed
            # JSON) are recorded as gaps; the upstream interaction stays.
            # PHI-safe: error_type only + drug-name lengths (drug names
            # are PHI-adjacent — clinical input may carry adjacent
            # narrative; same iter-234 / iter-239 discipline).
            logger.warning(
                "bitnet_classifier_failed",
                extra={
                    "error_type": type(exc).__name__,
                    "drug_a_length": len(it.drug_a),
                    "drug_b_length": len(it.drug_b),
                },
            )
            stamped.append(it)
            failed_stamps += 1
            continue

        # Safety-downgrade alert: upstream said this pair is dangerous,
        # BitNet says it's not. The audit chain records both, but
        # operators must see this as a WARNING in the live log so they
        # can investigate the disagreement before any clinical action.
        upstream_dangerous = it.severity.lower() in _SAFETY_DOWNGRADE_SEVERITY
        bitnet_safe = result.severity_name in ("none", "minor")
        if upstream_dangerous and bitnet_safe:
            disagreement_count += 1
            logger.warning(
                "BITNET_SAFETY_DOWNGRADE_DISAGREEMENT pair=%s+%s "
                "upstream=%s bitnet=%s feature_hash=%s repro_hash=%s "
                "weights_id=%s — upstream verdict preserved; investigate "
                "before any clinical action",
                it.drug_a, it.drug_b, it.severity, result.severity_name,
                result.feature_hash, result.repro_hash, result.weights_id,
            )

        stamped.append(
            DrugInteraction(
                drug_a=it.drug_a,
                drug_b=it.drug_b,
                severity=it.severity,           # upstream verdict preserved
                description=it.description,
                score=it.score,
                bitnet_severity=result.severity_name,
                bitnet_repro_hash=result.repro_hash,
                bitnet_weights_id=result.weights_id,
            )
        )
    # DEBUG — Layer 4.5 stamping completion signal so operators can track
    # the success rate of repro-hash stamping and the live disagreement
    # count without needing to grep WARNING records. Demoted to DEBUG to
    # avoid polluting scripts/run_clinical_regression_eval.py stdout-bound
    # logging.basicConfig (negative-control parser would break on INFO).
    logger.debug(
        "bitnet_stamping_complete",
        extra={
            "interaction_count": len(interactions),
            "stamped_count": len(stamped) - failed_stamps,
            "failed_stamps": failed_stamps,
            "disagreement_count": disagreement_count,
        },
    )
    return stamped


def _openevidence_check_interactions(
    medications: list[str],
    already_found: set[tuple[str, str]],
) -> list[DrugInteraction]:
    """
    Layer 2: OpenEvidence API — clinically authoritative drug interaction check.

    OpenEvidence is purpose-built for medicine (Mayo Clinic, Elsevier ClinicalKey AI,
    90%+ USMLE accuracy). Unlike a general LLM, it returns evidence-grounded answers
    with citations to peer-reviewed literature.

    API: POST https://api.openevidence.com/analysis
    Auth: Token-based (OPENEVIDENCE_API_KEY)

    Fallback: when OPENEVIDENCE_API_KEY is absent or the live call fails, the
    function falls back to engine.openevidence_cache so the demo dashboard and
    tests can show realistic responses. Cached entries are identified by the
    "[CACHED <date>]" prefix in the description field and a structured INFO
    log line.  The moment a live key is set, the live path takes precedence and
    the cache is never consulted.
    """
    import json
    import os

    api_key = os.environ.get("OPENEVIDENCE_API_KEY")
    if api_key:
        med_names = [m.split()[0] if " " in m else m for m in medications]
        query = (
            f"Are there clinically significant drug-drug interactions between any of "
            f"these medications: {', '.join(med_names)}? "
            f"For each interaction found, state the two drugs, severity "
            f"(serious or contraindicated), and a one-sentence clinical description."
        )

        try:
            import httpx

            resp = httpx.post(
                "https://api.openevidence.com/analysis",
                headers={
                    "Authorization": f"Token {api_key}",
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                },
                json={"text": query, "model": "oe-v2"},
                timeout=5,
            )
            if resp.status_code != 200:
                logger.warning("OpenEvidence API returned %d", resp.status_code)
            else:
                data = resp.json()
                # OpenEvidence returns a narrative analysis — parse for drug pairs
                analysis_text = data.get("text", "") or data.get("analysis", "") or str(data)
                if not analysis_text:  # pragma: no cover — str(data) never empty
                    return []

                results = _parse_interaction_narrative(
                    analysis_text, med_names, already_found, source="OpenEvidence"
                )
                if results:
                    logger.info(
                        "OpenEvidence detected %d additional interactions", len(results)
                    )
                return results

        except Exception as e:
            logger.warning(
                "openevidence_api_call_failed",
                extra={
                    "error_type": type(e).__name__,
                    "med_count": len(medications),
                },
            )
            # Fall through to cache fallback below.

    # No key or live call failed — consult the cached fixture set.
    return _openevidence_cache_fallback(medications, already_found)


def _openevidence_cache_fallback(
    medications: list[str],
    already_found: set[tuple[str, str]],
) -> list[DrugInteraction]:
    """Return cached OpenEvidence responses for the given medication list.

    Produces DrugInteraction objects in the same format as the live path so
    downstream layers (BitNet stamp, audit chain) require no changes.  Each
    entry's description is prefixed "[CACHED <date>]" for audit transparency.
    """
    try:
        from engine.openevidence_cache import lookup_cached, canonical_pair_key
    except ImportError as exc:
        logger.warning(
            "openevidence_cache_import_failed",
            extra={"error_type": type(exc).__name__},
        )
        return []

    meds_lower = [m.strip().lower() for m in medications]
    results: list[DrugInteraction] = []

    for i, med_a in enumerate(meds_lower):
        for med_b in meds_lower[i + 1:]:
            pair = canonical_pair_key(med_a, med_b)
            if pair in already_found or tuple(reversed(pair)) in already_found:
                continue
            cached = lookup_cached(med_a, med_b)
            if cached is None:
                continue
            date_tag = cached.retrieved_at or "unknown"
            description = f"[CACHED {date_tag}] {cached.clinical_summary}"
            results.append(
                DrugInteraction(
                    drug_a=cached.drug_pair_canonical[0],
                    drug_b=cached.drug_pair_canonical[1],
                    severity=cached.severity,
                    description=description,
                    score=medication_severity_score(cached.severity),
                )
            )
            logger.info(
                "openevidence_cache: returning cached entry pair=%s+%s severity=%s source=%s",
                cached.drug_pair_canonical[0],
                cached.drug_pair_canonical[1],
                cached.severity,
                cached.source,
            )

    # DEBUG — cache-fallback completion signal. Tracks how many of the
    # caller's pairs hit the cache when the live OpenEvidence API was
    # unavailable. Lets operators correlate cache-hit rate with API
    # outage windows. PHI-safe: counts only.
    n_pairs_checked = len(meds_lower) * (len(meds_lower) - 1) // 2
    logger.debug(
        "openevidence_cache_fallback_complete",
        extra={
            "med_count": len(meds_lower),
            "pairs_checked": n_pairs_checked,
            "cache_hits": len(results),
            "skipped_already_found": len(already_found),
        },
    )
    return results


def _parse_interaction_narrative(
    text: str,
    med_names: list[str],
    already_found: set[tuple[str, str]],
    source: str = "OpenEvidence",
) -> list[DrugInteraction]:
    """Parse a narrative text for drug interaction mentions."""
    text_lower = text.lower()
    results = []
    meds_lower = [m.lower() for m in med_names]

    # Check all medication pairs against the narrative
    for i, med_a in enumerate(meds_lower):
        for med_b in meds_lower[i + 1 :]:
            if (med_a, med_b) in already_found or (med_b, med_a) in already_found:
                continue
            # Both meds mentioned in the analysis text = potential interaction
            if med_a in text_lower and med_b in text_lower:
                # Check for severity indicators
                severity = "moderate"
                if any(
                    w in text_lower
                    for w in [
                        "contraindicated",
                        "avoid",
                        "do not combine",
                        "prohibited",
                    ]
                ):
                    severity = "contraindicated"
                elif any(
                    w in text_lower
                    for w in [
                        "serious",
                        "significant",
                        "major",
                        "dangerous",
                        "bleeding risk",
                        "serotonin syndrome",
                        "qt prolongation",
                    ]
                ):
                    severity = "serious"

                if severity in ("serious", "contraindicated"):
                    # Extract a description snippet around the drug mentions
                    desc = f"{source}-detected interaction between {med_a} and {med_b}"
                    results.append(
                        DrugInteraction(
                            drug_a=med_a,
                            drug_b=med_b,
                            severity=severity,
                            description=desc,
                            score=medication_severity_score(severity),
                        )
                    )
    # DEBUG — narrative-parsing completion signal. Lets operators track
    # how often narrative parsing surfaces a serious / contraindicated
    # interaction the structured API didn't return. PHI-safe: scalars
    # only; med_names + the parsed narrative text are NOT logged
    # (med_names live in synthetic Synthea cohort but the narrative is
    # caller-supplied and could carry downstream PHI).
    logger.debug(
        "narrative_parse_complete",
        extra={
            "source": source,
            "text_length": len(text),
            "med_count": len(med_names),
            "result_count": len(results),
            "serious_count": sum(1 for r in results if r.severity == "serious"),
            "contra_count": sum(
                1 for r in results if r.severity == "contraindicated"
            ),
        },
    )
    return results


def _rxnorm_check_interactions(
    medications: list[str],
    already_found: set[tuple[str, str]],
) -> list[DrugInteraction]:
    """Layer 3: RxNorm API — proper drug normalization + NIH interaction DB.

    Uses the rxnorm_client module for:
    1. Canonical drug name resolution (brand → generic → ingredient)
    2. RxCUI-based interaction lookup (NIH Drug Interaction API)
    Same data source used by Epic, Cerner, and all certified EHR systems.
    """
    import logging

    logger = logging.getLogger(__name__)

    try:
        from engine.rxnorm_client import normalize_medication_list, get_interactions_for_list
    except ImportError as exc:
        logger.error(
            "rxnorm_client_import_failed",
            extra={
                "error_type": type(exc).__name__,
                "med_count": len(medications),
            },
        )
        return []

    resolved = normalize_medication_list(medications)
    rxcuis = [rc.rxcui for rc in resolved.values() if rc is not None]

    if len(rxcuis) < 2:
        # Debug-level: the negative-control eval triggers this on every
        # 0-or-1-drug pair (clean negatives often have only 1 covered
        # drug) and stdout-captured JSON output would be polluted at INFO.
        logger.debug(
            "rxnorm_pre_flight_insufficient_rxcuis",
            extra={
                "med_count": len(medications),
                "resolved_count": len(rxcuis),
            },
        )
        return []

    rxnorm_interactions = get_interactions_for_list(rxcuis)
    results = []

    for ri in rxnorm_interactions:
        pair = (ri.drug_a, ri.drug_b)
        if pair in already_found or tuple(reversed(pair)) in already_found:
            continue

        if ri.severity in ("serious", "contraindicated"):
            results.append(
                DrugInteraction(
                    drug_a=ri.drug_a,
                    drug_b=ri.drug_b,
                    severity=ri.severity,
                    description=f"RxNorm/{ri.source}: {ri.description[:200]}",
                    score=medication_severity_score(ri.severity),
                )
            )
            already_found.add(pair)

    if results:
        logger.info("RxNorm API detected %d additional interactions", len(results))
    return results


def _call_openai_json(prompt: str, api_key: str) -> str | None:
    """Call OpenAI API for structured JSON drug interaction response."""
    import logging

    logger = logging.getLogger(__name__)
    try:
        import httpx

        resp = httpx.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": "gpt-5.5",
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are a clinical pharmacist with expertise in drug-drug "
                            "interactions. You ONLY return valid JSON arrays."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.1,
                "max_tokens": 512,
            },
            timeout=5,
        )
        if resp.status_code != 200:
            logger.info("OpenAI returned %d", resp.status_code)
            return None
        data = resp.json()
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        logger.info(
            "openai_call_failed",
            extra={"error_type": type(e).__name__},
        )
        return None


def _call_google_json(prompt: str, api_key: str, model_id: str) -> str | None:
    """Call Google GenAI API for structured JSON drug interaction response."""
    import logging

    logger = logging.getLogger(__name__)
    try:
        import httpx

        resp = httpx.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{model_id}:generateContent",
            headers={"x-goog-api-key": api_key},
            json={
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": 0.1, "maxOutputTokens": 512},
            },
            timeout=5,
        )
        if resp.status_code != 200:
            logger.info("%s returned %d", model_id, resp.status_code)
            return None
        data = resp.json()
        candidates = data.get("candidates", [])
        if candidates:
            return candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "")
    except Exception as e:
        logger.info(
            "google_genai_call_failed",
            extra={
                "model_id": model_id,
                "error_type": type(e).__name__,
            },
        )
    return None


def _llm_check_interactions(
    medications: list[str],
    already_found: set[tuple[str, str]],
) -> list[DrugInteraction]:
    """
    Layer 4: Medical LLM cascade for drug interactions.

    Tries models in order of clinical strength, using whichever API keys
    are available. Each model uses the same structured prompt.

    Cascade: OpenAI GPT-5.5 → Gemini 3.1 Pro
    """
    import json
    import logging
    import os

    logger = logging.getLogger(__name__)

    openai_key = os.environ.get("OPENAI_API_KEY")
    google_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")

    if not openai_key and not google_key:
        return []

    med_names = [m.split()[0] if " " in m else m for m in medications]

    prompt = f"""You are a clinical pharmacist. Given these medications: {', '.join(med_names)}

Check for drug-drug interactions. ONLY report clinically significant interactions
(serious or contraindicated). Do NOT report minor or theoretical interactions.

Respond with ONLY a JSON array. Each element must have:
- "drug_a": first drug name (lowercase)
- "drug_b": second drug name (lowercase)
- "severity": "serious" or "contraindicated"
- "description": one-sentence clinical description

If NO significant interactions exist, respond with: []

JSON array:"""

    # Build model cascade based on available API keys
    attempts: list[tuple[str, callable]] = []
    if openai_key:
        attempts.append(("OpenAI-GPT-5.5", lambda: _call_openai_json(prompt, openai_key)))
    if google_key:
        attempts.append(("Gemini-3.1-Pro", lambda: _call_google_json(prompt, google_key, "gemini-3.1-pro-preview")))

    for model_label, call_fn in attempts:
        text = call_fn()
        if not text:
            continue

        # Extract JSON (handle markdown code blocks)
        text = text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
            text = text.rsplit("```", 1)[0]
        text = text.strip()

        if not text or text == "[]":
            return []

        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            logger.info("%s returned invalid JSON, trying next", model_label)
            continue
        if not isinstance(parsed, list):
            continue

        results = []
        for item in parsed:
            a = item.get("drug_a", "").lower()
            b = item.get("drug_b", "").lower()
            if (a, b) in already_found or (b, a) in already_found:
                continue
            sev = item.get("severity", "moderate")
            if sev not in ("serious", "contraindicated"):
                continue
            results.append(
                DrugInteraction(
                    drug_a=a,
                    drug_b=b,
                    severity=sev,
                    description=f"{model_label}: {item.get('description', 'LLM-detected interaction')}",
                    score=medication_severity_score(sev),
                )
            )
        if results:
            logger.info("%s detected %d additional drug interactions", model_label, len(results))
        return results

    return []


@dataclass(frozen=True)
class AllergyConflict:
    allergen: str
    medication: str
    cross_reaction_group: str
    description: str


def check_allergy_conflicts(
    allergies: list[str], medications: list[str]
) -> list[AllergyConflict]:
    """Cross-reference patient allergies against prescribed medications.

    Two-layer detection:
    1. Local cross-reaction table (fast, deterministic)
    2. SNOMED CT drug class hierarchy (broader coverage)
    """
    allergies_lower = [a.lower().strip() for a in allergies]
    meds_lower = [m.lower().strip() for m in medications]
    conflicts = []
    found_pairs: set[tuple[str, str]] = set()

    # Layer 1: Deterministic table
    for allergen, cross_drugs, description in _ALLERGY_CROSS_REACTIONS:
        allergen_match = any(allergen in a for a in allergies_lower)
        if not allergen_match:
            continue
        for drug in cross_drugs:
            drug_match = any(drug in m for m in meds_lower)
            if drug_match:
                conflicts.append(
                    AllergyConflict(
                        allergen=allergen,
                        medication=drug,
                        cross_reaction_group=allergen,
                        description=description,
                    )
                )
                found_pairs.add((allergen, drug))

    # Layer 2: SNOMED CT drug class hierarchy (broader coverage)
    try:
        from engine.snomed_client import is_allergy_cross_reactive
        for allergy in allergies_lower:
            for med in meds_lower:
                if (allergy, med) in found_pairs:
                    continue
                if is_allergy_cross_reactive(allergy, med):
                    conflicts.append(
                        AllergyConflict(
                            allergen=allergy,
                            medication=med,
                            cross_reaction_group="SNOMED-hierarchy",
                            description=f"Cross-reactivity detected via drug class hierarchy",
                        )
                    )
                    found_pairs.add((allergy, med))
    except ImportError:
        pass

    # PHI-safe DEBUG — counts only. Allergen + medication names are
    # PHI by HIPAA Safe Harbor (allergy-to-X is identifying when paired
    # with other quasi-identifiers). The per-source breakdown lets
    # operators measure how often the SNOMED CT class hierarchy
    # contributes detections beyond the deterministic table.
    table_count = sum(1 for c in conflicts if c.cross_reaction_group != "SNOMED-hierarchy")
    snomed_count = sum(1 for c in conflicts if c.cross_reaction_group == "SNOMED-hierarchy")
    logger.debug(
        "allergy_conflicts_check_complete",
        extra={
            "allergy_count": len(allergies_lower),
            "med_count": len(meds_lower),
            "conflict_count": len(conflicts),
            "table_layer_hits": table_count,
            "snomed_layer_hits": snomed_count,
        },
    )

    return conflicts


# ── Lab-based medication contraindications ────────────────────────────────────

# Maps (lab_name_pattern, medication_pattern) -> threshold logic
# Each entry: (lab_keywords, med_keywords, threshold, direction, severity, description, recommendation)
_LAB_MED_CONTRAINDICATIONS: list[tuple[list[str], list[str], float, str, str, str, str]] = [
    (
        ["egfr", "gfr", "glomerular filtration"],
        ["metformin"],
        30.0, "below", "critical",
        "Metformin is contraindicated when eGFR falls below 30 mL/min due to risk of lactic acidosis",
        "STOP metformin immediately. Consider insulin or DPP-4 inhibitor (dose-adjusted for renal function).",
    ),
    (
        ["egfr", "gfr", "glomerular filtration"],
        ["metformin"],
        45.0, "below", "high",
        "Metformin dose reduction recommended when eGFR is 30-45 mL/min",
        "Reduce metformin to maximum 1000mg/day. Monitor renal function every 3 months.",
    ),
    (
        ["inr"],
        ["warfarin"],
        3.5, "above", "high",
        "INR above therapeutic range (2.0-3.0) indicates excessive anticoagulation and bleeding risk",
        "Hold warfarin dose. Check for new interacting medications (NSAIDs, antibiotics). Recheck INR in 2-3 days.",
    ),
    (
        ["potassium", "k+"],
        ["lisinopril", "losartan", "spironolactone"],
        5.5, "above", "critical",
        "Hyperkalemia risk with ACE inhibitors/ARBs/aldosterone antagonists when potassium exceeds 5.5 mEq/L",
        "Hold potassium-sparing medications. Obtain stat ECG. Consider calcium gluconate if K+ > 6.0.",
    ),
    (
        ["hba1c", "hemoglobin a1c", "a1c"],
        ["metformin"],
        9.0, "above", "moderate",
        "HbA1c above 9% suggests inadequate glycemic control on current metformin regimen",
        "Consider adding second-line agent (GLP-1 agonist or SGLT2 inhibitor). Reinforce lifestyle modifications.",
    ),
]


@dataclass(frozen=True)
class LabMedContraindication:
    lab_name: str
    lab_value: float
    lab_unit: str
    medication: str
    threshold: float
    direction: str  # "above" or "below"
    severity: str
    description: str
    recommendation: str


def check_lab_medication_contraindications(
    observations: list[dict], medications: list[str]
) -> list[LabMedContraindication]:
    """
    Cross-reference lab results against medications for contraindications.

    Detects clinically dangerous combinations like declining GFR + metformin,
    elevated INR + warfarin, or hyperkalemia + ACE inhibitors.

    Args:
        observations: List of observation dicts with keys: observation_name, value, unit
        medications: List of active medication names
    """
    meds_lower = [m.lower().strip() for m in medications]
    contraindications = []

    for lab_keywords, med_keywords, threshold, direction, severity, desc, rec in _LAB_MED_CONTRAINDICATIONS:
        # Check if any medication matches
        med_match = None
        for med_kw in med_keywords:
            for m in meds_lower:
                if med_kw in m:
                    med_match = m
                    break
            if med_match:
                break
        if not med_match:
            continue

        # Find matching lab observations
        for obs in observations:
            obs_name = (obs.get("observation_name") or obs.get("name") or "").lower()
            if not any(kw in obs_name for kw in lab_keywords):
                continue
            try:
                val = float(obs.get("value", 0))
            except (ValueError, TypeError):
                continue
            unit = obs.get("unit") or obs.get("lab_unit") or ""

            triggered = (
                (direction == "below" and val < threshold) or
                (direction == "above" and val > threshold)
            )
            if triggered:
                contraindications.append(
                    LabMedContraindication(
                        lab_name=obs.get("observation_name") or obs.get("name") or "Unknown",
                        lab_value=val,
                        lab_unit=unit,
                        medication=med_match,
                        threshold=threshold,
                        direction=direction,
                        severity=severity,
                        description=desc,
                        recommendation=rec,
                    )
                )

    # Deduplicate: keep only the highest severity per (lab_name, medication) pair
    seen = {}
    severity_rank = {"critical": 4, "high": 3, "moderate": 2, "low": 1}
    for c in contraindications:
        key = (c.lab_name, c.medication)
        existing = seen.get(key)
        if not existing or severity_rank.get(c.severity, 0) > severity_rank.get(existing.severity, 0):
            seen[key] = c

    final = sorted(seen.values(), key=lambda c: severity_rank.get(c.severity, 0), reverse=True)

    # INFO — lab×medication contraindications are SaMD-relevant safety
    # signals. PHI-safe: counts + severity distribution only, NEVER
    # lab values, lab names, or medication names (all PHI-adjacent).
    sev_counts: dict[str, int] = {}
    for c in final:
        sev_counts[c.severity] = sev_counts.get(c.severity, 0) + 1
    # DEBUG (not INFO) — same rationale as drug_interactions_check_complete
    # above; the regression-eval script pipes INFO to stdout and the
    # negative-control JSON parser breaks if these log lines fire there.
    logger.debug(
        "lab_medication_contraindications_check_complete",
        extra={
            "observation_count": len(observations),
            "med_count": len(meds_lower),
            "contraindication_count": len(final),
            "severity_counts": sev_counts,
        },
    )

    return final


# ── Lab trend analysis ────────────────────────────────────────────────────────

@dataclass(frozen=True)
class LabTrend:
    lab_name: str
    values: list[float]
    dates: list[str]
    direction: str  # "declining", "rising", "stable"
    rate_of_change: float  # per-measurement average change
    severity: str
    description: str
    recommendation: str


def detect_lab_trends(observations: list[dict]) -> list[LabTrend]:
    """
    Detect clinically significant trends in sequential lab values.

    Groups observations by lab name, orders by date, and identifies
    declining or rising patterns that warrant clinical attention.
    """
    # Group observations by name
    by_name: dict[str, list[tuple[str, float]]] = {}
    for obs in observations:
        name = (obs.get("observation_name") or obs.get("name") or "").strip()
        date = obs.get("effective_date") or obs.get("date") or ""
        try:
            val = float(obs.get("value", 0))
        except (ValueError, TypeError):
            continue
        if name and date:
            by_name.setdefault(name, []).append((date, val))

    trends = []
    for name, points in by_name.items():
        if len(points) < 2:
            continue
        # Sort by date
        points.sort(key=lambda x: x[0])
        values = [p[1] for p in points]
        dates = [p[0] for p in points]

        # Calculate trend
        changes = [values[i+1] - values[i] for i in range(len(values)-1)]
        avg_change = sum(changes) / len(changes)
        total_change = values[-1] - values[0]

        name_lower = name.lower()

        # GFR declining trend
        if any(kw in name_lower for kw in ["egfr", "gfr", "glomerular"]):
            if total_change < -5:  # Decline of 5+ mL/min
                severity = "critical" if values[-1] < 30 else "high" if values[-1] < 45 else "moderate"
                trends.append(LabTrend(
                    lab_name=name,
                    values=values,
                    dates=dates,
                    direction="declining",
                    rate_of_change=round(avg_change, 2),
                    severity=severity,
                    description=(
                        f"eGFR declining: {values[0]:.0f} → {values[-1]:.0f} mL/min/1.73m² "
                        f"(Δ {total_change:+.0f} over {len(values)} measurements). "
                        f"{'Approaching contraindication threshold for nephrotoxic medications.' if values[-1] < 45 else 'Monitor closely.'}"
                    ),
                    recommendation=(
                        "Review all renally-cleared medications for dose adjustment. "
                        "Nephrology referral if not already involved. "
                        "Recheck eGFR in 4-6 weeks."
                    ),
                ))

        # INR trending high
        elif any(kw in name_lower for kw in ["inr"]):
            if total_change > 0.5 and values[-1] > 3.0:
                trends.append(LabTrend(
                    lab_name=name,
                    values=values,
                    dates=dates,
                    direction="rising",
                    rate_of_change=round(avg_change, 2),
                    severity="high",
                    description=(
                        f"INR rising above therapeutic range: {values[0]:.1f} → {values[-1]:.1f}. "
                        "Check for new interacting medications or dietary changes."
                    ),
                    recommendation="Hold warfarin. Investigate cause. Recheck INR in 2-3 days.",
                ))

    # DEBUG — trend detection is best-effort safety signaling. PHI-safe:
    # observation count + grouped lab-name count + per-direction trend
    # counts only, never values/dates/lab-names (lab names + values are
    # PHI in identified EHR data).
    direction_counts: dict[str, int] = {}
    for t in trends:
        direction_counts[t.direction] = direction_counts.get(t.direction, 0) + 1
    logger.debug(
        "lab_trends_detect_complete",
        extra={
            "observation_count": len(observations),
            "distinct_lab_count": len(by_name),
            "trend_count": len(trends),
            "direction_counts": direction_counts,
        },
    )

    return trends


# ── Provider disagreement detection ───────────────────────────────────────────

@dataclass(frozen=True)
class ProviderDisagreement:
    topic: str
    provider_a: str
    provider_a_position: str
    provider_b: str
    provider_b_position: str
    severity: str
    description: str
    recommendation: str


def detect_provider_disagreements(blocks: list[dict]) -> list[ProviderDisagreement]:
    """
    Detect conflicting clinical recommendations from different providers.

    Compares notes and targets across observations/blocks from different
    providers to find disagreements in treatment goals.
    """
    disagreements = []

    # Look for BP target conflicts in observation notes
    bp_targets: list[dict] = []
    for block in blocks:
        content = (block.get("content") or "").lower()
        source = block.get("source") or block.get("metadata", {}).get("performer") or ""
        title = (block.get("title") or "").lower()
        notes = block.get("metadata", {}).get("notes") or ""

        # Check for BP target mentions
        if "blood pressure" in title or "bp" in title:
            import re
            # Match patterns like "<130/80", "target: 130/80", "<140/90"
            target_match = re.search(r'target[:\s]*<?(\d{2,3})/(\d{2,3})', content + " " + notes.lower())
            if target_match:
                systolic = int(target_match.group(1))
                diastolic = int(target_match.group(2))
                bp_targets.append({
                    "systolic": systolic,
                    "diastolic": diastolic,
                    "source": source,
                    "content": content,
                })

    # Compare BP targets from different providers
    for i in range(len(bp_targets)):
        for j in range(i + 1, len(bp_targets)):
            a, b = bp_targets[i], bp_targets[j]
            if a["source"] == b["source"]:
                continue
            systolic_diff = abs(a["systolic"] - b["systolic"])
            if systolic_diff >= 10:
                disagreements.append(ProviderDisagreement(
                    topic="Blood pressure target",
                    provider_a=a["source"],
                    provider_a_position=f"Target <{a['systolic']}/{a['diastolic']} mmHg",
                    provider_b=b["source"],
                    provider_b_position=f"Target <{b['systolic']}/{b['diastolic']} mmHg",
                    severity="high",
                    description=(
                        f"Provider disagreement on BP target: {a['source']} recommends "
                        f"<{a['systolic']}/{a['diastolic']}, but {b['source']} recommends "
                        f"<{b['systolic']}/{b['diastolic']}. {systolic_diff} mmHg systolic difference."
                    ),
                    recommendation=(
                        "Schedule care coordination meeting between providers. "
                        "Consider patient comorbidities (CKD vs cardiovascular risk) "
                        "to establish unified BP target."
                    ),
                ))

    # DEBUG — provider-disagreement detection footprint. PHI-safe:
    # block count + BP-target candidate count + disagreement count
    # only. NEVER provider names, BP values, or block content (all
    # downstream PHI). Topic count helps operators see whether new
    # disagreement-detection rules (beyond BP) start firing.
    topic_counts: dict[str, int] = {}
    for d in disagreements:
        topic_counts[d.topic] = topic_counts.get(d.topic, 0) + 1
    logger.debug(
        "provider_disagreements_detect_complete",
        extra={
            "block_count": len(blocks),
            "bp_target_candidates": len(bp_targets),
            "disagreement_count": len(disagreements),
            "topic_counts": topic_counts,
        },
    )

    return disagreements
