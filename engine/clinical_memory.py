"""
ClinicalMem Engine — persistent clinical memory powered by mind-mem.

Maps FHIR R4 resources to mind-mem memory blocks, providing:
- Hybrid search (BM25 + vector + RRF) over clinical data
- Contradiction detection across patient records
- Importance scoring with clinical acuity awareness
- Confidence gating (abstention when evidence is insufficient)
- Hash-chain audit trail for all clinical decisions (via mind-mem AuditChain)
"""
import json
import logging
import os
import tempfile
import time
from dataclasses import dataclass
from typing import Any

from engine.clinical_scoring import (
    AllergyConflict,
    ClinicalConfidence,
    DrugInteraction,
    LabMedContraindication,
    LabTrend,
    ProviderDisagreement,
    check_allergy_conflicts,
    check_drug_interactions,
    check_lab_medication_contraindications,
    detect_lab_trends,
    detect_provider_disagreements,
    confidence_gate,
    clinical_importance,
    is_negation_query,
)
from engine.llm_synthesizer import (
    ClinicalNarrative,
    explain_conflict,
    generate_clinical_handoff,
)
from engine.fhir_client import (
    BundleFHIRClient,
    FHIRClient,
    FHIRContext,
    extract_condition_name,
    extract_medication_name,
    coding_display,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ClinicalBlock:
    """A clinical memory block — maps a FHIR resource to mind-mem storage."""

    block_id: str
    patient_id: str
    resource_type: str
    title: str
    content: str
    metadata: dict[str, Any]
    timestamp: str
    source: str  # provider / system that created this


@dataclass(frozen=True)
class ClinicalRecallResult:
    """Result from clinical memory recall."""

    blocks: list[dict[str, Any]]
    confidence: ClinicalConfidence
    query: str
    patient_id: str
    audit_hash: str


@dataclass(frozen=True)
class MedicationSafetyReport:
    """Comprehensive medication safety assessment."""

    patient_id: str
    medications: list[str]
    interactions: list[DrugInteraction]
    allergy_conflicts: list[AllergyConflict]
    confidence: ClinicalConfidence
    summary: str
    audit_hash: str


class ClinicalMemEngine:
    """
    Core engine: FHIR data -> mind-mem blocks -> clinical intelligence.

    Each patient gets their own corpus (namespace) in mind-mem.
    Uses mind-mem AuditChain for tamper-proof logging and
    HybridBackend for BM25 + RRF search when available.
    """

    def __init__(self, data_dir: str | None = None):
        self._data_dir = data_dir or os.environ.get(
            "CLINICALMEM_DATA_DIR",
            os.path.join(tempfile.gettempdir(), "clinicalmem_data"),
        )
        os.makedirs(self._data_dir, exist_ok=True)
        self._patient_blocks: dict[str, list[ClinicalBlock]] = {}

        # mind-mem integration
        self._mind_mem_available = False
        self._audit_chain_mm = None  # mind_mem.audit_chain.AuditChain
        self._hybrid_backend = None  # mind_mem.hybrid_recall.HybridBackend
        self._init_mind_mem()

    def _init_mind_mem(self) -> None:
        """Initialize mind-mem AuditChain and HybridBackend if available."""
        try:
            from mind_mem.audit_chain import AuditChain
            from mind_mem.hybrid_recall import HybridBackend

            self._audit_chain_mm = AuditChain(self._data_dir)
            self._hybrid_backend = HybridBackend({"rrf_k": 60, "bm25_weight": 1.0})
            self._mind_mem_available = True
            logger.info("mind-mem initialized: AuditChain + HybridBackend active")
        except ImportError:
            logger.warning("mind-mem not installed — using fallback search and audit")
        except Exception as e:
            logger.warning("mind-mem init failed: %s — using fallback", e)

    def _patient_dir(self, patient_id: str) -> str:
        safe_id = patient_id.replace("/", "_").replace("..", "")
        path = os.path.join(self._data_dir, safe_id)
        os.makedirs(path, exist_ok=True)
        return path

    def _append_audit(self, action: str, details: dict[str, Any]) -> str:
        """Append to audit log. Uses mind-mem AuditChain when available."""
        if self._audit_chain_mm is not None:
            return self._append_audit_mindmem(action, details)
        return self._append_audit_fallback(action, details)

    def _append_audit_mindmem(self, action: str, details: dict[str, Any]) -> str:
        """Append via mind-mem's Merkle-chain audit ledger."""
        # Map clinical actions to mind-mem's valid operations
        op_map = {
            "ingest_fhir": "create_block",
            "store_observation": "append_block",
            "recall": "apply_proposal",
            "medication_safety_check": "apply_proposal",
            "detect_contradictions": "apply_proposal",
            "patient_summary": "apply_proposal",
            "treatment_dependencies": "apply_proposal",
        }
        operation = op_map.get(action, "apply_proposal")
        target = details.get("patient_id", "system")

        entry = self._audit_chain_mm.append(
            operation=operation,
            target=f"patient/{target}",
            agent="clinicalmem",
            reason=f"clinical_{action}",
            payload=details,
        )
        return entry.entry_hash

    def _append_audit_fallback(self, action: str, details: dict[str, Any]) -> str:
        """Fallback: in-memory hash chain when mind-mem is unavailable."""
        import hashlib

        if not hasattr(self, "_audit_chain_fallback"):
            self._audit_chain_fallback: list[dict[str, Any]] = []

        prev_hash = (
            self._audit_chain_fallback[-1]["hash"]
            if self._audit_chain_fallback
            else "genesis"
        )
        entry = {
            "action": action,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "details": details,
            "prev_hash": prev_hash,
        }
        entry_bytes = json.dumps(entry, sort_keys=True).encode()
        entry["hash"] = hashlib.sha256(entry_bytes).hexdigest()
        self._audit_chain_fallback.append(entry)
        return entry["hash"]

    # ── Ingest FHIR data ──────────────────────────────────────────────────

    def ingest_from_bundle(self, bundle: dict, patient_id: str) -> dict[str, int]:
        """Ingest a FHIR Bundle directly (for demo/testing without a live FHIR server).

        Parses the bundle and feeds resources directly into ingest_from_fhir
        via a BundleFHIRClient adapter — no mocking or monkey-patching required.
        """
        entries = bundle.get("entry", [])
        resources_by_type: dict[str, list] = {}
        for entry in entries:
            res = entry.get("resource", {})
            rt = res.get("resourceType", "")
            resources_by_type.setdefault(rt, []).append(res)

        fhir = BundleFHIRClient(resources_by_type, patient_id)
        return self.ingest_from_fhir(fhir)

    def ingest_from_fhir(self, fhir: FHIRClient) -> dict[str, int]:
        """Pull patient data from FHIR server and store as clinical blocks."""
        pid = fhir.patient_id
        counts = {"medications": 0, "conditions": 0, "allergies": 0, "observations": 0}

        try:
            meds = fhir.get_medications()
            for res in meds:
                name = extract_medication_name(res)
                dosage = ""
                for d in res.get("dosageInstruction", []):
                    dosage = d.get("text", "")
                    break
                block = ClinicalBlock(
                    block_id=res.get("id", f"med-{counts['medications']}"),
                    patient_id=pid,
                    resource_type="MedicationRequest",
                    title=f"Medication: {name}",
                    content=f"Active medication {name}. Dosage: {dosage}. "
                    f"Prescribed by: {(res.get('requester') or {}).get('display', 'Unknown')}. "
                    f"Date: {res.get('authoredOn', 'Unknown')}.",
                    metadata={
                        "medication_name": name,
                        "status": res.get("status"),
                        "dosage": dosage,
                        "authored_on": res.get("authoredOn"),
                        "requester": (res.get("requester") or {}).get("display"),
                    },
                    timestamp=res.get("authoredOn", ""),
                    source=(res.get("requester") or {}).get("display", "FHIR"),
                )
                self._store_block(block)
                counts["medications"] += 1
        except Exception as e:
            logger.error("Failed to ingest medications: %s", e)

        try:
            conditions = fhir.get_conditions()
            for res in conditions:
                name = extract_condition_name(res)
                onset = res.get("onsetDateTime") or (
                    (res.get("onsetPeriod") or {}).get("start", "Unknown")
                )
                block = ClinicalBlock(
                    block_id=res.get("id", f"cond-{counts['conditions']}"),
                    patient_id=pid,
                    resource_type="Condition",
                    title=f"Condition: {name}",
                    content=f"Active condition: {name}. Onset: {onset}. "
                    f"Recorded: {res.get('recordedDate', 'Unknown')}.",
                    metadata={
                        "condition_name": name,
                        "clinical_status": (
                            (res.get("clinicalStatus") or {})
                            .get("coding", [{}])[0]
                            .get("code")
                        ),
                        "severity": (res.get("severity") or {}).get("text"),
                        "onset": onset,
                    },
                    timestamp=res.get("recordedDate", ""),
                    source="FHIR",
                )
                self._store_block(block)
                counts["conditions"] += 1
        except Exception as e:
            logger.error("Failed to ingest conditions: %s", e)

        try:
            allergies = fhir.get_allergies()
            for res in allergies:
                code = res.get("code", {})
                name = code.get("text") or coding_display(code.get("coding", []))
                reactions = []
                for r in res.get("reaction", []):
                    for m in r.get("manifestation", []):
                        reactions.append(
                            m.get("text") or coding_display(m.get("coding", []))
                        )
                block = ClinicalBlock(
                    block_id=res.get("id", f"allergy-{counts['allergies']}"),
                    patient_id=pid,
                    resource_type="AllergyIntolerance",
                    title=f"Allergy: {name}",
                    content=f"Allergy to {name}. Criticality: {res.get('criticality', 'Unknown')}. "
                    f"Reactions: {', '.join(reactions) if reactions else 'Not specified'}.",
                    metadata={
                        "allergen": name,
                        "criticality": res.get("criticality"),
                        "reactions": reactions,
                        "verification": (
                            (res.get("verificationStatus") or {})
                            .get("coding", [{}])[0]
                            .get("code")
                        ),
                    },
                    timestamp=res.get("recordedDate", ""),
                    source="FHIR",
                )
                self._store_block(block)
                counts["allergies"] += 1
        except Exception as e:
            logger.error("Failed to ingest allergies: %s", e)

        try:
            for category in ["vital-signs", "laboratory"]:
                obs_list = fhir.get_observations(category=category)
                for res in obs_list:
                    code = res.get("code", {})
                    name = code.get("text") or coding_display(code.get("coding", []))
                    value = ""
                    value_numeric = None
                    unit = ""
                    if "valueQuantity" in res:
                        vq = res["valueQuantity"]
                        value_numeric = vq.get("value")
                        unit = vq.get("unit", "")
                        value = f"{value_numeric} {unit}"
                    elif "valueString" in res:
                        value = res["valueString"]

                    # Extract notes
                    notes = " ".join(
                        n.get("text", "") for n in res.get("note", [])
                    )

                    # Extract performer
                    performers = res.get("performer", [])
                    performer = performers[0].get("display", "FHIR") if performers else "FHIR"

                    # Build richer content including notes and targets
                    content_parts = [
                        f"{name}: {value}.",
                        f"Date: {res.get('effectiveDateTime', 'Unknown')}.",
                        f"Status: {res.get('status', 'Unknown')}.",
                        f"Recorded by: {performer}.",
                    ]
                    if notes:
                        content_parts.append(f"Notes: {notes}")

                    # Extract component values (e.g., BP systolic/diastolic)
                    components = {}
                    for comp in res.get("component", []):
                        comp_name = (comp.get("code", {}).get("coding", [{}])[0]
                                     .get("display", ""))
                        comp_vq = comp.get("valueQuantity", {})
                        if comp_vq:
                            components[comp_name] = f"{comp_vq.get('value')} {comp_vq.get('unit', '')}"
                    if components:
                        content_parts.append(
                            "Components: " + "; ".join(
                                f"{k}: {v}" for k, v in components.items()
                            )
                        )

                    block = ClinicalBlock(
                        block_id=res.get("id", f"obs-{counts['observations']}"),
                        patient_id=pid,
                        resource_type="Observation",
                        title=f"{category}: {name}",
                        content=" ".join(content_parts),
                        metadata={
                            "observation_name": name,
                            "value": value_numeric if value_numeric is not None else value,
                            "unit": unit,
                            "category": category,
                            "effective_date": res.get("effectiveDateTime"),
                            "interpretation": (
                                (res.get("interpretation") or [{}])[0].get("text")
                            ),
                            "notes": notes,
                            "components": components,
                            "performer": performer,
                        },
                        timestamp=res.get("effectiveDateTime", ""),
                        source=performer,
                    )
                    self._store_block(block)
                    counts["observations"] += 1
        except Exception as e:
            logger.error("Failed to ingest observations: %s", e)

        audit_hash = self._append_audit(
            "ingest_fhir",
            {"patient_id": pid, "counts": counts},
        )
        logger.info("Ingested FHIR data for patient %s: %s (audit: %s)", pid, counts, audit_hash)
        return counts

    def _store_block(self, block: ClinicalBlock) -> None:
        """Store a clinical block in the patient's memory.

        Also writes a Markdown file for mind-mem's BM25 corpus search.
        """
        pid = block.patient_id
        if pid not in self._patient_blocks:
            self._patient_blocks[pid] = []
        self._patient_blocks[pid].append(block)

        # Write mind-mem compatible Markdown block for BM25 search
        if self._mind_mem_available:
            self._write_block_markdown(block)

    def _write_block_markdown(self, block: ClinicalBlock) -> None:
        """Write a clinical block as mind-mem compatible Markdown for BM25 corpus."""
        pdir = self._patient_dir(block.patient_id)
        corpus_dir = os.path.join(pdir, "corpus")
        os.makedirs(corpus_dir, exist_ok=True)

        # One file per resource type per patient
        safe_type = block.resource_type.replace("/", "_")
        fpath = os.path.join(corpus_dir, f"{safe_type}.md")

        # Append mind-mem block format: [ID] header + key-value fields
        md_block = (
            f"\n[{block.block_id}]\n"
            f"Title: {block.title}\n"
            f"Status: active\n"
            f"Type: {block.resource_type}\n"
            f"Source: {block.source}\n"
            f"Timestamp: {block.timestamp}\n"
            f"Content: {block.content}\n"
        )
        for key, val in block.metadata.items():
            if val is not None and key not in ("type", "source"):
                md_block += f"{key}: {val}\n"

        with open(fpath, "a", encoding="utf-8") as f:
            f.write(md_block)

    # ── Recall ────────────────────────────────────────────────────────────

    def recall(
        self, patient_id: str, query: str, top_k: int = 10
    ) -> ClinicalRecallResult:
        """
        Recall clinical context for a patient query.

        Uses mind-mem HybridBackend (BM25 + RRF) when available,
        with MIND-kernel confidence gating.
        """
        blocks = self._patient_blocks.get(patient_id, [])
        if not blocks:
            empty_conf = confidence_gate([], [])
            audit_hash = self._append_audit(
                "recall",
                {"patient_id": patient_id, "query": query, "results": 0},
            )
            return ClinicalRecallResult(
                blocks=[],
                confidence=empty_conf,
                query=query,
                patient_id=patient_id,
                audit_hash=audit_hash,
            )

        # Try mind-mem HybridBackend first
        if self._hybrid_backend is not None:
            return self._recall_mindmem(patient_id, query, blocks, top_k)

        return self._recall_fallback(patient_id, query, blocks, top_k)

    def _recall_mindmem(
        self,
        patient_id: str,
        query: str,
        blocks: list[ClinicalBlock],
        top_k: int,
    ) -> ClinicalRecallResult:
        """Recall using mind-mem's HybridBackend (real BM25 + RRF fusion)."""
        pdir = self._patient_dir(patient_id)

        try:
            mm_results = self._hybrid_backend.search(
                query=query,
                workspace=pdir,
                limit=top_k,
            )
        except Exception as e:
            logger.warning("mind-mem hybrid search failed: %s — falling back", e)
            return self._recall_fallback(patient_id, query, blocks, top_k)

        if not mm_results:
            # mind-mem found nothing (maybe corpus not indexed yet), fall back
            return self._recall_fallback(patient_id, query, blocks, top_k)

        # Map mind-mem results back to clinical blocks
        block_map = {b.block_id: b for b in blocks}
        result_blocks = []
        bm25_scores = []
        entity_overlaps = []

        for item in mm_results:
            bid = item.get("_id", "") or item.get("id", "")
            score = item.get("rrf_score", 0.0) or item.get("score", 0.0)
            matched_block = block_map.get(bid)

            if matched_block:
                # Compute entity overlap for confidence gating
                query_terms = set(query.lower().split())
                meta_values = " ".join(
                    str(v) for v in matched_block.metadata.values()
                ).lower()
                entity_hits = sum(1 for t in query_terms if t in meta_values)
                overlap = entity_hits / (len(query_terms) + 1e-6)

                result_blocks.append({
                    "block_id": matched_block.block_id,
                    "title": matched_block.title,
                    "content": matched_block.content,
                    "resource_type": matched_block.resource_type,
                    "score": round(score, 4),
                    "metadata": matched_block.metadata,
                    "search_backend": "mind-mem/hybrid",
                })
                bm25_scores.append(min(score * 10, 1.0))  # Normalize RRF score
                entity_overlaps.append(overlap)
            else:
                # Block from mind-mem corpus but not in our map — include raw
                result_blocks.append({
                    "block_id": bid,
                    "title": item.get("Title", bid),
                    "content": item.get("Content", item.get("excerpt", "")),
                    "resource_type": item.get("Type", "Unknown"),
                    "score": round(score, 4),
                    "metadata": {},
                    "search_backend": "mind-mem/hybrid",
                })
                bm25_scores.append(min(score * 10, 1.0))
                entity_overlaps.append(0.0)

        conf = confidence_gate(bm25_scores, entity_overlaps)

        audit_hash = self._append_audit(
            "recall",
            {
                "patient_id": patient_id,
                "query": query,
                "results": len(result_blocks),
                "confidence": conf.score,
                "abstained": conf.should_abstain,
                "backend": "mind-mem/hybrid",
            },
        )

        return ClinicalRecallResult(
            blocks=result_blocks,
            confidence=conf,
            query=query,
            patient_id=patient_id,
            audit_hash=audit_hash,
        )

    def _recall_fallback(
        self,
        patient_id: str,
        query: str,
        blocks: list[ClinicalBlock],
        top_k: int,
    ) -> ClinicalRecallResult:
        """Fallback recall using approximate BM25 term matching."""
        is_negation = is_negation_query(query)
        query_terms = set(query.lower().split())

        scored = []
        for block in blocks:
            content_lower = block.content.lower()
            title_lower = block.title.lower()

            term_hits = sum(
                1 for t in query_terms if t in content_lower or t in title_lower
            )
            bm25_approx = term_hits / (len(query_terms) + 1e-6)

            meta_values = " ".join(str(v) for v in block.metadata.values()).lower()
            entity_hits = sum(1 for t in query_terms if t in meta_values)
            entity_overlap = entity_hits / (len(query_terms) + 1e-6)

            importance = clinical_importance(
                access_count=1,
                days_since_access=0,
                connection_degree=len(block.metadata),
                is_acute=block.resource_type in ("Condition", "AllergyIntolerance"),
            )

            final_score = (bm25_approx * 0.6 + entity_overlap * 0.4) * importance

            if is_negation:
                has_negation = any(
                    neg in content_lower
                    for neg in ["not ", "no ", "negative", "denies", "without"]
                )
                if not has_negation:
                    final_score *= 0.5

            scored.append((block, final_score, bm25_approx, entity_overlap))

        scored.sort(key=lambda x: x[1], reverse=True)
        top = scored[:top_k]

        bm25_scores = [s[2] for s in top]
        entity_overlaps_list = [s[3] for s in top]
        conf = confidence_gate(bm25_scores, entity_overlaps_list)

        result_blocks = [
            {
                "block_id": b.block_id,
                "title": b.title,
                "content": b.content,
                "resource_type": b.resource_type,
                "score": round(score, 4),
                "metadata": b.metadata,
                "search_backend": "fallback/bm25-approx",
            }
            for b, score, _, _ in top
        ]

        audit_hash = self._append_audit(
            "recall",
            {
                "patient_id": patient_id,
                "query": query,
                "results": len(result_blocks),
                "confidence": conf.score,
                "abstained": conf.should_abstain,
                "backend": "fallback/bm25-approx",
            },
        )

        return ClinicalRecallResult(
            blocks=result_blocks,
            confidence=conf,
            query=query,
            patient_id=patient_id,
            audit_hash=audit_hash,
        )

    # ── Medication Safety ─────────────────────────────────────────────────

    def medication_safety_check(
        self, patient_id: str, fhir: FHIRClient | None = None
    ) -> MedicationSafetyReport:
        """
        Comprehensive medication safety assessment.

        Checks drug-drug interactions and allergy cross-reactions.
        """
        blocks = self._patient_blocks.get(patient_id, [])

        med_names = []
        allergy_names = []

        for block in blocks:
            if block.resource_type == "MedicationRequest":
                med_names.append(block.metadata.get("medication_name", ""))
            elif block.resource_type == "AllergyIntolerance":
                allergy_names.append(block.metadata.get("allergen", ""))

        interactions = check_drug_interactions(med_names)
        allergy_conflicts = check_allergy_conflicts(allergy_names, med_names)

        # Confidence is based on data completeness
        data_completeness = min(len(med_names) / 3, 1.0)  # Expect at least 3 meds
        conf = confidence_gate(
            [data_completeness],
            [1.0 if allergy_names else 0.5],
        )

        # Build summary
        parts = []
        if interactions:
            parts.append(
                f"{len(interactions)} drug interaction(s) detected: "
                + "; ".join(
                    f"{i.drug_a} + {i.drug_b} ({i.severity}: {i.description})"
                    for i in interactions
                )
            )
        else:
            parts.append("No known drug interactions detected.")

        if allergy_conflicts:
            parts.append(
                f"{len(allergy_conflicts)} allergy conflict(s): "
                + "; ".join(
                    f"{c.allergen} allergy vs {c.medication} ({c.description})"
                    for c in allergy_conflicts
                )
            )
        else:
            parts.append("No allergy conflicts detected.")

        summary = " ".join(parts)

        audit_hash = self._append_audit(
            "medication_safety_check",
            {
                "patient_id": patient_id,
                "medications": med_names,
                "allergies": allergy_names,
                "interaction_count": len(interactions),
                "allergy_conflict_count": len(allergy_conflicts),
            },
        )

        return MedicationSafetyReport(
            patient_id=patient_id,
            medications=med_names,
            interactions=interactions,
            allergy_conflicts=allergy_conflicts,
            confidence=conf,
            summary=summary,
            audit_hash=audit_hash,
        )

    # ── Contradiction Detection ───────────────────────────────────────────

    def detect_contradictions(self, patient_id: str) -> list[dict[str, Any]]:
        """
        Detect contradictions in patient clinical data.

        Checks for:
        - Allergy vs prescription conflicts (e.g., Penicillin allergy + Amoxicillin)
        - Dangerous drug-drug interactions (e.g., Warfarin + Ibuprofen)
        - Lab-medication contraindications (e.g., declining GFR + Metformin)
        - Lab value trends (e.g., GFR 45→38→32 declining trajectory)
        - Provider disagreements (e.g., conflicting BP targets)
        """
        blocks = self._patient_blocks.get(patient_id, [])
        contradictions = []

        # 1. Allergy-medication conflicts
        safety = self.medication_safety_check(patient_id)
        for conflict in safety.allergy_conflicts:
            contradictions.append({
                "type": "allergy_medication_conflict",
                "severity": "critical",
                "description": (
                    f"Patient has {conflict.allergen} allergy but is prescribed "
                    f"{conflict.medication} ({conflict.description})"
                ),
                "recommendation": (
                    f"STOP {conflict.medication} immediately. "
                    f"Use alternative outside the {conflict.cross_reaction_group} class."
                ),
                "blocks_involved": [conflict.allergen, conflict.medication],
            })

        # 2. Drug interactions
        for interaction in safety.interactions:
            if interaction.severity in ("contraindicated", "serious"):
                contradictions.append({
                    "type": "drug_interaction",
                    "severity": "high" if interaction.severity == "serious" else "critical",
                    "description": (
                        f"{interaction.drug_a} + {interaction.drug_b}: "
                        f"{interaction.description}"
                    ),
                    "recommendation": (
                        f"Review co-prescription of {interaction.drug_a} and "
                        f"{interaction.drug_b}. Consider alternatives or close monitoring."
                    ),
                    "blocks_involved": [interaction.drug_a, interaction.drug_b],
                })

        # 3. Lab-medication contraindications (e.g., declining GFR + Metformin)
        med_names = [
            b.metadata.get("medication_name", "")
            for b in blocks if b.resource_type == "MedicationRequest"
        ]
        obs_dicts = [
            {
                "observation_name": b.metadata.get("observation_name", ""),
                "value": b.metadata.get("value", ""),
                "unit": b.metadata.get("unit", ""),
                "effective_date": b.metadata.get("effective_date", ""),
            }
            for b in blocks if b.resource_type == "Observation"
        ]
        lab_contras = check_lab_medication_contraindications(obs_dicts, med_names)
        for lc in lab_contras:
            contradictions.append({
                "type": "lab_medication_contraindication",
                "severity": lc.severity,
                "description": (
                    f"{lc.lab_name} = {lc.lab_value} {lc.lab_unit} "
                    f"({lc.direction} threshold {lc.threshold}): {lc.description}"
                ),
                "recommendation": lc.recommendation,
                "blocks_involved": [lc.lab_name, lc.medication],
            })

        # 4. Lab trends (e.g., declining GFR trajectory)
        lab_trends = detect_lab_trends(obs_dicts)
        for trend in lab_trends:
            contradictions.append({
                "type": "lab_trend_alert",
                "severity": trend.severity,
                "description": trend.description,
                "recommendation": trend.recommendation,
                "blocks_involved": [trend.lab_name],
            })

        # 5. Provider disagreements (e.g., conflicting BP targets)
        block_dicts = [
            {
                "title": b.title,
                "content": b.content,
                "source": b.source,
                "metadata": {
                    **b.metadata,
                    "notes": " ".join(
                        str(v) for v in b.metadata.values() if isinstance(v, str)
                    ),
                },
            }
            for b in blocks
        ]
        provider_conflicts = detect_provider_disagreements(block_dicts)
        for pc in provider_conflicts:
            contradictions.append({
                "type": "provider_disagreement",
                "severity": pc.severity,
                "description": pc.description,
                "recommendation": pc.recommendation,
                "blocks_involved": [pc.provider_a, pc.provider_b],
            })

        self._append_audit(
            "detect_contradictions",
            {
                "patient_id": patient_id,
                "contradiction_count": len(contradictions),
                "types_found": list({c["type"] for c in contradictions}),
            },
        )

        return contradictions

    # ── Audit Trail ───────────────────────────────────────────────────────

    def get_audit_trail(self, limit: int = 50) -> list[dict[str, Any]]:
        """Return the hash-chain audit trail."""
        if self._audit_chain_mm is not None:
            return self._get_audit_trail_mindmem(limit)
        fallback = getattr(self, "_audit_chain_fallback", [])
        return fallback[-limit:]

    def _get_audit_trail_mindmem(self, limit: int) -> list[dict[str, Any]]:
        """Read audit trail from mind-mem's JSONL ledger."""
        chain_path = self._audit_chain_mm._chain_path
        if not os.path.isfile(chain_path):
            return []
        entries = []
        with open(chain_path, "r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if stripped:
                    entries.append(json.loads(stripped))
        return entries[-limit:]

    def verify_audit_chain(self) -> bool:
        """Verify integrity of the audit chain (tamper detection)."""
        if self._audit_chain_mm is not None:
            is_valid, errors = self._audit_chain_mm.verify()
            if errors:
                logger.warning("Audit chain verification errors: %s", errors)
            return is_valid
        # Fallback verification
        fallback = getattr(self, "_audit_chain_fallback", [])
        for i, entry in enumerate(fallback):
            if i == 0:
                if entry.get("prev_hash") != "genesis":
                    return False
            else:
                if entry.get("prev_hash") != fallback[i - 1]["hash"]:
                    return False
        return True

    # ── Patient Summary ───────────────────────────────────────────────────

    def patient_summary(self, patient_id: str) -> dict[str, Any]:
        """Generate a structured patient summary from stored blocks."""
        blocks = self._patient_blocks.get(patient_id, [])

        by_type: dict[str, list[ClinicalBlock]] = {}
        for block in blocks:
            by_type.setdefault(block.resource_type, []).append(block)

        meds = [
            {
                "name": b.metadata.get("medication_name"),
                "dosage": b.metadata.get("dosage"),
                "prescribed_by": b.metadata.get("requester"),
                "date": b.metadata.get("authored_on"),
            }
            for b in by_type.get("MedicationRequest", [])
        ]

        conditions = [
            {
                "name": b.metadata.get("condition_name"),
                "severity": b.metadata.get("severity"),
                "onset": b.metadata.get("onset"),
            }
            for b in by_type.get("Condition", [])
        ]

        allergies = [
            {
                "allergen": b.metadata.get("allergen"),
                "criticality": b.metadata.get("criticality"),
                "reactions": b.metadata.get("reactions", []),
            }
            for b in by_type.get("AllergyIntolerance", [])
        ]

        observations = [
            {
                "name": b.metadata.get("observation_name"),
                "value": b.metadata.get("value"),
                "category": b.metadata.get("category"),
                "date": b.metadata.get("effective_date"),
            }
            for b in by_type.get("Observation", [])
        ]

        audit_hash = self._append_audit(
            "patient_summary", {"patient_id": patient_id}
        )

        return {
            "patient_id": patient_id,
            "total_blocks": len(blocks),
            "medications": meds,
            "conditions": conditions,
            "allergies": allergies,
            "recent_observations": observations[:10],
            "audit_hash": audit_hash,
        }

    # ── LLM-Grounded Clinical Synthesis ───────────────────────────────────

    def explain_clinical_conflict(
        self, patient_id: str, conflict_index: int = 0
    ) -> ClinicalNarrative:
        """
        Generate a patient-specific LLM explanation for a detected conflict.

        Uses deterministic detection + GenAI synthesis pattern:
        - Detection: rule-based (reliable, auditable)
        - Explanation: LLM-generated (expressive, context-aware)
        - Abstention: hard gate when evidence is insufficient
        """
        contradictions = self.detect_contradictions(patient_id)
        if not contradictions or conflict_index >= len(contradictions):
            return ClinicalNarrative(
                narrative="ABSTAIN: No conflicts detected for this patient.",
                evidence_citations=[],
                confidence_score=0.0,
                abstained=True,
                model_used="abstention_gate",
                audit_context={"reason": "no_conflicts"},
            )

        conflict = contradictions[conflict_index]
        patient_ctx = self.patient_summary(patient_id)

        # Get evidence blocks relevant to this conflict
        involved = conflict.get("blocks_involved", [])
        blocks = self._patient_blocks.get(patient_id, [])
        evidence = [
            {
                "block_id": b.block_id,
                "title": b.title,
                "content": b.content,
                "resource_type": b.resource_type,
                "metadata": b.metadata,
            }
            for b in blocks
            if any(
                term.lower() in b.content.lower() or term.lower() in b.title.lower()
                for term in involved
            )
        ]

        narrative = explain_conflict(conflict, patient_ctx, evidence)

        self._append_audit(
            "explain_conflict",
            {
                "patient_id": patient_id,
                "conflict_type": conflict.get("type"),
                "conflict_index": conflict_index,
                "abstained": narrative.abstained,
                "model_used": narrative.model_used,
                "citations": len(narrative.evidence_citations),
            },
        )

        return narrative

    def clinical_handoff(self, patient_id: str) -> ClinicalNarrative:
        """
        Generate a complete clinical care handoff note with LLM synthesis.

        Combines all detected findings into a structured clinician-ready note
        with evidence citations. Demonstrates GenAI + deterministic safety rails.
        """
        contradictions = self.detect_contradictions(patient_id)
        patient_ctx = self.patient_summary(patient_id)
        safety = self.medication_safety_check(patient_id)

        blocks = self._patient_blocks.get(patient_id, [])
        evidence = [
            {
                "block_id": b.block_id,
                "title": b.title,
                "content": b.content,
                "resource_type": b.resource_type,
                "metadata": b.metadata,
            }
            for b in blocks
        ]

        safety_report = {
            "interaction_count": len(safety.interactions),
            "allergy_conflict_count": len(safety.allergy_conflicts),
        }

        narrative = generate_clinical_handoff(
            patient_ctx, contradictions, safety_report, evidence
        )

        self._append_audit(
            "clinical_handoff",
            {
                "patient_id": patient_id,
                "contradiction_count": len(contradictions),
                "abstained": narrative.abstained,
                "model_used": narrative.model_used,
                "citations": len(narrative.evidence_citations),
            },
        )

        return narrative
