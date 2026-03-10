# ClinicalMem Schema Reference

> This file makes any coding agent ClinicalMem-literate. Drop it into your project as CLAUDE.md, .cursor/rules/, codex.md, or GEMINI.md. The agent will use exact tool names, parameters, and response formats — zero hallucination.

## MCP Server — 18 Tools

Server: FastMCP 2.x | Transport: Streamable HTTP | Port: 8080

### store_clinical_observation
Store a clinical observation or note in persistent memory.
```
Parameters:
  patient_id: str          # REQUIRED — FHIR patient ID
  observation_type: str    # REQUIRED — e.g. "clinical_note", "lab_result", "medication_change"
  title: str               # REQUIRED — brief title
  content: str             # REQUIRED — full text content
  source: str = "manual"   # who/what created this
  fhir_server_url: str     # SHARP-on-MCP header
  fhir_access_token: str   # SHARP-on-MCP header

Returns: { status, block_id, audit_hash, message }
```

### recall_patient_context
Retrieve patient history by query using hybrid search (BM25 + vector + RRF).
```
Parameters:
  patient_id: str          # REQUIRED
  query: str               # REQUIRED — natural language clinical query
  top_k: int = 10          # max results
  fhir_server_url: str
  fhir_access_token: str

Returns: {
  status, patient_id, query,
  confidence: { score: float, level: str, should_abstain: bool, reason: str },
  results: list[block],
  result_count: int,
  audit_hash: str
}
```

### check_medication_conflicts
Detect drug-drug interactions for a patient's active medications.
```
Parameters:
  patient_id: str          # REQUIRED
  fhir_server_url: str
  fhir_access_token: str

Returns: {
  status, patient_id, medications: list[str],
  interactions: list[{ drug_a, drug_b, severity, description, score }],
  interaction_count: int,
  allergy_conflicts: list[{ allergen, medication, cross_reaction_group, description }],
  allergy_conflict_count: int,
  recommendations: list[str],
  confidence: { score, level },
  summary: str,
  audit_hash: str
}
```

### check_allergy_conflicts
Cross-reference medications against known patient allergies. Detects cross-reactivity (e.g., Penicillin allergy vs Amoxicillin).
```
Parameters:
  patient_id: str          # REQUIRED
  fhir_server_url: str
  fhir_access_token: str

Returns: {
  status, patient_id,
  allergy_conflicts: list[{ allergen, medication, cross_reaction_group, description }],
  conflict_count: int,
  audit_hash: str
}
```

### detect_belief_drift
Comprehensive clinical contradiction scanner. Detects 5 types: allergy-medication conflicts, drug-drug interactions, lab-medication contraindications, lab trend alerts, provider disagreements.
```
Parameters:
  patient_id: str          # REQUIRED
  fhir_server_url: str
  fhir_access_token: str

Returns: {
  status, patient_id,
  contradictions: list[{ type, severity, description, ... }],
  contradiction_count: int,
  critical_count: int,
  high_count: int,
  types_found: list[str],
  has_critical: bool,
  escalation: str | null
}
```

### ingest_patient_data
Pull and store all FHIR data for a patient into clinical memory.
```
Parameters:
  patient_id: str          # REQUIRED
  fhir_server_url: str     # REQUIRED
  fhir_access_token: str   # REQUIRED

Returns: { status, patient_id, ingested: dict, total_blocks: int, message }
```

### explain_clinical_conflict
Generate LLM explanation for a detected clinical conflict. Uses deterministic detection + GenAI synthesis + abstention gate.
```
Parameters:
  patient_id: str          # REQUIRED
  conflict_index: int = 0  # 0-based index of conflict to explain
  fhir_server_url: str
  fhir_access_token: str

Returns: {
  status, patient_id,
  narrative: str,
  evidence_citations: list[str],
  confidence_score: float,
  abstained: bool,
  model_used: str
}
```

### clinical_care_handoff
Generate a complete clinician-ready handoff note with evidence citations.
```
Parameters:
  patient_id: str          # REQUIRED
  fhir_server_url: str
  fhir_access_token: str

Returns: {
  status, patient_id,
  handoff_note: str,
  evidence_citations: list[str],
  confidence_score: float,
  abstained: bool,
  model_used: str
}
```

### get_treatment_dependencies
Show causal chain of treatment decisions (condition → medications → observations).
```
Parameters:
  patient_id: str          # REQUIRED
  fhir_server_url: str
  fhir_access_token: str

Returns: {
  status, patient_id,
  dependencies: list[{ condition, severity, onset, related_medications }],
  audit_hash: str
}
```

### get_clinical_audit_trail
Retrieve SHA-256 hash-chain audit trail of all clinical decisions.
```
Parameters:
  limit: int = 50          # max entries (most recent first)

Returns: {
  status,
  chain_integrity: "verified" | "TAMPERED",
  entry_count: int,
  entries: list[audit_entry]
}
```

### summarize_patient_history
Condensed patient overview with MIND Lang importance-scored relevance.
```
Parameters:
  patient_id: str          # REQUIRED
  fhir_server_url: str
  fhir_access_token: str

Returns: { status, patient_id, conditions, medications, allergies, observations, ... }
```

### what_if_medication_change
Simulate safety impact of adding, removing, or swapping a medication BEFORE it reaches the patient.
```
Parameters:
  patient_id: str          # REQUIRED
  action: str              # REQUIRED — "add", "remove", or "swap"
  medication: str          # REQUIRED — medication to add/remove (or remove in swap)
  swap_to: str = ""        # for swap action only — the replacement
  fhir_server_url: str
  fhir_access_token: str

Returns: {
  status,
  scenario: str,
  patient_id: str,
  safe_to_proceed: bool,
  risk_delta: int,
  new_interactions: list[{ drug_a, drug_b, severity, description }],
  new_allergy_conflicts: list[{ allergen, medication, description }],
  removed_risks: list[str],
  recommendation: str,
  audit_hash: str
}
```

### verify_clinical_claims
Hallucination detector — verify LLM-generated clinical text is grounded in patient evidence.
```
Parameters:
  text: str                # REQUIRED — LLM-generated clinical text to verify
  patient_id: str          # REQUIRED
  fhir_server_url: str
  fhir_access_token: str

Returns: {
  status,
  grounding_score: float,
  grounded_count: int,
  ungrounded_count: int,
  flagged_hallucinations: list[str],
  claims: list[{ claim, grounded: bool, confidence: float, evidence_block_ids: list }]
}
```

### scan_for_phi
Scan clinical text for Protected Health Information (HIPAA categories).
```
Parameters:
  text: str                # REQUIRED — text to scan

Returns: {
  status,
  is_safe: bool,
  phi_count: int,
  categories_found: list[str],
  redacted_text: str,
  matches: list[{ category, text, confidence }]
}
```

### check_fda_safety_alerts
Query real FDA safety data (openFDA) for patient's active medications.
```
Parameters:
  patient_id: str          # REQUIRED
  fhir_server_url: str
  fhir_access_token: str

Returns: {
  status,
  medications: list[str],
  alert_count: int,
  alerts: list[{ drug, type, severity, description, source }],
  black_box_warnings: list[str],
  total_adverse_event_reports: int,
  highest_severity: str,
  audit_hash: str
}
```

### find_matching_trials
Match patient conditions to active ClinicalTrials.gov trials.
```
Parameters:
  patient_id: str          # REQUIRED
  fhir_server_url: str
  fhir_access_token: str

Returns: {
  status,
  patient_conditions: list[str],
  total_trials_found: int,
  trials: list[{ nct_id, title, status, conditions, interventions, phase, enrollment, locations, url }],
  audit_hash: str
}
```

### consensus_verify_finding
Multi-LLM consensus verification — 6 US-based models must agree on critical findings.
```
Parameters:
  finding: str             # REQUIRED — clinical safety finding to verify
  patient_id: str          # REQUIRED
  fhir_server_url: str
  fhir_access_token: str

Returns: {
  status,
  finding: str,
  consensus_level: str,    # "strong", "majority", "weak", "no_consensus"
  agreement: str,          # e.g. "5/6"
  confidence_score: float,
  should_report: bool,
  verdicts: list[{ model, agrees: bool, confidence: float, reasoning: str }],
  audit_hash: str
}
```

### health_check
Container health check for Azure/K8s.
```
Parameters: (none)

Returns: { status: "healthy", engine_ready: bool, mind_mem_available: bool, audit_chain_active: bool }
```

---

## A2A Agent — 13 Tools

Agent: Google ADK | Model: gemini-2.5-flash | Protocol: A2A

### FHIR Query Tools
| Tool | Description |
|------|-------------|
| `get_patient_demographics` | Patient demographics from FHIR |
| `get_active_medications` | Active medication list |
| `get_active_conditions` | Active condition list |
| `get_recent_observations` | Recent lab results and vitals |

### Clinical Memory Tools
| Tool | Description |
|------|-------------|
| `recall_clinical_context` | Hybrid search (BM25+vector+RRF) with confidence gating |
| `store_clinical_note` | Store observation with audit trail |

### Safety Analysis Tools
| Tool | Description |
|------|-------------|
| `medication_safety_review` | Drug interactions + allergy conflicts |
| `detect_record_contradictions` | 5-type contradiction scanner |
| `explain_clinical_conflict` | LLM explanation with evidence citations |

### v4.0 Tools
| Tool | Description |
|------|-------------|
| `what_if_scenario` | Simulate medication changes before applying |
| `check_fda_alerts` | Real openFDA safety data |
| `find_clinical_trials` | ClinicalTrials.gov matching |
| `consensus_verify` | 6-model LLM consensus verification |

---

## Engine Modules — 13

| Module | Purpose |
|--------|---------|
| `clinical_memory.py` | Core memory engine — block storage, recall, ingestion, audit chain |
| `clinical_scoring.py` | MIND Lang kernels — abstention, importance, adversarial |
| `fhir_client.py` | FHIR R4 client — patient data retrieval |
| `llm_synthesizer.py` | Medical LLM cascade — GPT-5.4 → MedGemma → Gemini Flash |
| `rxnorm_client.py` | RxNorm REST API — drug name → RxCUI, pairwise interactions |
| `snomed_client.py` | SNOMED CT — allergy → drug class hierarchies (8 classes) |
| `umls_mapper.py` | UMLS Metathesaurus — ICD-10 ↔ SNOMED CT ↔ LOINC ↔ RxNorm |
| `consensus_engine.py` | 6-model US-based LLM consensus with parallel execution |
| `fda_client.py` | openFDA API — adverse events, label warnings, recalls |
| `trials_client.py` | ClinicalTrials.gov API v2 — trial matching |
| `what_if.py` | Medication change simulation — add/remove/swap |
| `phi_detector.py` | PHI detection — SSN, MRN, phone, email, DOB, addresses |
| `hallucination_detector.py` | Claim extraction + evidence grounding verification |

---

## SHARP-on-MCP Headers

All MCP tools accept optional FHIR context via SHARP-on-MCP headers:
```
X-FHIR-Server-URL: https://fhir.example.com/r4
X-FHIR-Access-Token: Bearer <token>
X-Patient-ID: patient-123
```
Or as tool parameters: `fhir_server_url`, `fhir_access_token`.

---

## Common Patterns

### Ingest + Safety Check
```python
# 1. Ingest patient data
result = ingest_patient_data(
    patient_id="patient-sarah-mitchell",
    fhir_server_url="https://fhir.example.com/r4",
    fhir_access_token="Bearer abc123",
)

# 2. Run safety check
conflicts = check_medication_conflicts(patient_id="patient-sarah-mitchell")

# 3. Explain critical finding
explanation = explain_clinical_conflict(patient_id="patient-sarah-mitchell", conflict_index=0)
```

### What-If Simulation
```python
# Simulate swapping Ibuprofen for Acetaminophen
result = what_if_medication_change(
    patient_id="patient-sarah-mitchell",
    action="swap",
    medication="Ibuprofen",
    swap_to="Acetaminophen",
)
# result.safe_to_proceed = True, result.risk_delta = -1
```

### Hallucination Check
```python
# Verify LLM-generated text is grounded in evidence
result = verify_clinical_claims(
    text="Patient is on Metformin 1000mg for Type 2 Diabetes",
    patient_id="patient-sarah-mitchell",
)
# result.grounding_score = 1.0 (fully grounded)
```

### PHI Scan Before Storage
```python
# Always scan before storing clinical notes
phi = scan_for_phi(text="Patient SSN 123-45-6789, DOB 03/14/1959")
if not phi["is_safe"]:
    # Use phi["redacted_text"] instead
    store_clinical_observation(
        patient_id="patient-sarah-mitchell",
        observation_type="clinical_note",
        title="Visit Note",
        content=phi["redacted_text"],
    )
```

### Multi-LLM Consensus
```python
# Verify critical finding across 6 models
result = consensus_verify_finding(
    finding="Warfarin + Ibuprofen combination poses serious bleeding risk",
    patient_id="patient-sarah-mitchell",
)
# result.consensus_level = "strong", result.agreement = "6/6"
```

---

## NEVER Do This

| Wrong | Correct |
|-------|---------|
| `check_drug_interactions(...)` | `check_medication_conflicts(...)` |
| `recall(query=...)` | `recall_patient_context(patient_id=..., query=...)` |
| `scan_phi(...)` | `scan_for_phi(text=...)` |
| `verify_claims(...)` | `verify_clinical_claims(text=..., patient_id=...)` |
| `what_if(...)` | `what_if_medication_change(patient_id=..., action=..., medication=...)` |
| `fda_alerts(...)` | `check_fda_safety_alerts(patient_id=...)` |
| `find_trials(...)` | `find_matching_trials(patient_id=...)` |
| `consensus(...)` | `consensus_verify_finding(finding=..., patient_id=...)` |

---

*Generated from ClinicalMem v4.0.0 source — STARGA Inc.*
