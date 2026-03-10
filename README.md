<p align="center">
  <img src="docs/logo.svg" alt="ClinicalMem Logo" width="180">
</p>

<h1 align="center">ClinicalMem</h1>

<p align="center">
  <strong>Persistent, Auditable, Contradiction-Safe Clinical Memory for Healthcare AI Agents</strong>
</p>

<p align="center">
  <a href="#tests"><img src="https://img.shields.io/badge/tests-356%20passed%20·%2080%25%20coverage-brightgreen?style=flat-square" alt="Tests"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg?style=flat-square" alt="License"></a>
  <a href="https://python.org"><img src="https://img.shields.io/badge/python-3.12+-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python"></a>
  <a href="#mcp-server"><img src="https://img.shields.io/badge/protocol-MCP-7C3AED?style=flat-square" alt="MCP"></a>
  <a href="#a2a-agent"><img src="https://img.shields.io/badge/protocol-A2A-F97316?style=flat-square" alt="A2A"></a>
  <a href="https://github.com/star-ga/mind-mem"><img src="https://img.shields.io/badge/powered%20by-mind--mem-00A67E?style=flat-square" alt="mind-mem"></a>
  <a href="#six-layer-safety-pipeline"><img src="https://img.shields.io/badge/safety-6%20layers-DC2626?style=flat-square" alt="Safety"></a>
  <a href="https://app.promptopinion.ai/marketplace"><img src="https://img.shields.io/badge/Prompt%20Opinion-Marketplace-FF6B35?style=flat-square" alt="Marketplace"></a>
</p>

<p align="center">
  Built for the <a href="https://agents-assemble.devpost.com/">Agents Assemble Healthcare AI Hackathon</a> by <a href="https://star.ga">STARGA Inc.</a>
</p>

<p align="center">
  <a href="https://clinicalmem-demo.pages.dev/demo"><strong>Live Demo Dashboard</strong></a> &nbsp;|&nbsp;
  <a href="https://youtu.be/iDKfbA89AnI"><strong>Video Demo</strong></a> &nbsp;|&nbsp;
  <a href="https://devpost.com/software/clinimalmem"><strong>DevPost</strong></a>
</p>

---

## The Problem

Sarah Mitchell is 67 years old. She has diabetes, hypertension, kidney disease, and atrial fibrillation &mdash; managed by **four different doctors** who don't talk to each other.

Last week, her ER doctor prescribed ibuprofen for knee pain. He didn't know she's on warfarin. **That combination can cause fatal bleeding.**

Today's clinical AI agents hallucinate. They forget patient history between conversations. They can't detect when a new prescription conflicts with an existing allergy. And there's no cryptographic proof of what they recommended or why.

## What ClinicalMem Does

ClinicalMem is a **deterministic safety layer** that anchors GenAI reasoning to verified clinical evidence:

| Capability | Description |
|-----------|-------------|
| **Drug Interaction Detection** | Four-tier pipeline: deterministic table &rarr; OpenEvidence API &rarr; RxNorm API (drug normalization) &rarr; Six-model LLM consensus |
| **Six-Model LLM Consensus** | GPT-5.4, Gemini 3.1 Pro, Gemini 3.1 Flash Lite, Grok 4.1, Claude Opus 4.6, Perplexity Sonar &mdash; all US-based |
| **Allergy Cross-Reaction Alerts** | SNOMED CT drug class hierarchy with 8 classes + alias expansion + cross-class detection |
| **UMLS Cross-Vocabulary Mapping** | ICD-10 &harr; SNOMED CT &harr; LOINC &harr; RxNorm via UMLS Metathesaurus |
| **What-If Medication Simulation** | Preview safety outcomes of medication changes before making them |
| **FDA Safety Alert Integration** | Real-time openFDA queries for active recalls and adverse event signals |
| **Clinical Trial Matching** | ClinicalTrials.gov search with eligibility pre-screening |
| **PHI Detection Guard** | Catches Protected Health Information before it enters the memory layer |
| **Hallucination Detection** | Validates LLM-generated text against retrieved evidence blocks |
| **Cross-Provider Contradiction Detection** | Surfaces conflicting care plans, declining lab trends, and medication-lab contraindications |
| **Persistent Clinical Memory** | Ingests FHIR R4 data into searchable memory blocks (BM25 + vector + RRF fusion) |
| **LLM-Grounded Synthesis** | Medical LLM cascade generates patient-specific narratives with explicit evidence citations |
| **Safe Abstention** | When evidence is insufficient, the system refuses to generate a narrative |
| **SHA-256 Audit Trail** | Every clinical decision is logged in a tamper-proof Merkle chain |

## Demo: Sarah Mitchell

67-year-old female with 4 providers, 7 medications, and **4 conflicts** &mdash; all detected automatically:

| # | Finding | Severity | Detection |
|---|---------|----------|-----------|
| 1 | **Warfarin + Ibuprofen** &mdash; ER prescribed NSAID without checking anticoagulant | CRITICAL | Drug interaction pipeline |
| 2 | **Penicillin allergy + Amoxicillin** &mdash; Urgent Care prescribed without checking allergies | CRITICAL | Allergy cross-reaction |
| 3 | **Declining GFR + Metformin** &mdash; eGFR 45&rarr;38&rarr;32, approaching contraindication | HIGH | Lab-medication monitor |
| 4 | **Conflicting BP targets** &mdash; Cardiologist <130/80 vs Nephrologist <140/90 | HIGH | Provider contradiction |

Plus 2 bonus findings discovered autonomously:
- **INR 3.8** &mdash; above therapeutic range, correlated with ibuprofen addition
- **eGFR declining trajectory** &mdash; 13-point drop over 6 months

## Architecture

```
                    FHIR R4                      Medical APIs
                  Patient Data              NIH RxNorm, OpenEvidence
                      |                     openFDA, ClinicalTrials.gov
                      v                            |
              +-------+----------------------------+-------+
              |           SHARED ENGINE (engine/)           |
              |                                             |
              |   Clinical Memory    MIND Kernels           |
              |   (BM25 + Vector)    (Scoring)              |
              |                                             |
              |   Drug Interactions  LLM Synthesizer        |
              |   (4-tier pipeline)  (evidence-cited)       |
              |                                             |
              |   RxNorm Client      SNOMED CT Client       |
              |   (drug normalize)   (allergy hierarchy)    |
              |                                             |
              |   UMLS Mapper        Consensus Engine       |
              |   (cross-vocabulary) (6 US-based LLMs)      |
              |                                             |
              |   What-If Simulator  FDA Client             |
              |   (digital twin)     (safety alerts)        |
              |                                             |
              |   PHI Detector       Hallucination Detector |
              |   (HIPAA guard)      (evidence grounding)   |
              |                                             |
              |   Trials Client      Audit Trail            |
              |   (ClinicalTrials)   (SHA-256 Merkle Chain) |
              +--------+------------------+--------+--------+
                       |                  |        |
                       v                  v        v
                  MCP Server         A2A Agent   Audit Trail
                 (FastMCP 2.x)     (Google ADK)  (Hash Chain)
                  18 tools          13 tools     Tamper-proof
                  SHARP-on-MCP      A2A Protocol
```

## Six-Layer Safety Pipeline

ClinicalMem uses a six-layer architecture that makes AI safe for healthcare:

| Layer | Name | Type | Latency |
|-------|------|------|---------|
| 1 | **Deterministic Table** | Rule-based | < 1ms |
| 2 | **OpenEvidence API** | Mayo Clinic / Elsevier ClinicalKey AI | ~2s |
| 3 | **RxNorm API** | Drug normalization + NIH interaction DB (Epic/Cerner standard) | ~1s |
| 4 | **Multi-LLM Consensus** | 6 US-based models: GPT-5.4, Gemini 3.1 Pro, Gemini 3.1 Flash Lite, Grok 4.1, Claude Opus 4.6, Perplexity Sonar | ~3s |
| 5 | **LLM Synthesis** | Evidence-cited clinical explanations | ~3s |
| 6 | **Abstention Gate** | "I don't know" when evidence insufficient | 0ms |

> Green layers (1) never hallucinate. Purple layers (4-5) are LLM-powered. Red layer (6) is a safety gate.

## MCP Server (18 Tools)

| Tool | Purpose |
|------|---------|
| `store_clinical_observation` | Store clinical note for patient |
| `recall_patient_context` | Hybrid search over clinical memory |
| `check_medication_conflicts` | Drug interaction detection (4-tier) |
| `check_allergy_conflicts` | Allergy cross-reaction alerts |
| `get_treatment_dependencies` | Condition-medication dependency map |
| `get_clinical_audit_trail` | Tamper-proof audit chain |
| `summarize_patient_history` | Importance-scored patient overview |
| `detect_belief_drift` | Cross-provider contradiction detection |
| `ingest_patient_data` | Pull all FHIR data into memory |
| `explain_clinical_conflict` | **GenAI**: LLM-grounded conflict explanation with citations |
| `clinical_care_handoff` | **GenAI**: Evidence-grounded care handoff note |
| `what_if_medication_change` | **Digital Twin**: Simulate medication changes |
| `verify_clinical_claims` | **GenAI**: LLM-verified clinical assertions |
| `scan_for_phi` | **HIPAA**: Detect PHI in free text |
| `check_fda_safety_alerts` | **FDA**: Active recalls + adverse events |
| `find_matching_trials` | **Research**: ClinicalTrials.gov matching |
| `consensus_verify_finding` | **GenAI**: Six-model LLM consensus verification |
| `health_check` | Container orchestrator health probe |

## A2A Agent (13 Tools)

| Tool | Category | Description |
|------|----------|-------------|
| `get_patient_demographics` | FHIR | Patient demographic data |
| `get_active_medications` | FHIR | Current medication list |
| `get_active_conditions` | FHIR | Active diagnoses |
| `get_recent_observations` | FHIR | Recent lab results + vitals |
| `recall_clinical_context` | Memory | Hybrid search with confidence gating |
| `store_clinical_note` | Memory | Persist clinical observations |
| `medication_safety_review` | Safety | Drug interactions + allergy conflicts |
| `detect_record_contradictions` | Safety | Scan for conflicting records |
| `explain_clinical_conflict` | GenAI | LLM-grounded clinical rationale |
| `what_if_scenario` | Digital Twin | Simulate medication changes |
| `check_fda_alerts` | FDA | Safety alerts + adverse events |
| `find_clinical_trials` | Research | ClinicalTrials.gov matching |
| `consensus_verify` | GenAI | Six-model LLM consensus |

## Why ClinicalMem

### vs. Typical Healthcare AI

| Capability | ClinicalMem | Typical Healthcare AI |
|-----------|-------------|----------------------|
| **Drug interactions** | 4-tier: deterministic + OpenEvidence + RxNorm (drug normalization + NIH DB) + Six-model consensus | Hardcoded lookup table |
| **LLM verification** | 6 US-based models (GPT-5.4, Gemini 3.1 Pro, Grok 4.1, Claude Opus 4.6, Perplexity Sonar, Gemini Flash) | Single model, no fallback |
| **Terminology** | SNOMED CT + RxNorm + UMLS Metathesaurus (ICD-10 &harr; SNOMED &harr; LOINC &harr; RxNorm) | Single vocabulary |
| **Evidence sources** | Mayo Clinic, Elsevier, NIH/NLM (Epic/Cerner standard), openFDA, ClinicalTrials.gov | None |
| **What-if simulation** | Digital twin for medication change preview | None |
| **PHI protection** | Regex-based PHI detection at ingestion boundary | None |
| **Hallucination guard** | Evidence grounding validation for LLM output | None |
| **Audit trail** | SHA-256 Merkle hash chain (HIPAA-grade) | None |
| **When uncertain** | Safe abstention &mdash; refuses to guess | Hallucinates |
| **Protocol support** | Both MCP (18 tools) AND A2A (13 tools) | One or neither |
| **Test coverage** | 429 tests, 80% line coverage | Untested |
| **Deployment** | Azure Container Apps (live, zero cold-start) | Localhost only |

### vs. Commercial Clinical Decision Support

| Feature | ClinicalMem | Epic CDS | IBM Watson Health | Nuance DAX |
|---------|-------------|----------|-------------------|------------|
| **Multi-LLM consensus** | 6 US-based models | Single rules engine | Single model (discontinued) | Single model |
| **Open standards** | MCP + A2A + FHIR R4 | Proprietary | Proprietary | Proprietary |
| **Drug interaction tiers** | 4 (deterministic &rarr; API &rarr; RxNorm &rarr; LLM) | 1 (lookup table) | 1 (NLP) | N/A |
| **Audit trail** | SHA-256 Merkle chain | Database logs | Database logs | None |
| **Safe abstention** | Built-in abstention gate | N/A | N/A | N/A |
| **What-if simulation** | Digital twin | Limited | None | None |
| **Open source** | MIT license | No | No | No |
| **Hallucination detection** | Evidence grounding validation | N/A | N/A | N/A |

> **Note**: ClinicalMem is a research prototype, not a replacement for FDA-cleared commercial systems. This comparison highlights architectural differences in approach.

## Example API Usage

### MCP Tool Call: Check Medication Conflicts

```json
{
  "tool": "check_medication_conflicts",
  "arguments": {
    "patient_id": "sarah-mitchell-001",
    "medications": ["Warfarin 5mg", "Ibuprofen 400mg", "Metformin 1000mg", "Lisinopril 10mg"]
  }
}
```

**Response:**

```json
{
  "findings": [
    {
      "severity": "CRITICAL",
      "type": "drug_interaction",
      "pair": ["Warfarin", "Ibuprofen"],
      "detection_layer": "deterministic",
      "latency_ms": 0.3,
      "evidence": "NSAID + anticoagulant: increased bleeding risk (GI hemorrhage, intracranial bleeding)",
      "consensus": {"models_agreed": 6, "models_total": 6, "confidence": 1.0},
      "rxnorm_cuis": ["RxCUI:11289", "RxCUI:5640"],
      "sources": ["Deterministic table", "OpenEvidence (Mayo Clinic)", "NIH Drug Interaction API", "6/6 LLM consensus"]
    }
  ],
  "total_findings": 1,
  "layers_executed": ["deterministic", "openevidence", "rxnorm", "consensus"],
  "audit_hash": "sha256:a3f2e8c1d4..."
}
```

### MCP Tool Call: What-If Medication Change

```json
{
  "tool": "what_if_medication_change",
  "arguments": {
    "patient_id": "sarah-mitchell-001",
    "action": "substitute",
    "remove": "Ibuprofen 400mg",
    "add": "Acetaminophen 500mg"
  }
}
```

**Response:**

```json
{
  "original_findings": 4,
  "simulated_findings": 3,
  "resolved": [{"pair": ["Warfarin", "Ibuprofen"], "severity": "CRITICAL"}],
  "new_risks": [],
  "recommendation": "Substitution removes 1 CRITICAL finding with no new risks introduced",
  "audit_hash": "sha256:b7d1f9a2e5..."
}
```

## Quick Start

### Run Tests

```bash
pip install -e ".[dev]"
python -m pytest tests/ -v --cov=engine --cov=mcp_server --cov=a2a_agent
```

### Local Development

```bash
# MCP Server (port 8080)
python -m mcp_server

# A2A Agent (port 8001)
uvicorn a2a_agent.app:a2a_app --host 0.0.0.0 --port 8001
```

### Docker

```bash
docker compose up --build
```

### Live Endpoints

| Service | URL |
|---------|-----|
| MCP Server | `https://clinicalmem-mcp.thankfulpond-9c3fdc1e.eastus.azurecontainerapps.io` |
| A2A Agent | `https://clinicalmem-a2a.thankfulpond-9c3fdc1e.eastus.azurecontainerapps.io` |
| Demo Dashboard | [clinicalmem-demo.pages.dev](https://clinicalmem-demo.pages.dev/demo) |
| Marketplace | [Prompt Opinion Marketplace](https://app.promptopinion.ai/marketplace) |

## Project Structure

```
clinicalmem/
├── engine/                     # Shared core engine (13 modules)
│   ├── __init__.py
│   ├── clinical_memory.py      # mind-mem adapted for clinical data
│   ├── clinical_scoring.py     # MIND Lang scoring kernels
│   ├── consensus_engine.py     # 6-model US-based LLM consensus
│   ├── fda_client.py           # openFDA drug safety alerts
│   ├── fhir_client.py          # FHIR R4 client with SSRF protection
│   ├── hallucination_detector.py # Evidence grounding validation
│   ├── llm_synthesizer.py      # Medical LLM cascade + abstention
│   ├── phi_detector.py         # PHI detection (HIPAA)
│   ├── rxnorm_client.py        # RxNorm drug normalization + interactions
│   ├── snomed_client.py        # SNOMED CT allergy cross-reactivity
│   ├── trials_client.py        # ClinicalTrials.gov matching
│   ├── umls_mapper.py          # UMLS cross-vocabulary mapping
│   └── what_if.py              # What-if medication simulation
├── mcp_server/                 # MCP Server (FastMCP 2.x)
│   ├── __init__.py
│   ├── __main__.py
│   ├── server.py               # 18 SHARP-on-MCP tools
│   └── tools/
│       └── __init__.py
├── a2a_agent/                  # A2A Agent (Google ADK)
│   ├── __init__.py
│   ├── agent.py                # ADK agent with 13 tools
│   ├── app.py                  # Starlette + A2A middleware
│   └── tools/
│       ├── __init__.py
│       ├── fhir_tools.py       # FHIR data access tools
│       ├── memory_tools.py     # Memory recall tools
│       └── safety_tools.py     # Drug/allergy/FDA/trials/consensus
├── tests/
│   ├── fixtures/
│   │   └── sarah_mitchell_bundle.json  # Synthetic FHIR patient
│   ├── test_engine/
│   │   ├── test_clinical_scoring.py    # 43 unit tests
│   │   ├── test_consensus_engine.py    # 25 consensus engine tests
│   │   ├── test_fda_client.py          # 16 FDA client tests
│   │   ├── test_hallucination_detector.py # 26 hallucination tests
│   │   ├── test_integration.py         # 61 integration tests
│   │   ├── test_phi_detector.py        # 25 PHI detection tests
│   │   ├── test_rxnorm_client.py       #  8 RxNorm API tests
│   │   ├── test_snomed_client.py       # 12 SNOMED CT tests
│   │   ├── test_trials_client.py       # 16 trials client tests
│   │   ├── test_umls_mapper.py         #  6 UMLS mapper tests
│   │   └── test_what_if.py             # 13 what-if simulation tests
│   ├── test_mcp/
│   │   └── test_mcp_tools.py           # 58 MCP tool tests
│   └── test_a2a/
│       └── test_a2a_tools.py           # 47 A2A tool tests
├── docs/
│   ├── demo.html               # Interactive demo dashboard
│   └── index.html              # Redirect to demo
├── .github/workflows/
│   ├── test.yaml               # CI: Run tests on push (429 tests, 80% coverage)
│   ├── deploy-mcp-prod.yaml    # CD: Deploy MCP to Azure
│   ├── deploy-a2a-prod.yaml    # CD: Deploy A2A to Azure
│   ├── deploy-env.yaml         # Shared deployment config
│   └── pages.yaml              # Deploy demo to Cloudflare Pages
├── Dockerfile.mcp              # MCP Server container
├── Dockerfile.a2a              # A2A Agent container
├── docker-compose.yml          # Local development stack
├── pyproject.toml              # Python project config
├── LICENSE                     # MIT License
└── NOTICE.md                   # Attribution notice
```

## Tests

**429 tests** covering the full clinical safety pipeline:

```
tests/test_engine/test_clinical_scoring.py       — 43 tests (scoring kernels)
tests/test_engine/test_consensus_engine.py       — 25 tests (6-model LLM consensus)
tests/test_engine/test_fda_client.py             — 16 tests (openFDA safety alerts)
tests/test_engine/test_hallucination_detector.py — 26 tests (evidence grounding)
tests/test_engine/test_integration.py            — 61 tests (engine + FHIR + SSRF)
tests/test_engine/test_llm_synthesizer.py        — 42 tests (LLM cascade + templates + abstention)
tests/test_engine/test_phi_detector.py           — 25 tests (PHI detection)
tests/test_engine/test_rxnorm_client.py          —  8 tests (RxNorm drug normalization)
tests/test_engine/test_snomed_client.py          — 12 tests (SNOMED CT cross-reactivity)
tests/test_engine/test_snomed_extended.py        — 19 tests (Snowstorm API + UMLS + FHIR mapping)
tests/test_engine/test_trials_client.py          — 16 tests (ClinicalTrials.gov matching)
tests/test_engine/test_umls_mapper.py            —  6 tests (UMLS cross-vocabulary mapping)
tests/test_engine/test_what_if.py                — 13 tests (what-if medication simulation)
tests/test_mcp/test_mcp_tools.py                 — 58 tests (all 18 MCP tools)
tests/test_a2a/test_a2a_tools.py                 — 47 tests (A2A tools + FHIR helpers)
tests/test_a2a/test_a2a_app.py                   — 12 tests (agent card + middleware + FHIR hook)
```

Coverage includes:
- Clinical importance scoring (acute vs routine vs historical)
- Confidence calibration and abstention gating
- Adversarial negation detection ("ruled out", "NOT allergic")
- FHIR R4 resource ingestion and normalization
- Drug interaction detection (all 4 tiers)
- Six-model LLM consensus verification (all US-based)
- RxNorm drug normalization (exact + approximate matching)
- SNOMED CT allergy cross-reactivity (8 drug classes, alias expansion)
- UMLS Metathesaurus cross-vocabulary mapping (ICD-10 &harr; SNOMED &harr; RxNorm)
- Allergy cross-reaction matching (including penicillin &rarr; cephalosporin)
- Lab-medication contraindication detection
- Lab trend analysis (declining GFR trajectory)
- Cross-provider contradiction detection (BP targets)
- What-if medication simulation (add/remove/substitute)
- FDA safety alert integration (recalls + adverse events)
- Clinical trial matching with eligibility pre-screening
- PHI detection and redaction (SSN, MRN, phone, email, DOB, address)
- Hallucination detection (unsupported claims, fabricated citations)
- SHA-256 hash-chain audit trail integrity
- LLM synthesis with evidence citations
- Safe abstention when evidence is insufficient
- SSRF protection (RFC 1918, link-local, IPv6 private ranges)
- Rate limiting middleware

## Tech Stack

| Component | Technology |
|-----------|------------|
| **Engine** | [mind-mem](https://github.com/star-ga/mind-mem) hybrid search (BM25 + vector + RRF fusion) |
| **Scoring** | [MIND Lang](https://github.com/star-ga/mind) kernel patterns (confidence, importance, negation) |
| **Drug Interactions** | Deterministic table + [OpenEvidence](https://openevidence.com/) + [RxNorm](https://rxnav.nlm.nih.gov/) (drug normalization) + Six-model LLM consensus |
| **Terminology** | [SNOMED CT](https://www.snomed.org/) + [UMLS Metathesaurus](https://www.nlm.nih.gov/research/umls/) (ICD-10, LOINC, RxNorm crosswalk) |
| **LLM Consensus** | GPT-5.4, Gemini 3.1 Pro, Gemini 3.1 Flash Lite, Grok 4.1, Claude Opus 4.6, Perplexity Sonar (all US-based) |
| **FDA** | [openFDA](https://open.fda.gov/) drug safety alerts, recalls, adverse events |
| **Clinical Trials** | [ClinicalTrials.gov](https://clinicaltrials.gov/) API v2 |
| **MCP** | [FastMCP 2.x](https://github.com/jlowin/fastmcp) with SHARP-on-MCP headers |
| **A2A** | [Google ADK](https://github.com/google/adk-python) with A2A protocol |
| **FHIR** | R4 client with synthetic patient data |
| **Deploy** | Docker + Azure Container Apps |
| **Demo** | Cloudflare Pages |
| **CI/CD** | GitHub Actions (test + deploy) |

## Performance & Cost

| Metric | Value |
|--------|-------|
| **Deterministic layer latency** | < 1ms (rule-based, zero API calls) |
| **RxNorm + OpenEvidence** | ~2-3s (parallel NIH/evidence API calls) |
| **Six-model LLM consensus** | ~3-5s (all 6 models queried in parallel) |
| **End-to-end safety check** | ~5-8s total (all 6 layers) |
| **Test suite execution** | ~4s for 429 tests |
| **Code coverage** | 80% line coverage across engine, MCP server, and A2A agent |

### Cost Analysis (per patient safety check)

| Component | Estimated Cost | Notes |
|-----------|---------------|-------|
| Deterministic table | $0.00 | Local lookup, no API |
| OpenEvidence API | $0.00 | Free tier |
| RxNorm / NIH API | $0.00 | Free public API, no key required |
| openFDA API | $0.00 | Free public API |
| ClinicalTrials.gov | $0.00 | Free public API |
| Six-model LLM consensus | ~$0.02-0.05 | Varies by token count; parallel execution |
| **Total per check** | **~$0.02-0.05** | Dominated by LLM inference cost |

> **Production optimization**: For high-volume deployment, the consensus layer can be reduced to 3 models (majority vote with fewer models) or cached for repeated drug pairs, reducing cost by 50-80%.

## Consensus Mechanism

The six-model consensus engine uses **structured majority voting**:

1. All 6 US-based LLMs receive the same structured prompt with drug pair, patient context, and evidence from layers 1-3
2. Each model returns a severity classification (CRITICAL / HIGH / MODERATE / LOW / NONE) and a confidence score
3. **Majority rule**: A finding is confirmed if &ge;4/6 models agree on severity &ge; MODERATE
4. **Split decision handling**: If exactly 3/6 agree, the finding is flagged as UNCERTAIN and escalated to the synthesis layer with a lower confidence score
5. **Unanimous disagreement**: If no majority exists, the abstention gate fires &mdash; the system reports "insufficient consensus" rather than guessing
6. **Model-specific bias mitigation**: Each model's response is weighted equally; no single model can override the consensus

| Scenario | Models Agreeing | Result |
|----------|----------------|--------|
| 6/6 agree CRITICAL | Unanimous | Confirmed CRITICAL (confidence: 1.0) |
| 4/6 agree HIGH | Supermajority | Confirmed HIGH (confidence: 0.67) |
| 3/6 agree, 3/6 disagree | Split | Flagged UNCERTAIN (confidence: 0.50) |
| No majority | Fragmented | Abstention &mdash; "insufficient consensus" |

## Failure Modes & Resilience

ClinicalMem is designed to **degrade gracefully** when external services are unavailable:

| Failure | Impact | Fallback |
|---------|--------|----------|
| OpenEvidence API down | Layer 2 skipped | Layers 1, 3, 4 still active; deterministic + RxNorm + consensus |
| RxNorm API down | Layer 3 skipped | Layers 1, 2, 4 still active; deterministic + OpenEvidence + consensus |
| openFDA API down | No FDA alerts | Safety check continues; FDA alerts are supplementary |
| ClinicalTrials.gov down | No trial matching | Safety check continues; trial matching is supplementary |
| 1-2 LLMs timeout | Reduced consensus pool | Remaining models vote; minimum 4 required for confirmation |
| 3+ LLMs timeout | Consensus degraded | Abstention gate fires; system reports "insufficient models available" |
| All LLMs down | Layers 4-5 unavailable | **Layers 1-3 still fully operational** (deterministic + APIs) |
| FHIR server down | No patient ingestion | Cached patient data in memory still queryable |

> **Key design principle**: The deterministic layer (Layer 1) and NIH APIs (Layer 3) have **zero LLM dependency**. Even if all 6 LLMs are completely unavailable, ClinicalMem still catches known drug interactions and normalizes medications via RxNorm.

### Retry & Timeout Policy

- All external API calls have a **5-second timeout** with 2 retries
- LLM providers have a **10-second timeout** per model
- Consensus engine runs all 6 models in **parallel** (not sequential)
- Circuit breaker: After 3 consecutive failures, an API is marked degraded for 60 seconds

## Clinical Validation Roadmap

ClinicalMem is currently validated against **synthetic patient data** (Sarah Mitchell). The following roadmap outlines the path to clinical deployment:

### Current State (Hackathon)

- &check; 356 automated tests including adversarial cases (negation detection, boundary conditions)
- &check; SSRF protection validated against RFC 1918, link-local, and IPv6 private ranges
- &check; PHI detection tested against 25 patterns (SSN, MRN, phone, email, DOB, address)
- &check; Hallucination detection validated against fabricated citations and unsupported claims
- &check; Drug interaction detection verified against known pairs (warfarin+NSAID, metformin+CKD)

### Next Steps (Post-Hackathon)

1. **Gold-standard benchmarking** &mdash; Validate drug interaction detection against the [ONC High-Priority Drug-Drug Interactions](https://www.healthit.gov/topic/safety/clinical-decision-support) list and [ISMP High-Alert Medications](https://www.ismp.org/recommendations/high-alert-medications-acute-list)
2. **Sensitivity/specificity measurement** &mdash; Calculate true positive, false positive, true negative, and false negative rates against a curated clinical dataset
3. **Clinician adjudication** &mdash; Partner with clinical pharmacists to review flagged findings and measure precision (% of flags that are clinically meaningful)
4. **De-identified dataset validation** &mdash; Test against [MIMIC-IV](https://physionet.org/content/mimiciv/) or [eICU](https://physionet.org/content/eicu-crd/) de-identified clinical data
5. **Human-in-the-loop boundaries** &mdash; ClinicalMem is designed as a **clinical decision support** tool, not an autonomous prescriber. All findings require clinician review before action

### Regulatory Pathway

| Framework | Status | Notes |
|-----------|--------|-------|
| **FDA SaMD** | Planned | Clinical Decision Support software &mdash; Class II under [21 CFR 820](https://www.accessdata.fda.gov/scripts/cdrh/cfdocs/cfcfr/CFRSearch.cfm?CFRPart=820) |
| **HIPAA** | Partial | PHI detection at ingestion boundary; full BAA required for production PHI |
| **SOC 2 Type II** | Planned | SHA-256 audit trail provides cryptographic foundation |
| **ONC Certification** | Planned | FHIR R4 + USCDI alignment for EHR interoperability |
| **CE Marking (EU MDR)** | Future | Required for European clinical deployment |

> **Important**: ClinicalMem is currently a **research prototype** demonstrated with synthetic data. It is not FDA-cleared and should not be used for clinical decisions without appropriate validation and regulatory approval.

## Limitations & Known Risks

| Limitation | Description | Mitigation |
|-----------|-------------|------------|
| **Synthetic data only** | All validation uses fictional patient Sarah Mitchell | Roadmap includes MIMIC-IV and clinician adjudication |
| **Deterministic table coverage** | Rule-based layer covers common pairs, not all ~10,000 known interactions | Layers 2-4 catch pairs not in the deterministic table |
| **LLM consensus cost** | Six-model inference adds ~$0.02-0.05 per check | Cacheable for repeated drug pairs; reducible to 3 models |
| **External API dependency** | RxNorm, OpenEvidence, openFDA may have downtime | Graceful degradation; deterministic layer always available |
| **English only** | Clinical NLP and LLM prompts are English-only | Multilingual support planned |
| **No EHR write-back** | Read-only FHIR integration; cannot modify EHR records | By design &mdash; ClinicalMem is advisory, not prescriptive |
| **PHI detection is pattern-based** | Regex patterns may miss novel PHI formats | Supplementary to institutional de-identification pipelines |

## Data Safety

- All patient data is **synthetic** (no PHI)
- Sarah Mitchell is a fictional patient created for demonstration
- FHIR resources are generated test fixtures
- No real clinical data is stored or processed
- ClinicalMem is a **research prototype** &mdash; not for clinical use without validation

## License

MIT &mdash; see [LICENSE](LICENSE)

Built on [mind-mem](https://github.com/star-ga/mind-mem) (Apache 2.0) by [STARGA Inc.](https://star.ga)
