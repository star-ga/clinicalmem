# DevPost Submission — ClinicalMem

> Use this document to fill in the DevPost submission form fields.
> Submit for BOTH tracks: MCP Server + A2A Agent.

---

## Project Name

ClinicalMem

## Tagline (short description)

The deterministic memory layer that prevents LLM hallucinations in clinical care — with auditable evidence, safe abstention, and cross-provider contradiction detection.

## About (What it does)

ClinicalMem solves the biggest barrier to AI in healthcare: **trust**.

Today's clinical AI agents hallucinate. They forget patient history between conversations. They can't detect when a new prescription conflicts with an existing allergy. They can't notice that two specialists are giving contradictory guidance. And there's no cryptographic proof of what they recommended or why.

ClinicalMem is a **deterministic safety layer** that anchors GenAI reasoning to verified clinical evidence:

- **Persistent Clinical Memory** — Ingests FHIR R4 patient data and stores it as searchable memory blocks using mind-mem's hybrid BM25 + vector + RRF fusion retrieval
- **LLM-Grounded Clinical Synthesis** — Uses GenAI to generate patient-specific clinical narratives, but only from retrieved evidence with explicit citations. When confidence is low, the system **abstains rather than hallucinating**
- **Four-Tier Drug Interaction Detection** — (1) Deterministic table catches known pairs in microseconds, (2) OpenEvidence API (Mayo Clinic / Elsevier ClinicalKey AI) for clinically authoritative evidence-grounded detection, (3) NIH/NLM Drug Interaction API (RxNorm — the same federal database used by Epic, Cerner, and all certified EHRs) for free, authoritative coverage, (4) Gemini LLM fallback for remaining pairs. Each tier is audited
- **Allergy Cross-Reaction Alerts** — Catches prescriptions that cross-react with known allergies (Penicillin allergy + Amoxicillin = anaphylaxis risk)
- **Cross-Provider Contradiction Detection** — Surfaces conflicting care plans, declining lab trends, and medication-lab contraindications across fragmented provider records
- **SHA-256 Hash-Chain Audit Trail** — Every clinical decision is logged in a tamper-proof Merkle chain, providing cryptographic proof that AI recommendations haven't been altered

### Demo Scenario: Sarah Mitchell

Sarah is a 67-year-old with Type 2 Diabetes, Hypertension, CKD Stage 3b, and Atrial Fibrillation, managed by 4 different providers. ClinicalMem catches:

1. **NSAID + Warfarin** — ER doctor prescribed Ibuprofen for knee pain, but Sarah is on Warfarin for AFib. ClinicalMem flags the serious bleeding risk.
2. **Amoxicillin + Penicillin Allergy** — Urgent Care prescribed Amoxicillin without checking allergies. ClinicalMem catches the cross-reactivity with her documented Penicillin anaphylaxis.
3. **Declining GFR + Metformin** — GFR dropping from 45 to 32 over 6 months, approaching the threshold where Metformin becomes contraindicated.
4. **Conflicting BP Targets** — Cardiologist says <130/80, nephrologist says <140/90. ClinicalMem flags the belief drift.

## How we built it

ClinicalMem is built on two open-source STARGA technologies:

1. **mind-mem** (v1.9.0) — Our persistent memory system for coding agents, adapted for clinical data. Provides hybrid BM25 + vector search, contradiction detection, causal dependency graphs, and hash-chain audit logging.

2. **MIND Lang Scoring Kernels** — Pure Python implementations of three MIND Lang kernels:
   - `abstention.mind` → Confidence gating (decides when to abstain)
   - `importance.mind` → Clinical importance scoring (prioritizes acute over historical)
   - `adversarial.mind` → Negation detection (distinguishes "allergic" from "NOT allergic")

### Architecture

- **Shared Engine** (`engine/`) — FHIR R4 client, clinical memory, scoring kernels, LLM synthesizer
- **MCP Server** (`mcp_server/`) — FastMCP 2.x with Streamable HTTP transport, 11 SHARP-on-MCP tools
- **A2A Agent** (`a2a_agent/`) — Google ADK agent with 5 clinical skills, AgentCard, API key middleware

### The GenAI Pattern: Deterministic Detection + LLM Synthesis

ClinicalMem uses a two-layer architecture that makes AI safe for healthcare:

1. **Detection Layer 1** (Deterministic) — Rule-based safety rails catch known drug interactions, allergy conflicts, lab contraindications, and provider disagreements. Microsecond response, never hallucinate.

2. **Detection Layer 2** (OpenEvidence API) — For medication pairs not in the deterministic table, ClinicalMem queries OpenEvidence — the same medical AI engine powering Elsevier's ClinicalKey AI, developed with Mayo Clinic. Returns evidence-grounded answers with peer-reviewed citations.

3. **Detection Layer 3** (NIH/NLM Drug Interaction API) — Queries RxNorm to resolve drug names to RxCUI identifiers, then checks the NIH Drug Interaction API — the same federal database used by Epic, Cerner, and all certified EHR systems. Free, no API key required, authoritative.

4. **Detection Layer 4** (Gemini LLM) — General-purpose LLM fallback for any remaining uncovered pairs. Structured JSON extraction with severity filtering.

4. **Synthesis Layer** (GenAI) — An LLM generates patient-specific clinical explanations from detected findings, citing evidence blocks by ID. The LLM never invents facts — it explains what the detection layers found.

5. **Abstention Gate** — When evidence is insufficient, the system refuses to generate a narrative. In healthcare, "I don't know" saves lives.

### Tech Stack

- Python 3.12
- mind-mem (hybrid BM25 + vector search engine)
- NIH/NLM Drug Interaction API (RxNorm — federal gold standard, used by Epic/Cerner)
- OpenEvidence API (clinically authoritative medical AI, Mayo Clinic / Elsevier)
- FastMCP 2.x (MCP server framework)
- Google ADK + a2a-sdk (A2A agent framework)
- httpx (async FHIR R4 client)
- FHIR R4 (HL7 healthcare data standard)
- Docker + Azure Container Apps (deployment)

### Testing

90 tests covering:
- Clinical scoring kernels (confidence gating, drug interactions, allergy conflicts, negation detection)
- FHIR client integration with mock server
- Engine ingestion, recall, safety checks, contradiction detection
- Hash-chain audit trail integrity verification
- LLM synthesis (conflict explanation, care handoff, abstention gate, immutability)

## Challenges we ran into

1. **Confidence calibration** — The initial confidence formula was a self-referential ratio that always returned ~1.0. We redesigned it to use absolute evidence strength, so low BM25 scores actually trigger abstention.

2. **Cross-provider contradiction detection** — Different specialists legitimately disagree (e.g., BP targets for cardiac vs renal patients). We had to distinguish harmful contradictions from acceptable clinical nuance.

3. **FHIR data normalization** — Medications come in many formats (brand names, generics, coded concepts). Our fuzzy matching uses substring containment rather than exact match to catch variations.

4. **Negation boundary conditions** — "ruled out" at end of string vs mid-sentence. Clinical NLP must handle terminal negation markers without trailing spaces.

## Accomplishments that we're proud of

- **Deterministic safety + GenAI synthesis** — Rules catch the conflicts reliably; LLMs explain them in clinical context. Best of both worlds.
- **4 conflicts caught in demo** — Drug interactions, allergy cross-reactions, declining lab trends, and provider disagreements — all detected automatically with patient-specific clinical rationale.
- **Safe abstention** — When evidence is insufficient, ClinicalMem says "I don't know" instead of hallucinating. In healthcare, this saves lives.
- **Tamper-proof audit trail** — SHA-256 Merkle chain provides cryptographic proof that clinical decision logs haven't been altered.
- **90 tests passing** — Comprehensive test coverage including integration tests with a synthetic FHIR patient bundle.

## What we learned

- MIND Lang's abstention kernel philosophy (refuse to answer when uncertain) is critical for clinical AI safety
- FHIR R4's bundled resource model maps naturally to mind-mem's block-based memory architecture
- Healthcare AI needs memory that is auditable, contradiction-aware, and confidence-gated — not just retrieval

## What's next for ClinicalMem

- Integration with RxNorm API for comprehensive drug interaction checking
- Temporal trend analysis (detect worsening lab values over time)
- Multi-patient analytics (population-level safety signals)
- Full mind-mem vector search with medical embeddings (BiomedBERT)
- Production deployment with HIPAA-compliant encryption at rest

## Built With

- python
- mind-mem
- mind-lang
- nih-rxnorm
- openevidence
- fhir
- fastmcp
- google-adk
- a2a-protocol
- docker
- azure-container-apps

## Try it out

- GitHub: https://github.com/star-ga/clinicalmem
- MCP Server: https://clinicalmem-mcp.thankfulpond-9c3fdc1e.eastus.azurecontainerapps.io
- A2A Agent: https://clinicalmem-a2a.thankfulpond-9c3fdc1e.eastus.azurecontainerapps.io
