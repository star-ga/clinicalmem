# DevPost Submission — ClinicalMem

> Use this document to fill in the DevPost submission form fields.
> Submit for BOTH tracks: MCP Server + A2A Agent.

---

## Project Name

ClinicalMem

## Tagline (short description)

Persistent, auditable, contradiction-safe clinical memory for healthcare AI agents — powered by mind-mem and MIND Lang scoring kernels.

## About (What it does)

ClinicalMem gives healthcare AI agents something they desperately need: **memory that understands medicine**.

Today's clinical AI can answer questions about a patient — but it forgets everything between conversations. It can't detect when a new prescription conflicts with an existing allergy. It can't notice that two specialists are giving contradictory guidance. It can't audit why it made a recommendation six months ago.

ClinicalMem fixes this by providing:

- **Persistent Clinical Memory** — Ingests FHIR R4 patient data (medications, conditions, allergies, observations) and stores it as searchable memory blocks using hybrid BM25 + vector retrieval
- **Drug Interaction Detection** — Automatically flags dangerous medication combinations (e.g., Warfarin + NSAIDs = bleeding risk) using a clinical knowledge base of 12 interaction pairs
- **Allergy Cross-Reaction Alerts** — Catches prescriptions that cross-react with known allergies (e.g., Penicillin allergy + Amoxicillin prescription = anaphylaxis risk) across 4 cross-reaction groups
- **Confidence Gating (Abstention)** — When evidence is insufficient, ClinicalMem abstains rather than guessing. In healthcare, "I don't know" is better than a wrong answer
- **Contradiction Detection** — Identifies conflicts between providers, medications, and clinical guidelines
- **SHA-256 Hash-Chain Audit Trail** — Every clinical decision is logged in a tamper-proof chain, providing HIPAA-grade accountability

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

- **Shared Engine** (`engine/`) — FHIR R4 client, clinical memory, scoring kernels
- **MCP Server** (`mcp_server/`) — FastMCP 2.x with Streamable HTTP transport, 9 SHARP-on-MCP tools
- **A2A Agent** (`a2a_agent/`) — Google ADK agent with 4 clinical skills, AgentCard, API key middleware

### Tech Stack

- Python 3.12
- mind-mem (hybrid BM25 + vector search engine)
- FastMCP 2.x (MCP server framework)
- Google ADK + a2a-sdk (A2A agent framework)
- httpx (async FHIR R4 client)
- FHIR R4 (HL7 healthcare data standard)
- Docker + Google Cloud Run (deployment)

### Testing

61 tests covering:
- Clinical scoring kernels (confidence gating, drug interactions, allergy conflicts, negation detection)
- FHIR client integration with mock server
- Engine ingestion, recall, safety checks, contradiction detection
- Hash-chain audit trail integrity verification

## Challenges we ran into

1. **Confidence calibration** — The initial confidence formula was a self-referential ratio that always returned ~1.0. We redesigned it to use absolute evidence strength, so low BM25 scores actually trigger abstention.

2. **Cross-provider contradiction detection** — Different specialists legitimately disagree (e.g., BP targets for cardiac vs renal patients). We had to distinguish harmful contradictions from acceptable clinical nuance.

3. **FHIR data normalization** — Medications come in many formats (brand names, generics, coded concepts). Our fuzzy matching uses substring containment rather than exact match to catch variations.

4. **Negation boundary conditions** — "ruled out" at end of string vs mid-sentence. Clinical NLP must handle terminal negation markers without trailing spaces.

## Accomplishments that we're proud of

- **Zero infrastructure** — ClinicalMem runs as a single Python process. No vector database, no graph database, no external LLM for extraction. Pure BM25 + local scoring.
- **4 conflicts caught in demo** — Drug interactions, allergy cross-reactions, declining lab trends, and provider disagreements — all detected automatically.
- **Tamper-proof audit trail** — SHA-256 hash chain provides cryptographic proof that clinical decision logs haven't been altered.
- **61 tests passing** — Comprehensive test coverage including integration tests with a synthetic FHIR patient bundle.

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
- fhir
- fastmcp
- google-adk
- a2a-protocol
- docker
- google-cloud-run

## Try it out

- GitHub: https://github.com/star-ga/clinicalmem
- MCP Server: [Cloud Run URL after deployment]
- A2A Agent: [Cloud Run URL after deployment]
