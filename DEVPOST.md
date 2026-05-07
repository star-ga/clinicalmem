## Inspiration

Sarah Mitchell is 67 years old. She has diabetes, hypertension, kidney disease, and atrial fibrillation — managed by four different doctors who don't talk to each other. Last week, her ER doctor prescribed ibuprofen for knee pain. He didn't know she's on warfarin. That combination can cause fatal bleeding.

Today's clinical AI agents hallucinate. They forget patient history between conversations. They can't detect when a new prescription conflicts with an existing allergy. And there's no cryptographic proof of what they recommended or why.

## What it does

ClinicalMem is a **deterministic safety layer** that anchors GenAI reasoning to verified clinical evidence:

- **Persistent Clinical Memory** — Ingests FHIR R4 patient data and stores it as searchable memory blocks using mind-mem's hybrid BM25 + vector + RRF fusion retrieval
- **Six-Model US-Based LLM Consensus** — Drug interaction findings are verified by 6 US-based LLMs in parallel: OpenAI GPT-5.5, Google Gemini 3.1 Pro, xAI Grok 4.3, Anthropic Claude Opus 4.7, Perplexity Sonar Pro (`sonar-reasoning-pro`), and NVIDIA Nemotron Ultra 253B (`nvidia/llama-3.1-nemotron-ultra-253b-v1` via NIM at `integrate.api.nvidia.com`). **All providers are US-headquartered**, ensuring HIPAA-compatible data residency end-to-end. Majority consensus required for high-confidence clinical verdicts; cascade scales gracefully to whatever API keys are present (1-6 providers).
- **Six-Tier Drug Interaction Detection** — (1) Deterministic table catches known pairs in microseconds, (2) OpenEvidence API (Mayo Clinic / Elsevier ClinicalKey AI) for clinically authoritative evidence-grounded detection, (3) RxNorm REST API — resolves drug names to RxCUI identifiers, normalizes medication lists, and checks pairwise interactions via the NIH Drug Interaction API (the same federal database used by Epic, Cerner, and all certified EHRs), (4) Six-model US-based LLM consensus for remaining pairs with majority voting, **(4.5) BitNet b1.58 reproducibility primitive — *not* the primary classifier (primary recall comes from layers 1-4). Layer 4.5 is a high-precision deterministic veto + audit-replay anchor: 100% recall + 0 FP on the safety-critical `contraindicated` class (44/44 + 0 FP under v8 Q16.16, iter-275 promotion), pure-integer Q16.16 forward pass with bit-identical output across every architecture (ARM, x86_64, CUDA, NPU). The FDA SaMD reproducibility primitive — every classification carries a SHA-256 `repro_hash` any auditor can re-verify in `<1 ms`.**
- **SNOMED CT Allergy Cross-Reactivity** — 8 drug class hierarchies (penicillin, cephalosporin, sulfonamide, fluoroquinolone, opioid, NSAID, ACE inhibitor, statin) with alias expansion. Catches prescriptions that cross-react with known allergies
- **UMLS Metathesaurus Crosswalk** — Maps between ICD-10, SNOMED CT, LOINC, and RxNorm vocabularies using CUI-based resolution
- **What-If Medication Simulation** — Clinicians can simulate adding, removing, or substituting medications to preview safety outcomes before making changes. Runs the full six-layer pipeline on hypothetical medication lists
- **FDA Safety Alert Integration** — Queries the openFDA Drug Enforcement and Drug Event APIs for active recalls, safety alerts, and adverse event signals for patient medications
- **Clinical Trial Matching** — Searches ClinicalTrials.gov for active trials matching patient conditions, with eligibility pre-screening based on age, gender, and active conditions
- **PHI Detection Guard** — Scans free-text clinical notes for Protected Health Information (SSN, MRN, phone, email, dates of birth, addresses) before storage, preventing accidental PHI leakage into the memory layer
- **Hallucination Detection** — Validates LLM-generated clinical text against retrieved evidence blocks. Flags unsupported claims, checks medication name accuracy, and detects fabricated citations
- **Cross-Provider Contradiction Detection** — Surfaces conflicting care plans, declining lab trends, and medication-lab contraindications across fragmented provider records
- **LLM-Grounded Clinical Synthesis** — Medical LLM cascade generates patient-specific clinical narratives with explicit evidence citations. When confidence is low, the system **abstains rather than hallucinating**
- **SHA-256 Hash-Chain Audit Trail** — Every clinical decision is logged in a tamper-proof Merkle chain, providing cryptographic proof that AI recommendations haven't been altered

### Demo Scenario: Sarah Mitchell

Sarah is a 67-year-old with Type 2 Diabetes, Hypertension, CKD Stage 3b, and Atrial Fibrillation, managed by 4 different providers. ClinicalMem catches:

1. **NSAID + Warfarin** — ER doctor prescribed Ibuprofen for knee pain, but Sarah is on Warfarin for AFib. ClinicalMem flags the serious bleeding risk.
2. **Amoxicillin + Penicillin Allergy** — Urgent Care prescribed Amoxicillin without checking allergies. SNOMED CT drug class hierarchy detects the cross-reactivity with her documented Penicillin anaphylaxis.
3. **Declining GFR + Metformin** — GFR dropping from 45 to 32 over 6 months, approaching the threshold where Metformin becomes contraindicated.
4. **Conflicting BP Targets** — Cardiologist says <130/80, nephrologist says <140/90. ClinicalMem flags the belief drift.

## How we built it

ClinicalMem is built on two open-source STARGA technologies:

1. **mind-mem** (v3.x) — Our persistent memory system for AI agents, adapted for clinical data. Provides hybrid BM25 + vector + RRF search, contradiction detection, causal dependency graphs, **SQLCipher at-rest encryption (HIPAA-compatible)**, **TAG_v1 audit-integrity preimages with Q16.16 fixed-point scoring**, and **tier-decay (TTL/LRU) for clinical-memory ageing**. The audit chain is replayable against the canonical `mm verify-chain` reference verifier.

2. **MIND Lang Scoring Kernels** — Pure Python implementations of three MIND Lang kernels:
   - `abstention.mind` → Confidence gating (decides when to abstain)
   - `importance.mind` → Clinical importance scoring (prioritizes acute over historical)
   - `adversarial.mind` → Negation detection (distinguishes "allergic" from "NOT allergic")

### Architecture

- **Shared Engine** (`engine/`) — 14 modules: FHIR R4 client, clinical memory, scoring kernels, LLM synthesizer, RxNorm client, SNOMED CT client, UMLS Metathesaurus mapper, consensus engine (6 US-based LLMs), FDA client, clinical trials client, what-if simulator, PHI detector, hallucination detector, **BitNet b1.58 ternary classifier (Q16.16 fixed-point, bit-identical across architectures)**
- **MCP Server** (`mcp_server/`) — FastMCP 2.x with Streamable HTTP transport, 18 SHARP-on-MCP tools
- **A2A Agent** (`a2a_agent/`) — Google ADK agent with 13 clinical tools, AgentCard, API key middleware

### The GenAI Pattern: Deterministic Detection + LLM Consensus + Safe Synthesis

ClinicalMem uses a six-layer architecture that makes AI safe for healthcare:

1. **Detection Layer 1** (Deterministic) — Rule-based safety rails catch known drug interactions, allergy conflicts, lab contraindications, and provider disagreements. Microsecond response, never hallucinate.
2. **Detection Layer 2** (OpenEvidence API) — For medication pairs not in the deterministic table, ClinicalMem queries OpenEvidence — the same medical AI engine powering Elsevier's ClinicalKey AI, developed with Mayo Clinic.
3. **Detection Layer 3** (RxNorm + NIH Drug Interaction API) — Resolves drug names to RxCUI identifiers via the RxNorm REST API, normalizes medication lists, then checks pairwise interactions through the NIH Drug Interaction API — the same federal database used by Epic, Cerner, and all certified EHR systems. Free, no API key required, authoritative.
4. **Detection Layer 4** (Six-Model LLM Consensus) — Verifies findings across 6 US-based LLMs in parallel: OpenAI GPT-5.5 (`gpt-5.5`), Google Gemini 3.1 Pro (`gemini-3.1-pro-preview`), xAI Grok 4.3 (`grok-4.3`), Anthropic Claude Opus 4.7 (`claude-opus-4-7`), Perplexity Sonar Pro (`sonar-reasoning-pro`), and NVIDIA Nemotron Ultra 253B (`nvidia/llama-3.1-nemotron-ultra-253b-v1` via NIM). **All six providers are US-headquartered** (San Francisco / Mountain View / Palo Alto / Santa Clara), ensuring HIPAA-compatible data residency. Cascade scales gracefully when API keys are missing; majority consensus required for HIGH verdict, abstention triggers when only 0-1 providers respond.
4.5. **Detection Layer 4.5** (BitNet b1.58 Ternary Classifier) — A clean-room Python implementation of the BitNet b1.58 architecture (Ma et al., arXiv:2402.17764). Pure-integer Q16.16 fixed-point forward pass over ternary weights ∈ {-1, 0, +1} — no multiplication, only addition and subtraction. Output is **bit-identical across ARM, x86_64, CUDA, and NPU targets** — the reproducibility the FDA's 2024 SaMD guidance expects. Each classification carries a `repro_hash` (SHA-256 over the canonical encoding of feature_hash, logits_q16, severity, weights_id) that any auditor with `engine/bitnet_classifier.py` and the ~118 KB v8 ternary weights bundle (`1f0f8859…`) can re-verify in &lt; 1 ms per pair, no proprietary toolchain required.
5. **Synthesis Layer** (Medical LLM Cascade) — Generates patient-specific clinical explanations from detected findings, citing evidence blocks by ID. The LLM never invents facts — it explains what the detection layers found.
6. **Abstention Gate** — When evidence is insufficient, the system refuses to generate a narrative. In healthcare, "I don't know" saves lives.

### UMLS Terminology Services

ClinicalMem integrates three NIH terminology services for standardized clinical vocabulary:

- **RxNorm Client** — Resolves medication names to RxCUI identifiers, normalizes medication lists, and retrieves pairwise drug interactions from the NIH REST API
- **SNOMED CT Client** — Maps allergies to drug class hierarchies (8 classes: penicillin, cephalosporin, sulfonamide, fluoroquinolone, opioid, NSAID, ACE inhibitor, statin) with alias expansion
- **UMLS Metathesaurus Mapper** — Cross-vocabulary mapping between ICD-10, SNOMED CT, LOINC, and RxNorm using CUI-based concept resolution

### Testing

356 tests covering clinical scoring kernels, FHIR client integration, engine ingestion/recall/safety checks, hash-chain audit trail integrity, LLM synthesis with abstention gating, RxNorm drug normalization, SNOMED CT cross-reactivity detection, UMLS crosswalk mapping, consensus engine verification, what-if simulation, FDA client, clinical trials matching, PHI detection, and hallucination detection.

## Challenges we ran into

1. **Confidence calibration** — The initial confidence formula was a self-referential ratio that always returned ~1.0. We redesigned it to use absolute evidence strength, so low BM25 scores actually trigger abstention.
2. **Cross-provider contradiction detection** — Different specialists legitimately disagree (e.g., BP targets for cardiac vs renal patients). We had to distinguish harmful contradictions from acceptable clinical nuance.
3. **FHIR data normalization** — Medications come in many formats (brand names, generics, coded concepts). Our fuzzy matching uses substring containment rather than exact match to catch variations.
4. **Negation boundary conditions** — "ruled out" at end of string vs mid-sentence. Clinical NLP must handle terminal negation markers without trailing spaces.
5. **UMLS vocabulary alignment** — Medications, allergies, and lab tests use different coding systems across providers (ICD-10, SNOMED CT, LOINC, RxNorm). We built a crosswalk module using UMLS Metathesaurus CUI-based resolution to unify them.
6. **Multi-LLM consensus orchestration** — Each LLM provider has a different API format (OpenAI, Anthropic Messages API, Google generateContent, xAI, Perplexity). We built provider-specific adapters and parallel execution with timeout handling to get consensus in under 5 seconds.
7. **PHI detection in free text** — Clinical notes contain unstructured text where PHI can appear in unexpected formats. We built regex-based pattern detection for SSN, MRN, phone numbers, emails, dates of birth, and physical addresses.

## Accomplishments that we're proud of

- **BitNet b1.58 as the FDA SaMD reproducibility primitive** — Layer 4.5 is **not** the primary DDI classifier (primary recall comes from RxNorm + OpenEvidence + NIH RxNav + 6-LLM consensus). Layer 4.5 exists to make every clinical decision **bit-identical across CPU, GPU, and NPU** — the determinism the FDA's 2024 AI/ML SaMD guidance expects from clinical decision support. Its job is *deterministic verification*, not headline accuracy. **100% recall + 0 false positives on the safety-critical `contraindicated` class (44/44)**, used as a high-precision veto: when Layer 4.5 disagrees with the upstream pipeline by predicting `none` or `minor` on a `contraindicated` pair, that disagreement triggers a `BITNET_SAFETY_DOWNGRADE_DISAGREEMENT` alert and the safer (more severe) verdict always wins. Live engine bundle is **v8** (iter-275 promotion, `1f0f8859…`, 193-dim hash + 26 ATC pharmacology flags + 13 pair-derived DDI rule bits × 256 hidden); **50,688 ternary {-1, 0, +1} weights** + 261 Q16.16 biases (50,949 params) serialize to ~118 KB JSON and run in `<1 ms` per pair on a $5 Raspberry Pi Zero. The pre-promotion v1 baseline (`cfadb4f6`, 8,581 params / 19 KB / hash-only 128-dim × 64-hidden) is preserved at `engine/bitnet_weights.v1.cfadb4f6.bak.json` for full audit-chain reconstruction (any auditor can replay decisions made before iter-275 under the prior bundle). Every classification carries a SHA-256 `repro_hash` any auditor can re-verify in `<1 ms` per pair, decades later, with just the weights bundle and the Python file — no proprietary toolchain. Full recipe + per-class accuracy + corpus build in `docs/bitnet_training.md`.
- **Validated against the 139-pair PCCP regression cohort + the NTI anchor set** — The live engine cohort (post iter-280 cohort growth) is **139 drug pairs across 4 severity classes** (44 contraindicated · 4 major · 22 moderate · 69 serious), built around the canonical FDA / AGS Beers / STOPP-START NTI anchors (warfarin, digoxin, lithium, phenytoin, methotrexate, plus iter-280's MAOI×SNRI tranylcypromine + venlafaxine). Headline numbers under cross-arch Q16.16 inference: **100% recall on contraindicated (44/44), 100% recall on major (4/4), 0 false positives on contraindicated**. v8 closed every v1/v6/v7 historical miss including the lurasidone+ketoconazole + tacrolimus+voriconazole pairs the hash-only architectural ceiling missed. 17% abstention rate on under-evidenced pairs — Layer 6 says "I don't know" rather than guess. Full confusion matrix + per-pair reproducibility recipe: `docs/clinical_validation.md`.
- **CMS-validated provider attribution (NPI)** — Sarah Mitchell's 4 demo Practitioners (PCP, Cardiologist, Nephrologist, ER physician) ship with full NPI identifiers in the FHIR R4 bundle. Every NPI passes the official CMS `80840`-prefix Luhn check (`engine/npi_registry.py::validate_npi`); the module also includes a public NPPES lookup client for live registry validation. Every `MedicationStatement.informationSource` and `Observation.performer` resolves back to a verifiable Practitioner record — exactly what an audit reviewer needs to reconstruct who prescribed what.
- **Six-model US-based LLM consensus** — Drug interaction verdicts verified by GPT-5.5, Gemini 3.1 Pro, Grok 4.3, Claude Opus 4.7, Perplexity Sonar Pro, and NVIDIA Nemotron Ultra 253B in parallel. Every provider is US-headquartered. No single model can hallucinate a finding into production; the cascade scales gracefully when API keys are missing (1-6 providers).
- **What-if medication simulation** — Clinicians can preview safety outcomes of medication changes before making them — like a flight simulator for prescriptions.
- **FDA safety alert integration** — Real-time queries to openFDA for active recalls and adverse event signals on patient medications.
- **Clinical trial matching** — Automatically finds relevant active trials on ClinicalTrials.gov for patient conditions with eligibility pre-screening.
- **PHI detection guard** — Catches Protected Health Information before it enters the memory layer, preventing accidental data leakage.
- **Hallucination detection** — Validates LLM-generated clinical text against evidence, flagging unsupported claims.
- **Deterministic safety + GenAI synthesis** — Rules catch the conflicts reliably; LLMs explain them in clinical context. Best of both worlds.
- **4 conflicts caught in demo** — Drug interactions, allergy cross-reactions, declining lab trends, and provider disagreements — all detected automatically with patient-specific clinical rationale.
- **Full UMLS terminology integration** — RxNorm, SNOMED CT, and UMLS Metathesaurus for standardized clinical vocabulary across 13 engine modules.
- **Safe abstention** — When evidence is insufficient, ClinicalMem says "I don't know" instead of hallucinating. In healthcare, this saves lives.
- **Tamper-proof audit trail** — SHA-256 Merkle chain provides cryptographic proof that clinical decision logs haven't been altered.
- **356 tests passing** — Comprehensive coverage across clinical scoring, FHIR integration, drug normalization, allergy cross-reactivity, UMLS crosswalk, LLM synthesis with abstention gating, consensus engine, what-if simulation, FDA integration, clinical trials, PHI detection, and hallucination detection.
- **Dual protocol support** — Both MCP (18 tools) and A2A (13 tools) for maximum interoperability.
- **Live on Azure** — MCP server and A2A agent deployed on Azure Container Apps with zero cold-start.

## What we learned

- MIND Lang's abstention kernel philosophy (refuse to answer when uncertain) is critical for clinical AI safety
- FHIR R4's bundled resource model maps naturally to mind-mem's block-based memory architecture
- Healthcare AI needs memory that is auditable, contradiction-aware, and confidence-gated — not just retrieval
- UMLS terminology services (RxNorm, SNOMED CT, Metathesaurus) are essential for interoperability across providers
- Multi-LLM consensus is more reliable than any single model — different models catch different edge cases
- What-if simulation transforms a safety tool into a clinical decision support system
- PHI detection must happen at the ingestion boundary, not as an afterthought

## What's next for ClinicalMem

- BiomedBERT vector embeddings for medical-specific semantic search
- Multi-patient analytics (population-level safety signals)
- Production deployment with HIPAA-compliant encryption at rest via OpenAI Healthcare API with HIPAA BAA
- Real-time medication monitoring via DailyMed FDA label integration for black-box warnings
- Temporal trend analysis with predictive alerts (detect worsening trajectories before they hit thresholds)
- Integration with Epic FHIR APIs for real EHR data access

## Built With

- Python 3.12
- mind-mem (persistent clinical memory with BM25 + vector + RRF fusion)
- MIND Lang (abstention, importance, adversarial scoring kernels)
- FastMCP 2.x (MCP server with Streamable HTTP)
- Google ADK (A2A agent)
- FHIR R4 (patient data standard)
- RxNorm REST API (drug normalization — NIH/NLM)
- SNOMED CT (allergy cross-reactivity hierarchies)
- UMLS Metathesaurus (cross-vocabulary mapping)
- OpenEvidence API (Mayo Clinic / Elsevier ClinicalKey AI)
- openFDA API (drug safety alerts and adverse events)
- ClinicalTrials.gov API (clinical trial matching)
- OpenAI GPT-5.5, Google Gemini 3.1 Pro, xAI Grok 4.3, Anthropic Claude Opus 4.7, Perplexity Sonar Pro, NVIDIA Nemotron Ultra 253B via NIM (six-model US-based consensus)
- Docker + Azure Container Apps (deployment)
- Cloudflare Pages (demo dashboard)
