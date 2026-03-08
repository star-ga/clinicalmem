# ClinicalMem

[![Tests](https://img.shields.io/badge/tests-90%2F90-brightgreen)](#tests)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.12+-blue.svg)](https://python.org)
[![MCP](https://img.shields.io/badge/protocol-MCP-purple)](#mcp-server)
[![A2A](https://img.shields.io/badge/protocol-A2A-orange)](#a2a-agent)

Persistent, auditable, contradiction-safe clinical memory for healthcare AI agents.

Built for the [Agents Assemble Healthcare AI Hackathon](https://agents-assemble.devpost.com/) by [STARGA Inc.](https://star.ga)

## What It Does

ClinicalMem gives healthcare AI agents a **persistent memory** that:

- **Catches drug interactions** — flags Warfarin + Ibuprofen bleeding risk, contraindications
- **Detects allergy conflicts** — cross-checks medications against allergy records (Penicillin → Amoxicillin)
- **Finds provider contradictions** — surfaces conflicting care plans across multiple providers
- **Maintains audit trails** — SHA-256 hash-chained, tamper-evident record of every clinical decision
- **Scores by clinical importance** — acute conditions rank higher than routine observations
- **Knows when to abstain** — confidence gating prevents hallucinated clinical guidance

Powered by [mind-mem](https://github.com/star-ga/mind-mem) hybrid search engine (BM25 + vector + RRF fusion) and [MIND Lang](https://github.com/star-ga/mind) scoring kernels.

## Architecture

```
              SHARED CORE ENGINE
      +---------------------------------+
      |   ClinicalMem Engine            |
      |   (mind-mem adapted)            |
      |                                 |
      |   - Hybrid Search (BM25+Vec)    |
      |   - Contradiction Detection     |
      |   - Causal Dependency Graph     |
      |   - Audit Chain                 |
      |   - Clinical Scoring            |
      |   - FHIR Data Integration       |
      +--------+------------------------+
               |
     +---------+-----------+
     |                     |
MCP Server            A2A Agent
(FastMCP 2.x)        (Google ADK)
11 tools              5 skills
SHARP-on-MCP          A2A protocol
Streamable HTTP       Azure Container Apps
```

## MCP Server (11 Tools)

| Tool | Purpose |
|------|---------|
| `store_clinical_observation` | Store clinical note for patient |
| `recall_patient_context` | Hybrid search over clinical memory |
| `check_medication_conflicts` | Drug interaction detection |
| `check_allergy_conflicts` | Allergy cross-reaction alerts |
| `get_treatment_dependencies` | Condition-medication dependency map |
| `get_clinical_audit_trail` | Tamper-proof audit chain |
| `summarize_patient_history` | Importance-scored patient overview |
| `detect_belief_drift` | Contradiction detection |
| `ingest_patient_data` | Pull all FHIR data into memory |
| `explain_clinical_conflict` | **GenAI**: LLM-grounded conflict explanation with citations |
| `clinical_care_handoff` | **GenAI**: Evidence-grounded care handoff note |

## A2A Agent (5 Skills)

| Skill | Description |
|-------|-------------|
| `medication-safety-review` | Drug interactions + allergy conflicts |
| `clinical-context-recall` | Hybrid search with confidence gating |
| `contradiction-assessment` | Scan for conflicting records |
| `care-transition-summary` | Structured handoff summary |
| `explain-conflict` | **GenAI**: LLM-grounded clinical rationale with evidence citations |

## Synthetic Patient: Sarah Mitchell

67-year-old female with 4 providers, 22 FHIR resources, and **4 planted conflicts** — all detected:

1. **Ibuprofen + Warfarin** — serious bleeding risk ✅
2. **Amoxicillin + Penicillin allergy** — anaphylaxis risk ✅
3. **Declining GFR + Metformin** — approaching contraindication (GFR 45→38→32) ✅
4. **BP target conflict** — provider belief drift (Cardiologist <130/80 vs Nephrologist <140/90) ✅

Plus 2 bonus findings the engine discovers autonomously:
- **INR 3.8** — above therapeutic range (2.0-3.0), correlated with ibuprofen addition
- **eGFR declining trajectory** — 13-point drop over 3 measurements, approaching metformin contraindication threshold

## Quick Start

```bash
# Run tests
python -m pytest tests/ -v

# MCP server
python -m mcp_server

# A2A agent
uvicorn a2a_agent.app:a2a_app --host 0.0.0.0 --port 8001

# Docker
docker compose up --build
```

## Tests

90 tests covering engine scoring, FHIR ingestion, medication safety, lab-medication contraindications, lab trend analysis, provider disagreement detection, contradiction detection, recall, audit chain integrity, and LLM synthesis (explain_conflict + clinical_handoff with abstention).

```
tests/test_engine/test_clinical_scoring.py — 43 tests
tests/test_engine/test_integration.py      — 47 tests
```

## Tech Stack

- **Engine:** mind-mem hybrid search (BM25 + vector + RRF fusion)
- **Scoring:** MIND Lang kernel patterns (confidence gating, importance, negation)
- **MCP:** FastMCP 2.x with SHARP-on-MCP headers
- **A2A:** Google ADK with A2A protocol
- **FHIR:** R4 client with synthetic patient data
- **Deploy:** Docker + Azure Container Apps

## License

MIT — see [LICENSE](LICENSE)

Built on [mind-mem](https://github.com/star-ga/mind-mem) (Apache 2.0) by [STARGA Inc.](https://star.ga)
