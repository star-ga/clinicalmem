# ClinicalMem — Deployment Architecture

> *"Every layer is independently auditable. The whole stack is reproducible
>  bit-for-bit on a Raspberry Pi or an enterprise GPU cluster."*

This document is the canonical deployment-architecture reference for
ClinicalMem. It complements the per-component docs (`docs/bitnet_training.md`,
`docs/federated_memory.md`, `docs/clinical_validation.md`,
`docs/fda_q_sub_draft.md`) by showing how the pieces fit together at runtime
and across sites.

## Single-site deployment (one hospital)

```
                            ┌───────────────────────────────┐
                            │        EHR / SMART-on-FHIR    │
                            │  (Epic · Cerner · MEDITECH)   │
                            └──────────────┬────────────────┘
                                           │  FHIR R4 Bundle
                                           │  (Patient + Conditions
                                           │   + MedicationStatement
                                           │   + Practitioners with NPI
                                           │   + AllergyIntolerance)
                                           ▼
              ┌──────────────────────────────────────────────────────┐
              │  engine.fhir_adapter.ingest_bundle()                  │
              │   ─ FHIR R4 validation                                │
              │   ─ PHI Safe Harbor scrub (18 identifier classes)     │
              │   ─ NPI Luhn validation (engine.npi_registry)         │
              │   ─ canonical SHA-256 of the bundle (audit anchor)    │
              └──────────────┬───────────────────────────────────────┘
                             │  ClinicalIngestResult
                             │  (medications, conditions, allergies,
                             │   observations, practitioner_npis)
                             ▼
   ┌─────────────────────────────────────────────────────────────────────┐
   │                        6-LAYER SAFETY PIPELINE                      │
   │  ┌─────────────────────────────────────────────────────────────┐    │
   │  │ L1   Deterministic table              <1 ms     OFFLINE    │    │
   │  │ L2   OpenEvidence API (Mayo / Elsevier)  ~2 s   ONLINE     │    │
   │  │ L3   NIH RxNorm DDI API                  ~1 s   ONLINE     │    │
   │  │ L4   5-LLM US-based consensus (parallel) ~3 s   ONLINE     │    │
   │  │ L4.5 BitNet b1.58 ternary (Q16.16 FP)    <1 ms  OFFLINE    │    │
   │  │      ─ FDA SaMD reproducibility primitive                  │    │
   │  │      ─ 100% recall on contraindicated (NTI cohort)         │    │
   │  │ L5   LLM synthesis (cited evidence)      ~3 s   ONLINE     │    │
   │  │ L6   Abstention gate ("I don't know")    0 ms   OFFLINE    │    │
   │  └─────────────────────────────────────────────────────────────┘    │
   └────────────────────────────┬────────────────────────────────────────┘
                                │  list[DrugInteraction] with
                                │  (severity, bitnet_severity,
                                │   bitnet_repro_hash, bitnet_weights_id)
                                ▼
              ┌──────────────────────────────────────────────────────┐
              │             mind-mem v3.x (local store)              │
              │   ─ SQLCipher at-rest encryption (HIPAA)             │
              │   ─ BM25 + vector + RRF fusion retrieval             │
              │   ─ TAG_v1 NUL-separated audit chain (Q16.16 hashes) │
              │   ─ Local Qwen-3.5-4B fine-tune for memory extract   │
              │   ─ Tier decay + contradiction detection             │
              └──────────────┬───────────────────────────────────────┘
                             │
                             ▼
              ┌──────────────────────────────────────────────────────┐
              │  engine.audit_export_part11.export_audit_trail()      │
              │   ─ JSON-LD Merkle hash chain                        │
              │   ─ NPI-attested signature                           │
              │   ─ FDA-replayable decades later                     │
              └──────────────────────────────────────────────────────┘
                             │
                             ▼
        ┌──────────────────────────────────────────┐    ┌─────────────┐
        │   MCP server (18 tools)                  │    │  A2A agent  │
        │   ─ Streamable HTTP                      │    │  (13 skills)│
        │   ─ Azure Container Apps                 │    │  Google ADK │
        └──────────────────────────────────────────┘    └─────────────┘
                             ▲                                  ▲
                             │                                  │
                             └─────────── HEALTHCARE AI ────────┘
                                          AGENT
```

## Multi-site federation (cross-hospital)

```
             ┌──── SITE A — Mass General ────┐         ┌──── SITE B — Mayo Clinic ────┐
             │                               │         │                               │
             │  6-layer pipeline (above)     │         │  6-layer pipeline (above)     │
             │           │                   │         │           │                   │
             │           ▼                   │         │           ▼                   │
             │  ┌───────────────────────┐    │         │  ┌───────────────────────┐    │
             │  │ Local mind-mem v3.x   │    │         │  │ Local mind-mem v3.x   │    │
             │  │ (SQLCipher, PHI here) │    │         │  │ (SQLCipher, PHI here) │    │
             │  └───────┬───────────────┘    │         │  └────────────────▲──────┘    │
             │          │                    │         │                   │            │
             │          ▼                    │         │                   │            │
             │  flows/JointMemoryFederation  │         │  flows/JointMemory…           │
             │     EGRESS path:              │         │     INGRESS path:             │
             │   ┌─────────────────────┐     │         │   ┌─────────────────────┐    │
             │   │ classify (lane gate)│     │         │   │ x25519_open + AEAD  │    │
             │   │ phi_strip           │     │         │   │ ed25519_verify      │    │
             │   │ FHIR struct guard   │     │         │   │ KeyEpoch deny-list  │    │
             │   │ stamp issued_at     │     │         │   │ freshness window    │    │
             │   │ ed25519_sign + ep   │     │         │   │ phi_recheck         │    │
             │   │ x25519_seal +       │     │         │   │ tier_clamp [0..5]   │    │
             │   │  chacha20-poly1305  │     │         │   │ severity_quorum 3/5 │    │
             │   └─────────┬───────────┘     │         │   └─────────────────────┘    │
             └─────────────┼─────────────────┘         └───────────────────────────────┘
                           │                                          ▲
                           │                                          │
                           ▼                                          │
                  ┌─────────────────────────────────────────────┐     │
                  │  mind-mem v3.x multi-machine transport      │     │
                  │  (patent-pending STARGA technology)         │     │
                  │   ─ MAP   — Mind Annotation Protocol        │     │
                  │   ─ MIC@2 — Mind Interchange Coding v2      │     │
                  │   ─ binary framing                          │     │
                  │   ─ at-least-once delivery + dedup          │     │
                  └────────────┬────────────────────────────────┘     │
                               │                                       │
                               └───────────────────────────────────────┘
                                  
              KNOWLEDGE LANE (de-identified, encrypted, signed):
                ─ Drug-pair severity verdicts + repro_hash + bundle_id
                ─ BitNet activations on novel pairs
                ─ Audit-chain witnesses (hash receipts only)
                ─ Anonymised provider-disagreement patterns
              
              PHI LANE (NEVER leaves the originating site):
                ─ Patient names, DOB, MRN, addresses, insurance IDs
                ─ FHIR Patient resources
                ─ Free-text clinical notes
                ─ Quarantined by typed runtime invariant, not policy
```

## Edge / offline deployment (Raspberry Pi Zero)

```
              ┌──────────────────────────────────────────────────┐
              │              Raspberry Pi Zero (~$15)            │
              │   ─ 512 MB RAM, ARM Cortex-A53 single-core       │
              │   ─ Python 3.10+, no GPU, no internet            │
              │                                                  │
              │   engine/bitnet_classifier.py  (15 KB)           │
              │   engine/bitnet_weights.json   (19 KB ternary)   │
              │   engine/clinical_scoring.py   (+ Layer 1 only)  │
              │                                                  │
              │   Forward pass: <1 ms per drug pair              │
              │   Same repro_hash as the datacenter run          │
              │   Same severity verdicts bit-for-bit             │
              └──────────────────────────────────────────────────┘
```

The edge deployment is the load-bearing demonstration of the bit-identical
reproducibility primitive: a $15 Pi Zero produces the same SHA-256
`repro_hash` for every drug pair as a datacenter A100. An FDA auditor with
the 19 KB weights bundle and the 15 KB Python file can replay any past
clinical decision on any device, decades later.

## Mock vs. live transport (current state, 2026-05-03)

| Component | Status | Code |
|---|---|---|
| 6-layer pipeline (L1–L6) | ✅ Live | `engine/clinical_scoring.py` |
| BitNet b1.58 trained classifier | ✅ Live | `engine/bitnet_classifier.py` + `engine/bitnet_weights.json` |
| OpenEvidence cache fallback | ✅ Live | `engine/openevidence_cache.py` + `docs/openevidence_cache.json` |
| OpenEvidence live API | ⏳ Pending key (academic license requested 2026-05-02) | Falls through to cache automatically |
| FHIR R4 SMART-on-FHIR ingress | ✅ Live | `engine/fhir_adapter.py` |
| 21 CFR Part 11 audit export | ✅ Live | `engine/audit_export_part11.py` |
| PCCP regression harness | ✅ Live | `scripts/run_clinical_regression_eval.py` |
| Federation typed contract | ✅ Live | `flows/JointMemoryFederation.flow.mind` (21 typed runtime invariants, plan_hash 6c6fb3ea…5846) |
| Federation **mock** transport (in-process queue) | ✅ Live | `scripts/federation_mock_demo.py` |
| Federation **live** transport (mind-mem MIC@2/MAP/binary) | ⏳ In flight by upstream mind-mem team | Will land in mind-mem v3.x release |
| MCP server (18 tools) | ✅ Live | `mcp_server*.py` deployed on Azure Container Apps |
| A2A agent (13 skills) | ✅ Live | `a2a_agent/` deployed on Azure Container Apps |

The mock transport is a faithful in-process simulation: same canonical
preimage encoding, same Ed25519 + X25519 + ChaCha20-Poly1305 cryptographic
primitives, same TAG_v1 audit chain. When the live mind-mem MIC@2/MAP/binary
transport ships, swapping the transport call in
`flows/JointMemoryFederation.flow.mind::mind_mem_publish` is the only
client-side change required.

---

*Apache-2.0 — STARGA, Inc. — 2026.*
*MIC@2, MAP, and binary framing are patent-pending STARGA technologies.*
