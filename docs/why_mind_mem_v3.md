# Why ClinicalMem requires `mind-mem >= 3.0.0`

ClinicalMem v4.1.0 depends on `mind-mem >= 3.0.0`. The earlier `>= 1.9.0` floor was carried from the v0.1 prototype; the surfaces ClinicalMem actually relies on are all v3.0.0+. This file records which surfaces matter and why.

## v3.0.0+ surfaces ClinicalMem builds on

| Surface | Module | Why ClinicalMem needs it |
|---|---|---|
| **SQLCipher at-rest encryption** | `mind_mem.block_store_encrypted` | HIPAA-compatible at-rest encryption is non-negotiable for clinical memory blocks. The SQLCipher path was added in mind-mem v3.0.0; before that the BlockStore was plaintext SQLite. Without v3.0.0+ the at-rest-encryption claim in our DevPost copy is unsupportable. |
| **TAG_v1 audit-integrity preimages** | `mind_mem.audit_chain` (v2.10.0+ schema) | The hash-chain audit trail uses NUL-separated hash preimages with Q16.16 fixed-point scoring in the preimage payload. The TAG_v1 schema lands in v2.10.0 and is the preimage shape ClinicalMem's verifier expects. |
| **Tier decay (TTL + LRU)** | `mind_mem.tier_decay` | Acute clinical findings persist; routine entries age. Tier decay is the mechanism — configurable, benchmarked on LoCoMo. We document this as the production replacement for "we forget like humans" memory metaphors. |
| **HybridBackend (BM25 + vector + RRF)** | `mind_mem.hybrid_recall.HybridBackend` | The hybrid retrieval path used by `engine/clinical_memory.py`. Available since v1.x but the v3 implementation adds Porter stemming + RM3 expansion and the RRF fusion exponent we tune in `mind-mem.json`. |
| **AuditChain serialisation** | `mind_mem.audit_chain.AuditChain` | The hash-chain primitive. Used directly in `engine/clinical_memory.py` to record every memory write. |
| **Native MCP integration (16 clients)** | `mm install-all` (v3.1.0+) | Not load-bearing for ClinicalMem itself, but the same MCP discipline ClinicalMem applies (one tool per public surface, no overlapping schemas) is the discipline mind-mem ships and validates against — i.e. ClinicalMem inherits a tested protocol contract. |

## What we do NOT use from mind-mem (and why)

| Surface | Reason ClinicalMem does not depend on it |
|---|---|
| `mind_mem.governance.proposal_queue` | ClinicalMem's writes are deterministic safety findings; they should not enter a human-approval queue. We bypass it intentionally. |
| `mind_mem.alerting` (webhook/Slack hooks) | The clinical alerts we emit go through the audit chain + the synthesizer — alerting handlers are operator-level concerns, not part of the demo. |
| `mind_mem.cross_encoder_rerank` | The rerank path is opt-in and config-gated; ClinicalMem's recall queries are direct enough that BM25 + vector + RRF is sufficient. The cross-encoder would add latency without a measurable retrieval-quality lift on the Sarah Mitchell scenario. |
| `star-ga/mind-mem-4b` (the local fine-tune model) | ClinicalMem's LLM path is the 6-LLM cloud-consensus surface. The local fine-tune is an option for air-gapped deployments and is documented separately under "offline mode." |

## Operational guidance

- ClinicalMem reads `mind-mem.json` from the working directory; the only key it sets explicitly is the BlockStore encryption mode. Defaults are otherwise mind-mem's defaults.
- The audit chain produced by ClinicalMem can be replayed against the canonical `mind-mem` reference verifier (`mm verify-chain`) — drift between what ClinicalMem records and what `mind-mem` expects is a release-blocking event.
- ClinicalMem's `engine/clinical_memory.py` is the single point of contact with mind-mem; refactors to that module must preserve the AuditChain + HybridBackend contracts.

## Cross-reference

- `engine/clinical_memory.py` — the only ClinicalMem module that imports from `mind_mem`.
- `tests/test_engine/test_clinical_memory.py` — the regression tests for the integration.
- `docs/regulatory_readiness.md` — how at-rest encryption, audit chain, and tier decay map to HIPAA + FDA SaMD audit requirements.
- mind-mem's own `docs/governance.md` — the canonical L4 retrieval-time governance description.

---

*Doc landed 2026-05-02 as part of the v2 hackathon plan (Day 1, item A — "mind-mem v3.x dep upgrade + tier-decay narrative").*
