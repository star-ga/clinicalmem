# ClinicalMem Changelog

## v4.1.0 — 2026-05-11

Hackathon submission cut for the **Agents Assemble — The Healthcare AI Endgame** competition.

### Added
- Path B BitNet b1.58 ensemble: bundle A (50,949 ternary params, 118 KB) cascades to bundle B (12,741 ternary params, 30 KB).
- 100% recall on the 139-pair PCCP cohort: 44/44 contraindicated · 4/4 major · 69/69 serious · 22/22 moderate · zero false positives.
- 21 typed runtime federation invariants (`flows/JointMemoryFederation.flow.mind`) enforced by the MIND compiler.
- In-browser Verify Replay: bit-identical Q16.16 fixed-point — server hash matches the browser BitNet replay byte-for-byte. <1 ms on a $15 Raspberry Pi Zero 2.
- 18 MCP tools (FastMCP 2.x · SHARP-on-MCP) + Google ADK A2A agent (5 skills · 13 tools), both deployed on Azure Container Apps.
- SHA-256 Merkle audit chain, TAG_v1 NUL-separated preimages, HIPAA § 164.312(b) aligned.
- Cross-architecture determinism harness (x86_64 / ARM64 / CUDA).
- mind-mem v4.0.1 federation HTTP wire transport bridge (4 endpoints flag-gated by `v4.federation`).
- Open-source HF model release: `huggingface.co/star-ga/clinicalmem-bitnet-b158`.

### Demo
- Live dashboard: https://clinicalmem-demo.pages.dev/
- 3-minute demo video: https://youtu.be/1mICQ3zBbvo
- Devpost submission: https://devpost.com/software/clinimalmem
- Prompt Opinion Marketplace MCP listing: https://app.promptopinion.ai/marketplace/mcp/019ccad4-0a5f-7da4-b9f4-a2ae212d9217

### Fixed
- A2A agent card now serves `supportedInterfaces` (and mirrors `securityRequirements`) so registries built against a2a-sdk 1.0.x — including the Prompt Opinion marketplace — accept the card without an SDK upgrade. Implemented as a request-scoped Starlette middleware on `/.well-known/agent-card.json`; no behaviour change for callers built against 0.3.x.

### Tests
- 1425 tests passing across engine, MCP server, A2A agent, federation invariants, and Q16.16 determinism stress.

### Notes
- Synthetic patient data only — not medical advice.
- Apache-2.0.
