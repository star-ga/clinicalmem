# ClinicalMem Changelog

## v4.1.1 — 2026-05-14

Post-submission hotfix for `/.well-known/agent-card.json` to match the
**A2A v1 schema** that Prompt Opinion announced at
https://docs.promptopinion.ai/a2a-v1-migration. v4.1.0 shipped a
forward-compat *mirror* (added v1 keys alongside v0); judges hitting
the live agent card with a v1-only validator would still see the v0
fields and could reject. v4.1.1 makes the served JSON v1-clean.

### Fixed
- `AgentCardCompatMiddleware` (a2a_agent/app.py) now **strips** the
  v0-only fields v1 removed (verified against `a2a-sdk==1.0.3`
  protobuf descriptors): top-level `url`, `preferredTransport`,
  `supportsAuthenticatedExtendedCard`, `additionalInterfaces`, and
  `capabilities.stateTransitionHistory`.
- `security` is now **moved** to `securityRequirements` (was mirrored
  in v4.1.0). v1 consumers that fail-closed on unknown keys would
  reject a card that has both.
- `supportedInterfaces[0].protocolVersion` is now `"1.0"` (was
  `"0.3.0"` — the SDK build version, which is the wrong dimension to
  expose; protocolVersion tracks the A2A spec).

### Stack
- `a2a-sdk` pin tightened from `>=0.2` to `>=0.3.24,<1.0`. `google-adk`
  `to_a2a()` wrapper still binds the 0.3.x AgentCard ctor signature;
  the JSON shape on the wire is upgraded by the middleware, not the
  SDK.

### Tests
- `TestAgentCardCompat` (tests/test_a2a/test_a2a_app.py) now has
  4 explicit v1-strip assertions covering top-level v0 removal,
  capability removal, security→securityRequirements move, and
  `protocolVersion == "1.0"`. 6/6 passing.

### Deploy
- Same Azure FQDN (`clinicalmem-a2a.thankfulpond-9c3fdc1e.eastus.azurecontainerapps.io`)
  → Devpost submission text + Prompt Opinion marketplace listing
  unchanged. New revision `clinicalmem-a2a--0000010` at image
  `clinicalmemacr.azurecr.io/clinicalmem-a2a:680c51a`, 100% traffic.

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
