# Why Mind Flow — six per-scenario flows + the `plan_hash` decision ID

ClinicalMem v4.2 ships six `.flow.mind` source files, each describing a
typed-graph contract for a clinical safety scenario. Every flow has a
content-addressed `plan_hash` (SHA-256 over the canonical source) that
becomes the **clinician-facing decision ID** for every recommendation
the flow produces. Three months later, a regulator types that hash into
ClinicalMem's verifier and replays the exact same flow against the same
inputs to confirm bit-identity.

## The six flows

| Flow | Catches | Input ports | Plan-hash |
|---|---|---|---|
| `MedicationSafetyReview.flow.mind` | NSAID + warfarin → fatal-bleeding risk | `patient_id, medications` | content-addressed |
| `AllergyCrossReactivity.flow.mind` | Amoxicillin prescribed for a penicillin-anaphylaxis patient via SNOMED CT class hierarchy | `patient_id, medications, allergies` | content-addressed |
| `LabContraindication.flow.mind` | GFR trending 45 → 32 while on metformin | `patient_id, medications, lab_history` | content-addressed |
| `ProviderDisagreement.flow.mind` | Cardiologist `BP < 130/80` vs nephrologist `BP < 140/90` | `patient_id, provider_records` | content-addressed |
| `WhatIfSimulation.flow.mind` | "What if we add aspirin to a warfarin regimen?" — runs the full safety review on a hypothetical regimen | `patient_id, current, perturbation` | content-addressed |
| `ClinicalTrialMatch.flow.mind` | Active-trial eligibility cross-checked against demographics + active conditions | `patient_id, conditions, demographics` | content-addressed |

The plan-hash for each flow is computed by `engine/flow_runner.py`'s
`compute_plan_hash(name)` (Apache-2.0). Same source bytes → same hash on
every machine.

## What makes `plan_hash` the load-bearing audit primitive

| Question a regulator asks | How `plan_hash` answers it |
|---|---|
| "What pipeline produced this clinical recommendation?" | The audit chain stores `(flow_name, plan_hash, input_hash, output_hash)`. The `plan_hash` is a 64-char SHA-256 — unique and content-addressed. |
| "Has the safety pipeline changed since this decision?" | Recompute `plan_hash` on the current source. A divergence is a release-blocking event. `verify_replay()` does this in one call. |
| "Can I reproduce this decision exactly?" | Yes — pin `plan_hash`, replay against the same inputs, assert byte-identity of output (every flow's intermediate node hashes are recorded in the audit chain). |
| "Did anyone tamper with the audit log?" | `engine/flow_runner.verify_replay()` returns `matches=False`; the integrity event surfaces in CI and on `/v1/replay`. |

## How this lands inside the existing 5-tier safety pipeline

The flow source files are **architectural contracts** — they describe
*what* the pipeline does in typed-graph form. The Python implementation
of *how* each node executes lives in `engine/` (clinical_scoring,
consensus_engine, llm_synthesizer, rxnorm_client, snomed_client,
phi_detector, hallucination_detector, bitnet_classifier).

Two surfaces are kept in sync:

| Surface | Kept-in-sync invariant | Where |
|---|---|---|
| `flows/MedicationSafetyReview.flow.mind` ↔ `engine/consensus_engine.py` etc. | Every `node` in the flow has a corresponding implementation entry-point in `engine/`. | `tests/test_engine/test_flow_runner.py` asserts the architectural surface (PHI scan → normalize → tier1 → tier2 → tier3 → BitNet → consensus → synth → audit). |
| `flows/MedicationSafetyReview.flow.mind` ↔ DEVPOST.md "six-layer pipeline" copy | The user-facing description matches the contract source. | The architectural-rule layer in `.arch-mind/rules.mind` flags drift. |

The `engine/flow_runner.py` parser is line-oriented and intentionally
small (~250 LOC); it extracts the *contract surface* (inputs, outputs,
nodes, invariants) but does not implement a full `mindc` parser. The
binary-execution path is opt-in via the upstream MIND Flow toolchain at
[mindlang.dev](https://mindlang.dev) — not required for verification.

## Why this is a ClinicalMem-only differentiator

| Property | ClinicalMem (with .flow.mind contracts) | Best of the field today |
|---|---|---|
| Per-decision `plan_hash` | ✓ — 64-char SHA-256, audit-chain primary key | none |
| Flow-source-as-documentation | ✓ — `.flow.mind` IS the documentation; the same file the system executes is what auditors read | partial — usually a Python file + a separate Markdown spec, often drifting |
| Tamper-detection on the safety pipeline itself | ✓ — `verify_replay` returns `matches=False` if the source changed | none |
| Typed `@verify` invariants on every flow | ✓ — `consensus.majority_or_abstain == true`, `phi.safe_for_external == true`, `synth.citations_valid == true`, `synth.unsupported_claims == 0`, etc. | none |
| Compositional flows (flow-calls-flow) | ✓ — `WhatIfSimulation` calls `MedicationSafetyReview` twice and diffs the outputs | none |

## Cross-reference

- `flows/*.flow.mind` — the six shipped flow sources.
- `engine/flow_runner.py` — the clean-room Python verifier (Apache-2.0; ~250 LOC).
- `tests/test_engine/test_flow_runner.py` — 51 tests covering catalogue, plan-hash determinism, tamper detection, and architectural-surface invariants.
- `docs/why_bitnet_b158.md` — Layer 4.5, called by `MedicationSafetyReview.flow.mind`.
- `docs/why_mind_mem_v3.md` — L4 retrieval-time governance the flows record into.
- `.arch-mind/rules.mind` — L1 architectural-governance gate that flags pipeline drift.
- `HACKATHON_PLAN_V2.md §D` — the rationale for shipping these flows in the v2 plan.

The upstream MIND Flow project lives at
[github.com/star-ga/mind-flow](https://github.com/star-ga/mind-flow)
(STARGA Commercial License). ClinicalMem ships the flow sources under
Apache-2.0 along with a clean-room Python verifier; running the
upstream `mindc` toolchain to compile the flows into binaries is
optional and not required for ClinicalMem to operate.

---

*Doc landed 2026-05-02 as part of the v2 hackathon plan (Day 4–6, item D
— "Six per-scenario .flow.mind contracts + plan_hash decision ID").*
