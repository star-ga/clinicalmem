# Federated Joint Clinical Memory

> *"PHI never leaves the building. Knowledge does."*

ClinicalMem is the first clinical-memory system designed to federate
drug-interaction findings, BitNet activations, and provider-disagreement
patterns across multiple sites — **without ever moving a single patient
identifier**. The PHI/non-PHI boundary is a typed runtime invariant in
[`flows/JointMemoryFederation.flow.mind`](../flows/JointMemoryFederation.flow.mind),
not a policy doc, not a checklist, not a vendor promise.

## Why federation matters in healthcare AI

A safety finding discovered at Mass General — "warfarin + ibuprofen at
INR > 3.5 has a previously-unreported acute renal injury risk in CKD-3b
patients" — should not be re-discovered, painfully, at every other
hospital that runs ClinicalMem. The same goes for novel BitNet
classifier activations on rare drug pairs, anonymised
provider-disagreement patterns across cardiology vs. nephrology, and
audit-chain witnesses from one site that want to be cross-verified at
another.

But healthcare's HIPAA constraint is non-negotiable: **PHI cannot
cross site boundaries** without explicit Business Associate Agreements,
patient consent, or de-identification to Safe Harbor standards.

The standard industry response — "we don't federate, we'll re-discover
in each silo" — leaves clinical-AI safety as a per-site problem and
makes the network effect impossible. ClinicalMem solves this with a
**typed two-lane separation** baked into the flow contract.

## The two lanes

| Lane | What flows | Where it goes |
|---|---|---|
| **Knowledge lane** | Drug-pair severity verdicts (with `repro_hash` + `bundle_id`), BitNet classifier outputs on novel pairs, audit-chain witnesses (hash receipts only), anonymised provider-disagreement patterns (e.g. "27% of cardiologists targeted <130/80 in CKD-3b + AFib"), de-identified flow-execution evidence | Across the federation, between sites, free to propagate |
| **PHI lane** | Patient names, DOB, MRN, addresses, insurance IDs, FHIR Patient resources, free-text clinical notes, MedicationStatement notes referencing the patient, Observation values when paired with patient context | **Stays inside the originating site**, encrypted at rest, BAA required for any access |

The classifier at `flows/JointMemoryFederation.flow.mind::classify` is
the load-bearing PHI/non-PHI boundary. Its output (`lane`) is checked
by a typed invariant before any data hits the federation transport:

```mind
node classify = @native federation_classify(finding)
invariant classify.lane in ["clinical_knowledge", "phi_lane"]
invariant classify.lane != "phi_lane" or scrubbed.empty == true
```

A misclassification doesn't fail open — it fails closed with a
structured `InvariantViolation`. The transport never sees the payload.

## Transport: STARGA's patent-pending protocols

The federation rides on `mind-mem`'s upcoming multi-machine networking
layer:

| Protocol | Role | IP status |
|---|---|---|
| **MAP** (Mind Annotation Protocol) | Typed annotation envelope | **Patent-pending — STARGA, Inc.** |
| **MIC@2** (Mind Interchange Coding v2) | Wire-format encoder/decoder | **Patent-pending — STARGA, Inc.** |
| **binary framing** | On-the-wire framing | **Patent-pending — STARGA, Inc.** |

`mind-mem` itself is published under **Apache-2.0** on PyPI. Section 3
of the Apache license carries an explicit patent grant: any
ClinicalMem deployment automatically gets the right to use STARGA's
patent-pending MIC@2 / MAP / binary protocols **for the purpose of
running mind-mem as shipped**. This is the right scope:

- Hospitals can deploy and federate freely under Apache-2.0
- Standalone use of MIC/MAP in unrelated products requires a separate
  STARGA license
- The Apache-2.0 retaliation clause (§ 3 final sentence) protects
  mind-mem from patent-attack forks

## End-to-end flow

```
                  ┌─── SITE A (e.g. Mass General) ─────────────┐
                  │                                            │
                  │  finding ─► classify ─► phi_strip ─► sign  │
                  │                                            │
                  │                                       ▼    │
                  └────────── over MIC@2 / MAP / binary ──────┘
                                                  │
                                                  │  (patent-pending
                                                  │   STARGA transport)
                                                  ▼
                  ┌─── SITE B (e.g. Mayo Clinic) ──────────────┐
                  │                                            │
                  │  ingested ◄── ingest ◄── phi_recheck ◄── verify
                  │                                            │
                  └────────────────────────────────────────────┘
```

Every step in both directions is enforced by a typed invariant in the
`.flow.mind` contract. The contract's content-addressed `plan_hash`
(SHA-256, currently `d96173f3...31d2`) is recorded in the audit chain
for every federation event; an auditor can replay any past
inter-site exchange against the source contract bit-identically.

## Defence in depth

Five hard constraints apply at every federation event:

1. **PHI classification gate** — `classify.lane in ["clinical_knowledge", "phi_lane"]`; `phi_lane` payloads are dropped before any transport call.
2. **Independent PHI scrubber** — `phi_strip` runs the 18 HIPAA Safe Harbor identifiers across the payload; any hit blocks the emit.
3. **Per-site Ed25519 signature** — every emitted record is signed with the originating site's private key; receivers verify before ingestion.
4. **Inbound PHI re-check** — `phi_strip` runs again on the receiver before the record reaches mind-mem's local store; defence-in-depth against a misconfigured peer.
5. **Tamper-evident audit chain** — every inter-site exchange emits a TAG_v1 hash receipt; an auditor with the originator's public key can re-verify the exchange decades later.

## What's NOT in scope

- Patient-data sharing across sites — that's a clinical-network
  product (CommonWell, eHealth Exchange) and requires explicit BAA +
  patient consent.
- Real-time cross-site clinical decision support based on individual
  patient context — that's a different architecture (federated query
  vs. federated knowledge).
- Federated learning of the BitNet classifier — could be a v2
  feature using FedAvg or DP-SGD, but the v1 contract just propagates
  trained-weights bundles and per-pair classifications.

## Status

- `flows/JointMemoryFederation.flow.mind` — **shipped** (typed
  contract, 6 invariants, plan_hash `d96173f3...31d2`)
- `mind-mem` MIC@2 / MAP / binary multi-machine transport —
  **in active development** by the upstream mind-mem team
- ClinicalMem federation client wiring — pending mind-mem v3.x
  release; the contract above pins the API surface

This document tracks ClinicalMem's federation architecture; for
mind-mem's transport-layer status, see the upstream mind-mem
documentation.

---

*Apache-2.0 — STARGA, Inc.*
*MIC@2, MAP, and binary framing are patent-pending STARGA technologies.*
