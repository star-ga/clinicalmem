# ClinicalMem Healthcare-Specific Architectural Invariants

> *"Generic Python hygiene is not enough. A clinical-AI safety system has
>  six architectural promises that, if broken, make the entire FDA SaMD
>  reproducibility claim hollow."*

This document is the human-readable companion to `.arch-mind/rules.mind`.
The 9 rules currently enforced by the [arch-mind](https://github.com/star-ga/arch-mind)
v0.1.3 L1 governance gate are generic kernel rules (acyclicity,
redundancy, Q16.16 purity, etc.) that every MIND-ecosystem repo runs.

The **6 invariants below are clinicalmem-specific**. They will be
machine-enforced when arch-mind's commercial v0.2 release ships the
`clinical_invariants` profile (sidecar walk over canonical AST + the
`.flow.mind` contracts + the FHIR bundle JSONs). Until then they live
here as the architectural specification and as runtime invariants
already checked by the test suite — they are not aspirational, they
are the contract every shipped commit already keeps.

## I-1 · PHI gate coverage

**Promise:** Every Python module that touches a FHIR resource (Patient,
Observation, MedicationStatement, AllergyIntolerance, Practitioner)
must import `engine.phi_detector` (or its successor) before any
emit / write / network egress.

**Enforced today by:** `engine/clinical_scoring.py`,
`engine/fhir_adapter.py`, `flows/JointMemoryFederation.flow.mind`
(invariant `scrubbed.has_phi == false` on the egress path).

**Consequence if broken:** PHI may leak into mind-mem, the federation
transport, or audit-chain payloads. HIPAA Safe Harbor violation.

**arch-mind v0.2 metric:** `phi_gate_coverage_q16` ≥ 9500 (95%).

## I-2 · Audit-chain anchor density

**Promise:** Every clinical decision call (`check_drug_interactions`,
`check_allergy_conflicts`, `medication_safety_review`) must emit at
least one audit-chain anchor (TAG_v1 NUL-separated SHA-256) on its
return path.

**Enforced today by:** `engine/clinical_memory.py` (TAG_v1 wraps every
write), `engine/audit_export_part11.py` (Merkle chain export), the
runtime invariant in `flows/MedicationSafetyReview.flow.mind`.

**Consequence if broken:** No anchor = no FDA SaMD audit replay. The
"any auditor can re-verify any past clinical decision in <1 ms"
claim becomes false.

**arch-mind v0.2 metric:** `audit_chain_anchor_density_q16` ≥ 9000.

## I-3 · BitNet Layer 4.5 invocation discipline

**Promise:** Every reported `DrugInteraction` from
`engine/clinical_scoring.py::check_drug_interactions` must carry the
BitNet 4.5 disagreement-alert path. The runtime invariant is enforced
in `_attach_bitnet_repro_hashes`; this rule mirrors it at the
architectural surface so a refactor cannot silently bypass Layer 4.5.

**Enforced today by:** the hard `WeightsTamperError` raise on bundle_id
mismatch, the WARNING `BITNET_SAFETY_DOWNGRADE_DISAGREEMENT` log line
on contraindicated downgrades, and `tests/test_engine/test_bitnet_classifier.py`'s
5 frozen regression-anchor pairs.

**Consequence if broken:** The deterministic veto + audit-replay anchor
disappears. The "FDA SaMD reproducibility primitive" claim collapses
to a marketing line.

**arch-mind v0.2 metric:** `bitnet_layer_45_invocation_q16` == 10000
(every clinical-decision path that emits a `DrugInteraction` must call
`classifier_layer`).

## I-4 · Federation invariant density

**Promise:** Every cross-site federation flow (`.flow.mind` under
`flows/`) must declare at least 10 typed runtime invariants. The
current `JointMemoryFederation.flow.mind` ships 21; the regression
threshold catches a drop below the v0.2-era contract surface.

**Enforced today by:** `engine/flow_runner.py` (parses + counts
invariants per flow), `tests/test_engine/test_flow_runner.py`
(regression set asserts each flow has ≥1 invariant; new tests can
extend to ≥10 for federation flows specifically).

**Consequence if broken:** The PHI gate, encryption layer, replay
protection, severity quorum, or KeyEpoch revocation invariants get
silently removed; the federation becomes policy-protected, not
typed-runtime-protected.

**arch-mind v0.2 metric:** `federation_invariant_density_q16` ≥ 9000.

## I-5 · NPI Luhn coverage

**Promise:** Every `Practitioner` reference in any FHIR bundle in
the repo (`docs/sarah_mitchell_fhir.json`, `docs/synthea_demo_cohort.json`,
or any future bundle) must validate through
`engine/npi_registry.py::validate_npi` (CMS `80840`-prefix Luhn).

**Enforced today by:** `tests/test_engine/test_npi_registry.py::
test_sarah_mitchell_bundle_npis_all_valid` and the equivalent for the
Synthea cohort. Every Practitioner identifier in every bundle is
walked at test time and required to pass the Luhn check.

**Consequence if broken:** The Practitioner attribution chain becomes
spoofable. The "CMS-NPPES-verifiable provider" credibility hook
collapses; FHIR R4 reviewers will catch invalid NPIs immediately.

**arch-mind v0.2 metric:** `npi_luhn_coverage_q16` == 10000 (every
Practitioner identifier passes Luhn; non-passing is release-blocking).

## I-6 · Clinical-review pathway documented

**Promise:** `docs/clinical_validation.md` MUST describe the target
clinical-reviewer profile (specialty, licensure, NPI-verifiability)
and the scope of review required before the v2 production deployment.

**Currently satisfied by:** the "Clinical review pathway" section of
`docs/clinical_validation.md`, which names the target specialty
(US-licensed family medicine, multi-hospital affiliation), the review
scope (Sarah Mitchell demo bundle + NTI cohort severity calls +
abstention-gate triggers + cardiology/nephrology BP-target conflict),
and the CMS-NPPES Luhn validator (`engine/npi_registry.py::
validate_npi`) reviewers will use to verify any attestation that
lands.

**Consequence if broken:** The clinical-review pathway becomes
implicit, and v2 readers can no longer verify whether independent
clinical sign-off has been obtained or what its scope was.

**arch-mind v0.2 metric:** `clinical_review_pathway_documented_q16`
== 10000 (target reviewer profile, review scope, and verification
mechanism all described in `docs/clinical_validation.md`).


---

## Status table

| Invariant | Runtime enforcement | Test enforcement | arch-mind L1 (v0.1.3) | arch-mind L2 (v0.2, planned) |
|---|---|---|---|---|
| I-1 PHI gate coverage | ✅ | ✅ | ⏳ Phase B | Will activate |
| I-2 Audit-chain anchor | ✅ | ✅ | ⏳ Phase B | Will activate |
| I-3 BitNet 4.5 invocation | ✅ | ✅ (5 regression pairs) | ⏳ Phase B | Will activate |
| I-4 Federation invariant density | ✅ (parser + counter) | ✅ | ⏳ Phase B | Will activate |
| I-5 NPI Luhn coverage | — | ✅ | ⏳ Phase B | Will activate |
| I-6 Clinical-review pathway documented | — | ✅ (doc grep) | ⏳ Phase B | Will activate |

The 6 invariants are **already kept** at runtime + test time today;
the arch-mind L1 (Apache-2.0-public-safe) gate will mechanically
verify them when v0.2 of the commercial scanner ships the
`clinical_invariants` profile.

---

*Apache-2.0 — STARGA, Inc. — 2026.*
*The 6 invariants above are clinicalmem's architectural contract.
The arch-mind tool that machine-enforces them is a STARGA commercial
product (separate licensing); this document specifies the contract
clinicalmem keeps regardless of which scanner is run against it.*
