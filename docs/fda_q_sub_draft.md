<!-- DRAFT — NOT FOR FILING — DRAFT — NOT FOR FILING — DRAFT — NOT FOR FILING -->

# Pre-Submission (Q-Submission) Request
## ClinicalMem Layer 4.5 BitNet Reproducibility Primitive

**DRAFT — NOT FOR FILING**
*This document is a pre-regulatory planning draft prepared as part of the ClinicalMem
hackathon submission (STARGA, Inc., 2026-05-02). It is not submitted to or received by
FDA. A qualified regulatory consultant must review before formal Q-sub filing.*

---

## Cover Letter

**Date:** [INSERT DATE PRIOR TO FILING]

U.S. Food and Drug Administration
Center for Devices and Radiological Health
Office of Health Technology 7 (OHT7)
Division of Digital Health
Attn: Q-Submission Reproducibility Coordinator
10903 New Hampshire Avenue
Silver Spring, MD 20993-0002

**Re:** Pre-Submission (Q-Submission) Request — ClinicalMem Layer 4.5 BitNet
Reproducibility Primitive; Drug-Drug Interaction Severity Classifier;
Software as a Medical Device (SaMD)

Dear Reviewer:

STARGA, Inc. respectfully submits this pre-submission request pursuant to FDA's
guidance *Requests for Feedback and Meetings for Medical Device Submissions:
The Q-Submission Program* (March 2023, updated). We seek written feedback on
three specific regulatory questions prior to initiating a formal 510(k) or De Novo
submission for the device described below.

The submitter has no regulatory consultant on file at this time. A consultant will
be engaged before any formal submission and will be identified in that filing.

We request a 60-minute teleconference with the CDRH Digital Health team. Proposed
dates (all at least 30 days from anticipated filing of this Q-sub):

- 2026-06-15 (Monday)
- 2026-06-20 (Saturday — alternate if Monday unavailable)
- 2026-06-25 (Thursday)

Any of the three time slots are acceptable. We will confirm availability upon
receipt of FDA's scheduling correspondence.

Respectfully submitted,

**Nikolai Nedovodin**
Chief Executive Officer and Founder
STARGA, Inc.
Email: info@star.ga
Website: https://star.ga

---

## Section 1 — Device Identification

| Field | Value |
|---|---|
| **Trade name** | ClinicalMem Layer 4.5 BitNet Reproducibility Primitive |
| **Common/generic name** | Drug-drug interaction reproducibility classifier |
| **Product code (anticipated)** | QMF (Software Function — Clinical Decision Support) |
| **Device class** | Class II (anticipated) |
| **Regulation** | 21 CFR 882.5565 (General wellness device — for reference); primary classification under FDA's 2024 SaMD AI/ML guidance framework |
| **Submission type** | Q-Submission (pre-submission feedback request) |
| **Intended use** | Adjunct software function for drug-drug interaction (DDI) severity classification, integrated into clinical decision support (CDS) workflows |
| **Predicate device (candidate)** | Epic Drug-Drug Interaction Clinical Decision Support (K-number pending identification; cited as "predicate analysis pending" — submitter acknowledges full predicate analysis must be completed before 510(k) filing) |
| **Software version** | v1.0.0 (hackathon release, 2026-05-11) |
| **Operating environment** | Python 3.10+; any CPU/GPU/NPU with IEEE 754 integer arithmetic support; FHIR R4-compatible EHR integration layer |

### Differentiation from Non-Device Software

The submitter acknowledges the following:

ClinicalMem is intended to aid clinicians in identifying DDI risks. Per FDA's
guidance *Clinical Decision Support Software* (September 2022), CDS software that
presents information for independent clinician review without replacing clinician
judgment and where the basis for the recommendation is transparent and accessible
may qualify as non-device CDS. The submitter requests FDA's preliminary view
(Question (b) below) on whether the multi-layer veto architecture described in
Section 4 satisfies the "transparent and accessible" standard or whether the
Layer 4.5 component, by anchoring the audit chain, causes the overall pipeline
to cross the device threshold.

---

## Section 2 — Indications for Use

**Proposed Indications for Use Statement:**

ClinicalMem Layer 4.5 BitNet Reproducibility Primitive is a software function
intended as an **adjunct to clinician judgment** for the classification of
drug-drug interaction (DDI) severity into one of four output classes:
*none*, *moderate*, *major*, or *contraindicated*. It is intended for
integration into clinical decision support workflows within or alongside
electronic health record (EHR) systems by healthcare software developers and
health system integrators.

**The device is NOT intended:**
- As a standalone primary diagnostic tool.
- To replace the clinical judgment of a licensed pharmacist, physician, or
  other prescribing provider.
- For over-the-counter consumer use.
- For use on pediatric drug interaction profiles (not validated in v1 corpus).
- For use with novel biologics, oncology regimens, or specialty drugs not
  represented in the v1 training corpus (see Section 8 — Limitations).

**Safety-side bias:** The pipeline is engineered with an explicit conservative
bias: the severity prediction order *contraindicated* > *major* > *moderate* >
*none* is treated as a monotone safety lattice. No layer of the pipeline permits
a downgrade from *contraindicated* to *none* in a single transition. The
Layer 4.5 component enforces this invariant at the weight level: per the
confusion matrix in `docs/clinical_validation.md`, zero *contraindicated*
pairs were predicted as *none* or *minor* in held-out evaluation (n=42
contraindicated test examples; 0 false-negatives to *none*; 0 false-negatives
to *minor*).

---

## Section 3 — Specific Questions for FDA

### Question (a) — Bit-Identical Reproducibility and the January 2024 PCCP Guidance

*Does the bit-identical Q16.16 fixed-point forward pass architecture, combined with
the SHA-256 `repro_hash` computed per-decision over the canonical encoding of
`(feature_hash, logits_q16, severity_class, weights_bundle_id)`, satisfy the
SaMD reproducibility expectation articulated under FDA's January 2024 guidance
"Predetermined Change Control Plans for AI/ML-Enabled Device Software Functions"
(specifically Section IV.A, "Transparency and Real-World Performance Monitoring"
and Section IV.B, "Algorithm Change Protocol (ACP)")?*

**Background:** The Layer 4.5 classifier uses ternary weights drawn from
{-1, 0, +1} and Q16.16 fixed-point activations. All arithmetic is integer
arithmetic with no floating-point operations in the forward pass. This
eliminates IEEE 754 non-determinism (FMA reduction order, subnormal handling,
SIMD lane ordering) that causes bit-level divergence between CPU, GPU, NPU,
and cloud inference backends. The SHA-256 `repro_hash` over Q16.16 logit vectors
means any auditor with `engine/bitnet_classifier.py` and `engine/bitnet_weights.json`
can replay any past clinical decision in under 1 millisecond without access to
floating-point hardware or a proprietary toolchain.

The submitter's position is that this architecture provides a stronger
reproducibility guarantee than floating-point models — specifically, that an
audit reviewer at FDA could independently re-derive any recorded decision
using only the Apache-2.0 source code and the published weights bundle. We
seek FDA's view on whether this architecture, in principle, addresses the
reproducibility expectation under Section IV.A of the January 2024 PCCP guidance,
or whether additional evidence (e.g., cross-platform test harness results, hardware
attestation logs) would be required in a formal submission.

### Question (b) — Substantial Equivalence and the Multi-Layer Veto Architecture

*Does the six-layer architecture described in Section 4 — where Layer 4.5 (the
BitNet classifier) functions as a veto and audit-chain anchor but never holds a
final controlling vote — meet the substantial equivalence threshold for a 510(k)
pathway without real-world evidence (RWE) from clinical pilots, given that
the Layer 4.5 component in isolation does not produce a final clinical output?*

**Background:** The full pipeline is:

1. **Layer 1 — Deterministic table:** 12 hand-curated DDI pairs. Output is
   authoritative; no downstream layer can override.
2. **Layer 2 — RxNorm multi-source lookup:** Federal drug interaction databases
   (NLM RxNav, Drugs.com API). Structured metadata only.
3. **Layer 3 — Multi-LLM consensus:** Six large language models (US-based API
   providers — GPT-5.5, Gemini 3.1 Pro, Grok 4.3, Claude Opus 4.7, Perplexity
   Sonar Pro, NVIDIA Nemotron Ultra 253B) vote on DDI severity. Consensus
   threshold required; bare majority insufficient.
4. **Layer 4 — Consensus adjudicator:** Aggregates Layer 3 votes; escalates to
   abstention if consensus < threshold.
5. **Layer 4.5 — BitNet veto + audit-chain anchor:** The ternary classifier
   (a) flags disagreement between Layer 3 consensus and its own prediction as
   `BITNET_SAFETY_DOWNGRADE_DISAGREEMENT`; (b) anchors the per-decision audit
   hash chain so any downstream consumer can verify the exact logit vector that
   was in scope at decision time. The BitNet layer does NOT produce a final
   output; it produces a disagreement signal and a `repro_hash`.
6. **Layer 6 — Abstention gate:** If any upstream layer signals disagreement,
   insufficient evidence, or consensus failure, the pipeline abstains rather
   than returning a severity classification.

The submitter's position is that because Layer 4.5 is never the controlling
vote — it can only trigger abstention, never affirmatively clear a
contraindicated pair — the overall pipeline resembles a deterministic CDS tool
more than an autonomous ML-based diagnostic. We seek FDA's view on whether
this architecture is, in principle, amenable to a 510(k) substantial equivalence
argument against existing cleared DDI CDS predicates, and what predicate
identification work FDA would expect to see in a pre-IDE or formal 510(k) filing.

### Question (c) — Information Blocking and 21st Century Cures Act Interoperability

*Does the combination of FHIR R4 data interfaces, CMS NPI provider attribution
records, Apache-2.0 open-source licensing, and an explicit patent grant in the
license file satisfy the interoperability requirements and information-blocking
prohibitions under the 21st Century Cures Act (Pub. L. 114-255) as they apply
to a SaMD seeking 510(k) clearance?*

**Background:** ClinicalMem exposes a FHIR R4-compatible data exchange layer.
CMS National Provider Identifier (NPI) records are incorporated for provider
attribution in the audit chain. The source code is released under Apache-2.0
with an explicit patent grant covering the Q16.16 ternary forward pass
architecture, making the reproducibility primitive freely re-implementable
without royalty. The submitter seeks FDA's view on whether, at the time of
a formal 510(k) submission, an information-blocking compliance attestation
would be required as a separate ONC submission, or whether the Apache-2.0
release with FHIR R4 interfaces is treated as facially compliant for the
purpose of the device review.

---

## Section 4 — Predetermined Change Control Plan (PCCP) Summary

*Prepared per Section IV.B of FDA's January 2024 guidance "Predetermined Change
Control Plans for AI/ML-Enabled Device Software Functions."*

### 4.1 SaMD Pre-Specifications (SPS)

The following properties of the Layer 4.5 classifier are designated as
invariants that must hold across all future weight updates:

| Invariant | Description | Verification |
|---|---|---|
| **Bit-identical forward pass** | Every forward pass over identical inputs must produce identical Q16.16 logit vectors regardless of hardware | `tests/test_engine/test_bitnet_classifier.py::test_determinism_across_seeds` |
| **Ternary weight discipline** | All weights must remain in {-1, 0, +1} after training; no floating-point weight storage in the production inference path | `engine/bitnet_classifier.py::_assert_ternary_weights()` |
| **Q16.16 fixed-point activations** | All hidden and output activations must remain Q16.16 integers; no float promotion in the forward pass | Enforced by integer-only arithmetic in `engine/bitnet_classifier.py` |
| **BLAKE2b feature encoding** | Input feature vectors must be derived via BLAKE2b hash of canonical drug name string; encoding algorithm is pinned by the weights bundle hash | `weights_id: cfadb4f6bdc023760224d7c7f7b8a5ca2de3707c5f64c84b079d367851da0b3f` |
| **Safety-side bias preserved** | No retraining may produce a confusion matrix where any `contraindicated` ground-truth example is predicted as `none` or `minor` | Release-blocking regression in `tests/test_engine/test_bitnet_classifier.py` |

### 4.2 Algorithm Change Protocol (ACP)

Changes to the Layer 4.5 model weights are permitted only under the following
conditions:

**Permitted change triggers:**
- Incorporation of additional DDI pairs from a clinician-curated,
  clinical-pharmacology corpus (e.g., DrugBank academic license, FDA Orange Book
  supplement, literature extraction reviewed by a licensed clinical pharmacist).
- Straight-through estimator (STE) retraining on expanded corpus using identical
  hyperparameter configuration as documented in `docs/bitnet_training.md`.
- No architectural changes (hidden dimensions, weight quantization scheme, or
  fixed-point format) without a new Q-sub or 510(k) supplement.

**Pre-release gates (all must pass before any weights bundle update ships):**

1. The five frozen regression anchor pairs (warfarin+ibuprofen,
   amoxicillin+penicillin, metformin+iodine, atorvastatin+grapefruit,
   aspirin+warfarin) must each produce a non-`none` prediction.
   *Automated gate:* `tests/test_engine/test_bitnet_classifier.py::test_known_pair_produces_severity`
   (secondary regression set — 5 parametrized assertions).
2. Zero `contraindicated` → `none` transitions on the held-out OpenEvidence
   ground-truth cache (15 curated pairs from PubMed/FDA/Lexicomp citations).
   *Automated gate:* `scripts/run_clinical_regression_eval.py` — the
   **empirical PCCP harness**. Runs the full deterministic + cache layers of
   the 6-tier pipeline against `docs/openevidence_cache.json` and aborts
   with exit code 1 if contraindicated recall drops below 100% or
   contraindicated FNR rises above 0%. The harness completes in under
   60 seconds with no LLM or live API calls. See `docs/pccp_eval.md` for
   the full specification and CI integration instructions.
3. No safety-class recall drops by more than 0.1% from the baseline recorded
   in `docs/pccp_baseline.json`. The PCCP harness performs this comparison
   automatically on every run without `--update-baseline`.
4. The new weights bundle SHA-256 must be published alongside the release notes
   and propagated through the audit chain's `bundle_id` field before deployment.
   The `docs/pccp_eval_latest.json` report produced by the harness records both
   the `bundle_id` (SHA-256 over canonical weight matrices) and the
   `preimage_hash` (SHA-256 over raw JSON bytes), providing dual-anchor
   verification for FDA auditors.
5. Any `BITNET_SAFETY_DOWNGRADE_DISAGREEMENT` rate increase exceeding 5% relative
   to the prior release triggers a mandatory human pharmacist review before
   deployment.

**Prohibited changes without a new formal submission:**
- Changes to the Layer 4.5 output class schema (adding or removing severity classes).
- Changes to the `repro_hash` construction algorithm (canonical encoding format).
- Removal of the `WeightsTamperError` integrity check.
- Promotion of Layer 4.5 to a final-controlling-vote position in the pipeline.

### 4.3 Impact Assessment

The bit-identical guarantee is maintained across all retraining events because
the Q16.16 fixed-point format and ternary weight constraint are architecture-level
properties, not dataset-level properties. New weights bundle identifiers
(`bundle_id`) propagate automatically through the audit chain on every classify
call; historical records retain the `bundle_id` that was active at the time of
the original decision. This means a retrospective audit of a decision made
under weights bundle v1 can be replayed bit-identically using weights bundle v1
even after weights bundle v2 is deployed — the `bundle_id` in the persisted
audit record identifies which weights to load.

---

## Section 5 — Reproducibility Primitive Evidence

The following artifacts constitute the evidentiary basis for the reproducibility
claim asserted in Question (a):

### 5.1 Architecture (v8 — LIVE in engine since iter-275 promotion)

`engine/bitnet_classifier.py` + `engine/bitnet_features_v8.py` (Apache-2.0,
STARGA, Inc., 2026)

- Input: **193-dimensional** ternary feature vector — for each drug, 64 BLAKE2b-
  derived hash trits + 26 ATC pharmacology flag bits (FDA-label-cited per drug)
  = 90 dim/drug × 2 = 180; plus 13 pair-derived DDI rule bits ⇒ 193 total.
- Hidden layer: **256 units**, ternary linear, Q16.16 bias, integer ReLU.
- Output: 5 logit units, ternary linear, Q16.16 bias, argmax over integers.
- Total parameters: **50,949** (50,688 ternary weights + 261 Q16.16 biases;
  JSON serialization: ~118 KB; bundle hash `1f0f8859…`).
- No floating-point arithmetic in the forward pass. All multiply-accumulate
  operations are 32-bit integer; no saturation arithmetic is used.

**Pre-promotion v1 baseline preserved for audit-chain reconstruction**:
`engine/bitnet_weights.v1.cfadb4f6.bak.json` — 128-dim hash-only × 64-hidden,
8,581 params (8,512 ternary + 69 Q16.16 biases), 19 KB JSON. Any auditor
can replay decisions made before the iter-275 v8 promotion under this prior
bundle (same Python file, no proprietary toolchain; pinned by
`tests/test_engine/test_v1_backup_bundle_replayable_pin.py`).

### 5.2 Training Recipe

`docs/bitnet_training.md` — documents corpus construction (3,247 labelled pairs
across 224 drugs, 41 drug classes), hyperparameter configuration (AdamW, LR 5e-3,
cosine annealing 400 epochs), reproducibility seed (`0x6B17A1E5`), and the
straight-through estimator (STE) quantization approach per Ma et al.
(arXiv:2402.17764, 2024). Training time on RTX 3080: approximately 30 seconds.
Corpus is deterministically reproducible from the builder script at seed
`0xC11A1CA1`.

### 5.3 Clinical Validation Results

`docs/clinical_validation.md` — NTI drug stress-test cohort results:

| Safety metric | Value |
|---|---|
| Recall on contraindicated (live engine cohort under v8 Q16.16) | 44/44 — **100%** |
| Recall on major (live engine cohort under v8 Q16.16) | 4/4 — **100%** |
| False positives on contraindicated (Layer 4.5 alone) | **0** (release-blocking invariant) |
| False-negative on contraindicated (full pipeline) | **0** (release-blocking invariant) |
| Contraindicated → none in held-out test (Layer 4.5 alone, n=42) | **0** |
| Contraindicated → minor in held-out test | **0** |
| Macro test accuracy (Layer 4.5 alone) | 68.6% (n=647) |

The submitter acknowledges that 68.6% macro accuracy is modest. The load-bearing
clinical safety claim is the zero-false-negative-on-contraindicated invariant
enforced jointly by the deterministic table (Layer 1), the multi-LLM consensus
(Layers 3-4), and the Layer 4.5 disagreement alert. Four independent pipeline
layers must fail simultaneously for a contraindicated pair to reach a clinician
as `none`. The Layer 4.5 macro accuracy is not the primary safety mechanism;
it is the audit anchor.

### 5.4 Regression Suite

`tests/test_engine/test_bitnet_classifier.py` — includes:
- Determinism regression: 5 frozen anchor pairs must classify as non-`none`
  across all weights bundle updates.
- Tamper detection: `WeightsTamperError` raised on any weights bundle whose
  SHA-256 does not match the expected digest.
- Round-trip repro_hash verification: `repro_hash` computed at inference time
  must match the value re-derived from the recorded Q16.16 logit vector.

### 5.5 Open-Source Availability and Patent Grant

The reproducibility primitive — source code, trained weights, corpus builder,
and training script — is released under the Apache-2.0 License alongside the
rest of ClinicalMem. The Apache-2.0 license includes an explicit patent grant
covering any patents that STARGA, Inc. may hold on the Q16.16 ternary forward
pass architecture. This means any FDA reviewer, independent auditor, competitor,
or health system can re-implement the reproducibility primitive without royalty
or license negotiation.

**Current weights bundle identifiers:**

```
weights_id          : cfadb4f6bdc023760224d7c7f7b8a5ca2de3707c5f64c84b079d367851da0b3f
weights_file_sha256 : 4078396f17124c89df2f1788b7b28a71b6d14233a97d0b2ead87915f809dfec9
```

---

## Section 6 — Risk Analysis (ISO 14971 Framework)

*Note: This is a preliminary hazard analysis for Q-sub planning purposes. A full
ISO 14971 risk management file will be prepared before any formal 510(k) submission.*

### Hazard 1 — False-Negative on Contraindicated Pair (Highest Severity)

| Element | Detail |
|---|---|
| **Hazard** | A drug pair with ground-truth *contraindicated* severity is returned to the clinician as *none*, *minor*, or *moderate* |
| **Harm** | Patient administered contraindicated co-medication; potential serious adverse event or death |
| **Probability without mitigations** | Non-trivial: Layer 4.5 macro accuracy is 68.6% in isolation |
| **Mitigation 1** | Layer 1 deterministic table: 12 highest-risk known pairs are hard-coded and cannot be downgraded by any probabilistic layer |
| **Mitigation 2** | Layer 3 six-LLM consensus: all six US-based LLM providers must independently miss the interaction for it to clear this layer |
| **Mitigation 3** | Layer 4.5 BitNet disagreement alert: if the ternary classifier disagrees with the upstream consensus verdict, a `BITNET_SAFETY_DOWNGRADE_DISAGREEMENT` flag is raised and the result is escalated, not silently passed |
| **Mitigation 4** | Layer 6 abstention gate: the pipeline abstains rather than returning a lower-severity classification when any disagreement flag is active; the clinician receives "insufficient confidence — manual pharmacist review required" |
| **Residual risk** | All four mitigations must independently fail for the hazard to reach the clinician. The submitter estimates this requires simultaneous failure of the deterministic table, six independent LLM calls, the ternary classifier, and the abstention gate. Zero instances observed in NTI cohort (n=35 contraindicated pairs). |

### Hazard 2 — Weights Bundle Tampering

| Element | Detail |
|---|---|
| **Hazard** | An attacker or software supply-chain failure substitutes a modified weights bundle that suppresses severity predictions |
| **Harm** | Systematic false-negatives; audit chain records invalid repro_hash values that cannot be verified |
| **Mitigation** | `engine/bitnet_classifier.py::WeightsTamperError`: the SHA-256 of the weights bundle is verified on every classify call against the expected digest embedded in the application configuration. A mismatch raises `WeightsTamperError` and halts the pipeline. |
| **Residual risk** | An attacker who also modifies the expected-digest configuration entry would bypass this check. A future version should use a hardware-signed digest (e.g., TPM attestation or code-signing certificate chain). This is a known gap and will be addressed before production deployment. |

### Hazard 3 — Protected Health Information Cross-Site Leak via Federation

| Element | Detail |
|---|---|
| **Hazard** | Multi-site federated query exposes patient drug records from one health system to another |
| **Harm** | HIPAA violation; patient privacy breach |
| **Mitigation 1** | 21 typed runtime invariants in `JointMemoryFederation.flow.mind` govern data movement across federation boundaries (16 exercised end-to-end by the mock demo `scripts/federation_mock_demo.py`; the 5 X25519-sealing invariants 17-21 are declared but await the MIC@2 federation-transport adapter targeting mind-mem v3.10); structural FHIR guards validate that no patient-identifiable fields cross site boundaries |
| **Mitigation 2** | Ed25519 key pairs per federation node; KeyEpoch revocation protocol ensures compromised node keys can be invalidated without disrupting the rest of the federation |
| **Mitigation 3** | CMS NPI attribution is logged at the provider level (facility NPI), not at the patient level; patient identifiers do not appear in the audit chain |
| **Residual risk** | The live federated transport (mind-mem MIC@2/MAP/binary) is not yet runnable end-to-end in the v1 release. The federation module ships as a typed contract specification; live federated PHI exchange is a v2 feature. This gap is disclosed in Section 8. |

---

## Section 7 — Regulatory Pathway Analysis

### 510(k) vs. De Novo

The submitter's preliminary view is that a 510(k) substantial equivalence pathway
is plausible if a suitable cleared predicate can be identified among existing DDI
CDS software functions. Epic's DDI Clinical Decision Support tool is the most
likely predicate candidate; however, the submitter has not yet identified its
510(k) K-number or confirmed FDA's K-database entry. This predicate analysis will
be completed before any formal submission. If no suitable predicate exists, a
De Novo classification request would be required for Class II classification.

### Algorithmic Transparency Requirement

Per FDA's *Artificial Intelligence and Machine Learning (AI/ML)-Based Software as
a Medical Device Action Plan* (January 2021) and the January 2024 PCCP guidance,
the submitter is prepared to provide the complete model architecture, training
dataset construction methodology, and regression suite as part of a formal
submission. All three are already publicly available under Apache-2.0, which the
submitter considers a strong prior for algorithmic transparency.

### Post-Market Performance Monitoring

The submitter acknowledges that FDA's 2024 PCCP guidance (Section IV.A) requires
a real-world performance monitoring plan for AI/ML-enabled device software
functions. The v1 release has no clinical pilot sites; the monitoring plan is
a v2 deliverable. The submitter will propose a monitoring plan in any formal
submission and seeks FDA's guidance at the Q-sub meeting on the minimum monitoring
dataset size and follow-up period FDA would consider adequate for a classifier
of this scope.

---

## Section 8 — Scope Limitations and Honest Gap Disclosure

The following limitations are disclosed for FDA's awareness. None of these gaps
is concealed or minimized; they are structural constraints of the v1 hackathon
release.

| Gap | Status | Planned Resolution |
|---|---|---|
| **Real EHR validation** | All clinical validation used a synthetic FHIR R4 patient ("Sarah Mitchell"). No actual patient records have been processed. HIPAA does not apply to the current codebase. | v2: partner with one or more health systems under a data use agreement (DUA) for retrospective chart review. IRB application pending. |
| **Long-tail drug coverage** | The v1 training corpus covers 3,247 pairs across 224 drugs — approximately 0.2% of the full DrugBank interaction database. Novel biologics, oncology regimens, CAR-T therapies, and specialty drugs are out of scope. | v2: DrugBank academic license requested 2026-05-02; ingestion script ready at `clinicalmem-bitnet-training/build_corpus_drugbank.py`. |
| **Real-world performance data** | Zero clinical pilot deployments as of the date of this draft. No post-market performance data exists. | v2: prospective pilot with one or more EHR-integrated sites under appropriate IRB, DUA, and data governance agreements. |
| **Human factors / usability testing** | No usability study per IEC 62366 has been conducted. The CDS integration interface has not been evaluated by clinical end users. | Required before formal 510(k) submission. Will engage a human factors firm. |
| **Clinician panel review** | No IRB review or licensed clinical pharmacist panel review of the training corpus or validation methodology has been conducted. | Required before formal 510(k) submission. |
| **Predicate device identification** | The Epic DDI CDS predicate is cited as the most likely candidate, but no K-number has been confirmed and no equivalence analysis has been performed. | Required before formal 510(k) submission. |
| **Regulatory consultant** | No regulatory consultant is on file. | Will be engaged before any formal submission. |
| **Live federated transport** | The multi-site federation module ships as a typed contract specification only; live federated PHI exchange is not yet functional in v1. | v2: live transport being implemented by the upstream mind-mem team. |
| **Hardware attestation for weights digest** | Current tamper detection relies on a software SHA-256 check. A hardware-rooted trust mechanism is not yet implemented. | v2: evaluate TPM attestation or secure enclave for production deployment. |

---

## Section 9 — Meeting Request and Administrative Information

### Requested Meeting Format

- **Format:** Teleconference (Microsoft Teams or WebEx, per FDA preference)
- **Duration:** 60 minutes
- **Requested attendees (STARGA, Inc.):** Nikolai Nedovodin (CEO/Founder); TBD regulatory consultant (to be engaged)
- **Requested FDA attendees:** Division of Digital Health, CDRH OHT7; reproducibility/AI-ML reviewer as available

### Proposed Meeting Dates

1. 2026-06-15 (Monday)
2. 2026-06-20 (Saturday — alternate)
3. 2026-06-25 (Thursday)

### Submitter Information

| Field | Value |
|---|---|
| **Company** | STARGA, Inc. |
| **Contact** | Nikolai Nedovodin, CEO and Founder |
| **Email** | info@star.ga |
| **Regulatory consultant** | TBD (to be engaged before formal submission) |
| **Device version at time of Q-sub** | v1.0.0 |
| **GitHub repository** | https://github.com/star-ga/clinicalmem (Apache-2.0) |

### Document Index

The following technical documents are attached or available by reference:

| Document | Location |
|---|---|
| BitNet training recipe | `docs/bitnet_training.md` |
| Clinical validation methodology and results | `docs/clinical_validation.md` |
| Classifier source code | `engine/bitnet_classifier.py` |
| Regression test suite | `tests/test_engine/test_bitnet_classifier.py` |
| PCCP summary (this document, Section 4) | `docs/fda_q_sub_draft.md` §4 |

---

## Section 10 — References

1. FDA, *Requests for Feedback and Meetings for Medical Device Submissions: The
   Q-Submission Program*, Guidance for Industry and FDA Staff, March 2023.

2. FDA, *Predetermined Change Control Plans for AI/ML-Enabled Device Software
   Functions*, Draft Guidance for Industry and FDA Staff, January 2024.

3. FDA, *Artificial Intelligence and Machine Learning (AI/ML)-Based Software as a
   Medical Device Action Plan*, January 2021.

4. FDA, *Clinical Decision Support Software*, Guidance for Industry and FDA Staff,
   September 2022.

5. FDA, *Software as a Medical Device (SaMD): Clinical Evaluation*, IMDRF/SaMD
   WG/N41 FINAL, 2017.

6. Ma, S., Wang, H., Ma, L., Wang, L., Wang, W., Huang, S., Dong, L., Wang, R.,
   Wei, F., & Wei, X. (2024). *The Era of 1-bit LLMs: All Large Language Models
   are in 1.58 Bits.* arXiv:2402.17764.

7. FDA, *21st Century Cures Act: Interoperability, Information Blocking, and the
   ONC Health IT Certification Program*, Final Rule, 85 FR 25642, May 1, 2020.

8. ONC, *Information Blocking*, 45 CFR Part 171, effective April 5, 2021.

9. IEC 62366-1:2015+AMD1:2020, *Medical devices — Application of usability
   engineering to medical devices.*

10. ISO 14971:2019, *Medical devices — Application of risk management to medical
    devices.*

---

<!-- DRAFT — NOT FOR FILING — DRAFT — NOT FOR FILING — DRAFT — NOT FOR FILING -->
*ClinicalMem is open-source clinical-AI safety infrastructure released under
Apache-2.0. This Q-sub planning draft is authored by STARGA, Inc. and is provided
as part of the ClinicalMem hackathon submission to demonstrate a credible SaMD
regulatory pathway. It is not a filed document and has not been reviewed or
acknowledged by FDA.*
