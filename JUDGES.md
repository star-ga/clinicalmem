# For Hackathon Judges — Read This in 60 Seconds

> *Agents Assemble: Healthcare AI Endgame · DevPost · 2026-05-11 deadline*
> *Track: Clinical AI safety primitive*

ClinicalMem is a **bit-identical clinical decision verifier**. Every
drug-pair safety verdict carries a Q16.16 fixed-point reproduction
hash that an FDA auditor can replay decades later, on a $15 Pi Zero
or an A100 GPU, and obtain the same answer to the bit.

If you have 60 seconds, read the 5 bullets below.
If you have 5 minutes, run the 4 commands in [§ 5-minute audit](#5-minute-audit).
If you have an hour, follow [§ Full audit trail](#full-audit-trail).

---

## The 5-bullet pitch

1. **6-layer clinical safety pipeline** with a Q16.16 ternary anchor
   at Layer 4.5. Layer 4.5's job is *deterministic verification*, not
   headline accuracy: same SHA-256 `repro_hash` on any chip in
   healthcare. Bundle is 8,581 parameters (8,512 ternary weights + 69 Q16.16 biases) / 19 KB.
2. **Recall gate (PCCP):** **100% recall** on the three severity
   classes represented in the 105-pair OpenEvidence-cited cohort —
   contraindicated (17/17), serious (66/66), moderate (22/22). The
   `major` class is empty in this cohort by design (it's tested
   separately in the 35-pair NTI cohort, see
   `docs/clinical_validation.md`). Every pair is evidence-backed
   (FDA labels + ACC/AHA + EULAR + Beers + KDIGO + ESC + PubMed
   primaries).
3. **Precision gate:** **0 / 8 false positives** on a clinical-pharmacology
   negative-control cohort that includes two CYP-pathway boundary
   cases (clopidogrel + pantoprazole, atorvastatin + amlodipine).
4. **PHI never leaves the building.** The PHI / non-PHI boundary is
   a **typed runtime invariant** in
   `flows/JointMemoryFederation.flow.mind`, not a policy doc. 21
   typed invariants enforce PHI gates + Ed25519 signing + X25519 +
   ChaCha20-Poly1305 sealing + severity-quorum gating + KeyEpoch
   revocation. The control plane is LIVE against `mind-mem v3.8.14`'s
   `MemoryMesh` + `EventFanout`.
5. **Apache-2.0 with explicit § 3 patent grant** — deploy in
   production tomorrow. The patent-pending MIC@2 / MAP / binary
   wire formats ride inside `mind-mem` (also Apache-2.0) — the grant
   transitively applies.

---

## 5-minute audit

The fastest path is a **single command** that runs all four gates
in sequence and prints a unified PASS/FAIL summary:

```bash
python3 scripts/run_all_gates.py
```

Expected output (≈ 1 second total):

```
[  PASS  ] PCCP recall                          PCCP GATE: PASS — all safety-class gates satisfied
[  PASS  ] Negative-control precision           PRECISION GATE: PASS — no false positives.
[  PASS  ] Federation 16-invariant demo         FEDERATION DEMO COMPLETE — exit 0
[  PASS  ] arch-mind L1 governance              OK: every rule passed.
ALL-GATES RESULT: PASS
```

If you'd rather run each gate individually:

```bash
# 1. PCCP recall gate — verifies 100% recall on contraindicated /
#    major / serious / moderate against the 105-pair OpenEvidence-cited
#    cohort.
python3 scripts/run_clinical_regression_eval.py

# 2. Negative-control precision gate — verifies 0 / 8 false positives
#    on the negative-control cohort (6 clean negatives + 2 CYP-pathway
#    boundary cases).
python3 scripts/run_negative_control_eval.py

# 3. Federation 16-invariant flow demo — both site audit-chain hashes
#    match (proves bit-identical canonical encoding).
python3 scripts/federation_mock_demo.py

# 4. arch-mind L1 governance gate — walks engine/*.py with ast,
#    scores 8 architectural kernels, enforces the rules profile.
python3 scripts/run_arch_mind_gate.py
```

Optional: full pytest suite — `python3 -m pytest tests/test_engine/ tests/test_scripts/ -q`. Should report **820+ passed**.

---

## Full audit trail

Every visible figure on the dashboard traces back to a checked-in
artifact. Audit map:

| Claim on the dashboard | Source of truth |
|---|---|
| `100% recall · contraindicated · 16/16` | `docs/openevidence_cache.json` (104 entries) → `docs/pccp_eval_latest.json` (per-pair verdicts) |
| `0 / 8 FP · precision = 1.0` | `docs/negative_control_cohort.json` (8 entries) → `scripts/run_negative_control_eval.py --json` |
| `8,512 ternary weights + 69 Q16.16 biases = 8,581 params / 19 KB` + bundle hash `cfadb4f6…` | `engine/bitnet_weights.json` + `engine/bitnet_classifier.py`. Pinned by `tests/test_engine/test_bitnet_param_count_pin.py`. |
| `21 typed federation invariants` | `flows/JointMemoryFederation.flow.mind` (plan_hash recorded in audit chain) |
| `Federation control plane LIVE — mind-mem v3.8.14 MemoryMesh` | `engine/federation_transport.py` (9 unit tests) + `mind_mem.memory_mesh.MemoryMesh` |
| `arch-mind 9 / 9 rules` | `docs/arch_mind/clinicalmem_rules.mind` + `docs/arch_mind/clinicalmem.scan.json` (run via `scripts/run_arch_mind_gate.py`) |
| `14 synthetic patients · 28 NPIs` | `docs/synthea_demo_cohort.json` (FHIR R4 bundle, all NPIs Luhn-valid). **Per-patient drug-pair → cache-entry traceability matrix:** `docs/cohort_coverage_matrix.md` (regenerate with `scripts/build_cohort_coverage.py`). |
| `21 CFR Part 11 audit export` | `engine/audit_export_part11.py` (30 tests) |
| `Apache-2.0 + patent grant` | `LICENSE` (top of repo) |
| `IRB-exempt synthetic cohort` | `docs/irb_exemption.md` (45 C.F.R. § 46.102(e)(1)) |
| `Clinical Advisor: Dr. Ludmila Afonicheva, MD` | `docs/clinical_validation.md` (US-licensed Family Medicine, NPI 1932159530) |

---

## Honest limitations (what is NOT claimed)

1. **No live EHR pilot.** ClinicalMem ingests synthetic FHIR R4
   bundles. A real EHR integration (Epic / Cerner / MEDITECH SMART-on-FHIR)
   has not been performed; the FHIR adapter is implementation-complete
   but not deployed.
2. **No FDA SaMD filing yet.** The Q16.16 reproducibility primitive,
   PCCP regression gate, and 21 CFR Part 11 audit-export module
   together form a credible *FDA-SaMD-ready* surface, but no
   submission to the FDA has been made.
3. **Federation wire transport is mock.** The control plane (peer
   registry, sync scopes, conflict resolution policy, sync audit log,
   governance pub/sub) is LIVE against `mind-mem v3.8.14`. The
   actual cross-machine bytes-on-wire still go through a single-process
   queue in `scripts/federation_mock_demo.py`. The mind-mem v3.9
   transport adapter (HTTP / gRPC over MIC@2) is not yet released.
4. **5-LLM consensus surface (Layer 4) is implementation-only.** The
   API-key-required path is wired but the eval ran against the
   deterministic Layer 1 / Layer 2 cache / Layer 4.5 BitNet path.
   Multi-LLM consensus rounds are out of scope of the precision +
   recall gates above.

---

## One-line architecture

`EHR (FHIR R4)` → `engine.fhir_adapter` → `Layer 1 table` → `Layer 2 OpenEvidence` → `Layer 3 RxNorm DDI` → `Layer 4 5-LLM consensus` → **`Layer 4.5 BitNet b1.58 (Q16.16, bit-identical)`** → `Layer 5 LLM synthesis` → `Layer 6 abstention gate` → `mind-mem store + 21 CFR Part 11 audit chain` → `MCP (18 tools) + A2A (13 skills)`.

For deeper architecture see `docs/architecture.md`. For the federation
control-plane integration see `docs/architecture.md § Mock vs. live transport`
and `docs/arch_mind_federation_audit.md`.

---

## Contact

- Repo: <https://github.com/star-ga/clinicalmem> (PUBLIC on submission day 2026-05-11; PRIVATE before that)
- Engineering: STARGA Inc. — `noreply@star.ga`
- Clinical Advisor: Dr. Ludmila Afonicheva, MD — Family Medicine, US-licensed (NPI on file in `docs/clinical_validation.md`)

*Apache-2.0 — STARGA, Inc. — 2026.*
