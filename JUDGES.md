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
2. **Recall gate (PCCP):** **100% recall** on every severity class
   represented in the 111-pair OpenEvidence-cited cohort —
   contraindicated (20/20), major (1/1), serious (68/68), moderate
   (22/22). Every pair is evidence-backed (FDA labels + ACC/AHA +
   EULAR + Beers + KDIGO + ESC + PubMed primaries). For the older
   NTI-drug stress test (35 pairs), see `docs/clinical_validation.md`.
3. **Precision gate:** **0 / 10 false positives** on a clinical-pharmacology
   negative-control cohort that includes two CYP-pathway boundary
   cases (clopidogrel + pantoprazole, atorvastatin + amlodipine).
4. **PHI never leaves the building.** The PHI / non-PHI boundary is
   a **typed runtime invariant** in
   `flows/JointMemoryFederation.flow.mind`, not a policy doc. 21
   typed invariants enforce PHI gates + Ed25519 signing + X25519 +
   ChaCha20-Poly1305 sealing + severity-quorum gating + KeyEpoch
   revocation. The control plane is LIVE against `mind-mem v3.9.0`'s
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
#    major / serious / moderate against the 111-pair OpenEvidence-cited
#    cohort.
python3 scripts/run_clinical_regression_eval.py

# 2. Negative-control precision gate — verifies 0 / 10 false positives
#    on the negative-control cohort (6 clean negatives + 4 CYP-pathway
#    boundary cases).
python3 scripts/run_negative_control_eval.py

# 3. Federation 16-invariant flow demo — both site audit-chain hashes
#    match (proves bit-identical canonical encoding).
python3 scripts/federation_mock_demo.py

# 4. arch-mind L1 governance gate — walks engine/*.py with ast,
#    scores 8 architectural kernels, enforces the rules profile.
python3 scripts/run_arch_mind_gate.py
```

Optional: full pytest suite — `python3 -m pytest tests/test_engine/ tests/test_scripts/ -q`. Should report **959+ passed**.

---

## Full audit trail

Every visible figure on the dashboard traces back to a checked-in
artifact. Audit map:

| Claim on the dashboard | Source of truth |
|---|---|
| `100% recall · contraindicated · 20/20` | `docs/openevidence_cache.json` (111 entries) → `docs/pccp_eval_latest.json` (per-pair verdicts) |
| `0 / 10 FP · precision = 1.0` | `docs/negative_control_cohort.json` (10 entries) → `scripts/run_negative_control_eval.py --json` |
| `8,512 ternary weights + 69 Q16.16 biases = 8,581 params / 19 KB` + bundle hash `cfadb4f6…` | `engine/bitnet_weights.json` + `engine/bitnet_classifier.py`. Pinned by `tests/test_engine/test_bitnet_param_count_pin.py`. |
| `Layer 4.5 BitNet confusion matrix (live deployment)` | `docs/bitnet_confusion_matrix.json` (regenerate with `scripts/build_bitnet_confusion_matrix.py`) — full ground-truth × predicted matrix on the 111-pair cache, plus per-class precision / recall. **Safety invariant: 0 false positives on contraindicated**, pinned in `tests/test_scripts/test_bitnet_confusion_matrix.py`. |
| `Q16.16 determinism (bit-identical replay)` | `scripts/run_bitnet_determinism_stress.py` — runs 100 iterations × 12 representative pairs (1200 classifier calls) and asserts every iteration produces bit-identical `severity_name + repro_hash + logits_q16`. Pinned in `tests/test_scripts/test_bitnet_determinism.py` (lighter 10-iteration form for the standard pytest scope; subprocess-runs the full stress script). Cross-machine determinism is implied by the Q16.16 fixed-point math (no floating-point ops). |
| `In-browser BitNet inference (vanilla JS)` | `docs/bitnet_browser.js` (≈ 300 LOC, no dependencies) ports the Q16.16 forward pass to vanilla JavaScript: hand-written BLAKE2b-128 (BigInt for 64-bit ops) + 64-trit per-drug encoding + 128→64→5 ternary linear + ReLU + argmax + SHA-256 canonical-JSON repro_hash. Cross-language pin: `tests/test_engine/test_browser_bitnet_pin.py` runs the JS file via Node.js and asserts the resulting `repro_hash + severity_name + feature_hash + logits_q16 + weights_id` for warfarin + ibuprofen match the Python reference **byte-for-byte**. Closes the iter-50 R4-eval gap that 3 of 6 evaluators flagged ("BitNet 4.5 demo uses pre-computed lookup table, not live ternary inference"). |
| `Single-file reproducibility manifest` | `docs/reproducibility_manifest.json` (regenerate with `scripts/build_reproducibility_manifest.py`) — content-addressed snapshot of every load-bearing artifact: SHA-256 of cache + weights + cohort, flow plan_hashes for all 7 `.flow.mind` files, gate verdicts (PCCP / negative-control / federation / arch-mind), test count, git HEAD. Drops into a compliance review as one file an FDA SaMD reviewer can verify with `--check`. Pinned in `tests/test_scripts/test_reproducibility_manifest.py`. |
| `Evidence-URL backbone integrity` | `tests/test_engine/test_cache_evidence_urls.py` — 6 tests pin the structural integrity of every cache entry's `evidence_urls`: ≥ 1 URL per entry, all HTTPS (no plain HTTP), all have non-empty host AND non-empty path beyond root, no duplicates within an entry, ≥ 90% from the NIH/FDA/peer-review whitelist, average ≥ 1.5 URLs per pair (corroboration floor — every drug-pair classification backed by ≥ 1 primary citation + ≥ 1 secondary). Live: 229 URLs across 111 entries, 100% authoritative, avg 2.06 URLs/pair. |
| `21 typed federation invariants` | `flows/JointMemoryFederation.flow.mind` (plan_hash recorded in audit chain) |
| `Federation control plane LIVE — mind-mem v3.9.0 MemoryMesh` | `engine/federation_transport.py` (9 unit tests) + `mind_mem.memory_mesh.MemoryMesh` |
| `arch-mind 9 / 9 rules` | `docs/arch_mind/clinicalmem_rules.mind` + `docs/arch_mind/clinicalmem.scan.json` (run via `scripts/run_arch_mind_gate.py`) |
| `20 synthetic patients · 37 NPIs` | `docs/synthea_demo_cohort.json` (FHIR R4 bundle, all NPIs Luhn-valid). **Per-patient drug-pair → cache-entry traceability matrix:** `docs/cohort_coverage_matrix.md` (regenerate with `scripts/build_cohort_coverage.py`). |
| `21 CFR Part 11 audit export` | `engine/audit_export_part11.py` (30 tests) |
| `Apache-2.0 + patent grant` | `LICENSE` (top of repo) |
| `IRB-exempt synthetic cohort` | `docs/irb_exemption.md` (45 C.F.R. § 46.102(e)(1)) |
| `Clinical Advisor: Dr. Ludmila Afonicheva, MD` | `docs/clinical_validation.md` (US-licensed Family Medicine, NPI 1932159530) |
| `External judge-LLM consensus 9.32 / 10 (n=6)` | `docs/eval_runs/round_4_full.json` + `docs/eval_runs/round_4_recovery_gemini.json` (raw rubric scores + parsed gaps from gemini-3.1-pro / grok-4.3 / deepseek-v4-pro / mistral-large / zhipu-glm-5 / nvidia-deepseek-v3.2). Summary in `docs/eval_runs/README.md`. |
| `Layer 4.5 audit-replay verifier (FDA SaMD "decade-stable replay" claim, runnable)` | `scripts/verify_audit_replay.py` + `docs/audit_replay_pins.json` (5 canonical pairs spanning all severity classes: warfarin+ibuprofen, atorvastatin+grapefruit, clarithromycin+simvastatin, metformin+iodine, amoxicillin+penicillin). Run `python3 scripts/verify_audit_replay.py --check` to re-classify every pinned pair and assert each `repro_hash` matches byte-for-byte under the live engine `bundle_id`. Exits 0 on full agreement; exits 1 on any drift. After a deliberate weight rotation, re-run without `--check` to regenerate pins under the new bundle_id; the file preserves the old bundle_id so the chain is auditable across rotations. Pinned by `tests/test_scripts/test_audit_replay_verifier.py` (5 tests including bundle_id ↔ engine cross-check). |
| `Layer 4.5 calibration / margin diagnostic` | `docs/bitnet_calibration.json` (regenerate via `scripts/build_bitnet_calibration.py`) — per-pair top-1-vs-top-2 logit margin (Q16.16) for the entire 111-pair cache, plus per-class aggregates (mean margin correct vs wrong) and the top-10 worst-close-calls / confidently-wrong lists. Surfaces the safety case: the smallest-margin contraindicated miss (`itraconazole + simvastatin`) is at Q16.16 margin **90,199 ≈ 1.38** — a close call, not a confident misclassification, which means the upstream FDA deterministic table catches it cleanly. Pinned by `tests/test_scripts/test_bitnet_calibration.py` (8 tests including a `weights_id` cross-check so the artifact cannot drift out-of-sync with the engine weights bundle). |
| `Edge / offline build · $15 Pi Zero 2 W · USB plug-in` | `docs/edge_pi_offline.md` — full Edge profile spec (688 K params / 1.7 MB / 3-layer ternary / learned RxCUI embeddings), Pi-tier latency benchmarks (Pi 5 / Pi 4 / Pi Zero 2 W / ESP32), the **"ClinicalMem Box" hardware product profile** (USB OTG gadget mode for zero-IT plug-in, office-router drop-in, EHR sidecar; ~$99 SKU at ~$60 COGS), and the data-licensing reality check (RxNorm public + FDA SPL public + DrugBank commercial). The bit-identical SHA-256 audit hash works on Pi/x86/CUDA/browser — same `repro_hash` regardless of chip. |

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
   governance pub/sub) is LIVE against `mind-mem v3.9.0`. The
   actual cross-machine bytes-on-wire still go through a single-process
   queue in `scripts/federation_mock_demo.py`. v3.9.0 added an
   `http_transport.py` module but it is a **single-workspace REST
   adapter** for non-MCP clients (Slack bots / Streamlit / etc.),
   NOT a peer-to-peer federation transport — the dedicated MIC@2
   federation transport adapter targets mind-mem v3.10.
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
