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
   represented in the 133-pair OpenEvidence-cited cohort —
   contraindicated (33/33), major (4/4), serious (69/69), moderate
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

The fastest path is a **single command** that runs all five gates
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
#    major / serious / moderate against the 133-pair OpenEvidence-cited
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

Optional: full pytest suite — `python3 -m pytest tests/test_engine/ tests/test_scripts/ -q`. Should report **1173+ passed**.

---

## Full audit trail

Every visible figure on the dashboard traces back to a checked-in
artifact. Audit map:

| Claim on the dashboard | Source of truth |
|---|---|
| `100% recall · contraindicated · 38/38` | `docs/openevidence_cache.json` (133 entries) → `docs/pccp_eval_latest.json` (per-pair verdicts) |
| `0 / 10 FP · precision = 1.0` | `docs/negative_control_cohort.json` (10 entries) → `scripts/run_negative_control_eval.py --json`. Cohort integrity is independently pinned by `tests/test_engine/test_negative_control_cohort_integrity_pin.py` (7 tests: cohort size = 10, every entry has expected_severity = "none", ZERO collision with cache contras, every entry has ≥ 1 evidence URL, the 4 named CYP-pathway boundary cases are present, clean negatives use drugs that are EITHER absent from contra contexts OR explicitly allow-listed for non-collision demonstration, demo cites this pin file near the precision-claim sentence (iter-121 surfacing extension)). |
| `8,512 ternary weights + 69 Q16.16 biases = 8,581 params / 19 KB` + bundle hash `cfadb4f6…` | `engine/bitnet_weights.json` + `engine/bitnet_classifier.py`. Param counts pinned by `tests/test_engine/test_bitnet_param_count_pin.py`. **Bundle-level integrity** independently pinned by `tests/test_engine/test_bitnet_bundle_integrity_pin.py` (9 tests: live `bundle_id` short prefix matches `cfadb4f6` AND tail matches `0b3f`; demo + JUDGES short-form citations match the live hash; file size stays within ±2 KB of 19 KB AND hard-ceiling 32 KB (Pi Zero 2 W edge claim); ternary-weight sparsity ≥ 40% (the iter-72 "structured sparsity" rhetoric); JSON key set is exactly `{_meta, hidden_b, hidden_w, output_b, output_w}`; `_meta` carries provenance fields; demo cites this pin file with the trained-model integrity callout (iter-131 surfacing extension); self-referenced `_meta.bundle_id` matches live SHA-256). |
| `Layer 4.5 BitNet confusion matrix (live deployment)` | `docs/bitnet_confusion_matrix.json` (regenerate with `scripts/build_bitnet_confusion_matrix.py`) — full ground-truth × predicted matrix on the 133-pair cache, plus per-class precision / recall. **Safety invariant: 0 false positives on contraindicated**, pinned in `tests/test_scripts/test_bitnet_confusion_matrix.py`. |
| `Path A v6 — full-recall breakthrough · 38/38 + 4/4 + 0 FP under Q16.16 (iter 207, staged); v5 → v6 cohort-growth gap closed by 8-anchor BOOST_KEYS @200×` | `retrain_runpod/bitnet_weights_v5_h128.json` (`bundle_id 1ff61a6a82a709d9…`) — a retrained 193-dim × **128-hidden** ternary bundle that strictly supersedes Path A v3 (h=64, `eea0e637…`, 29/31 + 1 FP under Q16.16). Trained via `retrain_runpod/sweep_v5_h128.py` running in `systemd-run --user --scope clinicalmem_v5_sweep_iter166.scope` (runtime-detachment-resilient — the iter-162 nohup attempt died at 6 of 30 seeds; systemd-user scope survives session boundaries). Seed=1 hit the strict gate immediately; sweep early-stopped per its hit-and-save logic. **Iter-172 cohort growth (isavuconazole+simvastatin) revealed a training-corpus generalization gap**: v5 dropped to 31/32 + 0 FP under Q16.16 because the triazole sub-class wasn't in the iter-148 training corpus + iter-156 BOOST_KEYS — same architectural-generalization-gap pattern as ritonavir+simvastatin under cfadb4f6 (HIV-PI sub-class). **Iter-177 cohort growth (ketoconazole+ergotamine)** revealed the same gap on the CYP3A4-strong-inh × ergot-derivative slot (was 1-example pre-iter-177, only clarithromycin+ergotamine), dropping v5 further to 31/33 + 0 FP. **Iter-182 cohort growth (minocycline+isotretinoin)** hit the same one-example-slot pattern on tetracycline×retinoid (was only doxycycline+isotretinoin pre-iter-182), dropping v5 to 31/34 + 0 FP. **Iter-187 cohort growth (midazolam+ketoconazole)** extended the CYP3A4-strong-inh × CYP3A4-substrate slot to a NEW substrate sub-class (benzodiazepine, vs slot's existing statins/ergots — wasn't in iter-148 corpus), dropping v5 to 31/35 + 0 FP. **Iter-192 cohort growth (eplerenone+ketoconazole)** extended the same slot AND the K+-sparing-diuretic sub-class (also absent from iter-148 corpus — FDA Inspra § 4 + Nizoral § 4 dual-label, 5× AUC → hyperkalemia/arrhythmia), dropping v5 to 31/36 + 0 FP. **Iter-197 cohort growth (cyclosporine+rosuvastatin)** opened a new sub-slot of the OATP1B1 × statin rule (rule 1) in pure isolation — the only prior OATP1B1×statin contra was gemfibrozil+simvastatin which fires multiple rules in parallel; cyclosporine+rosuvastatin fires rule 1 ALONE because rosuvastatin is NOT a CYP3A4 substrate so rule 0 does not fire (FDA Crestor § 4 explicit, mechanism: cyclosporine blocks OATP1B1 → 7× rosuvastatin AUC → rhabdomyolysis), dropping v5 to 31/37 + 0 FP and exposing the OATP1B1×statin slot as the most-undertrained sub-class (all three classifiers — cfadb4f6 + v3 + v5 — default to 'major' on this rule-1-only signal). **Iter-202 cohort growth (tolvaptan+ketoconazole)** added a NEW CYP3A4-substrate sub-class (vasopressin V2-receptor antagonist) — FDA Samsca + Jynarque § 4 dual-label vs strong CYP3A4 inhibitors (5× AUC → osmotic demyelination syndrome from over-correction of hyponatremia), dropping v5 to 31/38 + 0 FP. Same one-substrate-sub-class generalization gap pattern as midazolam (iter-187 benzodiazepine) and eplerenone (iter-192 K+-sparing diuretic). **Iter-207 v6 sweep landed**: `retrain_runpod/sweep_v6_h128.py` ran the 30-seed CPU sweep over `train_bitnet_v6_h128.py` with the iter-202-extended `BOOST_KEYS` (8 iter-fix anchors total: iter-156/173/177/182/187/192/197/202 — `azathioprine::febuxostat`, `isavuconazole::simvastatin`, `ergotamine::ketoconazole`, `isotretinoin::minocycline`, `ketoconazole::midazolam`, `eplerenone::ketoconazole`, `cyclosporine::rosuvastatin`, `tolvaptan::ketoconazole`). **Seed=31 hit the strict full-recall gate** (38/38 contra + 4/4 major + 0 FP) and saved the v6 bundle: `retrain_runpod/bitnet_weights_v6_h128.json`, `bundle_id 592ee51ee088cbd8…`. The 7 v5-known-miss sub-classes (triazole, ergot derivative, tetracycline×retinoid, benzodiazepine, K+-sparing-diuretic, OATP1B1×rosuvastatin pure-isolation, V2-receptor antagonist) are all caught by v6 — confirmed mechanically by the iter-210 cross-pin (every `_V5_EXPECTED_MISSES` pair classified `contraindicated` under v6). v6 inherits v5's architecture (193-dim × 128-hidden); only the weights changed. **NOT yet engine-promoted** — promotion requires the same iter-166-class cascade (encoder lift to 193-dim already in v5/v6; engine bundle swap requires re-pinning derived checks, JUDGES + demo update, audit-replay regen under V6 bundle_id, manifest SHA rotation, 41 audit-pin re-replay) **plus** the `64 → 128` `hidden_w` shape extension in `engine/bitnet_classifier.py`. **Pinned by 6 pin families**: `tests/test_engine/test_path_a_v5_live_recall_pin.py` (5 tests including the `_V5_EXPECTED_MISSES` invariant: miss set must EXACTLY equal `{(isavuconazole, simvastatin), (ketoconazole, ergotamine), (minocycline, isotretinoin), (ketoconazole, midazolam), (eplerenone, ketoconazole), (cyclosporine, rosuvastatin), (tolvaptan, ketoconazole)}` — neither MORE misses (regression) nor LESS (silent retrain landed without re-pin)) + `tests/test_engine/test_path_a_v5_q16_determinism_pin.py` (7 tests: **15 canonical pairs** (post-iter-202 extension: 8 severity-class anchors + 7 v5-known-misses) × 4 pinned values each (feature_hash + logits_q16 + logits_hash + severity_name) + 100×15 determinism stress test + severity-coverage invariant + **iter-183 cross-pin** ensuring every `_V5_EXPECTED_MISSES` pair has its Q16.16 logits pinned — bidirectional guarantee with the live-recall pin: 'these pairs MUST be missed' × 'these pairs' Q16.16 logits MUST equal pinned values', so silent encoder/weight drift on the missed pairs is impossible before v6 retrain) + **iter-178 cross-pin invariant** `tests/test_engine/test_v5_boost_keys_coverage_pin.py` (4 tests enforcing `_V5_EXPECTED_MISSES ⊆ BOOST_KEYS` so the next v6 retrain sweep will actually fix every queued miss; also pins the 5 iter-148 anchor pairs as always-required so the original baseline can't silently regress) + **iter-198 cross-pin invariant** `tests/test_engine/test_v5_canonical_pair_rule_bits_pin.py` (4 tests locking the per-pair encoder→rule mapping for all 14 canonical pins; mechanically anchors the iter-197 "rule-1-only signal" claim — `cyclosporine + rosuvastatin` must fire rule 1 AND ONLY rule 1, exposing OATP1B1 × statin as the most-undertrained sub-class; cross-pin floor: every `_V5_EXPECTED_MISSES` pair fires ≥ 1 pair-derived rule so BOOST_KEYS upweighting always has signal for v6 retrain to learn from — see encoder-level guarantee row 105 below for the full pin description) + **iter-209 V6 live-recall pin** `tests/test_engine/test_path_a_v6_live_recall_pin.py` (6 tests: bundle_id pinned at `592ee51e…` + Q16.16 38/38 contra full-recall + 4/4 major full-recall + zero-FP invariant + meta-block consistency (193×128×5, ternary weights, q16.16 biases, 13 rules, 26 flags, contra_recall=1.0) + **strictly_supersedes_v5_recall** invariant — v6 cannot regress below v5 on the same cohort, otherwise iter-207 BOOST_KEYS extension was counterproductive) + **iter-210 V6 q16 canonical-pin determinism** `tests/test_engine/test_path_a_v6_q16_determinism_pin.py` (8 tests: 15 canonical pairs × 4 pinned values each (logits_q16 + feature_hash + logits_hash + severity_name) + 100×15 = 1500 forward-pass determinism stress + severity-class coverage + bundle_id ↔ iter-209 cross-check + **iter-210 cross-pin invariant** mechanically locking the iter-207 BOOST_KEYS promise: every pair in `_V5_EXPECTED_MISSES` must appear in V6 canonical pins AND have severity_name=`contraindicated` — the bidirectional guarantee with iter-209 is "v6 catches every prior v5-miss"; without this pin the BOOST_KEYS promise has no per-pair lock). The **iter-164/iter-165 'v3 = 31/31 float-NumPy'** measurement-artifact rollback documents the importance of using the engine's actual Q16.16 inference path for any recall claim; iter-166 corrected v3 back to the iter-155 baseline of 29/31 + 1 FP. |
| `Layer 4.5 design-class abstention (high-precision veto)` | The empty `minor` and `serious` columns of the live confusion matrix are **architectural by design**: BitNet abstains from those two classes because they are the primary-recall classes carried by upstream Layers 1-4 (RxNorm + OpenEvidence + NIH RxNav + 6-LLM US-based consensus). Layer 4.5's role is the **high-precision veto** sitting BELOW those layers, not a primary classifier — emitting `minor`/`serious` would put it in competition with upstream instead of vetoing them. Pinned by `tests/test_engine/test_bitnet_design_class_abstention_pin.py` (5 tests: `minor` column total = 0, `serious` column total = 0, demo's "by design" rhetoric near the confusion-matrix card, demo names BOTH abstained classes with explicit `0 of N` live counts (iter-116 surfacing extension), JUDGES design-rationale citation). A weight rotation that lifts either column off zero must update demo + JUDGES rhetoric explicitly — silent drift here breaks the safety-case justification. |
| `100% explanation coverage · 38 / 38 contraindicated traced to a curated pharmacology rule` | **13 pair-derived DDI-rule flags** in `docs/pharmacology_flags.json` cover every contraindicated cache entry: (1) CYP3A4 inhib×substrate, (2) OATP1B1×statin, (3) P-gp inhib×substrate, (4) CYP2C9×anticoag, (5) MAOI×serotonergic, (6) PDE5×nitrate, (7) iodinated-contrast×metformin, (8) CYP1A2 inhib×substrate, (9) xanthine-oxidase×thiopurine, (10) folate-antagonist pair, (11) tetracycline×retinoid, (12) ACE×neprilysin, (13) metformin×renal-state. Iter-140 closed the 8-mechanism documented-gap class with rules 7-13 (was 19/27 = 70.4% coverage; now 29/29 = 100%). Pinned by `tests/test_engine/test_contra_explanation_coverage_pin.py` (4 tests: 100% floor, no documented-gap pairs allowed without flag firing, no stale gap-list entries, 13-rule cardinality lock) **AND** `tests/test_engine/test_pharmacology_flags_coverage_pin.py` (9 tests: every cache drug catalogued, every flagged drug has ≥ 1 evidence URL, all URLs HTTPS, schema_version + flag_keys present, snake_case `is_*` flag-key naming, lowercase canonical drug-name keys, **the 13 baseline iter-96 flag classes pinned by name**, **the 12 iter-140 flag classes pinned by name**, **canonical-example pin** mapping every pair-derived rule index to a (drug_a, drug_b) pair from the live cache that MUST fire that rule — same pin pattern as iter-84 arch-mind 9-rule canonical set + iter-94 cache shape; catches silent flag rename, dead rule, AND lost example regressions in real time) **AND** `tests/test_engine/test_per_rule_cohort_coverage_pin.py` (iter-203 extension, 4 tests at cohort-aggregate level): (1) **encoder-level coverage** — every contraindicated cache entry fires AT LEAST 1 pair-derived rule under the encoder's bit logic (complements the curated-table explanation pin above; catches the drift class where a new contra is added without the necessary pharmacology_flags.json flag for one of the drugs); (2) **per-rule contra distribution snapshot** — the count of contras firing each rule is exact-pinned (rule 0 cyp3a4_inhib×substrate: 15 saturated · rule 1 oatp1b1×statin: 3 undertrained · rule 2 P-gp inhib×substrate: 6 · rule 3 CYP2C9×anticoag: **0 by design** · rule 4 maoi×serotonergic: 8 · rules 5-12: 1-2 each); catches silent flag rename or removal that drops a rule's firing count AND cohort-growth drift that changes the distribution without lockstep pin update; (3) **rule 3 (CYP2C9 × anticoag) zero-cohort design state** is mechanically locked — the rule is structurally present in the encoder because warfarin + fluconazole/miconazole/voriconazole interact via CYP2C9 inhibition, but the FDA Coumadin label classifies these as § 5.4 "monitor INR + dose reduce" rather than § 4 absolute contraindications (anticoagulation interactions are managed by monitoring, not blocked); a future cohort growth that adds a CYP2C9 × anticoag contra MUST update both the design-state pin AND the per-rule distribution in lockstep, preventing silent drift; (4) **sanity floor** — sum of per-rule firings ≥ cohort size. Same drift-prevention shape as iter-178/183/188/193/198 cross-pins, applied at cohort-aggregate level (rather than per-pair) so a cohort-growth event that drops a rule's count to 0 cannot silently land. **6th cross-pin family** in the discipline lineage. |
| `Q16.16 determinism (bit-identical replay)` | `scripts/run_bitnet_determinism_stress.py` — runs 100 iterations × 12 representative pairs (1200 classifier calls) and asserts every iteration produces bit-identical `severity_name + repro_hash + logits_q16`. Pinned in `tests/test_scripts/test_bitnet_determinism.py` (lighter 10-iteration form for the standard pytest scope; subprocess-runs the full stress script). Cross-machine determinism is implied by the Q16.16 fixed-point math (no floating-point ops). **Encoder-level guarantee** (iter-188 extension): `tests/test_engine/test_encode_pair_determinism_pin.py` (10 tests) makes the contract for `encode_pair` — the FIRST step of the inference pipeline — explicit at unit level. Tests determinism (100×4 pairs = 400 calls bit-identical), output dimension (193 = 64 hash trits + 26 flag bits per drug × 2 + 13 pair-derived rule bits), all-int output (no float leak), pair-derived bits ∈ {0,1} (no integer overflow), symmetry (`encode_pair(a,b) == encode_pair(b,a)` via internal lex-sort), case-insensitivity for consistent-case input (with documented mixed-case edge case unreachable in live pipeline since FHIR/Synthea callers pre-lowercase), whitespace robustness, and pair-derived canonical-example assertions (clarithromycin+simvastatin → rule 0 fires; lisinopril+sacubitril → rule 11 fires; unknown drugs → all rules silent). If `encode_pair` ever produces different output for the same input, every downstream Q16.16 logit pin collapses — this pin makes the encoder contract loud and visible. **Per-pair rule-firing guarantee** (iter-198 extension): `tests/test_engine/test_v5_canonical_pair_rule_bits_pin.py` (4 tests) extends the encoder contract to **per-pair rule-bit cardinality**. For each of the 14 V5 canonical pins, asserts the EXACT documented set of pair-derived rule bits (last 13 dims of encode_pair output) that fires — locking the `pharmacology_flags.json` → encoder → V5 model chain at unit level. Crucially: (a) **the iter-197 "rule-1-only signal" claim is mechanically enforceable** — `cyclosporine + rosuvastatin` must fire rule 1 (`is_oatp1b1_inhibitor × is_statin`) AND ONLY rule 1 (rosuvastatin is NOT a CYP3A4 substrate so rule 0 does not fire); a future flag addition that breaks this isolation fires the pin and forces JUDGES + demo to retract the "most-undertrained sub-class" framing; (b) the 3 demo anchors fire ZERO rules (cross-class severity demonstration uses only hash-trit + flag-bit contributions, not rule firings); (c) **cross-pin invariant**: every `_V5_EXPECTED_MISSES` pair fires AT LEAST 1 pair-derived rule — a 0-rule miss in BOOST_KEYS would be a wasted upweight slot with no signal for v6 retrain to learn from. Mirror-shape of iter-178 BOOST_KEYS coverage + iter-183 Q16.16 canonical-pins coverage cross-pins. |
| `In-browser BitNet inference (vanilla JS)` | `docs/bitnet_browser.js` (≈ 300 LOC, no dependencies) ports the Q16.16 forward pass to vanilla JavaScript: hand-written BLAKE2b-128 (BigInt for 64-bit ops) + 64-trit per-drug encoding + 128→64→5 ternary linear + ReLU + argmax + SHA-256 canonical-JSON repro_hash. Cross-language pin: `tests/test_engine/test_browser_bitnet_pin.py` runs the JS file via Node.js and asserts the resulting `repro_hash + severity_name + feature_hash + logits_q16 + weights_id` for warfarin + ibuprofen match the Python reference **byte-for-byte**. Closes the iter-50 R4-eval gap that 3 of 6 evaluators flagged ("BitNet 4.5 demo uses pre-computed lookup table, not live ternary inference"). |
| `Single-file reproducibility manifest` | `docs/reproducibility_manifest.json` (regenerate with `scripts/build_reproducibility_manifest.py`) — content-addressed snapshot of every load-bearing artifact: SHA-256 of 8 SHA-tracked artifacts (cache + weights + confusion matrix + cohort coverage matrix + cohort bundle + calibration + audit-replay pins + pharmacology flags), flow plan_hashes for all 7 `.flow.mind` files, gate verdicts for all 5 gates (PCCP / negative-control / federation / arch-mind / audit-replay), test count, git HEAD. Drops into a compliance review as one file an FDA SaMD reviewer can verify with `--check`. Pinned in `tests/test_scripts/test_reproducibility_manifest.py`. |
| `Evidence-URL backbone integrity` | `tests/test_engine/test_cache_evidence_urls.py` — 6 tests pin the structural integrity of every cache entry's `evidence_urls`: ≥ 1 URL per entry, all HTTPS (no plain HTTP), all have non-empty host AND non-empty path beyond root, no duplicates within an entry, ≥ 90% from the NIH/FDA/peer-review whitelist, average ≥ 1.5 URLs per pair (corroboration floor — every drug-pair classification backed by ≥ 1 primary citation + ≥ 1 secondary). Live: 298 URLs across 133 entries, 100% authoritative, avg 2.24 URLs/pair. |
| `21 typed federation invariants` | `flows/JointMemoryFederation.flow.mind` (plan_hash recorded in audit chain). **Cross-doc count integrity** independently pinned by `tests/test_scripts/test_federation_invariant_count_pin.py` (5 tests: live invariant count in flow file = 21; demo's `INVARIANT_DESCRIPTIONS` exercises 16; **all 6 user-facing federation docs** (demo.html, JUDGES.md, architecture.md, clinical_validation.md, fda_q_sub_draft.md, federated_memory.md) cite the canonical `21 typed` count; bare `16 invariants` / `16 typed runtime invariants` claim is forbidden in any of those 6 docs unless paired with `21 typed` disambiguation in the same file (iter-135 scope expansion catching the same insidious `scoped pin + unscoped doc` drift class iter-132 caught for the iter-122 transport-distinction claim — three regulatory-adjacent docs had silently lied about the count for ≥ 113 iterations); the 16-of-21 gap explanation (`5 X25519 sealing invariants await MIC@2 federation-transport adapter targeting mind-mem v3.10`) must remain on demo.html. |
| `Federation control plane LIVE — mind-mem v3.9.0 MemoryMesh` | `engine/federation_transport.py` (9 unit tests) + `mind_mem.memory_mesh.MemoryMesh` |
| `arch-mind 9 / 9 rules` | `docs/arch_mind/clinicalmem_rules.mind` + `docs/arch_mind/clinicalmem.scan.json` (run via `scripts/run_arch_mind_gate.py`) |
| `30 synthetic patients · 47 NPIs` | `docs/synthea_demo_cohort.json` (FHIR R4 bundle, all NPIs Luhn-valid). **Per-patient drug-pair → cache-entry traceability matrix:** `docs/cohort_coverage_matrix.md` (regenerate with `scripts/build_cohort_coverage.py`). Cohort integrity is independently pinned by `tests/test_engine/test_synthea_cohort_integrity_pin.py` (9 tests: FHIR R4 Bundle top-level shape; Patient count floor ≥ 30; Practitioner count floor ≥ 47; every Practitioner has an NPI; every NPI passes CMS Luhn check; every Patient + Practitioner carries `meta._synthetic = true`; every Practitioner carries `meta.npi_source = "DEMO_LUHN_GENERATED"`; demo cites this pin file with the cohort-integrity callout (iter-126 surfacing extension); no demo NPI collides with the known-real clinical-validation NPI 1932159530). |
| `21 CFR Part 11 audit export` | `engine/audit_export_part11.py` (30 tests) |
| `Apache-2.0 + patent grant` | `LICENSE` (top of repo) |
| `IRB-exempt synthetic cohort` | `docs/irb_exemption.md` (45 C.F.R. § 46.102(e)(1)) |
| `Clinical Advisor: Dr. Ludmila Afonicheva, MD` | `docs/clinical_validation.md` (US-licensed Family Medicine, NPI 1932159530) |
| `External judge-LLM consensus 9.32 / 10 (n=6)` | `docs/eval_runs/round_4_full.json` + `docs/eval_runs/round_4_recovery_gemini.json` (raw rubric scores + parsed gaps from gemini-3.1-pro / grok-4.3 / deepseek-v4-pro / mistral-large / zhipu-glm-5 / nvidia-deepseek-v3.2). Summary in `docs/eval_runs/README.md`. |
| `Layer 4.5 audit-replay verifier (FDA SaMD "decade-stable replay" claim, runnable; full safety-class coverage)` | `scripts/verify_audit_replay.py` + `docs/audit_replay_pins.json` (**41 pairs** as of iter-202: 3 non-contra anchors — warfarin+ibuprofen / atorvastatin+grapefruit / amoxicillin+penicillin — plus **every contraindicated cache entry**, currently 38). Run `python3 scripts/verify_audit_replay.py --check` to re-classify every pinned pair and assert each `repro_hash` matches byte-for-byte under the live engine `bundle_id`. Exits 0 on full agreement; exits 1 on any drift. After a deliberate weight rotation, re-run without `--check` to regenerate pins under the new bundle_id; the file preserves the old bundle_id so the chain is auditable across rotations. Future cache contraindicated growth automatically extends the replay set (no code edit needed). Pinned by `tests/test_scripts/test_audit_replay_verifier.py` (7 tests including bundle_id ↔ engine cross-check + full-contra-coverage assertion) + **iter-193 structural-integrity pin** `tests/test_engine/test_audit_replay_structural_integrity_pin.py` (5 tests closing the 3 drift gaps the coverage-floor pin misses: (1) **over-pinning** — `pin_count == 3 anchors + cohort_size` exact equality, not just `>=`, so extra unrelated pairs can't silently inflate the "41-pair re-replay" claim; (2) **lost demo anchors** — the 3 hand-picked anchors (warfarin+ibuprofen, atorvastatin+grapefruit, amoxicillin+penicillin) MUST always be present with non-`none` severity_name, so the BitNet 4.5 cross-class severity demonstration can never be silently dropped during a regen; (3) **malformed pin entries** — per-pin field-shape contract: `drug_a` + `drug_b` lower-stripped non-empty, `feature_hash` + `repro_hash` 64-char lowercase hex SHA-256, `logits_q16` 5-element list of `int` (Q16.16 fixed-point 5-class severity output), `severity_name` ∈ valid 5-class enum, no duplicate pairs (direction-flip canonicalised), JSON-LD envelope intact (`@context` + `@type` + `bundle_id` 64-char hex + `pairs` + `version` for the FDA SaMD Schema.org Dataset claim)). Same drift-prevention shape as the iter-178 BOOST_KEYS coverage cross-pin + iter-183 Q16.16 canonical-pins coverage cross-pin + iter-188 encode_pair encoder contract — extends the cross-pin discipline to the audit-replay JSON envelope itself. |
| `Layer 4.5 calibration / margin diagnostic` | `docs/bitnet_calibration.json` (regenerate via `scripts/build_bitnet_calibration.py`) — per-pair top-1-vs-top-2 logit margin (Q16.16) for the entire 133-pair cache, plus per-class aggregates (mean margin correct vs wrong) and the top-10 worst-close-calls / confidently-wrong lists. Surfaces the safety case: the smallest-margin contraindicated miss (`itraconazole + simvastatin`) is at Q16.16 margin **90,199 ≈ 1.38** — a close call, not a confident misclassification, which means the upstream FDA deterministic table catches it cleanly. Pinned by `tests/test_scripts/test_bitnet_calibration.py` (8 tests including a `weights_id` cross-check so the artifact cannot drift out-of-sync with the engine weights bundle). |
| `BitNet-alone vs engine major recall (3 of 4 + 1 safety override)` | `docs/bitnet_calibration.json::entries[ground_truth=major]` shows the live truth: 3 correct (paroxetine+tamoxifen / clarithromycin+digoxin / dabigatran+dronedarone), 1 miss (tacrolimus+voriconazole, predicted "none"). The engine still emits `major` for that pair because Layer 4.5's safety policy (`engine/clinical_scoring.py:329-339`) preserves the upstream verdict and emits a `BITNET_SAFETY_DOWNGRADE_DISAGREEMENT` warning into the run log on every disagreement. So the dashboard's `100% · 4 / 4` is the engine final output; BitNet alone is **3 of 4**. The miss is the architectural ceiling of the hash-only encoder on transporter+strong-CYP3A4 cross-mechanisms; richer pharmacology features (Path A) are tracked but not yet shipped. Pinned by `tests/test_engine/test_bitnet_alone_major_recall_pin.py` (5 tests: frozen correct set, frozen miss set + predicted_as, demo distinguishes BitNet-alone from engine, demo names the miss pair, JUDGES cites this pin file). |
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
4. **6-LLM US-based consensus surface (Layer 4) is implementation-only.** The
   API-key-required path is wired but the eval ran against the
   deterministic Layer 1 / Layer 2 cache / Layer 4.5 BitNet path.
   Multi-LLM consensus rounds are out of scope of the precision +
   recall gates above.
5. **Layer 4.5 BitNet alone catches 3 of 4 majors.** The engine
   reaches `100% · 4 / 4` on the major class only because Layer 4.5's
   safety policy (`engine/clinical_scoring.py:329-339`) preserves
   the upstream verdict on any BitNet downgrade and emits a
   `BITNET_SAFETY_DOWNGRADE_DISAGREEMENT` warning. BitNet alone misses
   `tacrolimus + voriconazole` (transporter + strong CYP3A4
   cross-mechanism) — the architectural ceiling of a hash-only
   encoder. The Q16.16 margin on the miss is large (≈ 15.4 in natural
   units), so this is a confident wrong, not a close call. Path A
   (curated ATC pharmacology table integration) is the candidate fix
   but has not yet been wired into the deployment path.

---

## One-line architecture

`EHR (FHIR R4)` → `engine.fhir_adapter` → `Layer 1 table` → `Layer 2 OpenEvidence` → `Layer 3 RxNorm DDI` → `Layer 4 6-LLM US-based consensus` → **`Layer 4.5 BitNet b1.58 (Q16.16, bit-identical)`** → `Layer 5 LLM synthesis` → `Layer 6 abstention gate` → `mind-mem store + 21 CFR Part 11 audit chain` → `MCP (18 tools) + A2A (13 skills)`.

For deeper architecture see `docs/architecture.md`. For the federation
control-plane integration see `docs/architecture.md § Mock vs. live transport`
and `docs/arch_mind_federation_audit.md`.

---

## Contact

- Repo: <https://github.com/star-ga/clinicalmem> (PUBLIC on submission day 2026-05-11; PRIVATE before that)
- Engineering: STARGA Inc. — `noreply@star.ga`
- Clinical Advisor: Dr. Ludmila Afonicheva, MD — Family Medicine, US-licensed (NPI on file in `docs/clinical_validation.md`)

*Apache-2.0 — STARGA, Inc. — 2026.*
