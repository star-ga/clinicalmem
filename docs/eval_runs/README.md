# LLM Evaluation Runs — ClinicalMem

External multi-LLM evaluations of the ClinicalMem submission. Used as
brutal-honesty third-party signal for hackathon readiness — the local
gates (PCCP / precision / federation / arch-mind) tell us the system
*works*; these tell us how a judge LLM with the same system prompt
*sees* it.

## Round 4 (2026-05-03)

10-dimension rubric, demo.html inlined verbatim into the prompt to fix
the round-3 UI-blindness regression.

| Provider | Composite | Verdict |
|---|---|---|
| `gemini-3.1-pro` | **9.5 / 10** | WIN_LIKELY |
| `grok-4.3` | **8.7 / 10** | WIN_LIKELY |
| `deepseek-v4-pro` | **9.8 / 10** | WIN_LIKELY |
| `mistral-large` | **9.5 / 10** | WIN_LIKELY |
| `zhipu-glm-5` | **9.1 / 10** | WIN_LIKELY |
| `nvidia-deepseek-v3.2` | **9.3 / 10** | WIN_LIKELY |
| **Mean (n=6)** | **9.32 / 10** | **6 / 6 WIN_LIKELY** |

Providers that errored (transient infra / trust-dir / max-retries): OpenAI
GPT-5.5, Anthropic Opus-4.7, Perplexity Sonar-Pro, NVIDIA Nemotron-Ultra,
Moonshot Kimi-K2.5. Excluded from the mean because no parsed score
landed; they did not return a verdict either way.

### Top 3 blocking gaps named ≥ 2× across the 6 evaluators

1. **Real-world clinical validation gap** (deepseek-v4-pro, nvidia-deepseek,
   gemini): "Synthetic Synthea cohort, no published clinical study or
   real-world deployment." Smallest fix flagged: partner with a health
   system for retrospective validation. Documented limitation:
   `JUDGES.md` § Honest limitations bullet 1, `docs/irb_exemption.md`.
2. **BitNet 4.5 demo uses lookup table, not live ternary inference**
   (zhipu-glm-5, deepseek-v4-pro, gemini): "What-If simulator + verify-replay
   path call into a 39-pair pre-computed JSON, not the full Q16.16
   forward pass on novel pairs in-browser." Smallest fix flagged:
   compile BitNet ternary forward to WASM. Real-world: the Q16.16
   determinism IS the live forward pass server-side; the demo's
   precomputed cache exists only for the 5-second judge interaction.
3. **Single MD advisor, no institutional affiliation** (nvidia-deepseek,
   grok): "Dr. Ludmila Afonicheva is solo-credentialed; no
   hospital/IRB sponsor named." Smallest fix flagged: add a sponsoring
   hospital. Documented in `docs/clinical_validation.md`.

### What the eval validated

- 5 / 6 evaluators gave perfect 10s on `reproducibility_fda_defensibility`
  (the Q16.16 BitNet bit-identical anchor was the dominant theme).
- 5 / 6 gave 10s on `license_ip_clarity` (Apache-2.0 + § 3 patent grant
  recognized as deploy-tomorrow).
- 4 / 6 gave 10s on `ui_visual_design` after the round-3 UI elevation
  (mermaid + clinical typography + view()-driven scroll motion).

### Files

- `round_4_full.json` — primary 11-provider sweep (raw prompts + parsed
  scores + errors).
- `round_4_recovery_gemini.json` — recovery-only re-run for the 6
  evaluators that errored in the primary sweep; Gemini was the only
  recovery success.

---

*Apache-2.0 — STARGA, Inc. — 2026.*
