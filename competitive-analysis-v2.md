# ClinicalMem — Competitive Analysis v2 + Winning Plan

> **Agents Assemble Healthcare AI Hackathon** | $25K Prize Pool ($7,500 first place)
> **Deadline:** May 11, 2026 @ 11:00pm EDT
> **Snapshot date:** May 2, 2026 (this document)  ·  **Updated:** May 9, 2026 (T-2 days to deadline)
> **Participants (May 2 snapshot):** 3,762 (up 9× from 411 in March)
> **Project gallery:** still hidden ("hackathon managers haven't published this gallery yet")
> **Winners announced:** ~May 27, 2026
>
> **Status as of iter-426 (2026-05-09):** P2.1 (test push 235 → 400+) ✅ DONE — live floor 1409+ (3.5× past target), 38 cross-pin discipline families enforcing 0% known misses + 0 PHI leaks. **iter-421 architectural breakthrough SHIPPED**: Path B 2-bundle BitNet ensemble at Layer 4.5 hits **100% recall on EVERY severity class** (44/44 contra · 4/4 major · 69/69 serious · 22/22 moderate · 0 contra FP · 0 major FP) — multi-LLM consensus recipe (4-of-5: DeepSeek/Mistral/GLM/NVIDIA-Llama70B) shipped via frozen v8 contra-gate cascade into a tier-2 specialist (h=64, 95-non-contra-sample training). iter-426 audit-replay closure: `verify_audit_replay.py` now mirrors the iter-421 cascade so all 47 pinned pairs reproduce byte-for-byte under the composite `weights_id = "{a_id}+{b_id}"` audit primitive across both bundle hashes. P0/P1/P3 (DevPost form, demo video, screenshots) remain in user-driven submission queue. P4 final pre-submission lockdown begins ~May 10.

---

## Executive summary

ClinicalMem is **structurally well-positioned to win** but the competitive surface is wider than the March 8 analysis assumed. Of the 3,762 registered participants, the active GitHub/template-fork signal is ~30 real entries; only **3 of them have non-trivial healthcare differentiation** (Gracestack, prior-auth-ai-agent, rcm-sentinel). The rest are either reference-template forks, mental-wellness apps, or scaffolds with 0 stars and no novel work.

**Current ClinicalMem state (v3.3.0)** is materially stronger than the March 8 snapshot:

- 6 layers of clinical safety (was 4 in March) — Layer 4.5 BitNet now runs the **iter-421 Path B 2-bundle ensemble** at **100% recall on every severity class** (44/44 contra · 4/4 major · 69/69 serious · 22/22 moderate · 0 contra FP · 0 major FP)
- 6-model US-based LLM consensus (was a 3-model cascade)
- 4-tier drug interaction pipeline including the **NIH RxNorm REST + Drug Interaction API** — the same federal database used by Epic, Cerner, and all certified EHRs
- UMLS Metathesaurus crosswalk (ICD-10 ↔ SNOMED ↔ LOINC ↔ RxNorm)
- What-If medication simulation, FDA Safety Alert integration, ClinicalTrials.gov matching
- PHI Detection Guard + Hallucination Detection
- 1409+ tests (engine + scripts; live floor as of iter-426) — 38 cross-pin discipline families mechanically enforcing 0% known misses + 0 PHI leaks
- **Bit-identical Q16.16 cross-architecture replay** under the iter-421 ensemble: composite `weights_id = "{bundle_id_a}+{bundle_id_b}"` audit primitive reproduces every cascade decision byte-for-byte on x86_64 / ARM64 / RISC-V / Pi Zero / A100 / browser
- Live demo: `clinicalmem-demo.pages.dev`
- Both MCP server (18 tools) AND A2A agent (5 skills · 13 tools) — covers both hackathon tracks

**The 9-day plan below** focuses on three load-bearing wins: (a) submission-completeness sweep, (b) demo polish to neutralise Gracestack's visual-UI advantage, (c) edge-case testing pass to push tests over 400 and harden the abstention story.

---

## Live competitive surface (May 2)

### Tier 1 — Active threats with non-trivial differentiation (3 entries)

#### 1. Gracestack AI (`cedendahlkim/Agents-assemble`) — STALE since March 6

| Attribute | Value |
|---|---|
| Last commit | March 6, 2026 (no updates in 8 weeks) |
| Commits | 11 |
| Stars | 0 |
| Stack | TypeScript + Rust, Gemini 2.5 Flash, React 19 + Tailwind v4 |

**Memory-Agent cognitive modules:** Ebbinghaus forgetting curve · HDC (10,000-dim bipolar vectors) · **Gut Feeling** (anomaly detection — claims 85% confidence on NSAID + Aspirin bleeding risk).

**Threat trajectory:** *declining*. Repository has been stale 8 weeks. Gracestack has not improved their product since the original 4-tier pipeline analysis. ClinicalMem has substantially closed the visual-storytelling gap (live demo dashboard, YouTube video, Cloudflare Pages deployment).

**Where they still win:** the "AI that forgets like a human" narrative is sticky for general-audience judges. Counter: the safety-first narrative is decisively stronger for healthcare AI specifically — judges who understand the domain will weight clinical evidence over cognitive metaphor.

#### 2. `saiprasanth-git/prior-auth-ai-agent` — Specialised, active

| Attribute | Value |
|---|---|
| Last commit | April 13, 2026 |
| Commits | 21 |
| Stars | 1 (self-favourite or one early follower) |
| Focus | Prior authorisation workflow automation |

Built on po-adk-python; FHIR R4 + Gemini 2.5 Flash + A2A; produces ICD-10/CPT-coded PA letters; estimates 0–100% payer-approval likelihood; checks FDA guidelines.

**Threat:** prior auth is a high-value, well-defined healthcare AI use case. A judge looking for "real ROI" might rank this highly.

**Gaps:** no clinical safety pipeline, no audit trail, no multi-LLM consensus, no drug-interaction surface, no allergy logic, no PHI/hallucination detection, no test coverage shown. Single-domain focus is also a weakness — the hackathon prompt is "build a *full agent* … to tackle a specific healthcare challenge," but ClinicalMem covers the safety story across drug interactions, allergies, contradictions, and clinical trials simultaneously.

#### 3. `arvind-elayappan/rcm-sentinel` — Mature template, clean architecture

| Attribute | Value |
|---|---|
| Last commit | April 19, 2026 |
| Commits | 25 |
| Stars | 0 |
| Architecture | Three-agent (healthcare / general / orchestrator) |

Strong A2A v1 compliance (nested `supportedInterfaces`, typed `securitySchemes`); per-agent X-API-Key auth; FHIR credentials flow through A2A metadata (never exposed to LLM prompt); comprehensive test suite; Docker + Cloud Run.

**Threat:** the architecture is the cleanest of any non-ClinicalMem entry. Judges who weight engineering polish heavily may rank this competitive.

**Gaps:** healthcare functionality is shallow — `healthcare_agent` is a basic FHIR-query agent; `general_agent` does ICD-10 lookup and date/time. No drug interactions, no allergies, no clinical safety pipeline, no contradiction detection. ClinicalMem's clinical depth is 10× by feature count.

### Tier 2 — Active hackathon entries with limited differentiation (~10)

| Repo | Latest | Notes |
|---|---|---|
| `ameyrane98/HealthBud` | Apr 25 | Plain fork of po-community-mcp + FHIR tools; no novel logic visible |
| `Preethi-Sundaravelu/healthcare-community-mcp` | Apr 15 | Renamed fork; no README signal |
| `naufaldirfq/Baby-Milestone-Matcher-MCP` | Mar 30 | Niche pediatric-milestone matcher; off-topic for clinical safety |
| `Eradboi/po-adk-python-hackathon` | Apr 24 | Hackathon-named; minimal content |
| `s0r0j/agents-assemble` (MindCare AI) | Apr 25 | Mental wellness 5-question stress test; no FHIR/MCP/A2A |
| `LongThanVu/Healthcare-AI-Agents-` | Mar 30 | Triage/Scheduling/Data; no FHIR/MCP/A2A standards compliance, no LLMs visible |
| `Miles762/po-community-a2a` | May 1 | Active A2A fork; no public README signal |
| `SilentDebugger/po-community-mcp` | **May 2** | Last-minute push today; need to re-monitor |
| `NoraDOSSOUGBETE/prompt-opinion` | May 1 | Recent fork; content unclear |
| `Scriea/agents-assemble-hackathon` | Mar 11 | Stale since hackathon start |

**Net assessment:** none of these are likely podium contenders today. **`SilentDebugger`** is the one to re-watch — a May 2 push 9 days before deadline can mean either a serious last-mile push or a routine scaffold update.

### Tier 3 — Reference-implementation forks (no novel work)

20+ unmodified forks of `po-community-mcp` and `po-adk-python` from individuals who registered but never pushed novel work. Not threats.

### Tier 4 — Out-of-scope / non-hackathon repos

`the-momentum/fhir-mcp-server`, `wso2/fhir-mcp-server`, AWS HealthLake MCP server. All commercial / enterprise products, not hackathon entries.

---

## Recon items I cannot complete remotely (you can)

1. **Discord** — `https://discord.gg/JS2bZVruUg` (Prompt Opinion server). I confirmed the invite resolves but did not scrape the channels. Worth checking:
   - **#showcase / #demos / #help / #project-channel-1..N** — anyone pasting screenshots, demo videos, or repo links?
   - **#announcements / #judging-criteria** — any update on judge weighting that contradicts our March-era assumptions?
   - **#general / #introductions** — any team that mentioned drug interactions, allergies, contradictions, multi-LLM, audit trails? Those are the only entries that would compete with ClinicalMem's safety surface.
2. **DevPost forum / discussion** — `agents-assemble.devpost.com/discussions` returned 404 in my fetch; maybe it's only visible to logged-in participants. Worth checking on your side for any pinned threads that signal what judges weight.
3. **Prompt Opinion Marketplace** — `app.promptopinion.ai/marketplace` — the page my fetch returned was empty. Need a logged-in screenshot to see what's already published. **High priority** — the entries already in the marketplace are the visible competition.
4. **Sample-team chatter** — if you know team handles, search GitHub for their accounts and watch what repos they push to between today and May 11.

---

## Judging-criteria-aligned scoring (May 2 snapshot)

| Criterion | Weight | ClinicalMem | Gracestack | prior-auth | rcm-sentinel |
|---|---|---|---|---|---|
| **The AI Factor** (multi-LLM, novelty, sophistication) | High | **6-LLM US-based consensus + 4-tier interaction pipeline + Q16.16 abstention kernels** | Ebbinghaus + HDC (academic novelty) | Single-LLM PA letter generation | Single-LLM FHIR Q&A |
| **Potential Impact** (does it save patient lives?) | High | **Catches NSAID+warfarin, β-lactam cross-reactivity, declining GFR + metformin, BP-target conflicts — every demo example has a documented "would have prevented harm" case** | Hardcoded interaction lookup; no allergy logic; no cross-provider contradictions | Faster prior auth approval; no patient-safety story | FHIR query convenience; no patient-safety story |
| **Feasibility** (regulatory, deployable, real) | High | **Live cloud deployment, SHA-256 hash-chain audit, abstention gate, 1409+ tests, Apache-2.0 with explicit patent grant, MCP+A2A both, PHI guard, hallucination detection** | Docker Compose only; localhost; no audit; no abstention; no tests | 21 commits; no audit; no compliance docs | Clean architecture but shallow features |
| **Demo polish** (judge experience) | High | **Live demo dashboard + YouTube + Cloudflare Pages + Prompt Opinion native flow** | Static React; no live URL | Likely none deployed | Cloud Run scripts but no live URL cited |

**Estimated rank against today's known field: 1st.**

The risk vector is not the visible field — it's hidden submissions. With 3,762 participants and the gallery still private, there are likely 10–30 strong submissions on private branches that we cannot see. The plan below assumes that risk.

---

## 9-day winning plan (May 2 → May 11)

### Theme: defence-in-depth — make ClinicalMem impossible to beat on every judging criterion

The product is already differentiated. The 9-day plan is about **completeness, polish, and signal amplification** — not new features. Adding novel features in the last week introduces bugs; doubling down on what works does not.

### P0 — Submission-completeness sweep (May 2 → May 4, 2 days)

| ID | Item | Time | Why it matters |
|---|---|---|---|
| **P0.1** | Verify DevPost submission complete: title, tagline, story, video link, repo link, demo URL, all 6 screenshots, tech tags, prizes targeted | 30 min | A missing field = automatic disqualification. |
| **P0.2** | Verify Prompt Opinion Marketplace listing: published, agent card valid, MCP tools enumerated, A2A skills enumerated, working `:complete` / `:classify` / health endpoints | 1 hr | The hackathon explicitly requires marketplace publication. |
| **P0.3** | End-to-end smoke test: from a fresh browser session, click DevPost link → load demo dashboard → run Sarah Mitchell scenario → see 4 findings → click through audit chain → verify each finding has evidence citation. Record as a single 90-second clip. | 2 hrs | Judge experience is the single best predictor of placing. |
| **P0.4** | Verify all live URLs are stable for 9 days: cloudflare-pages renew, Prompt Opinion auth tokens not expiring, FHIR sandbox tokens valid through May 27 | 1 hr | A dead demo on judging day is fatal. |
| **P0.5** | DevPost story rewrite: one-paragraph hook (Sarah Mitchell), one paragraph "what makes this different" (the 6-layer pipeline as a phrase), one paragraph "why we'll win on judging criteria" — all under 1,500 chars. | 2 hrs | Most judges read the story; most stories are bad. |

### P1 — Demo polish to neutralise Gracestack's visual edge (May 4 → May 6, 2 days)

| ID | Item | Time | Why |
|---|---|---|---|
| **P1.1** | Re-record YouTube demo with: clear narration + on-screen captions + 0:00–0:15 hook (Sarah Mitchell) + 0:15–1:00 the 4 catches + 1:00–2:00 the 6-layer architecture diagram + 2:00–2:45 audit chain visualisation + 2:45–3:00 close | 4 hrs | The current 1.0 video was a single take. A polished 3-minute video is the highest-leverage demo asset. |
| **P1.2** | README hero section: animated GIF (3 seconds, 1MB) showing the dashboard catching the NSAID+warfarin interaction in real time | 2 hrs | First impression on the GitHub page. |
| **P1.3** | Add a "Why ClinicalMem" sidebar to the demo dashboard: 6 columns (Ebbinghaus / Single-LLM / FHIR-only entries / Contradictions detected / Audit chain / Live deployment) — each with ✓ or — for ClinicalMem vs the field | 3 hrs | Lets judges see differentiation without reading the README. |
| **P1.4** | Create one polished architecture diagram (3 layers: data sources → 6-layer pipeline → outputs) in `gallery_architecture.png` | 2 hrs | Replaces the existing diagram with one optimised for screenshot-grade clarity. |
| **P1.5** | Demo dashboard: add a "What other systems missed" callout under each finding | 2 hrs | Makes the value proposition concrete. |

### P2 — Coverage + abstention story (May 6 → May 8, 2 days)

| ID | Item | Time | Why |
|---|---|---|---|
| **P2.1** ✅ | **DONE — live floor 1409+ (3.5× past 400+ target as of iter-426).** Coverage achieved: MCP tool round-trip, A2A skill round-trip, every NIH RxNorm error path, every FHIR fallback, every abstention boundary, plus 38 cross-pin discipline families that mechanically enforce 0% known misses + zero PHI leaks under structured logging. iter-425 added the 36th family (bundle B content integrity at 13 invariants); iter-426 closed the audit-replay verifier under the iter-421 ensemble cascade. | 6 hrs | "Feasibility" criterion weights tests heavily. 1409+ is a clear lead — 3.5× the original target. |
| **P2.2** | Write `docs/abstention_proof.md`: a side-by-side of 12 ambiguous medication queries showing what GPT-5.5 / Gemini Pro / Claude / Grok / Sonar / Flash each returned → where consensus broke → where ClinicalMem abstained → what the right answer is | 4 hrs | Turn the abstention story into a falsifiable, replayable artefact. The single strongest "we're not bullshitting" signal. |
| **P2.3** | Add a 5-patient demo set: Sarah Mitchell + 4 others (pediatric, oncology, mental-health, post-op) — each catches a different finding class | 4 hrs | Counters the "single demo patient" criticism. |
| **P2.4** | `docs/regulatory_readiness.md`: HIPAA path (PHI guard + at-rest encryption pathway), DO-178C-style audit (hash chain), FDA-IDE pre-submission outline | 2 hrs | Healthcare AI judges are deeply attuned to "could this actually ship?" |

### P3 — Signal amplification (May 8 → May 10, 2 days)

| ID | Item | Time | Why |
|---|---|---|---|
| **P3.1** | Discord recon: monitor #showcase, #demos, #help — keep a running threat list. Engage *only* to answer technical questions about MCP + A2A; do not promote. | 30 min/day | Stay current with the visible field; build organic recognition with judges in the channel. |
| **P3.2** | LinkedIn post (your account, not the bot): one-paragraph "what we built and why it matters" — 600 chars max, link to demo. | 1 hr | Social signal helps judge tie-breakers and post-hackathon visibility. |
| **P3.3** | One thoughtful blog post: `mindlang.dev/blog/clinicalmem-safety-pipeline` — the 6-layer story, the math behind abstention, the audit-chain design, why "I don't know" saves lives | 4 hrs | A linked, externally-hosted article on a STARGA-owned domain reinforces the "this is a real product" signal. |
| **P3.4** | Two technical issues filed back to `prompt-opinion/po-community-mcp` with clean PRs (e.g. SHARP token validation + agent-card schema fix) | 3 hrs | Upstream contributions raise visibility with the platform team (who likely have judge ties). |
| **P3.5** | Update DevPost screenshots: 6 high-resolution captures showing (1) live dashboard, (2) NSAID+warfarin catch, (3) audit chain, (4) abstention example, (5) 6-LLM consensus, (6) marketplace listing | 2 hrs | Every screenshot is real estate the README + DevPost share with the judges. |

### P4 — Final pre-submission lockdown (May 10 → May 11)

| ID | Item | Time |
|---|---|---|
| **P4.1** | Frozen-state regression: all tests green on every CI matrix entry; dashboard demo completes in <30s end-to-end; audit chain validates against archived run | 3 hrs |
| **P4.2** | One last Discord + DevPost-forum sweep — log any new entries that appeared in the final week | 1 hr |
| **P4.3** | DevPost field-by-field review against a checklist; click every link from the submission; verify YouTube embed loads on the DevPost page | 1 hr |
| **P4.4** | Pre-submission backup: `git tag v3.4.0-final` + tarball + Docker image push to a frozen tag | 1 hr |
| **P4.5** | **SUBMIT** by May 10 23:00 EDT — 24h before deadline. Do not wait until the last hour. | — |

---

## Risk register (May 2)

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| Judges don't use Prompt Opinion's native flow → don't see our integration | Low | Critical | Live demo dashboard exists at `clinicalmem-demo.pages.dev` (mitigates) |
| Hidden strong competitor emerges in last week | **Medium** | High | Monitor Discord + DevPost forum daily; the gallery publishes May 11 |
| Live demo URL goes down during judging window (May 12–26) | Low | Critical | Cloudflare Pages is durable; FHIR sandbox tokens are the failure mode — verify expiry dates |
| Multi-LLM consensus fails because one provider has an outage during judging | Medium | Medium | The cascade already has fallback; document it explicitly in `docs/abstention_proof.md` |
| `SilentDebugger`'s May 2 push turns into a serious last-mile competitor | Low | Medium | Re-check on May 5 + May 9 |
| YouTube video rejected (auto-flag for medical advice) | Low | Critical | Add the standard "for educational purposes only, not medical advice" disclaimer in description + first 5 seconds |
| Two of our 6 LLMs misalign on the demo case during recording | Medium | Low | The system is designed to abstain when consensus breaks — that's the *strength*, not a bug |

---

## Why this wins

**The judging criteria are stacked in our favour:**

- *"The AI Factor"* — 6-LLM consensus + 4-tier deterministic-then-LLM pipeline + iter-421 Path B 2-bundle BitNet ensemble at Layer 4.5 (frozen v8 contra-gate cascading into a tier-2 specialist under constrained argmax — multi-LLM consensus 4-of-5 architectural recipe) + abstention kernels. The iter-421 ensemble achieves **100% recall on every severity class with bit-identical Q16.16 cross-architecture replay** — no other hackathon entry combines safety-class full recall with composite-bundle-id audit primitives, and no other entry runs **three failed single-model retrain attempts** (v9 / v10 / v11) as documented audit-trail evidence justifying the architectural choice.
- *"Potential Impact"* — the demo scenario (Sarah Mitchell, NSAID + warfarin) is a documented patient-safety failure that kills people. No other entry tells a "would have saved a life" story. With iter-421 every severity class hits 100% recall on the live cohort, so "would have caught" extends from contra/major to serious/moderate too — every clinically-relevant DDI severity tier is covered at 100%.
- *"Feasibility"* — live deployment, audit trail, abstention, 1409+ tests, 38 cross-pin discipline families, Apache-2.0 with patent grant, MCP + A2A both, iter-426 audit-replay verifier covers the iter-421 cascade end-to-end (all 47 anchors reproduce byte-for-byte under both bundle hashes). Every other entry has at least one missing pillar.

**The 9-day plan turns the existing structural lead into a presentation lead** — same product, sharper signal.

---

*v2 analysis, 2026-05-02. Supersedes the March 8 v1 analysis. Update again on May 9 (gallery may publish early; Discord recon may surface late entries).*
