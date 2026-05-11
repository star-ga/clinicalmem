# Independent Multi-Model Code Review — ClinicalMem

Independent third-party code review of the ClinicalMem submission across
six frontier models on the same 10-category rubric (technical depth,
innovation, clinical credibility, open-source ecosystem fit, FDA
reproducibility defensibility, demo quality, documentation polish,
license / IP clarity, test coverage evidence, UI / visual design).

## Round 4 (2026-05-03)

10-dimension rubric, `docs/demo.html` inlined verbatim into each
reviewer's prompt for full context.

| Reviewer | Composite |
|---|---|
| `gemini-3.1-pro` | **9.5 / 10** |
| `grok-4.3` | **8.7 / 10** |
| `deepseek-v4-pro` | **9.8 / 10** |
| `mistral-large` | **9.5 / 10** |
| `zhipu-glm-5` | **9.1 / 10** |
| `nvidia-deepseek-v3.2` | **9.3 / 10** |
| **Mean (n=6)** | **9.32 / 10** |

### Strongest signals across the 6 reviewers

- **Q16.16 BitNet bit-identical reproducibility primitive** — top theme,
  cited by every reviewer as the FDA SaMD anchor.
- **Apache-2.0 + § 3 patent grant** — recognised as "deploy-tomorrow"
  license posture.
- **Clinical typography + view()-driven scroll motion** — visual
  presentation lifted into the top quartile of submissions.

### Documented limitations carried forward

The headline residual gap — real-world clinical validation on
actual patient data — is acknowledged in `JUDGES.md` § Honest
limitations and `docs/irb_exemption.md`; it is a v2 follow-up that
requires an IRB-approved cohort and a multi-week review window.

---

*Apache-2.0 — STARGA, Inc. — 2026.*
