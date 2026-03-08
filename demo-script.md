# ClinicalMem Demo Script (3 minutes)

> Record with screen share + voiceover. Show the A2A agent chat interface on Prompt Opinion.

---

## [0:00 - 0:20] Hook — The Problem

**Voiceover:**
"Sarah Mitchell is 67 years old. She has diabetes, hypertension, kidney disease, and atrial fibrillation — managed by four different doctors who don't talk to each other. Last week, her ER doctor prescribed ibuprofen for knee pain. He didn't know she's on warfarin. That combination can cause fatal bleeding."

**Screen:** Show Sarah's patient summary (conditions, 4 providers, medications list)

---

## [0:20 - 0:50] The Solution — ClinicalMem

**Voiceover:**
"ClinicalMem is a persistent clinical memory layer for healthcare AI agents. It ingests patient data from FHIR, runs it through a six-layer safety pipeline, and catches conflicts that individual providers miss."

**Screen:** Show architecture diagram or the tool list (11 MCP tools / 5 A2A skills)

---

## [0:50 - 1:30] Demo — Medication Safety Review

**Action:** Type: "Run a complete medication safety review for Sarah Mitchell"

**Voiceover (as results appear):**
"The medication safety review runs instantly. Layer 1 — our deterministic table — catches ibuprofen plus warfarin in microseconds. That's a serious bleeding risk. It also flags the amoxicillin prescribed by urgent care — Sarah has a documented penicillin allergy. Amoxicillin is a beta-lactam cross-reactant. That's anaphylaxis risk."

**Screen:** Show the drug interactions and allergy conflicts in the response, with severity scores and audit hash.

---

## [1:30 - 2:10] Demo — Contradiction Detection

**Action:** Type: "Check for any contradictions in Sarah's records"

**Voiceover (as results appear):**
"Now ClinicalMem scans across all four providers. It finds Sarah's GFR has been declining — 45, then 38, then 32 — approaching the threshold where metformin becomes contraindicated for lactic acidosis risk. And it catches that her cardiologist wants blood pressure below 130 over 80, but her nephrologist says below 140 over 90. That's a 10-point disagreement that nobody flagged."

**Screen:** Show contradiction results — 5 types detected, escalation message, audit hash, chain integrity = verified.

---

## [2:10 - 2:40] Demo — LLM Explanation with Evidence Citations

**Action:** Type: "Explain the most critical conflict"

**Voiceover:**
"Now the synthesis layer kicks in. ClinicalMem uses MedGemma — Google's purpose-built medical model — to generate a patient-specific explanation. But it only cites evidence it actually found. See those block IDs in brackets? Every claim traces back to a specific piece of Sarah's record. And if the evidence were insufficient, the system would refuse to answer. In healthcare, 'I don't know' saves lives."

**Screen:** Show the narrative with [block_id] citations, confidence score, model_used field, abstained: false.

---

## [2:40 - 3:00] Closing — Why This Matters

**Voiceover:**
"Four conflicts caught. Six detection layers. Every finding audited in a tamper-proof hash chain. ClinicalMem doesn't replace doctors — it gives them a persistent memory that never forgets, never hallucinates, and knows when to say 'I don't know.' Built on mind-mem and MIND Lang by STARGA."

**Screen:** Show the audit trail with SHA-256 hashes, then the GitHub repo URL.

---

## Recording Tips

- Use Prompt Opinion chat interface if possible (shows A2A agent in action)
- If PO isn't ready, use a terminal with `curl` commands to the MCP server
- Keep responses visible long enough for judges to read key fields
- Highlight: audit_hash, chain_integrity, model_used, severity scores
- Total runtime target: 2:45 - 3:00
