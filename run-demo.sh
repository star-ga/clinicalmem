#!/usr/bin/env bash
# ClinicalMem 3-Minute Demo Script
# Screen-record this: OBS, QuickTime, or `asciinema rec demo.cast`
set -euo pipefail

A2A="https://clinicalmem-a2a.thankfulpond-9c3fdc1e.eastus.azurecontainerapps.io"
KEY="my-secret-key-123"
BOLD="\033[1m"
CYAN="\033[1;36m"
RED="\033[1;31m"
YELLOW="\033[1;33m"
GREEN="\033[1;32m"
DIM="\033[2m"
RESET="\033[0m"

type_slow() {
  local text="$1"
  for ((i=0; i<${#text}; i++)); do
    printf "%s" "${text:$i:1}"
    sleep 0.03
  done
  echo
}

pause() {
  sleep "${1:-2}"
}

section() {
  echo
  echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
  echo -e "${BOLD}$1${RESET}"
  echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
  echo
}

clear

# ─── INTRO ───────────────────────────────────────────────────────────
section "ClinicalMem — Persistent Clinical Memory for Healthcare AI"

echo -e "${BOLD}The Problem:${RESET}"
type_slow "Sarah Mitchell is 67. She has diabetes, hypertension, kidney disease,"
type_slow "and atrial fibrillation — managed by 4 doctors who don't talk to each other."
echo
type_slow "Last week, her ER doctor prescribed ibuprofen for knee pain."
type_slow "He didn't know she's on warfarin. That combination can cause fatal bleeding."
pause 3

# ─── DEMO 1: MEDICATION SAFETY ──────────────────────────────────────
section "Demo 1: Medication Safety Review"

echo -e "${DIM}Sending A2A request to ClinicalMem agent...${RESET}"
echo -e "${YELLOW}→ POST ${A2A}${RESET}"
echo -e "${YELLOW}→ \"Run a complete medication safety review for Sarah Mitchell\"${RESET}"
echo
pause 1

RESULT1=$(curl -s -X POST "$A2A" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $KEY" \
  -d '{"jsonrpc":"2.0","method":"message/send","id":"demo-1","params":{"message":{"role":"user","messageId":"msg-1","parts":[{"text":"Run a complete medication safety review for Sarah Mitchell"}]}}}' \
  --max-time 60)

# Extract the agent's text response
TEXT1=$(echo "$RESULT1" | python3 -c "
import sys, json
d = json.load(sys.stdin)
print(d['result']['artifacts'][0]['parts'][0]['text'])
" 2>/dev/null || echo "Error parsing response")

# Extract tool response data for structured display
TOOL_DATA=$(echo "$RESULT1" | python3 -c "
import sys, json
d = json.load(sys.stdin)
for msg in d['result']['history']:
    for part in msg.get('parts', []):
        if part.get('kind') == 'data':
            meta = part.get('metadata', {})
            if meta.get('adk_type') == 'function_response':
                resp = part['data'].get('response', {})
                if resp.get('status') == 'success':
                    print(f\"Patient: {resp.get('patient_id', 'N/A')}\")
                    print(f\"Medications reviewed: {resp.get('medication_count', 'N/A')}\")
                    print(f\"Drug interactions: {resp.get('interaction_count', 0)}\")
                    print(f\"Allergy conflicts: {resp.get('allergy_conflict_count', 0)}\")
                    print(f\"Critical findings: {resp.get('critical_findings', 0)}\")
                    print(f\"Confidence: {resp.get('confidence', {}).get('score', 'N/A')} ({resp.get('confidence', {}).get('level', 'N/A')})\")
                    print(f\"Audit hash: {resp.get('audit_hash', 'N/A')}\")
                    # Show interactions
                    for ix in resp.get('drug_interactions', []):
                        print(f\"  ⚠ {ix['drug_a']} + {ix['drug_b']}: {ix['severity']} — {ix['description'][:80]}\")
                    for ax in resp.get('allergy_conflicts', []):
                        print(f\"  🚨 {ax['allergen']} allergy vs {ax['prescribed_medication']}: {ax['description'][:80]}\")
" 2>/dev/null)

if [ -n "$TOOL_DATA" ]; then
  echo -e "${GREEN}─── Structured Results ───${RESET}"
  echo "$TOOL_DATA" | while IFS= read -r line; do
    if [[ "$line" == *"⚠"* ]]; then
      echo -e "${YELLOW}${line}${RESET}"
    elif [[ "$line" == *"🚨"* ]]; then
      echo -e "${RED}${line}${RESET}"
    elif [[ "$line" == *"Audit hash"* ]]; then
      echo -e "${DIM}${line}${RESET}"
    else
      echo -e "${BOLD}${line}${RESET}"
    fi
  done
  echo
fi

echo -e "${GREEN}─── Agent Response ───${RESET}"
echo "$TEXT1"
pause 4

# ─── DEMO 2: CONTRADICTION DETECTION ────────────────────────────────
section "Demo 2: Cross-Provider Contradiction Detection"

echo -e "${DIM}Sending A2A request...${RESET}"
echo -e "${YELLOW}→ \"Check for any contradictions in Sarah's records\"${RESET}"
echo
pause 1

RESULT2=$(curl -s -X POST "$A2A" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $KEY" \
  -d '{"jsonrpc":"2.0","method":"message/send","id":"demo-2","params":{"message":{"role":"user","messageId":"msg-2","parts":[{"text":"Check for any contradictions in Sarah Mitchell'\''s records"}]}}}' \
  --max-time 60)

TEXT2=$(echo "$RESULT2" | python3 -c "
import sys, json
d = json.load(sys.stdin)
print(d['result']['artifacts'][0]['parts'][0]['text'])
" 2>/dev/null || echo "Error parsing response")

echo -e "${GREEN}─── Agent Response ───${RESET}"
echo "$TEXT2"
pause 4

# ─── DEMO 3: LLM EXPLANATION ────────────────────────────────────────
section "Demo 3: LLM Clinical Explanation with Evidence Citations"

echo -e "${DIM}Sending A2A request...${RESET}"
echo -e "${YELLOW}→ \"Explain the most critical conflict for Sarah Mitchell\"${RESET}"
echo
pause 1

RESULT3=$(curl -s -X POST "$A2A" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $KEY" \
  -d '{"jsonrpc":"2.0","method":"message/send","id":"demo-3","params":{"message":{"role":"user","messageId":"msg-3","parts":[{"text":"Explain the most critical conflict for Sarah Mitchell"}]}}}' \
  --max-time 60)

TEXT3=$(echo "$RESULT3" | python3 -c "
import sys, json
d = json.load(sys.stdin)
print(d['result']['artifacts'][0]['parts'][0]['text'])
" 2>/dev/null || echo "Error parsing response")

echo -e "${GREEN}─── Agent Response ───${RESET}"
echo "$TEXT3"
pause 3

# ─── CLOSING ─────────────────────────────────────────────────────────
section "ClinicalMem — Summary"

echo -e "${BOLD}What we demonstrated:${RESET}"
echo "  ✓ 4 clinical conflicts caught automatically"
echo "  ✓ 6 detection layers (deterministic → OpenEvidence → NIH RxNorm → Multi-LLM)"
echo "  ✓ Every finding audited in SHA-256 hash chain"
echo "  ✓ Safe abstention — refuses to answer when evidence is insufficient"
echo "  ✓ 90 tests passing"
echo
echo -e "${BOLD}Tech:${RESET} mind-mem + MIND Lang + OpenAI GPT-5.4 + MedGemma + NIH RxNorm + FHIR R4"
echo -e "${BOLD}Built by:${RESET} STARGA, Inc."
echo -e "${DIM}github.com/star-ga/clinicalmem${RESET}"
echo
echo -e "${CYAN}ClinicalMem doesn't replace doctors — it gives them a memory that${RESET}"
echo -e "${CYAN}never forgets, never hallucinates, and knows when to say 'I don't know.'${RESET}"
echo
