"""Generate 3 gallery images for DevPost using nano-banana-pro-preview."""
import base64
import os
import pathlib
from google import genai
from google.genai import types

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY", ""))

IMAGES = [
    {
        "filename": "gallery_architecture.png",
        "prompt": (
            "Create a clean, professional software architecture infographic on a white/light gray (#f8fafc) background. "
            "Title: 'ClinicalMem v4.0 Architecture' in bold dark text. "
            "Show a vertical flow diagram with three input boxes at top (FHIR R4 Patient Data in blue, "
            "Medical APIs: RxNorm/SNOMED/UMLS/openFDA/ClinicalTrials.gov in teal, "
            "6-Model LLM Consensus: GPT-5.4/Gemini 3.1/Grok 4.1/Claude Opus 4.6/Perplexity Sonar in purple). "
            "Arrows flow down into a large central rounded rectangle labeled 'Shared Engine (13 Modules)' containing a grid of module cards: "
            "Clinical Memory, MIND Kernels, Drug Safety (4-Tier), LLM Synthesis, RxNorm Client, SNOMED CT Client, UMLS Mapper, "
            "Consensus Engine, FDA Client, Trials Client, What-If Simulator, PHI Detector, Hallucination Detector. "
            "Below that, a horizontal bar labeled 'SHA-256 Audit Trail (Merkle Chain)' in red. "
            "At bottom, two output boxes side by side: 'MCP Server (18 Tools)' and 'A2A Agent (13 Tools)'. "
            "Use a red heart with white cross icon as the logo. "
            "Style: flat design, rounded corners, subtle shadows, Inter font style, medical/clinical feel with red (#dc2626) accents, "
            "light backgrounds, no gradients on background. Modern SaaS dashboard aesthetic. 1200x800 pixels."
        ),
    },
    {
        "filename": "gallery_findings.png",
        "prompt": (
            "Create a clean, professional clinical safety dashboard infographic on a white/light gray (#f8fafc) background. "
            "Title: 'ClinicalMem Safety Findings — Sarah Mitchell' in bold dark text with a red heart+cross icon. "
            "Show a patient card at top: 'Sarah Mitchell, 67F, 4 Providers, 7 Medications' with badges '2 CRITICAL' (red) and '2 HIGH' (amber). "
            "Below, show 4 finding cards in a 2x2 grid: "
            "1. 'Warfarin + Ibuprofen' tagged CRITICAL in red — 'Bleeding risk, verified by 6-model LLM consensus' "
            "2. 'Penicillin Allergy + Amoxicillin' tagged CRITICAL in red — 'Cross-reactivity via SNOMED CT drug class hierarchy' "
            "3. 'Declining GFR + Metformin' tagged HIGH in amber — 'eGFR 45→32, approaching contraindication' "
            "4. 'Conflicting BP Targets' tagged HIGH in amber — 'Cardiologist <130/80 vs Nephrologist <140/90' "
            "Below the grid, show a 'What-If Simulation' card: 'Substitute Ibuprofen → Acetaminophen: 0 new conflicts (safe)' in green. "
            "Style: flat design, white cards with subtle borders and shadows, rounded corners, medical/clinical feel, "
            "red (#dc2626) for critical, amber (#d97706) for high, green (#16a34a) for safe. Modern dashboard aesthetic. 1200x800 pixels."
        ),
    },
    {
        "filename": "gallery_pipeline.png",
        "prompt": (
            "Create a clean, professional infographic showing a 6-layer safety pipeline on a white/light gray (#f8fafc) background. "
            "Title: '6-Layer Clinical Safety Pipeline' in bold dark text with a red heart+cross icon. "
            "Show 6 horizontal layers stacked vertically, each as a rounded card with a number circle on the left: "
            "Layer 1: 'Deterministic Table' — green (#16a34a) — 'Rule-based, <1ms, never hallucinates' "
            "Layer 2: 'OpenEvidence API' — blue (#2563eb) — 'Mayo Clinic / Elsevier ClinicalKey AI, ~2s' "
            "Layer 3: 'RxNorm + NIH Drug API' — teal (#0d9488) — 'Drug normalization, pairwise interactions, ~1s' "
            "Layer 4: 'Six-Model LLM Consensus' — purple (#7c3aed) — 'GPT-5.4, Gemini 3.1 Pro, Grok 4.1, Claude Opus 4.6, Perplexity Sonar, Gemini Flash — all US-based, ~3s' "
            "Layer 5: 'LLM Synthesis' — purple (#7c3aed) — 'Evidence-cited clinical explanations, ~3s' "
            "Layer 6: 'Abstention Gate' — red (#dc2626) — 'Refuses to guess when evidence insufficient, 0ms' "
            "At bottom, a legend bar with colored dots: Deterministic | Evidence APIs | LLM Consensus | Safety Gate. "
            "Also show a subtitle: '356 tests | 18 MCP tools | 13 A2A tools | Live on Azure'. "
            "Style: flat design, subtle shadows, rounded corners, left-aligned numbers in colored circles, "
            "light background, modern SaaS aesthetic. 1200x800 pixels."
        ),
    },
]

for img_info in IMAGES:
    print(f"Generating {img_info['filename']}...")
    response = client.models.generate_content(
        model="nano-banana-pro-preview",
        contents=img_info["prompt"],
        config=types.GenerateContentConfig(response_modalities=["IMAGE", "TEXT"]),
    )
    for part in response.candidates[0].content.parts:
        if hasattr(part, "inline_data") and part.inline_data and part.inline_data.data:
            data = part.inline_data.data
            if isinstance(data, str):
                data = base64.b64decode(data)
            path = pathlib.Path(img_info["filename"])
            path.write_bytes(data)
            print(f"  Saved {path} ({len(data)} bytes)")
            break
    else:
        print(f"  WARNING: No image data in response for {img_info['filename']}")

print("Done!")
