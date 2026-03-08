"""Generate DevPost gallery images for ClinicalMem."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np


def set_style(fig, ax):
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#0d1117')
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis('off')


def draw_box(ax, x, y, w, h, label, color, sublabel=None, fontsize=11):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15",
                         facecolor=color, edgecolor='white', linewidth=1.5, alpha=0.9)
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2 + (0.15 if sublabel else 0), label,
            ha='center', va='center', fontsize=fontsize, fontweight='bold', color='white')
    if sublabel:
        ax.text(x + w/2, y + h/2 - 0.25, sublabel,
                ha='center', va='center', fontsize=8, color='#b0b0b0')


def arrow(ax, x1, y1, x2, y2):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color='#58a6ff', lw=2))


# ─── Image 1: Architecture Diagram ──────────────────────────────────
fig, ax = plt.subplots(1, 1, figsize=(12, 8))
set_style(fig, ax)

# Title
ax.text(6, 7.5, 'ClinicalMem Architecture', ha='center', va='center',
        fontsize=22, fontweight='bold', color='white')

# FHIR Source
draw_box(ax, 0.3, 5.5, 2.2, 1.2, 'FHIR R4', '#1f6feb', 'Patient Data')

# Shared Engine
draw_box(ax, 3.5, 4.8, 5, 2.5, '', '#161b22')
ax.text(6, 6.9, 'Shared Engine', ha='center', va='center',
        fontsize=14, fontweight='bold', color='#58a6ff')

# Engine subcomponents
draw_box(ax, 3.7, 5.5, 2.2, 0.9, 'Clinical Memory', '#21262d', 'BM25 + Vector', fontsize=9)
draw_box(ax, 6.1, 5.5, 2.2, 0.9, 'MIND Kernels', '#21262d', 'Scoring', fontsize=9)
draw_box(ax, 3.7, 4.9, 2.2, 0.5, 'Drug Interactions', '#21262d', fontsize=9)
draw_box(ax, 6.1, 4.9, 2.2, 0.5, 'LLM Synthesizer', '#21262d', fontsize=9)

# MCP Server
draw_box(ax, 1, 2, 3.5, 1.5, 'MCP Server', '#238636', '11 Tools · FastMCP 2.x')

# A2A Agent
draw_box(ax, 5.5, 2, 3.5, 1.5, 'A2A Agent', '#8b5cf6', '5 Skills · Google ADK')

# Audit Trail
draw_box(ax, 9.5, 5.5, 2.2, 1.2, 'Audit Trail', '#da3633', 'SHA-256 Chain')

# External APIs
draw_box(ax, 9.5, 2, 2.2, 1.5, 'Medical APIs', '#d29922',
         'NIH · OpenEvidence\nGPT-5.4 · MedGemma', fontsize=10)

# Arrows
arrow(ax, 2.5, 6.1, 3.5, 6.1)  # FHIR -> Engine
arrow(ax, 6, 4.8, 2.75, 3.5)   # Engine -> MCP
arrow(ax, 6, 4.8, 7.25, 3.5)   # Engine -> A2A
arrow(ax, 8.5, 5.5, 9.5, 5.7)  # Engine -> Audit
arrow(ax, 8.5, 5.1, 9.5, 2.8)  # Engine -> APIs

# Bottom label
ax.text(6, 0.5, 'Deterministic Detection + LLM Synthesis + Safe Abstention',
        ha='center', va='center', fontsize=12, color='#8b949e', style='italic')

plt.tight_layout()
plt.savefig('gallery_architecture.png', dpi=150, bbox_inches='tight',
            facecolor='#0d1117', edgecolor='none')
plt.close()
print("Saved gallery_architecture.png")


# ─── Image 2: Six-Layer Safety Pipeline ─────────────────────────────
fig, ax = plt.subplots(1, 1, figsize=(12, 8))
set_style(fig, ax)

ax.text(6, 7.5, 'Six-Layer Clinical Safety Pipeline', ha='center', va='center',
        fontsize=22, fontweight='bold', color='white')

layers = [
    ('Layer 1: Deterministic Table', 'Known drug pairs in microseconds', '#238636', '< 1ms'),
    ('Layer 2: OpenEvidence API', 'Mayo Clinic / Elsevier ClinicalKey AI', '#1f6feb', '~2s'),
    ('Layer 3: NIH RxNorm API', 'Federal gold standard (Epic/Cerner)', '#1f6feb', '~1s'),
    ('Layer 4: Multi-LLM Cascade', 'GPT-5.4 → MedGemma → Gemini Flash', '#8b5cf6', '~3s'),
    ('Layer 5: LLM Synthesis', 'Evidence-cited clinical explanations', '#8b5cf6', '~3s'),
    ('Layer 6: Abstention Gate', '"I don\'t know" when evidence insufficient', '#da3633', '0ms'),
]

for i, (name, desc, color, timing) in enumerate(layers):
    y = 6.3 - i * 1.05
    w = 8
    x = 2
    box = FancyBboxPatch((x, y), w, 0.85, boxstyle="round,pad=0.1",
                         facecolor=color, edgecolor='white', linewidth=1.2, alpha=0.85)
    ax.add_patch(box)
    ax.text(x + 0.3, y + 0.45, name, ha='left', va='center',
            fontsize=12, fontweight='bold', color='white')
    ax.text(x + 0.3, y + 0.15, desc, ha='left', va='center',
            fontsize=9, color='#d0d0d0')
    ax.text(x + w - 0.3, y + 0.45, timing, ha='right', va='center',
            fontsize=10, color='#f0f0f0', fontweight='bold')

    if i < len(layers) - 1:
        ax.annotate('', xy=(6, y), xytext=(6, y + 0.0),
                    arrowprops=dict(arrowstyle='->', color='#58a6ff', lw=1.5))

# Legend
ax.text(2, 0.5, 'Green = Deterministic (never hallucinate)   ',
        ha='left', va='center', fontsize=10, color='#238636')
ax.text(6, 0.5, 'Purple = LLM-powered   ',
        ha='left', va='center', fontsize=10, color='#8b5cf6')
ax.text(9, 0.5, 'Red = Safety gate',
        ha='left', va='center', fontsize=10, color='#da3633')

plt.tight_layout()
plt.savefig('gallery_pipeline.png', dpi=150, bbox_inches='tight',
            facecolor='#0d1117', edgecolor='none')
plt.close()
print("Saved gallery_pipeline.png")


# ─── Image 3: Demo Findings Summary ─────────────────────────────────
fig, ax = plt.subplots(1, 1, figsize=(12, 8))
set_style(fig, ax)

ax.text(6, 7.5, 'Sarah Mitchell — 4 Conflicts Detected', ha='center', va='center',
        fontsize=22, fontweight='bold', color='white')
ax.text(6, 7.0, '67yo · T2DM · HTN · CKD 3b · AFib · 4 Providers · 9 Medications',
        ha='center', va='center', fontsize=11, color='#8b949e')

findings = [
    ('CRITICAL', 'Warfarin + Ibuprofen', 'Serious bleeding risk — ER prescribed NSAID\nwithout checking anticoagulant therapy', '#da3633'),
    ('CRITICAL', 'Penicillin Allergy + Amoxicillin', 'Beta-lactam cross-reactivity — Urgent Care\nprescribed without checking allergy history', '#da3633'),
    ('HIGH', 'Declining GFR + Metformin', 'eGFR 45→38→32 over 6 months — approaching\nmetformin contraindication threshold', '#d29922'),
    ('HIGH', 'Conflicting BP Targets', 'Cardiologist: <130/80 vs Nephrologist: <140/90\n10-point provider disagreement unflagged', '#d29922'),
]

for i, (severity, title, desc, color) in enumerate(findings):
    y = 5.6 - i * 1.45
    # Severity badge
    badge = FancyBboxPatch((0.5, y + 0.35), 1.8, 0.5, boxstyle="round,pad=0.08",
                           facecolor=color, edgecolor='none', alpha=0.9)
    ax.add_patch(badge)
    ax.text(1.4, y + 0.6, severity, ha='center', va='center',
            fontsize=10, fontweight='bold', color='white')
    # Finding box
    box = FancyBboxPatch((2.5, y), 9, 1.2, boxstyle="round,pad=0.12",
                         facecolor='#161b22', edgecolor=color, linewidth=2, alpha=0.95)
    ax.add_patch(box)
    ax.text(2.8, y + 0.85, title, ha='left', va='center',
            fontsize=13, fontweight='bold', color='white')
    ax.text(2.8, y + 0.35, desc, ha='left', va='center',
            fontsize=9, color='#b0b0b0', linespacing=1.4)

# Audit footer
ax.text(6, 0.4, 'SHA-256 Audit: 1454275c...ddf903  |  Chain Integrity: VERIFIED  |  90 Tests Passing',
        ha='center', va='center', fontsize=10, color='#58a6ff',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#161b22', edgecolor='#58a6ff', alpha=0.8))

plt.tight_layout()
plt.savefig('gallery_findings.png', dpi=150, bbox_inches='tight',
            facecolor='#0d1117', edgecolor='none')
plt.close()
print("Saved gallery_findings.png")

print("\nAll 3 gallery images generated:")
print("  1. gallery_architecture.png  — System architecture")
print("  2. gallery_pipeline.png      — Six-layer safety pipeline")
print("  3. gallery_findings.png      — Demo findings summary")
