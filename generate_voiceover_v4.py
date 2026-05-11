"""Generate the v4 2:40 demo voiceover via edge-tts.

Same voice (`en-US-GuyNeural`) and base rate as v1/v2 — narration is
*timeline-locked* to the on-screen action budget per section.

Per-section budget (matches demo-script-v4.md timing summary):
    §1 hook         0:00 – 0:12  (12s)
    §2 sarah        0:12 – 0:40  (28s)
    §3 catch        0:40 – 1:10  (30s)
    §4 contradict   1:10 – 1:40  (30s)
    §5 federation   1:40 – 2:00  (20s)
    §6 layer-4.5    2:00 – 2:25  (25s)
    §7 close        2:25 – 2:40  (15s)
                    -----------
                    total 2:40

For each section:
  1. Render the narration at the chosen rate.
  2. Probe duration.
  3. If shorter than its time budget → pad with TRAILING silence so the
     screen action has time to land before the next section starts.
  4. If longer than its time budget → re-render at a faster rate (rate
     bump in 5% increments, capped at +30%) and re-check.

Run: python3 generate_voiceover_v4.py
Outputs:
  vo_v4_NN_label.mp3   (one per section, exact-budget length)
  voiceover_v4.mp3     (concatenated, 2:40 total)

Requires: edge-tts >= 7.x, ffmpeg / ffprobe on PATH.
"""
from __future__ import annotations

import asyncio
import subprocess
import sys
from pathlib import Path

import edge_tts

# Same voice + same rate as v1/v2 — narrator must sound IDENTICAL across
# the hackathon's video lineage. Rate bumping is forbidden — if a
# section's text doesn't fit its budget at -5%, the TEXT gets trimmed,
# not the voice sped up. (A +25% rate makes the same TTS model sound
# chipmunky-fast and is perceived as a different narrator.)
VOICE = "en-US-GuyNeural"
BASE_RATE = "-5%"
MAX_RATE_BUMP = 0  # locked — never bump the rate

REPO_ROOT = Path(__file__).resolve().parent
OUT_DIR = REPO_ROOT  # write next to the script for parity with v2

# (label, on-screen budget seconds, narration text)
# Narration matches demo-script-v4.md sections verbatim.
SEGMENTS: list[tuple[str, int, str]] = [
    (
        "S1_hook",
        12,
        "A doctor just prescribed a drug that could kill a patient. "
        "They didn't know. ClinicalMem knew — in under a second.",
    ),
    (
        "S2_sarah",
        28,
        # Trimmed harder for -5% fit (target ≤27.5s, no rate bumps).
        "Meet Sarah Mitchell. Sixty-seven. Diabetes, hypertension, kidney "
        "disease, atrial fibrillation. Four providers, no shared records. "
        "Her ER doctor prescribed ibuprofen — didn't know she was on "
        "warfarin. Her urgent-care clinic prescribed amoxicillin — didn't "
        "check her penicillin allergy. Seven thousand Americans die from "
        "exactly these mistakes.",
    ),
    (
        "S3_catch",
        30,
        # Trimmed for -5% fit (target ≤29.5s, no rate bumps).
        "Two critical findings, in under one second. Warfarin plus "
        "ibuprofen: NSAID on a blood-thinner patient. Major bleeding "
        "risk. Caught by layer one — before any LLM saw the case. "
        "Penicillin plus amoxicillin: cross-reactant. Anaphylaxis risk. "
        "Every finding locked in a tamper-proof digital fingerprint — "
        "FDA-verifiable, decades from now.",
    ),
    (
        "S4_contradict",
        30,
        # Trimmed to fit 30s at -5% (same narrator cadence as v2).
        # Cut: "across all four providers" → "across providers"; cut
        # "She's approaching the threshold where metformin becomes
        # contraindicated" → "Metformin contraindication zone."
        "Now we scan across providers. Sarah's kidney function is "
        "declining — eGFR forty-five, thirty-eight, thirty-two. "
        "Metformin contraindication zone. No provider flagged it. "
        "And her cardiologist sets blood-pressure below one-thirty "
        "over eighty. Her nephrologist says below one-forty over "
        "ninety. Same patient, same week. Ten-point gap. Neither knew.",
    ),
    (
        "S5_federation",
        20,
        # Trimmed for -5% fit (was 20.5s, target ≤19.5s).
        "ClinicalMem spots the conflict — without moving Sarah's private "
        "data. It propagates as de-identified clinical knowledge across "
        "providers, sealed with military-grade encryption, gated by "
        "twenty-one typed runtime safety checks. HIPAA built into the "
        "code, not a checkbox on a form.",
    ),
    (
        "S6_verify",
        25,
        # Trimmed for -5% fit. Cut "five-dollar"/"Decades from now"
        # phrasing — kept the green-match payoff.
        "Layer four-point-five runs identically on every device. "
        "One hundred eighteen kilobytes. Under one millisecond on a "
        "Raspberry Pi Zero. One hundred percent recall — forty-four "
        "out of forty-four — zero false positives. Click Verify Replay. "
        "Your browser computes the same hash the server recorded. "
        "Green match.",
    ),
    (
        "S7_close",
        15,
        # Trimmed for -5% fit. Kept the killer "I wish I had known" line.
        "ClinicalMem doesn't replace doctors. It makes sure they never "
        "have to say: I wish I had known. ClinicalMem. Apache 2.",
    ),
]


async def _probe_duration(path: Path) -> float:
    """Return audio duration in seconds via ffprobe."""
    proc = await asyncio.create_subprocess_exec(
        "ffprobe", "-v", "quiet", "-show_entries", "format=duration",
        "-of", "csv=p=0", str(path),
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
    )
    out, _ = await proc.communicate()
    return float(out.decode().strip())


def _bump_rate(base: str, n: int) -> str:
    """Step the edge-tts rate string by `n` × 5% (positive = faster)."""
    base_pct = int(base.rstrip("%"))
    new_pct = base_pct + n * 5
    return f"{new_pct:+d}%"


async def _render_within_budget(label: str, budget: int, text: str) -> Path:
    """Render `text` so audio duration is <= `budget` seconds.

    Steps the rate up in +5% increments (capped at MAX_RATE_BUMP) until
    the duration fits, then returns the rendered path. The audio is NOT
    yet padded to budget length — that happens in the concat step.
    """
    rate = BASE_RATE
    path = OUT_DIR / f"vo_v4_{label}.mp3"
    for bump in range(MAX_RATE_BUMP + 1):
        if bump > 0:
            rate = _bump_rate(BASE_RATE, bump)
        comm = edge_tts.Communicate(text, VOICE, rate=rate)
        await comm.save(str(path))
        dur = await _probe_duration(path)
        # Reserve 0.5s headroom so the section doesn't bleed into the next
        if dur <= budget - 0.5:
            print(f"  ✓ {label}  rate={rate}  speech={dur:.2f}s "
                  f"budget={budget}s  pad={budget - dur:.2f}s")
            return path
        print(f"  ↻ {label}  rate={rate}  speech={dur:.2f}s > "
              f"budget={budget - 0.5:.1f}s — bumping rate")
    print(f"  ⚠ {label}  hit max rate bump; speech={dur:.2f}s "
          f"vs budget {budget}s — narration may overlap next section")
    return path


def _make_silence(seconds: float, path: Path) -> None:
    """Generate a silent MP3 of given length (mono, 24 kHz)."""
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i",
         "anullsrc=r=24000:cl=mono", "-t", f"{seconds:.3f}",
         "-q:a", "9", str(path)],
        check=True, capture_output=True,
    )


async def main() -> None:
    print(f"Voice: {VOICE}  base rate: {BASE_RATE}")
    print(f"Sections: {len(SEGMENTS)}  total budget: "
          f"{sum(b for _, b, _ in SEGMENTS)}s\n")

    rendered: list[tuple[str, int, Path, float]] = []
    for label, budget, text in SEGMENTS:
        print(f"[{label}] budget={budget}s  text={len(text)} chars")
        path = await _render_within_budget(label, budget, text)
        dur = await _probe_duration(path)
        rendered.append((label, budget, path, dur))

    # Build the concat list with per-section trailing silence so each
    # section ends EXACTLY at its budget boundary.
    print("\nPadding sections to budget boundaries:")
    pad_paths: list[Path] = []
    concat_entries: list[Path] = []
    for label, budget, path, dur in rendered:
        concat_entries.append(path)
        pad = max(0.0, budget - dur)
        if pad > 0.05:
            silence = OUT_DIR / f"vo_v4_{label}_pad.mp3"
            _make_silence(pad, silence)
            pad_paths.append(silence)
            concat_entries.append(silence)
            print(f"  {label}: speech={dur:.2f}s + pad={pad:.2f}s = {budget}s")
        else:
            print(f"  {label}: speech={dur:.2f}s (no pad needed)")

    list_path = OUT_DIR / "concat_v4.txt"
    list_path.write_text(
        "\n".join(f"file '{p}'" for p in concat_entries) + "\n"
    )

    out = OUT_DIR / "voiceover_v4.mp3"
    subprocess.run(
        ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(list_path),
         "-c", "copy", str(out)],
        check=True, capture_output=True,
    )
    final_dur = await _probe_duration(out)
    print(f"\n✓ Wrote {out}  duration={final_dur:.2f}s  target=160s")
    if abs(final_dur - 160) > 1.0:
        print(f"  WARNING: total duration drifted from 2:40 target by "
              f"{final_dur - 160:+.1f}s")


if __name__ == "__main__":
    asyncio.run(main())
