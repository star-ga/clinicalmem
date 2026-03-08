"""Generate a single continuous voiceover track using edge-tts, then merge with demo video."""
import asyncio
import os
import subprocess

import edge_tts

# One continuous narration — pauses are handled by silence gaps between segments
segments = [
    "Sarah Mitchell is 67 years old. She has diabetes, hypertension, kidney disease, and atrial fibrillation — managed by four different doctors who don't talk to each other. Last week, her ER doctor prescribed ibuprofen for knee pain. He didn't know she's on warfarin. That combination can cause fatal bleeding.",

    "ClinicalMem is a persistent clinical memory layer for healthcare AI agents. It ingests patient data from FHIR, runs it through a six-layer safety pipeline, and catches conflicts that individual providers miss.",

    "We're now sending a medication safety review to the live ClinicalMem agent.",

    "Two critical findings. First, Sarah has a documented penicillin allergy, but urgent care prescribed amoxicillin — a beta-lactam cross-reactant. That's anaphylaxis risk. Second, ibuprofen plus warfarin — a serious bleeding risk that the ER doctor missed.",

    "Now checking for contradictions across all four providers.",

    "Nine findings detected. Her GFR has been declining from 45 to 32, approaching the threshold where metformin becomes contraindicated. And her cardiologist wants blood pressure below 130 over 80, but her nephrologist says below 140 over 90. A ten-point disagreement nobody flagged.",

    "The synthesis layer uses a medical LLM cascade to explain each conflict. Every claim cites a specific piece of Sarah's record. If evidence were insufficient, the system would refuse to answer. In healthcare, I don't know saves lives.",

    "Four conflicts caught. Six detection layers. Every finding audited in a tamper-proof hash chain. ClinicalMem doesn't replace doctors — it gives them a memory that never forgets, never hallucinates, and knows when to say I don't know.",
]

VOICE = "en-US-GuyNeural"
PAUSE_SECONDS = 1.5  # silence between segments


async def generate_segments():
    paths = []
    for i, text in enumerate(segments):
        print(f"Generating segment {i+1}/{len(segments)}...")
        path = f"vo_segment_{i:02d}.mp3"
        communicate = edge_tts.Communicate(text, VOICE, rate="-5%")
        await communicate.save(path)
        # Get duration
        r = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration", "-of", "csv=p=0", path],
            capture_output=True, text=True,
        )
        dur = float(r.stdout.strip())
        print(f"  {path} — {dur:.1f}s")
        paths.append(path)
    return paths


paths = asyncio.run(generate_segments())

# Generate silence file
print(f"\nGenerating {PAUSE_SECONDS}s silence gap...")
subprocess.run([
    "ffmpeg", "-y", "-f", "lavfi", "-i",
    f"anullsrc=r=44100:cl=mono",
    "-t", str(PAUSE_SECONDS),
    "-c:a", "libmp3lame", "-q:a", "9",
    "silence.mp3",
], capture_output=True)

# Build concat list: segment, silence, segment, silence, ...
print("Building concat list...")
with open("concat_list.txt", "w") as f:
    for i, path in enumerate(paths):
        f.write(f"file '{path}'\n")
        if i < len(paths) - 1:
            f.write(f"file 'silence.mp3'\n")

# Concatenate into single track
print("Concatenating into single voiceover track...")
r = subprocess.run([
    "ffmpeg", "-y", "-f", "concat", "-safe", "0",
    "-i", "concat_list.txt",
    "-c:a", "libmp3lame", "-q:a", "2",
    "voiceover_combined.mp3",
], capture_output=True, text=True)
if r.returncode != 0:
    print(f"Concat error: {r.stderr[-500:]}")
    raise SystemExit(1)

# Check voiceover duration
r = subprocess.run(
    ["ffprobe", "-v", "quiet", "-show_entries", "format=duration", "-of", "csv=p=0", "voiceover_combined.mp3"],
    capture_output=True, text=True,
)
vo_duration = float(r.stdout.strip())
print(f"Voiceover duration: {vo_duration:.1f}s")

# Get video duration
r = subprocess.run(
    ["ffprobe", "-v", "quiet", "-show_entries", "format=duration", "-of", "csv=p=0", "clinicalmem-demo-slow.mp4"],
    capture_output=True, text=True,
)
video_duration = float(r.stdout.strip())
print(f"Video duration: {video_duration:.1f}s")

# If voiceover is longer than video, speed up video to match
# If voiceover is shorter, that's fine — video continues silently
if vo_duration > video_duration:
    # Slow down the video to match voiceover length
    speed = video_duration / vo_duration
    print(f"Adjusting video speed to {speed:.3f}x to match voiceover...")
    subprocess.run([
        "ffmpeg", "-y",
        "-i", "clinicalmem-demo-slow.mp4",
        "-filter:v", f"setpts={1/speed}*PTS",
        "-an",
        "clinicalmem-demo-matched.mp4",
    ], capture_output=True)
    video_input = "clinicalmem-demo-matched.mp4"
else:
    video_input = "clinicalmem-demo-slow.mp4"

# Merge video + voiceover
print("Merging video + voiceover...")
r = subprocess.run([
    "ffmpeg", "-y",
    "-i", video_input,
    "-i", "voiceover_combined.mp3",
    "-c:v", "copy" if video_input == "clinicalmem-demo-slow.mp4" else "libx264",
    "-c:a", "aac",
    "-b:a", "192k",
    "-shortest",
    "-movflags", "faststart",
    "clinicalmem-demo-final.mp4",
], capture_output=True, text=True)
if r.returncode != 0:
    print(f"Merge error: {r.stderr[-500:]}")
    raise SystemExit(1)

# Final check
r = subprocess.run(
    ["ffprobe", "-v", "quiet", "-show_entries", "format=duration", "-of", "csv=p=0", "clinicalmem-demo-final.mp4"],
    capture_output=True, text=True,
)
final_duration = float(r.stdout.strip())
size = os.path.getsize("clinicalmem-demo-final.mp4")
print(f"\nFinal video: clinicalmem-demo-final.mp4")
print(f"Duration: {final_duration:.1f}s")
print(f"Size: {size / 1024 / 1024:.1f}MB")
print("Done! Upload to YouTube to replace the previous version.")
