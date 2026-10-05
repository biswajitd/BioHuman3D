"""
Verify the two defects reported from real exports:

1. Every export narrated the *skeleton*, whatever body system was selected.
2. All four exported videos contained byte-identical audio.

Checks:
  A. Every body system maps to its own tour.
  B. No narration paragraph is shared between tours.
  C. Two different systems synthesise *different* audio (hash comparison).
  D. The voice catalogue resolves dialect x gender, and reports a missing
     dialect instead of silently using a different accent.
  E. A real export runs through the quality pipeline (supersampling + ambient
     occlusion) and contains both video and audio streams.

    python tools\\verify_systems.py
"""
from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import tempfile
import wave
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("QT_LOGGING_RULES", "*=false")

for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QApplication

from app.audio.scripts import (BODY_SYSTEMS, BUILTIN_TOURS, SYSTEM_TOUR_MAP,
                               get_tour, subject_for_layer, tour_for_system,
                               tour_list, video_subjects)
from app.audio.voices import (DIALECTS, filter_voices, resolve_voice,
                              catalogue_summary, dialects_available,
                              enumerate_pyttsx3_voices)
from app.config import AppConfig
from app.core.model_registry import LayerRegistry
from app.core.viewport import create_viewport
from app.video.exporter import TourVideoExporter, VideoSettings, video_support_status

PASS, FAIL = [], []


def check(name: str, ok: bool, detail: str = "") -> None:
    (PASS if ok else FAIL).append(name)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""),
          flush=True)


# ---------------------------------------------------------------------------
print("=" * 74)
print(" A/B. Per-system scripts")
print("=" * 74)

missing = [s.label for s in BODY_SYSTEMS if s.tour_id and s.tour_id not in BUILTIN_TOURS]
check("Every body system has a tour", not missing, f"missing: {missing}" if missing else
      f"{len(SYSTEM_TOUR_MAP)} systems mapped")

for label in ("Venous", "Arterial", "Nervous", "Lymphatic", "Urinary"):
    tour = tour_for_system(label)
    check(f"  {label} → its own tour",
          tour is not None and tour.id != "skeletal_overview",
          tour.title if tour else "none")

# Organ-level subjects: "generate a video on the liver" must be possible.
for layer_id, expected in (("liver", "liver_overview"),
                           ("heart", "heart_overview"),
                           ("kidneys", "kidneys_overview"),
                           ("brain", "brain_overview"),
                           ("lungs", "lungs_overview")):
    tour = subject_for_layer(layer_id)
    check(f"  structure '{layer_id}' → {expected}",
          tour is not None and tour.id == expected,
          tour.title if tour else "none")

# The regression that actually caused the repeated skeletal exports: the UI was
# offered a hard-coded subset of tours, so selecting anything else silently failed.
all_tours = tour_list()
check("Every authored tour is offered in the UI",
      len(all_tours) == len(BUILTIN_TOURS),
      f"{len(all_tours)} offered vs {len(BUILTIN_TOURS)} authored")
unreachable = [tid for tid in SYSTEM_TOUR_MAP.values()
               if tid not in {t.id for t in all_tours}]
check("Every system tour is reachable", not unreachable, str(unreachable))
organ_subjects = [s for s in video_subjects() if s[0] == "Organ"]
check("Organ subjects offered", len(organ_subjects) >= 5,
      ", ".join(s[2] for s in organ_subjects))

all_text = []
for tour in BUILTIN_TOURS.values():
    all_text.extend(kf.narration.strip() for kf in tour.keyframes)
duplicates = [text for text, n in Counter(all_text).items() if n > 1]
check("No narration paragraph reused between tours", not duplicates,
      f"{len(all_text)} unique paragraphs across {len(BUILTIN_TOURS)} tours"
      if not duplicates else f"{len(duplicates)} duplicated")

word_counts = {t.title: t.word_count for t in BUILTIN_TOURS.values()}
check("Every script has substance (>= 60 words)", all(w >= 60 for w in word_counts.values()),
      f"min={min(word_counts.values())} max={max(word_counts.values())} words")

# ---------------------------------------------------------------------------
print()
print("=" * 74)
print(" D. Voice catalogue (dialect x gender)")
print("=" * 74)

voices = enumerate_pyttsx3_voices()
print(f"  installed: {catalogue_summary(voices)}", flush=True)
for voice in voices:
    print(f"     - {voice.display}  [{voice.id.split(chr(92))[-1]}]", flush=True)

print(f"  dialects available: {dialects_available(voices)}", flush=True)

female_indian = filter_voices(voices, "en-IN", "female")
male_indian = filter_voices(voices, "en-IN", "male")
print(f"  en-IN female -> {len(female_indian)}", flush=True)
print(f"  en-IN male   -> {len(male_indian)}", flush=True)

if not voices:
    # No SAPI voices on this machine (e.g. Linux CI). Offline voice resolution
    # cannot be exercised; neural voices are the default narration engine.
    print("  [SKIP] system-voice resolution — no offline voices installed on this machine", flush=True)
else:
    resolved, note = resolve_voice(voices, "en-IN", "female")
    check("Dialect/gender resolution returns a voice", resolved is not None,
          resolved.display if resolved else "none")
    if resolved is not None and resolved.dialect != "en-IN":
        check("Missing dialect is reported, not hidden",
              "India" in note and "Falling back" in note,
              note[:110] if note else "no note (a matching voice exists)")
    check("Fallback honours the requested gender", resolved is not None
          and resolved.gender == "female",
          f"asked for Indian English female, got {resolved.display if resolved else 'none'}")

    any_resolved, any_note = resolve_voice(voices, "any", "female")
    check("Gender-only filter works", any_resolved is not None,
          any_resolved.display if any_resolved else "none")

check("Dialect list includes Indian English", any(d.code == "en-IN" for d in DIALECTS))

# ---------------------------------------------------------------------------
print()
print("=" * 74)
print(" C. Two systems synthesise different audio")
print("=" * 74)

app = QApplication(sys.argv[:1])
config = AppConfig.load()

from app.video.narration import NarrationSynthWorker  # noqa: E402

work = Path(tempfile.mkdtemp(prefix="biohuman_verify_"))
results = {}


#: Keep QThread objects alive. Without a strong reference Python garbage-collects
#: a running QThread as soon as the creating frame returns, which kills the
#: process with a native crash rather than a traceback.
_WORKERS: list = []


def synth_and_hash(label: str, tour, done_callback) -> None:
    cues = [type("Cue", (), {"index": 0, "text": tour.keyframes[0].narration,
                             "title": tour.keyframes[0].title})()]
    worker = NarrationSynthWorker(cues, work / label, rate=175, volume=0.9,
                                  enabled=True)
    _WORKERS.append(worker)

    def finished(clips):
        clip = clips[0]
        if clip.ok and clip.path:
            data = clip.path.read_bytes()
            results[label] = (hashlib.sha256(data).hexdigest()[:16],
                              round(clip.duration, 2))
        else:
            results[label] = (None, 0.0)
        done_callback()

    worker.completed.connect(finished)
    worker.failed.connect(lambda m: (results.update({label: (None, 0.0)}),
                                     print("   synth error:", m, flush=True),
                                     done_callback()))
    worker.start()


pipeline_state = {"stage": 0}
check_output: dict = {}


def advance():
    """Run synthesis for two systems, then the export, then report."""
    if pipeline_state["stage"] == 0:
        synth_and_hash("skeletal", get_tour("skeletal_overview"), advance)
        pipeline_state["stage"] = 1
        return
    if pipeline_state["stage"] == 1:
        synth_and_hash("venous", get_tour("venous_overview"), advance)
        pipeline_state["stage"] = 2
        return
    if pipeline_state["stage"] == 2:
        skel, ven = results.get("skeletal"), results.get("venous")
        print(f"  skeletal first cue: sha={skel[0]}  {skel[1]}s", flush=True)
        print(f"  venous   first cue: sha={ven[0]}  {ven[1]}s", flush=True)
        check("Audio differs between systems",
              bool(skel and ven and skel[0] and ven[0] and skel[0] != ven[0]),
              "distinct narration tracks" if skel[0] != ven[0] else "IDENTICAL audio")
        pipeline_state["stage"] = 3
        start_export()
        return


# ---------------------------------------------------------------------------
print()
print("=" * 74)
print(" E. Export pipeline (supersampling + ambient occlusion)")
print("=" * 74)

registry = LayerRegistry(config.paths.models)
viewport = create_viewport(registry, config)
viewport.resize(900, 600)
viewport.show()

from app.core.scene_controller import SceneController  # noqa: E402
from app.audio.manager import AudioManager  # noqa: E402
from app.ai.manager import AIManager  # noqa: E402

audio = AudioManager(config)
scene = SceneController(config, registry, viewport, audio, AIManager(config, registry), None)

out = ROOT / "_verify_export.mp4"
ok_support, support_detail = video_support_status()
check("Video encoder available", ok_support, support_detail)

exporter = TourVideoExporter(viewport, config)
export_state = {"done": False, "path": None, "error": None}


def on_done(path):
    export_state.update(done=True, path=path)


def on_fail(message):
    export_state.update(done=True, error=message)


exporter.finished.connect(on_done)
exporter.failed.connect(on_fail)


def start_export():
    real, placeholders = viewport.load_scene()
    check("Scene loaded", real > 0, f"{real} real / {placeholders} placeholder layers")

    # A short synthetic tour keeps the render quick while still exercising
    # supersampling and the SSAO pass.
    from app.audio.scripts import Tour, TourKeyframe
    tour = Tour(id="verify_quality", title="Quality verification", system="Test",
                keyframes=[
                    TourKeyframe(title="Open", narration="Quality check, first beat.",
                                 view="anterior",
                                 layers={"skin": 0.12, "muscles": 0.85, "skeleton": 0.6},
                                 travel_seconds=0.6),
                    TourKeyframe(title="Close", narration="Quality check, second beat.",
                                 view="isometric", focus_layer="skeleton",
                                 layers={"skeleton": 1.0, "muscles": 0.15},
                                 travel_seconds=0.6),
                ])
    settings = VideoSettings(width=480, height=270, fps=10, quality=9,
                             supersample=2, ssao=True, burn_captions=True,
                             include_audio=True, lead_in=0.6, pause_after=0.4)
    print(f"  settings: {settings.describe()}", flush=True)
    print(f"  render size: {settings.render_dimensions()}", flush=True)
    exporter.start(tour, settings, output=out)


def poll():
    if pipeline_state["stage"] < 3 or not export_state["done"]:
        QTimer.singleShot(700, poll)
        return

    check("SSAO pass applied", getattr(exporter, "_ssao_applied", False))
    if export_state["error"]:
        check("Export succeeded", False, export_state["error"][:150])
    else:
        path = Path(export_state["path"])
        check("Export produced a file", path.exists() and path.stat().st_size > 5000,
              f"{round(path.stat().st_size / 1024, 1)} KB")
        import imageio_ffmpeg
        info = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-i", str(path)],
                              capture_output=True, text=True)
        text = info.stderr or ""
        check("Output has a video stream", "Video:" in text)
        check("Output has an audio stream", "Audio:" in text)
        expected = "480x270"
        check("Output resolution matches the request", expected in text,
              next((l.strip() for l in text.splitlines() if "Video:" in l), ""))

    print()
    print("=" * 74)
    print(f" RESULT: {len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        for name in FAIL:
            print(f"   FAILED: {name}")
    print("=" * 74)
    app.quit()


QTimer.singleShot(500, advance)
QTimer.singleShot(2000, poll)
QTimer.singleShot(300000, app.quit)
app.exec()
sys.exit(1 if FAIL else 0)
