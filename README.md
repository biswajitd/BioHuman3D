# BioHuman3D

A desktop 3D anatomy and health simulation studio — PyQt6 + VTK + local/cloud LLMs
with synchronized voiceover tours.

![stack](https://img.shields.io/badge/PyQt6-6.6%2B-41CD52) ![vtk](https://img.shields.io/badge/VTK-9.3%2B-1E88E5) ![python](https://img.shields.io/badge/python-3.10%2B-3776AB)

See **[BLUEPRINT.md](BLUEPRINT.md)** for the full architecture, threading contract
and roadmap.

---

## Quick start

**Windows — just double-click `run.bat`.** It resolves Python, installs what's
missing, generates starter geometry and launches the app.

```bat
run.bat            :: launch
run.bat doctor     :: diagnose environment (GPU, local LLMs, audio)
run.bat check      :: verify everything without launching
run.bat setup      :: create/refresh .venv and install dependencies
run.bat anatomy    :: download and install BodyParts3D (scan-derived anatomy)
```

Or do it manually:

```powershell
cd BioHuman3D
python -m venv .venv
.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt              # CORE - required
pip install -r requirements-extras.txt       # OPTIONAL - voiceover

python tools\generate_demo_models.py         # placeholder anatomy (recommended)
python main.py
```

## Anatomy: reference body and real atlas data

**Out of the box** the app builds a *reference body* on first launch (about 3 s):
400 individually named structures — every vertebra and rib, long bones with
epiphyses, ~90 muscles placed between their real origins and insertions,
named arteries, veins, nerves and organs — proportioned with the Drillis &
Contini anthropometric ratios for a 1.75 m adult. Click any structure to get
its name; isolate, highlight, slice and animate it individually.

It is still a *schematic* model. **For scan-derived anatomy, install
BodyParts3D** (≈1,500 named parts from a full-body MRI, CC BY 4.0) with one
command:

```powershell
run.bat anatomy                     :: or: python tools\fetch_anatomy.py
python tools\fetch_anatomy.py --restore-reference    :: back to the reference body
```

The fetcher downloads the archive and name table (cached), names and sorts
every mesh into its body-system layer, converts millimetres to metres, orients
it (Z up, anterior −Y, patient's right −X) and writes one structure-labelled
file per layer. Muscle actions, disease simulation and MRI pairing then run on
the real meshes unchanged. If your network blocks the download, fetch the files
in a browser from the BodyParts3D download page and pass them with `--zip`
and `--table`.

**Renderer quality** (applies to both): 2× supersampling on export, screen-space
ambient occlusion, three-point lighting with rim light, `-preset slow` H.264.
Cross-sections are rendered with **solid cut faces** for every sliced structure.

To import any other dataset (OBJ/STL/PLY/GLB/VTP folder):

```powershell
python tools\import_anatomy.py D:\anatomy\MyAtlas --dry-run
python tools\import_anatomy.py D:\anatomy\MyAtlas
python tools\import_anatomy.py --self-test
```

### Free, commercially usable anatomy datasets

| Dataset | License | Entry point |
|---|---|---|
| **BodyParts3D** (DBCLS, Japan) — thousands of named anatomical parts, OBJ | **CC BY 4.0** — commercial use allowed with attribution | [license](https://dbarchive.biosciencedbc.jp/en/bodyparts3d/lic.html) · [download](https://dbarchive.biosciencedbc.jp/en/bodyparts3d/download.html) |
| **Z-Anatomy** — community atlas authored in Blender | CC BY-SA 4.0 | [gauthier-kervyn.itch.io/z-anatomy](https://gauthier-kervyn.itch.io/z-anatomy) |

BodyParts3D requires this attribution in any product that uses it:

> BodyParts3D, © The Database Center for Life Science licensed under CC
> Attribution 4.0 International

Z-Anatomy is share-alike: derivatives of *the model* must carry the same licence.
Neither dataset is bundled here — download and import them yourself so the
licence terms and attribution stay under your control.

### Required vs optional dependencies

| File | Contents | If it fails |
|---|---|---|
| `requirements.txt` | PyQt6, VTK, numpy, requests | The app cannot start |
| `requirements-extras.txt` | pyttsx3 (offline voices), pygame-ce (audio playback) | The app still runs; voiceover is silent or system-voice only |

The app boots even with nothing installed beyond PyQt6: missing VTK shows a
labelled placeholder, missing models fall back to procedural geometry, and
missing LLM servers surface as an explanatory message rather than a crash.

### Python version notes

* **3.10 – 3.14** are supported.
* On **Python 3.14**, do **not** install upstream `pygame` — it publishes no
  cp314 wheels, so pip attempts a source build and fails with
  `ModuleNotFoundError: No module named 'distutils.msvccompiler'` (Python 3.14
  removed `distutils`). Use **`pygame-ce`**, the community fork: identical
  `import pygame` API, prebuilt wheels for 3.12–3.14. This is already what
  `requirements-extras.txt` specifies.

Optional CUDA stack:

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install transformers accelerate bitsandbytes
```

---

## Using the app

| Action | Control |
|---|---|
| Orbit / zoom / pan | Left-drag · wheel · Shift+drag |
| Select a structure | Left-click (hover highlights) |
| Isolate a structure | Double-click, or the ◆ button on a layer row |
| Fade a layer | Opacity slider on the layer row |
| Slice the body | *Display & Slice → Cross-section* |
| Save a 2× PNG | Ctrl+S (written to `%LOCALAPPDATA%\BioHuman3D\screenshots`) |
| Reset the scene | Ctrl+R |
| View presets | Ctrl+1 … Ctrl+6 |
| Play / pause a tour | Space or ▶ in the overlay bar |
| Generate a narrated video | ☰ menu → *Tour video* → *Generate narrated video* |
| Settings (API keys) | Ctrl+, |

### Narration: the export subject follows what you are looking at

**17 authored tours — 12 body systems plus 5 organs** (Liver, Heart, Lungs,
Kidneys, Brain), 2,411 words, and no paragraph is shared between tours.

The export subject is **not** a separate dropdown you have to remember to change.
It follows your selection:

| You select | Video subject |
|---|---|
| Click the **liver** in the 3D view or the layer list | **Liver** — liver-specific narration |
| Choose **Venous** in the sidebar | **Venous System** |
| Choose **Urinary** in the sidebar | **Urinary System** |
| Nothing (fresh launch) | First system in the registry order |

Every subject is also directly selectable in the *Audio & Video* tab, grouped as
`Body system:` and `Organ:`.

> **Why this was broken before:** the controller returned a hard-coded list of
> three tours with the skeleton first, and the sidebar's system list was a
> separate hard-coded tuple of seven labels. So the export panel only ever had
> three choices, defaulted to the skeleton, and syncing to any other system
> failed silently because the tour it asked for was never in the list. Both lists
> now derive from the registry, and the smoke test asserts they cannot diverge.
>
> Each export also carries its provenance in the container metadata, so a file
> identifies its own subject and voice:
> `ffprobe -v error -show_entries format_tags=comment video.mp4`

Verify it yourself:

```powershell
python tools\verify_systems.py      # 21 checks: scripts, audio distinctness, voices, export
```

### Voice and language

Right panel → **Audio & Video** → *Voice & Narration*:

| Control | Choices |
|---|---|
| **Engine** | Microsoft neural voices (free, online — default) · Azure AI Speech (your key) · Google Cloud TTS (your key) · ElevenLabs multilingual · offline system voices · silent |
| **Language** | 11 English dialects (India first, then UK, US, Australia, Canada, Ireland, South Africa, New Zealand, Singapore, Nigeria, Kenya) · 13 Indian languages (Hindi, Bengali, Tamil, Telugu, Marathi, Gujarati, Kannada, Malayalam, Odia, Punjabi, Assamese, Urdu, Nepali) · 24 international languages |
| **Voice** | Female / Male — every language has both |

Neural voices are natural, human-like voices (e.g. Indian English *Neerja* /
*Prabhat*, Hindi *Swara* / *Madhur*, Tamil *Pallavi* / *Valluvar*). Audio is
cached, shared by live narration and video export, and falls back to the
offline system voice automatically when there is no internet.

* **Free neural voices** need `pip install edge-tts` (in `requirements-extras.txt`)
  and an internet connection. They use Microsoft's Edge read-aloud service and
  are intended for personal/evaluation use — for a product, add an **Azure
  Speech** key and region in Settings (same voices, licensed for production).
* **Google Cloud TTS** picks the best voice per language and gender from
  Google's live catalogue (Chirp3-HD → Neural2 → WaveNet).

**Narration in other languages is translated first**, then spoken. The
translation engine is chosen in Settings: your configured AI model (local
Ollama/LM Studio or OpenAI/Anthropic — prompted for standard medical
terminology), Google Cloud Translation, or Azure Translator. Every translated
sentence is stored in `%LOCALAPPDATA%\BioHuman3D\translations\<language>.json`
with its English source. **Have a clinician or medical translator review that
file**: set `"reviewed": true` on corrected entries and they are never
overwritten.

Video captions are drawn with Qt's text engine, so Devanagari, Bengali, Tamil
and the other Indic scripts are shaped correctly and Urdu/Arabic run
right-to-left. Captions advance sentence by sentence with the speech.

### Narrated tour video (MP4)

Right panel → **Audio & Video** tab → *Tour Video*. Pick a tour and a resolution
(720p / 1080p / 1440p / 4K), then **Generate video**. The tour is rendered
frame-by-frame at the chosen resolution, the narration is synthesised and mixed
into the file, and the player opens automatically when it finishes.

* Narration is synthesised **first**, so the video timeline is built from the
  measured speech durations — the voice and the visuals cannot drift apart.
* Rendering happens **off-screen** at full output resolution, independent of the
  window size, and is batched so the UI stays responsive (with a progress bar
  and Cancel).
* **Captions** can be burned into the frames, so the video still explains itself
  when muted.
* Output goes to `%LOCALAPPDATA%\BioHuman3D\videos\`; *Show in folder* reveals it.

The right panel is **tabbed** (*Assistant* / *Audio & Video*) rather than one long
column, so every control stays reachable on a laptop-height display.

---

## Simulate tab: muscle actions, disease progression, MRI pairing

### Muscle actions
Pick any of 45 muscles (or click one in 3D) and press **Play action**. The
joint moves through the muscle's real actions — e.g. biceps brachii:
supination, elbow flexion, shoulder flexion — with the agonist glowing and
thickening as it shortens, synergists amber and antagonists blue. Narration
gives origin, insertion, innervation with root levels, normal range of motion
(AAOS) and a clinical note (Gray's / Moore's). **Animate motion** shows every
agonist and antagonist of one joint motion. Joint centres are estimated from the
bones, so this works on imported atlases too. **Export video** renders it to MP4.

### Disease simulation
Eight conditions, each staged with the system clinicians use and morphing the
anatomy stage by stage:

| Condition | Staging |
|---|---|
| Coronary artery disease → myocardial infarction | AHA lesion types; 4th Universal Definition of MI |
| Fatty liver disease → cirrhosis (MASLD) | Steatosis → MASH → fibrosis F0–F4 (METAVIR) |
| Chronic kidney disease | KDIGO G1–G5 |
| COPD / emphysema | GOLD 1–4 |
| Ischaemic stroke (left MCA) | Hyperacute → acute → chronic |
| Knee osteoarthritis | Kellgren–Lawrence 0–4 |
| Osteoporosis with vertebral fracture | WHO T-score categories |
| Hypertension with LV hypertrophy | ACC/AHA 2017 |

Use the **Stage** slider to inspect a stage, **Play progression** for a narrated
tour, **Export video** for MP4. Each stage states its defining criteria and the
guideline it follows. These are educational simulations, not diagnoses.

### Cross-sections paired with MRI
**Simulate MRI** (T1 / T2 / PD) builds an MRI-like study from the model, or
**Load scan…** opens a real NIfTI file or DICOM folder (`pip install nibabel
pydicom`). The paired view sits beside the 3D view; moving the cut in either one
moves the other. Slices follow radiological convention (axial viewed from the
feet), hovering names the structure, clicking selects it in 3D, and *Labels*
outlines every structure. Simulated studies are always marked **SIMULATED**.

---

## Local AI

The app probes, in parallel, on startup:

| Backend | Endpoint |
|---|---|
| Ollama | `http://127.0.0.1:11434` |
| LM Studio | `http://127.0.0.1:1234` |
| llama.cpp server | `http://127.0.0.1:8080` |
| vLLM | `http://127.0.0.1:8000` |

It also inventories **on-disk** model caches (Ollama blobs, LM Studio folders,
`~/.cache/huggingface/hub`) and reports GPU capability from `nvidia-smi` or
`torch.cuda`. If nothing is running you get a concrete, actionable message instead
of a silent failure.

**Local mode** routes to the best available local server.
**Cloud mode** uses OpenAI or Anthropic (add a key in Settings or set
`OPENAI_API_KEY` / `ANTHROPIC_API_KEY`).
**Auto** prefers a healthy local model and falls back to cloud.

---

## Adding real anatomy

1. Export each anatomical system as an **OBJ / STL / PLY / VTP / GLB** file.
2. Name it after the layer id (`skin.glb`, `skeleton.glb`, `heart.glb`, …)
   and drop it in `app/assets/models/`.
3. Restart — no code changes needed.

GLB is preferred: VTK reads GLTF node names, so `Femur.L`, `Heart`, `Aorta` become
individually pickable structures inside their layer.

To add a system that isn't in the registry, append a `LayerSpec` row in
[model_registry.py](app/core/model_registry.py).

**Sources of open anatomy data you can use:**

* **Z-Anatomy** — Blender source, CC-BY-SA, the most complete open atlas.
* **BodyParts3D** — CC-BY-SA, ~2000 mesh parts, ideal for per-organ layers.
* **OpenStax / NIH 3D Print Exchange** — mixed licences, check each asset.

Keep each layer under ~500 k triangles (or split into multiple nodes) to hold
60 fps on an RTX 4060.

---

## Project layout

```
main.py                    entry point
app/config.py              paths + persisted settings
app/core/                  viewport, registry, tours, VTK helpers
app/ai/                    detection, providers, prompts, manager
app/anatomy/               reference body, structure files, atlas frame,
                           muscle knowledge base, kinematic rig, muscle tours
app/disease/               condition catalogue, stage effects, progression tours
app/imaging/               MRI volumes (NIfTI/DICOM), MRI simulator, paired slice pane
app/audio/                 languages, neural TTS, system TTS, authored tour scripts
app/i18n/                  narration translation + translation memory
app/video/                 exporter, narration synthesis, Qt captions
app/ui/                    shell, theme, QSS, panels, simulation page
tools/generate_demo_models.py   reference body
tools/fetch_anatomy.py          download + install BodyParts3D
tests/                          pytest unit tests
```

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| Status bar or bottom controls cut off | The window no longer fits the work area. Geometry is clamped to the available screen on every launch; if a stale size persists, delete the `ui.geometry` key from `%LOCALAPPDATA%\BioHuman3D\settings.json`. |
| `Video encoding needs imageio-ffmpeg` | `pip install imageio imageio-ffmpeg` |
| Exported video has no sound | `pyttsx3` is missing or no system voice is selected. The video still renders with captions. |
| Narration stops after the first paragraph | Fixed: a fresh speech engine is created per paragraph (reusing one hangs SAPI5). Update to the current `app/audio/manager.py`. |
| `distutils.msvccompiler` while installing `pygame` | Python 3.14 removed `distutils`. Install the fork instead: `pip install pygame-ce`. Never blocks the app. |
| `3D viewport unavailable` | `pip install vtk` (needs 9.3+ for PyQt6), restart |
| Transparent layers look wrong | Settings → *Correct transparency (depth peeling)*; some drivers prefer it off |
| No narration | `pip install pyttsx3 pygame`; check a voice is selected in *Audio & Narration* |
| `No AI model available` | Start `ollama serve` / enable LM Studio's local server, or add a cloud key |
| AI answer fails mid-stream | The provider returned an HTTP error — the message includes the status code |
| Low frame rate | Set the layer decimation ratio in `viewport.load_scene(decimate_ratio=0.4)` |

---

## Licence note

The application code here is yours to use freely. Anatomy **datasets** carry their
own licences (Z-Anatomy and BodyParts3D are CC-BY-SA) — attribution is required
if you redistribute derived models.
