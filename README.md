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

## Visual quality and real anatomy data

**Read this if the renders look "not realistic enough".**

The app ships with **procedural placeholder geometry** — ellipsoids and tubes
arranged in a rough anatomical stack. It exists so the viewer, tours, picking,
isolation and export all work out of the box. It is *not* a substitute for real
anatomical data, and no amount of rendering work will make a sphere look like a
liver. Photoreal anatomy comes from real mesh data.

**What was improved in the renderer itself** (all of which also lift real data):

| Change | Effect |
|---|---|
| 2× supersampling on export (renders 4K to output 1080p, then box-filtered down) | Clean, properly anti-aliased silhouettes |
| Screen-space ambient occlusion (`vtkSSAOPass`) | Contact shadows in creases and between organs — surfaces read as solid, not flat |
| Three-point lighting rig + rim light | Form and depth instead of flat headlight shading |
| Doubled mesh resolution in the demo generator (40–48 → 64–96) | Smooth silhouettes on close-ups |
| `-preset slow` + `+faststart` encoding | Better compression at equal quality, streamable output |

Toggle **Ultra** in the export card to control supersampling + occlusion.

**To get production realism, import a real dataset:**

```powershell
# 1. see how a dataset would be sorted, without changing anything
python tools\import_anatomy.py D:\anatomy\BodyParts3D --dry-run

# 2. merge each body system into one .vtp the app can load
python tools\import_anatomy.py D:\anatomy\BodyParts3D
```

The importer assigns every mesh to a layer by matching its name against ordered
keyword rules ("aortic valve" → `heart`, not `arteries`), merges each system into
a single file, and reports anything it could not classify rather than guessing.
Datasets that use opaque ids (FMA/TA codes) are handled with a name table —
supply `--map id,name.csv`, or drop a two-column `.tsv` in the folder.

Verify the pipeline works before converting anything:

```powershell
python tools\import_anatomy.py --self-test     # builds a labelled dataset and imports it
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

### Voice: dialect × gender

Right panel → **Audio & Video** → *Voice & Narration*. Choose a **Dialect** and a
**Voice** (Female / Male / Any); the exact installed voice is then resolved and
named in the panel.

> **Important — Indian English is not installed on this PC by default.**
> Windows ships `en-US` only (David, Zira) unless the Speech feature for India is
> added. BioHuman3D does **not** silently pass off a US voice as Indian English:
> it reports the gap, falls back to a voice matching your requested *gender*, and
> offers setup instructions. `run.bat doctor` reports the same.
>
> To add Indian English: **Settings → Time & Language → Language & region →**
> **Add a language → English (India) → tick "Speech"**, then restart the app.
> Windows then provides *Heera* (female) and *Ravi* (male).
> For regional Indian accents without a language pack, configure an ElevenLabs
> key — its multilingual voices expose accent/gender labels and are fetched live.

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
app/audio/                 TTS engine + authored tour scripts
app/ui/                    shell, theme, QSS, panels
tools/generate_demo_models.py
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
