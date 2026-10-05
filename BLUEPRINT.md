# BioHuman3D — System Architecture & Project Blueprint

A desktop 3D anatomy + health simulation platform (BioDigital-Human-class UX)
built on **PyQt6 + VTK + local/cloud LLMs + a synchronized narration engine**.

Target hardware: Intel i9 / RTX 4060 / DDR5 — the stack is designed around
GPU-accelerated rendering (VTK/OpenGL), optional CUDA inference, and zero-blocking UI.

---

## 1. Design Principles

| Principle | Implementation |
|---|---|
| **Never block the UI thread** | Every I/O-bound task (LLM streaming, TTS, network probes, mesh loading) runs on a `QThread` worker and reports back via Qt signals. |
| **The viewport is the product** | VTK renders at 60 fps with depth peeling, FXAA and progressive image quality; all heavy geometry is cleaned/decimated once at load time. |
| **One actor group per anatomical layer** | Hide / fade / isolate become O(1) property changes instead of mesh surgery. |
| **Graceful degradation** | Missing VTK, missing models, missing Ollama, missing API keys → the app still boots and tells the user exactly what is missing. |
| **Provider-agnostic AI** | A single `LLMProvider` interface fronts Ollama, LM Studio, llama.cpp, vLLM, OpenAI and Anthropic. Swapping backends never touches UI code. |

---

## 2. Directory Structure

```
BioHuman3D/
├── main.py                       # Entry point: bootstrap, theme, high-DPI, demo assets
├── requirements.txt
├── BLUEPRINT.md                  # This document
├── README.md
│
├── app/
│   ├── config.py                 # AppConfig + Paths + persisted QSettings
│   │
│   ├── core/                     # Rendering & scene domain (no widgets-with-chrome)
│   │   ├── vtk_utils.py          # VTK import guard, reader factory, mesh cleanup, actor factory
│   │   ├── model_registry.py     # Dataset definitions: layers, files, colours, groups
│   │   ├── interaction.py        # Interactor style: hover-pick, isolate-on-double-click
│   │   ├── viewport.py           # ★ Interactive3DViewport widget (VTK + Qt)
│   │   ├── tour.py               # Camera keyframe interpolator (guided tours)
│   │   └── scene_controller.py   # Orchestrator: viewport ⇄ audio ⇄ AI ⇄ UI
│   │
│   ├── ai/                       # Local + cloud LLM subsystem
│   │   ├── detector.py           # Scans localhost for Ollama / LM Studio / llama.cpp / vLLM
│   │   │                         #   + HuggingFace cache + CUDA/NVIDIA capability
│   │   ├── providers.py          # LLMProvider ABC + 6 concrete streaming providers
│   │   ├── prompts.py            # System prompt + anatomy-context builders
│   │   └── manager.py            # AIManager: detection thread, routing, streaming
│   │
│   ├── audio/                    # Narration / voiceover subsystem
│   │   ├── manager.py            # AudioManager: queued TTS, caching, sync signals
│   │   └── scripts.py            # Narration scripts + authored guided tours
│   │
│   ├── ui/
│   │   ├── theme.py              # Palette tokens → QSS template substitution
│   │   ├── style.qss             # $TOKEN-based dark stylesheet (compiled at runtime)
│   │   ├── main_window.py        # ★ Application shell: title bar + 3-panel layout
│   │   └── widgets/
│   │       ├── controls.py       # IconButton, PillToggle, LabeledSlider, SectionCard
│   │       ├── collapsible.py    # Animated collapsible section
│   │       ├── layer_panel.py    # Left: Anatomy Layers & Tools
│   │       ├── viewport_panel.py # Centre: 3D viewport + floating glass toolbars
│   │       └── ai_panel.py       # Right: AI Assistant & Audio Controls
│   │
│   └── assets/
│       ├── models/               # .glb / .gltf / .obj / .stl / .vtp  (per-layer files)
│       ├── audio/cache/          # Rendered TTS clips (content-hashed)
│       └── icons/                # Inline SVG icon paths
│
└── tools/
    └── generate_demo_models.py   # Procedural placeholder anatomy so the app runs OOTB
```

**Asset rule:** one file (or one `vitep`-style GLTF node group) per layer id declared in
`model_registry.py`. Adding a new organ = dropping a file + one registry row.

---

## 3. Runtime Architecture

```
                       ┌──────────────────────────────────────────────┐
                       │              MainWindow (QMainWindow)        │
                       │  TitleBar │ Splitter │ StatusBar             │
                       └───────┬──────────────┬───────────────┬───────┘
                               │              │               │
                  ┌────────────▼───┐  ┌───────▼────────┐  ┌───▼─────────────┐
                  │  LayerPanel    │  │ ViewportPanel  │  │    AIPanel      │
                  │  (left)        │  │  (centre)      │  │    (right)      │
                  └────────┬───────┘  └───────┬────────┘  └───┬─────────────┘
                           │ signals          │               │
                           └──────────┬───────┴───────────────┘
                                      ▼
                        ┌───────────────────────────────┐
                        │      SceneController          │  ← single source of truth
                        │  (layer state, tours, context)│
                        └───┬───────────┬───────────┬───┘
                            │           │           │
                ┌───────────▼──┐ ┌──────▼─────┐ ┌───▼───────────┐
                │  Viewport    │ │ AudioMgr   │ │  AIManager    │
                │  (VTK/OGL)   │ │ (TTS)      │ │  (LLMs)       │
                └──────────────┘ └────────────┘ └───┬───────────┘
                                                    │
                                      ┌─────────────▼──────────────┐
                                      │       AI Detector          │
                                      │  Ollama :11434             │
                                      │  LM Studio :1234           │
                                      │  llama.cpp :8080           │
                                      │  vLLM :8000                │
                                      │  ~/.cache/huggingface      │
                                      │  nvidia-smi / torch.cuda   │
                                      └────────────────────────────┘
```

### Threading contract

| Worker | Thread | Communicates via |
|---|---|---|
| `DetectionWorker` | `QThread` | `report_ready(DetectionReport)` |
| `ChatWorker` (per request) | `QThread` | `chunk(str)`, `finished(str)`, `failed(str)` |
| `NarrationWorker` | `QThread` | `cue_started(int)`, `word(str)`, `cue_finished(int)` |
| Mesh loading | main thread (VTK is not thread-safe for render) | direct return |

VTK objects are **owned by the main thread only**. Workers hand back plain Python data
(text, paths, numbers) — never `vtkActor`.

---

## 4. 3D Viewport Strategy

**Why VTK over Panda3D / PyOpenGL here**

* Native `vtkActor` opacity + **depth peeling** = correct transparency for nested
  anatomical layers (skin over muscle over bone). This is the single hardest
  requirement and VTK gives it for free.
* `vtkCellPicker` returns the exact picked actor → instant "what is this structure?".
* `vtkPlane` clipping gives cross-section/slicing tools in ~20 lines.
* VTK ships a Qt-native interactor (`QVTKRenderWindowInteractor`) designed for
  embedding; Panda3D requires a manual OpenGL window hand-off.

**Layer model**

```
layer_id  →  LayerState{ actors[], opacity, visible, color, z_offset }
```

* **Hide** → `actor.SetVisibility(False)`
* **Fade** → `actor.GetProperty().SetOpacity(a)` + `ForceTranslucentOn()` (depth peeling on)
* **Isolate** → target keeps opacity 1.0, every other layer drops to 0.06 (X-ray ghost)
* **Explode** → each layer translated along a registry-defined offset vector
* **Slice**  → one shared `vtkPlane` added to every mapper in the scene

**Performance notes**

* Decimation (`vtkQuadricDecimation`) + `vtkWindowedSincPolyDataFilter` at load only.
* `SetDesiredUpdateRate` / interactive LOD: coarse render while dragging, HQ on release.
* Cap total triangles ≈ 2–3 M for a 4060 at 60 fps; split organ models into
  multiple actors when a layer exceeds ~500 k tris.

---

## 5. AI Subsystem

**Detection (runs at startup, ~0.5 s, fully parallel)**

| Probe | Endpoint / Path | Payload used |
|---|---|---|
| Ollama | `GET http://127.0.0.1:11434/api/tags` | `models[].name`, `size` |
| LM Studio | `GET http://127.0.0.1:1234/v1/models` | `data[].id` |
| llama.cpp server | `GET http://127.0.0.1:8080/v1/models` | `data[].id` |
| vLLM | `GET http://127.0.0.1:8000/v1/models` | `data[].id` |
| HuggingFace cache | `~/.cache/huggingface/hub/models--*` | GGUF / safetensors size |
| CUDA | `nvidia-smi --query-gpu=...` + `torch.cuda` | GPU name, VRAM, driver |

**Routing**

```
user toggle  ──►  Local        ──►  auto-pick best available local backend
                  Cloud        ──►  OpenAI / Anthropic (needs API key)
                  Auto         ──►  Local if healthy, else Cloud, else explain
```

**Provider contract**

```python
class LLMProvider(ABC):
    id: str; label: str; kind: Literal["local", "cloud"]
    def list_models(self) -> list[ModelInfo]: ...
    def health(self) -> HealthStatus: ...
    def stream_chat(self, messages, model, **opts) -> Iterator[str]: ...
```

Streaming everywhere (NDJSON for Ollama, SSE for the OpenAI-compatible and
Anthropic APIs) so the transcript renders token-by-token.

---

## 6. Audio Subsystem

* **Backends:** `pyttsx3` (offline SAPI5/NSSpeech, zero cost), `ElevenLabs`
  (cached MP3 rendered to `assets/audio/cache/` and played through `pygame.mixer`),
  `NullBackend` (silent, keeps the feature testable).
* **Queue:** cues are enqueued, never spoken twice, and cancellable.
* **Synchronisation:** `AudioManager.cue_started(index)` drives camera keyframes;
  pyttsx3's `started-word` callback emits `word(str)` for live transcript highlighting.
* **Guidance:** always create the `pyttsx3` engine **inside** the worker thread —
  SAPI5 binds to the COM apartment of the creating thread and will silently fail otherwise.

---

## 7. Build / Run

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python tools\generate_demo_models.py    # optional placeholder anatomy
python main.py
```

Optional CUDA extra:

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

---

## 8. Roadmap Beyond the Bootstrap

1. **Real anatomy data** — Z-Anatomy (Blender/CC-BY-SA) or BodyParts3D, exported as
   per-system GLB with node names matching layer ids.
2. **Pathology database** — SQLite table of conditions mapped to structure ids,
   surfaced as clickable "lesions" that re-colour the affected layer.
3. **DICOM ingestion** — `vtkDICOMImageReader` → `vtkMarchingCubes` for
   patient-specific anatomy alongside the generic atlas.
4. **Physiology simulation** — cardiac cycle / ventilation driven by a simulation
   clock that morphs keyframed meshes or animates a displacement filter.
5. **WebXR export** — `vtkWebAssembly` or a Three.js companion for tablet sharing.
6. **Collaborative session** — WebSocket relay of layer state + camera for teaching.
