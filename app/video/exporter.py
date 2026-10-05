"""
Narrated tour video export.

Pipeline
--------
    1. SYNTHESISE  narration cues -> WAV clips (measures real speech duration)
    2. PLAN        build a timeline from those durations (audio-first timing,
                   so the visuals can never drift from the voice)
    3. RENDER      draw the tour frame-by-frame off-screen at the chosen
                   resolution, spline-easing the camera and fading the layers
    4. MUX         encode H.264 and mux the narration track with ffmpeg

Design notes
------------
* **Off-screen renderer.** Export uses its own ``vtkRenderWindow`` rather than
  resizing the visible widget, so the UI keeps working and the output
  resolution is independent of the window size. Actors are shared, so layer
  opacity changes apply to both.
* **Main-thread rendering.** VTK is not thread-safe. Frames are drawn in small
  batches driven by a ``QTimer``, which keeps the UI responsive and the
  progress bar honest. Only WAV synthesis and the ffmpeg mux run on worker
  threads.
* **Graceful degradation.** Without ``imageio-ffmpeg`` the export reports a
  clear message; without pyttsx3 it produces a captioned silent video using
  estimated timings.
"""
from __future__ import annotations

import math
import shutil
import subprocess
import textwrap
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from PyQt6.QtCore import QObject, QThread, QTimer, pyqtSignal

from app.core.video_math import ease_in_out_cubic, lerp_state
from app.video.captions import chunk_at, chunk_text
from app.core.viewport import BACKGROUNDS, VIEW_PRESETS
from app.core.vtk_utils import VTK_AVAILABLE, vtk
from app.video.narration import (AudioTrackBuilder, NarrationClip,
                                 NarrationSynthWorker, estimate_duration)

try:
    from vtkmodules.util.numpy_support import vtk_to_numpy
except Exception:  # pragma: no cover
    from vtk.util.numpy_support import vtk_to_numpy  # type: ignore

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except Exception:  # pragma: no cover
    np = None  # type: ignore[assignment]
    NUMPY_AVAILABLE = False

try:
    import imageio.v2 as imageio
    IMAGEIO_AVAILABLE = True
except Exception:  # pragma: no cover
    imageio = None  # type: ignore[assignment]
    IMAGEIO_AVAILABLE = False

try:
    import imageio_ffmpeg
    IMAGEIO_FFMPEG_AVAILABLE = True
except Exception:  # pragma: no cover
    imageio_ffmpeg = None  # type: ignore[assignment]
    IMAGEIO_FFMPEG_AVAILABLE = False


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
@dataclass
class VideoSettings:
    """Export parameters exposed in the UI."""

    width: int = 1920
    height: int = 1080
    fps: int = 30
    quality: int = 9              # imageio/ffmpeg quality (0-10, higher = better)
    travel_seconds: float = 1.7   # camera glide between keyframes
    pause_after: float = 0.9      # breathing room after each narration finishes
    lead_in: float = 1.0          # opening hold on keyframe 1
    burn_captions: bool = True
    include_audio: bool = True

    # -- quality ----------------------------------------------------------
    #: Render at N× the output resolution and box-filter down. This is the single
    #: biggest visual win available without new mesh data: it removes the stair
    #: stepping on silhouettes that makes procedural geometry look cheap.
    supersample: int = 2
    #: Screen-space ambient occlusion — contact shadows in creases and between
    #: organs, which is what makes a render read as solid rather than flat.
    ssao: bool = True

    def even_dimensions(self) -> tuple:
        """H.264 requires even width/height."""
        return (self.width - self.width % 2, self.height - self.height % 2)

    def render_dimensions(self) -> tuple:
        """Internal render size, including supersampling."""
        width, height = self.even_dimensions()
        factor = max(1, int(self.supersample))
        return (width * factor, height * factor)

    def describe(self) -> str:
        out_w, out_h = self.even_dimensions()
        render_w, render_h = self.render_dimensions()
        bits = [f"{out_w}×{out_h} @ {self.fps}fps"]
        if self.supersample > 1:
            bits.append(f"{self.supersample}× SSAA (renders {render_w}×{render_h})")
        if self.ssao:
            bits.append("ambient occlusion")
        if self.burn_captions:
            bits.append("captions")
        return " · ".join(bits)


@dataclass
class Segment:
    """One keyframe's slot on the video timeline."""

    index: int
    title: str
    narration: str
    start: float = 0.0
    duration: float = 0.0
    travel: float = 1.7
    camera_from: Dict = field(default_factory=dict)
    camera_to: Dict = field(default_factory=dict)
    layers_from: Dict[str, float] = field(default_factory=dict)
    layers_to: Dict[str, float] = field(default_factory=dict)
    effect: Optional[object] = None          # SceneEffect animated during this segment
    chunks: List[str] = field(default_factory=list)   # caption chunks, in speech order
    speech: float = 0.0                      # spoken duration (s)
    speech_start: float = 0.0                # offset of speech within the segment

    @property
    def end(self) -> float:
        return self.start + self.duration


# ---------------------------------------------------------------------------
# Mux worker
# ---------------------------------------------------------------------------
class _MuxWorker(QThread):
    """Encodes/muxes the final container with ffmpeg."""

    done = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(self, video: Path, audio: Optional[Path], out: Path,
                 comment: str = "", parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._video = Path(video)
        self._audio = Path(audio) if audio else None
        self._out = Path(out)
        self._comment = comment

    def run(self) -> None:  # noqa: D102
        try:
            has_audio = self._audio is not None and self._audio.exists()
            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
            command = [ffmpeg, "-y", "-i", str(self._video)]
            if has_audio:
                command += ["-i", str(self._audio),
                            "-c:a", "aac", "-b:a", "192k", "-shortest"]
            command += ["-c:v", "copy"]
            if self._comment:
                # Self-identifying metadata: which build, which subject and which
                # voice produced this file. Makes "it exported the wrong thing"
                # answerable with a single ffprobe instead of guesswork.
                command += ["-metadata", f"comment={self._comment}"]
            command.append(str(self._out))

            result = subprocess.run(command, capture_output=True, text=True,
                                    timeout=900, check=False)
            if result.returncode != 0:
                if not has_audio:
                    # Nothing to mux; a plain move is good enough.
                    shutil.move(str(self._video), str(self._out))
                    self.done.emit(str(self._out))
                    return
                self.failed.emit(
                    f"ffmpeg exited with code {result.returncode}: "
                    f"{(result.stderr or '')[-400:]}")
                return
            self.done.emit(str(self._out))
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# Exporter
# ---------------------------------------------------------------------------
class TourVideoExporter(QObject):
    """Renders a narrated guided tour to an MP4 file."""

    progress = pyqtSignal(int, str)      # percent, stage description
    finished = pyqtSignal(str)           # output path
    failed = pyqtSignal(str)
    cancelled = pyqtSignal()

    #: Frames rendered per timer tick before yielding to the event loop.
    BATCH = 4

    def __init__(self, viewport, config, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._viewport = viewport
        self._config = config
        self._settings = VideoSettings()

        self._queue: List[Segment] = []
        self._cursor = 0
        self._frame_in_segment = 0
        self._writer = None
        self._render_window = None
        self._renderer = None
        self._caption = None
        self._title = None
        self._mark = None
        self._video_path: Optional[Path] = None
        self._audio_path: Optional[Path] = None
        self._out_path: Optional[Path] = None
        self._clips: List[NarrationClip] = []
        self._synth: Optional[NarrationSynthWorker] = None
        self._mux: Optional[_MuxWorker] = None
        self._cancel = False
        self._ssao_applied = False
        self._started_at = 0.0
        self._frame_total = 0
        self._frame_done = 0
        self._total_seconds = 0.0
        self._work_dir: Optional[Path] = None
        self._tour = None
        self._meta: Dict[str, str] = {}
        self._active_effect = None
        self._overlay = None

        self._timer = QTimer(self)
        self._timer.setInterval(0)          # drain as fast as the UI allows
        self._timer.timeout.connect(self._render_batch)

    # ------------------------------------------------------------ public
    @property
    def is_running(self) -> bool:
        return (self._synth is not None and self._synth.isRunning()) \
            or self._timer.isActive() \
            or (self._mux is not None and self._mux.isRunning())

    def prerequisites_ok(self) -> tuple:
        """``(ok, message)`` describing whether export can run at all."""
        if not VTK_AVAILABLE:
            return False, "VTK is not available — nothing to render."
        if not IMAGEIO_AVAILABLE:
            return False, "Video encoding needs imageio:  pip install imageio"
        if not IMAGEIO_FFMPEG_AVAILABLE:
            return False, ("Video encoding needs imageio-ffmpeg:  "
                           "pip install imageio-ffmpeg")
        if not NUMPY_AVAILABLE:
            return False, "Video export needs numpy:  pip install numpy"
        return True, ""

    @staticmethod
    def default_output(config, tour_id: str) -> Path:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        return config.paths.user_data / "videos" / f"{tour_id}_{stamp}.mp4"

    def estimate_seconds(self, tour, settings: Optional[VideoSettings] = None) -> float:
        """Rough duration of the finished video (before real synthesis)."""
        settings = settings or self._settings
        rate = int(self._config.get("audio.rate", 175))
        total = 0.0
        for index, keyframe in enumerate(tour.keyframes):
            travel = settings.lead_in if index == 0 else settings.travel_seconds
            spoken = estimate_duration(getattr(keyframe, "narration", ""), rate)
            total += max(travel, spoken) + settings.pause_after
        return total

    def start(self, tour, settings: VideoSettings,
              output: Optional[Path] = None,
              meta: Optional[Dict[str, str]] = None) -> bool:
        ok, message = self.prerequisites_ok()
        if not ok:
            self.failed.emit(message)
            return False
        if self.is_running:
            return False

        self._settings = settings
        self._meta = dict(meta or {})
        self._cancel = False
        self._started_at = time.perf_counter()
        self._clips = []
        self._queue = []
        self._cursor = 0
        self._frame_done = 0
        self._frame_total = 0
        self._out_path = Path(output) if output else self.default_output(
            self._config, getattr(tour, "id", "tour"))
        self._out_path.parent.mkdir(parents=True, exist_ok=True)

        work = self._out_path.parent / f".work_{self._out_path.stem}"
        work.mkdir(parents=True, exist_ok=True)
        self._work_dir = work
        self._video_path = work / "silent.mp4"

        self.progress.emit(0, "Preparing narration…")

        # -- stage 1: synthesise narration (worker thread) -------------------
        cues = self._cues_for(tour)
        self._tour = tour
        self._synth = NarrationSynthWorker(
            cues, work / "audio",
            rate=int(self._config.get("audio.rate", 175)),
            volume=float(self._config.get("audio.volume", 0.9)),
            voice_id=str(self._config.get("audio.voice", "") or ""),
            enabled=bool(settings.include_audio)
            and bool(self._config.get("audio.enabled", True)),
            neural=self._neural_request(),
            cache_dir=self._config.paths.audio_cache,
            parent=self,
        )
        # THREAD SAFETY: connect to *bound methods*, never lambdas. A lambda is
        # not a QObject, so PyQt falls back to a direct connection and the slot
        # would run on the worker thread. That would build a vtkRenderWindow and
        # render off the GUI thread — VTK is not thread-safe and the process dies
        # with a fail-fast access violation (0xC0000409), not a Python exception.
        self._synth.progress.connect(self._on_synth_progress)
        self._synth.completed.connect(self._on_synth_completed)
        self._synth.failed.connect(self._on_synth_failed)
        self._synth.start()
        return True

    def _attribution(self) -> str:
        """Credit line required by the anatomy licence (BodyParts3D: CC BY 4.0)."""
        sources = {getattr(state, "source", "") for state in self._viewport.layers.values()}
        if "bodyparts3d" in sources:
            return "Anatomy: BodyParts3D © The Database Center for Life Science, CC BY 4.0"
        return ""

    def _neural_request(self):
        if str(self._config.get("audio.backend", "edge")) not in ("edge", "azure", "google"):
            return None
        from app.audio.neural_tts import from_config
        return from_config(self._config)

    # -- synthesis callbacks (always delivered on the GUI thread) ----------
    def _on_synth_progress(self, percent: int, message: str) -> None:
        self.progress.emit(int(percent * 0.15), message)

    def _on_synth_failed(self, message: str) -> None:
        # Non-fatal: fall back to estimated timings and a silent video.
        self.progress.emit(2, message)

    def _on_synth_completed(self, clips: List[NarrationClip]) -> None:
        if self._cancel:
            return
        self._on_synth_done(getattr(self, "_tour", None), clips)

    def cancel(self) -> None:
        self._cancel = True
        if self._synth is not None and self._synth.isRunning():
            self._synth.cancel()
        self._timer.stop()
        self._teardown_renderer()
        self._close_writer()
        self._cleanup_work()
        self.cancelled.emit()

    # ------------------------------------------------------------ planning
    @staticmethod
    def _cues_for(tour) -> List:
        return [type("Cue", (), {
            "index": index,
            "text": getattr(keyframe, "narration", ""),
            "title": getattr(keyframe, "title", ""),
        })() for index, keyframe in enumerate(tour.keyframes)]

    def _on_synth_done(self, tour, clips: List[NarrationClip]) -> None:
        if self._cancel or tour is None:
            return
        self._clips = clips
        self._queue = self._plan(tour, clips)
        if not self._queue:
            self.failed.emit("The tour has no keyframes to render.")
            return

        self._frame_total = sum(
            max(1, int(segment.duration * self._settings.fps)) for segment in self._queue)
        self._build_renderer()

        self.progress.emit(15, "Rendering frames…")
        self._timer.start()

    def _plan(self, tour, clips: Sequence[NarrationClip]) -> List[Segment]:
        """Build the timeline. Narration durations drive segment lengths."""
        settings = self._settings
        bounds = self._scene_bounds()
        start = 0.0

        # Current live state is the starting point for the opening keyframe.
        previous_camera = self._viewport.camera_state() or self._camera_for(
            bounds, None, "isometric")
        previous_layers = {layer_id: state.opacity
                           for layer_id, state in self._viewport.layers.items()}

        segments: List[Segment] = []
        for index, keyframe in enumerate(tour.keyframes):
            clip = clips[index] if index < len(clips) else None
            spoken = clip.duration if clip else estimate_duration(
                getattr(keyframe, "narration", ""))

            travel = settings.lead_in if index == 0 else settings.travel_seconds
            duration = max(travel, spoken) + settings.pause_after
            effect = getattr(keyframe, "effect", None)
            if effect is not None:          # always show at least one full action cycle
                duration = max(duration, float(getattr(effect, "cycle_seconds", 0.0)) + 0.3)

            layers_to = dict(previous_layers)
            for layer_id, opacity in (getattr(keyframe, "layers", {}) or {}).items():
                layers_to[layer_id] = float(opacity)
            # Keyframes that only change the camera still show the previously
            # faded state, so untouched layers keep their last value.

            camera_to = self._camera_for(
                bounds, keyframe, getattr(keyframe, "view", "") or "isometric")

            segments.append(Segment(
                index=index,
                title=getattr(keyframe, "title", "") or f"Step {index + 1}",
                narration=getattr(keyframe, "narration", ""),
                start=start,
                duration=duration,
                travel=travel,
                camera_from=dict(previous_camera),
                camera_to=dict(camera_to),
                layers_from=dict(previous_layers),
                layers_to=layers_to,
                effect=effect,
                chunks=chunk_text(getattr(keyframe, "narration", "")),
                speech=spoken,
            ))

            if clip is not None:
                clip.start = start
            start += duration
            previous_camera = camera_to
            previous_layers = layers_to

        self._total_seconds = start
        return segments

    # ------------------------------------------------------------- camera
    def _scene_bounds(self) -> tuple:
        """Bounds of everything currently in the scene."""
        if self._viewport is None:
            return (-1, 1, -1, 1, -1, 1)
        bounds = [1e18, -1e18, 1e18, -1e18, 1e18, -1e18]
        for state in self._viewport.layers.values():
            for polydata in state.polydata:
                b = polydata.GetBounds()
                bounds[0] = min(bounds[0], b[0]); bounds[1] = max(bounds[1], b[1])
                bounds[2] = min(bounds[2], b[2]); bounds[3] = max(bounds[3], b[3])
                bounds[4] = min(bounds[4], b[4]); bounds[5] = max(bounds[5], b[5])
        if bounds[0] > bounds[1]:
            return (-1, 1, -1, 1, -1, 1)
        return tuple(bounds)

    def _layer_bounds(self, layer_id: str):
        state = self._viewport.layers.get(layer_id)
        if state is None or not state.polydata:
            return None
        bounds = [1e18, -1e18, 1e18, -1e18, 1e18, -1e18]
        for polydata in state.polydata:
            b = polydata.GetBounds()
            bounds[0] = min(bounds[0], b[0]); bounds[1] = max(bounds[1], b[1])
            bounds[2] = min(bounds[2], b[2]); bounds[3] = max(bounds[3], b[3])
            bounds[4] = min(bounds[4], b[4]); bounds[5] = max(bounds[5], b[5])
        return tuple(bounds) if bounds[0] <= bounds[1] else None

    def _camera_for(self, scene_bounds: tuple, keyframe, preset: str) -> Dict:
        """Camera state looking at the keyframe's focus layer (or the whole body).

        Mirrors ``Interactive3DViewport.set_view`` but works off explicit bounds,
        so export never has to disturb the live camera.
        """
        bounds = scene_bounds
        if keyframe is not None and getattr(keyframe, "focus_layer", ""):
            focus = self._layer_bounds(keyframe.focus_layer)
            if focus is not None:
                bounds = focus
        structures = getattr(keyframe, "focus_structures", None) if keyframe is not None else None
        explicit = getattr(keyframe, "focus_bounds", None) if keyframe is not None else None
        if explicit or (structures and hasattr(self._viewport, "structure_bounds")):
            focus = tuple(explicit) if explicit else self._viewport.structure_bounds(structures)
            if focus is not None:
                from app.core.viewport import camera_for_bounds
                return camera_for_bounds(focus, preset or "anterior", 1.6)

        direction, up = VIEW_PRESETS.get(preset, VIEW_PRESETS["isometric"])
        center = ((bounds[0] + bounds[1]) / 2.0,
                  (bounds[2] + bounds[3]) / 2.0,
                  (bounds[4] + bounds[5]) / 2.0)
        diagonal = math.dist((bounds[0], bounds[2], bounds[4]),
                             (bounds[1], bounds[3], bounds[5])) or 1.0
        distance = diagonal * 1.75

        norm = math.sqrt(sum(c * c for c in direction)) or 1.0
        return {
            "position": [center[i] + direction[i] / norm * distance for i in range(3)],
            "focal": list(center),
            "up": list(up),
            "angle": 30.0,
        }

    # ------------------------------------------------------------ renderer
    def _build_renderer(self) -> None:
        """Off-screen render window that shares the scene's actors.

        Renders at ``render_dimensions()`` (output size × supersampling factor)
        and downsamples on capture, so edges are properly anti-aliased.
        """
        width, height = self._settings.render_dimensions()

        self._render_window = vtk.vtkRenderWindow()
        self._render_window.SetOffScreenRendering(True)
        self._render_window.SetSize(width, height)
        self._render_window.SetMultiSamples(0)

        self._renderer = vtk.vtkRenderer()
        top, bottom = BACKGROUNDS.get(
            self._config.get("render.background", "studio"), BACKGROUNDS["studio"])
        self._renderer.SetBackground(*bottom)
        self._renderer.SetBackground2(*top)
        self._renderer.SetGradientBackground(True)
        self._render_window.AddRenderer(self._renderer)

        # Three-point rig plus a rim light. Flat, single-source lighting is a
        # large part of why early renders read as "cheap plastic".
        headlight = vtk.vtkLight()
        headlight.SetLightTypeToHeadlight()
        headlight.SetIntensity(0.78)
        self._renderer.AddLight(headlight)

        for position, intensity, colour in (
            ((1.0, -0.7, 0.5), 0.50, (0.74, 0.84, 1.00)),    # key
            ((-1.1, 0.8, -0.5), 0.34, (0.62, 0.88, 1.00)),   # fill
            ((0.0, 1.2, -0.9), 0.22, (0.90, 0.95, 1.00)),    # rim
            ((0.0, 0.0, -1.0), 0.14, (0.55, 0.72, 1.00)),    # undershot
        ):
            light = vtk.vtkLight()
            light.SetPosition(*position)
            light.SetFocalPoint(0.0, 0.0, 0.0)
            light.SetIntensity(intensity)
            light.SetColor(*colour)
            self._renderer.AddLight(light)

        for state in self._viewport.layers.values():
            for actor in state.actors:
                self._renderer.AddActor(actor)

        self._apply_ambient_occlusion()

        # Titles and captions are drawn by Qt after capture (see captions.py):
        # VTK text cannot shape Indic or Arabic scripts.
        from app.video.captions import CaptionRenderer
        from app.audio.languages import language
        width, height = self._settings.even_dimensions()
        self._overlay = CaptionRenderer(
            width, height, rtl=language(str(self._config.get("audio.language", "en-IN"))).rtl,
            attribution=self._attribution())

    def _apply_ambient_occlusion(self) -> None:
        """Add a screen-space ambient occlusion pass.

        SSAO darkens creases and the gaps where organs meet, which is what makes
        a render read as physically solid. Guarded because it needs an OpenGL2
        backend and a depth-capable window; if it is unavailable we simply keep
        the standard forward-rendered look.
        """
        if not self._settings.ssao:
            return
        if not hasattr(vtk, "vtkSSAOPass") or not hasattr(vtk, "vtkRenderStepsPass"):
            return
        try:
            steps = vtk.vtkRenderStepsPass()
            ssao = vtk.vtkSSAOPass()
            ssao.SetDelegatePass(steps)
            ssao.SetRadius(0.045)          # fraction of the scene diagonal
            ssao.SetKernelSize(64)
            ssao.SetBlur(True)
            self._renderer.SetPass(ssao)
            self._ssao_applied = True
        except Exception:
            self._ssao_applied = False

    def _teardown_renderer(self) -> None:
        if getattr(self, "_active_effect", None) is not None:
            try:
                self._active_effect.end(self._viewport)
            finally:
                self._active_effect = None
        if self._renderer is not None and self._render_window is not None:
            try:
                for state in self._viewport.layers.values():
                    for actor in state.actors:
                        self._renderer.RemoveActor(actor)
                self._render_window.RemoveRenderer(self._renderer)
                self._render_window.Finalize()
            except Exception:
                pass
        self._renderer = None
        self._render_window = None

    # -------------------------------------------------------------- frames
    def _render_batch(self) -> None:
        if self._cancel:
            return
        try:
            for _ in range(self.BATCH):
                if self._cursor >= len(self._queue):
                    self._finish_rendering()
                    return
                self._render_frame()

            # A segment renders a partial final frame, so the raw counter can
            # nudge past the estimate; clamp it for a clean readout.
            done = min(self._frame_done, self._frame_total)
            percent = 15 + int(75 * done / max(1, self._frame_total))
            self.progress.emit(percent, f"Rendering frames {done}/{self._frame_total}…")
        except Exception as exc:
            self._timer.stop()
            self._teardown_renderer()
            self._close_writer()
            self._cleanup_work()
            self.failed.emit(f"Frame rendering failed: {type(exc).__name__}: {exc}")

    def _render_frame(self) -> None:
        segment = self._queue[self._cursor]
        fps = self._settings.fps
        local_time = self._frame_in_segment / float(fps)

        alpha = 1.0
        if segment.travel > 0 and local_time < segment.travel:
            alpha = ease_in_out_cubic(local_time / segment.travel)

        camera = lerp_state(segment.camera_from, segment.camera_to, alpha)
        self._apply_camera(camera)
        self._apply_layers(segment, alpha)
        if self._frame_in_segment == 0:
            self._switch_effect(segment.effect)
        if self._active_effect is not None:
            self._active_effect.apply(self._viewport, local_time)

        frame = self._capture()
        if frame is not None and self._overlay is not None:
            caption = ""
            if self._settings.burn_captions:
                caption = chunk_at(segment.chunks, local_time - segment.speech_start, segment.speech)
            frame = self._overlay.compose(frame, segment.title, caption)
        if frame is not None:
            if self._writer is None:
                self._writer = imageio.get_writer(
                    str(self._video_path),
                    fps=fps,
                    codec="libx264",
                    quality=self._settings.quality,
                    macro_block_size=None,
                    # slow preset = better compression at the same quality;
                    # faststart moves the index to the front for streaming.
                    # NOTE: do not add -pix_fmt here — imageio already emits it
                    # and a duplicate triggers "Multiple -pix_fmt options" noise.
                    ffmpeg_params=["-preset", "slow",
                                   "-movflags", "+faststart"],
                )
            self._writer.append_data(frame)

        self._frame_done += 1
        self._frame_in_segment += 1
        if local_time >= segment.duration:
            self._cursor += 1
            self._frame_in_segment = 0

    def _switch_effect(self, effect) -> None:
        if self._active_effect is not None:
            self._active_effect.end(self._viewport)
        self._active_effect = effect
        if effect is not None:
            effect.begin(self._viewport)

    def _apply_camera(self, state: Dict) -> None:
        camera = self._renderer.GetActiveCamera()
        if state.get("position"):
            camera.SetPosition(*state["position"])
        if state.get("focal"):
            camera.SetFocalPoint(*state["focal"])
        if state.get("up"):
            camera.SetViewUp(*state["up"])
        camera.SetViewAngle(float(state.get("angle", 30.0)))
        camera.OrthogonalizeViewUp()
        self._renderer.ResetCameraClippingRange()

    def _apply_layers(self, segment: Segment, alpha: float) -> None:
        for layer_id, state in self._viewport.layers.items():
            start = segment.layers_from.get(layer_id, state.opacity)
            end = segment.layers_to.get(layer_id, start)
            value = start + (end - start) * alpha
            for actor in state.actors:
                actor.GetProperty().SetOpacity(value)

    def _capture(self):
        """Grab one frame, downsampling the supersampled render if enabled."""
        self._render_window.Render()
        filt = vtk.vtkWindowToImageFilter()
        filt.SetInput(self._render_window)
        filt.SetInputBufferTypeToRGB()
        filt.ReadFrontBufferOff()
        filt.SetScale(1)
        filt.Update()

        image = filt.GetOutput()
        width, height, _ = image.GetDimensions()
        scalars = vtk_to_numpy(image.GetPointData().GetScalars())
        frame = np.flipud(scalars.reshape(height, width, -1))

        factor = max(1, int(self._settings.supersample))
        if factor > 1:
            frame = _box_downsample(frame, factor)
        return np.ascontiguousarray(frame)

    # ------------------------------------------------------------ finishing
    def _finish_rendering(self) -> None:
        self._timer.stop()
        self._close_writer()
        self._teardown_renderer()

        if self._cancel:
            self._cleanup_work()
            return

        # Restore live layer opacities (export drove them directly).
        for layer_id, state in self._viewport.layers.items():
            self._viewport.set_layer_opacity(layer_id, state.opacity)

        self.progress.emit(92, "Mixing narration…")
        self._audio_path = AudioTrackBuilder().build(
            self._clips,
            self._total_seconds + 0.5,
            Path(self._work_dir) / "narration.wav",
        ) if self._settings.include_audio else None

        self.progress.emit(95, "Encoding video…")
        self._mux = _MuxWorker(self._video_path, self._audio_path, self._out_path,
                               self._meta_comment(), self)
        self._mux.done.connect(self._on_mux_done)
        self._mux.failed.connect(self._on_mux_failed)
        self._mux.start()

    def _on_mux_done(self, path: str) -> None:
        self._cleanup_work()
        self.progress.emit(100, "Done")
        self.finished.emit(path)

    def _on_mux_failed(self, message: str) -> None:
        self._cleanup_work()
        self.failed.emit(f"Encoding failed: {message}")

    def _meta_comment(self) -> str:
        """Provenance string written into the container metadata."""
        tour = getattr(self, "_tour", None)
        parts = ["BioHuman3D"]
        if tour is not None:
            parts.append(f"subject={getattr(tour, 'id', '?')}")
            parts.append(f"title={getattr(tour, 'title', '?')}")
        if self._meta:
            parts.extend(f"{k}={v}" for k, v in self._meta.items() if v)
        parts.append(f"render={self._settings.describe()}")
        return " | ".join(parts)

    def _close_writer(self) -> None:
        if self._writer is not None:
            try:
                self._writer.close()
            except Exception:
                pass
            self._writer = None

    def _cleanup_work(self) -> None:
        """Remove the scratch directory, keeping the finished output."""
        work = getattr(self, "_work_dir", None)
        if work is not None and Path(work).exists():
            shutil.rmtree(work, ignore_errors=True)


def _box_downsample(frame, factor: int):
    """Average ``factor``×``factor`` blocks — a correct 2× SSAA resolve.

    Uses an integer block mean rather than a resampling library so video export
    has no dependency beyond numpy.
    """
    height, width = frame.shape[0], frame.shape[1]
    height -= height % factor
    width -= width % factor
    if height == 0 or width == 0:
        raise ValueError("frame smaller than the supersampling factor")
    trimmed = frame[:height, :width, ...]
    channels = trimmed.shape[2]
    reshaped = trimmed.reshape(height // factor, factor, width // factor, factor,
                               channels)
    return reshaped.mean(axis=(1, 3)).astype(trimmed.dtype)


# ---------------------------------------------------------------------------
# Availability probe for the UI
# ---------------------------------------------------------------------------
def video_support_status() -> tuple:
    """``(available, detail)`` for tooltips and the doctor tool."""
    if not IMAGEIO_AVAILABLE:
        return False, "needs imageio"
    if not IMAGEIO_FFMPEG_AVAILABLE:
        return False, "needs imageio-ffmpeg"
    if not NUMPY_AVAILABLE:
        return False, "needs numpy"
    try:
        version = imageio_ffmpeg.get_ffmpeg_version()
    except Exception:
        version = "unknown"
    return True, f"ffmpeg {version} ready"
