"""
Guided tour playback.

``TourEngine`` interpolates the camera between authored keyframes on a 60 Hz
timer while the narration plays. It is deliberately decoupled from both the
viewport and the audio engine: it emits :attr:`keyframeEntered` and waits to be
told when the narration for that beat has finished, so audio and visuals stay in
lock-step instead of drifting.
"""
from __future__ import annotations

import math
import time
from enum import Enum, auto
from typing import Dict, List, Optional, Sequence

from PyQt6.QtCore import QObject, QTimer, pyqtSignal


class _Phase(Enum):
    IDLE = auto()
    TRAVEL = auto()
    DWELL = auto()
    FINISHED = auto()


def ease_in_out_cubic(t: float) -> float:
    """Smooth acceleration/deceleration — no camera 'snap' at either end."""
    t = max(0.0, min(1.0, t))
    return 4 * t ** 3 if t < 0.5 else 1 - pow(-2 * t + 2, 3) / 2


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _lerp_vec(a: Sequence[float], b: Sequence[float], t: float) -> List[float]:
    return [_lerp(a[i], b[i], t) for i in range(min(len(a), len(b)))]


class TourEngine(QObject):
    """Drives a :class:`~app.audio.scripts.Tour` against a viewport."""

    keyframeEntered = pyqtSignal(int, object)   # index, TourKeyframe
    progress = pyqtSignal(int, float)           # index, 0..1
    started = pyqtSignal(str)                   # tour id
    finished = pyqtSignal(str)                  # tour id
    stateChanged = pyqtSignal(bool)             # running?

    TICK_MS = 16                 # ~60 Hz
    MAX_DWELL_SECONDS = 120.0    # safety valve if narration never reports back

    def __init__(self, viewport, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._viewport = viewport
        self._tour = None
        self._index = -1
        self._phase = _Phase.IDLE
        self._elapsed = 0.0
        self._travel = 1.0
        self._dwell = 0.0
        self._from_state: Dict[str, Sequence[float]] = {}
        self._to_state: Dict[str, Sequence[float]] = {}
        self._waiting_for_narration = False
        self._paused = False

        self._timer = QTimer(self)
        self._timer.setInterval(self.TICK_MS)
        self._timer.timeout.connect(self._tick)
        self._last_tick = time.perf_counter()

    # ------------------------------------------------------------- public
    @property
    def tour(self):
        return self._tour

    @property
    def current_index(self) -> int:
        return self._index

    @property
    def is_running(self) -> bool:
        return self._phase in (_Phase.TRAVEL, _Phase.DWELL) and not self._paused

    def load(self, tour) -> None:
        self.stop()
        self._tour = tour
        self._index = -1
        self._phase = _Phase.IDLE

    def start(self, index: int = 0) -> None:
        if self._tour is None or not self._tour.keyframes:
            return
        self._tour_id = getattr(self._tour, "id", "")
        self._index = -1
        self._phase = _Phase.IDLE
        self._paused = False
        self.started.emit(self._tour_id)
        self.stateChanged.emit(True)
        self._last_tick = time.perf_counter()
        self._timer.start()
        self._advance_to(max(0, min(index, len(self._tour.keyframes) - 1)))

    def stop(self) -> None:
        paused = self._paused
        self._timer.stop()
        self._phase = _Phase.IDLE
        self._index = -1
        self._waiting_for_narration = False
        if not paused:
            self.stateChanged.emit(False)

    def pause(self) -> None:
        self._paused = True
        self.stateChanged.emit(False)

    def resume(self) -> None:
        self._paused = False
        self._last_tick = time.perf_counter()
        self.stateChanged.emit(True)

    def next_keyframe(self) -> None:
        self._advance_to(self._index + 1)

    def previous_keyframe(self) -> None:
        self._advance_to(max(0, self._index - 1))

    def jump_to(self, index: int) -> None:
        self._advance_to(index)

    def notify_narration_finished(self, index: int) -> None:
        """Called by the controller when the TTS cue for *index* completes."""
        if index != self._index:
            return
        self._waiting_for_narration = False
        if self._phase is _Phase.DWELL:
            self._advance_to(self._index + 1)

    # ------------------------------------------------------------ internals
    def _advance_to(self, index: int) -> None:
        if self._tour is None:
            return
        if index >= len(self._tour.keyframes):
            self._phase = _Phase.FINISHED
            self._timer.stop()
            self.stateChanged.emit(False)
            self.finished.emit(self._tour_id)
            return

        self._index = index
        keyframe = self._tour.keyframes[index]

        # -- apply the layer state for this beat ---------------------------
        for layer_id, opacity in (keyframe.layers or {}).items():
            self._viewport.set_layer_opacity(layer_id, opacity)
            if layer_id in getattr(self._viewport, "layers", {}):
                self._viewport.layers[layer_id].base_opacity = opacity
                self._viewport.set_layer_visible(layer_id, True)
        if keyframe.focus_layer:
            self._viewport.set_highlight(keyframe.focus_layer)

        # -- camera --------------------------------------------------------
        self._from_state = self._viewport.camera_state() or {}
        target = keyframe.camera
        if target:
            self._to_state = dict(target)
        elif keyframe.view:
            self._viewport.set_view(keyframe.view)
            self._to_state = self._viewport.camera_state() or self._from_state
        else:
            self._to_state = self._from_state

        self._travel = max(0.35, float(keyframe.travel_seconds))
        self._dwell = max(0.0, float(keyframe.dwell_seconds))
        self._elapsed = 0.0
        self._waiting_for_narration = self._dwell <= 0.0
        self._phase = _Phase.TRAVEL

        self.keyframeEntered.emit(index, keyframe)
        if not self._timer.isActive():
            self._last_tick = time.perf_counter()
            self._timer.start()

    def _tick(self) -> None:
        now = time.perf_counter()
        dt = now - self._last_tick
        self._last_tick = now
        if self._paused or self._phase is _Phase.IDLE:
            return

        self._elapsed += dt

        if self._phase is _Phase.TRAVEL:
            raw = min(1.0, self._elapsed / self._travel) if self._travel else 1.0
            eased = ease_in_out_cubic(raw)
            self._apply_interpolated(eased)
            self.progress.emit(self._index, raw)
            if raw >= 1.0:
                self._phase = _Phase.DWELL
                self._elapsed = 0.0
            return

        if self._phase is _Phase.DWELL:
            if self._waiting_for_narration:
                if self._elapsed >= self.MAX_DWELL_SECONDS:
                    self._waiting_for_narration = False
                    self._advance_to(self._index + 1)
                return
            if self._elapsed >= max(self._dwell, 0.6):
                self._advance_to(self._index + 1)

    def _apply_interpolated(self, t: float) -> None:
        if not self._from_state or not self._to_state:
            return
        state = {
            "position": _lerp_vec(self._from_state.get("position", (0, 0, 0)),
                                  self._to_state.get("position", (0, 0, 0)), t),
            "focal": _lerp_vec(self._from_state.get("focal", (0, 0, 0)),
                               self._to_state.get("focal", (0, 0, 0)), t),
            "up": _lerp_vec(self._from_state.get("up", (0, 0, 1)),
                            self._to_state.get("up", (0, 0, 1)), t),
            "angle": _lerp(float(self._from_state.get("angle", 30.0)),
                           float(self._to_state.get("angle", 30.0)), t),
        }
        self._viewport.apply_camera_state(state)
