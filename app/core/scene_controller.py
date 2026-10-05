"""
SceneController — the application's single source of truth.

Widgets emit intent ("fade the muscles to 30%"); the controller owns the state
change, applies it to the viewport, keeps the AI context in sync, and broadcasts
the resulting status. Guided-tour playback and audio synchronisation also live
here, because they couple the viewport and the audio engine.

Keeping this layer thin is deliberate: it is the only place where the UI, the
renderer and the AI meet, so it stays readable and testable.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

from PyQt6.QtCore import QObject, pyqtSignal

from app.audio.scripts import Tour, get_tour, system_presets, tour_list
from app.core.model_registry import LayerRegistry
from app.core.tour import TourEngine


class SceneController(QObject):
    """Mediates between the viewport, the audio engine and the AI manager."""

    statusMessage = pyqtSignal(str, int)       # text, timeout ms (0 = sticky)
    structureChanged = pyqtSignal(str, str)    # layer_id, label
    layerStateChanged = pyqtSignal(str, str, object)   # layer_id, prop, value
    tourStarted = pyqtSignal(str, str)         # tour id, title
    tourKeyframe = pyqtSignal(int, int, str)   # index, total, title
    tourFinished = pyqtSignal(str)
    sceneLoaded = pyqtSignal(int, int)         # real layers, placeholder layers
    busyChanged = pyqtSignal(bool)

    def __init__(self, config, registry: LayerRegistry, viewport, audio, ai,
                 parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self.config = config
        self.registry = registry
        self.viewport = viewport
        self.audio = audio
        self.ai = ai

        self._selected_id = ""
        self._selected_label = ""
        self._current_view = "isometric"
        self._active_tour: Optional[Tour] = None

        self.tour = TourEngine(viewport, self)

        # -- wiring ---------------------------------------------------------
        self.tour.keyframeEntered.connect(self._on_keyframe_entered)
        self.tour.finished.connect(self._on_tour_finished)
        if audio is not None:
            audio.cueFinished.connect(self._on_cue_finished)

    # ------------------------------------------------------------ startup
    def load_scene(self, *, allow_placeholders: bool = True) -> None:
        self.busyChanged.emit(True)
        try:
            real, placeholders = self.viewport.load_scene(allow_placeholders=allow_placeholders)
        finally:
            self.busyChanged.emit(False)

        self.sceneLoaded.emit(real, placeholders)
        if placeholders and not real:
            self.statusMessage.emit(
                "Placeholder geometry in use — run tools\\generate_demo_models.py "
                "or add real models to app\\assets\\models.", 9000)
        elif placeholders:
            self.statusMessage.emit(
                f"{real} dataset(s) loaded, {placeholders} placeholder layer(s).", 6000)
        else:
            self.statusMessage.emit(f"{real} anatomical layers loaded.", 4000)

        self._sync_ai_context()

    # --------------------------------------------------------- structure
    def select_structure(self, layer_id: str, label: str = "") -> None:
        """Called when the user picks in the viewport or clicks a sidebar row."""
        self._selected_id = layer_id
        self._selected_label = label or (self.registry.label(layer_id) if layer_id else "")
        if layer_id:
            highlight = getattr(self.viewport, "highlight_structure", None)
            if not (label and highlight is not None and highlight(label)):
                self.viewport.set_highlight(layer_id)
        self.structureChanged.emit(self._selected_id, self._selected_label)
        self._sync_ai_context()

    def clear_selection(self) -> None:
        self._selected_id = ""
        self._selected_label = ""
        self.viewport.set_highlight(None)
        self.structureChanged.emit("", "")
        self._sync_ai_context()

    @property
    def selected_id(self) -> str:
        return self._selected_id

    @property
    def selected_label(self) -> str:
        return self._selected_label

    # ----------------------------------------------------- layer control
    def set_layer_opacity(self, layer_id: str, opacity: float) -> None:
        self.viewport.set_layer_opacity(layer_id, opacity)
        state = self.viewport.layers.get(layer_id)
        if state is not None and opacity > 0.5:
            state.base_opacity = opacity       # remember the "restored" value
        self.layerStateChanged.emit(layer_id, "opacity", opacity)
        self._sync_ai_context()

    def set_layer_visible(self, layer_id: str, visible: bool) -> None:
        self.viewport.set_layer_visible(layer_id, visible)
        self.layerStateChanged.emit(layer_id, "visible", visible)
        self._sync_ai_context()

    def isolate_layer(self, layer_id: str) -> None:
        self.viewport.isolate_layer(layer_id)
        self.select_structure(layer_id)
        self.statusMessage.emit(f"Isolated: {self.registry.label(layer_id)}", 3000)

    def show_all(self) -> None:
        self.viewport.show_all_layers()
        for layer_id, state in self.viewport.layers.items():
            self.layerStateChanged.emit(layer_id, "opacity", state.base_opacity)
            self.layerStateChanged.emit(layer_id, "visible", True)
        self._sync_ai_context()

    def apply_preset(self, preset_name: str) -> None:
        presets = system_presets()
        visible = presets.get(preset_name)
        if visible is None:
            # Unknown label: report it rather than returning silently, so a stale
            # UI list cannot fail invisibly.
            self.statusMessage.emit(f"Unknown system preset: {preset_name}", 4000)
            return
        self.viewport.apply_layer_preset(visible)
        self.statusMessage.emit(f"Preset applied: {preset_name}", 2500)
        self._sync_ai_context()

    # ------------------------------------------------------------- camera
    def set_view(self, preset: str) -> None:
        self._current_view = preset
        self.viewport.set_view(preset)
        self._sync_ai_context()

    def reset_view(self) -> None:
        self.viewport.reset_camera()
        self.statusMessage.emit("Camera reset", 2000)

    # -------------------------------------------------------------- tours
    def available_tours(self) -> List[Tour]:
        """Every authored tour, in body-system order.

        This previously returned a hard-coded triplet (skeletal, muscular,
        cardiorespiratory). That single line silently defeated the whole
        per-system narration design: the export dropdown only ever offered three
        subjects with *skeletal first*, so any export without an explicit change
        rendered the skeleton — and syncing to any other system failed because
        ``findData()`` could not find a tour that was never added to the list.

        Never hand-list tours again: derive them from the registry.
        """
        return tour_list()

    def start_tour(self, tour_id: str, *, speak: bool = True) -> bool:
        tour = get_tour(tour_id)
        if tour is None:
            return False

        self._active_tour = tour
        self.tour.load(tour)
        self.tourStarted.emit(tour.id, tour.title)

        if speak and self.audio is not None and self.audio.is_available():
            self.audio.play_cues(tour.cues())
        elif speak and self.audio is not None:
            self.statusMessage.emit(
                "Voiceover unavailable — install pyttsx3 or configure a cloud voice.", 6000)

        self.tour.start(0)
        return True

    def stop_tour(self) -> None:
        self.tour.stop()
        if self.audio is not None:
            self.audio.stop()
        if self._active_tour is not None:
            self.tourFinished.emit(self._active_tour.id)

    def next_tour_step(self) -> None:
        if self.audio is not None:
            self.audio.stop()
        self.tour.next_keyframe()

    def previous_tour_step(self) -> None:
        if self.audio is not None:
            self.audio.stop()
        self.tour.previous_keyframe()

    def _on_keyframe_entered(self, index: int, keyframe) -> None:
        total = len(self._active_tour.keyframes) if self._active_tour else 0
        self.tourKeyframe.emit(index, total, getattr(keyframe, "title", ""))

        # Mirror the beat's layer state into the sidebar without extra signals.
        for layer_id, opacity in (getattr(keyframe, "layers", {}) or {}).items():
            self.layerStateChanged.emit(layer_id, "opacity", opacity)

        if self.audio is not None and getattr(keyframe, "narration", ""):
            # If a cloud/silent backend finished early, keep the beat moving.
            pass

    def _on_cue_finished(self, index: int) -> None:
        self.tour.notify_narration_finished(index)

    def _on_tour_finished(self, tour_id: str) -> None:
        self.tourFinished.emit(tour_id)
        if self.audio is not None:
            self.audio.stop()
        self.statusMessage.emit("Guided tour complete", 5000)

    # ---------------------------------------------------------- narration
    def narrate_selection(self) -> None:
        """Speak the registry description of the current selection immediately."""
        if not self._selected_id or self.audio is None:
            self.statusMessage.emit("Select a structure first.", 3000)
            return
        spec = self.registry.spec(self._selected_id)
        text = spec.description if spec else ""
        if not text:
            # No authored script — ask the AI for a short spoken line.
            if self.ai is not None:
                self.ai.request_voiceover()
            return
        self.audio.speak(text, title=self._selected_label)
        self.statusMessage.emit(f"Speaking: {self._selected_label}", 3000)

    def speak_text(self, text: str) -> None:
        if self.audio is not None and text.strip():
            self.audio.speak(text)

    # -------------------------------------------------------------- export
    def snapshot(self, path: Path, magnification: int = 2) -> Optional[Path]:
        result = self.viewport.snapshot(path, magnification)
        if result:
            self.statusMessage.emit(f"Screenshot saved: {result.name}", 5000)
        else:
            self.statusMessage.emit("Screenshot failed.", 4000)
        return result

    # ------------------------------------------------------------- context
    def _sync_ai_context(self) -> None:
        """Push the current scene state into the AI manager for grounding."""
        if self.ai is None:
            return
        try:
            visible = self.viewport.visible_layers()
        except Exception:
            visible = []
        self.ai.set_scene_context(
            selected_id=self._selected_id,
            selected_label=self._selected_label,
            visible_layers=visible,
            view=self._current_view,
        )
