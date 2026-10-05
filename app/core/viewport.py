"""
Interactive 3D anatomy viewport.

``Interactive3DViewport`` embeds a VTK render window inside Qt and exposes an
anatomy-oriented API on top of it:

    load_scene()                       # build layers from the registry
    set_layer_opacity(id, 0.25)        # fade (proper alpha via depth peeling)
    set_layer_visible(id, False)       # hide
    isolate_layer(id)                  # solo one layer, ghost the rest
    set_view("anterior")               # camera presets
    rotate()/zoom()/pan()              # programmatic camera control
    apply_clip_plane("x", 0.4)         # cross-section slicing
    camera_state()/apply_camera_state()# keyframes for guided tours
    snapshot(path, magnification=2)    # high-resolution export

Threading: all VTK objects live on the GUI thread. Workers must hand back plain
Python data only.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from PyQt6.QtCore import QTimer, pyqtSignal
from PyQt6.QtWidgets import QLabel, QVBoxLayout, QWidget

from app.anatomy import structures as st
from app.core.model_registry import LayerRegistry, LayerSpec
from app.core import vtk_utils as vu
from app.core.vtk_utils import VTK_AVAILABLE, vtk, QVTKRenderWindowInteractor


# ---------------------------------------------------------------------------
# Scene data
# ---------------------------------------------------------------------------
@dataclass
class LayerState:
    """Everything the viewport needs to know about one manipulable layer."""

    id: str
    label: str
    group: str = ""
    actors: List[object] = field(default_factory=list)
    polydata: List[object] = field(default_factory=list)
    opacity: float = 1.0
    visible: bool = True
    color: Tuple[float, float, float] = (0.8, 0.8, 0.8)
    base_opacity: float = 1.0            # restored after an isolate/ghost
    origin: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    explode_vector: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    is_placeholder: bool = False
    structures: List[str] = field(default_factory=list)   # parallel to ``actors``
    source: str = ""                     # provenance tag of the model file

    @property
    def actor_count(self) -> int:
        return len(self.actors)


#: camera preset name → (direction, view-up). Assumes Z-up anatomical models.
VIEW_PRESETS: Dict[str, Tuple[Tuple[float, float, float], Tuple[float, float, float]]] = {
    "anterior":  ((0.0, -1.0, 0.0), (0.0, 0.0, 1.0)),
    "posterior": ((0.0, 1.0, 0.0),  (0.0, 0.0, 1.0)),
    "left":      ((-1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    "right":     ((1.0, 0.0, 0.0),  (0.0, 0.0, 1.0)),
    "superior":  ((0.0, 0.0, 1.0),  (0.0, -1.0, 0.0)),
    "inferior":  ((0.0, 0.0, -1.0), (0.0, 1.0, 0.0)),
    "isometric": ((0.85, -0.85, 0.55), (0.0, 0.0, 1.0)),
}

BACKGROUNDS: Dict[str, Tuple[Tuple[float, float, float], Tuple[float, float, float]]] = {
    "studio":   ((0.043, 0.055, 0.078), (0.012, 0.016, 0.024)),
    "slate":    ((0.078, 0.086, 0.102), (0.031, 0.035, 0.043)),
    "clinical": ((0.925, 0.933, 0.945), (0.780, 0.800, 0.826)),
    "black":    ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
}

_GHOST_OPACITY = 0.06


def camera_for_bounds(bounds: Sequence[float], preset: str = "isometric",
                      distance_factor: float = 1.75) -> Dict[str, List[float]]:
    """Camera state framing *bounds* from a view preset (used by tours and export)."""
    direction, up = VIEW_PRESETS.get(preset or "isometric", VIEW_PRESETS["isometric"])
    center = [(bounds[0] + bounds[1]) / 2.0, (bounds[2] + bounds[3]) / 2.0,
              (bounds[4] + bounds[5]) / 2.0]
    diagonal = math.dist((bounds[0], bounds[2], bounds[4]), (bounds[1], bounds[3], bounds[5])) or 1.0
    # Small structures still get some context, and the lower third of the frame
    # is reserved for captions, so frame a little wider and look slightly low.
    diagonal = max(diagonal, 0.42)
    distance = diagonal * distance_factor
    center[2] -= diagonal * 0.12
    norm = math.sqrt(sum(c * c for c in direction)) or 1.0
    return {"position": [center[i] + direction[i] / norm * distance for i in range(3)],
            "focal": center, "up": list(up), "angle": 30.0}
_HUGE_MESH_CELLS = 300_000     # above this we use a cheap bounding-box highlight


# ---------------------------------------------------------------------------
# Widget
# ---------------------------------------------------------------------------
class Interactive3DViewport(QVTKRenderWindowInteractor):
    """Embeddable VTK viewport with an anatomy-aware scene API."""

    structurePicked = pyqtSignal(str, str)   # layer_id, human label ("" = empty space)
    hoverChanged = pyqtSignal(str)           # hovered layer label ("" = none)
    cameraChanged = pyqtSignal()
    fpsUpdated = pyqtSignal(float)
    sceneLoaded = pyqtSignal(int, int)       # loaded_layers, placeholder_layers

    # ---------------------------------------------------------------- setup
    def __init__(self, registry: LayerRegistry, config, parent: Optional[QWidget] = None):
        if not VTK_AVAILABLE:
            raise RuntimeError(f"VTK unavailable: {vu.VTK_IMPORT_ERROR}")
        super().__init__(parent)

        self._registry = registry
        self._config = config
        self._layers: Dict[str, LayerState] = {}
        self._actor_to_layer: Dict[int, str] = {}
        self._actor_to_structure: Dict[object, str] = {}
        self._structure_index: Dict[str, Tuple[str, int]] = {}   # lower name -> (layer, idx)
        self._highlight_cache: Dict[str, object] = {}
        self._highlight_actor = None
        self._highlight_layer: Optional[str] = None
        self._clip_plane = None
        self._explode = 0.0
        self._initialized = False
        self._last_pick_id: Optional[str] = None

        self.setMinimumSize(560, 400)

        # -- render window / renderer --------------------------------------
        self._render_window = self.GetRenderWindow()
        self._renderer = vtk.vtkRenderer()
        self._render_window.AddRenderer(self._renderer)

        self._configure_gl()
        self._configure_background(self._config.get("render.background", "studio"))
        self._configure_lighting()
        self._configure_orientation_widget()

        from app.core.interaction import AnatomyInteractorStyle
        self._style = AnatomyInteractorStyle(self)
        self.SetInteractorStyle(self._style)

        # -- fps probe ------------------------------------------------------
        self._fps_timer = QTimer(self)
        self._fps_timer.setInterval(1000)
        self._fps_timer.timeout.connect(self._sample_fps)
        self._fps_timer.start()
        self._last_frame_time = time.time()

    # -- OpenGL / lighting -------------------------------------------------
    def _configure_gl(self) -> None:
        """Anti-aliasing + depth peeling (correct order-independent transparency)."""
        if self._config.get("render.depth_peeling", True):
            self._render_window.SetAlphaBitPlanes(True)
            self._render_window.SetMultiSamples(0)
            self._renderer.SetUseDepthPeeling(True)
            self._renderer.SetMaximumNumberOfPeels(8)
            self._renderer.SetOcclusionRatio(0.08)
        else:
            self._renderer.SetUseDepthPeeling(False)
            self._render_window.SetMultiSamples(4)

        if self._config.get("render.fxaa", True) and hasattr(self._renderer, "SetUseFXAA"):
            self._renderer.SetUseFXAA(True)

        self._render_window.SetDesiredUpdateRate(30.0)   # LOD while dragging

    def _configure_background(self, preset: str) -> None:
        top, bottom = BACKGROUNDS.get(preset, BACKGROUNDS["studio"])
        self._renderer.SetBackground(*bottom)
        self._renderer.SetBackground2(*top)
        self._renderer.SetGradientBackground(True)

    def _configure_lighting(self) -> None:
        """Key + fill + rim: enough shape definition to read anatomy without lambertian mush."""
        # Key light follows the camera.
        key = vtk.vtkLight()
        key.SetLightTypeToHeadlight()
        key.SetIntensity(0.85)
        key.SetColor(1.0, 0.98, 0.95)
        self._renderer.AddLight(key)

        fill = vtk.vtkLight()
        fill.SetPosition(1.0, -0.6, 0.4)
        fill.SetFocalPoint(0.0, 0.0, 0.0)
        fill.SetIntensity(0.42)
        fill.SetColor(0.72, 0.82, 1.0)
        self._renderer.AddLight(fill)

        rim = vtk.vtkLight()
        rim.SetPosition(-1.0, 0.9, -0.6)
        rim.SetFocalPoint(0.0, 0.0, 0.0)
        rim.SetIntensity(0.30)
        rim.SetColor(0.65, 0.90, 1.0)
        self._renderer.AddLight(rim)

    def _configure_orientation_widget(self) -> None:
        if not self._config.get("render.show_axes", True):
            return
        axes = vtk.vtkAxesActor()
        axes.SetXAxisLabelText("R")
        axes.SetYAxisLabelText("A")
        axes.SetZAxisLabelText("S")
        axes.SetTotalLength(0.16, 0.16, 0.16)
        for cap in (axes.GetXAxisCaptionActor2D(),
                    axes.GetYAxisCaptionActor2D(),
                    axes.GetZAxisCaptionActor2D()):
            prop = cap.GetTextActor().GetTextProperty()
            prop.SetColor(0.62, 0.70, 0.80)
            prop.SetFontSize(11)
        self._axes = axes
        self._marker = vtk.vtkOrientationMarkerWidget()
        self._marker.SetOrientationMarker(axes)
        self._marker.SetInteractor(self)
        self._marker.SetViewport(0.0, 0.0, 0.16, 0.22)

    # -- Qt lifecycle -------------------------------------------------------
    def showEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        super().showEvent(event)
        if not self._initialized:
            self.Initialize()
            try:
                self._marker.SetEnabled(1)
                self._marker.InteractiveOff()
            except Exception:
                pass
            self._initialized = True
            self._renderer.ResetCamera()
            self._render_window.Render()

    # ---------------------------------------------------------------- scene
    def load_scene(self, *, allow_placeholders: bool = True,
                   decimate_ratio: float = 0.0) -> Tuple[int, int]:
        """Populate the scene from the layer registry.

        Returns ``(real_layers, placeholder_layers)``. Layers without a backing
        file receive a deterministic placeholder primitive so the UI, picking and
        opacity tools remain fully exercisable before real anatomy is installed.
        """
        self.clear_scene()
        loaded = placeholders = 0
        for spec in self._registry.specs:
            resolved = self._registry.resolve(spec.id)
            polydata_list = []
            names: List[str] = []
            source = ""

            if resolved is not None:
                for path in resolved.paths:
                    if path.suffix.lower() in (".glb", ".gltf"):
                        for name, poly in vu.read_gltf_nodes(path):
                            if poly is not None and poly.GetNumberOfPoints():
                                polydata_list.append(
                                    vu.condition_polydata(poly, decimate_ratio=decimate_ratio))
                                names.append(st.display_name(name))
                    else:
                        poly, _err = vu.read_polydata(path)
                        if poly is None or not poly.GetNumberOfPoints():
                            continue
                        source = source or st.source_of(poly)
                        if st.is_labelled(poly):
                            # Condition once for the whole layer (cell labels
                            # survive smoothing + normals), then split into one
                            # pickable actor per named structure.
                            field_data = vtk.vtkFieldData()
                            field_data.ShallowCopy(poly.GetFieldData())
                            conditioned = vu.condition_polydata(
                                poly, smooth_iterations=0 if source.startswith("procedural") else 12)
                            conditioned.GetFieldData().ShallowCopy(field_data)
                            for name, piece in st.split_structures(conditioned, spec.label):
                                if decimate_ratio:
                                    piece = vu.condition_polydata(
                                        piece, decimate_ratio=decimate_ratio, smooth_iterations=0)
                                polydata_list.append(piece)
                                names.append(st.display_name(name))
                        else:
                            polydata_list.append(
                                vu.condition_polydata(poly, decimate_ratio=decimate_ratio))
                            names.append("")

            is_placeholder = False
            if not polydata_list and allow_placeholders:
                poly = vu.make_placeholder_polydata(spec.id)
                if poly is not None:
                    polydata_list = [poly]
                    is_placeholder = True

            if not polydata_list:
                continue

            if is_placeholder:
                names = [""]
            self._add_layer(spec, polydata_list, is_placeholder, names, source)
            if is_placeholder:
                placeholders += 1
            else:
                loaded += 1

        # Order matters for transparency: render thick/outer layers first.
        self._apply_layer_ordering()
        self.reset_camera()
        self.sceneLoaded.emit(loaded, placeholders)
        return loaded, placeholders

    def _add_layer(self, spec: LayerSpec, polydata_list: List[object], placeholder: bool,
                   names: Optional[List[str]] = None, source: str = "") -> None:
        state = LayerState(
            id=spec.id,
            label=spec.label,
            group=spec.group,
            opacity=spec.default_opacity,
            visible=spec.default_visible,
            color=tuple(spec.color),
            base_opacity=spec.default_opacity,
            explode_vector=tuple(spec.explode_vector),
            is_placeholder=placeholder,
            source=source,
        )
        names = list(names or [])
        names += [""] * (len(polydata_list) - len(names))

        for poly, name in zip(polydata_list, names):
            actor = vu.make_surface_actor(poly, spec.color, spec.default_opacity)
            actor.SetVisibility(spec.default_visible)
            if placeholder:
                actor.GetProperty().SetRepresentationToWireframe()
                actor.GetProperty().SetLineWidth(1.2)

            self._renderer.AddActor(actor)
            state.actors.append(actor)
            state.polydata.append(poly)
            self._actor_to_layer[id(actor)] = spec.id
            self._actor_to_layer[actor.GetAddressAsString("")] = spec.id
            if name:
                self._actor_to_structure[id(actor)] = name
                self._structure_index.setdefault(name.lower(), (spec.id, len(state.actors) - 1))
            state.structures.append(name)

        self._layers[spec.id] = state

    def _apply_layer_ordering(self) -> None:
        """Back-to-front hint for the renderer (helps transparency correctness)."""
        order = ["skeleton", "cartilage", "tendons", "muscles", "fascia", "skin"]
        for index, layer_id in enumerate(order):
            state = self._layers.get(layer_id)
            if not state:
                continue
            for actor in state.actors:
                if hasattr(actor, "SetRenderOrder"):
                    actor.SetRenderOrder(min(index + 1, 5))

    # ------------------------------------------------------------ accessors
    @property
    def renderer(self):
        return self._renderer

    @property
    def layers(self) -> Dict[str, LayerState]:
        return self._layers

    def layer_state(self, layer_id: str) -> Optional[LayerState]:
        return self._layers.get(layer_id)

    def layer_id_for_actor(self, actor) -> Optional[str]:
        if actor is None:
            return None
        key = self._actor_to_layer.get(id(actor))
        if key:
            return key
        try:
            return self._actor_to_layer.get(actor.GetAddressAsString(""))
        except Exception:
            return None

    def pickable_actors(self) -> List[object]:
        out = []
        for state in self._layers.values():
            if state.visible and state.opacity > 0.08:
                out.extend(a for a in state.actors
                           if a.GetVisibility() and a.GetProperty().GetOpacity() > 0.08)
        return out

    def visible_layers(self) -> List[str]:
        return [s.id for s in self._layers.values() if s.visible and s.opacity > 0.15]

    def clear_scene(self) -> None:
        """Remove every layer actor (used before a reload)."""
        if self._highlight_actor is not None:
            self._renderer.RemoveActor(self._highlight_actor)
            self._highlight_actor = None
        for state in self._layers.values():
            for actor in state.actors:
                self._renderer.RemoveActor(actor)
        self._layers.clear()
        self._actor_to_layer.clear()
        self._actor_to_structure.clear()
        self._structure_index.clear()
        self._highlight_cache.clear()
        self._clip_plane = None

    # ------------------------------------------------------------ structures
    def structure_for_actor(self, actor) -> str:
        """Anatomical name of the structure an actor draws ("" if unnamed)."""
        return self._actor_to_structure.get(id(actor), "") if actor is not None else ""

    def structure_names(self, layer_id: Optional[str] = None) -> List[str]:
        """Named structures in one layer, or in the whole scene."""
        layers = [self._layers[layer_id]] if layer_id in self._layers else (
            [] if layer_id else list(self._layers.values()))
        return [name for state in layers for name in state.structures if name]

    def has_structure_names(self) -> bool:
        return bool(self._structure_index)

    def find_structures(self, query: str, layer_id: str = "") -> List[Tuple[str, str]]:
        """``(layer_id, name)`` for every structure whose name contains *query*
        (case-insensitive). All words of a multi-word query must match."""
        words = [w for w in (query or "").lower().split() if w]
        out = []
        for state in self._layers.values():
            if layer_id and state.id != layer_id:
                continue
            for name in state.structures:
                low = name.lower()
                if name and all(w in low for w in words):
                    out.append((state.id, name))
        return out

    def locate_structure(self, name: str) -> Optional[Tuple[str, int]]:
        """``(layer_id, actor index)`` for an exact (case-insensitive) name."""
        return self._structure_index.get((name or "").lower())

    def structure_actor(self, name: str):
        hit = self.locate_structure(name)
        return self._layers[hit[0]].actors[hit[1]] if hit else None

    def structure_polydata(self, name: str):
        hit = self.locate_structure(name)
        return self._layers[hit[0]].polydata[hit[1]] if hit else None

    def structure_bounds(self, names: Sequence[str]):
        """Union bounds of named structures, or ``None``."""
        bounds = [1e18, -1e18, 1e18, -1e18, 1e18, -1e18]
        found = False
        for name in names:
            poly = self.structure_polydata(name)
            if poly is None:
                continue
            found = True
            b = poly.GetBounds()
            for k in range(3):
                bounds[2 * k] = min(bounds[2 * k], b[2 * k])
                bounds[2 * k + 1] = max(bounds[2 * k + 1], b[2 * k + 1])
        return tuple(bounds) if found else None

    def highlight_structure(self, name: str) -> bool:
        """Neon edge overlay on a single named structure."""
        poly = self.structure_polydata(name)
        if poly is None:
            return False
        if self._highlight_actor is not None:
            self._renderer.RemoveActor(self._highlight_actor)
        key = f"structure::{name.lower()}"
        cached = self._highlight_cache.get(key)
        if cached is None:
            builder = vu.make_outline_actor if poly.GetNumberOfCells() > _HUGE_MESH_CELLS else vu.make_edge_actor
            cached = builder(poly, color=(0.15, 0.88, 1.0))
            self._highlight_cache[key] = cached
        self._highlight_actor = cached
        self._highlight_layer = self.locate_structure(name)[0]
        self._renderer.AddActor(cached)
        self._render()
        return True

    def isolate_structures(self, names: Sequence[str], ghost_opacity: float = _GHOST_OPACITY,
                           *, focus: bool = True) -> int:
        """Show only the named structures at full opacity; ghost everything else.
        Returns how many of *names* were found."""
        wanted = {n.lower() for n in names}
        hits = 0
        for state in self._layers.values():
            any_hit = False
            for actor, name in zip(state.actors, state.structures):
                hit = bool(name) and name.lower() in wanted
                any_hit |= hit
                hits += hit
                actor.SetVisibility(True)
                actor.GetProperty().SetOpacity(1.0 if hit else ghost_opacity)
            state.visible = True
            state.opacity = 1.0 if any_hit else ghost_opacity
        if focus and hits:
            self.focus_structures(names)
        self._render()
        return hits

    def focus_structures(self, names: Sequence[str]) -> None:
        bounds = self.structure_bounds(names)
        if bounds is None:
            return
        self._renderer.ResetCamera(bounds)
        self._renderer.ResetCameraClippingRange()
        self._render()

    # ------------------------------------------------------- layer control
    def set_layer_opacity(self, layer_id: str, opacity: float) -> None:
        """Fade a layer. Values ≥ 0.995 switch the actor back to opaque rendering."""
        state = self._layers.get(layer_id)
        if state is None:
            return
        opacity = max(0.0, min(1.0, float(opacity)))
        state.opacity = opacity
        for actor in state.actors:
            # Opacity alone drives translucency; see the note in vtk_utils.make_surface_actor
            # about why ForceOpaque/ForceTranslucent are deliberately not used.
            actor.GetProperty().SetOpacity(opacity)
            if state.is_placeholder:
                actor.SetVisibility(state.visible and opacity > 0.01)
        self._render()

    def set_layer_visible(self, layer_id: str, visible: bool) -> None:
        state = self._layers.get(layer_id)
        if state is None:
            return
        state.visible = bool(visible)
        for actor in state.actors:
            actor.SetVisibility(state.visible)
        self._render()

    def isolate_layer(self, layer_id: str, ghost_opacity: float = _GHOST_OPACITY) -> None:
        """Solo *layer_id*; everything else becomes a faint X-ray ghost."""
        if layer_id not in self._layers:
            return
        for lid, state in self._layers.items():
            if lid == layer_id:
                self.set_layer_opacity(lid, state.base_opacity)
                self.set_layer_visible(lid, True)
            else:
                self.set_layer_opacity(lid, ghost_opacity)
        self.focus_layer(layer_id)
        self.set_highlight(layer_id)

    def show_all_layers(self, restore_opacity: bool = True) -> None:
        for lid, state in self._layers.items():
            self.set_layer_visible(lid, True)
            self.set_layer_opacity(lid, state.base_opacity if restore_opacity else 1.0)

    def hide_all_layers(self, except_ids: Sequence[str] = ()) -> None:
        for lid in self._layers:
            if lid not in except_ids:
                self.set_layer_visible(lid, False)

    def set_layer_color(self, layer_id: str, color: Sequence[float]) -> None:
        state = self._layers.get(layer_id)
        if state is None:
            return
        state.color = tuple(color[:3])
        for actor in state.actors:
            actor.GetProperty().SetColor(*color[:3])
        self._highlight_cache.pop(layer_id, None)
        self._render()

    def apply_layer_preset(self, visible_ids: Sequence[str]) -> None:
        """Named 'system' presets, e.g. only skeletal + muscular."""
        wanted = set(visible_ids)
        for lid, state in self._layers.items():
            self.set_layer_visible(lid, lid in wanted)
            self.set_layer_opacity(lid, state.base_opacity if lid in wanted else state.opacity)

    # ------------------------------------------------------------ highlight
    def set_highlight(self, layer_id: Optional[str]) -> None:
        """Draw a neon edge overlay on one layer (selection / hover)."""
        if self._highlight_actor is not None:
            self._renderer.RemoveActor(self._highlight_actor)
            self._highlight_actor = None
        self._highlight_layer = layer_id
        if not layer_id:
            self._render()
            return

        cached = self._highlight_cache.get(layer_id)
        if cached is None:
            state = self._layers.get(layer_id)
            if state is None or not state.polydata:
                self._render()
                return
            poly = state.polydata[0] if len(state.polydata) == 1 else _append_all(state.polydata)
            cells = poly.GetNumberOfCells() if poly else 0
            builder = vu.make_outline_actor if cells > _HUGE_MESH_CELLS else vu.make_edge_actor
            cached = builder(poly, color=(0.15, 0.88, 1.0))
            self._highlight_cache[layer_id] = cached

        self._highlight_actor = cached
        self._renderer.AddActor(self._highlight_actor)
        self._render()

    # --------------------------------------------------------------- camera
    def reset_camera(self, *, animate: bool = False) -> None:
        self._renderer.ResetCamera()
        cam = self._renderer.GetActiveCamera()
        cam.Zoom(1.15)
        self._renderer.ResetCameraClippingRange()
        self._render()

    def focus_layer(self, layer_id: str) -> None:
        state = self._layers.get(layer_id)
        if state is None or not state.polydata:
            return
        bounds = [1e18, -1e18, 1e18, -1e18, 1e18, -1e18]
        for poly in state.polydata:
            b = poly.GetBounds()
            bounds[0] = min(bounds[0], b[0]); bounds[1] = max(bounds[1], b[1])
            bounds[2] = min(bounds[2], b[2]); bounds[3] = max(bounds[3], b[3])
            bounds[4] = min(bounds[4], b[4]); bounds[5] = max(bounds[5], b[5])
        if bounds[0] > bounds[1]:
            return
        self._renderer.ResetCamera(bounds)
        self._renderer.ResetCameraClippingRange()
        self._render()

    def set_view(self, preset: str) -> None:
        """Snap the camera to a standard anatomical projection."""
        direction, up = VIEW_PRESETS.get(preset, VIEW_PRESETS["isometric"])
        bounds = self._renderer.ComputeVisiblePropBounds()
        if bounds[1] < bounds[0]:
            return
        center = ((bounds[0] + bounds[1]) / 2, (bounds[2] + bounds[3]) / 2,
                  (bounds[4] + bounds[5]) / 2)
        diagonal = math.dist((bounds[0], bounds[2], bounds[4]),
                             (bounds[1], bounds[3], bounds[5])) or 1.0
        distance = diagonal * 1.75

        norm = math.sqrt(sum(c * c for c in direction)) or 1.0
        cam = self._renderer.GetActiveCamera()
        cam.SetFocalPoint(*center)
        cam.SetPosition(*(center[i] + direction[i] / norm * distance for i in range(3)))
        cam.SetViewUp(*up)
        cam.OrthogonalizeViewUp()
        self._renderer.ResetCameraClippingRange()
        self._render()
        self.cameraChanged.emit()

    def rotate(self, dx_degrees: float, dy_degrees: float) -> None:
        """Orbit the camera. ``dx`` spins around the superior axis, ``dy`` tilts."""
        cam = self._renderer.GetActiveCamera()
        cam.Azimuth(-float(dx_degrees))
        cam.Elevation(float(dy_degrees))
        cam.OrthogonalizeViewUp()
        self._renderer.ResetCameraClippingRange()
        self._render()
        self.cameraChanged.emit()

    def zoom(self, factor: float) -> None:
        """*factor* > 1 moves closer, < 1 moves away."""
        cam = self._renderer.GetActiveCamera()
        cam.Zoom(max(1.005, min(20.0, float(factor))))
        self._renderer.ResetCameraClippingRange()
        self._render()
        self.cameraChanged.emit()

    def pan(self, dx_pixels: float, dy_pixels: float) -> None:
        """Translate the camera in the view plane, preserving scale."""
        width, height = self._render_window.GetSize()
        if width <= 0 or height <= 0:
            return
        cam = self._renderer.GetActiveCamera()
        focal = cam.GetFocalPoint()
        position = cam.GetPosition()
        distance = math.dist(focal, position) or 1.0
        world_height = 2.0 * distance * math.tan(math.radians(cam.GetViewAngle()) / 2.0)
        world_width = world_height * (width / height)

        forward = _normalize([focal[i] - position[i] for i in range(3)])
        right = _normalize(_cross(forward, cam.GetViewUp()))
        up = _cross(right, forward)

        tx = -(dx_pixels / width) * world_width
        ty = -(dy_pixels / height) * world_height
        offset = [right[i] * tx + up[i] * ty for i in range(3)]

        cam.SetPosition(*[position[i] + offset[i] for i in range(3)])
        cam.SetFocalPoint(*[focal[i] + offset[i] for i in range(3)])
        self._renderer.ResetCameraClippingRange()
        self._render()
        self.cameraChanged.emit()

    def dolly(self, factor: float) -> None:
        """Physical camera move (keeps perspective consistent)."""
        cam = self._renderer.GetActiveCamera()
        focal = cam.GetFocalPoint()
        position = cam.GetPosition()
        vector = [focal[i] - position[i] for i in range(3)]
        scale = max(0.05, min(0.95, 1.0 - 1.0 / max(factor, 1e-3)))
        cam.SetPosition(*[position[i] + vector[i] * scale for i in range(3)])
        self._renderer.ResetCameraClippingRange()
        self._render()

    # -------------------------------------------------------- camera states
    def camera_state(self) -> Dict[str, List[float]]:
        cam = self._renderer.GetActiveCamera()
        return {
            "position": list(cam.GetPosition()),
            "focal": list(cam.GetFocalPoint()),
            "up": list(cam.GetViewUp()),
            "angle": float(cam.GetViewAngle()),
        }

    def apply_camera_state(self, state: Dict[str, Sequence[float]]) -> None:
        cam = self._renderer.GetActiveCamera()
        if "position" in state:
            cam.SetPosition(*state["position"])
        if "focal" in state:
            cam.SetFocalPoint(*state["focal"])
        if "up" in state:
            cam.SetViewUp(*state["up"])
        if "angle" in state:
            cam.SetViewAngle(float(state["angle"]))
        cam.OrthogonalizeViewUp()
        self._renderer.ResetCameraClippingRange()
        self._render()

    # ------------------------------------------------------------ clip/explode
    def apply_clip_plane(self, axis: str, normalized: float) -> None:
        """Slice the scene with one shared plane. ``normalized`` ∈ [0, 1]."""
        bounds = self._renderer.ComputeVisiblePropBounds()
        if bounds[1] < bounds[0]:
            return
        index = {"x": 0, "y": 1, "z": 2}.get(axis, 0)
        lo, hi = bounds[index * 2], bounds[index * 2 + 1]
        lo, hi = min(lo, hi), max(lo, hi)
        position = lo + (hi - lo) * max(0.0, min(1.0, normalized))

        normal = [0.0, 0.0, 0.0]
        normal[index] = 1.0
        origin = [0.0, 0.0, 0.0]
        origin[index] = position

        if self._clip_plane is None:
            self._clip_plane = vtk.vtkPlane()
            for state in self._layers.values():
                for actor in state.actors:
                    actor.GetMapper().AddClippingPlane(self._clip_plane)
        self._clip_plane.SetNormal(*normal)
        self._clip_plane.SetOrigin(*origin)
        self._render()

    def disable_clip_plane(self) -> None:
        if self._clip_plane is None:
            return
        for state in self._layers.values():
            for actor in state.actors:
                mapper = actor.GetMapper()
                mapper.RemoveAllClippingPlanes()
        self._clip_plane = None
        self._render()

    def set_explode(self, factor: float) -> None:
        """Fan layers apart along their registry-defined offset vectors."""
        self._explode = max(0.0, min(1.0, float(factor)))
        for state in self._layers.values():
            offset = [state.explode_vector[i] * self._explode * 1.4 for i in range(3)]
            for actor in state.actors:
                actor.SetPosition(*offset)
        self._render()

    # ------------------------------------------------------------ appearance
    def set_background(self, preset: str) -> None:
        self._configure_background(preset)
        self._render()

    def set_axes_visible(self, visible: bool) -> None:
        try:
            self._marker.SetEnabled(1 if visible else 0)
        except Exception:
            pass
        self._render()

    def snapshot(self, path: Path, magnification: int = 2) -> Optional[Path]:
        """Save the current frame at *magnification*× for reports and slides."""
        try:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            w2i = vtk.vtkWindowToImageFilter()
            w2i.SetInput(self._render_window)
            w2i.SetScale(magnification)
            w2i.SetInputBufferTypeToRGB()
            w2i.ReadFrontBufferOff()
            w2i.Update()

            writer = vtk.vtkPNGWriter()
            writer.SetFileName(str(path))
            writer.SetInputConnection(w2i.GetOutputPort())
            writer.Write()
            return path
        except Exception:
            return None

    # ------------------------------------------------------------- internal
    def _render(self) -> None:
        self._render_window.Render()

    def _sample_fps(self) -> None:
        try:
            dt = self._renderer.GetLastRenderTimeInSeconds()
            if dt > 0:
                self.fpsUpdated.emit(1.0 / dt)
        except Exception:
            pass

    # -- called by the interactor style ------------------------------------
    def _on_hover_changed(self, layer_id: Optional[str], actor=None) -> None:
        """Hover feedback: hand-shaped cursor over a structure, arrow over empty space."""
        from PyQt6.QtCore import Qt
        self.setCursor(Qt.CursorShape.PointingHandCursor if layer_id
                       else Qt.CursorShape.ArrowCursor)
        name = self.structure_for_actor(actor)
        self.hoverChanged.emit(name or (self._registry.label(layer_id) if layer_id else ""))

    def _on_click_picked(self, actor) -> None:
        layer_id = self.layer_id_for_actor(actor)
        self._last_pick_id = layer_id
        if layer_id:
            name = self.structure_for_actor(actor)
            if not (name and self.highlight_structure(name)):
                self.set_highlight(layer_id)
            self.structurePicked.emit(layer_id, name or self._registry.label(layer_id))
        else:
            self.set_highlight(None)
            self.structurePicked.emit("", "")

    def _on_camera_moved(self, _style) -> None:
        self.cameraChanged.emit()


# ---------------------------------------------------------------------------
# Fallback widget — keeps the shell usable when VTK is missing
# ---------------------------------------------------------------------------
class PlaceholderViewport(QWidget):
    """Shown instead of :class:`Interactive3DViewport` when VTK cannot be imported."""

    structurePicked = pyqtSignal(str, str)
    hoverChanged = pyqtSignal(str)
    cameraChanged = pyqtSignal()
    fpsUpdated = pyqtSignal(float)
    sceneLoaded = pyqtSignal(int, int)

    def __init__(self, reason: str = "", parent: Optional[QWidget] = None):
        super().__init__(parent)
        from PyQt6.QtCore import Qt

        self.setObjectName("ViewportPlaceholder")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(48, 48, 48, 48)

        title = QLabel("3D viewport unavailable")
        title.setObjectName("PlaceholderTitle")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)

        detail = QLabel(
            f"{reason}\n\nInstall the rendering stack with:\n"
            "    pip install vtk PyQt6\n\nthen restart BioHuman3D."
        )
        detail.setObjectName("PlaceholderBody")
        detail.setAlignment(Qt.AlignmentFlag.AlignCenter)
        detail.setWordWrap(True)

        layout.addStretch(1)
        layout.addWidget(title)
        layout.addWidget(detail)
        layout.addStretch(1)

    # -- no-op API so SceneController can bind unconditionally -------------
    def load_scene(self, **_kwargs): return (0, 0)
    def set_layer_opacity(self, *_a, **_k): pass
    def set_layer_visible(self, *_a, **_k): pass
    def isolate_layer(self, *_a, **_k): pass
    def show_all_layers(self, *_a, **_k): pass
    def hide_all_layers(self, *_a, **_k): pass
    def set_highlight(self, *_a, **_k): pass
    def reset_camera(self, *_a, **_k): pass
    def focus_layer(self, *_a, **_k): pass
    def set_view(self, *_a, **_k): pass
    def rotate(self, *_a, **_k): pass
    def zoom(self, *_a, **_k): pass
    def pan(self, *_a, **_k): pass
    def apply_clip_plane(self, *_a, **_k): pass
    def disable_clip_plane(self, *_a, **_k): pass
    def set_explode(self, *_a, **_k): pass
    def set_background(self, *_a, **_k): pass
    def camera_state(self): return {}
    def apply_camera_state(self, *_a, **_k): pass
    def snapshot(self, *_a, **_k): return None
    def visible_layers(self): return []
    def clear_scene(self): pass
    def structure_for_actor(self, *_a): return ""
    def structure_names(self, *_a, **_k): return []
    def has_structure_names(self): return False
    def find_structures(self, *_a, **_k): return []
    def locate_structure(self, *_a): return None
    def structure_actor(self, *_a): return None
    def structure_polydata(self, *_a): return None
    def structure_bounds(self, *_a): return None
    def highlight_structure(self, *_a): return False
    def isolate_structures(self, *_a, **_k): return 0
    def focus_structures(self, *_a, **_k): pass
    layers: Dict[str, LayerState] = {}


#: Platforms that cannot provide a native OpenGL drawable.
_HEADLESS_PLATFORMS = {"offscreen", "minimal"}


def create_viewport(registry: LayerRegistry, config, parent=None):
    """Factory: real VTK viewport when possible, informative placeholder otherwise.

    Guards against a crash mode that cannot be caught: when Qt runs on a
    platform without a native window (``offscreen`` / ``minimal``), VTK's Win32
    render window fails to acquire a pixel format and the process dies with an
    access violation rather than raising. Detecting the platform up front keeps
    CI, headless tests and remote sessions alive.
    """
    if not VTK_AVAILABLE:
        return PlaceholderViewport(vu.VTK_IMPORT_ERROR, parent)

    try:
        from PyQt6.QtGui import QGuiApplication
        platform = QGuiApplication.platformName()
    except Exception:
        platform = ""

    if platform in _HEADLESS_PLATFORMS:
        return PlaceholderViewport(
            f"Qt platform '{platform}' has no OpenGL drawable, so the VTK render "
            f"window is disabled. Run the app on a normal desktop session to use "
            f"the 3D viewport.",
            parent,
        )

    try:
        return Interactive3DViewport(registry, config, parent)
    except Exception as exc:  # pragma: no cover
        return PlaceholderViewport(f"Failed to initialise VTK: {exc}", parent)


# ---------------------------------------------------------------------------
# Tiny vector math (avoids pulling numpy into the viewport hot path)
# ---------------------------------------------------------------------------
def _append_all(polys: Sequence[object]):
    append = vtk.vtkAppendPolyData()
    for poly in polys:
        append.AddInputData(poly)
    append.Update()
    return append.GetOutput()


def _cross(a: Sequence[float], b: Sequence[float]) -> List[float]:
    return [a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0]]


def _normalize(v: Sequence[float]) -> List[float]:
    n = math.sqrt(sum(c * c for c in v)) or 1.0
    return [c / n for c in v]
