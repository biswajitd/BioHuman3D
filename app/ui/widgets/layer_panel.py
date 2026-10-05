"""
Left panel — **Anatomy Layers & Tools**.

Contains three collapsible cards:

1. *Anatomy Layers* — one :class:`LayerRow` per anatomical system, grouped, with
   an eye toggle (hide/show), an isolate button and a per-layer opacity slider.
2. *Display & Slice* — background preset, cross-section plane, explode view.
3. *Guided Tours* — narrated camera tours with transport controls.

The panel owns no state: it emits intent, and :class:`SceneController` decides
what actually happens. That keeps the widget reusable and testable.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (QComboBox, QFrame, QHBoxLayout, QLabel, QPushButton,
                             QScrollArea, QSizePolicy, QSlider, QVBoxLayout,
                             QWidget)

from app.core.model_registry import LayerSpec
from app.audio.scripts import system_presets
from app.ui.widgets.collapsible import CollapsibleSection
from app.ui.widgets.controls import (GLYPH, ChipButton, IconButton, LabeledSlider,
                                     SearchField, SegmentBar, panel_header)


# ---------------------------------------------------------------------------
# One layer row
# ---------------------------------------------------------------------------
class LayerRow(QFrame):
    """Compact row: colour swatch · name · eye · isolate · opacity slider."""

    opacityChanged = pyqtSignal(str, float)     # layer_id, 0..1
    visibilityToggled = pyqtSignal(str, bool)
    isolateRequested = pyqtSignal(str)
    picked = pyqtSignal(str)

    def __init__(self, spec: LayerSpec, *, available: bool = True,
                 parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.spec = spec
        self._layer_id = spec.id
        self.setObjectName("LayerRow")
        self.setProperty("selected", "false")
        self.setProperty("dimmed", "false")

        root = QVBoxLayout(self)
        root.setContentsMargins(7, 6, 7, 6)
        root.setSpacing(4)

        # -- top row --------------------------------------------------------
        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        top.setSpacing(8)

        r, g, b = spec.color
        swatch = QLabel()
        swatch.setObjectName("LayerSwatch")
        swatch.setStyleSheet(
            f"background-color: rgb({int(r*255)},{int(g*255)},{int(b*255)});"
            "border-radius: 3px;"
        )
        swatch.setToolTip(f"Layer colour: {spec.group}")

        name_box = QVBoxLayout()
        name_box.setContentsMargins(0, 0, 0, 0)
        name_box.setSpacing(0)
        name = QLabel(spec.label)
        name.setObjectName("LayerName")
        meta = QLabel(f"{spec.group}" + ("" if available else " · no model"))
        meta.setObjectName("LayerMeta")
        name_box.addWidget(name)
        name_box.addWidget(meta)

        self._eye = IconButton(GLYPH["eye"], "Hide layer", checkable=True, role="eye")
        self._eye.setChecked(spec.default_visible)
        self._eye.toggled.connect(self._on_eye_toggled)

        self._solo = IconButton(GLYPH["solo"], "Isolate this layer (ghost the rest)")
        self._solo.clicked.connect(lambda: self.isolateRequested.emit(self._layer_id))

        top.addWidget(swatch)
        top.addLayout(name_box, 1)
        top.addWidget(self._eye)
        top.addWidget(self._solo)

        # -- opacity --------------------------------------------------------
        opacity_row = QHBoxLayout()
        opacity_row.setContentsMargins(18, 0, 2, 0)
        opacity_row.setSpacing(6)
        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(0, 100)
        self._slider.setValue(int(spec.default_opacity * 100))
        self._slider.setProperty("compact", "true")
        self._slider.setToolTip("Opacity — fade this layer without hiding it")
        self._slider.valueChanged.connect(self._on_slider)
        self._pct = QLabel(f"{int(spec.default_opacity * 100)}%")
        self._pct.setObjectName("LayerMeta")
        self._pct.setFixedWidth(30)
        self._pct.setAlignment(Qt.AlignmentFlag.AlignRight)

        opacity_row.addWidget(self._slider, 1)
        opacity_row.addWidget(self._pct)

        root.addLayout(top)
        root.addLayout(opacity_row)

        if not available:
            self._slider.setEnabled(False)
            self._eye.setEnabled(False)
            self._solo.setEnabled(False)
            self.setProperty("dimmed", "true")

    # -- API ---------------------------------------------------------------
    @property
    def layer_id(self) -> str:
        return self._layer_id

    def set_opacity(self, value: float, *, emit: bool = False) -> None:
        """Update the UI from outside without bouncing a signal back."""
        ivalue = int(max(0.0, min(1.0, value)) * 100)
        blocked = self._slider.blockSignals(True)
        self._slider.setValue(ivalue)
        self._slider.blockSignals(blocked)
        self._pct.setText(f"{ivalue}%")
        if emit:
            self.opacityChanged.emit(self._layer_id, ivalue / 100.0)

    def opacity(self) -> float:
        return self._slider.value() / 100.0

    def set_visible(self, visible: bool, *, emit: bool = False) -> None:
        blocked = self._eye.blockSignals(True)
        self._eye.setChecked(visible)
        self._eye.blockSignals(blocked)
        self._eye.setText(GLYPH["eye"] if visible else GLYPH["eye_off"])
        if emit:
            self.visibilityToggled.emit(self._layer_id, visible)

    def set_selected(self, selected: bool) -> None:
        self.setProperty("selected", "true" if selected else "false")
        from app.ui.theme import repolish
        repolish(self)

    def matches(self, needle: str) -> bool:
        if not needle:
            return True
        haystack = f"{self.spec.label} {self.spec.group} {self.spec.id}".lower()
        return needle.lower() in haystack

    # -- internals ---------------------------------------------------------
    def _on_eye_toggled(self, checked: bool) -> None:
        self._eye.setText(GLYPH["eye"] if checked else GLYPH["eye_off"])
        self.visibilityToggled.emit(self._layer_id, checked)

    def _on_slider(self, value: int) -> None:
        self._pct.setText(f"{value}%")
        self.opacityChanged.emit(self._layer_id, value / 100.0)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        self.picked.emit(self._layer_id)
        super().mousePressEvent(event)


# ---------------------------------------------------------------------------
# The panel
# ---------------------------------------------------------------------------
class LayerPanel(QWidget):
    """Anatomy layer list + viewport tool drawer."""

    layerOpacityChanged = pyqtSignal(str, float)
    layerVisibilityChanged = pyqtSignal(str, bool)
    isolateRequested = pyqtSignal(str)
    layerSelected = pyqtSignal(str, str)         # id, label
    presetRequested = pyqtSignal(str)
    showAllRequested = pyqtSignal()
    hideAllRequested = pyqtSignal()

    clipChanged = pyqtSignal(str, float)         # axis, 0..1
    clipCleared = pyqtSignal()
    explodeChanged = pyqtSignal(float)
    backgroundChanged = pyqtSignal(str)

    tourRequested = pyqtSignal(str)
    tourStopRequested = pyqtSignal()
    tourStepRequested = pyqtSignal(int)          # +1 next, -1 previous
    tourNarrateRequested = pyqtSignal()

    def __init__(self, tours: Sequence = (), parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("SidePanel")
        self.setMinimumWidth(280)
        self.setMaximumWidth(420)

        self._rows: Dict[str, LayerRow] = {}
        self._group_headers: Dict[str, QLabel] = {}

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        root.addWidget(panel_header(
            "Anatomy Layers & Tools",
            "Click a structure in the 3D view to inspect it",
        ))

        # -- search + presets ----------------------------------------------
        search_wrap = QWidget()
        search_layout = QVBoxLayout(search_wrap)
        search_layout.setContentsMargins(12, 0, 12, 8)
        search_layout.setSpacing(7)

        self._search = SearchField("Filter layers…")
        self._search.textChanged.connect(self._apply_filter)
        search_layout.addWidget(self._search)

        preset_row = QHBoxLayout()
        preset_row.setSpacing(6)
        # Built from the registry, never hand-listed: a hard-coded list here
        # previously hid half the body systems (no Urinary, Arterial, Venous,
        # Lymphatic, Integumentary or Articular) and so made their tours
        # unreachable from the export panel.
        self._preset = QComboBox()
        self._preset.setToolTip(
            "Choose a body system. This also sets the video export subject so "
            "the narration matches what you are looking at.")
        for name in system_presets():
            self._preset.addItem(name, "" if name == "All systems" else name)
        self._preset.currentIndexChanged.connect(self._on_preset)
        preset_row.addWidget(self._preset, 1)

        show_all = IconButton(GLYPH["refresh"], "Show every layer at default opacity")
        show_all.clicked.connect(self.showAllRequested)
        hide_all = IconButton(GLYPH["sound_off"], "Hide every layer")
        hide_all.clicked.connect(self.hideAllRequested)
        preset_row.addWidget(show_all)
        preset_row.addWidget(hide_all)
        search_layout.addLayout(preset_row)
        root.addWidget(search_wrap)

        # -- scrollable card stack ------------------------------------------
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        holder = QWidget()
        holder_layout = QVBoxLayout(holder)
        holder_layout.setContentsMargins(10, 0, 10, 10)
        holder_layout.setSpacing(9)

        self.layers_section = CollapsibleSection("Anatomy Layers", expanded=True)
        self._layer_list_host = QWidget()
        self._layer_list_layout = QVBoxLayout(self._layer_list_host)
        self._layer_list_layout.setContentsMargins(0, 0, 0, 0)
        self._layer_list_layout.setSpacing(3)
        self.layers_section.add_widget(self._layer_list_host)

        holder_layout.addWidget(self.layers_section)
        holder_layout.addWidget(self._build_display_section())
        holder_layout.addWidget(self._build_tours_section(tours))
        holder_layout.addStretch(1)

        scroll.setWidget(holder)
        root.addWidget(scroll, 1)

    # ------------------------------------------------------------ build
    def _build_display_section(self) -> CollapsibleSection:
        section = CollapsibleSection("Display & Slice", expanded=True)

        background_row = QHBoxLayout()
        background_row.setSpacing(6)
        background_label = QLabel("Background")
        background_label.setObjectName("LayerMeta")
        self._background = QComboBox()
        for label, key in (("Studio", "studio"), ("Slate", "slate"),
                           ("Clinical", "clinical"), ("Black", "black")):
            self._background.addItem(label, key)
        self._background.currentIndexChanged.connect(
            lambda: self.backgroundChanged.emit(self._background.currentData()))
        background_row.addWidget(background_label)
        background_row.addWidget(self._background, 1)
        section.add_layout(background_row)

        axis_row = QHBoxLayout()
        axis_row.setSpacing(6)
        axis_label = QLabel("Slice axis")
        axis_label.setObjectName("LayerMeta")
        self._clip_axis = SegmentBar([("x", "X"), ("y", "Y"), ("z", "Z")], current="y")
        self._clip_axis.setSizePolicy(QSizePolicy.Policy.Maximum,
                                      QSizePolicy.Policy.Fixed)
        axis_row.addWidget(axis_label)
        axis_row.addStretch(1)
        axis_row.addWidget(self._clip_axis)
        section.add_layout(axis_row)

        self._clip_slider = LabeledSlider("Cross-section", value=0, formatter=self._fmt_clip)
        self._clip_slider.slider.setToolTip("0% = no slice · drag to cut through the body")
        self._clip_slider.valueChanged.connect(self._on_clip)
        section.add_widget(self._clip_slider)

        clear_clip = ChipButton("Clear slice", "Remove the clipping plane")
        clear_clip.clicked.connect(self._on_clear_clip)
        section.add_widget(clear_clip)

        self._explode_slider = LabeledSlider("Explode view", value=0,
                                             formatter=lambda v: f"{v}%")
        self._explode_slider.slider.setToolTip("Fan the anatomical layers apart")
        self._explode_slider.valueChanged.connect(
            lambda v: self.explodeChanged.emit(v / 100.0))
        section.add_widget(self._explode_slider)

        return section

    def _build_tours_section(self, tours: Sequence) -> CollapsibleSection:
        section = CollapsibleSection("Guided Tours", expanded=False)

        self._tour_combo = QComboBox()
        for tour in tours:
            self._tour_combo.addItem(tour.title, getattr(tour, "id", ""))
        section.add_widget(self._tour_combo)

        transport = QHBoxLayout()
        transport.setSpacing(6)
        self._btn_prev = IconButton(GLYPH["prev"], "Previous keyframe")
        self._btn_prev.clicked.connect(lambda: self.tourStepRequested.emit(-1))
        self._btn_play = QPushButton(f"{GLYPH['play']}  Play tour")
        self._btn_play.setObjectName("PrimaryButton")
        self._btn_play.clicked.connect(self._on_play_clicked)
        self._btn_next = IconButton(GLYPH["next"], "Next keyframe")
        self._btn_next.clicked.connect(lambda: self.tourStepRequested.emit(1))
        self._btn_stop = IconButton(GLYPH["stop"], "Stop tour")
        self._btn_stop.clicked.connect(self.tourStopRequested)

        transport.addWidget(self._btn_prev)
        transport.addWidget(self._btn_play, 1)
        transport.addWidget(self._btn_next)
        transport.addWidget(self._btn_stop)
        section.add_layout(transport)

        self._narrate = ChipButton(f"{GLYPH['speaker']}  Narrate current selection",
                                   "Speak the description of the selected structure")
        self._narrate.clicked.connect(self.tourNarrateRequested)
        section.add_widget(self._narrate)

        self._cue_label = QLabel("")
        self._cue_label.setObjectName("LayerMeta")
        self._cue_label.setWordWrap(True)
        section.add_widget(self._cue_label)

        self._tours_section = section
        self._is_playing = False
        return section

    # --------------------------------------------------------- population
    def populate(self, specs: Iterable[LayerSpec], available_ids: Sequence[str] = ()) -> None:
        """Build one row per spec, bucketed under its system group.

        Idempotent: calling it again (e.g. after the scene finishes loading)
        rebuilds the list instead of appending a duplicate set of rows.
        """
        self._clear_rows()
        available = set(available_ids)
        groups: Dict[str, List[LayerSpec]] = {}
        order: List[str] = []
        for spec in specs:
            if spec.group not in groups:
                groups[spec.group] = []
                order.append(spec.group)
            groups[spec.group].append(spec)

        for group in order:
            header = QLabel(group.upper())
            header.setObjectName("GroupHeader")
            self._group_headers[group] = header
            self._layer_list_layout.addWidget(header)

            for spec in groups[group]:
                row = LayerRow(spec, available=(not available or spec.id in available))
                row.opacityChanged.connect(self.layerOpacityChanged)
                row.visibilityToggled.connect(self.layerVisibilityChanged)
                row.isolateRequested.connect(self.isolateRequested)
                row.picked.connect(lambda lid, s=spec: self.layerSelected.emit(lid, s.label))
                self._rows[spec.id] = row
                row.setProperty("group", group)
                self._layer_list_layout.addWidget(row)

        self.layers_section.set_badge(f"{len(self._rows)} layers")

    def _clear_rows(self) -> None:
        """Tear down previously built rows before a rebuild."""
        while self._layer_list_layout.count():
            item = self._layer_list_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()
        self._rows.clear()
        self._group_headers.clear()

    # ---------------------------------------------------------- external
    def set_layer_opacity(self, layer_id: str, opacity: float) -> None:
        row = self._rows.get(layer_id)
        if row is not None:
            row.set_opacity(opacity)

    def set_layer_visible(self, layer_id: str, visible: bool) -> None:
        row = self._rows.get(layer_id)
        if row is not None:
            row.set_visible(visible)

    def select_layer(self, layer_id: str) -> None:
        for lid, row in self._rows.items():
            row.set_selected(lid == layer_id)

    def set_tour_progress(self, index: int, total: int, title: str) -> None:
        self._cue_label.setText(f"Step {index + 1}/{total} — {title}" if title else "")

    def set_playing(self, playing: bool) -> None:
        self._is_playing = playing
        self._btn_play.setText(f"{GLYPH['pause'] if playing else GLYPH['play']}  "
                               + ("Pause tour" if playing else "Play tour"))

    def current_tour_id(self) -> str:
        return self._tour_combo.currentData() or ""

    # ------------------------------------------------------------ internal
    def _on_play_clicked(self) -> None:
        if self._is_playing:
            self.tourStopRequested.emit()
        else:
            self.tourRequested.emit(self.current_tour_id())

    def _on_preset(self) -> None:
        """Emit the chosen body system.

        The selection is deliberately NOT reset to index 0 any more: keeping it
        visible tells the user which system is active, which is also the video
        export subject.
        """
        preset = self._preset.currentData()
        if preset:
            self.presetRequested.emit(preset)

    def _on_clip(self, _value: int) -> None:
        value = self._clip_slider.value()
        if value <= 0:
            self.clipCleared.emit()
        else:
            self.clipChanged.emit(self._clip_axis.current, value / 100.0)

    def _on_clear_clip(self) -> None:
        self._clip_slider.set_value(0)
        self.clipCleared.emit()

    @staticmethod
    def _fmt_clip(value: int) -> str:
        return "off" if value <= 0 else f"{value}%"

    def _apply_filter(self, needle: str) -> None:
        """Show only matching rows, and hide group headers with no matches."""
        needle = (needle or "").strip()
        visible_groups: Dict[str, int] = {}
        for layer_id, row in self._rows.items():
            matched = row.matches(needle)
            row.setVisible(matched)
            group = row.property("group") or ""
            visible_groups[group] = visible_groups.get(group, 0) + (1 if matched else 0)
        for group, header in self._group_headers.items():
            header.setVisible(visible_groups.get(group, 0) > 0)
