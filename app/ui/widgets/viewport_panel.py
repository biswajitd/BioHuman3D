"""
Centre panel — **3D Viewport**.

The viewport widget itself fills the panel; the controls float above it in
translucent "glass" bars using a ``QGridLayout`` where every overlay occupies the
same cell as the render surface. Overlays are real child widgets, so their
buttons receive clicks normally while the background stays see-through.
"""
from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (QFrame, QGridLayout, QHBoxLayout, QLabel, QPushButton,
                             QSizePolicy, QVBoxLayout, QWidget)

from app.ui.widgets.controls import (GLYPH, IconButton, SegmentBar, StatusChip)


class ViewportPanel(QWidget):
    """3D surface + floating toolbar + status readouts."""

    viewRequested = pyqtSignal(str)          # anterior | posterior | left | right | superior | isometric
    resetRequested = pyqtSignal()
    snapshotRequested = pyqtSignal()
    tourToggleRequested = pyqtSignal()
    tourStopRequested = pyqtSignal()
    tourStepRequested = pyqtSignal(int)
    narrateRequested = pyqtSignal()

    def __init__(self, viewport: QWidget, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("CenterPanel")

        grid = QGridLayout(self)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(0)

        self.viewport = viewport
        self.viewport.setSizePolicy(QSizePolicy.Policy.Expanding,
                                    QSizePolicy.Policy.Expanding)
        grid.addWidget(self.viewport, 0, 0)

        # ---------------------------------------------------------- overlays
        # Placed in the same grid cell as the render surface and positioned by
        # alignment, so they float over the 3D scene. Each one sits in a thin
        # margin wrapper so it never touches the panel edge, and no two overlays
        # share an anchor (that is what caused them to collide).
        grid.addWidget(self._wrap(self._build_top_bar(), (12, 12, 12, 0)), 0, 0,
                       Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter)
        grid.addWidget(self._wrap(self._build_tour_bar(), (0, 0, 12, 12)), 0, 0,
                       Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignRight)
        grid.addWidget(self._wrap(self._build_bottom_bar(), (12, 0, 0, 12)), 0, 0,
                       Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignLeft)

        self._overlays = [w for w in self.findChildren(QFrame) if w.objectName() == "OverlayBar"]
        self._apply_overlay_margins()

    # ------------------------------------------------------------ build
    @staticmethod
    def _wrap(inner: QWidget, margins: tuple) -> QWidget:
        """Put *inner* in a transparent wrapper that keeps it off the panel edge."""
        holder = QWidget()
        layout = QVBoxLayout(holder)
        layout.setContentsMargins(*margins)
        layout.setSpacing(0)
        layout.addWidget(inner)
        holder.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Maximum)
        return holder

    def _overlay_frame(self) -> QFrame:
        frame = QFrame()
        frame.setObjectName("OverlayBar")
        frame.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        return frame

    def _build_top_bar(self) -> QFrame:
        bar = self._overlay_frame()
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(8, 7, 8, 7)
        layout.setSpacing(8)

        self._view_segments = SegmentBar(
            [("anterior", "Anterior"), ("posterior", "Posterior"), ("left", "Left"),
             ("right", "Right"), ("superior", "Superior"), ("isometric", "3D")],
            current="isometric",
        )
        self._view_segments.changed.connect(self.viewRequested)

        reset = IconButton(GLYPH["reset"], "Reset camera  (R)")
        reset.clicked.connect(self.resetRequested)
        snapshot = IconButton(GLYPH["camera"], "Save high-resolution PNG  (Ctrl+S)")
        snapshot.clicked.connect(self.snapshotRequested)
        narrate = IconButton(GLYPH["speaker"], "Narrate the selected structure")
        narrate.clicked.connect(self.narrateRequested)

        layout.addWidget(self._view_segments)
        layout.addWidget(self._separator())
        layout.addWidget(reset)
        layout.addWidget(snapshot)
        layout.addWidget(narrate)
        return bar

    def _build_tour_bar(self) -> QFrame:
        bar = self._overlay_frame()
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(6)

        self._btn_play = IconButton(GLYPH["play"], "Play / pause the guided tour")
        self._btn_play.clicked.connect(self.tourToggleRequested)
        self._btn_prev = IconButton(GLYPH["prev"], "Previous step")
        self._btn_prev.clicked.connect(lambda: self.tourStepRequested.emit(-1))
        self._btn_next = IconButton(GLYPH["next"], "Next step")
        self._btn_next.clicked.connect(lambda: self.tourStepRequested.emit(1))
        self._btn_stop = IconButton(GLYPH["stop"], "End the tour")
        self._btn_stop.clicked.connect(self.tourStopRequested)

        self._tour_title = QLabel("Guided tours")
        self._tour_title.setObjectName("PanelHint")
        self._tour_title.setMaximumWidth(150)

        layout.addWidget(self._btn_play)
        layout.addWidget(self._btn_prev)
        layout.addWidget(self._btn_next)
        layout.addWidget(self._btn_stop)
        layout.addWidget(self._separator())
        layout.addWidget(self._tour_title)
        bar.setVisible(True)
        return bar

    def _build_bottom_bar(self) -> QFrame:
        bar = self._overlay_frame()
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(10, 6, 10, 6)
        layout.setSpacing(10)

        self._selection_chip = StatusChip("No structure selected", accent=False)
        # Kept short: the transport bar shares the bottom edge on the right.
        self._hint_chip = StatusChip("Drag to orbit · scroll to zoom", accent=False)
        self._fps_chip = StatusChip("— fps", accent=False)

        layout.addWidget(self._selection_chip)
        layout.addWidget(self._hint_chip)
        layout.addWidget(self._fps_chip)
        return bar

    @staticmethod
    def _separator() -> QFrame:
        line = QFrame()
        line.setObjectName("VerticalDivider")
        line.setFixedSize(1, 18)
        return line

    def _apply_overlay_margins(self) -> None:
        for overlay in self._overlays:
            overlay.setContentsMargins(0, 0, 0, 0)

    # ------------------------------------------------------------ public API
    def set_selection(self, label: str) -> None:
        text = label if label else "No structure selected"
        self._selection_chip.setText(text)
        self._selection_chip.setProperty("accent", "true" if label else "false")
        from app.ui.theme import repolish
        repolish(self._selection_chip)

    def set_fps(self, fps: float) -> None:
        self._fps_chip.setText(f"{fps:.0f} fps")

    def set_hover(self, label: str) -> None:
        if label:
            self._hint_chip.setText(f"\u2192 {label}")

    def set_view_preset(self, preset: str) -> None:
        self._view_segments.set_current(preset)

    def set_tour_state(self, playing: bool, title: str = "") -> None:
        self._btn_play.setText(GLYPH["pause"] if playing else GLYPH["play"])
        self._tour_title.setText(title or "Guided tours")

    def set_status_hint(self, text: str) -> None:
        self._hint_chip.setText(text)
