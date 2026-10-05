"""
Paired MRI pane: the imaging slice that matches the 3D cross-section.

Sits beside the 3D view. Moving the cross-section in 3D moves the MRI slice to
the same anatomical plane, and scrolling the MRI moves the 3D cut. Hovering
the image names the structure under the cursor (from the label volume of a
simulated study, or — for a real scan — the model structure at that point).
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np
from PyQt6.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QImage, QPainter, QPen
from PyQt6.QtWidgets import (QCheckBox, QComboBox, QHBoxLayout, QLabel, QSlider,
                             QVBoxLayout, QWidget)

from app.imaging.volume import PLANE_FOR_AXIS, REGIONS, ImagingVolume, align_to_region
from app.ui.widgets.controls import ChipButton, SegmentBar

#: Distinct overlay colours for label outlines (cycled).
_OVERLAY = np.array([[255, 99, 71], [64, 196, 255], [255, 214, 10], [120, 220, 120], [220, 120, 255],
                     [255, 160, 60], [80, 255, 200], [255, 105, 180]], dtype=np.uint8)


class _SliceCanvas(QWidget):
    hovered = pyqtSignal(float, float)          # row, col in slice pixels (-1 = outside)
    clicked = pyqtSignal(float, float)
    wheel = pyqtSignal(int)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setMinimumSize(200, 200)
        self._image: Optional[QImage] = None
        self._aspect = 1.0
        self._caption = ""
        self._corner = ""
        self._orient = ("", "", "", "")         # top, bottom, left, right labels
        self._rect = QRectF()

    def set_image(self, image: Optional[QImage], aspect: float, caption: str, corner: str,
                  orient: tuple) -> None:
        self._image, self._aspect, self._caption, self._corner, self._orient = image, aspect, caption, corner, orient
        self.update()

    def _target(self) -> QRectF:
        if self._image is None:
            return QRectF()
        w, h = self._image.width(), self._image.height() * self._aspect
        scale = min((self.width() - 24) / max(w, 1), (self.height() - 44) / max(h, 1))
        tw, th = w * scale, h * scale
        return QRectF((self.width() - tw) / 2, (self.height() - th) / 2 + 6, tw, th)

    def paintEvent(self, _event) -> None:  # noqa: N802
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(4, 6, 9))
        if self._image is None:
            p.setPen(QColor(150, 160, 175))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                       "No imaging at this plane.\nSimulate an MRI or load a scan,\nthen move the cross-section.")
            return
        self._rect = self._target()
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        p.drawImage(self._rect, self._image)
        p.setPen(QColor(120, 210, 255))
        f = QFont(self.font())
        f.setPointSizeF(max(7.5, f.pointSizeF() * 0.95))
        p.setFont(f)
        top, bottom, left, right = self._orient
        r = self._rect
        p.drawText(QRectF(r.left(), r.top() - 18, r.width(), 16), Qt.AlignmentFlag.AlignHCenter, top)
        p.drawText(QRectF(r.left(), r.bottom() + 2, r.width(), 16), Qt.AlignmentFlag.AlignHCenter, bottom)
        p.drawText(QRectF(r.left() - 16, r.center().y() - 8, 14, 16), Qt.AlignmentFlag.AlignRight, left)
        p.drawText(QRectF(r.right() + 2, r.center().y() - 8, 14, 16), Qt.AlignmentFlag.AlignLeft, right)
        p.setPen(QColor(235, 238, 245))
        p.drawText(QRectF(8, 4, self.width() - 16, 18), Qt.AlignmentFlag.AlignLeft, self._caption)
        if self._corner:
            p.setPen(QColor(255, 190, 70))
            p.drawText(QRectF(8, 4, self.width() - 16, 18), Qt.AlignmentFlag.AlignRight, self._corner)

    def _to_pixel(self, pos: QPointF):
        if self._image is None or self._rect.isEmpty() or not self._rect.contains(pos):
            return -1.0, -1.0
        col = (pos.x() - self._rect.left()) / self._rect.width() * self._image.width()
        row = (pos.y() - self._rect.top()) / self._rect.height() * self._image.height()
        return row, col

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        self.hovered.emit(*self._to_pixel(event.position()))

    def mousePressEvent(self, event) -> None:  # noqa: N802
        row, col = self._to_pixel(event.position())
        if row >= 0:
            self.clicked.emit(row, col)

    def wheelEvent(self, event) -> None:  # noqa: N802
        self.wheel.emit(1 if event.angleDelta().y() > 0 else -1)

    def leaveEvent(self, _event) -> None:  # noqa: N802
        self.hovered.emit(-1.0, -1.0)


class SlicePane(QWidget):
    """MRI slice synchronised with the viewport's clip plane."""

    planeRequested = pyqtSignal(str, float)       # axis, world position → move 3D cut
    structureClicked = pyqtSignal(str)            # name under the cursor
    sequenceRequested = pyqtSignal(str)           # "T1" | "T2" | "PD" (simulated studies)
    closeRequested = pyqtSignal()

    def __init__(self, viewport, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("SlicePane")
        self._viewport = viewport
        self._volume: Optional[ImagingVolume] = None
        self._axis = "z"
        self._position = 1.2
        self._window = 1.0
        self._level = 0.5
        self._name_lookup: Optional[Callable] = None

        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(6)

        head = QHBoxLayout()
        self._title = QLabel("MRI pairing")
        self._title.setObjectName("LayerMeta")
        self._planes = SegmentBar([("z", "Axial"), ("y", "Coronal"), ("x", "Sagittal")], current="z")
        self._planes.changed.connect(self._on_plane_changed)
        close = ChipButton("✕")
        close.setToolTip("Hide the MRI pane")
        close.clicked.connect(self.closeRequested)
        head.addWidget(self._title, 1)
        head.addWidget(self._planes)
        head.addWidget(close)
        root.addLayout(head)

        self._canvas = _SliceCanvas()
        self._canvas.hovered.connect(self._on_hover)
        self._canvas.clicked.connect(self._on_click)
        self._canvas.wheel.connect(self._step)
        root.addWidget(self._canvas, 1)

        self._readout = QLabel("")
        self._readout.setObjectName("LayerMeta")
        root.addWidget(self._readout)

        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(0, 1000)
        self._slider.valueChanged.connect(self._on_slider)
        root.addWidget(self._slider)

        controls = QHBoxLayout()
        self._sequence = QComboBox()
        for key, label in (("T1", "T1"), ("T2", "T2"), ("PD", "PD")):
            self._sequence.addItem(label, key)
        self._sequence.setToolTip("Contrast weighting (simulated studies)")
        self._sequence.currentIndexChanged.connect(lambda: self.sequenceRequested.emit(self._sequence.currentData()))
        self._overlay = QCheckBox("Labels")
        self._overlay.setToolTip("Outline labelled structures")
        self._overlay.stateChanged.connect(lambda: self.refresh())
        self._region = QComboBox()
        for region in REGIONS:
            self._region.addItem(f"Align: {region}", region)
        self._region.setToolTip("Place the scan over this part of the model")
        self._region.activated.connect(self._on_align)
        controls.addWidget(self._sequence)
        controls.addWidget(self._overlay)
        controls.addWidget(self._region, 1)
        root.addLayout(controls)

        wl = QHBoxLayout()
        self._w = QSlider(Qt.Orientation.Horizontal)
        self._l = QSlider(Qt.Orientation.Horizontal)
        for s in (self._w, self._l):
            s.setRange(1, 1000)
            s.valueChanged.connect(self._on_window)
        wl.addWidget(QLabel("W"))
        wl.addWidget(self._w, 1)
        wl.addWidget(QLabel("L"))
        wl.addWidget(self._l, 1)
        root.addLayout(wl)
        self._range = (0.0, 1.0)

    # ------------------------------------------------------------- volume
    @property
    def volume(self) -> Optional[ImagingVolume]:
        return self._volume

    def set_volume(self, volume: Optional[ImagingVolume], *, keep_window: bool = False) -> None:
        self._volume = volume
        if volume is None:
            self._canvas.set_image(None, 1.0, "", "", ("", "", "", ""))
            return
        self._sequence.setVisible(volume.simulated)
        self._title.setText(volume.modality)
        self._title.setToolTip(volume.description)
        lo, hi = float(np.percentile(volume.data[::4, ::4, ::4], 0.5)), float(volume.data.max())
        self._range = (lo, max(hi, lo + 1e-6))
        if not keep_window:
            self._window, self._level = volume.default_window()
            self._sync_wl_sliders()
        axis, pos = self._viewport.clip_state() if hasattr(self._viewport, "clip_state") else ("", 0.0)
        if axis:
            self._axis, self._position = axis, pos
            self._planes.set_current(axis)
        else:
            b = volume.bounds()
            k = {"x": 0, "y": 1, "z": 2}[self._axis]
            self._position = (b[2 * k] + b[2 * k + 1]) / 2
        self._sync_slider()
        self.refresh()

    def set_name_lookup(self, lookup: Optional[Callable]) -> None:
        """Fallback naming for real scans: world point → model structure name."""
        self._name_lookup = lookup

    # ------------------------------------------------------------- syncing
    def follow_clip(self, axis: str, position: float) -> None:
        """Called when the 3D cross-section moves."""
        if not axis:
            return
        self._axis, self._position = axis, position
        self._planes.set_current(axis)
        self._sync_slider()
        self.refresh()

    def _sync_slider(self) -> None:
        if self._volume is None:
            return
        b = self._volume.bounds()
        k = {"x": 0, "y": 1, "z": 2}[self._axis]
        lo, hi = b[2 * k], b[2 * k + 1]
        value = int(round((self._position - lo) / ((hi - lo) or 1.0) * 1000))
        blocked = self._slider.blockSignals(True)
        self._slider.setValue(max(0, min(1000, value)))
        self._slider.blockSignals(blocked)

    def _on_slider(self, value: int) -> None:
        if self._volume is None:
            return
        b = self._volume.bounds()
        k = {"x": 0, "y": 1, "z": 2}[self._axis]
        self._position = b[2 * k] + (b[2 * k + 1] - b[2 * k]) * value / 1000.0
        self.refresh()
        self.planeRequested.emit(self._axis, self._position)

    def _step(self, direction: int) -> None:
        if self._volume is None:
            return
        k = {"x": 0, "y": 1, "z": 2}[self._axis]
        self._position += direction * self._volume.spacing[k]
        self._sync_slider()
        self.refresh()
        self.planeRequested.emit(self._axis, self._position)

    def _on_plane_changed(self, axis: str) -> None:
        self._axis = axis
        if self._volume is not None:
            b = self._volume.bounds()
            k = {"x": 0, "y": 1, "z": 2}[axis]
            self._position = (b[2 * k] + b[2 * k + 1]) / 2
        self._sync_slider()
        self.refresh()
        self.planeRequested.emit(self._axis, self._position)

    def _on_align(self) -> None:
        if self._volume is None:
            return
        region = self._region.currentData()
        align_to_region(self._volume, self._viewport, region)
        self._sync_slider()
        self.refresh()

    # --------------------------------------------------------------- window
    def _sync_wl_sliders(self) -> None:
        lo, hi = self._range
        span = hi - lo
        for slider, value in ((self._w, self._window / span), (self._l, (self._level - lo) / span)):
            blocked = slider.blockSignals(True)
            slider.setValue(int(max(1, min(1000, value * 1000))))
            slider.blockSignals(blocked)

    def _on_window(self) -> None:
        lo, hi = self._range
        span = hi - lo
        self._window = max(1e-6, self._w.value() / 1000 * span)
        self._level = lo + self._l.value() / 1000 * span
        self.refresh()

    # -------------------------------------------------------------- render
    def refresh(self) -> None:
        vol = self._volume
        if vol is None:
            return
        img, aspect = vol.slice(self._axis, self._position)
        if img is None:
            self._canvas.set_image(None, 1.0, "", "", ("", "", "", ""))
            self._readout.setText("This plane is outside the scan — move the cross-section, "
                                  "or use Align to place the scan.")
            return
        lo = self._level - self._window / 2
        grey = np.clip((img - lo) / self._window, 0.0, 1.0)
        rgb = np.repeat((grey * 255).astype(np.uint8)[:, :, None], 3, axis=2)
        if self._overlay.isChecked() and vol.labels is not None:
            lab, _ = vol.slice(self._axis, self._position, vol.labels)
            if lab is not None:
                edge = np.zeros(lab.shape, bool)
                edge[1:, :] |= lab[1:, :] != lab[:-1, :]
                edge[:, 1:] |= lab[:, 1:] != lab[:, :-1]
                edge &= lab > 0
                rgb[edge] = _OVERLAY[lab[edge] % len(_OVERLAY)]
        rgb = np.ascontiguousarray(rgb)
        h, w, _ = rgb.shape
        qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
        plane = PLANE_FOR_AXIS[self._axis]
        caption = f"{plane}  ·  {self._position * 1000:+.0f} mm"
        corner = "SIMULATED" if vol.simulated else ""
        orient = {"z": ("A", "P", "R", "L"), "y": ("S", "I", "R", "L"), "x": ("S", "I", "A", "P")}[self._axis]
        self._canvas.set_image(qimg, aspect, caption, corner, orient)
        self._readout.setText(vol.description)

    # --------------------------------------------------------------- hover
    def _name_at(self, row: float, col: float) -> str:
        vol = self._volume
        if vol is None or row < 0:
            return ""
        point = vol.display_to_world(self._axis, self._position, row, col)
        name = vol.label_at(point)
        if not name and self._name_lookup is not None:
            name = self._name_lookup(point) or ""
        return name

    def _on_hover(self, row: float, col: float) -> None:
        if self._volume is None:
            return
        if row < 0:
            self._readout.setText(self._volume.description)
            return
        name = self._name_at(row, col)
        point = self._volume.display_to_world(self._axis, self._position, row, col)
        vox = self._volume.voxel_at(point)
        signal = f"  ·  signal {self._volume.data[vox]:.2f}" if vox is not None else ""
        self._readout.setText((name or "—") + signal)

    def _on_click(self, row: float, col: float) -> None:
        name = self._name_at(row, col)
        if name:
            self.structureClicked.emit(name)

    def set_sequence(self, sequence: str) -> None:
        index = self._sequence.findData(sequence)
        if index >= 0:
            blocked = self._sequence.blockSignals(True)
            self._sequence.setCurrentIndex(index)
            self._sequence.blockSignals(blocked)
