"""
Simulation page: muscle actions, disease progression and MRI cross-sections.

The panel only emits intents; ``MainWindow`` owns the rig, the tour engine and
the exporter. Sections are added by ``add_section`` so each feature stays in its
own module.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (QCheckBox, QComboBox, QHBoxLayout, QLabel, QPushButton,
                             QScrollArea, QVBoxLayout, QWidget)

from app.anatomy.muscles import MOTIONS, MUSCLES, describe, muscle_for_structure
from app.ui.widgets.collapsible import CollapsibleSection
from app.ui.widgets.controls import ChipButton, SegmentBar


def _row(label: str, widget: QWidget) -> QHBoxLayout:
    row = QHBoxLayout()
    caption = QLabel(label)
    caption.setObjectName("LayerMeta")
    caption.setMinimumWidth(64)
    row.addWidget(caption)
    row.addWidget(widget, 1)
    return row


class SimulationPanel(QWidget):
    """Scrollable page hosting the simulation sections."""

    muscleTourRequested = pyqtSignal(str, str)        # muscle key, side
    motionTourRequested = pyqtSignal(str, str)        # motion id, side
    exportRequested = pyqtSignal(str, str, str)       # kind ("muscle"|"motion"|…), key, side
    stopRequested = pyqtSignal()
    mriSimulateRequested = pyqtSignal(float)          # voxel size, metres
    mriLoadRequested = pyqtSignal(str)                # file or DICOM folder
    mriPaneToggled = pyqtSignal(bool)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setMinimumHeight(120)
        holder = QWidget()
        self._layout = QVBoxLayout(holder)
        self._layout.setContentsMargins(10, 0, 10, 10)
        self._layout.setSpacing(9)
        self._layout.addStretch(1)
        scroll.setWidget(holder)
        outer.addWidget(scroll)
        self.add_section(self._build_muscle_section())
        self.add_section(self._build_mri_section())

    def add_section(self, section: QWidget) -> None:
        self._layout.insertWidget(self._layout.count() - 1, section)

    # ------------------------------------------------------------ muscles
    def _build_muscle_section(self) -> CollapsibleSection:
        section = CollapsibleSection("Muscle Actions", expanded=True)
        intro = QLabel("Pick a muscle to see it contract and move its joint through the real "
                       "action. Agonist glows orange, synergists amber, antagonists blue.")
        intro.setObjectName("LayerMeta")
        intro.setWordWrap(True)
        section.add_widget(intro)

        self._side = SegmentBar([("right", "Right"), ("left", "Left")], current="right")
        section.add_layout(_row("Side", self._side))

        self._muscle = QComboBox()
        last_region = ""
        for muscle in sorted(MUSCLES, key=lambda m: (m.region, m.name)):
            if muscle.region != last_region and self._muscle.count():
                self._muscle.insertSeparator(self._muscle.count())
            last_region = muscle.region
            self._muscle.addItem(f"{muscle.region} · {muscle.name}", muscle.key)
        self._muscle.currentIndexChanged.connect(self._refresh_muscle_card)
        section.add_layout(_row("Muscle", self._muscle))

        buttons = QHBoxLayout()
        play = QPushButton("▶  Play action")
        play.setObjectName("PrimaryButton")
        play.clicked.connect(lambda: self.muscleTourRequested.emit(self.current_muscle(), self.current_side()))
        export = ChipButton("Export video")
        export.clicked.connect(lambda: self.exportRequested.emit("muscle", self.current_muscle(),
                                                                 self.current_side()))
        stop = ChipButton("Stop")
        stop.clicked.connect(self.stopRequested)
        buttons.addWidget(play, 1)
        buttons.addWidget(export)
        buttons.addWidget(stop)
        section.add_layout(buttons)

        self._card = QLabel("")
        self._card.setObjectName("LayerMeta")
        self._card.setWordWrap(True)
        self._card.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        section.add_widget(self._card)

        self._motion = QComboBox()
        for motion in MOTIONS.values():
            self._motion.addItem(f"{motion.label}  ({motion.normal_range})", motion.id)
        section.add_layout(_row("Motion", self._motion))
        motion_buttons = QHBoxLayout()
        animate = ChipButton("▶  Animate motion")
        animate.setToolTip("Every agonist and antagonist of one joint motion together")
        animate.clicked.connect(lambda: self.motionTourRequested.emit(self.current_motion(), self.current_side()))
        export_motion = ChipButton("Export video")
        export_motion.clicked.connect(lambda: self.exportRequested.emit("motion", self.current_motion(),
                                                                        self.current_side()))
        motion_buttons.addWidget(animate, 1)
        motion_buttons.addWidget(export_motion)
        section.add_layout(motion_buttons)

        self._rig_note = QLabel("")
        self._rig_note.setObjectName("LayerMeta")
        self._rig_note.setWordWrap(True)
        self._rig_note.setVisible(False)
        section.add_widget(self._rig_note)
        self._refresh_muscle_card()
        return section

    # ---------------------------------------------------------------- MRI
    def _build_mri_section(self) -> CollapsibleSection:
        section = CollapsibleSection("Cross-section & MRI", expanded=True)
        intro = QLabel("Pair every 3D cross-section with the matching MRI slice. Load a real "
                       "study (NIfTI or a DICOM folder), or simulate T1/T2/PD contrast from the "
                       "model for teaching. Move the slice in either view; the other follows.")
        intro.setObjectName("LayerMeta")
        intro.setWordWrap(True)
        section.add_widget(intro)

        self._mri_res = QComboBox()
        for label, size in (("4 mm  (fast)", 0.004), ("3 mm", 0.003), ("2 mm  (detailed, slower)", 0.002)):
            self._mri_res.addItem(label, size)
        section.add_layout(_row("Voxel", self._mri_res))

        buttons = QHBoxLayout()
        simulate = QPushButton("Simulate MRI")
        simulate.setObjectName("PrimaryButton")
        simulate.clicked.connect(lambda: self.mriSimulateRequested.emit(float(self._mri_res.currentData())))
        load_file = ChipButton("Load scan…")
        load_file.setToolTip("NIfTI (.nii / .nii.gz)")
        load_file.clicked.connect(self._choose_scan_file)
        load_dir = ChipButton("DICOM folder…")
        load_dir.clicked.connect(self._choose_scan_folder)
        buttons.addWidget(simulate, 1)
        buttons.addWidget(load_file)
        buttons.addWidget(load_dir)
        section.add_layout(buttons)

        self._mri_toggle = QCheckBox("Show paired MRI view")
        self._mri_toggle.toggled.connect(self.mriPaneToggled)
        section.add_widget(self._mri_toggle)

        self._mri_status = QLabel("No imaging loaded.")
        self._mri_status.setObjectName("LayerMeta")
        self._mri_status.setWordWrap(True)
        section.add_widget(self._mri_status)
        return section

    def _choose_scan_file(self) -> None:
        from PyQt6.QtWidgets import QFileDialog
        path, _ = QFileDialog.getOpenFileName(self, "Open MRI (NIfTI)", "",
                                              "NIfTI (*.nii *.nii.gz);;All files (*)")
        if path:
            self.mriLoadRequested.emit(path)

    def _choose_scan_folder(self) -> None:
        from PyQt6.QtWidgets import QFileDialog
        path = QFileDialog.getExistingDirectory(self, "Open DICOM series folder")
        if path:
            self.mriLoadRequested.emit(path)

    def set_mri_status(self, text: str) -> None:
        self._mri_status.setText(text)

    def set_mri_pane_checked(self, checked: bool) -> None:
        blocked = self._mri_toggle.blockSignals(True)
        self._mri_toggle.setChecked(checked)
        self._mri_toggle.blockSignals(blocked)

    def current_muscle(self) -> str:
        return self._muscle.currentData() or ""

    def current_side(self) -> str:
        return self._side.current or "right"

    def current_motion(self) -> str:
        return self._motion.currentData() or ""

    def follow_structure(self, name: str) -> bool:
        """Select the muscle (and side) the user just clicked in the 3D view."""
        muscle = muscle_for_structure(name or "")
        if muscle is None:
            return False
        index = self._muscle.findData(muscle.key)
        if index >= 0:
            self._muscle.setCurrentIndex(index)
        low = name.lower()
        if "left" in low:
            self._side.set_current("left")
        elif "right" in low:
            self._side.set_current("right")
        return True

    def set_rig_status(self, available: bool, detail: str = "") -> None:
        self._rig_note.setText(detail if detail else
                               "" if available else
                               "Joint animation needs named limb bones (humerus, femur …). "
                               "Muscles will be shown contracting in place.")
        self._rig_note.setVisible(bool(self._rig_note.text()))

    def _refresh_muscle_card(self) -> None:
        key = self.current_muscle()
        muscle = next((m for m in MUSCLES if m.key == key), None)
        self._card.setText(describe(muscle) if muscle else "")
