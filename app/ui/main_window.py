"""
Application shell.

Layout:

    ┌──────────────────────────────────────────────────────────────────────┐
    │ TitleBar   brand · scene summary · settings · window buttons         │
    ├───────────────┬──────────────────────────────────┬───────────────────┤
    │ LayerPanel    │ ViewportPanel                    │ AIPanel           │
    │ layers+tools  │  (VTK surface + glass overlays)  │ assistant + audio │
    ├───────────────┴──────────────────────────────────┴───────────────────┤
    │ StatusBar   provider · hardware · transient messages                 │
    └──────────────────────────────────────────────────────────────────────┘

The window owns the objects and wires signals; it holds almost no logic, which
is what keeps the three subsystems independently testable.
"""
from __future__ import annotations

import datetime as _dt
import subprocess
import sys
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import QByteArray, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QGuiApplication, QKeySequence, QShortcut
from PyQt6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox,
                             QFormLayout, QHBoxLayout, QLabel, QLineEdit,
                             QMainWindow, QMenu, QMessageBox, QSplitter,
                             QStatusBar, QToolButton, QVBoxLayout, QWidget)

from app.ai.manager import AIManager
from app.audio.manager import AudioManager
from app.audio.scripts import get_tour, subject_for_layer, tour_for_system
from app.audio.voices import install_help
from app.core.model_registry import LayerRegistry
from app.core.scene_controller import SceneController
from app.core.viewport import create_viewport
from app.ui.widgets.controls import GLYPH, IconButton
from app.ui.widgets.ai_panel import AIPanel, VIDEO_PRESETS
from app.ui.widgets.layer_panel import LayerPanel
from app.ui.widgets.simulation_panel import SimulationPanel
from app.ui.widgets.viewport_panel import ViewportPanel
from app.video.exporter import TourVideoExporter, VideoSettings
from app.video.player import VideoPlayerDialog, open_in_default_player


# ---------------------------------------------------------------------------
# Title bar
# ---------------------------------------------------------------------------
class TitleBar(QWidget):
    """Slim application header. Draggable when the window is frameless."""

    settingsRequested = pyqtSignal()
    sceneResetRequested = pyqtSignal()

    def __init__(self, *, frameless: bool = False, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("TitleBar")
        self._frameless = frameless
        self._drag_offset = None

        layout = QHBoxLayout(self)
        layout.setContentsMargins(14, 0, 10, 0)
        layout.setSpacing(11)

        brand = QLabel("B")
        brand.setObjectName("BrandMark")
        brand.setAlignment(Qt.AlignmentFlag.AlignCenter)

        titles = QVBoxLayout()
        titles.setContentsMargins(0, 0, 0, 0)
        titles.setSpacing(0)
        title = QLabel("BioHuman3D")
        title.setObjectName("TitleBarTitle")
        subtitle = QLabel("Anatomy & Health Studio")
        subtitle.setObjectName("TitleBarSubtitle")
        titles.addWidget(title)
        titles.addWidget(subtitle)

        layout.addWidget(brand)
        layout.addLayout(titles)
        layout.addStretch(1)

        self._scene_label = QLabel("")
        self._scene_label.setObjectName("TitleBarSubtitle")
        layout.addWidget(self._scene_label)
        layout.addSpacing(8)

        settings = IconButton(GLYPH["tools"], "Settings  (Ctrl+,)")
        settings.clicked.connect(self.settingsRequested)
        reset = IconButton(GLYPH["reset"], "Reset scene  (Ctrl+R)")
        reset.clicked.connect(self.sceneResetRequested)
        layout.addWidget(settings)
        layout.addWidget(reset)

        if frameless:
            layout.addSpacing(6)
            for glyph, tip, slot, danger in (
                (GLYPH["minimize"], "Minimise", self._minimise, False),
                (GLYPH["maximize"], "Maximise", self._toggle_max, False),
                (GLYPH["close"], "Close", self._close, True),
            ):
                button = IconButton(glyph, tip)
                button.setObjectName("WindowButton")
                button.setProperty("danger", "true" if danger else "false")
                button.clicked.connect(slot)
                layout.addWidget(button)

    def set_scene_summary(self, text: str) -> None:
        self._scene_label.setText(text)

    def set_menu(self, menu: QMenu) -> None:
        """Attach an application menu button to the left of the settings icon."""
        button = IconButton(GLYPH["menu"], "Application menu")
        button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        button.setMenu(menu)
        layout = self.layout()
        # Insert directly after the brand/title block, before the stretch.
        layout.insertWidget(2, button)

    # -- frameless drag ----------------------------------------------------
    def mousePressEvent(self, event) -> None:  # noqa: N802
        if self._frameless and event.button() == Qt.MouseButton.LeftButton:
            window = self.window()
            self._drag_offset = event.globalPosition().toPoint() - window.frameGeometry().topLeft()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._frameless and self._drag_offset is not None:
            self.window().move(event.globalPosition().toPoint() - self._drag_offset)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        self._drag_offset = None
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802
        if self._frameless:
            self._toggle_max()
        super().mouseDoubleClickEvent(event)

    def _minimise(self) -> None:
        self.window().showMinimized()

    def _toggle_max(self) -> None:
        window = self.window()
        window.showNormal() if window.isMaximized() else window.showMaximized()

    def _close(self) -> None:
        self.window().close()


# ---------------------------------------------------------------------------
# Settings dialog
# ---------------------------------------------------------------------------
class SettingsDialog(QDialog):
    """Cloud keys + inference knobs. Reads/writes :class:`AppConfig` in place."""

    def __init__(self, config, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._config = config
        self.setWindowTitle("BioHuman3D — Settings")
        self.setMinimumWidth(470)

        root = QVBoxLayout(self)
        form = QFormLayout()
        form.setSpacing(9)

        self._openai = QLineEdit(config.api_key("openai"))
        self._openai.setEchoMode(QLineEdit.EchoMode.Password)
        self._openai.setPlaceholderText("sk-…  (or set OPENAI_API_KEY)")
        form.addRow("OpenAI key", self._openai)

        self._anthropic = QLineEdit(config.api_key("anthropic"))
        self._anthropic.setEchoMode(QLineEdit.EchoMode.Password)
        self._anthropic.setPlaceholderText("sk-ant-…  (or set ANTHROPIC_API_KEY)")
        form.addRow("Anthropic key", self._anthropic)

        self._eleven = QLineEdit(config.api_key("elevenlabs"))
        self._eleven.setEchoMode(QLineEdit.EchoMode.Password)
        self._eleven.setPlaceholderText("Optional — premium voiceovers")
        form.addRow("ElevenLabs key", self._eleven)
        self._eleven_voice = QLineEdit(str(config.get("audio.elevenlabs_voice", "")))
        self._eleven_voice.setPlaceholderText("Voice id from your ElevenLabs library")
        form.addRow("ElevenLabs voice", self._eleven_voice)

        # -- neural voices & translation -----------------------------------
        self._azure_key = QLineEdit(str(config.get("audio.azure_key", "")))
        self._azure_key.setEchoMode(QLineEdit.EchoMode.Password)
        self._azure_key.setPlaceholderText("Azure AI Speech key (production neural voices)")
        form.addRow("Azure Speech key", self._azure_key)
        self._azure_region = QLineEdit(str(config.get("audio.azure_region", "")))
        self._azure_region.setPlaceholderText("e.g. centralindia, southeastasia, eastus")
        form.addRow("Azure region", self._azure_region)
        self._google_key = QLineEdit(str(config.get("audio.google_key", "")))
        self._google_key.setEchoMode(QLineEdit.EchoMode.Password)
        self._google_key.setPlaceholderText("Google Cloud API key (Text-to-Speech / Translation)")
        form.addRow("Google Cloud key", self._google_key)
        self._i18n_engine = QComboBox()
        for label, key in (("Your AI model (local or cloud)", "llm"),
                           ("Google Cloud Translation", "google"),
                           ("Azure AI Translator", "azure")):
            self._i18n_engine.addItem(label, key)
        self._i18n_engine.setCurrentIndex(max(0, self._i18n_engine.findData(config.get("i18n.engine", "llm"))))
        form.addRow("Translate narration with", self._i18n_engine)
        self._translator_key = QLineEdit(str(config.get("i18n.azure_key", "")))
        self._translator_key.setEchoMode(QLineEdit.EchoMode.Password)
        self._translator_key.setPlaceholderText("Azure Translator key (if chosen above)")
        form.addRow("Azure Translator key", self._translator_key)

        self._temperature = QComboBox()
        for value in (0.0, 0.2, 0.3, 0.5, 0.7, 1.0):
            self._temperature.addItem(str(value), value)
        index = self._temperature.findData(float(config.get("ai.temperature", 0.3)))
        self._temperature.setCurrentIndex(max(0, index))
        form.addRow("Temperature", self._temperature)

        self._depth_peeling = QCheckBox("Correct transparency (depth peeling)")
        self._depth_peeling.setChecked(bool(config.get("render.depth_peeling", True)))
        form.addRow("", self._depth_peeling)

        self._fxaa = QCheckBox("Anti-aliasing (FXAA)")
        self._fxaa.setChecked(bool(config.get("render.fxaa", True)))
        form.addRow("", self._fxaa)

        self._auto_explain = QCheckBox("Auto-explain on selection")
        self._auto_explain.setChecked(bool(config.get("ui.auto_explain_on_select", False)))
        form.addRow("", self._auto_explain)

        root.addLayout(form)

        note = QLabel("Renderer changes apply on restart. AI and audio settings apply immediately.")
        note.setObjectName("PanelHint")
        note.setWordWrap(True)
        root.addWidget(note)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save
                                   | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self._save)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    def _save(self) -> None:
        self._config["ai.openai_key"] = self._openai.text().strip()
        self._config["ai.anthropic_key"] = self._anthropic.text().strip()
        self._config["audio.elevenlabs_key"] = self._eleven.text().strip()
        self._config["audio.elevenlabs_voice"] = self._eleven_voice.text().strip()
        self._config["audio.azure_key"] = self._azure_key.text().strip()
        self._config["audio.azure_region"] = self._azure_region.text().strip()
        self._config["audio.google_key"] = self._google_key.text().strip()
        self._config["i18n.engine"] = self._i18n_engine.currentData()
        self._config["i18n.azure_key"] = self._translator_key.text().strip()
        if not self._config.get("i18n.azure_region", ""):
            self._config["i18n.azure_region"] = self._azure_region.text().strip()
        self._config["ai.temperature"] = float(self._temperature.currentData())
        self._config["render.depth_peeling"] = self._depth_peeling.isChecked()
        self._config["render.fxaa"] = self._fxaa.isChecked()
        self._config["ui.auto_explain_on_select"] = self._auto_explain.isChecked()
        self._config.save()
        self.accept()


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------
class MainWindow(QMainWindow):
    """BioHuman3D application shell."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.setWindowTitle("BioHuman3D — Anatomy & Health Studio")
        # Kept low enough to fit a small laptop panel once panels are collapsed.
        self.setMinimumSize(1040, 640)
        self._apply_startup_geometry()

        # ---------------------------------------------------------- subsystems
        self.registry = LayerRegistry(config.paths.models)
        self.viewport = create_viewport(self.registry, config)
        self.audio = AudioManager(config, self)
        self.ai = AIManager(config, self.registry, self)
        self.controller = SceneController(config, self.registry, self.viewport,
                                          self.audio, self.ai, self)
        self.video = TourVideoExporter(self.viewport, self.config, self)

        self._build_ui()
        self._wire_signals()
        self._install_shortcuts()

        QTimer.singleShot(80, self._startup)

    # ============================================================== build UI
    def _build_ui(self) -> None:
        central = QWidget()
        central.setObjectName("Root")
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self.title_bar = TitleBar(frameless=bool(self.config.get("ui.frameless", False)))
        root.addWidget(self.title_bar)

        # -- three-panel splitter ------------------------------------------
        tours = self.controller.available_tours()
        self.layer_panel = LayerPanel(tours)
        self.viewport_panel = ViewportPanel(self.viewport)
        self.ai_panel = AIPanel(tours)
        self.simulation_panel = SimulationPanel()
        self.ai_panel.set_simulation_widget(self.simulation_panel)
        self._rig = None

        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.setChildrenCollapsible(True)
        self.splitter.setHandleWidth(1)
        self.splitter.addWidget(self.layer_panel)
        # Centre: the 3D view, and (on demand) the paired MRI slice beside it.
        from app.imaging.slice_pane import SlicePane
        self.slice_pane = SlicePane(self.viewport)
        self.slice_pane.setVisible(False)
        self.center_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.center_splitter.setChildrenCollapsible(True)
        self.center_splitter.setHandleWidth(2)
        self.center_splitter.addWidget(self.viewport_panel)
        self.center_splitter.addWidget(self.slice_pane)
        self.splitter.addWidget(self.center_splitter)
        self.splitter.addWidget(self.ai_panel)
        self._mri_sim = None
        self._mri_worker = None
        self._translation_worker = None
        self._pending_localization = None
        self._disease_scene = None
        self._disease_isolation = None
        self._disease_condition = ""
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setStretchFactor(2, 0)
        self.splitter.setSizes([
            int(self.config.get("ui.left_width", 320)),
            1000,
            int(self.config.get("ui.right_width", 380)),
        ])
        root.addWidget(self.splitter, 1)

        self.setCentralWidget(central)
        self.title_bar.set_menu(self._build_menu())

        # -- status bar -----------------------------------------------------
        self.status = QStatusBar()
        self.status.setSizeGripEnabled(False)
        self.setStatusBar(self.status)
        self._status_provider = QLabel("Scanning…")
        self._status_scene = QLabel("")
        self.status.addPermanentWidget(self._status_scene)
        self.status.addPermanentWidget(self._status_provider)

    # ============================================================ geometry
    def _available_geometry(self):
        screen = QGuiApplication.primaryScreen()
        return screen.availableGeometry() if screen else None

    def _apply_startup_geometry(self) -> None:
        """Restore the saved window geometry, or fit a sensible default on screen.

        Everything here exists because of one failure mode: a window larger than
        the desktop silently hides the status bar and the bottom of the side
        panels. A hard-coded 1680x980 default does exactly that on a 1536x816
        work area, and the user simply cannot reach the controls at the bottom.
        """
        available = self._available_geometry()

        saved = str(self.config.get("ui.geometry", "") or "")
        if saved:
            try:
                if self.restoreGeometry(QByteArray.fromBase64(saved.encode("ascii"))):
                    # A restored geometry must FIT, not merely overlap: an
                    # oversized window still "intersects" the screen while its
                    # lower edge sits behind the taskbar.
                    self._clamp_to_screen(available)
                    return
            except Exception:
                pass

        if available is None:
            self.resize(1400, 880)
            return

        self.resize(min(1680, int(available.width() * 0.96)),
                    min(980, int(available.height() * 0.94)))
        self._clamp_to_screen(available)
        # Centre only when we are choosing the size ourselves.
        self.move(available.left() + max(0, (available.width() - self.width()) // 2),
                  available.top() + max(0, (available.height() - self.height()) // 2))
        self._clamp_to_screen(available)

    def _clamp_to_screen(self, available, passes: int = 1) -> None:
        """Shrink and reposition so the whole frame (incl. status bar) is visible.

        Works from the **measured overflow** rather than a predicted decoration
        size. That matters because ``frameGeometry()`` under-reports decorations
        until the native window exists — measuring "deco = 24px" when it is
        really 31px leaves the status bar 7px off the bottom of the screen, which
        is precisely the kind of near-miss that looks like it works.
        """
        if available is None or self.isMaximized() or self.isFullScreen():
            return

        for _ in range(passes):
            frame = self.frameGeometry()
            overflow_w = frame.width() - available.width()
            overflow_h = frame.height() - available.height()
            if overflow_w <= 0 and overflow_h <= 0:
                break
            client = self.geometry()
            width = max(self.minimumWidth(), client.width() - max(0, overflow_w))
            height = max(self.minimumHeight(), client.height() - max(0, overflow_h))
            if (width, height) == (client.width(), client.height()):
                break           # already at the layout floor; nothing more to give
            self.resize(width, height)

        # Re-anchor using the FRAME, not the client rect. A window whose height
        # fits can still sit partly off-screen if it is positioned too low: the
        # status bar ends up behind the taskbar even though `height <= available`.
        frame = self.frameGeometry()
        dx = dy = 0
        if frame.left() < available.left():
            dx = available.left() - frame.left()
        elif frame.right() > available.right():
            dx = available.right() - frame.right()
        if frame.top() < available.top():
            dy = available.top() - frame.top()
        elif frame.bottom() > available.bottom():
            dy = available.bottom() - frame.bottom()
        if dx or dy:
            self.move(self.x() + dx, self.y() + dy)

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        if getattr(self, "_geometry_checked", False):
            return
        self._geometry_checked = True
        available = self._available_geometry()
        self._clamp_to_screen(available)
        # Decorations become measurable only after the native frame is created, so
        # re-clamp on later event-loop turns. Each call is a SINGLE pass: the frame
        # geometry is not recomputed synchronously, so looping inside one call
        # would re-subtract the same overflow and shrink the window to its floor.
        for delay in (0, 120, 400):
            QTimer.singleShot(
                delay, lambda: self._clamp_to_screen(self._available_geometry()))

    def _save_geometry(self) -> None:
        try:
            self.config["ui.geometry"] = bytes(
                self.saveGeometry().toBase64()).decode("ascii")
        except Exception:
            pass

    # ================================================================ menu
    def _build_menu(self) -> QMenu:
        menu = QMenu(self)

        media = menu.addMenu("Tour video")
        make = media.addAction("Generate narrated video")
        make.triggered.connect(self._open_video_export)
        play = media.addAction("Play last video…")
        play.triggered.connect(lambda: self._play_video(self.ai_panel._last_video_path
                                                        if self.ai_panel._last_video_path
                                                        else ""))
        media.addSeparator()
        folder = media.addAction("Open exports folder")
        folder.triggered.connect(self._open_exports_folder)

        file_menu = menu.addMenu("File")
        shot = file_menu.addAction("Save screenshot")
        shot.setShortcut(QKeySequence("Ctrl+S"))
        shot.triggered.connect(self._take_snapshot)
        settings = file_menu.addAction("Settings…")
        settings.setShortcut(QKeySequence("Ctrl+,"))
        settings.triggered.connect(self.open_settings)
        file_menu.addSeparator()
        quit_action = file_menu.addAction("Exit")
        quit_action.triggered.connect(self.close)

        view = menu.addMenu("View")
        for label, preset in (("Anterior", "anterior"), ("Posterior", "posterior"),
                              ("Left", "left"), ("Right", "right"),
                              ("Superior", "superior"), ("3D isometric", "isometric")):
            action = view.addAction(label)
            action.triggered.connect(lambda _c=False, p=preset: self.controller.set_view(p))
        view.addSeparator()
        reset = view.addAction("Reset scene")
        reset.setShortcut(QKeySequence("Ctrl+R"))
        reset.triggered.connect(self._reset_scene)
        panels = view.addAction("Reset panel widths")
        panels.triggered.connect(self._reset_panel_widths)

        tours = menu.addMenu("Guided tours")
        for tour in self.controller.available_tours():
            action = tours.addAction(f"Play — {tour.title}")
            action.triggered.connect(
                lambda _c=False, t=tour.id: self.controller.start_tour(t))
        tours.addSeparator()
        stop = tours.addAction("Stop tour")
        stop.triggered.connect(self.controller.stop_tour)

        about = menu.addAction("About BioHuman3D")
        about.triggered.connect(self._show_about)
        return menu

    def _reset_panel_widths(self) -> None:
        self.splitter.setSizes([320, max(600, self.width() - 700), 380])
        self.status_message("Panel widths reset", 2500)

    def _open_exports_folder(self) -> None:
        folder = self.config.paths.user_data / "videos"
        folder.mkdir(parents=True, exist_ok=True)
        self._reveal(folder)

    def _show_about(self) -> None:
        from app import __version__
        from app.core.vtk_utils import VTK_AVAILABLE, vtk
        from app.video.exporter import video_support_status

        vtk_version = vtk.vtkVersion.GetVTKVersion() if VTK_AVAILABLE else "not installed"
        video_ok, video_detail = video_support_status()
        QMessageBox.about(
            self, "About BioHuman3D",
            f"<b>BioHuman3D {__version__}</b><br>"
            f"Anatomy &amp; Health Studio<br><br>"
            f"PyQt6 · VTK {vtk_version} · "
            f"{'QtMultimedia playback' if True else ''}<br>"
            f"Video export: {'ready — ' + video_detail if video_ok else video_detail}<br>"
            f"Models: {len(getattr(self.viewport, 'layers', {}))} layers<br>"
            f"Models folder: {self.config.paths.models}",
        )

    # ================================================================ wiring
    def _wire_signals(self) -> None:
        # -- left panel → controller ---------------------------------------
        lp = self.layer_panel
        lp.layerOpacityChanged.connect(self.controller.set_layer_opacity)
        lp.layerVisibilityChanged.connect(self.controller.set_layer_visible)
        lp.isolateRequested.connect(self.controller.isolate_layer)
        lp.layerSelected.connect(self.controller.select_structure)
        lp.layerSelected.connect(lambda layer_id, _label: self._sync_subject_to_layer(layer_id))
        # Routed through a handler so choosing a system ALSO switches the video
        # subject — the bug was that export stayed on the previous tour.
        lp.presetRequested.connect(self._on_system_selected)
        lp.showAllRequested.connect(self.controller.show_all)
        lp.hideAllRequested.connect(lambda: self.viewport.hide_all_layers())
        lp.clipChanged.connect(self.viewport.apply_clip_plane)
        lp.clipCleared.connect(self.viewport.disable_clip_plane)
        lp.explodeChanged.connect(self.viewport.set_explode)
        lp.backgroundChanged.connect(self.viewport.set_background)
        lp.tourRequested.connect(self.controller.start_tour)
        lp.tourStopRequested.connect(self.controller.stop_tour)
        lp.tourStepRequested.connect(self._step_tour)
        lp.tourNarrateRequested.connect(self.controller.narrate_selection)

        # -- centre panel ---------------------------------------------------
        vp = self.viewport_panel
        vp.viewRequested.connect(self.controller.set_view)
        vp.resetRequested.connect(self.controller.reset_view)
        vp.snapshotRequested.connect(self._take_snapshot)
        vp.tourToggleRequested.connect(self._toggle_tour)
        vp.tourStopRequested.connect(self.controller.stop_tour)
        vp.tourStepRequested.connect(self._step_tour)
        vp.narrateRequested.connect(self.controller.narrate_selection)

        # -- viewport → shell ------------------------------------------------
        self.viewport.structurePicked.connect(self._on_structure_picked)
        self.viewport.hoverChanged.connect(self.viewport_panel.set_hover)
        self.viewport.fpsUpdated.connect(self.viewport_panel.set_fps)

        # -- right panel → managers -----------------------------------------
        ap = self.ai_panel
        ap.modeChanged.connect(self.ai.set_mode)
        ap.modelSelected.connect(self.ai.set_target)
        ap.refreshRequested.connect(self.ai.refresh_detection)
        ap.promptSubmitted.connect(self._submit_prompt)
        ap.quickActionRequested.connect(self._quick_action)
        ap.cancelRequested.connect(self.ai.cancel)
        ap.clearChatRequested.connect(self._clear_chat)
        ap.readAloudRequested.connect(self.controller.speak_text)
        ap.openSettingsRequested.connect(self.open_settings)
        ap.audioEnabledChanged.connect(self._on_audio_enabled)
        ap.audioBackendChanged.connect(self._on_audio_backend)
        ap.audioVoiceChanged.connect(self._on_audio_voice)
        ap.audioDialectChanged.connect(self._on_audio_dialect)
        ap.audioGenderChanged.connect(self._on_audio_gender)
        ap.voiceSetupRequested.connect(self._show_voice_help)
        ap.audioRateChanged.connect(self._on_audio_rate)
        ap.audioVolumeChanged.connect(self._on_audio_volume)
        ap.stopAudioRequested.connect(self.audio.stop)
        ap.replayCueRequested.connect(self._replay_cue)

        # -- right panel → video exporter ------------------------------------
        ap.videoGenerateRequested.connect(self._generate_video)
        ap.videoCancelRequested.connect(self._cancel_video)
        ap.videoPlayRequested.connect(self._play_video)
        ap.videoFolderRequested.connect(self._show_video_in_folder)
        ap.videoOptionsChanged.connect(self._update_video_estimate)

        # -- simulation page ---------------------------------------------------
        sp = self.simulation_panel
        sp.muscleTourRequested.connect(self._play_muscle_tour)
        sp.motionTourRequested.connect(self._play_motion_tour)
        sp.exportRequested.connect(self._export_simulation)
        sp.stopRequested.connect(self.controller.stop_tour)
        from app.i18n.translate import Translator
        self.translator = Translator(self.config, self.ai)
        self.controller.tour_resolver = lambda tour: self._localize(
            tour, lambda local: self.controller.start_tour(local.id))
        sp.diseaseStageRequested.connect(self._on_disease_stage)
        sp.diseaseTourRequested.connect(self._play_disease_tour)
        sp.diseaseResetRequested.connect(self._reset_disease)
        sp.mriSimulateRequested.connect(self._simulate_mri)
        sp.mriLoadRequested.connect(self._load_scan)
        sp.mriPaneToggled.connect(self._show_mri_pane)
        if hasattr(self.viewport, "clipMoved"):
            self.viewport.clipMoved.connect(self.slice_pane.follow_clip)
        self.slice_pane.planeRequested.connect(self._on_slice_plane)
        self.slice_pane.structureClicked.connect(self._on_mri_structure_clicked)
        self.slice_pane.sequenceRequested.connect(self._on_mri_sequence)
        self.slice_pane.closeRequested.connect(lambda: self._show_mri_pane(False))
        self.slice_pane.set_name_lookup(self._structure_at_point)

        # -- video exporter → panel -------------------------------------------
        self.video.progress.connect(self._on_video_progress)
        self.video.finished.connect(self._on_video_finished)
        self.video.failed.connect(self._on_video_failed)
        self.video.cancelled.connect(self._on_video_cancelled)

        # -- AI manager → panel ---------------------------------------------
        self.ai.detectionStarted.connect(
            lambda: self.ai_panel.set_status("Scanning local models…", "warn"))
        self.ai.detectionReady.connect(self._on_detection)
        self.ai.detectionFailed.connect(
            lambda msg: self.ai_panel.set_status(f"Detection failed: {msg}", "error"))
        self.ai.responseStarted.connect(self._on_response_started)
        self.ai.tokenReceived.connect(self.ai_panel.append_token)
        self.ai.responseFinished.connect(self._on_response_finished)
        self.ai.responseFailed.connect(self._on_response_failed)
        self.ai.providerChanged.connect(self._on_provider_changed)

        # -- audio manager → panel ------------------------------------------
        self.audio.cueStarted.connect(self.ai_panel.highlight_cue)
        self.audio.failed.connect(lambda msg: self.status_message(msg, 6000))

        # -- controller → shell ---------------------------------------------
        self.controller.statusMessage.connect(self.status_message)
        self.controller.structureChanged.connect(self._on_structure_changed)
        self.controller.layerStateChanged.connect(self._on_layer_state_changed)
        self.controller.sceneLoaded.connect(self._on_scene_loaded)
        self.controller.tourStarted.connect(self._on_tour_started)
        self.controller.tourKeyframe.connect(self.layer_panel.set_tour_progress)
        self.controller.tourFinished.connect(self._on_tour_finished)

        # -- title bar -------------------------------------------------------
        self.title_bar.settingsRequested.connect(self.open_settings)
        self.title_bar.sceneResetRequested.connect(self._reset_scene)

    def _install_shortcuts(self) -> None:
        bindings = {
            "R": self.controller.reset_view,
            "F": self._focus_selection,
            "Space": self._toggle_tour,
            "Ctrl+S": self._take_snapshot,
            "Ctrl+R": self._reset_scene,
            "Ctrl+,": self.open_settings,
            "Ctrl+K": lambda: self.layer_panel._search.setFocus(),  # noqa: SLF001
            "Esc": self.controller.clear_selection,
            "Ctrl+1": lambda: self.controller.set_view("anterior"),
            "Ctrl+2": lambda: self.controller.set_view("posterior"),
            "Ctrl+3": lambda: self.controller.set_view("left"),
            "Ctrl+4": lambda: self.controller.set_view("right"),
            "Ctrl+5": lambda: self.controller.set_view("superior"),
            "Ctrl+6": lambda: self.controller.set_view("isometric"),
        }
        for sequence, slot in bindings.items():
            shortcut = QShortcut(QKeySequence(sequence), self)
            shortcut.activated.connect(slot)

    # ============================================================== startup
    def _startup(self) -> None:
        # Populate the layer list from the registry before loading geometry.
        self.layer_panel.populate(self.registry.specs, self.registry.available_ids())
        self.controller.load_scene(allow_placeholders=True)

        # Audio options (voice enumeration can be slow, so defer it).
        self._refresh_audio_options()
        QTimer.singleShot(400, self._load_voices)

        # Kick off AI detection in the background.
        self.ai.refresh_detection()

        self._update_scene_summary()

        # Prime the export panel: script preview + expected video length.
        first_tour = get_tour(self.ai_panel.current_video_tour_id())
        if first_tour is not None:
            self.ai_panel.set_cues(first_tour.cues())
        self._update_video_estimate()

    def _load_voices(self) -> None:
        self._refresh_audio_options(reload_voices=True)
        if not self.audio.voices():
            self.ai_panel.append_note(
                "No system voices found. Install pyttsx3 (pip install pyttsx3) for "
                "offline narration, or add an ElevenLabs key for cloud voices.")

    def _refresh_audio_options(self, *, reload_voices: bool = False) -> None:
        """Push the voice catalogue and the resolved selection into the panel.

        Single place where dialect/gender/voice state reaches the UI, so the
        sidebar, the settings dialog and the export card can never disagree
        about which voice will actually narrate.
        """
        voices = self.audio.voices(refresh=reload_voices)
        voice, note = self.audio.resolve_selection(apply=False)
        self.ai_panel.set_audio_options(
            backends=self.audio.available_backends(),
            voices=voices,
            backend=str(self.config.get("audio.backend", "pyttsx3")),
            voice=voice.id if voice else "",
            rate=int(self.config.get("audio.rate", 175)),
            volume=int(float(self.config.get("audio.volume", 0.9)) * 100),
            dialects=self.audio.dialect_options(),
            dialect=self.audio.current_dialect(),
            gender=self.audio.current_gender(),
            note=note,
        )

    # ============================================================ AI events
    def _on_detection(self, report) -> None:
        hardware = report.hardware
        local = report.local_models()
        inventory = report.local_model_count()

        if report.available_backends():
            labels = ", ".join(b.label for b in report.available_backends())
            self.ai_panel.set_status(f"Live: {labels}", "ok")
        elif inventory:
            self.ai_panel.set_status(
                f"{inventory} local model(s) on disk but no server is running.", "warn")
        else:
            self.ai_panel.set_status(
                "No local models found. Start Ollama/LM Studio or add a cloud API key.",
                "off")

        self.ai_panel.set_hardware_summary(hardware.summary())
        self.ai_panel.set_models(self.ai.available_models(),
                                 self.ai.current_provider.id if self.ai.current_provider else "",
                                 self.ai.current_model)
        self._status_provider.setText(self.ai.status_line())

        notes = " ".join(hardware.notes)
        local_desc = f"{len(local)} live" if local else f"{inventory} on disk"
        self.status_message(
            f"Model scan complete in {report.elapsed_s:.2f}s — local: {local_desc}."
            + (f" {notes}" if notes else ""),
            7000,
        )

    def _on_provider_changed(self, provider_id: str, model_id: str) -> None:
        self._status_provider.setText(self.ai.status_line())
        self.ai_panel.set_models(self.ai.available_models(), provider_id, model_id)

    def _on_response_started(self, provider: str, model: str) -> None:
        self.ai_panel.set_busy(True)
        self.ai_panel.start_assistant(provider, model)

    def _on_response_finished(self, _text: str, seconds: float) -> None:
        self.ai_panel.set_busy(False)
        self.ai_panel.finish_assistant(seconds)

    def _on_response_failed(self, message: str) -> None:
        self.ai_panel.set_busy(False)
        self.ai_panel.append_error(message)
        self.status_message("AI request failed — see the assistant panel.", 6000)

    def _submit_prompt(self, text: str) -> None:
        self.ai_panel.append_user(text)
        if not self.ai.ask(text):
            self.ai_panel.set_busy(False)

    def _quick_action(self, key: str) -> None:
        actions = {
            "explain": lambda: self.ai.explain_selection("standard"),
            "conditions": self.ai.ask_conditions,
            "relations": self.ai.ask_relations,
            "physiology": self.ai.ask_physiology,
            "quiz": self.ai.ask_quiz,
        }
        handler = actions.get(key)
        if handler is None:
            return
        label = {
            "explain": f"Explain {self.controller.selected_label or 'the visible anatomy'}",
            "conditions": "Which conditions affect this?",
            "relations": "Describe the anatomical relations.",
            "physiology": "Explain the physiology.",
            "quiz": "Quiz me on this.",
        }.get(key, key)
        self.ai_panel.append_user(label)
        if not handler():
            self.ai_panel.set_busy(False)

    def _clear_chat(self) -> None:
        self.ai.clear_history()
        self.ai_panel.clear_chat()

    # ======================================================== scene events
    def _sync_subject_to_layer(self, layer_id: str) -> None:
        """Point the video exporter at the structure the user is looking at.

        This is what makes "select the liver → generate" produce a liver video:
        the export subject follows the selection instead of a stale dropdown.
        """
        if not layer_id:
            return
        tour = subject_for_layer(layer_id)
        if tour is not None and self.ai_panel.select_video_tour(tour.id):
            self.ai_panel.set_cues(tour.cues())
            self.status_message(
                f"Video subject: {tour.title} — its own narration script", 3500)
            self._update_video_estimate()

    def _on_structure_picked(self, layer_id: str, label: str) -> None:
        if layer_id:
            self._sync_subject_to_layer(layer_id)
        if not layer_id:
            self.controller.clear_selection()
            return
        self.controller.select_structure(layer_id, label)
        if layer_id == "muscles":
            self.simulation_panel.follow_structure(label)
        if bool(self.config.get("ui.auto_explain_on_select", False)):
            self.ai_panel.append_user(f"Explain {label}")
            self.ai.explain_selection("brief")

    def _on_structure_changed(self, layer_id: str, label: str) -> None:
        self.layer_panel.select_layer(layer_id)
        self.viewport_panel.set_selection(label)
        self._update_scene_summary()

    def _on_layer_state_changed(self, layer_id: str, prop: str, value) -> None:
        if prop == "opacity":
            self.layer_panel.set_layer_opacity(layer_id, float(value))
        elif prop == "visible":
            self.layer_panel.set_layer_visible(layer_id, bool(value))

    def _on_scene_loaded(self, real: int, placeholders: int) -> None:
        self._rig = None                       # joints are re-estimated lazily
        self._disease_scene = None             # baselines belong to the old meshes
        self.layer_panel.populate(self.registry.specs, self.registry.available_ids())
        self.simulation_panel.set_rig_status(self.rig() is not None)
        self._update_scene_summary()
        if placeholders and not real:
            self.ai_panel.append_note(
                "Showing placeholder geometry. Run "
                "`python tools\\generate_demo_models.py` or drop real anatomy models "
                "into app/assets/models to replace it.")

    # ============================================================ simulation
    def rig(self):
        """Kinematic rig for the loaded scene (``None`` without named limb bones)."""
        if self._rig is None and getattr(self.viewport, "has_structure_names", lambda: False)():
            from app.anatomy.kinematics import Rig
            rig = Rig(self.viewport)
            self._rig = rig if rig.available else False
        return self._rig or None

    def _simulation_tour(self, kind: str, key: str, side: str):
        if kind == "muscle":
            from app.anatomy.muscle_tours import muscle_tour
            return muscle_tour(self.viewport, self.rig(), key, side)
        if kind == "motion":
            from app.anatomy.muscle_tours import motion_tour
            return motion_tour(self.viewport, self.rig(), key, side)
        if kind == "disease":
            from app.disease.tours import disease_tour
            self._reset_disease()
            return disease_tour(self.viewport, self.disease_scene(), key)
        return None

    # ------------------------------------------------------------ disease
    def disease_scene(self):
        if self._disease_scene is None:
            from app.disease.effects import DiseaseScene
            self._disease_scene = DiseaseScene(self.viewport)
        return self._disease_scene

    def _on_disease_stage(self, condition_id: str, index: int) -> None:
        """Show one stage statically (the Stage slider)."""
        from app.core.effects import IsolateStructuresEffect
        from app.disease.catalog import CONDITION_BY_ID
        from app.disease.effects import blend_stages
        from app.disease.tours import _layers, involved
        condition = CONDITION_BY_ID.get(condition_id)
        if condition is None:
            return
        self.controller.stop_tour()
        scene = self.disease_scene()
        if condition_id != self._disease_condition:
            self._reset_disease()
            self._disease_condition = condition_id
            for layer_id, opacity in _layers(condition).items():
                self.controller.set_layer_visible(layer_id, opacity > 0)
                if opacity > 0:
                    self.controller.set_layer_opacity(layer_id, opacity)
            names = involved(scene, condition)
            if condition.isolate:
                self._disease_isolation = IsolateStructuresEffect(names, ghost=0.10, context_layers=())
                self._disease_isolation.begin(self.viewport)
                self._disease_isolation.apply(self.viewport, 0.0)
            focus = scene.matching(condition.focus)
            if focus:
                self.controller.set_view(condition.view)
                self.viewport.focus_structures(focus)
        scene.apply(blend_stages(condition, index, index, 1.0))
        stage = condition.stages[index]
        self.status_message(f"{condition.name}: {stage.label} — {stage.title}", 6000)

    def _play_disease_tour(self, condition_id: str) -> None:
        self._start_generated_tour(self._simulation_tour("disease", condition_id, ""), "Disease")

    def _reset_disease(self) -> None:
        if self._disease_isolation is not None:
            self._disease_isolation.end(self.viewport)
            self._disease_isolation = None
        if self._disease_scene is not None:
            self._disease_scene.reset()
        self._disease_condition = ""

    def _start_generated_tour(self, tour, group: str) -> None:
        if tour is None:
            self.status_message("That simulation is not available for the loaded anatomy.", 5000)
            return
        self.ai_panel.add_video_subject(group, tour.title, tour)
        self.ai_panel.set_cues(tour.cues())
        self.controller.start_tour(tour.id)

    def _play_muscle_tour(self, key: str, side: str) -> None:
        self._start_generated_tour(self._simulation_tour("muscle", key, side), "Muscle")

    def _play_motion_tour(self, motion_id: str, side: str) -> None:
        self._start_generated_tour(self._simulation_tour("motion", motion_id, side), "Motion")

    def _export_simulation(self, kind: str, key: str, side: str) -> None:
        tour = self._simulation_tour(kind, key, side)
        if tour is None:
            self.status_message("That simulation is not available for the loaded anatomy.", 5000)
            return
        self.controller.stop_tour()
        self.ai_panel.add_video_subject(kind.capitalize(), tour.title, tour)
        self.ai_panel.show_tab("media")
        self._generate_video(tour.id, self.ai_panel.current_video_preset())

    # ============================================================ language
    def _localize(self, tour, then):
        """Tour in the narration language, translating first if needed.

        Returns the localized tour when it is ready now, else ``None`` and calls
        ``then(localized)`` on the GUI thread when translation finishes."""
        from app.audio.languages import language
        from app.i18n.translate import localized_tour, tour_sources
        code = self.audio.current_language()
        lang = language(code)
        if lang.is_english or "@" in tour.id or self.audio.engine() == "none":
            return tour
        memory = self.translator.memory(code)
        if not memory.missing(tour_sources(tour)):
            return localized_tour(tour, code, memory)
        if self._translation_worker is not None and self._translation_worker.isRunning():
            self.status_message("A translation is already running…", 4000)
            return None
        ok, reason = self.translator.ready()
        if not ok:
            QMessageBox.warning(self, "Translation needed",
                                f"Narration in {lang.name} needs translating first.\n\n{reason}")
            return None
        from app.i18n.worker import TranslationWorker
        self._pending_localization = (tour, code, then)
        self._translation_worker = TranslationWorker(self.translator, tour_sources(tour), code, parent=self)
        self._translation_worker.progress.connect(self._on_translation_progress)
        self._translation_worker.done.connect(self._on_translation_done)
        self.status_message(f"Translating narration to {lang.name}…", 0)
        self._translation_worker.start()
        return None

    def _on_translation_progress(self, percent: int, message: str) -> None:
        self.status_message(f"{message} ({percent}%)", 0)

    def _on_translation_done(self, ok: bool, message: str) -> None:
        pending, self._pending_localization = self._pending_localization, None
        if not ok or pending is None:
            self.status_message(f"Translation failed: {message}" if message else "Translation cancelled.", 8000)
            return
        from app.i18n.translate import localized_tour
        tour, code, then = pending
        self.status_message(f"Narration translated. Translations are saved in "
                            f"{self.translator.folder} for review.", 8000)
        then(localized_tour(tour, code, self.translator.memory(code)))

    # ================================================================ MRI
    def _show_mri_pane(self, show: bool) -> None:
        self.slice_pane.setVisible(bool(show))
        self.simulation_panel.set_mri_pane_checked(bool(show))
        if show:
            total = max(800, self.center_splitter.width())
            self.center_splitter.setSizes([int(total * 0.58), int(total * 0.42)])
            volume = self.slice_pane.volume
            axis, _pos = self.viewport.clip_state()
            if volume is not None and not axis:
                b = volume.bounds()
                self.viewport.apply_clip_world("z", (b[4] + b[5]) / 2)
            self._face_cut(self.viewport.clip_state()[0])

    def _face_cut(self, axis: str) -> None:
        """Turn the 3D camera to the cut face, as a radiologist views the slice:
        axial from the feet, coronal from the front, sagittal from the side."""
        preset = {"z": "inferior", "y": "anterior", "x": "left"}.get(axis)
        if preset:
            self.controller.set_view(preset)

    def _on_slice_plane(self, axis: str, position: float) -> None:
        changed = axis != self.viewport.clip_state()[0]
        self.viewport.apply_clip_world(axis, position)
        if changed:
            self._face_cut(axis)

    def _simulate_mri(self, spacing: float) -> None:
        if self._mri_worker is not None and self._mri_worker.isRunning():
            self.status_message("MRI simulation already running…", 3000)
            return
        from app.imaging.synthetic_mri import scene_items
        from app.imaging.worker import MRISimulationWorker
        self.controller.stop_tour()
        items = scene_items(self.viewport)
        self._mri_worker = MRISimulationWorker(items, spacing, parent=self)
        # Bound methods only: the slots must run on the GUI thread.
        self._mri_worker.progress.connect(self._on_mri_progress)
        self._mri_worker.completed.connect(self._on_mri_ready)
        self._mri_worker.failed.connect(self._on_mri_failed)
        self.simulation_panel.set_mri_status("Simulating MRI…")
        self._mri_worker.start()

    def _on_mri_progress(self, percent: int, message: str) -> None:
        self.simulation_panel.set_mri_status(f"{percent}% · {message}")

    def _on_mri_ready(self, simulator, volume) -> None:
        self._mri_sim = simulator
        self.slice_pane.set_sequence("T1")
        self.slice_pane.set_volume(volume)
        self.simulation_panel.set_mri_status(
            f"{volume.modality}. {volume.description}. Simulated for teaching — not patient data.")
        self._show_mri_pane(True)

    def _on_mri_failed(self, message: str) -> None:
        self.simulation_panel.set_mri_status(message)
        self.status_message(message, 6000)

    def _on_mri_sequence(self, sequence: str) -> None:
        if self._mri_sim is None or self.slice_pane.volume is None or not self.slice_pane.volume.simulated:
            return
        offset = self.slice_pane.volume.offset.copy()
        volume = self._mri_sim.render(sequence)
        volume.offset = offset
        self.slice_pane.set_volume(volume)

    def _load_scan(self, path: str) -> None:
        from app.imaging.worker import ScanLoadWorker
        self.simulation_panel.set_mri_status(f"Loading {path} …")
        self._scan_worker = ScanLoadWorker(path, parent=self)
        self._scan_worker.completed.connect(self._on_scan_loaded)
        self._scan_worker.failed.connect(self._on_mri_failed)
        self._scan_worker.start()

    def _on_scan_loaded(self, volume) -> None:
        from app.imaging.volume import align_to_region, guess_region
        region = align_to_region(volume, self.viewport, guess_region(volume))
        self.slice_pane.set_volume(volume)
        self.simulation_panel.set_mri_status(
            f"{volume.description}. Placed over: {region} (use Align in the MRI pane to change).")
        self._show_mri_pane(True)

    def _on_mri_structure_clicked(self, name: str) -> None:
        hit = self.viewport.locate_structure(name)
        if hit is None:
            return
        layer_id = hit[0]
        self.controller.select_structure(layer_id, name)
        if layer_id == "muscles":
            self.simulation_panel.follow_structure(name)

    def _structure_at_point(self, point) -> str:
        """Smallest model structure whose bounds contain *point* (real-scan hover)."""
        best = None
        for state in self.viewport.layers.values():
            if not state.visible:
                continue
            for name, poly in zip(state.structures, state.polydata):
                if not name:
                    continue
                b = poly.GetBounds()
                if b[0] <= point[0] <= b[1] and b[2] <= point[1] <= b[3] and b[4] <= point[2] <= b[5]:
                    size = (b[1] - b[0]) * (b[3] - b[2]) * (b[5] - b[4])
                    if best is None or size < best[0]:
                        best = (size, name)
        return f"≈ {best[1]} (model)" if best else ""

    def _update_scene_summary(self) -> None:
        try:
            loaded = len(self.viewport.layers)
        except Exception:
            loaded = 0
        selection = self.controller.selected_label or "nothing selected"
        self.title_bar.set_scene_summary(f"{loaded} layers · {selection}")
        self._status_scene.setText(f"{loaded} layers loaded")

    # ========================================================== tour events
    def _toggle_tour(self) -> None:
        if self.controller.tour.is_running:
            self.controller.stop_tour()
            self.layer_panel.set_playing(False)
            self.viewport_panel.set_tour_state(False)
        else:
            self.controller.start_tour(self.layer_panel.current_tour_id())

    def _step_tour(self, direction: int) -> None:
        if direction > 0:
            self.controller.next_tour_step()
        else:
            self.controller.previous_tour_step()

    def _on_tour_started(self, _tour_id: str, title: str) -> None:
        tour = self.controller.tour.tour
        self.layer_panel.set_playing(True)
        self.viewport_panel.set_tour_state(True, title)
        if tour is not None:
            self.ai_panel.set_cues(tour.cues())
        self.status_message(f"Playing: {title}", 4000)

    def _on_tour_finished(self, _tour_id: str) -> None:
        self.layer_panel.set_playing(False)
        self.viewport_panel.set_tour_state(False)

    def _replay_cue(self, index: int) -> None:
        tour = self.controller.tour.tour
        if tour is None or not (0 <= index < len(tour.keyframes)):
            return
        self.controller.tour.jump_to(index)
        self.audio.speak(tour.keyframes[index].narration, index=index)

    # ========================================================= audio events
    def _on_audio_enabled(self, enabled: bool) -> None:
        self.config["audio.enabled"] = bool(enabled)
        if not enabled:
            self.audio.stop()

    def _on_audio_backend(self, backend: str) -> None:
        self.config["audio.backend"] = backend
        self.audio.apply_settings()
        self._refresh_audio_options()
        ok, reason = self.audio.engine_status()
        if not ok and reason:
            self.ai_panel.append_note(reason)
        if backend == "elevenlabs" and not self.config.api_key("elevenlabs"):
            self.ai_panel.append_note(
                "ElevenLabs selected but no API key is set — add one in Settings.")
        elif backend == "pyttsx3" and not self.audio.is_available():
            self.ai_panel.append_note(
                "pyttsx3 is not installed. Run: pip install pyttsx3")

    def _on_audio_voice(self, voice_id: str) -> None:
        self.audio.set_voice(voice_id)
        self._refresh_audio_options()
        self._update_video_estimate()

    def _on_audio_dialect(self, code: str) -> None:
        """Switch the narration dialect for live voiceover AND video export."""
        self.audio.set_dialect(code)
        self._refresh_audio_options()
        self._update_video_estimate()
        summary = self.audio.selection_summary()
        self.status_message(summary, 6000)

    def _on_audio_gender(self, gender: str) -> None:
        self.audio.set_gender(gender)
        self._refresh_audio_options()
        self._update_video_estimate()
        self.status_message(self.audio.selection_summary(), 6000)

    def _show_voice_help(self, dialect_code: str) -> None:
        QMessageBox.information(self, "Add a narration voice",
                                install_help(dialect_code or "en-IN"))

    def _on_system_selected(self, label: str) -> None:
        """A body system was chosen in the sidebar.

        Applies that system's layer visibility *and* points the video exporter at
        the matching scripted tour. Keeping the two in step is what stops a
        "venous system" export from narrating the skeleton.
        """
        self.controller.apply_preset(label)
        tour = tour_for_system(label)
        if tour is not None:
            if self.ai_panel.select_video_tour(tour.id):
                self.status_message(
                    f"Video subject: {tour.title} — its own narration script", 4000)
            self.ai_panel.set_cues(tour.cues())
        self._update_video_estimate()

    def _on_audio_rate(self, rate: int) -> None:
        self.config["audio.rate"] = int(rate)
        self.audio.apply_settings()

    def _on_audio_volume(self, volume: int) -> None:
        self.config["audio.volume"] = int(volume) / 100.0
        self.audio.apply_settings()

    # ======================================================== video export
    def _update_video_estimate(self) -> None:
        """Refresh the "estimated length" line under the export controls."""
        ok, message = self.video.prerequisites_ok()
        if not ok:
            self.ai_panel.set_video_estimate(0, enabled=False, detail=f"⚠ {message}")
            return

        tour = get_tour(self.ai_panel.current_video_tour_id())
        if tour is None:
            self.ai_panel.set_video_estimate(0, enabled=False, detail="No tour available.")
            return

        width, height = VIDEO_PRESETS.get(self.ai_panel.current_video_preset(),
                                          (1920, 1080))
        options = self.ai_panel.video_options()
        settings = VideoSettings(width=width, height=height, fps=30,
                                 supersample=options["supersample"],
                                 ssao=options["ssao"])
        self.ai_panel.set_video_estimate(
            self.video.estimate_seconds(tour, settings),
            voice_hint=f"{self.audio.selection_summary()}\n{settings.describe()}")

    def _open_video_export(self) -> None:
        """Menu entry: jump to the media tab and start rendering immediately."""
        self.ai_panel.show_tab("media")
        self._generate_video(self.ai_panel.current_video_tour_id(),
                             self.ai_panel.current_video_preset())

    def _generate_video(self, tour_id: str, preset_key: str = "1080p") -> None:
        tour = get_tour(tour_id)
        if tour is None:
            self.status_message("Select a guided tour first.", 4000)
            return
        tour = self._localize(tour, lambda local: self._generate_video(local.id, preset_key))
        if tour is None:
            return                           # translating first; export resumes when done
        if self.video.is_running:
            self.status_message("A video export is already running.", 4000)
            return

        options = self.ai_panel.video_options()
        width, height = VIDEO_PRESETS.get(preset_key, (1920, 1080))
        settings = VideoSettings(
            width=width, height=height, fps=30,
            burn_captions=options["burn_captions"],
            include_audio=options["include_audio"],
            supersample=options["supersample"],
            ssao=options["ssao"],
        )

        self.ai_panel.set_video_busy(True, percent=0, message="Preparing narration…")
        self.status_message(
            f"Rendering “{tour.title}” — {settings.describe()}", 0)
        meta = {
            "dialect": self.audio.current_dialect(),
            "gender": self.audio.current_gender(),
            "voice": (self.audio.resolve_selection(apply=False)[0] or
                      type("V", (), {"name": "none"})()).name,
            "app": __import__("app").__version__,
        }
        if not self.video.start(tour, settings, meta=meta):
            self.ai_panel.set_video_busy(False, message="Could not start the export.")
            self.ai_panel.set_video_error("Export could not start.")

    def _cancel_video(self) -> None:
        if self.video.is_running:
            self.video.cancel()

    def _on_video_progress(self, percent: int, message: str) -> None:
        self.ai_panel.set_video_busy(True, percent=percent, message=message)
        self.status.showMessage(f"Video export — {message}")

    def _on_video_finished(self, path: str) -> None:
        self.ai_panel.set_video_busy(False)
        self.ai_panel.set_video_result(path, message=f"Ready: {Path(path).name}")
        self.status_message(f"Video saved: {path}", 8000)
        # The user asked for "generate and play" — open the player immediately.
        self._play_video(path)

    def _on_video_failed(self, message: str) -> None:
        self.ai_panel.set_video_busy(False)
        self.ai_panel.set_video_error(message)
        self.status_message(f"Video export failed: {message}", 9000)

    def _on_video_cancelled(self) -> None:
        self.ai_panel.set_video_busy(False, message="Export cancelled.")
        self.status_message("Video export cancelled", 4000)

    def _play_video(self, path: str) -> None:
        if not path or not Path(path).exists():
            self.status_message("No video to play yet — generate one first.", 5000)
            return
        dialog = VideoPlayerDialog(Path(path), title="Guided tour video", parent=self)
        dialog.exportAgainRequested.connect(
            lambda: (dialog.accept(), self._open_video_export()))
        dialog.exec()

    def _show_video_in_folder(self, path: str) -> None:
        target = Path(path) if path else (self.config.paths.user_data / "videos")
        self._reveal(target)

    @staticmethod
    def _reveal(path: Path) -> None:
        """Show a file or folder in the OS file browser."""
        path = Path(path)
        try:
            if sys.platform == "win32":
                if path.is_dir():
                    subprocess.Popen(["explorer", str(path)])
                else:
                    subprocess.Popen(["explorer", "/select,", str(path)])
            elif sys.platform == "darwin":
                subprocess.Popen(["open", "-R", str(path)])
            else:
                subprocess.Popen(["xdg-open",
                                  str(path if path.is_dir() else path.parent)])
        except Exception:
            pass

    # ============================================================== actions
    def _focus_selection(self) -> None:
        if self.controller.selected_id:
            self.viewport.focus_layer(self.controller.selected_id)
        else:
            self.viewport.reset_camera()

    def _take_snapshot(self) -> None:
        stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        target = self.config.paths.user_data / "screenshots" / f"biohuman3d_{stamp}.png"
        self.controller.snapshot(target, magnification=2)

    def _reset_scene(self) -> None:
        self.controller.stop_tour()
        self.controller.clear_selection()
        self.controller.show_all()
        self.controller.reset_view()
        self.layer_panel._clip_slider.set_value(0)      # noqa: SLF001
        self.layer_panel._explode_slider.set_value(0)   # noqa: SLF001
        self.viewport.set_explode(0.0)
        self.viewport.disable_clip_plane()
        self.status_message("Scene reset", 2500)

    def open_settings(self) -> None:
        dialog = SettingsDialog(self.config, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.status_message("Settings saved", 3000)
            self.ai.refresh_detection()
            self._refresh_audio_options(reload_voices=True)

    # ============================================================== helpers
    def status_message(self, text: str, timeout_ms: int = 4000) -> None:
        """Show a transient message in the status bar."""
        self.status.showMessage(text, timeout_ms)

    # ============================================================ lifecycle
    def closeEvent(self, event) -> None:  # noqa: N802
        try:
            sizes = self.splitter.sizes()
            if len(sizes) == 3:
                self.config["ui.left_width"] = sizes[0]
                self.config["ui.right_width"] = sizes[2]
            self._save_geometry()
            self.config.save()
        except Exception:
            pass

        if self.video.is_running:
            self.video.cancel()
        if self.controller.tour.is_running:
            self.controller.stop_tour()
        self.ai.cancel()
        self.audio.shutdown()

        # Give the workers a bounded moment to unwind.
        for worker in (self.ai._detection, self.ai._chat):  # noqa: SLF001
            try:
                if worker is not None and worker.isRunning():
                    worker.wait(1200)
            except Exception:
                pass

        super().closeEvent(event)
