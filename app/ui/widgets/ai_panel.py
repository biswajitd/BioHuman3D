"""
Right panel — **AI Assistant · Audio & Video**.

Two *tabs* rather than one long scrolling column. On a laptop-height display a
single column pushed the audio and video controls below the fold where they
could not be reached; splitting the panel guarantees each feature has the full
panel height available.

    [ Assistant ]  [ Audio & Video ]
    ├─ Assistant        model routing, streaming transcript, quick actions
    └─ Audio & Video    voice settings, narration script, tour video export

The panel is presentational only: every action is emitted as a signal and
handled by the AI manager, the audio manager or the video exporter.
"""
from __future__ import annotations

import re
from typing import List, Optional, Sequence

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QTextCursor
from PyQt6.QtWidgets import (QCheckBox, QComboBox, QHBoxLayout, QLabel,
                             QProgressBar, QPushButton, QScrollArea,
                             QSizePolicy, QStackedWidget, QTextEdit,
                             QVBoxLayout, QWidget)

from app.audio.languages import ENGINES
from app.audio.scripts import get_tour, video_subjects
from app.ui.widgets.collapsible import CollapsibleSection
from app.ui.widgets.controls import (GLYPH, ChipButton, IconButton,
                                     LabeledSlider, SegmentBar, StatusDot,
                                     panel_header)

#: Export resolution presets: key -> (width, height)
VIDEO_PRESETS = {
    "720p": (1280, 720),
    "1080p": (1920, 1080),
    "1440p": (2560, 1440),
    "4K": (3840, 2160),
}


class AIPanel(QWidget):
    """Medical-assistant chat, narration settings and tour video export."""

    # -- assistant ---------------------------------------------------------
    modeChanged = pyqtSignal(str)                 # auto | local | cloud
    modelSelected = pyqtSignal(str, str)          # provider_id, model_id
    refreshRequested = pyqtSignal()
    promptSubmitted = pyqtSignal(str)
    quickActionRequested = pyqtSignal(str)
    cancelRequested = pyqtSignal()
    clearChatRequested = pyqtSignal()
    readAloudRequested = pyqtSignal(str)
    openSettingsRequested = pyqtSignal()

    # -- audio -------------------------------------------------------------
    audioEnabledChanged = pyqtSignal(bool)
    audioBackendChanged = pyqtSignal(str)
    audioVoiceChanged = pyqtSignal(str)
    audioDialectChanged = pyqtSignal(str)
    audioGenderChanged = pyqtSignal(str)
    audioRateChanged = pyqtSignal(int)
    audioVolumeChanged = pyqtSignal(int)
    stopAudioRequested = pyqtSignal()
    replayCueRequested = pyqtSignal(int)
    voiceSetupRequested = pyqtSignal(str)          # dialect code needing setup

    # -- video -------------------------------------------------------------
    videoGenerateRequested = pyqtSignal(str, str)   # tour_id, preset key
    videoCancelRequested = pyqtSignal()
    videoPlayRequested = pyqtSignal(str)            # path
    videoFolderRequested = pyqtSignal(str)          # path
    videoOptionsChanged = pyqtSignal()

    def __init__(self, tours: Sequence = (), parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("SidePanelRight")
        self.setMinimumWidth(320)
        self.setMaximumWidth(560)
        # Keep the vertical floor low so the window can always fit the screen;
        # the contents scroll instead of pushing the status bar off-screen.
        self.setMinimumHeight(150)

        self._streaming = False
        self._last_answer = ""
        self._answer_start = 0
        self._cue_rows: List[QPushButton] = []
        self._video_ready = False

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(panel_header(
            "AI Assistant · Audio & Video",
            "Answers are grounded in the structure you are viewing",
        ))

        # -- tab switcher ----------------------------------------------------
        switcher = QWidget()
        switcher_layout = QHBoxLayout(switcher)
        switcher_layout.setContentsMargins(12, 0, 12, 8)
        self._tabs = SegmentBar(
            [("assistant", "Assistant"), ("media", "Audio & Video"), ("simulate", "Simulate")],
            current="assistant")
        self._tabs.changed.connect(self._on_tab_changed)
        switcher_layout.addWidget(self._tabs)
        root.addWidget(switcher)

        # -- pages -----------------------------------------------------------
        self._stack = QStackedWidget()
        # Without this the stacked widget adopts the tallest page's minimum as its
        # own, and a QStackedWidget reports the MAXIMUM over all pages — so the
        # hidden Audio & Video page would set a floor the window cannot shrink past.
        self._stack.setMinimumHeight(0)
        self._stack.addWidget(self._scroll_page([
            self._build_model_section(),
            self._build_chat_section(),
        ]))
        self._stack.addWidget(self._scroll_page([
            self._build_audio_section(),
            self._build_video_section(),
            self._build_script_section(),
        ]))
        # Third page: filled by MainWindow with the simulation panel (muscle
        # actions, disease progression, MRI cross-sections).
        self._simulate_holder = QWidget()
        self._simulate_layout = QVBoxLayout(self._simulate_holder)
        self._simulate_layout.setContentsMargins(0, 0, 0, 0)
        self._stack.addWidget(self._simulate_holder)
        root.addWidget(self._stack, 1)

    # ------------------------------------------------------------------ util
    @staticmethod
    def _scroll_page(widgets: Sequence[QWidget]) -> QWidget:
        """Stack widgets inside a vertical scroll area with the section styling."""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        # Let the page compress and scroll instead of imposing a height floor.
        scroll.setMinimumHeight(120)
        holder = QWidget()
        layout = QVBoxLayout(holder)
        layout.setContentsMargins(10, 0, 10, 10)
        layout.setSpacing(9)
        for widget in widgets:
            layout.addWidget(widget)
        layout.addStretch(1)
        scroll.setWidget(holder)
        return scroll

    _PAGES = {"assistant": 0, "media": 1, "simulate": 2}

    def _on_tab_changed(self, key: str) -> None:
        self._stack.setCurrentIndex(self._PAGES.get(key, 0))

    def set_simulation_widget(self, widget: QWidget) -> None:
        """Install the simulation page (built by the main window)."""
        while self._simulate_layout.count():
            item = self._simulate_layout.takeAt(0)
            if item.widget() is not None:
                item.widget().setParent(None)
        self._simulate_layout.addWidget(widget)

    def show_tab(self, key: str) -> None:
        self._tabs.set_current(key, emit=False)
        self._on_tab_changed(key)

    # =================================================================== AI
    def _build_model_section(self) -> CollapsibleSection:
        section = CollapsibleSection("Model Routing", expanded=True)

        mode_row = QHBoxLayout()
        self._mode = SegmentBar(
            [("auto", "Auto"), ("local", "Local"), ("cloud", "Cloud")], current="auto")
        self._mode.changed.connect(self.modeChanged)
        refresh = IconButton(GLYPH["refresh"], "Rescan for local models and API keys")
        refresh.clicked.connect(self.refreshRequested)
        mode_row.addWidget(self._mode, 1)
        mode_row.addWidget(refresh)
        section.add_layout(mode_row)

        status_row = QHBoxLayout()
        status_row.setSpacing(7)
        self._dot = StatusDot("off")
        self._status = QLabel("Scanning local models…")
        self._status.setObjectName("LayerMeta")
        self._status.setWordWrap(True)
        status_row.addWidget(self._dot)
        status_row.addWidget(self._status, 1)
        section.add_layout(status_row)

        self._model_combo = QComboBox()
        self._model_combo.setToolTip("Select the model that answers your questions")
        self._model_combo.currentIndexChanged.connect(self._on_model_changed)
        section.add_widget(self._model_combo)

        self._hardware = QLabel("")
        self._hardware.setObjectName("LayerMeta")
        self._hardware.setWordWrap(True)
        section.add_widget(self._hardware)

        configure = ChipButton("Configure API keys…", "Open settings for cloud providers")
        configure.clicked.connect(self.openSettingsRequested)
        section.add_widget(configure)

        self._model_section = section
        return section

    def _build_chat_section(self) -> CollapsibleSection:
        section = CollapsibleSection("Assistant", expanded=True)

        self._chat_view = QTextEdit()
        self._chat_view.setObjectName("ChatView")
        self._chat_view.setReadOnly(True)
        self._chat_view.setMinimumHeight(130)
        self._chat_view.setSizePolicy(QSizePolicy.Policy.Expanding,
                                      QSizePolicy.Policy.Expanding)
        section.add_widget(self._chat_view)

        chips_top = QHBoxLayout()
        chips_top.setSpacing(5)
        for label, key in (("Explain selection", "explain"),
                           ("Common conditions", "conditions"),
                           ("Relations", "relations")):
            chip = ChipButton(label)
            chip.clicked.connect(lambda _c=False, k=key: self.quickActionRequested.emit(k))
            chips_top.addWidget(chip)
        section.add_layout(chips_top)

        chips_bottom = QHBoxLayout()
        chips_bottom.setSpacing(5)
        for label, key in (("Physiology", "physiology"), ("Quiz me", "quiz")):
            chip = ChipButton(label)
            chip.clicked.connect(lambda _c=False, k=key: self.quickActionRequested.emit(k))
            chips_bottom.addWidget(chip)
        read_aloud = ChipButton(f"{GLYPH['speaker']} Read aloud")
        read_aloud.clicked.connect(lambda: self.readAloudRequested.emit(self._last_answer))
        chips_bottom.addWidget(read_aloud)
        section.add_layout(chips_bottom)

        self._input = QTextEdit()
        self._input.setObjectName("ChatInput")
        self._input.setPlaceholderText(
            "Ask about the selected structure…  (Enter to send, Shift+Enter for a new line)")
        self._input.setFixedHeight(62)
        self._input.installEventFilter(self)
        section.add_widget(self._input)

        send_row = QHBoxLayout()
        send_row.setSpacing(6)
        clear = ChipButton("Clear")
        clear.clicked.connect(self.clearChatRequested)
        self._send = QPushButton(f"{GLYPH['send']}  Ask")
        self._send.setObjectName("PrimaryButton")
        self._send.clicked.connect(self._emit_prompt)
        self._cancel = ChipButton("Stop")
        self._cancel.clicked.connect(self.cancelRequested)
        self._cancel.setVisible(False)
        send_row.addWidget(clear)
        send_row.addStretch(1)
        send_row.addWidget(self._cancel)
        send_row.addWidget(self._send)
        section.add_layout(send_row)

        self._chat_section = section
        return section

    # ================================================================ AUDIO
    def _build_audio_section(self) -> CollapsibleSection:
        section = CollapsibleSection("Voice & Narration", expanded=True)

        top = QHBoxLayout()
        top.setSpacing(6)
        self._audio_toggle = ChipButton(f"{GLYPH['speaker']}  Voiceover on")
        self._audio_toggle.setCheckable(True)
        self._audio_toggle.setChecked(True)
        self._audio_toggle.toggled.connect(self._on_audio_toggled)
        stop = IconButton(GLYPH["stop"], "Stop speaking")
        stop.clicked.connect(self.stopAudioRequested)
        top.addWidget(self._audio_toggle)
        top.addStretch(1)
        top.addWidget(stop)
        section.add_layout(top)

        backend_row = QHBoxLayout()
        backend_label = QLabel("Engine")
        backend_label.setObjectName("LayerMeta")
        self._backend = QComboBox()
        for key, label in ENGINES:
            self._backend.addItem(label, key)
        self._backend.currentIndexChanged.connect(
            lambda: self.audioBackendChanged.emit(self._backend.currentData()))
        backend_row.addWidget(backend_label)
        backend_row.addWidget(self._backend, 1)
        section.add_layout(backend_row)

        # -- dialect × gender -------------------------------------------------
        # Indian English is listed first and marked when not installed, so a
        # missing voice reads as a gap to fix rather than an absent feature.
        dialect_row = QHBoxLayout()
        dialect_label = QLabel("Language")
        dialect_label.setObjectName("LayerMeta")
        self._dialect = QComboBox()
        self._dialect.setMaxVisibleItems(24)
        self._dialect.setToolTip(
            "Narration language and accent — English dialects (Indian English first), Indian "
            "languages and major international languages. Applies to live voiceover and to "
            "exported videos; non-English narration is translated first.")
        self._dialect.currentIndexChanged.connect(
            lambda: self.audioDialectChanged.emit(self._dialect.currentData() or ""))
        dialect_row.addWidget(dialect_label)
        dialect_row.addWidget(self._dialect, 1)
        section.add_layout(dialect_row)

        gender_row = QHBoxLayout()
        gender_label = QLabel("Voice")
        gender_label.setObjectName("LayerMeta")
        self._gender = SegmentBar(
            [("female", "Female"), ("male", "Male"), ("any", "Any")], current="any")
        self._gender.changed.connect(self.audioGenderChanged)
        gender_row.addWidget(gender_label)
        gender_row.addWidget(self._gender, 1)
        section.add_layout(gender_row)

        voice_row = QHBoxLayout()
        voice_label = QLabel("Speaking")
        voice_label.setObjectName("LayerMeta")
        self._voice = QComboBox()
        self._voice.setToolTip("Exact installed voice used for narration")
        self._voice.currentIndexChanged.connect(
            lambda: self.audioVoiceChanged.emit(self._voice.currentData() or ""))
        voice_row.addWidget(voice_label)
        voice_row.addWidget(self._voice, 1)
        section.add_layout(voice_row)

        self._voice_note = QLabel("")
        self._voice_note.setObjectName("LayerMeta")
        self._voice_note.setWordWrap(True)
        section.add_widget(self._voice_note)

        self._voice_help = ChipButton("How to add Indian English voices")
        self._voice_help.clicked.connect(
            lambda: self.voiceSetupRequested.emit(self._dialect.currentData() or "en-IN"))
        self._voice_help.setVisible(False)
        section.add_widget(self._voice_help)

        self._rate = LabeledSlider("Speaking rate", minimum=80, maximum=320, value=175,
                                   suffix=" wpm", formatter=lambda v: f"{v} wpm")
        self._rate.valueChanged.connect(lambda v: self.audioRateChanged.emit(int(v)))
        section.add_widget(self._rate)

        self._volume = LabeledSlider("Volume", minimum=0, maximum=100, value=90)
        self._volume.valueChanged.connect(lambda v: self.audioVolumeChanged.emit(int(v)))
        section.add_widget(self._volume)

        self._audio_section = section
        return section

    # ================================================================ VIDEO
    def _build_video_section(self) -> CollapsibleSection:
        section = CollapsibleSection("Tour Video (narrated MP4)", expanded=True)

        intro = QLabel(
            "Renders the guided tour frame-by-frame at your chosen resolution and "
            "mixes the narration into the file — ready to share or upload."
        )
        intro.setObjectName("LayerMeta")
        intro.setWordWrap(True)
        section.add_widget(intro)

        # Body-system selector. Each system has its OWN narration script, and this
        # follows whatever system is selected in the left panel — the earlier bug
        # was that export stayed pinned to one tour while the user changed system.
        subject_row = QHBoxLayout()
        subject_label = QLabel("Subject")
        subject_label.setObjectName("LayerMeta")
        self._video_tour = QComboBox()
        self._video_tour.setToolTip(
            "Each subject has its own narration script. This follows the system or "
            "structure you selected, so the video always matches what you are viewing.")
        # Built from the registry. A previous hard-coded list in the controller
        # limited this to three tours with the skeleton first, which is why every
        # export came out as a skeletal video.
        self._tours_by_id = {}
        last_group = ""
        for group, label, tour_id in video_subjects():
            tour = get_tour(tour_id)
            if tour is None:
                continue
            self._tours_by_id[tour_id] = tour
            if group != last_group and self._video_tour.count():
                self._video_tour.insertSeparator(self._video_tour.count())
            last_group = group
            self._video_tour.addItem(f"{group}: {label}", tour_id)
        self._video_tour.currentIndexChanged.connect(self._on_video_tour_changed)
        subject_row.addWidget(subject_label)
        subject_row.addWidget(self._video_tour, 1)
        section.add_layout(subject_row)

        self._video_subject = QLabel("")
        self._video_subject.setObjectName("LayerMeta")
        self._video_subject.setWordWrap(True)
        section.add_widget(self._video_subject)

        preset_row = QHBoxLayout()
        preset_label = QLabel("Quality")
        preset_label.setObjectName("LayerMeta")
        self._video_preset = QComboBox()
        for key in ("720p", "1080p", "1440p", "4K"):
            width, height = VIDEO_PRESETS[key]
            self._video_preset.addItem(f"{key}  ·  {width}×{height}", key)
        self._video_preset.setCurrentIndex(1)          # 1080p default
        self._video_preset.currentIndexChanged.connect(
            lambda: self.videoOptionsChanged.emit())
        preset_row.addWidget(preset_label)
        preset_row.addWidget(self._video_preset, 1)
        section.add_layout(preset_row)

        options_row = QHBoxLayout()
        options_row.setSpacing(10)
        self._video_audio = QCheckBox("Narration")
        self._video_audio.setChecked(True)
        self._video_audio.setToolTip("Synthesise and mux the voiceover into the video")
        self._video_captions = QCheckBox("Captions")
        self._video_captions.setChecked(True)
        self._video_captions.setToolTip("Burn narration text into the video frames")
        self._video_ultra = QCheckBox("Ultra")
        self._video_ultra.setChecked(True)
        self._video_ultra.setToolTip(
            "2× supersampling for crisp edges, plus screen-space ambient occlusion "
            "for contact shadows. Renders slower, looks markedly better.")
        self._video_ultra.stateChanged.connect(
            lambda: self.videoOptionsChanged.emit())
        options_row.addWidget(self._video_audio)
        options_row.addWidget(self._video_captions)
        options_row.addWidget(self._video_ultra)
        options_row.addStretch(1)
        section.add_layout(options_row)

        self._video_estimate = QLabel("")
        self._video_estimate.setObjectName("LayerMeta")
        section.add_widget(self._video_estimate)

        self._video_progress = QProgressBar()
        self._video_progress.setRange(0, 100)
        self._video_progress.setValue(0)
        self._video_progress.setTextVisible(False)
        self._video_progress.setVisible(False)
        section.add_widget(self._video_progress)

        self._video_status = QLabel("")
        self._video_status.setObjectName("LayerMeta")
        self._video_status.setWordWrap(True)
        section.add_widget(self._video_status)

        action_row = QHBoxLayout()
        action_row.setSpacing(6)
        self._btn_generate = QPushButton(f"{GLYPH['play']}  Generate video")
        self._btn_generate.setObjectName("PrimaryButton")
        self._btn_generate.clicked.connect(
            lambda: self.videoGenerateRequested.emit(self.current_video_tour_id(),
                                                     self.current_video_preset()))
        self._btn_video_cancel = ChipButton("Cancel")
        self._btn_video_cancel.clicked.connect(self.videoCancelRequested)
        self._btn_video_cancel.setVisible(False)
        action_row.addWidget(self._btn_generate, 1)
        action_row.addWidget(self._btn_video_cancel)
        section.add_layout(action_row)

        result_row = QHBoxLayout()
        result_row.setSpacing(6)
        self._btn_play_video = QPushButton(f"{GLYPH['play']}  Play video")
        self._btn_play_video.setObjectName("PrimaryButton")
        self._btn_play_video.setEnabled(False)
        self._btn_play_video.clicked.connect(
            lambda: self.videoPlayRequested.emit(self._last_video_path))
        self._btn_folder = ChipButton("Show in folder")
        self._btn_folder.setEnabled(False)
        self._btn_folder.clicked.connect(
            lambda: self.videoFolderRequested.emit(self._last_video_path))
        result_row.addWidget(self._btn_play_video, 1)
        result_row.addWidget(self._btn_folder)
        section.add_layout(result_row)

        self._last_video_path = ""
        self._video_section = section
        return section

    def _build_script_section(self) -> CollapsibleSection:
        section = CollapsibleSection("Narration Script", expanded=False)

        self._cue_host = QWidget()
        self._cue_layout = QVBoxLayout(self._cue_host)
        self._cue_layout.setContentsMargins(0, 0, 0, 0)
        self._cue_layout.setSpacing(3)
        self._cue_empty = QLabel("Play or export a tour to see its script here.")
        self._cue_empty.setObjectName("LayerMeta")
        self._cue_empty.setWordWrap(True)
        self._cue_layout.addWidget(self._cue_empty)
        section.add_widget(self._cue_host)

        self._script_section = section
        return section

    # -- video API ---------------------------------------------------------
    def current_video_tour_id(self) -> str:
        return self._video_tour.currentData() or ""

    def select_video_tour(self, tour_id: str) -> bool:
        """Follow the body system chosen in the sidebar.

        Returns True when the selection actually changed.
        """
        if not tour_id:
            return False
        index = self._video_tour.findData(tour_id)
        if index < 0 or index == self._video_tour.currentIndex():
            return False
        self._video_tour.setCurrentIndex(index)
        return True

    def add_video_subject(self, group: str, label: str, tour) -> None:
        """Offer a generated tour (muscle action, disease stage …) for export
        and select it. Re-adding the same id refreshes it."""
        tour_id = getattr(tour, "id", "")
        if not tour_id:
            return
        self._tours_by_id[tour_id] = tour
        index = self._video_tour.findData(tour_id)
        if index < 0:
            self._video_tour.addItem(f"{group}: {label}", tour_id)
            index = self._video_tour.count() - 1
        else:
            self._video_tour.setItemText(index, f"{group}: {label}")
        if index == self._video_tour.currentIndex():
            self._on_video_tour_changed()
        else:
            self._video_tour.setCurrentIndex(index)

    def video_subject_text(self) -> str:
        return self._video_subject.text()

    def _on_video_tour_changed(self) -> None:
        tour = self._tours_by_id.get(self._video_tour.currentData() or "")
        if tour is not None:
            subtitle = getattr(tour, "subtitle", "") or ""
            beats = len(getattr(tour, "keyframes", []) or [])
            words = getattr(tour, "word_count", 0)
            self._video_subject.setText(
                f"{subtitle}  ·  {beats} narrated chapters  ·  {words} words "
                f"(its own script, not shared with other systems)"
                if subtitle else
                f"{beats} narrated chapters  ·  {words} words")
        self.videoOptionsChanged.emit()

    def current_video_preset(self) -> str:
        return self._video_preset.currentData() or "1080p"

    def video_options(self) -> dict:
        """Current export options, ready to splat into :class:`VideoSettings`."""
        ultra = self._video_ultra.isChecked()
        return {
            "include_audio": self._video_audio.isChecked(),
            "burn_captions": self._video_captions.isChecked(),
            "supersample": 2 if ultra else 1,
            "ssao": ultra,
        }

    def set_video_estimate(self, seconds: float, enabled: bool = True,
                           detail: str = "", voice_hint: str = "") -> None:
        if not enabled:
            self._video_estimate.setText(detail or "Video export unavailable.")
            self._btn_generate.setEnabled(False)
            return
        minutes, secs = divmod(int(round(seconds)), 60)
        text = (f"Estimated length ≈ {minutes}:{secs:02d}  ·  renders off-screen "
                f"while you work")
        if voice_hint:
            text += f"\n{voice_hint}"
        self._video_estimate.setText(text)
        self._btn_generate.setEnabled(True)

    def set_video_busy(self, busy: bool, *, percent: int = 0, message: str = "") -> None:
        self._video_progress.setVisible(busy)
        self._btn_video_cancel.setVisible(busy)
        self._btn_generate.setEnabled(not busy)
        if busy:
            self._video_progress.setValue(percent)
            self._video_status.setText(message)
            self._btn_play_video.setEnabled(False)
            self._btn_folder.setEnabled(False)
            self._video_ready = False
        elif message:
            self._video_status.setText(message)

    def set_video_progress(self, percent: int, message: str) -> None:
        self._video_progress.setValue(max(0, min(100, percent)))
        self._video_status.setText(message)

    def set_video_result(self, path: str, message: str = "") -> None:
        self._last_video_path = path or ""
        ready = bool(path)
        self._video_ready = ready
        self._btn_play_video.setEnabled(ready)
        self._btn_folder.setEnabled(ready)
        self._video_status.setText(
            message or (f"Ready: {path}" if ready else ""))

    def set_video_error(self, message: str) -> None:
        self._video_ready = False
        self._btn_play_video.setEnabled(False)
        self._btn_folder.setEnabled(False)
        self._video_status.setText(f"⚠ {message}")

    # ============================================================ public API
    def status_text(self) -> str:
        return self._status.text()

    def set_status(self, text: str, state: str = "off") -> None:
        self._status.setText(text)
        self._dot.set_state(state)

    def set_hardware_summary(self, text: str) -> None:
        self._hardware.setText(text)

    def set_models(self, models: Sequence, current_provider: str = "",
                   current_model: str = "") -> None:
        self._model_combo.blockSignals(True)
        self._model_combo.clear()
        self._model_combo.addItem("Auto (best available)", ("", ""))
        last_provider = ""
        for model in models:
            provider = getattr(model, "provider_id", "")
            kind = getattr(model, "kind", "local")
            if provider != last_provider:
                self._model_combo.insertSeparator(self._model_combo.count())
                self._model_combo.addItem(f"— {provider.upper()} ({kind}) —", ("", ""))
                last_provider = provider
            size = getattr(model, "size_gb", 0.0)
            suffix = f"  ·  {size:.1f} GB" if size else ""
            self._model_combo.addItem(f"   {model.label}{suffix}", (provider, model.id))

        if current_provider and current_model:
            for index in range(self._model_combo.count()):
                if self._model_combo.itemData(index) == (current_provider, current_model):
                    self._model_combo.setCurrentIndex(index)
                    break
        self._model_combo.blockSignals(False)

    def set_mode(self, mode: str) -> None:
        self._mode.set_current(mode)

    def set_busy(self, busy: bool) -> None:
        self._streaming = busy
        self._send.setVisible(not busy)
        self._cancel.setVisible(busy)
        self._send.setEnabled(not busy)

    def set_audio_state(self, enabled: bool) -> None:
        self._audio_toggle.blockSignals(True)
        self._audio_toggle.setChecked(enabled)
        self._audio_toggle.blockSignals(False)
        self._audio_toggle.setText(
            f"{GLYPH['speaker']}  Voiceover on" if enabled
            else f"{GLYPH['sound_off']}  Voiceover off")

    def set_audio_options(self, backends: Sequence[str], voices: Sequence,
                          backend: str = "", voice: str = "",
                          rate: int = 175, volume: int = 90,
                          dialects: Sequence = (), dialect: str = "any",
                          gender: str = "any", note: str = "") -> None:
        self._backend.blockSignals(True)
        self._backend.clear()
        labels = dict(ENGINES)
        for key in backends:
            self._backend.addItem(labels.get(key, key), key)
        index = self._backend.findData(backend)
        if index >= 0:
            self._backend.setCurrentIndex(index)
        self._backend.blockSignals(False)

        if dialects:
            self._dialect.blockSignals(True)
            self._dialect.clear()
            for code, label in dialects:
                self._dialect.addItem(label, code)
                if not code:                      # group heading: visible, not selectable
                    item = self._dialect.model().item(self._dialect.count() - 1)
                    if item is not None:
                        item.setEnabled(False)
            dindex = self._dialect.findData(dialect)
            if dindex >= 0:
                self._dialect.setCurrentIndex(dindex)
            self._dialect.blockSignals(False)

        self._gender.set_current(gender or "any", emit=False)

        self._voice.blockSignals(True)
        self._voice.clear()
        self._voice.addItem("Auto (best match for language and voice)", "")
        for item in voices:
            self._voice.addItem(str(item), getattr(item, "id", ""))
        vindex = self._voice.findData(voice)
        if vindex >= 0:
            self._voice.setCurrentIndex(vindex)
        self._voice.blockSignals(False)

        self._rate.set_value(rate)
        self._volume.set_value(volume)
        self.set_voice_note(note)

    def set_voice_note(self, note: str, warn: bool = True) -> None:
        """Show why the active voice may differ from the requested dialect."""
        self._voice_note.setText(note or "")
        self._voice_note.setVisible(bool(note))
        self._voice_help.setVisible(bool(note) and warn and "not installed" in note.lower()
                                    and "neural" not in note.lower())

    # -- transcript --------------------------------------------------------
    def append_user(self, text: str) -> None:
        self.show_tab("assistant")
        self._append_html(
            f"<span style='color:#22D3EE;font-weight:700;'>You</span>"
            f"<br><span style='color:#E8EFFA;'>{_escape(text)}</span>")

    def start_assistant(self, provider: str, model: str) -> None:
        self._last_answer = ""
        self._append_html(
            f"<span style='color:#34D399;font-weight:700;'>Anatomy Copilot</span>"
            f"<span style='color:#63738D;font-size:11px;'>  {_escape(provider)} · "
            f"{_escape(model)}</span><br>")
        self._answer_start = self._chat_view.textCursor().position()

    def append_token(self, chunk: str) -> None:
        self._last_answer += chunk
        cursor = self._chat_view.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertText(chunk)
        self._chat_view.setTextCursor(cursor)
        self._chat_view.ensureCursorVisible()

    def finish_assistant(self, seconds: float = 0.0) -> None:
        if self._last_answer.strip():
            cursor = self._chat_view.textCursor()
            cursor.setPosition(self._answer_start)
            cursor.movePosition(QTextCursor.MoveOperation.End,
                                QTextCursor.MoveMode.KeepAnchor)
            cursor.removeSelectedText()
            cursor.insertHtml(_render_markdown(self._last_answer))
            self._chat_view.setTextCursor(cursor)

        self._append_html(
            f"<span style='color:#63738D;font-size:10.5px;'>"
            f"answered in {seconds:.1f}s</span>")

    def append_error(self, message: str) -> None:
        escaped = _escape(message).replace("\n", "<br>")
        self._append_html(f"<span style='color:#F87171;'>⚠ {escaped}</span>")

    def append_note(self, message: str) -> None:
        self._append_html(
            f"<span style='color:#63738D; font-size:11px;'>{_escape(message)}</span>")

    def clear_chat(self) -> None:
        self._chat_view.clear()
        self._last_answer = ""

    def last_answer(self) -> str:
        return self._last_answer

    # -- narration cues ----------------------------------------------------
    def set_cues(self, cues: Sequence) -> None:
        while self._cue_layout.count():
            item = self._cue_layout.takeAt(0)
            widget = item.widget()
            if widget is None:
                continue
            if widget is self._cue_empty:
                widget.setParent(None)
            else:
                widget.setParent(None)
                widget.deleteLater()
        self._cue_rows.clear()

        for cue in cues:
            row = QPushButton()
            row.setObjectName("CueRow")
            row.setFlat(True)
            row.setCursor(Qt.CursorShape.PointingHandCursor)
            index = getattr(cue, "index", 0)
            title = getattr(cue, "title", "") or f"Step {index + 1}"
            row.setText(f"{title}\n{_ellipsis(getattr(cue, 'text', ''), 90)}")
            row.setStyleSheet("text-align:left;")
            row.clicked.connect(lambda _c=False, i=index: self.replayCueRequested.emit(i))
            self._cue_rows.append(row)
            self._cue_layout.addWidget(row)

        if not cues:
            self._cue_layout.addWidget(self._cue_empty)

    def highlight_cue(self, index: int) -> None:
        from app.ui.theme import repolish
        for position, row in enumerate(self._cue_rows):
            row.setProperty("active", "true" if position == index else "false")
            repolish(row)

    # -- internals ---------------------------------------------------------
    def _append_html(self, html: str) -> None:
        cursor = self._chat_view.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        if not self._chat_view.document().isEmpty():
            cursor.insertBlock()
        cursor.insertHtml(html)
        self._chat_view.setTextCursor(cursor)
        self._chat_view.ensureCursorVisible()

    def _emit_prompt(self) -> None:
        text = self._input.toPlainText().strip()
        if not text or self._streaming:
            return
        self._input.clear()
        self.promptSubmitted.emit(text)

    def _on_model_changed(self) -> None:
        data = self._model_combo.currentData()
        if not data:
            return
        provider, model = data
        if provider and model:
            self.modelSelected.emit(provider, model)

    def _on_audio_toggled(self, checked: bool) -> None:
        self._audio_toggle.setText(
            f"{GLYPH['speaker']}  Voiceover on" if checked
            else f"{GLYPH['sound_off']}  Voiceover off")
        self.audioEnabledChanged.emit(checked)

    def eventFilter(self, obj, event):  # noqa: N802
        from PyQt6.QtCore import QEvent
        if obj is self._input and event.type() == QEvent.Type.KeyPress:
            key = event.key()
            if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                    return False
                self._emit_prompt()
                return True
        return super().eventFilter(obj, event)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _escape(text: str) -> str:
    return (text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


_BULLET = re.compile(r"^\s*[-*\u2022]\s+(.*)$")
_NUMBERED = re.compile(r"^\s*(\d+)[.)]\s+(.*)$")
_BOLD = re.compile(r"\*\*(.+?)\*\*", re.S)
_ITALIC = re.compile(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", re.S)
_CODE = re.compile(r"`([^`]+)`")


def _inline(text: str) -> str:
    text = _BOLD.sub(r"<b>\1</b>", text)
    text = _ITALIC.sub(r"<i>\1</i>", text)
    text = _CODE.sub(
        r"<code style='background-color:#0D121D;color:#22D3EE;'>\1</code>", text)
    return text


def _render_markdown(text: str) -> str:
    """Minimal, safe markdown → HTML for the transcript."""
    html: List[str] = []
    in_list = False

    for raw in _escape(text or "").split("\n"):
        line = raw.rstrip()
        bullet = _BULLET.match(line)
        numbered = _NUMBERED.match(line)

        if bullet or numbered:
            if not in_list:
                html.append("<ul style='margin:2px 0 6px 16px;'>")
                in_list = True
            body = bullet.group(1) if bullet else numbered.group(2)
            html.append(f"<li>{_inline(body)}</li>")
            continue

        if in_list:
            html.append("</ul>")
            in_list = False

        if not line.strip():
            html.append("<br>")
        elif line.startswith("### "):
            html.append(f"<p style='margin:6px 0 2px 0;'><b>{_inline(line[4:])}</b></p>")
        else:
            html.append(f"<p style='margin:2px 0;'>{_inline(line)}</p>")

    if in_list:
        html.append("</ul>")
    return "".join(html)


def _ellipsis(text: str, limit: int) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= limit else text[: limit - 1] + "…"
