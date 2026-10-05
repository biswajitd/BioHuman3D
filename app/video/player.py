"""
In-app video playback.

``VideoPlayerDialog`` plays an exported tour inside BioHuman3D using Qt's
multimedia stack (already bundled with PyQt6 — no extra dependency). If Qt
Multimedia is unavailable on the host, the dialog degrades to a labelled panel
with an "Open in default player" button instead of failing.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import Qt, QUrl, pyqtSignal
from PyQt6.QtWidgets import (QDialog, QHBoxLayout, QLabel, QPushButton, QSlider,
                             QVBoxLayout, QWidget)

try:
    from PyQt6.QtMultimedia import QAudioOutput, QMediaPlayer
    from PyQt6.QtMultimediaWidgets import QVideoWidget
    MULTIMEDIA_AVAILABLE = True
except Exception:  # pragma: no cover
    MULTIMEDIA_AVAILABLE = False


def open_in_default_player(path: Path) -> bool:
    """Open a media file with the OS default application."""
    try:
        if sys.platform == "win32":
            os.startfile(str(path))          # noqa: S606 - intended behaviour
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(path)])
        else:
            subprocess.Popen(["xdg-open", str(path)])
        return True
    except Exception:
        return False


def _format_time(ms: int) -> str:
    seconds = max(0, ms // 1000)
    return f"{seconds // 60:d}:{seconds % 60:02d}"


class VideoPlayerDialog(QDialog):
    """Player window for an exported tour video."""

    exportAgainRequested = pyqtSignal()

    def __init__(self, path: Path, title: str = "Tour video",
                 parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._path = Path(path)
        self.setWindowTitle(f"BioHuman3D — {title}")
        self.setMinimumSize(760, 520)

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(9)

        header = QLabel(f"<b>{title}</b><br>"
                        f"<span style='color:#9AABC4; font-size:11px;'>{self._path}</span>")
        header.setTextFormat(Qt.TextFormat.RichText)
        header.setWordWrap(True)
        root.addWidget(header)

        if not MULTIMEDIA_AVAILABLE:
            note = QLabel(
                "Qt Multimedia is not available in this PyQt6 build, so the video "
                "cannot be played inside the app.\n\n"
                "The file was created successfully — use the button below to open it "
                "in your system player."
            )
            note.setWordWrap(True)
            root.addWidget(note)
            root.addStretch(1)
            root.addLayout(self._button_row())
            return

        # -- video surface ---------------------------------------------------
        self._video = QVideoWidget()
        self._video.setMinimumHeight(360)
        self._video.setStyleSheet("background:#05080E; border-radius:8px;")
        root.addWidget(self._video, 1)

        self._player = QMediaPlayer(self)
        self._audio = QAudioOutput(self)
        self._audio.setVolume(0.9)
        self._player.setAudioOutput(self._audio)
        self._player.setVideoOutput(self._video)
        self._player.setSource(QUrl.fromLocalFile(str(self._path)))

        # -- transport -------------------------------------------------------
        transport = QHBoxLayout()
        transport.setSpacing(8)

        self._play = QPushButton("▶  Play")
        self._play.setObjectName("PrimaryButton")
        self._play.clicked.connect(self._toggle)

        self._position = QSlider(Qt.Orientation.Horizontal)
        self._position.setRange(0, 0)
        self._position.sliderMoved.connect(self._player.setPosition)

        self._time = QLabel("0:00 / 0:00")
        self._time.setObjectName("LayerMeta")

        self._volume = QSlider(Qt.Orientation.Horizontal)
        self._volume.setRange(0, 100)
        self._volume.setValue(90)
        self._volume.setFixedWidth(90)
        self._volume.valueChanged.connect(lambda v: self._audio.setVolume(v / 100.0))

        transport.addWidget(self._play)
        transport.addWidget(self._position, 1)
        transport.addWidget(self._time)
        transport.addWidget(QLabel("🔊"))
        transport.addWidget(self._volume)
        root.addLayout(transport)

        root.addLayout(self._button_row())

        # -- signals ---------------------------------------------------------
        self._player.positionChanged.connect(self._on_position)
        self._player.durationChanged.connect(self._on_duration)
        self._player.playbackStateChanged.connect(self._on_state)
        self._player.errorOccurred.connect(self._on_error)

        self._player.play()

    # -- helpers -----------------------------------------------------------
    def _button_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(8)

        again = QPushButton("Generate another…")
        again.setObjectName("GhostButton")
        again.clicked.connect(self.exportAgainRequested)

        reveal = QPushButton("Show in folder")
        reveal.setObjectName("GhostButton")
        reveal.clicked.connect(
            lambda: subprocess.Popen(["explorer", "/select,", str(self._path)])
            if sys.platform == "win32" else open_in_default_player(self._path.parent)
        )

        external = QPushButton("Open in system player")
        external.setObjectName("GhostButton")
        external.clicked.connect(lambda: open_in_default_player(self._path))

        close = QPushButton("Close")
        close.clicked.connect(self.accept)

        row.addStretch(1)
        row.addWidget(again)
        row.addWidget(reveal)
        row.addWidget(external)
        row.addWidget(close)
        return row

    def _toggle(self) -> None:
        if not MULTIMEDIA_AVAILABLE:
            return
        if self._player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self._player.pause()
        else:
            self._player.play()

    def _on_position(self, position: int) -> None:
        if not self._position.isSliderDown():
            self._position.setValue(position)
        self._time.setText(f"{_format_time(position)} / {_format_time(self._duration)}")

    def _on_duration(self, duration: int) -> None:
        self._duration = duration
        self._position.setRange(0, duration)

    _duration = 0

    def _on_state(self, state) -> None:
        playing = state == QMediaPlayer.PlaybackState.PlayingState
        self._play.setText("⏸  Pause" if playing else "▶  Play")

    def _on_error(self, _error, message: str = "") -> None:
        self._time.setText("playback error")

    # -- lifecycle ---------------------------------------------------------
    def closeEvent(self, event) -> None:  # noqa: N802
        if MULTIMEDIA_AVAILABLE:
            try:
                self._player.stop()
            except Exception:
                pass
        super().closeEvent(event)
