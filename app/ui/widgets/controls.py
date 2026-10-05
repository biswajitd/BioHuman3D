"""
Small reusable controls.

Icons are Unicode glyphs rather than PNG/SVG assets so the project has no binary
dependencies and scales cleanly with the UI font. Swap ``GLYPH`` values for
``QIcon`` calls if you later add a branded icon set.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont
from PyQt6.QtWidgets import (QFrame, QHBoxLayout, QLabel, QLineEdit, QPushButton,
                             QSizePolicy, QSlider, QToolButton, QVBoxLayout,
                             QWidget)


#: Consistent iconography across the app.
GLYPH: Dict[str, str] = {
    "eye": "\u25c9",         # ◉ visible
    "eye_off": "\u25cb",     # ○ hidden
    "solo": "\u25c6",        # ◆ isolate
    "play": "\u25b6",        # ▶
    "pause": "\u23f8",       # ⏸
    "stop": "\u25a0",        # ■
    "next": "\u23ed",        # ⏭
    "prev": "\u23ee",        # ⏮
    "refresh": "\u21bb",     # ↻
    "reset": "\u27f2",       # ⟲
    "camera": "\u2316",      # ⌖ snapshot
    "search": "\u2315",      # ⌕
    "chevron_down": "\u25be",  # ▾
    "chevron_right": "\u25b8", # ▸
    "close": "\u2715",       # ✕
    "minimize": "\u2013",    # –
    "maximize": "\u2610",    # ☐
    "speaker": "\u266b",     # ♫
    "sound_off": "\u266a",   # ♪
    "brain": "\u25c8",       # ◈
    "layers": "\u29c9",      # ⧉
    "tools": "\u2692",       # ⚒
    "send": "\u27a4",        # ➤
    "menu": "\u2630",        # ☰
    "video": "\u25a3",       # ▣
}


class IconButton(QToolButton):
    """Square tool button around a single glyph."""

    def __init__(self, glyph: str, tooltip: str = "", *, checkable: bool = False,
                 role: str = "", size: int = 13, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setText(glyph)
        self.setToolTip(tooltip)
        self.setCheckable(checkable)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        font = QFont(self.font())
        font.setPointSize(size)
        self.setFont(font)
        if role:
            self.setProperty("role", role)


class ChipButton(QPushButton):
    """Pill-shaped quick action used above the chat input."""

    def __init__(self, text: str, tooltip: str = "", parent: Optional[QWidget] = None):
        super().__init__(text, parent)
        self.setObjectName("Chip")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        if tooltip:
            self.setToolTip(tooltip)


class SegmentBar(QWidget):
    """Exclusive segmented control: ``[ Local | Cloud | Auto ]``.

    Emits :attr:`changed` with the *key* (not the index) so call sites stay
    readable and stable when buttons are reordered.
    """

    changed = pyqtSignal(str)

    def __init__(self, options: Sequence[tuple], *, current: str = "",
                 parent: Optional[QWidget] = None):
        """``options`` = ``[(key, label), ...]``"""
        super().__init__(parent)
        self.setObjectName("SegmentBar")
        self._buttons: Dict[str, QPushButton] = {}
        self._current = ""

        layout = QHBoxLayout(self)
        layout.setContentsMargins(3, 3, 3, 3)
        layout.setSpacing(2)

        for key, label in options:
            button = QPushButton(label)
            button.setObjectName("SegmentButton")
            button.setCheckable(True)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.clicked.connect(lambda _checked, k=key: self.set_current(k, emit=True))
            layout.addWidget(button)
            self._buttons[key] = button

        if options:
            self.set_current(current or options[0][0], emit=False)

    @property
    def current(self) -> str:
        return self._current

    def set_current(self, key: str, *, emit: bool = False) -> None:
        if key not in self._buttons:
            return
        for k, button in self._buttons.items():
            button.setChecked(k == key)
        if key != self._current:
            self._current = key
            if emit:
                self.changed.emit(key)
        else:
            self._current = key

    def set_enabled(self, key: str, enabled: bool, tooltip: str = "") -> None:
        button = self._buttons.get(key)
        if button is not None:
            button.setEnabled(enabled)
            button.setToolTip(tooltip)


class LabeledSlider(QWidget):
    """``Label ————o———— 42%`` row with an optional value formatter."""

    valueChanged = pyqtSignal(float)

    def __init__(self, label: str, *, minimum: int = 0, maximum: int = 100,
                 value: int = 100, compact: bool = True, suffix: str = "%",
                 formatter=None, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._suffix = suffix
        self._formatter = formatter

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 2, 0, 2)
        root.setSpacing(3)

        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        self._label = QLabel(label)
        self._label.setObjectName("LayerMeta")
        self._value = QLabel("")
        self._value.setObjectName("LayerMeta")
        self._value.setAlignment(Qt.AlignmentFlag.AlignRight)
        top.addWidget(self._label)
        top.addStretch(1)
        top.addWidget(self._value)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(minimum, maximum)
        self.slider.setValue(value)
        self.slider.setProperty("compact", "true" if compact else "false")
        self.slider.valueChanged.connect(self._on_changed)

        root.addLayout(top)
        root.addWidget(self.slider)
        self._refresh_label(value)

    # -- API ---------------------------------------------------------------
    def value(self) -> int:
        return self.slider.value()

    def set_value(self, value: int, *, emit: bool = False) -> None:
        blocked = self.slider.blockSignals(not emit)
        self.slider.setValue(int(value))
        self.slider.blockSignals(blocked)
        self._refresh_label(int(value))

    def normalized(self) -> float:
        span = self.slider.maximum() - self.slider.minimum() or 1
        return (self.slider.value() - self.slider.minimum()) / span

    def set_label(self, text: str) -> None:
        self._label.setText(text)

    # -- internals ---------------------------------------------------------
    def _on_changed(self, value: int) -> None:
        self._refresh_label(value)
        self.valueChanged.emit(float(value))

    def _refresh_label(self, value: int) -> None:
        if self._formatter is not None:
            self._value.setText(self._formatter(value))
        else:
            self._value.setText(f"{value}{self._suffix}")


class StatusDot(QLabel):
    """Coloured availability indicator driven by a QSS dynamic property."""

    def __init__(self, state: str = "off", parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("Dot")
        self.setFixedSize(8, 8)
        self.set_state(state)

    def set_state(self, state: str) -> None:
        self.setProperty("state", state)
        from app.ui.theme import repolish
        repolish(self)


class Divider(QFrame):
    """One-pixel horizontal rule."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("Divider")
        self.setFrameShape(QFrame.Shape.HLine)
        self.setFixedHeight(1)


class SearchField(QLineEdit):
    """Rounded search input with a leading glyph."""

    def __init__(self, placeholder: str = "Search…", parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("SearchField")
        self.setPlaceholderText(f"{GLYPH['search']}  {placeholder}")
        self.setClearButtonEnabled(True)
        self.setMinimumHeight(30)


class StatusChip(QLabel):
    """Small translucent pill used over the 3D viewport."""

    def __init__(self, text: str = "", *, accent: bool = False,
                 parent: Optional[QWidget] = None):
        super().__init__(text, parent)
        self.setObjectName("StatusChip")
        self.setProperty("accent", "true" if accent else "false")
        self.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)


def panel_header(title: str, hint: str = "") -> QWidget:
    """Standard ``PANEL TITLE`` header block used by all three panels."""
    wrapper = QWidget()
    wrapper.setObjectName("PanelHeader")
    layout = QVBoxLayout(wrapper)
    layout.setContentsMargins(14, 12, 14, 6)
    layout.setSpacing(2)

    title_label = QLabel(title.upper())
    title_label.setObjectName("PanelHeaderTitle")
    layout.addWidget(title_label)

    if hint:
        hint_label = QLabel(hint)
        hint_label.setObjectName("PanelHint")
        hint_label.setWordWrap(True)
        layout.addWidget(hint_label)

    return wrapper
