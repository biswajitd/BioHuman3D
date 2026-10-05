"""
Animated collapsible section.

``CollapsibleSection`` gives the panels their layered, tool-drawer feel: a card
with a clickable header that smoothly expands/collapses its body. Animation runs
on ``maximumHeight`` (an animatable int property), which is the only reliable way
to animate a widget's size in Qt without a custom ``QPropertyAnimation``
subclass.
"""
from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import QEasingCurve, QPropertyAnimation, Qt, pyqtSignal
from PyQt6.QtWidgets import (QFrame, QHBoxLayout, QLabel, QPushButton,
                             QSizePolicy, QVBoxLayout, QWidget)

from app.ui.widgets.controls import GLYPH


class CollapsibleSection(QFrame):
    """Card-style section with an animated body."""

    toggled = pyqtSignal(bool)

    def __init__(self, title: str, *, expanded: bool = True,
                 badge: str = "", parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("Card")
        self._expanded = expanded

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # -- header ---------------------------------------------------------
        self._header = QPushButton()
        self._header.setObjectName("CardHeader")
        self._header.setCheckable(True)
        self._header.setChecked(expanded)
        self._header.setCursor(Qt.CursorShape.PointingHandCursor)
        self._header.clicked.connect(self._on_header_clicked)

        header_layout = QHBoxLayout(self._header)
        header_layout.setContentsMargins(10, 8, 10, 8)
        header_layout.setSpacing(8)

        self._chevron = QLabel(GLYPH["chevron_down"] if expanded else GLYPH["chevron_right"])
        self._chevron.setObjectName("CardCount")
        self._title = QLabel(title)
        self._badge = QLabel(badge)
        self._badge.setObjectName("CardCount")
        self._badge.setVisible(bool(badge))

        header_layout.addWidget(self._chevron)
        header_layout.addWidget(self._title)
        header_layout.addStretch(1)
        header_layout.addWidget(self._badge)

        # -- body -----------------------------------------------------------
        self.body = QWidget()
        self.body.setObjectName("CardBody")
        self._body_layout = QVBoxLayout(self.body)
        self._body_layout.setContentsMargins(10, 2, 10, 10)
        self._body_layout.setSpacing(7)

        self._animation = QPropertyAnimation(self.body, b"maximumHeight", self)
        self._animation.setDuration(170)
        self._animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._animation.finished.connect(self._on_animation_finished)

        root.addWidget(self._header)
        root.addWidget(self.body)

        if not expanded:
            self.body.setMaximumHeight(0)
            self.body.setVisible(False)

    # -- API ---------------------------------------------------------------
    def add_widget(self, widget: QWidget) -> None:
        self._body_layout.addWidget(widget)

    def add_layout(self, layout) -> None:
        self._body_layout.addLayout(layout)

    def body_layout(self) -> QVBoxLayout:
        return self._body_layout

    def set_badge(self, text: str) -> None:
        self._badge.setText(text)
        self._badge.setVisible(bool(text))

    def is_expanded(self) -> bool:
        return self._expanded

    def set_expanded(self, expanded: bool, *, animate: bool = True) -> None:
        if expanded == self._expanded:
            return
        self._expanded = expanded
        self._header.setChecked(expanded)
        self._header.setProperty("expanded", "true" if expanded else "false")
        self._chevron.setText(GLYPH["chevron_down"] if expanded else GLYPH["chevron_right"])

        if not animate:
            self.body.setVisible(expanded)
            self.body.setMaximumHeight(16777215 if expanded else 0)
            self.toggled.emit(expanded)
            return

        if expanded:
            self.body.setVisible(True)
            target = max(1, self.body.sizeHint().height())
            start = 0
        else:
            target = 0
            start = self.body.height()

        self._animation.stop()
        self._animation.setStartValue(start)
        self._animation.setEndValue(target)
        self._animation.start()
        self.toggled.emit(expanded)

    # -- internals ---------------------------------------------------------
    def _on_header_clicked(self) -> None:
        self.set_expanded(not self._expanded)

    def _on_animation_finished(self) -> None:
        if self._expanded:
            # Release the constraint so the body can grow with new content.
            self.body.setMaximumHeight(16777215)
        else:
            self.body.setVisible(False)
