"""
Caption and title overlays for exported video, drawn with Qt.

VTK's text actors cannot shape complex scripts: Devanagari, Bengali, Tamil and
the other Indic scripts need conjunct and vowel-sign shaping, Arabic and Urdu
need joining and right-to-left layout. Qt's text engine (HarfBuzz) does all of
that and falls back across installed fonts (on Windows: Nirmala UI for Indic
scripts, Segoe UI for Arabic, Microsoft YaHei for Chinese …), so captions are
rendered here and composited onto each frame.

Narration is shown a sentence at a time, timed to the speech, rather than as a
whole paragraph clipped at the frame edge.
"""
from __future__ import annotations

import re
from typing import List, Optional, Sequence, Tuple

import numpy as np
from PyQt6.QtCore import QRectF, Qt
from PyQt6.QtGui import QColor, QFont, QImage, QPainter, QPainterPath, QTextOption

#: Sentence / clause boundaries across scripts (Latin, Devanagari danda,
#: Arabic and CJK punctuation).
_SENTENCE = re.compile(r"(?<=[.!?।॥؟。！？])\s+|(?<=[;:])\s+(?=\S{12,})")


def chunk_text(text: str, max_chars: int = 120) -> List[str]:
    """Split narration into caption-sized chunks on sentence boundaries."""
    text = " ".join((text or "").split())
    if not text:
        return []
    parts = [p.strip() for p in _SENTENCE.split(text) if p.strip()]
    out: List[str] = []
    for part in parts:
        while len(part) > max_chars:
            cut = part.rfind(", ", 0, max_chars)
            cut = cut if cut > max_chars // 3 else part.rfind(" ", 0, max_chars)
            if cut <= 0:
                break
            out.append(part[:cut + 1].strip())
            part = part[cut + 1:].strip()
        if part:
            out.append(part)
    return out


def chunk_at(chunks: Sequence[str], seconds: float, spoken: float) -> str:
    """The chunk being spoken at *seconds*, allotting time by character count."""
    if not chunks:
        return ""
    if spoken <= 0:
        return chunks[0]
    total = sum(len(c) for c in chunks) or 1
    elapsed = 0.0
    for chunk in chunks:
        elapsed += spoken * len(chunk) / total
        if seconds < elapsed:
            return chunk
    return chunks[-1]


class CaptionRenderer:
    """Composites title, caption and attribution onto RGB frames."""

    def __init__(self, width: int, height: int, *, rtl: bool = False,
                 brand: str = "BioHuman3D", attribution: str = "") -> None:
        self.width, self.height = int(width), int(height)
        self.rtl = rtl
        self.brand = brand
        self.attribution = attribution
        base = max(12, int(self.height / 26))
        self.caption_font = QFont()
        self.caption_font.setPixelSize(base)
        self.caption_font.setWeight(QFont.Weight.DemiBold)
        self.title_font = QFont()
        self.title_font.setPixelSize(max(10, int(base * 0.72)))
        self.small_font = QFont()
        self.small_font.setPixelSize(max(9, int(base * 0.48)))

    def compose(self, frame: np.ndarray, title: str = "", caption: str = "") -> np.ndarray:
        frame = np.ascontiguousarray(frame[:, :, :3], dtype=np.uint8)
        h, w, _ = frame.shape
        # PyQt copies the buffer it is given, so paint on our own image and read back.
        image = QImage(frame.tobytes(), w, h, 3 * w, QImage.Format.Format_RGB888).copy()
        painter = QPainter(image)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
        margin = w * 0.025

        if title:
            painter.setFont(self.title_font)
            painter.setPen(QColor(170, 205, 235))
            painter.drawText(QRectF(margin, margin * 0.6, w * 0.7, h * 0.08),
                             int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop), title)
        painter.setFont(self.small_font)
        painter.setPen(QColor(120, 140, 165))
        painter.drawText(QRectF(w * 0.5, margin * 0.6, w * 0.5 - margin, h * 0.05),
                         int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop), self.brand)
        if self.attribution:
            painter.drawText(QRectF(w * 0.35, h - margin * 0.6 - h * 0.035, w * 0.65 - margin, h * 0.035),
                             int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignBottom), self.attribution)

        if caption:
            option = QTextOption()
            option.setWrapMode(QTextOption.WrapMode.WordWrap)
            option.setAlignment(Qt.AlignmentFlag.AlignHCenter)
            option.setTextDirection(Qt.LayoutDirection.RightToLeft if self.rtl else Qt.LayoutDirection.LeftToRight)
            painter.setFont(self.caption_font)
            box_w = w * 0.82
            measure = painter.boundingRect(QRectF(0, 0, box_w, h), caption, option)
            text_h = min(measure.height(), h * 0.30)
            pad = self.caption_font.pixelSize() * 0.55
            box = QRectF((w - box_w) / 2 - pad, h - margin * 1.6 - text_h - 2 * pad, box_w + 2 * pad, text_h + 2 * pad)
            path = QPainterPath()
            path.addRoundedRect(box, pad * 0.8, pad * 0.8)
            painter.fillPath(path, QColor(3, 6, 12, 168))
            painter.setPen(QColor(34, 211, 238, 150))
            painter.drawPath(path)
            painter.setPen(QColor(245, 248, 255))
            painter.drawText(box.adjusted(pad, pad, -pad, -pad), caption, option)
        painter.end()
        ptr = image.constBits()
        ptr.setsize(image.sizeInBytes())
        rows = np.frombuffer(ptr, np.uint8).reshape(h, image.bytesPerLine())
        return rows[:, :3 * w].reshape(h, w, 3).copy()      # copy: the QImage is freed on return
