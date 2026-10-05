"""
Narration synthesis and audio-track assembly for video export.

Two responsibilities:

1. :class:`NarrationSynthesizer` turns narration cues into WAV files using the
   offline system voices (pyttsx3 / SAPI5) and reports each clip's true
   duration. The video timeline is then built from those durations, so speech
   and visuals stay in sync instead of being approximated.

2. :class:`AudioTrackBuilder` mixes those clips onto a single timeline and writes
   one WAV that ffmpeg muxes into the final MP4.

Threading: ``pyttsx3`` must be created inside the thread that uses it (SAPI5
binds to the creating thread's COM apartment), so synthesis runs in a QThread
worker that builds its own engine in ``run()``.
"""
from __future__ import annotations

import math
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Sequence

import numpy as np
from PyQt6.QtCore import QObject, QThread, pyqtSignal

from app.core.wincom import co_initialize as _co_initialize
from app.core.wincom import co_uninitialize as _co_uninitialize

try:
    import pyttsx3
    PYTTsx3_AVAILABLE = True
except Exception:  # pragma: no cover
    pyttsx3 = None  # type: ignore[assignment]
    PYTTsx3_AVAILABLE = False


#: Fallback speaking rate (words/second) when TTS is unavailable.
FALLBACK_WORDS_PER_SECOND = 2.4

#: Target sample rate for the mixed track handed to ffmpeg.
MIX_RATE = 44100


@dataclass
class NarrationClip:
    """One synthesised (or estimated) narration segment."""

    index: int
    text: str
    title: str = ""
    path: Optional[Path] = None
    duration: float = 0.0
    source: str = "estimated"        # "tts" | "estimated"
    start: float = 0.0               # placement on the timeline, seconds

    @property
    def ok(self) -> bool:
        return self.source == "tts" and self.path is not None and self.path.exists()


def estimate_duration(text: str, rate_wpm: int = 175) -> float:
    """Rough spoken duration when real synthesis is not possible."""
    words = max(1, len((text or "").split()))
    return max(2.0, words / max(1.0, rate_wpm / 60.0)) + 0.4


# ---------------------------------------------------------------------------
# Synthesis worker
# ---------------------------------------------------------------------------
class NarrationSynthWorker(QThread):
    """Renders every cue to a WAV file on a worker thread."""

    progress = pyqtSignal(int, str)          # percent, message
    completed = pyqtSignal(list)             # List[NarrationClip]
    failed = pyqtSignal(str)

    def __init__(self, cues: Sequence, work_dir: Path, *, rate: int = 175,
                 volume: float = 0.9, voice_id: str = "", enabled: bool = True,
                 parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._cues = list(cues)
        self._dir = Path(work_dir)
        self._rate = int(rate)
        self._volume = float(volume)
        self._voice_id = voice_id
        self._enabled = enabled
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def run(self) -> None:  # noqa: D102
        self._dir.mkdir(parents=True, exist_ok=True)
        _co_initialize()

        clips: List[NarrationClip] = []
        total = max(1, len(self._cues))
        use_tts = self._enabled and PYTTsx3_AVAILABLE

        for position, cue in enumerate(self._cues):
            if self._cancel:
                break

            text = (getattr(cue, "text", "") or "").strip()
            index = int(getattr(cue, "index", position))
            title = getattr(cue, "title", "") or f"Step {index + 1}"

            self.progress.emit(int(position / total * 100),
                               f"Synthesising narration {position + 1}/{total}…")

            clip = NarrationClip(index=index, text=text, title=title)
            target = self._dir / f"cue_{index:03d}.wav"

            if use_tts and text and self._synthesise_one(text, target):
                clip.path = target
                clip.duration = _wav_duration(target)
                clip.source = "tts"

            if clip.source != "tts":
                # Fall back to a timed estimate so a silent video still works.
                clip.duration = estimate_duration(text, self._rate)

            clips.append(clip)

        _co_uninitialize()
        self.completed.emit(clips)

    def _synthesise_one(self, text: str, target: Path) -> bool:
        """Render a single cue using a **fresh** engine.

        Do NOT reuse one ``pyttsx3`` engine for several ``save_to_file`` calls
        in a loop. On Windows/SAPI5 the driver's queue state does not survive a
        second ``runAndWait``: the first cue renders correctly and the second
        either hangs forever or kills the process with a fail-fast
        (0xC0000409) — verified with an isolated three-cue reproduction.
        A fresh engine per cue is reliable, and the cost is negligible next to
        the speech itself.
        """
        try:
            if target.exists():
                target.unlink()

            # Engine must be created and driven on THIS thread.
            engine = pyttsx3.init()  # type: ignore[union-attr]
            try:
                engine.setProperty("rate", self._rate)
                engine.setProperty("volume", self._volume)
                if self._voice_id:
                    for voice in engine.getProperty("voices"):
                        if voice.id == self._voice_id:
                            engine.setProperty("voice", voice.id)
                            break
                engine.save_to_file(text, str(target))
                engine.runAndWait()
            finally:
                try:
                    engine.stop()
                except Exception:
                    pass
                del engine

            return target.exists() and target.stat().st_size > 1024
        except Exception as exc:
            self.failed.emit(f"Narration synthesis failed: {exc}")
            return False


def _wav_duration(path: Path) -> float:
    """Duration of a WAV file in seconds."""
    try:
        with wave.open(str(path), "rb") as handle:
            frames = handle.getnframes()
            rate = handle.getframerate() or MIX_RATE
            return frames / float(rate)
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Track assembly
# ---------------------------------------------------------------------------
class AudioTrackBuilder:
    """Mixes narration clips onto a single timeline WAV."""

    def __init__(self, rate: int = MIX_RATE) -> None:
        self.rate = rate

    def build(self, clips: Sequence[NarrationClip], total_seconds: float,
              out_path: Path) -> Optional[Path]:
        """Write a mono 16-bit WAV of ``total_seconds`` with clips at ``clip.start``.

        Returns ``None`` when no clip produced usable audio, so the caller can
        fall back to a silent video.
        """
        usable = [c for c in clips if c.ok and c.duration > 0]
        if not usable:
            return None

        total_samples = int(max(total_seconds, 0.1) * self.rate) + self.rate
        track = np.zeros(total_samples, dtype=np.float32)

        for clip in usable:
            samples = self._read_mono(clip.path)
            if samples.size == 0:
                continue
            offset = int(max(0.0, clip.start) * self.rate)
            end = min(total_samples, offset + samples.size)
            if end <= offset:
                continue
            track[offset:end] += samples[: end - offset]

        # Headroom: overlapping cadences can clip, so normalise if needed.
        peak = float(np.max(np.abs(track))) if track.size else 0.0
        if peak > 0.99:
            track *= 0.99 / peak

        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pcm = np.clip(track * 32767.0, -32768, 32767).astype(np.int16)

        with wave.open(str(out_path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(self.rate)
            handle.writeframes(pcm.tobytes())

        return out_path

    def _read_mono(self, path: Optional[Path]) -> np.ndarray:
        """Read a WAV as float32 mono at :attr:`rate`, resampling if needed."""
        if path is None:
            return np.zeros(0, dtype=np.float32)
        try:
            with wave.open(str(path), "rb") as handle:
                channels = handle.getnchannels()
                width = handle.getsampwidth()
                rate = handle.getframerate() or self.rate
                raw = handle.readframes(handle.getnframes())
        except Exception:
            return np.zeros(0, dtype=np.float32)

        if width == 2:
            data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        elif width == 1:
            data = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
        elif width == 4:
            data = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
        else:
            return np.zeros(0, dtype=np.float32)

        if channels > 1:
            data = data.reshape(-1, channels).mean(axis=1)

        if rate != self.rate and data.size:
            target_len = int(data.size * self.rate / float(rate))
            if target_len > 0:
                source_x = np.linspace(0.0, 1.0, data.size, endpoint=False)
                target_x = np.linspace(0.0, 1.0, target_len, endpoint=False)
                data = np.interp(target_x, source_x, data).astype(np.float32)

        # Short fade in/out removes clicks at clip boundaries.
        fade = min(int(0.015 * self.rate), data.size // 8)
        if fade > 1:
            ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
            data[:fade] *= ramp
            data[-fade:] *= ramp[::-1]

        return data
