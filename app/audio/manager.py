"""
AudioManager — text-to-speech for anatomy voiceovers.

Design notes that matter in practice
-----------------------------------
1. **pyttsx3 must be created inside the worker thread.** On Windows the SAPI5
   driver binds to the COM apartment of the creating thread; building the engine
   on the GUI thread and calling ``runAndWait()`` from a worker silently produces
   no audio or crashes.
2. **Cues, not strings.** Every utterance is a :class:`NarrationCue` carrying an
   index so the camera tour can synchronise to it and the transcript can
   highlight the active paragraph.
3. **Cloud voices are cached.** ElevenLabs output is written to
   ``assets/audio/cache/<sha1>.mp3`` and replayed through ``pygame.mixer`` — a
   replayed tour costs nothing.
4. **Graceful degradation.** Missing ``pyttsx3``/``pygame``/API key downgrades to
   a silent backend that still emits timing signals, so the UI logic stays
   testable and the tours still advance.
"""
from __future__ import annotations

import hashlib
import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

from PyQt6.QtCore import QObject, QThread, pyqtSignal

from app.audio.languages import ENGINES, LANGUAGES, NEURAL_ENGINES, language
from app.audio.neural_tts import TTSError, engine_ready, from_config, synthesize
from app.audio.voices import (ANY_DIALECT, DIALECT_BY_CODE, DIALECTS, GENDER_ANY, VoiceOption,
                              dialects_available, enumerate_elevenlabs_voices,
                              enumerate_pyttsx3_voices, resolve_voice)
from app.core.wincom import co_initialize, co_uninitialize

# ---------------------------------------------------------------------------
# Optional dependencies
# ---------------------------------------------------------------------------
try:
    import pyttsx3
    PYTTsx3_AVAILABLE = True
except Exception:
    pyttsx3 = None  # type: ignore[assignment]
    PYTTsx3_AVAILABLE = False

try:
    import pygame
    PYGAME_AVAILABLE = True
except Exception:
    pygame = None  # type: ignore[assignment]
    PYGAME_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except Exception:
    requests = None  # type: ignore[assignment]
    REQUESTS_AVAILABLE = False


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
@dataclass
class Voice:
    """A selectable system or cloud voice."""

    id: str
    name: str
    backend: str = "pyttsx3"
    language: str = ""
    gender: str = ""

    def __str__(self) -> str:
        suffix = f" · {self.language}" if self.language else ""
        return f"{self.name}{suffix}"


@dataclass
class NeuralVoice:
    """A cloud neural voice for the selected language (display + id only)."""

    id: str
    language: str
    gender: str
    engine: str

    @property
    def display(self) -> str:
        lang = language(self.language)
        short = self.id.split("-")[-1].replace("Neural", "") if "Neural" in self.id else self.id
        return f"{short} — {lang.name} — {self.gender.capitalize()} (neural)"

    def __str__(self) -> str:
        return self.display


@dataclass
class NarrationCue:
    """One spoken paragraph, optionally bound to a camera keyframe."""

    index: int
    text: str
    title: str = ""
    camera: Optional[Dict[str, Sequence[float]]] = None
    layer_states: Dict[str, float] = field(default_factory=dict)
    duration_hint: float = 0.0


# ---------------------------------------------------------------------------
# Worker thread
# ---------------------------------------------------------------------------
class NarrationWorker(QThread):
    """Owns the speech engine and drains a cue queue."""

    cueStarted = pyqtSignal(int, str)     # index, text
    wordSpoken = pyqtSignal(str)
    cueFinished = pyqtSignal(int)
    failed = pyqtSignal(str)

    def __init__(self, *, rate: int = 175, volume: float = 0.9,
                 voice_id: str = "", backend: str = "pyttsx3",
                 cache_dir: Optional[Path] = None, api_key: str = "",
                 elevenlabs_voice: str = "", neural=None,
                 parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._queue: "queue.Queue[Optional[NarrationCue]]" = queue.Queue()
        self._rate = rate
        self._volume = volume
        self._voice_id = voice_id
        self._backend = backend
        self._cache_dir = cache_dir
        self._api_key = api_key
        self._el_voice = elevenlabs_voice
        self._neural = neural
        self._neural_failed = False

        self._stop_flag = threading.Event()
        self._engine = None
        self._engine_lock = threading.Lock()
        self._mixer_ready = False

    # -- public API (called from the GUI thread) ---------------------------
    def enqueue(self, cue: NarrationCue) -> None:
        self._queue.put(cue)

    def stop_speaking(self) -> None:
        """Stop the current utterance; the queue is preserved."""
        with self._engine_lock:
            if self._engine is not None:
                try:
                    self._engine.stop()
                except Exception:
                    pass

    def shutdown(self) -> None:
        self._stop_flag.set()
        self.stop_speaking()
        self._queue.put(None)

    # -- internals ---------------------------------------------------------
    def _configure_engine(self, engine) -> None:
        """Apply rate/volume/voice and attach the word-tracking callback."""
        engine.setProperty("rate", int(self._rate))
        engine.setProperty("volume", float(self._volume))
        if self._voice_id:
            for voice in engine.getProperty("voices"):
                if voice.id == self._voice_id:
                    engine.setProperty("voice", voice.id)
                    break

        def _on_word(name, location, length):
            # Not emitted by every SAPI voice; harmless when unsupported.
            self.wordSpoken.emit(name)

        try:
            engine.connect("started-word", _on_word)
        except Exception:
            pass

    def _ensure_mixer(self) -> bool:
        if not PYGAME_AVAILABLE:
            return False
        if not self._mixer_ready:
            try:
                pygame.mixer.init(frequency=44100, channels=1, buffer=1024)  # type: ignore[union-attr]
                self._mixer_ready = True
            except Exception as exc:
                self.failed.emit(f"Audio device unavailable: {exc}")
                return False
        return True

    def _cache_path(self, text: str) -> Path:
        digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]
        base = self._cache_dir or Path.cwd()
        base.mkdir(parents=True, exist_ok=True)
        return base / f"{digest}.mp3"

    def _speak_elevenlabs(self, cue: NarrationCue) -> bool:
        """Render (or replay from cache) a premium cloud voice."""
        if not (REQUESTS_AVAILABLE and self._api_key):
            return False
        path = self._cache_path(cue.text)
        if not path.exists():
            try:
                response = requests.post(  # type: ignore[union-attr]
                    f"https://api.elevenlabs.io/v1/text-to-speech/{self._el_voice}",
                    headers={"xi-api-key": self._api_key,
                             "Content-Type": "application/json"},
                    json={"text": cue.text,
                          "model_id": "eleven_multilingual_v2",
                          "voice_settings": {"stability": 0.45, "similarity_boost": 0.75}},
                    timeout=45,
                )
                if response.status_code >= 400:
                    self.failed.emit(f"ElevenLabs HTTP {response.status_code}")
                    return False
                path.write_bytes(response.content)
            except Exception as exc:
                self.failed.emit(f"ElevenLabs request failed: {exc}")
                return False

        if not self._ensure_mixer():
            return False
        try:
            pygame.mixer.music.load(str(path))       # type: ignore[union-attr]
            pygame.mixer.music.set_volume(float(self._volume))  # type: ignore[union-attr]
            pygame.mixer.music.play()                # type: ignore[union-attr]
            while pygame.mixer.music.get_busy():     # type: ignore[union-attr]
                if self._stop_flag.is_set():
                    pygame.mixer.music.stop()        # type: ignore[union-attr]
                    return True
                self.msleep(120)
            return True
        except Exception as exc:
            self.failed.emit(f"Playback failed: {exc}")
            return False

    def _play_wav(self, path: Path) -> bool:
        """Play a WAV and block until it ends (or stop is requested)."""
        if self._ensure_mixer():
            try:
                pygame.mixer.music.load(str(path))            # type: ignore[union-attr]
                pygame.mixer.music.set_volume(float(self._volume))  # type: ignore[union-attr]
                pygame.mixer.music.play()                     # type: ignore[union-attr]
                while pygame.mixer.music.get_busy():          # type: ignore[union-attr]
                    if self._stop_flag.is_set():
                        pygame.mixer.music.stop()             # type: ignore[union-attr]
                        return True
                    self.msleep(80)
                return True
            except Exception as exc:
                self.failed.emit(f"Playback failed: {exc}")
        import sys
        if sys.platform == "win32":                           # no pygame: Windows' own player
            try:
                import winsound
                winsound.PlaySound(str(path), winsound.SND_FILENAME)
                return True
            except Exception as exc:
                self.failed.emit(f"Playback failed: {exc}")
        return False

    def _speak_neural(self, cue: NarrationCue) -> bool:
        if self._neural is None or self._neural_failed:
            return False
        try:
            path = synthesize(cue.text, self._neural, self._cache_dir or Path.cwd())
        except TTSError as exc:
            # Report once, then fall back to the system voice for the rest of the tour.
            self._neural_failed = True
            self.failed.emit(f"Neural voice unavailable — using the system voice. {exc}")
            return False
        return self._play_wav(path)

    def _speak_pyttsx3(self, text: str) -> bool:
        """Speak one utterance with a **fresh** engine.

        Never reuse a single pyttsx3 engine across consecutive utterances. On
        Windows/SAPI5 a second ``runAndWait`` on the same engine hangs forever
        or fail-fast crashes the process (0xC0000409) — reproduced in isolation
        with three consecutive cues. A fresh engine per utterance is reliable;
        creation costs a few hundred milliseconds, which is negligible next to
        the speech itself.
        """
        if not PYTTsx3_AVAILABLE:
            return False

        try:
            engine = pyttsx3.init()                  # type: ignore[union-attr]
        except Exception as exc:
            self.failed.emit(f"Could not start the speech engine: {exc}")
            return False

        # Published so stop_speaking() can interrupt the active utterance.
        with self._engine_lock:
            self._engine = engine

        try:
            self._configure_engine(engine)
            engine.say(text)
            engine.runAndWait()
            return True
        except Exception as exc:
            self.failed.emit(f"Text-to-speech failed: {exc}")
            return False
        finally:
            with self._engine_lock:
                self._engine = None
            try:
                engine.stop()
            except Exception:
                pass
            del engine

    # -- thread body -------------------------------------------------------
    def run(self) -> None:  # noqa: D102
        # COM must be initialised on this thread before any speech object exists.
        co_initialize()

        while not self._stop_flag.is_set():
            try:
                cue = self._queue.get(timeout=0.25)
            except queue.Empty:
                continue
            if cue is None:                          # shutdown sentinel
                break
            if self._stop_flag.is_set():
                break

            self.cueStarted.emit(cue.index, cue.text)

            spoken = False
            if self._backend in ("edge", "azure", "google"):
                spoken = self._speak_neural(cue)
            elif self._backend == "elevenlabs":
                spoken = self._speak_elevenlabs(cue)
            if not spoken and self._backend != "none":
                spoken = self._speak_pyttsx3(cue.text)
            if not spoken:
                # Silent fallback: pace the cue so tours still advance sensibly.
                estimate = cue.duration_hint or max(2.0, len(cue.text.split()) / 2.6)
                deadline = time.time() + estimate
                while time.time() < deadline and not self._stop_flag.is_set():
                    self.msleep(80)

            self.cueFinished.emit(cue.index)

        self.stop_speaking()
        co_uninitialize()


# ---------------------------------------------------------------------------
# Facade
# ---------------------------------------------------------------------------
class AudioManager(QObject):
    """Queue-based narration controller used by the UI and the tour engine."""

    cueStarted = pyqtSignal(int, str)
    cueFinished = pyqtSignal(int)
    wordSpoken = pyqtSignal(str)
    queueChanged = pyqtSignal(int)
    stateChanged = pyqtSignal(bool)          # True = speaking
    failed = pyqtSignal(str)

    def __init__(self, config, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._config = config
        self._worker: Optional[NarrationWorker] = None
        self._queue: List[NarrationCue] = []
        self._index = 0
        self._speaking = False
        self._voices_cache: List[VoiceOption] = []
        self._resolved: Optional[VoiceOption] = None
        self._resolve_note = ""

    # ----------------------------------------------------------- lifecycle
    def _worker_options(self) -> Dict:
        return {
            "rate": int(self._config.get("audio.rate", 175)),
            "volume": float(self._config.get("audio.volume", 0.9)),
            "voice_id": str(self._config.get("audio.voice", "") or ""),
            "backend": str(self._config.get("audio.backend", "pyttsx3")),
            "cache_dir": self._config.paths.audio_cache,
            "api_key": self._config.api_key("elevenlabs"),
            "elevenlabs_voice": str(self._config.get("audio.elevenlabs_voice", "")),
            "neural": self.neural_request(),
        }

    # ------------------------------------------------------------ neural / language
    def engine(self) -> str:
        return str(self._config.get("audio.backend", "edge"))

    def neural_request(self):
        """Voice request for the configured neural engine, or ``None``."""
        if self.engine() not in ("edge", "azure", "google"):
            return None
        return from_config(self._config)

    def engine_status(self) -> tuple:
        """``(ok, reason)`` for the configured engine."""
        engine = self.engine()
        if engine == "none":
            return True, ""
        if engine == "pyttsx3":
            return PYTTsx3_AVAILABLE, "" if PYTTsx3_AVAILABLE else "pip install pyttsx3"
        return engine_ready(from_config(self._config))

    def current_language(self) -> str:
        code = str(self._config.get("audio.language", "") or "")
        if not code:
            dialect = str(self._config.get("audio.dialect", "") or "")
            code = dialect if dialect and dialect != ANY_DIALECT else "en-IN"
        return code

    def set_language(self, code: str) -> None:
        self._config["audio.language"] = code
        # The offline engine resolves by dialect; keep it in step where it can.
        self._config["audio.dialect"] = code if code in DIALECT_BY_CODE else ANY_DIALECT
        self._config["audio.voice"] = ""
        self._config["audio.neural_voice"] = ""
        self.resolve_selection(apply=True)

    def _ensure_worker(self) -> NarrationWorker:
        if self._worker is None or not self._worker.isRunning():
            self._worker = NarrationWorker(parent=self, **self._worker_options())
            self._worker.cueStarted.connect(self._on_cue_started)
            self._worker.cueFinished.connect(self._on_cue_finished)
            self._worker.wordSpoken.connect(self.wordSpoken)
            self._worker.failed.connect(self.failed)
            self._worker.start()
        return self._worker

    def shutdown(self) -> None:
        if self._worker is not None:
            self._worker.shutdown()
            self._worker.wait(2500)

    # ------------------------------------------------------------- speaking
    def is_available(self) -> bool:
        backend = self.engine()
        if backend == "none":
            return False
        if backend == "elevenlabs":
            return bool(self._config.api_key("elevenlabs"))
        if backend in ("edge", "azure", "google"):
            return engine_ready(from_config(self._config))[0] or PYTTsx3_AVAILABLE
        return PYTTsx3_AVAILABLE

    def available_backends(self) -> List[str]:
        """Every engine; the panel marks the ones that still need setup."""
        return [key for key, _label in ENGINES]

    def speak(self, text: str, *, index: int = -1, title: str = "",
              interrupt: bool = True) -> None:
        """Speak a one-off line (used by the 'Read answer aloud' button)."""
        text = (text or "").strip()
        if not text or not self._config.get("audio.enabled", True):
            return
        if interrupt:
            self.stop()
        self._ensure_worker().enqueue(
            NarrationCue(index=index if index >= 0 else self._next_index(),
                         text=self._clean(text), title=title)
        )

    def play_cues(self, cues: Sequence[NarrationCue], *, start_at: int = 0) -> None:
        """Queue a full narration script (a guided tour)."""
        self.stop()
        self._queue = [c for c in cues if c.text.strip()]
        self._index = max(0, min(start_at, len(self._queue) - 1))
        self.queueChanged.emit(len(self._queue))
        self._pump()

    def _pump(self) -> None:
        if self._index >= len(self._queue):
            return
        worker = self._ensure_worker()
        for cue in self._queue[self._index:]:
            worker.enqueue(cue)

    def stop(self) -> None:
        self._queue.clear()
        self._index = 0
        self.queueChanged.emit(0)
        if self._worker is not None:
            self._worker.stop_speaking()
            # Drain any pending cues still sitting in the thread's queue.
            try:
                while True:
                    self._worker._queue.get_nowait()  # noqa: SLF001 - internal drain
            except Exception:
                pass
        self._speaking = False
        self.stateChanged.emit(False)

    def pause(self) -> None:
        """pyttsx3 exposes no portable pause; stopping is the honest behaviour."""
        self.stop()

    # ------------------------------------------------------------- settings
    def apply_settings(self) -> None:
        """Recreate the worker so rate/volume/voice changes take effect."""
        was_speaking = self._speaking
        self.stop()
        if self._worker is not None:
            self._worker.shutdown()
            self._worker.wait(1500)
            self._worker = None
        if was_speaking:
            self._ensure_worker()

    def voices(self, refresh: bool = False) -> List[VoiceOption]:
        """Installed voices, classified by dialect and gender (cached).

        For neural engines: the female and male neural voice of the language."""
        if self.engine() in ("edge", "azure", "google"):
            lang = language(self.current_language())
            return [NeuralVoice(lang.female, lang.code, "female", self.engine()),
                    NeuralVoice(lang.male, lang.code, "male", self.engine())]
        if self._voices_cache and not refresh:
            return self._voices_cache
        self._voices_cache = enumerate_pyttsx3_voices()
        if self._config.api_key("elevenlabs"):
            self._voices_cache += enumerate_elevenlabs_voices(
                self._config.api_key("elevenlabs"))
        return self._voices_cache

    # -------------------------------------------------------- voice catalogue
    def dialect_options(self) -> List[tuple]:
        """``(code, label)`` pairs for the UI. Dialects with no installed voice
        are marked rather than hidden, so a missing Indian English voice is
        visible as a gap to fix instead of silently absent."""
        neural = self.engine() not in ("pyttsx3", "none")
        installed = set() if neural else set(dialects_available(self.voices()))
        options = []
        last_group = ""
        for lang in LANGUAGES:
            if lang.group != last_group:
                options.append(("", f"── {lang.group} ──"))
                last_group = lang.group
            suffix = ""
            if not neural:
                if lang.code not in DIALECT_BY_CODE:
                    suffix = "   (neural voices only)"
                elif lang.code not in installed:
                    suffix = "   (not installed)"
            options.append((lang.code, f"{lang.label}{suffix}"))
        return options

    def current_dialect(self) -> str:
        """Selected narration language (BCP-47); kept for older call sites."""
        return self.current_language()

    def current_gender(self) -> str:
        return str(self._config.get("audio.gender", GENDER_ANY) or GENDER_ANY)

    def set_dialect(self, code: str) -> None:
        if code:
            self.set_language(code)

    def set_gender(self, gender: str) -> None:
        self._config["audio.gender"] = gender or GENDER_ANY
        self._config["audio.voice"] = ""
        self.resolve_selection(apply=True)

    def set_voice(self, voice_id: str) -> None:
        """Pin an explicit voice, overriding dialect/gender resolution."""
        if self.engine() in ("edge", "azure", "google"):
            lang = language(self.current_language())
            self._config["audio.neural_voice"] = voice_id or ""
            if voice_id in (lang.female, lang.male):
                self._config["audio.gender"] = "male" if voice_id == lang.male else "female"
            self.resolve_selection(apply=True)
            return
        self._config["audio.voice"] = voice_id or ""
        self._resolved = None
        self._resolve_note = ""
        self.resolve_selection(apply=True)

    def resolve_selection(self, apply: bool = True) -> tuple:
        """Resolve dialect + gender to a concrete voice.

        Returns ``(voice, note)``. ``note`` is non-empty whenever a fallback was
        applied, so the UI can explain *why* the narration is not in the
        requested dialect instead of silently switching accent.
        """
        if self.engine() in ("edge", "azure", "google"):
            req = from_config(self._config)
            voice = NeuralVoice(req.voice_name(), req.language, req.gender, req.engine)
            ok, reason = engine_ready(req)
            note = "" if ok else f"{reason} Until then the offline system voice is used."
            if not language(req.language).is_english:
                note = (note + " " if note else "") + (
                    f"Narration is translated to {language(req.language).name} before it is spoken.")
            self._resolved, self._resolve_note = voice, note
            if apply:
                self.apply_settings()
            return voice, note

        catalogue = self.voices()

        # An explicitly pinned voice wins over dialect/gender resolution, so a
        # deliberate choice is not silently undone on the next settings change.
        pinned = str(self._config.get("audio.voice", "") or "")
        if pinned:
            match = next((v for v in catalogue if v.id == pinned), None)
            if match is not None:
                self._resolved = match
                self._resolve_note = ""
                return match, ""

        voice, note = resolve_voice(catalogue, self.current_dialect(),
                                    self.current_gender())
        self._resolved = voice
        self._resolve_note = note
        if voice is not None and apply:
            self._config["audio.voice"] = voice.id
            self.apply_settings()
        return voice, note

    def selection_summary(self) -> str:
        """Single line describing the narration voice that will actually be used."""
        if self._resolved is None:
            self.resolve_selection(apply=False)
        if self._resolved is None:
            return "No speech voice available"
        base = f"Narration voice: {self._resolved.display}"
        return f"{base}  ·  {self._resolve_note}" if self._resolve_note else base

    def resolve_note(self) -> str:
        if self._resolved is None and not self._resolve_note:
            self.resolve_selection(apply=False)
        return self._resolve_note

    # ------------------------------------------------------------ internals
    def _on_cue_started(self, index: int, _text: str) -> None:
        self._speaking = True
        self.stateChanged.emit(True)
        self.cueStarted.emit(index, _text)

    def _on_cue_finished(self, index: int) -> None:
        is_last_cue = bool(self._queue) and index == self._queue[-1].index
        if is_last_cue:
            self._speaking = False
            self.stateChanged.emit(False)
        self.cueFinished.emit(index)

    def _next_index(self) -> int:
        self._index += 1
        return self._index

    @staticmethod
    def _clean(text: str) -> str:
        """Strip markdown so the TTS does not read asterisks and hashes aloud."""
        import re
        text = re.sub(r"```.*?```", " code block omitted ", text, flags=re.S)
        text = re.sub(r"[*_`>#]+", "", text)
        text = re.sub(r"^\s*[-•]\s*", "", text, flags=re.M)
        text = re.sub(r"\[(.*?)\]\(.*?\)", r"\1", text)
        return re.sub(r"\s{2,}", " ", text).strip()
