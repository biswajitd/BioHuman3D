"""
Neural text-to-speech: natural human-like voices in many languages.

``synthesize(text, request)`` returns a WAV file (cached by content hash), so
live narration and video export share one code path and a repeated tour costs
nothing. Engines:

* ``edge``   Microsoft neural voices through the Edge read-aloud service
             (``pip install edge-tts``). No key; needs internet. Intended for
             personal and evaluation use — use ``azure`` for production.
* ``azure``  Azure AI Speech REST API with your key + region. Same voices,
             production SLA and licensing.
* ``google`` Google Cloud Text-to-Speech REST API with your API key; the best
             available voice per language/gender is picked from Google's live
             catalogue (Chirp3-HD > Neural2 > WaveNet > Standard).
* ``elevenlabs`` ElevenLabs ``eleven_multilingual_v2`` with your voice id.

All engines are called from worker threads only.
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
from xml.sax.saxutils import escape

from app.audio.languages import Language, language

try:
    import requests
except Exception:  # pragma: no cover
    requests = None  # type: ignore[assignment]

try:
    import edge_tts  # type: ignore
    EDGE_AVAILABLE = True
except Exception:  # pragma: no cover
    edge_tts = None  # type: ignore[assignment]
    EDGE_AVAILABLE = False

#: Narration rate in the UI is words-per-minute on the old SAPI scale; neural
#: voices speak naturally at roughly this pace, so it maps to 0 % adjustment.
BASE_WPM = 175


class TTSError(RuntimeError):
    pass


@dataclass
class VoiceRequest:
    engine: str                       # edge | azure | google | elevenlabs
    language: str = "en-IN"           # BCP-47
    gender: str = "female"            # female | male
    voice: str = ""                   # explicit voice id (overrides language/gender)
    rate_wpm: int = BASE_WPM
    azure_key: str = ""
    azure_region: str = ""
    google_key: str = ""
    elevenlabs_key: str = ""
    elevenlabs_voice: str = ""

    @property
    def lang(self) -> Language:
        return language(self.language)

    def voice_name(self) -> str:
        if self.voice:
            return self.voice
        return self.lang.voice("male" if self.gender == "male" else "female")

    def rate_percent(self) -> int:
        return int(max(-50, min(80, round((self.rate_wpm / BASE_WPM - 1.0) * 100))))

    def cache_key(self, text: str) -> str:
        ident = "|".join([self.engine, self.language, self.gender, self.voice_name(),
                          str(self.rate_percent()), self.elevenlabs_voice if self.engine == "elevenlabs" else "",
                          text])
        return hashlib.sha1(ident.encode("utf-8")).hexdigest()[:20]


def from_config(config) -> VoiceRequest:
    """Build a request from :class:`AppConfig` (keys from config or environment)."""
    import os
    gender = str(config.get("audio.gender", "female") or "female")
    return VoiceRequest(
        engine=str(config.get("audio.backend", "edge")),
        language=str(config.get("audio.language", "en-IN") or "en-IN"),
        gender="male" if gender == "male" else "female",
        voice=str(config.get("audio.neural_voice", "") or ""),
        rate_wpm=int(config.get("audio.rate", BASE_WPM)),
        azure_key=(config.get("audio.azure_key", "") or os.environ.get("AZURE_SPEECH_KEY", "")).strip(),
        azure_region=(config.get("audio.azure_region", "") or os.environ.get("AZURE_SPEECH_REGION", "")).strip(),
        google_key=(config.get("audio.google_key", "") or os.environ.get("GOOGLE_TTS_API_KEY", "")).strip(),
        elevenlabs_key=config.api_key("elevenlabs"),
        elevenlabs_voice=str(config.get("audio.elevenlabs_voice", "") or ""),
    )


def engine_ready(request: VoiceRequest) -> tuple:
    """``(ok, reason)`` — whether *request.engine* can run at all."""
    if request.engine == "edge":
        return (EDGE_AVAILABLE, "" if EDGE_AVAILABLE else "Install the free neural voices:  pip install edge-tts")
    if request.engine == "azure":
        ok = bool(request.azure_key and request.azure_region)
        return ok, "" if ok else "Add your Azure Speech key and region in Settings."
    if request.engine == "google":
        ok = bool(request.google_key)
        return ok, "" if ok else "Add your Google Cloud Text-to-Speech API key in Settings."
    if request.engine == "elevenlabs":
        ok = bool(request.elevenlabs_key and request.elevenlabs_voice)
        return ok, "" if ok else "Add your ElevenLabs key (and voice id) in Settings."
    return False, f"{request.engine} is not a neural engine"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def synthesize(text: str, request: VoiceRequest, cache_dir: Path) -> Path:
    """Render *text* to a mono WAV (cached). Raises :class:`TTSError`."""
    text = (text or "").strip()
    if not text:
        raise TTSError("nothing to say")
    ok, reason = engine_ready(request)
    if not ok:
        raise TTSError(reason)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / f"{request.engine}_{request.cache_key(text)}.wav"
    if target.exists() and target.stat().st_size > 1024:
        return target

    tmp_dir = Path(tempfile.mkdtemp(prefix="bh3d_tts_"))
    try:
        if request.engine == "edge":
            raw = _edge(text, request, tmp_dir)
        elif request.engine == "azure":
            raw = _azure(text, request, tmp_dir)
        elif request.engine == "google":
            raw = _google(text, request, tmp_dir)
        elif request.engine == "elevenlabs":
            raw = _elevenlabs(text, request, tmp_dir)
        else:
            raise TTSError(f"unknown engine {request.engine}")
        if raw.suffix.lower() == ".wav":
            shutil.copy2(raw, target)
        else:
            to_wav(raw, target)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    if not target.exists() or target.stat().st_size <= 1024:
        raise TTSError("the speech service returned no audio")
    return target


def to_wav(source: Path, target: Path, rate: int = 24000) -> None:
    """Decode any audio (MP3 from cloud engines) to mono 16-bit WAV via ffmpeg."""
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:
        raise TTSError("Decoding neural audio needs imageio-ffmpeg:  pip install imageio-ffmpeg") from exc
    flags = 0x08000000 if hasattr(subprocess, "CREATE_NO_WINDOW") else 0   # no console flash on Windows
    proc = subprocess.run([exe, "-y", "-loglevel", "error", "-i", str(source), "-ac", "1",
                           "-ar", str(rate), "-sample_fmt", "s16", str(target)],
                          capture_output=True, timeout=120, creationflags=flags)
    if proc.returncode != 0:
        raise TTSError(f"audio decode failed: {proc.stderr.decode(errors='replace')[-200:]}")


# ---------------------------------------------------------------------------
# Engines
# ---------------------------------------------------------------------------
def _edge(text: str, req: VoiceRequest, tmp: Path) -> Path:
    out = tmp / "speech.mp3"

    async def run() -> None:
        communicate = edge_tts.Communicate(text, req.voice_name(), rate=f"{req.rate_percent():+d}%")
        await communicate.save(str(out))

    try:
        asyncio.run(run())
    except Exception as exc:
        raise TTSError(f"Microsoft neural voice unavailable ({type(exc).__name__}: {exc}). "
                       "Check the internet connection, or choose Azure/Google with your own key.") from exc
    return out


def _azure(text: str, req: VoiceRequest, tmp: Path) -> Path:
    if requests is None:
        raise TTSError("requests is not installed")
    ssml = (f"<speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis' xml:lang='{req.language}'>"
            f"<voice name='{req.voice_name()}'><prosody rate='{req.rate_percent():+d}%'>"
            f"{escape(text)}</prosody></voice></speak>")
    response = requests.post(
        f"https://{req.azure_region}.tts.speech.microsoft.com/cognitiveservices/v1",
        headers={"Ocp-Apim-Subscription-Key": req.azure_key,
                 "Content-Type": "application/ssml+xml",
                 "X-Microsoft-OutputFormat": "riff-24khz-16bit-mono-pcm",
                 "User-Agent": "BioHuman3D"},
        data=ssml.encode("utf-8"), timeout=60)
    if response.status_code >= 400:
        raise TTSError(f"Azure Speech HTTP {response.status_code}: {response.text[:200]}")
    out = tmp / "speech.wav"
    out.write_bytes(response.content)
    return out


_google_voices: Dict[str, List[dict]] = {}


def _google_voice(req: VoiceRequest) -> Optional[str]:
    """Best Google voice for the language and gender, from the live catalogue."""
    if req.voice and not req.voice.endswith("Neural"):
        return req.voice
    if req.language not in _google_voices:
        response = requests.get("https://texttospeech.googleapis.com/v1/voices",
                                params={"languageCode": req.language, "key": req.google_key}, timeout=30)
        _google_voices[req.language] = response.json().get("voices", []) if response.ok else []
    wanted = "MALE" if req.gender == "male" else "FEMALE"
    ranked = []
    for v in _google_voices[req.language]:
        if req.language not in v.get("languageCodes", []):
            continue
        name = v.get("name", "")
        quality = (4 if "Chirp3-HD" in name else 3 if "Neural2" in name else
                   2 if "Wavenet" in name else 1 if "Standard" in name else 0)
        ranked.append(((v.get("ssmlGender") == wanted), quality, name))
    ranked.sort(reverse=True)
    return ranked[0][2] if ranked else None


def _google(text: str, req: VoiceRequest, tmp: Path) -> Path:
    if requests is None:
        raise TTSError("requests is not installed")
    voice: Dict[str, str] = {"languageCode": req.language,
                             "ssmlGender": "MALE" if req.gender == "male" else "FEMALE"}
    name = _google_voice(req)
    if name:
        voice["name"] = name
    response = requests.post(
        "https://texttospeech.googleapis.com/v1/text:synthesize",
        params={"key": req.google_key},
        json={"input": {"text": text}, "voice": voice,
              "audioConfig": {"audioEncoding": "LINEAR16", "sampleRateHertz": 24000,
                              "speakingRate": max(0.5, min(1.8, req.rate_wpm / BASE_WPM))}},
        timeout=60)
    if response.status_code >= 400:
        raise TTSError(f"Google TTS HTTP {response.status_code}: {response.text[:200]}")
    out = tmp / "speech.wav"
    out.write_bytes(base64.b64decode(response.json()["audioContent"]))
    return out


def _elevenlabs(text: str, req: VoiceRequest, tmp: Path) -> Path:
    if requests is None:
        raise TTSError("requests is not installed")
    response = requests.post(
        f"https://api.elevenlabs.io/v1/text-to-speech/{req.elevenlabs_voice}",
        headers={"xi-api-key": req.elevenlabs_key, "Content-Type": "application/json"},
        json={"text": text, "model_id": "eleven_multilingual_v2",
              "voice_settings": {"stability": 0.45, "similarity_boost": 0.75}},
        timeout=90)
    if response.status_code >= 400:
        raise TTSError(f"ElevenLabs HTTP {response.status_code}")
    out = tmp / "speech.mp3"
    out.write_bytes(response.content)
    return out
