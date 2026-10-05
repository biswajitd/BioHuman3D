"""
Narration translation with a reviewable translation memory.

Medical narration must be *accurate* in every language, so translation is
treated as content, not as a throwaway API call:

* every sentence is translated once and stored in
  ``%LOCALAPPDATA%/BioHuman3D/translations/<lang>.json`` together with the
  English source, the engine that produced it and a ``reviewed`` flag;
* a clinician or translator can edit that file; entries marked
  ``"reviewed": true`` are never overwritten by a machine;
* the cache is keyed by the exact English source, so editing a script
  re-translates only the sentences that changed.

Engines (chosen in Settings):

* ``llm``     the AI model already configured in BioHuman3D (local or cloud),
              with a prompt that enforces standard medical terminology;
* ``google``  Google Cloud Translation v2 (API key);
* ``azure``   Azure AI Translator (key + region).
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import threading
from dataclasses import replace
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence

from app.audio.languages import Language, language

try:
    import requests
except Exception:  # pragma: no cover
    requests = None  # type: ignore[assignment]


class TranslationError(RuntimeError):
    pass


def _key(text: str) -> str:
    return hashlib.sha1(text.strip().encode("utf-8")).hexdigest()[:16]


class TranslationMemory:
    """JSON store of translated sentences for one language."""

    def __init__(self, folder: Path, lang_code: str) -> None:
        self.path = Path(folder) / f"{lang_code}.json"
        self._lock = threading.Lock()
        self._data: Dict[str, dict] = {}
        if self.path.exists():
            try:
                self._data = json.loads(self.path.read_text(encoding="utf-8"))
            except Exception:
                self._data = {}

    def get(self, source: str) -> Optional[str]:
        entry = self._data.get(_key(source))
        return entry.get("text") if entry and entry.get("source", "").strip() == source.strip() else None

    def put(self, source: str, text: str, engine: str) -> None:
        with self._lock:
            old = self._data.get(_key(source))
            if old and old.get("reviewed"):
                return
            self._data[_key(source)] = {"source": source.strip(), "text": text.strip(),
                                        "engine": engine, "reviewed": False}

    def save(self) -> None:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self._data, ensure_ascii=False, indent=1), encoding="utf-8")

    def missing(self, sources: Iterable[str]) -> List[str]:
        return [s for s in sources if s.strip() and self.get(s) is None]


# ---------------------------------------------------------------------------
# Engines
# ---------------------------------------------------------------------------
PROMPT = (
    "You are a professional medical translator preparing narration for an anatomy and "
    "health education video. Translate the user's English text into {language} ({native}).\n"
    "Rules:\n"
    "- Use the standard medical and anatomical terminology taught in {language}-medium "
    "medical education. Where professionals in that language normally use the English or "
    "Latin term (for example muscle or nerve names), keep that term, written in the "
    "{language} script if that is the usual practice.\n"
    "- Keep every fact, number, unit, range and nerve-root level exactly as in the source.\n"
    "- Write natural spoken sentences for a voice-over, not a word-for-word gloss.\n"
    "- Output ONLY the translation, with no notes, quotes or transliteration."
)


def _llm_translate(text: str, lang: Language, provider, model: str) -> str:
    from app.ai.providers import ChatMessage
    messages = [ChatMessage("system", PROMPT.format(language=lang.name, native=lang.native)),
                ChatMessage("user", text)]
    out = provider.complete(messages, model, temperature=0.1, max_tokens=1200)
    out = re.sub(r"^\s*(translation\s*:)\s*", "", out.strip(), flags=re.I).strip().strip('"“”')
    if not out:
        raise TranslationError("the AI model returned an empty translation")
    return out


def _google_translate(texts: Sequence[str], lang: Language, key: str) -> List[str]:
    response = requests.post("https://translation.googleapis.com/language/translate/v2",
                             params={"key": key},
                             json={"q": list(texts), "source": "en", "target": lang.translate_to,
                                   "format": "text"}, timeout=60)
    if response.status_code >= 400:
        raise TranslationError(f"Google Translation HTTP {response.status_code}: {response.text[:200]}")
    return [t["translatedText"] for t in response.json()["data"]["translations"]]


def _azure_translate(texts: Sequence[str], lang: Language, key: str, region: str) -> List[str]:
    target = {"zh-CN": "zh-Hans", "pt-PT": "pt-pt", "fil": "fil"}.get(lang.translate_to, lang.translate_to)
    response = requests.post("https://api.cognitive.microsofttranslator.com/translate",
                             params={"api-version": "3.0", "from": "en", "to": target},
                             headers={"Ocp-Apim-Subscription-Key": key, "Ocp-Apim-Subscription-Region": region,
                                      "Content-Type": "application/json"},
                             json=[{"Text": t} for t in texts], timeout=60)
    if response.status_code >= 400:
        raise TranslationError(f"Azure Translator HTTP {response.status_code}: {response.text[:200]}")
    return [item["translations"][0]["text"] for item in response.json()]


class Translator:
    """Translate narration for a language, filling the translation memory."""

    def __init__(self, config, ai_manager=None) -> None:
        self._config = config
        self._ai = ai_manager
        self.folder = config.paths.user_data / "translations"

    def engine(self) -> str:
        return str(self._config.get("i18n.engine", "llm") or "llm")

    def memory(self, lang_code: str) -> TranslationMemory:
        return TranslationMemory(self.folder, lang_code)

    def ready(self) -> tuple:
        engine = self.engine()
        if engine == "google":
            key = self._google_key()
            return bool(key), "" if key else "Add a Google Cloud Translation API key in Settings."
        if engine == "azure":
            ok = bool(self._azure_key() and self._azure_region())
            return ok, "" if ok else "Add an Azure Translator key and region in Settings."
        provider = self._ai.current_provider if self._ai is not None else None
        model = self._ai.current_model() if self._ai is not None else ""
        ok = provider is not None and bool(model)
        return ok, "" if ok else ("No AI model is available for translation — start a local model "
                                  "(Ollama/LM Studio), add a cloud key, or choose Google/Azure "
                                  "translation in Settings.")

    def _google_key(self) -> str:
        return (self._config.get("i18n.google_key", "") or self._config.get("audio.google_key", "")
                or os.environ.get("GOOGLE_TRANSLATE_API_KEY", "")).strip()

    def _azure_key(self) -> str:
        return (self._config.get("i18n.azure_key", "") or os.environ.get("AZURE_TRANSLATOR_KEY", "")).strip()

    def _azure_region(self) -> str:
        return (self._config.get("i18n.azure_region", "") or self._config.get("audio.azure_region", "")
                or os.environ.get("AZURE_TRANSLATOR_REGION", "")).strip()

    def translate_all(self, sources: Sequence[str], lang_code: str,
                      progress: Optional[Callable[[int, str], None]] = None,
                      cancelled: Optional[Callable[[], bool]] = None) -> TranslationMemory:
        lang = language(lang_code)
        memory = self.memory(lang_code)
        if lang.is_english:
            return memory
        todo = memory.missing(dict.fromkeys(sources))
        engine = self.engine()
        ok, reason = self.ready()
        if todo and not ok:
            raise TranslationError(reason)
        done = 0
        batch = 20 if engine in ("google", "azure") else 1
        for start in range(0, len(todo), batch):
            if cancelled and cancelled():
                break
            chunk = todo[start:start + batch]
            if engine == "google":
                results = _google_translate(chunk, lang, self._google_key())
            elif engine == "azure":
                results = _azure_translate(chunk, lang, self._azure_key(), self._azure_region())
            else:
                results = [_llm_translate(chunk[0], lang, self._ai.current_provider, self._ai.current_model())]
            for src, out in zip(chunk, results):
                memory.put(src, out, engine)
            done += len(chunk)
            memory.save()
            if progress:
                progress(int(done * 100 / max(1, len(todo))), f"Translating to {lang.name}: {done}/{len(todo)}")
        memory.save()
        return memory


# ---------------------------------------------------------------------------
# Tours
# ---------------------------------------------------------------------------
def tour_sources(tour) -> List[str]:
    out = [tour.title]
    for kf in tour.keyframes:
        out += [kf.title, kf.narration]
    return [s for s in out if s and s.strip()]


def localized_tour(tour, lang_code: str, memory: TranslationMemory):
    """Copy of *tour* with translated titles and narration (effects shared)."""
    from app.audio.scripts import Tour, register_tour
    lang = language(lang_code)
    if lang.is_english:
        return tour

    def tr(text: str) -> str:
        return memory.get(text) or text

    keyframes = [replace(kf, title=tr(kf.title), narration=tr(kf.narration)) for kf in tour.keyframes]
    local = Tour(id=f"{tour.id}@{lang.code}", title=tr(tour.title),
                 subtitle=f"{tour.subtitle} · {lang.native}" if tour.subtitle else lang.native,
                 system=tour.system, keyframes=keyframes)
    return register_tour(local)
