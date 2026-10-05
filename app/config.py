"""
Application configuration.

Two distinct concerns live here:

* ``Paths``     — immutable, computed filesystem locations for assets/caches.
* ``AppConfig`` — user-mutable preferences persisted through ``QSettings``.

Keeping them separate means tests can build a ``Paths`` object pointing at a
temporary directory without touching the registry.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Filesystem layout
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Paths:
    """Resolved project directories. Call :meth:`ensure` before writing."""

    root: Path = PROJECT_ROOT
    assets: Path = PROJECT_ROOT / "app" / "assets"
    models: Path = PROJECT_ROOT / "app" / "assets" / "models"
    audio: Path = PROJECT_ROOT / "app" / "assets" / "audio"
    audio_cache: Path = PROJECT_ROOT / "app" / "assets" / "audio" / "cache"
    icons: Path = PROJECT_ROOT / "app" / "assets" / "icons"
    qss: Path = PROJECT_ROOT / "app" / "ui" / "style.qss"
    user_data: Path = field(
        default_factory=lambda: Path(
            os.environ.get("LOCALAPPDATA")
            or Path.home() / ".local" / "share"
        ) / "BioHuman3D"
    )

    def ensure(self) -> "Paths":
        """Create every directory that the app may write into."""
        for directory in (
            self.assets, self.models, self.audio,
            self.audio_cache, self.icons, self.user_data,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        return self

    @property
    def settings_file(self) -> Path:
        return self.user_data / "settings.json"

    @property
    def chat_log(self) -> Path:
        return self.user_data / "chat_history.jsonl"


# ---------------------------------------------------------------------------
# User preferences
# ---------------------------------------------------------------------------
DEFAULTS: Dict[str, Any] = {
    # ── rendering ─────────────────────────────────────────────────────────
    "render.fxaa": True,
    "render.depth_peeling": True,
    "render.background": "studio",          # studio | slate | black | clinical
    "render.show_axes": True,

    # ── AI ─────────────────────────────────────────────────────────────────
    "ai.mode": "auto",                       # auto | local | cloud
    "ai.provider": "",                       # provider id, empty = auto-pick
    "ai.model": "",
    "ai.temperature": 0.3,
    "ai.max_tokens": 900,
    "ai.openai_key": "",
    "ai.anthropic_key": "",
    "ai.openai_model": "gpt-4o",
    "ai.anthropic_model": "claude-3-5-sonnet-latest",

    # ── audio ──────────────────────────────────────────────────────────────
    "audio.enabled": True,
    "audio.backend": "edge",                 # edge | azure | google | elevenlabs | pyttsx3 | none
    "audio.language": "en-IN",               # BCP-47 narration language
    "audio.gender": "female",                # female | male
    "audio.neural_voice": "",                # explicit neural voice id (optional)
    "audio.azure_key": "",
    "audio.azure_region": "",                # e.g. centralindia, eastus
    "audio.google_key": "",
    "i18n.engine": "llm",                    # llm | google | azure
    "i18n.google_key": "",
    "i18n.azure_key": "",
    "i18n.azure_region": "",
    "audio.voice": "",
    "audio.rate": 175,                       # words / minute (pyttsx3 scale)
    "audio.volume": 0.9,                     # 0.0 – 1.0
    "audio.elevenlabs_key": "",
    "audio.elevenlabs_voice": "21m00Tcm4TlvDq8ikWAM",

    # ── shell ──────────────────────────────────────────────────────────────
    "ui.frameless": False,
    "ui.left_width": 320,
    "ui.right_width": 380,
    "ui.auto_explain_on_select": False,
    "ui.geometry": "",                       # base64 QMainWindow.saveGeometry()
}


def _coerce(value: str, default: Any) -> Any:
    """JSON-decode a persisted value, falling back to the default's type."""
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return default


class AppConfig:
    """Dict-like, dot-namespaced configuration with JSON-file persistence.

    Usage::

        cfg = AppConfig.load()
        cfg["ai.mode"] = "local"
        cfg.save()
    """

    def __init__(self, data: Optional[Dict[str, Any]] = None, paths: Optional[Paths] = None):
        self.paths = paths or Paths()
        self._data: Dict[str, Any] = dict(DEFAULTS)
        if data:
            self._data.update({k: v for k, v in data.items() if k in DEFAULTS or "." in k})
        self._listeners: list = []

    # -- factory -----------------------------------------------------------
    @classmethod
    def load(cls, paths: Optional[Paths] = None) -> "AppConfig":
        paths = paths or Paths()
        paths.ensure()
        data: Dict[str, Any] = {}
        if paths.settings_file.exists():
            try:
                data = json.loads(paths.settings_file.read_text("utf-8"))
            except Exception:
                data = {}
        return cls(data, paths)

    # -- dict protocol -----------------------------------------------------
    def __getitem__(self, key: str) -> Any:
        return self._data.get(key, DEFAULTS.get(key))

    def __setitem__(self, key: str, value: Any) -> None:
        if self._data.get(key) == value:
            return
        self._data[key] = value
        for cb in list(self._listeners):
            try:
                cb(key, value)
            except Exception:
                pass

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, DEFAULTS.get(key, default))

    def update(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            self[k] = v

    def as_dict(self) -> Dict[str, Any]:
        return dict(self._data)

    # -- persistence -------------------------------------------------------
    def save(self) -> None:
        try:
            self.paths.settings_file.write_text(
                json.dumps(self._data, indent=2), encoding="utf-8"
            )
        except Exception:
            pass

    # -- change notification ----------------------------------------------
    def on_change(self, callback) -> None:
        """Register ``callback(key, value)`` fired on every mutation."""
        self._listeners.append(callback)

    # -- secret helpers ----------------------------------------------------
    def api_key(self, provider_id: str) -> str:
        """Resolve a cloud key from config, falling back to the environment."""
        env_map = {
            "openai": ("ai.openai_key", "OPENAI_API_KEY"),
            "anthropic": ("ai.anthropic_key", "ANTHROPIC_API_KEY"),
            "elevenlabs": ("audio.elevenlabs_key", "ELEVENLABS_API_KEY"),
        }
        cfg_key, env_key = env_map.get(provider_id, ("", ""))
        return (self.get(cfg_key, "") or os.environ.get(env_key, "")).strip()
