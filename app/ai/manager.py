"""
AIManager — the single façade the UI talks to.

Responsibilities
----------------
* Run detection on a worker thread and publish a :class:`DetectionReport`.
* Resolve "which provider + model should answer right now?" from user intent
  (Local / Cloud / Auto) plus live availability.
* Stream chat responses on a worker thread, emitting ``token`` signals so the
  transcript types itself out.
* Keep conversation history and the scene context that grounds each answer.

The UI never touches providers or threads directly.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict
from typing import Dict, Iterable, List, Optional, Sequence

from PyQt6.QtCore import QObject, QThread, pyqtSignal

from app.ai import prompts
from app.ai.detector import (DetectionReport, ModelInfo, detect_all)
from app.ai.providers import (ChatMessage, LLMProvider, ProviderError,
                              build_provider)


# ---------------------------------------------------------------------------
# Workers
# ---------------------------------------------------------------------------
class DetectionWorker(QThread):
    """Scans localhost + disk for local models and inspects the GPU."""

    reportReady = pyqtSignal(object)     # DetectionReport
    failed = pyqtSignal(str)

    def __init__(self, config, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._config = config

    def run(self) -> None:  # noqa: D102
        try:
            report = detect_all(self._config, include_cloud=True, include_hardware=True)
            self.reportReady.emit(report)
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class ChatWorker(QThread):
    """Streams one assistant turn from a provider."""

    token = pyqtSignal(str)              # incremental text
    completed = pyqtSignal(str, float)   # full text, elapsed seconds
    failed = pyqtSignal(str)

    def __init__(self, provider: LLMProvider, model: str,
                 messages: Sequence[ChatMessage], options: Dict,
                 parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._provider = provider
        self._model = model
        self._messages = list(messages)
        self._options = options
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def run(self) -> None:  # noqa: D102
        started = time.perf_counter()
        collected: List[str] = []
        try:
            for chunk in self._provider.stream_chat(self._messages, self._model,
                                                    **self._options):
                if self._cancelled:
                    break
                collected.append(chunk)
                self.token.emit(chunk)
            self.completed.emit("".join(collected), time.perf_counter() - started)
        except ProviderError as exc:
            self.failed.emit(str(exc))
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------
class AIManager(QObject):
    """Provider routing + streaming chat with scene grounding."""

    detectionStarted = pyqtSignal()
    detectionReady = pyqtSignal(object)       # DetectionReport
    detectionFailed = pyqtSignal(str)

    tokenReceived = pyqtSignal(str)
    responseStarted = pyqtSignal(str, str)    # provider label, model id
    responseFinished = pyqtSignal(str, float)  # full text, seconds
    responseFailed = pyqtSignal(str)

    providerChanged = pyqtSignal(str, str)    # provider id, model id

    def __init__(self, config, registry=None, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._config = config
        self._registry = registry
        self._report: Optional[DetectionReport] = None
        self._detection: Optional[DetectionWorker] = None
        self._chat: Optional[ChatWorker] = None

        # Conversation state
        self._history: List[ChatMessage] = []
        self._max_history_turns = 12

        # Scene grounding
        self._selected_id = ""
        self._selected_label = ""
        self._visible_layers: List[str] = []
        self._view = "isometric"

        # Resolved target
        self._provider: Optional[LLMProvider] = None
        self._model: str = ""
        self._busy = False
        self._pending_prompt = ""

    # ------------------------------------------------------------- detection
    def refresh_detection(self) -> None:
        """Kick off a background scan (safe to call repeatedly)."""
        if self._detection is not None and self._detection.isRunning():
            return
        self.detectionStarted.emit()
        self._detection = DetectionWorker(self._config, self)
        self._detection.reportReady.connect(self._on_report)
        self._detection.failed.connect(self.detectionFailed)
        self._detection.start()

    def _on_report(self, report: DetectionReport) -> None:
        self._report = report
        self.detectionReady.emit(report)
        self._resolve_target(announce=True)

    @property
    def report(self) -> Optional[DetectionReport]:
        return self._report

    @property
    def is_busy(self) -> bool:
        return self._busy

    @property
    def current_model(self) -> str:
        return self._model

    @property
    def current_provider(self) -> Optional[LLMProvider]:
        return self._provider

    # -------------------------------------------------------------- routing
    def available_models(self) -> List[ModelInfo]:
        """Everything selectable, ordered: healthy local, disk-only local, cloud."""
        if self._report is None:
            return []
        out: List[ModelInfo] = []
        for backend in self._report.backends:
            if backend.id.endswith("_disk") or backend.id == "huggingface":
                continue                     # inventory only, not directly servable
            for model in backend.models:
                out.append(model)
        return out

    def _resolve_target(self, *, announce: bool = False) -> bool:
        """Pick provider+model according to the Local / Cloud / Auto preference."""
        if self._report is None:
            return False

        mode = str(self._config.get("ai.mode", "auto")).lower()
        want_kind = {"local": "local", "cloud": "cloud"}.get(mode)   # None for auto

        # 1) Explicit user choice wins if still healthy.
        forced_provider = str(self._config.get("ai.provider", "") or "")
        forced_model = str(self._config.get("ai.model", "") or "")
        if forced_provider:
            backend = self._report.get(forced_provider)
            if backend and backend.available:
                models = [m.id for m in backend.models]
                if forced_model in models or not forced_model:
                    self._set_target(forced_provider, forced_model or (models[0] if models else ""))
                    if announce:
                        self.providerChanged.emit(forced_provider, self._model)
                    return True

        # 2) Preference-based selection.
        candidates = [b for b in self._report.backends if b.available and b.models]
        if want_kind:
            candidates = [b for b in candidates if b.kind == want_kind]
        if not candidates and want_kind is None:
            candidates = [b for b in self._report.backends if b.available and b.models]

        if not candidates:
            self._provider = None
            self._model = ""
            return False

        local_first = sorted(candidates, key=lambda b: 0 if b.kind == "local" else 1)
        backend = local_first[0]

        if backend.kind == "local":
            best = self._report.best_local()
            model = best.id if best and any(m.id == best.id for m in backend.models) \
                else backend.models[0].id
        else:
            preferred = str(self._config.get(
                "ai.anthropic_model" if backend.id == "anthropic" else "ai.openai_model", ""))
            ids = [m.id for m in backend.models]
            model = preferred if preferred in ids else ids[0]

        self._set_target(backend.id, model)
        if announce:
            self.providerChanged.emit(backend.id, self._model)
        return True

    def _set_target(self, provider_id: str, model_id: str) -> None:
        provider = build_provider(provider_id, self._config)
        self._provider = provider
        self._model = model_id

    def set_target(self, provider_id: str, model_id: str) -> bool:
        """Explicit user selection from the UI dropdown."""
        provider = build_provider(provider_id, self._config)
        if provider is None:
            return False
        self._provider = provider
        self._model = model_id
        self._config["ai.provider"] = provider_id
        self._config["ai.model"] = model_id
        self.providerChanged.emit(provider_id, model_id)
        return True

    def set_mode(self, mode: str) -> None:
        """``auto`` | ``local`` | ``cloud`` — cleared explicit pick so routing re-resolves."""
        self._config["ai.mode"] = mode
        self._config["ai.provider"] = ""
        self._config["ai.model"] = ""
        self._resolve_target(announce=True)

    # --------------------------------------------------------------- context
    def set_scene_context(self, *, selected_id: str = "", selected_label: str = "",
                          visible_layers: Sequence[str] = (), view: str = "") -> None:
        self._selected_id = selected_id
        self._selected_label = selected_label
        self._visible_layers = list(visible_layers)
        self._view = view

    def _context_block(self) -> str:
        notes = ""
        if self._registry is not None and self._selected_id:
            spec = self._registry.spec(self._selected_id)
            if spec is not None:
                notes = spec.description
        return prompts.build_context_block(
            selected_id=self._selected_id,
            selected_label=self._selected_label,
            visible_layers=self._visible_layers,
            registry=self._registry,
            view=self._view,
            anatomy_notes=notes,
        )

    def clear_history(self) -> None:
        self._history.clear()

    def history(self) -> List[Dict[str, str]]:
        return [m.as_dict() for m in self._history]

    # ------------------------------------------------------------- inference
    def ask(self, user_prompt: str, *, use_history: bool = True,
            context: bool = True) -> bool:
        """Stream an answer. Returns False when no provider can serve the request."""
        if self._busy:
            return False
        if self._provider is None or not self._model:
            if not self._resolve_target():
                self.responseFailed.emit(
                    "No AI model available.\n\n"
                    "• Start a local server (Ollama: `ollama serve`, LM Studio: "
                    "enable the local server)\n"
                    "• or add an OpenAI / Anthropic API key in Settings, then "
                    "press Refresh."
                )
                return False

        messages = [ChatMessage("system", prompts.SYSTEM_PROMPT)]
        if context:
            messages.append(ChatMessage("system", self._context_block()))
        if use_history:
            messages.extend(self._history[-self._max_history_turns * 2:])
        messages.append(ChatMessage("user", user_prompt))

        options = {
            "temperature": float(self._config.get("ai.temperature", 0.3)),
            "max_tokens": int(self._config.get("ai.max_tokens", 900)),
        }

        self._busy = True
        self._pending_prompt = user_prompt
        self.responseStarted.emit(self._provider.label, self._model)
        self._chat = ChatWorker(self._provider, self._model, messages, options, self)
        self._chat.token.connect(self.tokenReceived)     # signal -> signal: queued
        # Connect to a bound method, never a lambda: a lambda is not a QObject, so
        # PyQt would use a direct connection and run this on the worker thread.
        self._chat.completed.connect(self._on_chat_completed)
        self._chat.failed.connect(self._on_failed)
        self._chat.start()
        return True

    def _on_chat_completed(self, text: str, seconds: float) -> None:
        self._on_completed(text, seconds, self._pending_prompt)

    def _on_completed(self, text: str, seconds: float, user_prompt: str) -> None:
        self._busy = False
        if text.strip():
            self._history.append(ChatMessage("user", user_prompt))
            self._history.append(ChatMessage("assistant", text))
        self.responseFinished.emit(text, seconds)
        self._persist_turn(user_prompt, text)

    def _on_failed(self, message: str) -> None:
        self._busy = False
        self.responseFailed.emit(message)

    def cancel(self) -> None:
        if self._chat is not None and self._chat.isRunning():
            self._chat.cancel()
        self._busy = False

    # ------------------------------------------------ quick anatomy actions
    def explain_selection(self, depth: str = "standard") -> bool:
        return self.ask(prompts.explain_selection_prompt(self._selected_label, depth=depth))

    def ask_conditions(self) -> bool:
        return self.ask(prompts.conditions_prompt(self._selected_label or "the visible anatomy"))

    def ask_relations(self) -> bool:
        return self.ask(prompts.relations_prompt(self._selected_label or "the visible anatomy"))

    def ask_physiology(self) -> bool:
        return self.ask(prompts.physiology_prompt(self._selected_label or "the visible anatomy"))

    def ask_quiz(self) -> bool:
        return self.ask(prompts.quiz_prompt(self._selected_label))

    def request_voiceover(self) -> bool:
        return self.ask(prompts.voiceover_prompt(self._selected_label, self._visible_layers))

    # ------------------------------------------------------------ diagnostics
    def status_line(self) -> str:
        if self._report is None:
            return "Scanning local models…"
        hardware = self._report.hardware
        if self._provider is not None and self._model:
            return f"{self._provider.label} · {self._model}"
        return f"No model selected · {hardware.summary()}"

    # ------------------------------------------------------------ persistence
    def _persist_turn(self, user_prompt: str, answer: str) -> None:
        try:
            path = self._config.paths.chat_log
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({
                    "ts": time.time(),
                    "selected": self._selected_label,
                    "provider": self._provider.id if self._provider else "",
                    "model": self._model,
                    "prompt": user_prompt,
                    "answer": answer,
                }, ensure_ascii=False) + "\n")
        except Exception:
            pass
