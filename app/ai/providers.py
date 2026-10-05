"""
LLM provider adapters.

One interface, six transports. All providers stream, so the transcript renders
token-by-token and the first token arrives in ~200 ms instead of waiting for a
900-token answer.

    LLMProvider (ABC)
      ├── OllamaProvider          POST /api/chat              (NDJSON)
      ├── OpenAICompatProvider    POST /v1/chat/completions   (SSE)
      │     ├── LM Studio
      │     ├── llama.cpp server
      │     └── vLLM
      ├── OpenAIProvider          POST /chat/completions      (SSE)
      └── AnthropicProvider       POST /messages              (SSE, versioned)

Adding a provider = subclass + register. UI code never changes.
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Optional

try:
    import requests
except Exception:  # pragma: no cover
    requests = None  # type: ignore[assignment]

from app.ai.detector import BackendStatus, ModelInfo


class ProviderError(RuntimeError):
    """Raised for configuration problems (missing key, unreachable host)."""


@dataclass
class ChatMessage:
    role: str      # "system" | "user" | "assistant"
    content: str

    def as_dict(self) -> Dict[str, str]:
        return {"role": self.role, "content": self.content}


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------
class LLMProvider(ABC):
    """Streaming chat provider."""

    id: str = "base"
    label: str = "Provider"
    kind: str = "local"          # local | cloud
    requires_key: bool = False

    def __init__(self, endpoint: str = "", api_key: str = "", **options) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.api_key = api_key or ""
        self.options = options
        self._session = requests.Session() if requests else None

    # -- capability --------------------------------------------------------
    @abstractmethod
    def list_models(self) -> List[ModelInfo]:
        """Enumerate models this provider can serve right now."""

    def health(self) -> BackendStatus:
        models = self.list_models()
        return BackendStatus(id=self.id, label=self.label, kind=self.kind,
                             available=bool(models), endpoint=self.endpoint,
                             detail=f"{len(models)} model(s)", models=models)

    def is_configured(self) -> bool:
        return bool(self.endpoint) and (not self.requires_key or bool(self.api_key))

    # -- inference ---------------------------------------------------------
    @abstractmethod
    def stream_chat(self, messages: List[ChatMessage], model: str,
                    *, temperature: float = 0.3, max_tokens: int = 900,
                    timeout: float = 180.0) -> Iterator[str]:
        """Yield response text chunks as they arrive."""

    # -- helpers -----------------------------------------------------------
    def complete(self, messages: List[ChatMessage], model: str, **kwargs) -> str:
        """Convenience wrapper for non-streaming call sites."""
        return "".join(self.stream_chat(messages, model, **kwargs))

    def _post_stream(self, url: str, payload: dict, headers: Dict[str, str],
                     timeout: float):
        if self._session is None:
            raise ProviderError("The 'requests' package is required for network providers.")
        headers = {"Content-Type": "application/json", **headers}
        response = self._session.post(url, json=payload, headers=headers,
                                      stream=True, timeout=(5.0, timeout))
        if response.status_code >= 400:
            body = response.text[:400]
            response.close()
            raise ProviderError(f"{self.label} HTTP {response.status_code}: {body}")
        return response

    @staticmethod
    def _iter_sse_data(response) -> Iterator[str]:
        """Yield the ``data:`` payloads of an SSE stream, stopping at ``[DONE]``."""
        for raw in response.iter_lines(decode_unicode=True):
            if not raw:
                continue
            line = raw.decode("utf-8") if isinstance(raw, bytes) else raw
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            yield data


# ---------------------------------------------------------------------------
# Ollama
# ---------------------------------------------------------------------------
class OllamaProvider(LLMProvider):
    id = "ollama"
    label = "Ollama"
    kind = "local"

    def __init__(self, endpoint: str = "http://127.0.0.1:11434", **options) -> None:
        super().__init__(endpoint=endpoint, **options)

    def list_models(self) -> List[ModelInfo]:
        if self._session is None:
            return []
        try:
            resp = self._session.get(f"{self.endpoint}/api/tags", timeout=1.5)
            data = resp.json()
        except Exception:
            return []
        return [
            ModelInfo(id=m.get("name", "unknown"), provider_id=self.id,
                      label=m.get("name", "unknown"),
                      size_gb=round((m.get("size") or 0) / 1e9, 2), kind="local")
            for m in data.get("models", [])
        ]

    def stream_chat(self, messages, model, *, temperature=0.3, max_tokens=900,
                    timeout=180.0) -> Iterator[str]:
        payload = {
            "model": model,
            "messages": [m.as_dict() for m in messages],
            "stream": True,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        response = self._post_stream(f"{self.endpoint}/api/chat", payload, {}, timeout)
        try:
            for raw in response.iter_lines(decode_unicode=True):
                if not raw:
                    continue
                line = raw.decode("utf-8") if isinstance(raw, bytes) else raw
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                if event.get("error"):
                    raise ProviderError(str(event["error"]))
                chunk = (event.get("message") or {}).get("content", "")
                if chunk:
                    yield chunk
                if event.get("done"):
                    break
        finally:
            response.close()


# ---------------------------------------------------------------------------
# OpenAI-compatible (LM Studio / llama.cpp / vLLM)
# ---------------------------------------------------------------------------
class OpenAICompatProvider(LLMProvider):
    """Any server exposing ``/v1/chat/completions`` with SSE streaming."""

    id = "openai_compat"
    label = "OpenAI-compatible server"
    kind = "local"

    def __init__(self, endpoint: str = "http://127.0.0.1:1234", **options) -> None:
        super().__init__(endpoint=endpoint, **options)

    def list_models(self) -> List[ModelInfo]:
        if self._session is None:
            return []
        try:
            data = self._session.get(f"{self.endpoint}/v1/models", timeout=1.5).json()
        except Exception:
            return []
        return [ModelInfo(id=m.get("id", "unknown"), provider_id=self.id,
                          label=m.get("id", "unknown"), kind="local")
                for m in data.get("data", [])]

    def stream_chat(self, messages, model, *, temperature=0.3, max_tokens=900,
                    timeout=180.0) -> Iterator[str]:
        payload = {
            "model": model,
            "messages": [m.as_dict() for m in messages],
            "stream": True,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        response = self._post_stream(f"{self.endpoint}/v1/chat/completions",
                                     payload, headers, timeout)
        try:
            for data in self._iter_sse_data(response):
                try:
                    event = json.loads(data)
                except ValueError:
                    continue
                choices = event.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                piece = delta.get("content")
                if piece:
                    yield piece
        finally:
            response.close()


class LMStudioProvider(OpenAICompatProvider):
    id = "lmstudio"
    label = "LM Studio"

    def __init__(self, endpoint: str = "http://127.0.0.1:1234", **options) -> None:
        super().__init__(endpoint=endpoint, **options)


class LlamaCppProvider(OpenAICompatProvider):
    id = "llamacpp"
    label = "llama.cpp server"

    def __init__(self, endpoint: str = "http://127.0.0.1:8080", **options) -> None:
        super().__init__(endpoint=endpoint, **options)


class VLLMProvider(OpenAICompatProvider):
    id = "vllm"
    label = "vLLM"

    def __init__(self, endpoint: str = "http://127.0.0.1:8000", **options) -> None:
        super().__init__(endpoint=endpoint, **options)


# ---------------------------------------------------------------------------
# Cloud
# ---------------------------------------------------------------------------
class OpenAIProvider(LLMProvider):
    id = "openai"
    label = "OpenAI GPT"
    kind = "cloud"
    requires_key = True

    def __init__(self, endpoint: str = "https://api.openai.com/v1",
                 api_key: str = "", **options) -> None:
        super().__init__(endpoint=endpoint, api_key=api_key, **options)

    def list_models(self) -> List[ModelInfo]:
        if not self.api_key or self._session is None:
            return []
        try:
            resp = self._session.get(f"{self.endpoint}/models",
                                     headers={"Authorization": f"Bearer {self.api_key}"},
                                     timeout=4.0)
            data = resp.json()
        except Exception:
            return []
        keep = ("gpt-4", "gpt-3.5", "o1", "o3")
        return [ModelInfo(id=m["id"], provider_id=self.id, label=m["id"], kind="cloud")
                for m in data.get("data", [])
                if any(k in m.get("id", "") for k in keep)][:20]

    def stream_chat(self, messages, model, *, temperature=0.3, max_tokens=900,
                    timeout=120.0) -> Iterator[str]:
        if not self.api_key:
            raise ProviderError("OpenAI API key is not configured.")
        payload = {
            "model": model,
            "messages": [m.as_dict() for m in messages],
            "stream": True,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        response = self._post_stream(
            f"{self.endpoint}/chat/completions", payload,
            {"Authorization": f"Bearer {self.api_key}"}, timeout)
        try:
            for data in self._iter_sse_data(response):
                try:
                    event = json.loads(data)
                except ValueError:
                    continue
                for choice in event.get("choices", []):
                    piece = (choice.get("delta") or {}).get("content")
                    if piece:
                        yield piece
        finally:
            response.close()


class AnthropicProvider(LLMProvider):
    id = "anthropic"
    label = "Anthropic Claude"
    kind = "cloud"
    requires_key = True
    API_VERSION = "2023-06-01"

    def __init__(self, endpoint: str = "https://api.anthropic.com/v1",
                 api_key: str = "", **options) -> None:
        super().__init__(endpoint=endpoint, api_key=api_key, **options)

    def list_models(self) -> List[ModelInfo]:
        # Anthropic has no public /models listing on all tiers; advertise known ids.
        if not self.api_key:
            return []
        return [ModelInfo(id=m, provider_id=self.id, label=m, kind="cloud")
                for m in ("claude-3-5-sonnet-latest", "claude-3-5-haiku-latest",
                          "claude-3-opus-latest")]

    def stream_chat(self, messages, model, *, temperature=0.3, max_tokens=900,
                    timeout=120.0) -> Iterator[str]:
        if not self.api_key:
            raise ProviderError("Anthropic API key is not configured.")

        # Anthropic takes the system prompt out-of-band.
        system_prompt = "\n\n".join(m.content for m in messages if m.role == "system")
        turns = [m.as_dict() for m in messages if m.role != "system"]

        payload = {
            "model": model,
            "messages": turns,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        }
        if system_prompt:
            payload["system"] = system_prompt

        response = self._post_stream(
            f"{self.endpoint}/messages", payload,
            {"x-api-key": self.api_key,
             "anthropic-version": self.API_VERSION}, timeout)
        try:
            for data in self._iter_sse_data(response):
                try:
                    event = json.loads(data)
                except ValueError:
                    continue
                if event.get("type") == "content_block_delta":
                    piece = (event.get("delta") or {}).get("text")
                    if piece:
                        yield piece
                elif event.get("type") == "error":
                    raise ProviderError(str(event.get("error")))
        finally:
            response.close()


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
PROVIDER_CLASSES = {
    cls.id: cls for cls in (
        OllamaProvider, LMStudioProvider, LlamaCppProvider, VLLMProvider,
        OpenAIProvider, AnthropicProvider,
    )
}


def build_provider(provider_id: str, config) -> Optional[LLMProvider]:
    """Instantiate a provider, injecting keys/endpoints from :class:`AppConfig`."""
    cls = PROVIDER_CLASSES.get(provider_id)
    if cls is None:
        return None

    kwargs: Dict[str, object] = {}
    if cls.requires_key:
        kwargs["api_key"] = config.api_key(provider_id)

    endpoint_override = config.get(f"ai.{provider_id}_endpoint", "")
    if endpoint_override:
        kwargs["endpoint"] = endpoint_override

    return cls(**kwargs)  # type: ignore[arg-type]
