"""
Local AI discovery.

Pure-Python, Qt-free (so it is trivially unit-testable and can run inside any
thread). Three complementary discovery strategies:

1. **Live endpoints** — HTTP probes against the well-known localhost ports used
   by Ollama, LM Studio, llama.cpp and vLLM.
2. **On-disk caches** — Ollama's GGUF blob store, LM Studio's model folder and
   the HuggingFace hub cache. Works even when the server is not running.
3. **Hardware capability** — `nvidia-smi`, NVML or `torch.cuda`, so the UI can
   honestly tell the user whether GPU inference is feasible.

Every probe is time-boxed: a dead port costs ~0.4 s and the whole scan runs in a
worker thread, so the UI never stutters.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

try:
    import requests
except Exception:  # pragma: no cover
    requests = None  # type: ignore[assignment]

# Probing a dead socket should feel instantaneous.
PROBE_TIMEOUT = 0.45
HF_CACHE_LIMIT = 40          # don't enumerate a 400-model cache


# ---------------------------------------------------------------------------
# Report dataclasses
# ---------------------------------------------------------------------------
@dataclass
class ModelInfo:
    """One addressable chat model."""

    id: str
    provider_id: str
    label: str = ""
    size_gb: float = 0.0
    family: str = ""
    kind: str = "local"          # local | cloud
    path: str = ""

    def __post_init__(self) -> None:
        if not self.label:
            self.label = self.id


@dataclass
class BackendStatus:
    """Availability + inventory for one provider."""

    id: str
    label: str
    kind: str = "local"          # local | cloud
    available: bool = False
    endpoint: str = ""
    detail: str = ""
    models: List[ModelInfo] = field(default_factory=list)

    @property
    def model_count(self) -> int:
        return len(self.models)


@dataclass
class HardwareReport:
    """What the machine can actually accelerate."""

    python: str = ""
    gpu_name: str = ""
    vram_gb: float = 0.0
    driver: str = ""
    cuda_available: bool = False
    torch_available: bool = False
    torch_version: str = ""
    cpu_count: int = 0
    notes: List[str] = field(default_factory=list)

    @property
    def gpu_ready(self) -> bool:
        return bool(self.gpu_name) and (self.cuda_available or self.vram_gb > 0)

    def summary(self) -> str:
        if not self.gpu_name:
            return "No NVIDIA GPU detected — local models will run on CPU."
        cuda = "CUDA ready" if self.cuda_available else "CUDA runtime not detected"
        return f"{self.gpu_name} · {self.vram_gb:.0f} GB VRAM · {cuda}"


@dataclass
class DetectionReport:
    """Full snapshot produced by :func:`detect_all`."""

    backends: List[BackendStatus] = field(default_factory=list)
    hardware: HardwareReport = field(default_factory=HardwareReport)
    scanned_at: float = field(default_factory=time.time)
    elapsed_s: float = 0.0

    # -- queries -----------------------------------------------------------
    def get(self, backend_id: str) -> Optional[BackendStatus]:
        return next((b for b in self.backends if b.id == backend_id), None)

    def available_backends(self) -> List[BackendStatus]:
        return [b for b in self.backends if b.available]

    def local_models(self) -> List[ModelInfo]:
        return [m for b in self.backends if b.kind == "local" and b.available
                for m in b.models]

    def cloud_models(self) -> List[ModelInfo]:
        return [m for b in self.backends if b.kind == "cloud" and b.available
                for m in b.models]

    def local_model_count(self) -> int:
        """Models discovered on disk even when no server is live."""
        return sum(len(b.models) for b in self.backends if b.kind == "local")

    def best_local(self) -> Optional[ModelInfo]:
        """Prefer a mid-size instruct/coder model already downloaded locally."""
        candidates = self.local_models()
        if not candidates:
            # Fall back to disk-only inventory (server offline) so the UI can
            # offer a 'start your server' nudge with a concrete model name.
            candidates = [m for b in self.backends if b.kind == "local" for m in b.models]
        if not candidates:
            return None

        def score(m: ModelInfo) -> float:
            name = m.id.lower()
            s = 0.0
            if any(k in name for k in ("llama3", "llama-3", "mistral", "qwen", "gemma")):
                s += 2.0
            if any(k in name for k in ("instruct", "chat", "it")):
                s += 1.0
            if any(k in name for k in ("70b", "72b", "405b")):
                s -= 2.5          # likely too large for a 4060
            if any(k in name for k in ("7b", "8b", "12b", "13b", "14b")):
                s += 1.5
            if "embed" in name or "whisper" in name or "vision" in name:
                s -= 3.0
            return s

        return max(candidates, key=score)


# ---------------------------------------------------------------------------
# HTTP probes
# ---------------------------------------------------------------------------
def _http_get_json(url: str, timeout: float = PROBE_TIMEOUT) -> Optional[dict]:
    if requests is None:
        return None
    try:
        response = requests.get(url, timeout=timeout)
        if response.status_code >= 400:
            return None
        return response.json()
    except Exception:
        return None


def probe_ollama(host: str = "127.0.0.1", port: int = 11434) -> BackendStatus:
    base = f"http://{host}:{port}"
    status = BackendStatus(id="ollama", label="Ollama", endpoint=base)
    payload = _http_get_json(f"{base}/api/tags")
    if payload is None:
        status.detail = f"Not reachable at {base}"
        return status

    for item in payload.get("models", []):
        name = item.get("name") or item.get("model") or "unknown"
        status.models.append(ModelInfo(
            id=name,
            provider_id="ollama",
            label=f"{name}",
            size_gb=round((item.get("size") or 0) / 1e9, 2),
            family=(item.get("details") or {}).get("family", ""),
            kind="local",
        ))
    status.available = True
    status.detail = f"{len(status.models)} model(s) at {base}"
    return status


def probe_openai_compatible(backend_id: str, label: str, base: str) -> BackendStatus:
    """LM Studio, llama.cpp server and vLLM all speak ``GET /v1/models``."""
    status = BackendStatus(id=backend_id, label=label, endpoint=base)
    payload = _http_get_json(f"{base}/v1/models")
    if payload is None:
        status.detail = f"Not reachable at {base}"
        return status

    for item in payload.get("data", []):
        model_id = item.get("id") or "unknown"
        status.models.append(ModelInfo(
            id=model_id,
            provider_id=backend_id,
            label=model_id,
            kind="local",
            path=item.get("path", "") or "",
        ))
    status.available = True
    status.detail = f"{len(status.models)} model(s) at {base}"
    return status


# ---------------------------------------------------------------------------
# On-disk inventories
# ---------------------------------------------------------------------------
def _dir_size_gb(path: Path) -> float:
    total = 0
    try:
        for entry in path.rglob("*"):
            if entry.is_file():
                total += entry.stat().st_size
    except Exception:
        pass
    return round(total / 1e9, 2)


def scan_ollama_store() -> BackendStatus:
    """Ollama keeps GGUF blobs under ``~/.ollama/models`` even when offline."""
    status = BackendStatus(id="ollama_disk", label="Ollama model store", kind="local")
    root = Path(os.environ.get("OLLAMA_MODELS", Path.home() / ".ollama" / "models"))
    manifests = root / "manifests"
    if not manifests.exists():
        status.detail = "No Ollama model store found"
        return status

    seen: Dict[str, float] = {}
    for manifest in manifests.rglob("*"):
        if not manifest.is_file():
            continue
        try:
            data = json.loads(manifest.read_text("utf-8"))
        except Exception:
            continue
        name = data.get("name") or manifest.parent.parent.name.replace("library", "").strip("/")
        repo = manifest.parent.name
        tag = manifest.name
        full = f"{repo}:{tag}"
        size = sum(layer.get("size", 0) for layer in data.get("layers", [])) / 1e9
        seen[full or name] = round(size, 2)

    for name, size in list(seen.items())[:HF_CACHE_LIMIT]:
        status.models.append(ModelInfo(id=name, provider_id="ollama", label=name,
                                       size_gb=size, kind="local",
                                       path=str(root)))
    status.available = bool(status.models)
    status.detail = f"{len(status.models)} GGUF manifest(s) on disk"
    return status


def scan_lmstudio_store() -> BackendStatus:
    status = BackendStatus(id="lmstudio_disk", label="LM Studio models", kind="local")
    root = Path.home() / ".lmstudio" / "models"
    if not root.exists():
        root = Path.home() / ".cache" / "lm-studio" / "models"
    if not root.exists():
        status.detail = "No LM Studio model folder found"
        return status

    for model_dir in sorted(p for p in root.rglob("*") if p.is_dir()):
        ggufs = list(model_dir.glob("*.gguf"))
        if not ggufs:
            continue
        status.models.append(ModelInfo(
            id=model_dir.name,
            provider_id="lmstudio",
            label=model_dir.name,
            size_gb=round(sum(g.stat().st_size for g in ggufs) / 1e9, 2),
            kind="local",
            path=str(model_dir),
        ))
    status.available = bool(status.models)
    status.detail = f"{len(status.models)} local model folder(s)"
    return status


def scan_huggingface_cache() -> BackendStatus:
    """Enumerate the HF hub cache — the source of truth for CUDA/transformers work."""
    status = BackendStatus(id="huggingface", label="HuggingFace cache", kind="local")
    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    hub = Path(os.environ.get("HUGGINGFACE_HUB_CACHE", hf_home / "hub"))
    if not hub.exists():
        status.detail = "No HuggingFace hub cache"
        return status

    repos = sorted(p for p in hub.glob("models--*") if p.is_dir())[:HF_CACHE_LIMIT]
    for repo in repos:
        pretty = repo.name.replace("models--", "").replace("--", "/")
        snapshots = repo / "snapshots"
        weights = []
        if snapshots.exists():
            weights = [f for f in snapshots.rglob("*")
                       if f.suffix in (".safetensors", ".bin", ".gguf", ".pt")]
        size_gb = round(sum(f.stat().st_size for f in weights if f.is_file()) / 1e9, 2)
        if not weights:
            continue
        status.models.append(ModelInfo(
            id=pretty, provider_id="huggingface", label=pretty,
            size_gb=size_gb, kind="local", path=str(repo),
        ))
    status.available = bool(status.models)
    status.detail = f"{len(status.models)} cached repo(s) in {hub}"
    return status


# ---------------------------------------------------------------------------
# Hardware
# ---------------------------------------------------------------------------
def detect_hardware() -> HardwareReport:
    import platform

    report = HardwareReport(
        python=platform.python_version(),
        cpu_count=os.cpu_count() or 0,
    )

    # 1) nvidia-smi gives the cleanest name/VRAM/driver triple.
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.run(
                ["nvidia-smi",
                 "--query-gpu=name,memory.total,driver_version",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=3, check=False,
            ).stdout.strip().splitlines()
            if out:
                name, vram_mb, driver = [p.strip() for p in out[0].split(",")[:3]]
                report.gpu_name = name
                report.vram_gb = round(float(vram_mb) / 1024.0, 1)
                report.driver = driver
                report.cuda_available = True
        except Exception as exc:
            report.notes.append(f"nvidia-smi probe failed: {exc}")

    # 2) torch is authoritative for "can I actually run CUDA kernels".
    try:
        import torch  # type: ignore
        report.torch_available = True
        report.torch_version = getattr(torch, "__version__", "")
        if torch.cuda.is_available():
            report.cuda_available = True
            if not report.gpu_name:
                report.gpu_name = torch.cuda.get_device_name(0)
            if not report.vram_gb:
                try:
                    props = torch.cuda.get_device_properties(0)
                    report.vram_gb = round(props.total_memory / 1e9, 1)
                except Exception:
                    pass
    except Exception:
        report.notes.append("PyTorch not installed — CUDA inference unavailable.")

    if not report.gpu_name and shutil.which("nvidia-smi") is None:
        report.notes.append("No NVIDIA tooling found on PATH.")

    return report


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
LOCAL_PROBES: List[Callable[[], BackendStatus]] = [
    probe_ollama,
    lambda: probe_openai_compatible("lmstudio", "LM Studio", "http://127.0.0.1:1234"),
    lambda: probe_openai_compatible("llamacpp", "llama.cpp server", "http://127.0.0.1:8080"),
    lambda: probe_openai_compatible("vllm", "vLLM", "http://127.0.0.1:8000"),
]

DISK_SCANS: List[Callable[[], BackendStatus]] = [
    scan_ollama_store,
    scan_lmstudio_store,
    scan_huggingface_cache,
]


def detect_local(timeout: float = PROBE_TIMEOUT, parallel: bool = True) -> List[BackendStatus]:
    """Probe every known local inference endpoint + on-disk store."""
    results: List[BackendStatus] = []
    if parallel:
        with ThreadPoolExecutor(max_workers=len(LOCAL_PROBES) + len(DISK_SCANS)) as pool:
            futures = [pool.submit(fn) for fn in (*LOCAL_PROBES, *DISK_SCANS)]
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as exc:
                    results.append(BackendStatus(id="unknown", label="probe",
                                                 detail=str(exc)))
    else:
        for fn in (*LOCAL_PROBES, *DISK_SCANS):
            try:
                results.append(fn())
            except Exception:
                pass

    # Live endpoints first, then on-disk inventories.
    preferred = ("ollama", "lmstudio", "llamacpp", "vllm",
                 "ollama_disk", "lmstudio_disk", "huggingface")
    order = {backend_id: index for index, backend_id in enumerate(preferred)}
    results.sort(key=lambda status: order.get(status.id, 99))
    return results


def detect_cloud(config=None) -> List[BackendStatus]:
    """Cloud providers are 'available' only when a key resolves."""
    def key(provider: str) -> str:
        if config is not None and hasattr(config, "api_key"):
            return config.api_key(provider)
        return os.environ.get({"openai": "OPENAI_API_KEY",
                               "anthropic": "ANTHROPIC_API_KEY"}.get(provider, ""), "")

    openai = BackendStatus(id="openai", label="OpenAI", kind="cloud",
                           endpoint="https://api.openai.com/v1")
    if key("openai"):
        openai.available = True
        openai.detail = "API key detected"
        for model in ("gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "o1-mini"):
            openai.models.append(ModelInfo(id=model, provider_id="openai",
                                           label=model, kind="cloud"))
    else:
        openai.detail = "No API key — set OPENAI_API_KEY or add it in Settings"

    anthropic = BackendStatus(id="anthropic", label="Anthropic Claude", kind="cloud",
                              endpoint="https://api.anthropic.com/v1")
    if key("anthropic"):
        anthropic.available = True
        anthropic.detail = "API key detected"
        for model in ("claude-3-5-sonnet-latest", "claude-3-5-haiku-latest",
                      "claude-3-opus-latest"):
            anthropic.models.append(ModelInfo(id=model, provider_id="anthropic",
                                              label=model, kind="cloud"))
    else:
        anthropic.detail = "No API key — set ANTHROPIC_API_KEY or add it in Settings"

    return [openai, anthropic]


def detect_all(config=None, *, include_cloud: bool = True,
               include_hardware: bool = True) -> DetectionReport:
    """Full scan. Safe to call from a worker thread."""
    started = time.perf_counter()
    backends = detect_local()
    if include_cloud:
        backends += detect_cloud(config)
    report = DetectionReport(
        backends=backends,
        hardware=detect_hardware() if include_hardware else HardwareReport(),
        elapsed_s=round(time.perf_counter() - started, 3),
    )
    return report
