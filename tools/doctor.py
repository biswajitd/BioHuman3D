"""
BioHuman3D environment doctor.

Run this before opening a support question — it checks every optional dependency,
the asset tree, the GPU, local LLM endpoints and the audio stack, then prints a
pass/fail table with a concrete fix for anything broken.

    python tools\\doctor.py
    python tools\\doctor.py --json     # machine-readable output
"""
from __future__ import annotations

import argparse
import contextlib
import importlib
import io
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Windows consoles still default to cp1252 in many shells; without this the
# diagnostic output raises UnicodeEncodeError instead of reporting the problem.
for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

RESULTS: list = []


def _import_quietly(name: str):
    """Import a module while swallowing any import-time banner output.

    pygame prints a startup banner to stdout on import, which would otherwise
    interleave with this report.
    """
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        return importlib.import_module(name)


def _is_pygame_ce() -> bool:
    """True when the installed ``pygame`` module is actually pygame-ce.

    Upstream pygame publishes no wheels for Python 3.14, so pip attempts a
    source build that fails (Python 3.14 removed ``distutils.msvccompiler``).
    pygame-ce is a drop-in fork: same ``import pygame`` name, but with wheels
    for 3.12-3.14. It sets ``pygame.IS_CE = 1``.
    """
    try:
        pygame = _import_quietly("pygame")
    except Exception:
        return False
    return bool(getattr(pygame, "IS_CE", 0))


def check(name: str, ok: bool, detail: str = "", fix: str = "", *, required: bool = False):
    RESULTS.append({
        "check": name, "ok": bool(ok), "detail": detail, "fix": fix,
        "required": required,
    })
    marker = "PASS" if ok else ("FAIL" if required else "WARN")
    line = f"[{marker}] {name:<34} {detail}"
    print(line)
    if not ok and fix:
        print(f"       -> {fix}")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose the BioHuman3D environment.")
    parser.add_argument("--json", action="store_true", help="emit JSON only")
    args = parser.parse_args()

    if not args.json:
        print("=" * 74)
        print(" BioHuman3D - environment doctor")
        print("=" * 74)

    # -- core dependencies -------------------------------------------------
    print("\n-- core runtime --")
    check("Python >= 3.10", sys.version_info >= (3, 10), sys.version.split()[0],
          "Install Python 3.10 or newer.", required=True)

    for module, requirement, fix in (
        ("PyQt6", "required", "pip install PyQt6"),
        ("vtk", "required", "pip install vtk   (needs 9.3+ for PyQt6 support)"),
        ("requests", "required", "pip install requests"),
        ("numpy", "optional", "pip install numpy"),
    ):
        try:
            mod = importlib.import_module(module)
            version = getattr(mod, "__version__", "installed")
            check(module, True, str(version), required=(requirement == "required"))
        except Exception as exc:
            check(module, False, str(exc), fix, required=(requirement == "required"))

    # VTK/Qt bridge is the single most common embedding failure.
    try:
        from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor  # noqa: F401
        check("VTK-PyQt6 bridge", True, "QVTKRenderWindowInteractor importable")
    except Exception as exc:
        check("VTK-PyQt6 bridge", False, str(exc),
              "Install matching PyQt6 and VTK wheels: pip install --upgrade PyQt6 vtk")

    # -- audio (optional) ---------------------------------------------------
    print("\n-- audio (optional) --")
    for module, fix in (
        ("pyttsx3", "pip install pyttsx3   (offline system voices)"),
        # pygame has no wheels for Python 3.14; pygame-ce is the drop-in fork.
        ("pygame", "pip install pygame-ce   (wheel available for 3.12-3.14)"),
    ):
        try:
            mod = _import_quietly(module)
            if module == "pygame":
                version = getattr(getattr(mod, "version", None), "ver", "") or ""
                if _is_pygame_ce():
                    version = f"{version} (pygame-ce fork)".strip()
            else:
                version = getattr(mod, "__version__", "")
            check(module, True, f"installed {version}".strip())
        except Exception as exc:
            check(module, False, str(exc), fix)

    # -- video export (optional but headline) -------------------------------
    print("\n-- video export (optional) --")
    try:
        from app.video.exporter import video_support_status
        video_ok, video_detail = video_support_status()
        check("narrated MP4 export", video_ok, video_detail,
              "pip install imageio imageio-ffmpeg")
    except Exception as exc:
        check("narrated MP4 export", False, f"{type(exc).__name__}: {exc}",
              "pip install imageio imageio-ffmpeg")

    try:
        from app.video.player import MULTIMEDIA_AVAILABLE
        check("in-app playback (QtMultimedia)", MULTIMEDIA_AVAILABLE,
              "available" if MULTIMEDIA_AVAILABLE else "absent from this PyQt6 build",
              "Upgrade PyQt6, or use 'Open in system player'.")
    except Exception as exc:
        check("in-app playback (QtMultimedia)", False, str(exc))

    # -- narration content --------------------------------------------------
    print("\n-- narration content --")
    try:
        from app.audio.scripts import (BODY_SYSTEMS, BUILTIN_TOURS,
                                       SYSTEM_TOUR_MAP)
        words = sum(t.word_count for t in BUILTIN_TOURS.values())
        check("scripted tours per body system", len(SYSTEM_TOUR_MAP) > 0,
              f"{len(SYSTEM_TOUR_MAP)} systems · {len(BUILTIN_TOURS)} tours · "
              f"{words:,} words")
    except Exception as exc:
        check("narration scripts", False, f"{type(exc).__name__}: {exc}")

    # -- narration voice coverage -------------------------------------------
    try:
        from app.audio.voices import dialects_available, enumerate_pyttsx3_voices
        voices = enumerate_pyttsx3_voices()
        installed = dialects_available(voices)
        check("Indian English (en-IN) narration voice", "en-IN" in installed,
              f"installed dialects: {', '.join(installed) or 'none'}",
              "Settings -> Time & Language -> Language & region -> Add a language "
              "-> 'English (India)' -> tick Speech, then restart. Until then "
              "narration falls back to an installed English voice.")
    except Exception as exc:
        check("voice catalogue", False, f"{type(exc).__name__}: {exc}")

    # -- project assets ----------------------------------------------------
    print("\n-- assets --")
    try:
        from app.config import Paths
        from app.core.model_registry import LayerRegistry

        paths = Paths().ensure()
        registry = LayerRegistry(paths.models)
        available = registry.available_ids()
        missing = registry.missing_ids()

        check("settings dir writable", paths.settings_file.parent.exists(),
              str(paths.settings_file.parent))
        check(f"3D models found ({len(available)})", bool(available),
              f"{len(available)} present, {len(missing)} missing",
              "python tools\\generate_demo_models.py")
        if missing:
            check("layers without a model file", True,
                  ", ".join(missing[:8]) + (" ..." if len(missing) > 8 else ""),
                  "Placeholder geometry will be used for these layers.")
    except Exception as exc:
        check("app package import", False, f"{type(exc).__name__}: {exc}",
              "Run from the project root: python tools\\doctor.py")

    # -- hardware + LLM endpoints -----------------------------------------
    print("\n-- hardware & local AI --")
    try:
        from app.ai.detector import detect_all
        report = detect_all(include_cloud=True, include_hardware=True)

        hw = report.hardware
        check("NVIDIA GPU", bool(hw.gpu_name), hw.gpu_name or "none detected",
              "Local GPU inference is unavailable; CPU inference still works.")
        check("CUDA runtime", hw.cuda_available,
              "available" if hw.cuda_available else "not available",
              "pip install torch --index-url https://download.pytorch.org/whl/cu121")

        live = report.available_backends()
        for backend in report.backends:
            if backend.kind == "cloud":
                check(f"cloud: {backend.label}", backend.available, backend.detail,
                      "Add an API key in Settings (Ctrl+,).")
            else:
                check(f"local: {backend.label}", backend.available, backend.detail,
                      "Start the server, or ignore on-disk inventories below.")

        inventory = report.local_model_count()
        check("local models on disk", inventory > 0, f"{inventory} found",
              "Pull a model: ollama pull llama3.1:8b")
        check("live local endpoint", bool([b for b in live if b.kind == "local"]),
              ", ".join(b.label for b in live if b.kind == "local") or "none running",
              "Start Ollama (ollama serve) or enable LM Studio's local server.")

        best = report.best_local()
        if best is not None:
            print(f"       suggested routing -> {best.provider_id} / {best.id}")
        print(f"       scan completed in {report.elapsed_s:.2f}s")
    except Exception as exc:
        check("AI detection", False, f"{type(exc).__name__}: {exc}")

    # -- summary -----------------------------------------------------------
    required_failures = [r for r in RESULTS if not r["ok"] and r["required"]]
    warnings = [r for r in RESULTS if not r["ok"] and not r["required"]]

    if args.json:
        print(json.dumps({
            "required_failures": required_failures,
            "warnings": warnings,
            "checks": RESULTS,
        }, indent=2))
    else:
        print("\n" + "=" * 74)
        if required_failures:
            print(f" {len(required_failures)} blocking issue(s) - the app will not start:")
            for item in required_failures:
                print(f"   - {item['check']}: {item['fix']}")
        elif warnings:
            print(f" Ready. {len(warnings)} non-blocking warning(s) above - the app "
                  f"will start with reduced functionality.")
        else:
            print(" All checks passed. Launch with:  python main.py")
        print("=" * 74)

    return 1 if required_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
