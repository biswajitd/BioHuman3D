"""
BioHuman3D — entry point.

Bootstraps the Qt application, applies the dark theme, ensures the asset tree
exists, and launches the main window. Every heavy subsystem (VTK, AI detection,
TTS) is imported lazily so a missing optional dependency degrades to a warning
banner instead of a crash.

Run:
    python main.py
"""
from __future__ import annotations

import os
import sys
import ctypes
from pathlib import Path

# --------------------------------------------------------------------------
# 0.  Environment hardening (must run before QApplication is constructed)
# --------------------------------------------------------------------------
# Qt6 scales automatically; these flags keep fractional-DPI crisp on Windows.
os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "1")
os.environ.setdefault("QT_SCALE_FACTOR_ROUNDING_POLICY", "PassThrough")
# OpenGL: prefer the discrete GPU on hybrid laptops.
os.environ.setdefault("QT_OPENGL", "desktop")

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _set_windows_app_id() -> None:
    """Give Windows a stable AppUserModelID so the taskbar icon is correct."""
    if sys.platform != "win32":
        return
    try:
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
            "BioHuman3D.AnatomyStudio.1"
        )
    except Exception:
        pass


def main() -> int:
    _set_windows_app_id()

    from PyQt6.QtCore import Qt
    from PyQt6.QtGui import QIcon
    from PyQt6.QtWidgets import QApplication

    # Imported after QApplication exists to avoid Qt/OpenGL init-order warnings.
    from app.config import AppConfig
    from app.ui.theme import apply_theme
    from app.ui.main_window import MainWindow

    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    app = QApplication(sys.argv)
    app.setApplicationName("BioHuman3D")
    app.setOrganizationName("BioHuman3D")
    app.setApplicationDisplayName("BioHuman3D — Anatomy & Health Studio")

    config = AppConfig.load()
    config.paths.ensure()

    apply_theme(app, config)

    # Optional window icon (silently skipped when the asset is absent).
    icon_path = config.paths.icons / "app.svg"
    if icon_path.exists():
        app.setWindowIcon(QIcon(str(icon_path)))

    window = MainWindow(config)
    window.show()

    # First-run convenience: generate placeholder anatomy if the user has no models.
    if not any(config.paths.models.glob("*.*")):
        window.status_message(
            "No 3D models found — run  python tools\\generate_demo_models.py  "
            "to create placeholder anatomy.",
            timeout_ms=12000,
        )

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
