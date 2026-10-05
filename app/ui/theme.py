"""
Theme engine.

Qt stylesheets have no variables, so the palette lives here as Python data and
is substituted into ``style.qss`` at load time (``$TOKEN`` syntax via
``string.Template``). That gives one source of truth for colours — change a hex
value here and every widget, chart and overlay follows.

``apply_theme`` also installs a matching ``QPalette`` so native chrome (menus,
tooltips, dialogs, scrollbars drawn by the platform) doesn't flash white.
"""
from __future__ import annotations

from pathlib import Path
from string import Template
from typing import Dict

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QFont, QFontDatabase, QPalette
from PyQt6.QtWidgets import QApplication

# ---------------------------------------------------------------------------
# Design tokens
# ---------------------------------------------------------------------------
PALETTE: Dict[str, str] = {
    # Surfaces (darkest → lightest)
    "BG_APP":      "#0A0D13",
    "BG_PANEL":    "#111726",
    "BG_PANEL_2":  "#151C2C",
    "BG_ELEV":     "#1A2334",
    "BG_HOVER":    "#212C42",
    "BG_INPUT":    "#0D121D",

    # Lines
    "BORDER":      "#243049",
    "BORDER_SOFT": "#1A2333",
    "BORDER_FOCUS": "#22D3EE",

    # Type
    "TEXT":        "#E8EFFA",
    "TEXT_DIM":    "#9AABC4",
    "TEXT_MUTED":  "#63738D",
    "TEXT_INVERT": "#05080E",

    # Brand / accent
    "ACCENT":      "#22D3EE",
    "ACCENT_HI":   "#67E8F9",
    "ACCENT_2":    "#0EA5E9",
    "ACCENT_DARK": "#0B7C8C",
    "ACCENT_GLOW": "rgba(34, 211, 238, 0.18)",
    "ACCENT_SOFT": "rgba(34, 211, 238, 0.10)",

    # Semantic
    "SUCCESS":     "#34D399",
    "WARNING":     "#FBBF24",
    "DANGER":      "#F87171",
    "MUSCLE":      "#E0525F",
    "BONE":        "#D8D2C0",

    # Anatomy layer accents (mirror model_registry colours)
    "LAYER_SKIN":     "#DCB49E",
    "LAYER_MUSCLE":   "#B8383D",
    "LAYER_BONE":     "#E8E2D0",
    "LAYER_VESSEL":   "#CC2936",
    "LAYER_VEIN":     "#2E52B8",
    "LAYER_NERVE":    "#F2CC33",
    "LAYER_ORGAN":    "#C77E9E",

    # Radii / metrics
    "RADIUS_SM":   "6px",
    "RADIUS_MD":   "10px",
    "RADIUS_LG":   "14px",
    "RADIUS_PILL": "999px",
}

#: Preferred UI font stack, best first.
FONT_CANDIDATES = (
    "Segoe UI Variable Display",
    "Segoe UI Variable Text",
    "Segoe UI",
    "Inter",
    "SF Pro Text",
    "Noto Sans",
    "DejaVu Sans",
)


def resolve_font_family() -> str:
    """Return the first installed font from :data:`FONT_CANDIDATES`."""
    try:
        installed = set(QFontDatabase.families())
        for candidate in FONT_CANDIDATES:
            if candidate in installed:
                return candidate
    except Exception:
        pass
    return "Segoe UI"


# ---------------------------------------------------------------------------
# Stylesheet compilation
# ---------------------------------------------------------------------------
def load_stylesheet(path: Path | None = None, extra: Dict[str, str] | None = None) -> str:
    """Read ``style.qss`` and substitute the palette tokens."""
    path = Path(path) if path else Path(__file__).with_name("style.qss")
    if not path.exists():
        return ""
    template = Template(path.read_text(encoding="utf-8"))
    tokens = dict(PALETTE)
    if extra:
        tokens.update(extra)
    # ``safe_substitute`` keeps any stray ``$`` in future QSS edits from
    # exploding the whole theme at runtime.
    return template.safe_substitute(**tokens)


# ---------------------------------------------------------------------------
# Application-level application
# ---------------------------------------------------------------------------
def apply_theme(app: QApplication, config=None) -> str:
    """Install Fusion style, dark palette, UI font and the compiled stylesheet."""
    app.setStyle("Fusion")

    family = resolve_font_family()
    base = QFont(family, 10)
    base.setHintingPreference(QFont.HintingPreference.PreferFullHinting)
    base.setStyleStrategy(QFont.StyleStrategy.PreferAntialias)
    app.setFont(base)

    # Native chrome must not fall back to the light default palette.
    pal = QPalette()
    pal.setColor(QPalette.ColorRole.Window, QColor(PALETTE["BG_APP"]))
    pal.setColor(QPalette.ColorRole.WindowText, QColor(PALETTE["TEXT"]))
    pal.setColor(QPalette.ColorRole.Base, QColor(PALETTE["BG_INPUT"]))
    pal.setColor(QPalette.ColorRole.AlternateBase, QColor(PALETTE["BG_PANEL"]))
    pal.setColor(QPalette.ColorRole.Text, QColor(PALETTE["TEXT"]))
    pal.setColor(QPalette.ColorRole.Button, QColor(PALETTE["BG_ELEV"]))
    pal.setColor(QPalette.ColorRole.ButtonText, QColor(PALETTE["TEXT"]))
    pal.setColor(QPalette.ColorRole.Highlight, QColor(PALETTE["ACCENT"]))
    pal.setColor(QPalette.ColorRole.HighlightedText, QColor(PALETTE["TEXT_INVERT"]))
    pal.setColor(QPalette.ColorRole.ToolTipBase, QColor(PALETTE["BG_ELEV"]))
    pal.setColor(QPalette.ColorRole.ToolTipText, QColor(PALETTE["TEXT"]))
    pal.setColor(QPalette.ColorRole.PlaceholderText, QColor(PALETTE["TEXT_MUTED"]))
    pal.setColor(QPalette.ColorRole.Link, QColor(PALETTE["ACCENT"]))
    for role in (QPalette.ColorRole.WindowText, QPalette.ColorRole.Text,
                 QPalette.ColorRole.ButtonText):
        pal.setColor(QPalette.ColorGroup.Disabled, role, QColor(PALETTE["TEXT_MUTED"]))
    app.setPalette(pal)

    qss_path = getattr(getattr(config, "paths", None), "qss", None)
    stylesheet = load_stylesheet(qss_path)
    app.setStyleSheet(stylesheet)
    return stylesheet


def repolish(widget) -> None:
    """Force a style recalculation after a dynamic property changes."""
    style = widget.style()
    style.unpolish(widget)
    style.polish(widget)
    widget.update()
