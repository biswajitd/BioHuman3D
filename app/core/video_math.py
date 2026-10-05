"""
Small numeric helpers shared by the tour engine and the video exporter.

Kept separate so the exporter does not import widget code, and so the easing
function stays identical between live playback and rendered output (a mismatch
would make the exported camera moves differ from the preview).
"""
from __future__ import annotations

import textwrap
from typing import Dict, List, Sequence

# Re-exported so both the live tour and the renderer ease identically.
from app.core.tour import ease_in_out_cubic  # noqa: F401


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def lerp_vec(a: Sequence[float], b: Sequence[float], t: float) -> List[float]:
    if not a or not b:
        return list(a or b or [])
    n = min(len(a), len(b))
    return [lerp(a[i], b[i], t) for i in range(n)]


def lerp_state(from_state: Dict, to_state: Dict, t: float) -> Dict:
    """Blend two camera states (position / focal / up / angle)."""
    if not from_state:
        return dict(to_state or {})
    if not to_state:
        return dict(from_state)
    angle_a = float(from_state.get("angle", 30.0))
    angle_b = float(to_state.get("angle", 30.0))
    return {
        "position": lerp_vec(from_state.get("position", ()), to_state.get("position", ()), t),
        "focal": lerp_vec(from_state.get("focal", ()), to_state.get("focal", ()), t),
        "up": lerp_vec(from_state.get("up", ()), to_state.get("up", ()), t),
        "angle": lerp(a=angle_a, b=angle_b, t=t),
    }


def wrap_caption(text: str, width: int = 78, max_lines: int = 3) -> str:
    """Wrap narration into a caption block for burning into video frames.

    Long narration is truncated with an ellipsis — the full text is still spoken
    in the audio track, the caption is a readability aid.
    """
    cleaned = " ".join((text or "").split())
    if not cleaned:
        return ""
    lines = textwrap.wrap(cleaned, width=width)
    prefix = "\n  "
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1].rstrip(".") + "…"
    return prefix.join(lines) + "\n "
