"""
Time-driven scene effects for tours and video export.

A *scene effect* is anything that changes the scene over time within one tour
keyframe: a muscle contracting and moving its joint, a disease stage
progressing, a structure being isolated. The same object drives both the live
tour (``TourEngine`` calls ``apply`` every tick) and the exported video
(``TourVideoExporter`` calls ``apply`` for every frame with the frame's exact
time), so what you see is exactly what gets rendered.

    effect.begin(viewport)          # once, when the keyframe starts
    effect.apply(viewport, seconds) # every tick / frame, seconds since begin
    effect.end(viewport)            # once, when the keyframe ends

Effects must be deterministic in ``seconds`` — export renders frames faster or
slower than real time.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple


class SceneEffect:
    """Base class: does nothing."""

    #: Natural length of one cycle in seconds (0 = static).
    cycle_seconds: float = 0.0

    def begin(self, viewport) -> None:  # pragma: no cover - trivial
        pass

    def apply(self, viewport, seconds: float) -> None:  # pragma: no cover - trivial
        pass

    def end(self, viewport) -> None:  # pragma: no cover - trivial
        pass


class CompositeEffect(SceneEffect):
    def __init__(self, effects: Iterable[SceneEffect]) -> None:
        self.effects: List[SceneEffect] = [e for e in effects if e is not None]
        self.cycle_seconds = max([e.cycle_seconds for e in self.effects] or [0.0])

    def begin(self, viewport) -> None:
        for e in self.effects:
            e.begin(viewport)

    def apply(self, viewport, seconds: float) -> None:
        for e in self.effects:
            e.apply(viewport, seconds)

    def end(self, viewport) -> None:
        for e in reversed(self.effects):
            e.end(viewport)


class IsolateStructuresEffect(SceneEffect):
    """Show named structures solid, ghost the rest of their layers.

    ``context`` layers keep their own (keyframe) opacity, so a muscle can be
    isolated while the skeleton stays visible for orientation.
    """

    def __init__(self, names: Sequence[str], ghost: float = 0.08,
                 context_layers: Sequence[str] = ("skeleton",), solid: float = 1.0) -> None:
        self.names = {n.lower() for n in names}
        self.ghost = ghost
        self.solid = solid
        self.context = set(context_layers)
        self._saved: Dict[int, Tuple[object, float, bool]] = {}

    def begin(self, viewport) -> None:
        self._saved.clear()
        for state in viewport.layers.values():
            for actor in state.actors:
                self._saved[id(actor)] = (actor, actor.GetProperty().GetOpacity(), actor.GetVisibility())

    def apply(self, viewport, seconds: float) -> None:
        for state in viewport.layers.values():
            if state.id in self.context:
                continue
            for actor, name in zip(state.actors, state.structures):
                hit = bool(name) and name.lower() in self.names
                actor.SetVisibility(True if hit else actor.GetVisibility())
                actor.GetProperty().SetOpacity(self.solid if hit else min(self.ghost,
                                                                         actor.GetProperty().GetOpacity()))

    def end(self, viewport) -> None:
        for actor, opacity, visible in self._saved.values():
            actor.GetProperty().SetOpacity(opacity)
            actor.SetVisibility(visible)
        self._saved.clear()


class TintEffect(SceneEffect):
    """Temporarily recolour named structures, restoring the originals on end."""

    def __init__(self, colours: Dict[str, Tuple[float, float, float]]) -> None:
        self.colours = {k.lower(): v for k, v in colours.items()}
        self._saved: Dict[int, Tuple[object, Tuple[float, float, float], float]] = {}

    def begin(self, viewport) -> None:
        self._saved.clear()
        for name, colour in self.colours.items():
            actor = viewport.structure_actor(name)
            if actor is None:
                continue
            prop = actor.GetProperty()
            self._saved[id(actor)] = (actor, prop.GetColor(), prop.GetAmbient())
            prop.SetColor(*colour)

    def end(self, viewport) -> None:
        for actor, colour, ambient in self._saved.values():
            actor.GetProperty().SetColor(*colour)
            actor.GetProperty().SetAmbient(ambient)
        self._saved.clear()


def lerp_colour(a, b, t: float) -> Tuple[float, float, float]:
    return tuple(a[i] + (b[i] - a[i]) * t for i in range(3))
