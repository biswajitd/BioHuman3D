"""
Muscle-action animation: a muscle contracts and its joint moves through the
real motion, with agonists and antagonists colour-coded.

One cycle (``CYCLE`` seconds):

    0.0 → 0.8   passive set-up: the limb moves to the motion's start angle
                (e.g. the elbow is flexed before demonstrating *extension*)
    0.8 → 2.6   concentric contraction: the agonist glows, thickens as it
                shortens, and drives the joint from start to end angle; the
                antagonist is shown lengthening (cool tint)
    2.6 → 3.2   hold at end of range
    3.2 → 4.2   relax back to the anatomical position

The cycle repeats for as long as the narration lasts, so a long explanation
shows several repetitions rather than a frozen pose.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from app.anatomy.kinematics import Rig, smoothstep
from app.anatomy.muscles import (MOTIONS, MUSCLES, Motion, Muscle, agonists_for,
                                 antagonists_for, muscle_for_structure)
from app.core.effects import SceneEffect, lerp_colour

try:
    from vtkmodules.util.numpy_support import vtk_to_numpy
except Exception:  # pragma: no cover
    from vtk.util.numpy_support import vtk_to_numpy  # type: ignore

AGONIST = (1.00, 0.38, 0.16)
AGONIST_PEAK = (1.00, 0.62, 0.22)
ANTAGONIST = (0.36, 0.58, 0.98)
SYNERGIST = (0.95, 0.62, 0.30)

CYCLE = 4.2
SETUP, CONTRACT, HOLD, RELAX = 0.8, 2.6, 3.2, 4.2


def _ease(t):
    return 0.5 - 0.5 * math.cos(math.pi * max(0.0, min(1.0, t)))


def phase(seconds: float) -> Tuple[float, float]:
    """``(progress, activation)`` within a cycle.

    *progress* is 0 at rest, 1 at the motion's start angle … 2 at its end angle.
    *activation* (0–1) is how strongly the agonist is contracting.
    """
    t = seconds % CYCLE
    if t < SETUP:
        return _ease(t / SETUP), 0.0
    if t < CONTRACT:
        k = (t - SETUP) / (CONTRACT - SETUP)
        return 1.0 + _ease(k), min(1.0, 0.35 + 0.65 * math.sin(math.pi * min(1.0, k * 1.2)))
    if t < HOLD:
        return 2.0, 0.55
    k = (t - HOLD) / (RELAX - HOLD)
    return 2.0 * (1.0 - _ease(k)), 0.0


def angle_at(motion: Motion, seconds: float) -> Tuple[float, float]:
    p, act = phase(seconds)
    if p <= 1.0:
        return motion.start * p, act
    return motion.start + (motion.end - motion.start) * (p - 1.0), act


class MuscleActionEffect(SceneEffect):
    """Animate *motion* on one side, highlighting agonists and antagonists."""

    cycle_seconds = CYCLE

    def __init__(self, rig: Rig, motion_id: str, side: str = "right",
                 focus_muscle: str = "") -> None:
        self.rig = rig
        self.motion = MOTIONS[motion_id]
        self.side = side
        self.limb_key = f"{side}_{self.motion.limb}"
        self.focus_muscle = focus_muscle.lower()
        self._agonists: List[str] = []
        self._antagonists: List[str] = []
        self._saved: Dict[str, Tuple[object, tuple, float, float]] = {}
        self._rest: Dict[str, np.ndarray] = {}

    # -- structure selection -------------------------------------------------
    def _structures(self, viewport, muscles: Sequence[Muscle]) -> List[str]:
        state = viewport.layers.get("muscles")
        if state is None:
            return []
        keys = {m.key for m in muscles}
        out = []
        for name in state.structures:
            low = name.lower()
            if self.side not in low:
                continue
            match = muscle_for_structure(name)
            if match is not None and match.key in keys:
                out.append(name)
        return out

    def structures(self, viewport) -> Tuple[List[str], List[str]]:
        agon = agonists_for(self.motion.id)
        if self.focus_muscle:
            agon = [m for m in agon if m.key == self.focus_muscle] + \
                   [m for m in agon if m.key != self.focus_muscle]
        return self._structures(viewport, agon), self._structures(viewport, antagonists_for(self.motion.id))

    # -- effect protocol ----------------------------------------------------
    def begin(self, viewport) -> None:
        self._agonists, self._antagonists = self.structures(viewport)
        self._saved.clear()
        for name in self._agonists + self._antagonists:
            actor = viewport.structure_actor(name)
            if actor is None:
                continue
            prop = actor.GetProperty()
            self._saved[name] = (actor, prop.GetColor(), prop.GetAmbient(), prop.GetOpacity())
            prop.SetOpacity(1.0)
            actor.SetVisibility(True)
        self._rest = {}
        for name in self._agonists:
            poly = viewport.structure_polydata(name)
            if poly is not None:
                self._rest[name] = vtk_to_numpy(poly.GetPoints().GetData()).copy()

    def apply(self, viewport, seconds: float) -> None:
        angle, act = angle_at(self.motion, seconds)
        self.rig.pose(self.limb_key, {self.motion.dof: angle})
        primary = self.focus_muscle
        for name in self._agonists:
            actor = self._saved.get(name, (None,))[0]
            if actor is None:
                continue
            is_focus = bool(primary) and primary in name.lower()
            colour = lerp_colour(AGONIST if is_focus or not primary else SYNERGIST, AGONIST_PEAK, act)
            actor.GetProperty().SetColor(*colour)
            actor.GetProperty().SetAmbient(0.14 + 0.45 * act)
            actor.GetProperty().SetOpacity(1.0)
            self._bulge(viewport, name, act * (1.0 if is_focus or not primary else 0.6))
        for name in self._antagonists:
            actor = self._saved.get(name, (None,))[0]
            if actor is not None:
                actor.GetProperty().SetColor(*ANTAGONIST)
                actor.GetProperty().SetAmbient(0.25)
                actor.GetProperty().SetOpacity(1.0)

    def end(self, viewport) -> None:
        self.rig.reset()
        for name, rest in self._rest.items():
            poly = viewport.structure_polydata(name)
            if poly is not None:
                vtk_to_numpy(poly.GetPoints().GetData())[:] = rest
                poly.GetPoints().Modified()
                poly.Modified()
        self._rest.clear()
        for name, (actor, colour, ambient, opacity) in self._saved.items():
            prop = actor.GetProperty()
            prop.SetColor(*colour)
            prop.SetAmbient(ambient)
            prop.SetOpacity(opacity)
        self._saved.clear()
        render = getattr(viewport, "_render", None)
        if render is not None:
            render()

    # -- contraction bulge (volume is conserved as the belly shortens) -------
    def _bulge(self, viewport, name: str, activation: float) -> None:
        if activation <= 0.01:
            if name in self._rest and not self.rig.is_bound(viewport.structure_polydata(name)):
                vtk_to_numpy(viewport.structure_polydata(name).GetPoints().GetData())[:] = self._rest[name]
            return
        poly = viewport.structure_polydata(name)
        if poly is None:
            return
        pts = vtk_to_numpy(poly.GetPoints().GetData())
        if not self.rig.is_bound(poly) and name in self._rest:
            pts[:] = self._rest[name]          # unposed muscle: bulge from rest, never compound
        c = pts.mean(axis=0)
        cov = np.cov((pts - c).T)
        axis = np.linalg.eigh(cov)[1][:, -1]
        along = (pts - c) @ axis
        half = np.abs(along).max() or 1.0
        radial = (pts - c) - np.outer(along, axis)
        profile = np.cos(np.clip(along / half, -1, 1) * math.pi / 2) ** 2
        pts += radial * (0.16 * activation * profile)[:, None]
        poly.GetPoints().Modified()
        poly.Modified()


def motions_for(muscle: Muscle) -> List[Motion]:
    return [MOTIONS[m] for m in muscle.actions if m in MOTIONS]


class ContractInPlaceEffect(SceneEffect):
    """For muscles without a limb-joint action (masseter, diaphragm, trunk):
    the belly glows and thickens rhythmically while the scene stays still."""

    cycle_seconds = 2.4

    def __init__(self, names: Sequence[str]) -> None:
        self.names = list(names)
        self._saved: Dict[str, Tuple[object, tuple, float, float]] = {}
        self._rest: Dict[str, np.ndarray] = {}

    def begin(self, viewport) -> None:
        for name in self.names:
            actor, poly = viewport.structure_actor(name), viewport.structure_polydata(name)
            if actor is None or poly is None:
                continue
            p = actor.GetProperty()
            self._saved[name] = (actor, p.GetColor(), p.GetAmbient(), p.GetOpacity())
            self._rest[name] = vtk_to_numpy(poly.GetPoints().GetData()).copy()

    def apply(self, viewport, seconds: float) -> None:
        act = 0.5 - 0.5 * math.cos(2 * math.pi * seconds / self.cycle_seconds)
        for name, (actor, _c, _a, _o) in self._saved.items():
            p = actor.GetProperty()
            p.SetColor(*lerp_colour(AGONIST, AGONIST_PEAK, act))
            p.SetAmbient(0.14 + 0.45 * act)
            p.SetOpacity(1.0)
            poly = viewport.structure_polydata(name)
            pts = vtk_to_numpy(poly.GetPoints().GetData())
            rest = self._rest[name]
            c = rest.mean(axis=0)
            axis = np.linalg.eigh(np.cov((rest - c).T))[1][:, -1]
            along = (rest - c) @ axis
            radial = (rest - c) - np.outer(along, axis)
            half = np.abs(along).max() or 1.0
            profile = np.cos(np.clip(along / half, -1, 1) * math.pi / 2) ** 2
            # Shortens along its axis and thickens across it (constant volume).
            pts[:] = c + np.outer(along * (1 - 0.10 * act), axis) + radial * (1 + 0.16 * act * profile)[:, None]
            poly.GetPoints().Modified()
            poly.Modified()

    def end(self, viewport) -> None:
        for name, (actor, colour, ambient, opacity) in self._saved.items():
            p = actor.GetProperty()
            p.SetColor(*colour)
            p.SetAmbient(ambient)
            p.SetOpacity(opacity)
            poly = viewport.structure_polydata(name)
            if poly is not None and name in self._rest:
                vtk_to_numpy(poly.GetPoints().GetData())[:] = self._rest[name]
                poly.GetPoints().Modified()
                poly.Modified()
        self._saved.clear()
        self._rest.clear()
