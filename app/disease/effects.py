"""
Applying disease stages to the scene.

Every change is computed from the *healthy baseline* of each affected structure
(points, normals, colour, opacity captured the first time it is touched), so
stages never compound and "back to healthy" is exact.

``DiseaseScene`` holds that baseline and the focal-lesion actors for one
viewport. ``StageTransitionEffect`` is the time-driven scene effect used by
tours and video export: it morphs from one stage to the next over the beat.
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from app.core.effects import SceneEffect
from app.core.vtk_utils import vtk
from app.disease.catalog import Condition, Effect

try:
    from vtkmodules.util.numpy_support import vtk_to_numpy
except Exception:  # pragma: no cover
    from vtk.util.numpy_support import vtk_to_numpy  # type: ignore


def _key(effect: Effect) -> str:
    return f"{effect['kind']}:{effect.get('id', '')}:{'|'.join(effect['match'])}"


def _identity(effect: Effect) -> Effect:
    """The "no change" version of an effect (start of its first appearance)."""
    e = dict(effect)
    kind = e["kind"]
    if kind in ("tint", "pinch"):
        e["amount"] = 0.0
    elif kind == "scale":
        f = e["factor"]
        e["factor"] = (1.0, 1.0, 1.0) if isinstance(f, (tuple, list)) else 1.0
    elif kind == "nodular":
        e["amplitude"] = 0.0
    elif kind == "lesion":
        e["radii"] = (0.0, 0.0, 0.0)
    elif kind == "opacity":
        e["value"] = None
    return e


def _lerp(a, b, t: float):
    if a is None or b is None:
        return b if t >= 0.5 else a
    if isinstance(a, (tuple, list)):
        if not isinstance(b, (tuple, list)):
            b = (b,) * len(a)
        return tuple(_lerp(x, y, t) for x, y in zip(a, b))
    if isinstance(b, (tuple, list)):
        return _lerp((a,) * len(b), b, t)
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return a + (b - a) * t
    return b if t >= 0.5 else a


def blend_stages(condition: Condition, a: int, b: int, t: float) -> List[Effect]:
    """Effects for a point *t* (0–1) of the way from stage *a* to stage *b*."""
    ea = {_key(e): e for e in condition.stages[a].effects} if a >= 0 else {}
    eb = {_key(e): e for e in condition.stages[b].effects}
    out = []
    for key in list(dict.fromkeys(list(ea) + list(eb))):
        start = ea.get(key) or _identity(eb[key])
        end = eb.get(key) or _identity(ea[key])
        mixed = dict(end)
        for field, value in end.items():
            if field in ("kind", "match", "id"):
                continue
            mixed[field] = _lerp(start.get(field), value, t)
        out.append(mixed)
    return out


def _noise(points: np.ndarray, frequency: float, seed: int = 3) -> np.ndarray:
    """Smooth deterministic 3D noise in [-1, 1] (sum of random plane waves)."""
    rng = np.random.default_rng(seed)
    dirs = rng.normal(size=(9, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    phases = rng.uniform(0, 2 * math.pi, 9)
    scales = rng.uniform(0.7, 1.4, 9)
    acc = np.zeros(len(points))
    for d, ph, sc in zip(dirs, phases, scales):
        acc += np.cos(points @ d * frequency * sc + ph)
    return acc / 3.0


class DiseaseScene:
    """Baseline store + effect application for one viewport."""

    def __init__(self, viewport) -> None:
        self.viewport = viewport
        self._base: Dict[str, dict] = {}          # structure name → baseline
        self._lesions: Dict[str, object] = {}     # lesion key → actor
        self._touched: set = set()

    # -------------------------------------------------------------- matching
    def matching(self, keywords: Sequence[str]) -> List[str]:
        names = []
        for state in self.viewport.layers.values():
            for name in state.structures:
                low = name.lower()
                if name and any(k.lower() in low for k in keywords):
                    names.append(name)
        # Prefer the most specific keyword set: "left cerebral hemisphere"
        # must not also pick up "right cerebral hemisphere".
        return names

    def _baseline(self, name: str) -> Optional[dict]:
        if name in self._base:
            return self._base[name]
        actor, poly = self.viewport.structure_actor(name), self.viewport.structure_polydata(name)
        if actor is None or poly is None:
            return None
        normals = poly.GetPointData().GetNormals()
        prop = actor.GetProperty()
        self._base[name] = dict(
            actor=actor, poly=poly,
            points=vtk_to_numpy(poly.GetPoints().GetData()).astype(float).copy(),
            normals=vtk_to_numpy(normals).astype(float).copy() if normals is not None else None,
            color=prop.GetColor(), opacity=prop.GetOpacity())
        return self._base[name]

    # -------------------------------------------------------------- apply
    def apply(self, effects: Iterable[Effect]) -> None:
        """Set the scene to exactly these effects (from the healthy baseline)."""
        effects = list(effects)
        signature = repr(effects)
        if signature == getattr(self, "_last", None):
            return                      # unchanged (e.g. holding a stage): nothing to do
        self._last = signature
        geometry: Dict[str, List[Effect]] = {}
        colours: Dict[str, Tuple[tuple, float]] = {}
        opacities: Dict[str, float] = {}
        lesions_seen = set()
        for effect in effects:
            names = self.matching(effect["match"])
            kind = effect["kind"]
            if kind == "lesion":
                key = _key(effect)
                lesions_seen.add(key)
                self._lesion(key, names, effect)
                continue
            for name in names:
                if self._baseline(name) is None:
                    continue
                self._touched.add(name)
                if kind == "tint":
                    colours[name] = (tuple(effect["color"]), float(effect["amount"]))
                elif kind == "opacity" and effect.get("value") is not None:
                    opacities[name] = float(effect["value"])
                else:
                    geometry.setdefault(name, []).append(effect)

        for name in list(self._touched):
            base = self._base[name]
            prop = base["actor"].GetProperty()
            colour, amount = colours.get(name, (base["color"], 0.0))
            prop.SetColor(*[base["color"][i] + (colour[i] - base["color"][i]) * amount for i in range(3)])
            if name in opacities:
                prop.SetOpacity(opacities[name])
            self._deform(base, geometry.get(name, []))

        for key, actor in self._lesions.items():
            if key not in lesions_seen:
                actor.SetVisibility(False)
        self._render()

    def _deform(self, base: dict, effects: List[Effect]) -> None:
        pts = base["points"].copy()
        changed_shape = False
        for effect in effects:
            kind = effect["kind"]
            if kind == "scale":
                f = effect["factor"]
                f = np.asarray(f if isinstance(f, (tuple, list)) else (f, f, f), float)
                c = base["points"].mean(axis=0)
                pts = c + (pts - c) * f
            elif kind == "nodular" and base["normals"] is not None and effect["amplitude"]:
                n = _noise(base["points"], float(effect["frequency"]))
                pts = pts + base["normals"] * (float(effect["amplitude"]) * n)[:, None]
                changed_shape = True
            elif kind == "pinch" and effect["amount"]:
                centre = self._anchor(base["points"], effect["at"])
                d = np.linalg.norm(base["points"] - centre, axis=1)
                r = float(effect["radius"])
                w = np.clip(1.0 - d / r, 0.0, 1.0) ** 2 * float(effect["amount"])
                near = d < r
                if near.any():
                    axis_pt = base["points"][near].mean(axis=0)
                    pts = pts + (axis_pt - pts) * w[:, None] * 0.85
                    changed_shape = True
        poly = base["poly"]
        vtk_to_numpy(poly.GetPoints().GetData())[:] = pts
        poly.GetPoints().Modified()
        if base["normals"] is not None and poly.GetPointData().GetNormals() is not None:
            if changed_shape:
                nrm = vtk.vtkPolyDataNormals()
                nrm.SetInputData(poly)
                nrm.SplittingOff()
                nrm.ConsistencyOn()
                nrm.Update()
                new = nrm.GetOutput().GetPointData().GetNormals()
                if new is not None and new.GetNumberOfTuples() == len(pts):
                    vtk_to_numpy(poly.GetPointData().GetNormals())[:] = vtk_to_numpy(new)
            else:
                vtk_to_numpy(poly.GetPointData().GetNormals())[:] = base["normals"]
            poly.GetPointData().GetNormals().Modified()
        poly.Modified()

    @staticmethod
    def _anchor(points: np.ndarray, at: Sequence[float]) -> np.ndarray:
        """Bounding-box fraction → nearest point on the surface."""
        lo, hi = points.min(axis=0), points.max(axis=0)
        target = lo + (hi - lo) * np.asarray(at, float)
        return points[int(np.argmin(np.linalg.norm(points - target, axis=1)))]

    def _lesion(self, key: str, names: List[str], effect: Effect) -> None:
        radii = np.asarray(effect["radii"], float)
        actor = self._lesions.get(key)
        if not names or radii.max() < 1e-5:
            if actor is not None:
                actor.SetVisibility(False)
            return
        pts = np.vstack([self._baseline(n)["points"] for n in names if self._baseline(n) is not None])
        centre = self._anchor(pts, effect["at"])
        inset = float(effect.get("inset", 0.0) or 0.0)
        if inset:          # lesions within an organ sit inside it, not on top of it
            inward = pts.mean(axis=0) - centre
            inward /= (np.linalg.norm(inward) or 1.0)
            centre = centre + inward * inset * float(radii.max())
        if actor is None:
            sphere = vtk.vtkSphereSource()
            sphere.SetThetaResolution(28)
            sphere.SetPhiResolution(20)
            sphere.SetRadius(1.0)
            sphere.Update()
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputData(sphere.GetOutput())
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.SetPickable(False)
            prop = actor.GetProperty()
            prop.SetInterpolationToPhong()
            prop.SetSpecular(0.25)
            prop.SetSpecularPower(20)
            self._lesions[key] = actor
            self.viewport.add_overlay(actor)
        actor.SetScale(*radii)
        actor.SetPosition(*centre)
        actor.GetProperty().SetColor(*effect["color"])
        actor.GetProperty().SetOpacity(float(effect.get("opacity", 1.0)))
        actor.SetVisibility(True)

    # -------------------------------------------------------------- reset
    def reset(self) -> None:
        """Back to the healthy baseline; lesion actors removed."""
        for name in list(self._touched):
            base = self._base[name]
            prop = base["actor"].GetProperty()
            prop.SetColor(*base["color"])
            prop.SetOpacity(base["opacity"])
            self._deform(base, [])
        self._touched.clear()
        for actor in self._lesions.values():
            self.viewport.remove_overlay(actor)
        self._lesions.clear()
        self._last = None
        self._render()

    def _render(self) -> None:
        render = getattr(self.viewport, "_render", None)
        if render is not None:
            render()


class StageTransitionEffect(SceneEffect):
    """Morph from stage *a* to stage *b* over the first part of the beat."""

    cycle_seconds = 3.5
    MORPH = 2.8

    def __init__(self, scene: DiseaseScene, condition: Condition, a: int, b: int,
                 reset_on_end: bool = False) -> None:
        self.scene = scene
        self.condition = condition
        self.a, self.b = a, b
        self.reset_on_end = reset_on_end

    def begin(self, viewport) -> None:
        self.apply(viewport, 0.0)

    def apply(self, viewport, seconds: float) -> None:
        t = min(1.0, max(0.0, seconds / self.MORPH))
        t = 0.5 - 0.5 * math.cos(math.pi * t)
        self.scene.apply(blend_stages(self.condition, self.a, self.b, t))

    def end(self, viewport) -> None:
        if self.reset_on_end:
            self.scene.reset()
