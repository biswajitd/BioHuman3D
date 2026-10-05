"""
Joint kinematics for muscle-action animation.

The rig is *inferred from the anatomy*, not authored per dataset:

* joint centres are estimated from the named long bones — the humeral head is
  the top cap of the humerus, the elbow its distal cap, the hip the medial part
  of the femur's top cap, and so on — so the same code drives the reference
  body and an imported BodyParts3D atlas;
* each limb is a chain (shoulder → elbow → wrist → hand tip; hip → knee →
  ankle → toe tip) with anatomical degrees of freedom whose positive
  directions are fixed by anatomy (flexion carries the distal joint anteriorly,
  abduction laterally, internal rotation turns the anterior surface medially …)
  and are verified numerically against the body frame;
* bones move rigidly with their segment; soft tissue (muscles, tendons,
  vessels, nerves, skin) is deformed with linear-blend skinning, so a muscle
  that crosses a joint stretches or shortens instead of being torn apart;
* forearm pronation is a *distributed twist* along the forearm, which is how
  the radius rolls over the ulna.

All arrays are numpy; a frame of a full limb costs a few milliseconds.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    from vtkmodules.util.numpy_support import vtk_to_numpy
except Exception:  # pragma: no cover
    from vtk.util.numpy_support import vtk_to_numpy  # type: ignore

ANTERIOR = np.array([0.0, -1.0, 0.0])
POSTERIOR = -ANTERIOR
UP = np.array([0.0, 0.0, 1.0])

#: Layers whose structures move rigidly (one segment per structure).
RIGID_LAYERS = ("skeleton", "cartilage")
#: Layers deformed by skinning. Viscera never move with a limb.
SOFT_LAYERS = ("muscles", "tendons", "arteries", "veins", "nerves", "lymphatics", "skin", "fascia")

DOFS = {
    "upper": ("shoulder_flex", "shoulder_abd", "shoulder_rot", "elbow_flex", "pronation",
              "wrist_flex", "wrist_dev"),
    "lower": ("hip_flex", "hip_abd", "hip_rot", "knee_flex", "ankle_dorsi", "ankle_inv"),
}


def smoothstep(e0: float, e1: float, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def rotation(axis: Sequence[float], degrees: float) -> np.ndarray:
    """3×3 rotation (Rodrigues)."""
    a = np.asarray(axis, float)
    a = a / (np.linalg.norm(a) or 1.0)
    th = math.radians(degrees)
    k = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(th) * k + (1 - math.cos(th)) * (k @ k)


def about(pivot: Sequence[float], rot: np.ndarray) -> np.ndarray:
    """4×4 rotation about a pivot point."""
    p = np.asarray(pivot, float)
    m = np.eye(4)
    m[:3, :3] = rot
    m[:3, 3] = p - rot @ p
    return m


def _apply(m: np.ndarray, pts: np.ndarray) -> np.ndarray:
    return pts @ m[:3, :3].T + m[:3, 3]


def _dist_to_segment(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ab = b - a
    t = np.clip(((p - a) @ ab) / (ab @ ab or 1.0), 0.0, 1.0)
    return np.linalg.norm(p - (a + t[:, None] * ab), axis=1)


# ---------------------------------------------------------------------------
# Joint estimation from bones
# ---------------------------------------------------------------------------
def _cap(points: np.ndarray, top: bool, fraction: float) -> np.ndarray:
    z = points[:, 2]
    lo, hi = z.min(), z.max()
    cut = hi - (hi - lo) * fraction if top else lo + (hi - lo) * fraction
    sel = points[z >= cut] if top else points[z <= cut]
    return sel if len(sel) else points


@dataclass
class Limb:
    side: str                    # "left" | "right"
    kind: str                    # "upper" | "lower"
    joints: List[np.ndarray]     # J1, J2, J3
    tip: np.ndarray
    axes: Dict[str, np.ndarray] = field(default_factory=dict)
    blend: Tuple[float, float, float] = (0.035, 0.025, 0.015)
    radius: Tuple[float, float] = (0.065, 0.11)

    @property
    def key(self) -> str:
        return f"{self.side}_{self.kind}"

    @property
    def s(self) -> float:
        return 1.0 if self.side == "left" else -1.0

    def chain(self) -> List[np.ndarray]:
        return [*self.joints, self.tip]


def _bone_points(bones: Dict[str, np.ndarray], side: str, *words: str) -> Optional[np.ndarray]:
    hits = [pts for name, pts in bones.items()
            if side in name and all(w in name for w in words)]
    return np.vstack(hits) if hits else None


_FOOT_WORDS = ("foot", "toe", "metatars", "tarsal", "calcane", "talus", "navicular", "cuneiform", "cuboid")
_HAND_WORDS = ("hand", "metacarp", "finger", "thumb", "carpal", "phalan")


def _is_foot(name: str) -> bool:
    return any(k in name for k in _FOOT_WORDS)


def _is_hand(name: str) -> bool:
    return not _is_foot(name) and any(k in name for k in _HAND_WORDS) and "carpal bones" not in name


def estimate_limbs(bones: Dict[str, np.ndarray]) -> Dict[str, Limb]:
    """Infer limb chains from bone point clouds keyed by lower-case name."""
    limbs: Dict[str, Limb] = {}
    for side in ("left", "right"):
        s = 1.0 if side == "left" else -1.0
        humerus = _bone_points(bones, side, "humerus")
        if humerus is not None:
            shoulder = _cap(humerus, True, 0.10).mean(axis=0)
            # The humeral head is medial to the greater tubercle.
            top = _cap(humerus, True, 0.14)
            medial = top[(top[:, 0] * s) <= np.median(top[:, 0] * s)]
            if len(medial):
                shoulder = medial.mean(axis=0)
            elbow = _cap(humerus, False, 0.07).mean(axis=0)
            forearm = [p for p in (_bone_points(bones, side, "radius"), _bone_points(bones, side, "ulna"))
                       if p is not None]
            if forearm:
                wrist = _cap(np.vstack(forearm), False, 0.06).mean(axis=0)
            else:
                wrist = elbow + (elbow - shoulder) * 0.82
            hand = [p for name, p in bones.items() if side in name and _is_hand(name)]
            tip = (_cap(np.vstack(hand), False, 0.05).mean(axis=0) if hand
                   else wrist + (wrist - elbow) * 0.70)
            limbs[f"{side}_upper"] = Limb(side, "upper", [shoulder, elbow, wrist], tip,
                                          blend=(0.040, 0.028, 0.016), radius=(0.055, 0.095))
        femur = _bone_points(bones, side, "femur")
        if femur is not None:
            top = _cap(femur, True, 0.12)
            medial = top[(top[:, 0] * s) <= np.percentile(top[:, 0] * s, 40)]
            hip = (medial if len(medial) else top).mean(axis=0)
            knee = _cap(femur, False, 0.06).mean(axis=0)
            tibia = _bone_points(bones, side, "tibia")
            if tibia is not None:        # joint centre lies between condyles and plateau
                knee = (knee + _cap(tibia, True, 0.05).mean(axis=0)) / 2
            leg = [p for p in (tibia, _bone_points(bones, side, "fibula"))
                   if p is not None]
            ankle = _cap(np.vstack(leg), False, 0.05).mean(axis=0) if leg else knee + (knee - hip)
            foot = [p for name, p in bones.items() if side in name and _is_foot(name)]
            if foot:
                fp = np.vstack(foot)
                tip = fp[np.argmin(fp[:, 1])]                 # most anterior point
            else:
                tip = ankle + np.array([0.0, -0.16, -0.05])
            limbs[f"{side}_lower"] = Limb(side, "lower", [hip, knee, ankle], tip,
                                          blend=(0.050, 0.030, 0.020), radius=(0.10, 0.16))
    for limb in limbs.values():
        _define_axes(limb)
    return limbs


def _oriented(axis: np.ndarray, pivot: np.ndarray, probe: np.ndarray, desired: np.ndarray) -> np.ndarray:
    """Flip *axis* so a small positive rotation moves *probe* along *desired*."""
    axis = axis / (np.linalg.norm(axis) or 1.0)
    moved = rotation(axis, 8.0) @ (probe - pivot) + pivot - probe
    return axis if float(moved @ desired) >= 0 else -axis


def _define_axes(limb: Limb) -> None:
    j1, j2, j3 = limb.joints
    tip = limb.tip
    lateral = np.array([limb.s, 0.0, 0.0])
    medial = -lateral
    x, y = np.array([1.0, 0, 0]), np.array([0, 1.0, 0])
    a = limb.axes
    if limb.kind == "upper":
        a["shoulder_flex"] = _oriented(x, j1, j2, ANTERIOR)
        a["shoulder_abd"] = _oriented(y, j1, j2, lateral)
        a["shoulder_rot"] = _oriented(j2 - j1, j1, j2 + ANTERIOR * 0.05, medial)
        a["elbow_flex"] = _oriented(x, j2, j3, ANTERIOR)
        a["pronation"] = _oriented(j3 - j2, j2, j3 + ANTERIOR * 0.05, medial)
        a["wrist_flex"] = _oriented(x, j3, tip, ANTERIOR)
        a["wrist_dev"] = _oriented(y, j3, tip, lateral)
    else:
        a["hip_flex"] = _oriented(x, j1, j2, ANTERIOR)
        a["hip_abd"] = _oriented(y, j1, j2, lateral)
        a["hip_rot"] = _oriented(j2 - j1, j1, j2 + ANTERIOR * 0.05, medial)
        a["knee_flex"] = _oriented(x, j2, j3, POSTERIOR)
        a["ankle_dorsi"] = _oriented(x, j3, tip, UP)
        foot_axis = tip - j3
        foot_axis[2] = 0.0
        a["ankle_inv"] = _oriented(foot_axis, j3, j3 + medial * 0.04 + np.array([0, -0.05, -0.04]), UP)


# ---------------------------------------------------------------------------
# Binding (skin weights) and posing
# ---------------------------------------------------------------------------
@dataclass
class _Binding:
    layer: str
    index: int
    poly: object
    rest_points: np.ndarray
    rest_normals: Optional[np.ndarray]
    weights: np.ndarray            # (N, 4): root, seg1, seg2, seg3
    twist_t: np.ndarray            # (N,) position along the forearm (upper limb only)


class Rig:
    """Poses limbs of a loaded :class:`Interactive3DViewport` scene."""

    def __init__(self, viewport) -> None:
        self._viewport = viewport
        self.limbs: Dict[str, Limb] = {}
        self._bindings: Dict[str, List[_Binding]] = {}
        self._posed: Dict[str, Dict[str, float]] = {}
        self.refresh()

    # ----------------------------------------------------------- building
    def refresh(self) -> None:
        self.reset()
        bones: Dict[str, np.ndarray] = {}
        for layer_id in RIGID_LAYERS[:1]:
            state = self._viewport.layers.get(layer_id)
            if state is None:
                continue
            for name, poly in zip(state.structures, state.polydata):
                if name and poly.GetNumberOfPoints():
                    bones[name.lower()] = vtk_to_numpy(poly.GetPoints().GetData()).astype(float)
        self.limbs = estimate_limbs(bones)
        self._bindings.clear()

    @property
    def available(self) -> bool:
        return bool(self.limbs)

    def limb_for(self, side: str, kind: str) -> Optional[Limb]:
        return self.limbs.get(f"{side}_{kind}")

    def _bind(self, limb: Limb) -> List[_Binding]:
        if limb.key in self._bindings:
            return self._bindings[limb.key]
        chain = limb.chain()
        lo = np.min(chain, axis=0) - limb.radius[1]
        hi = np.max(chain, axis=0) + limb.radius[1]
        other = "right" if limb.side == "left" else "left"
        out: List[_Binding] = []
        for layer_id in RIGID_LAYERS + SOFT_LAYERS:
            state = self._viewport.layers.get(layer_id)
            if state is None:
                continue
            for index, (name, poly) in enumerate(zip(state.structures, state.polydata)):
                low = (name or "").lower()
                if other in low.split() or low.startswith(other):
                    continue
                b = poly.GetBounds()
                if b[1] < lo[0] or b[0] > hi[0] or b[3] < lo[1] or b[2] > hi[1] or b[5] < lo[2] or b[4] > hi[2]:
                    continue
                pts = vtk_to_numpy(poly.GetPoints().GetData()).astype(float)
                rigid = layer_id in RIGID_LAYERS
                if not rigid and not self._member(limb, name, pts):
                    continue
                weights, twist = self._weights(limb, pts, rigid)
                if weights[:, 1:].max() < 1e-3:
                    continue
                normals = poly.GetPointData().GetNormals()
                out.append(_Binding(layer_id, index, poly, pts.copy(),
                                    vtk_to_numpy(normals).astype(float).copy() if normals is not None else None,
                                    weights, twist))
        self._bindings[limb.key] = out
        return out

    def _member(self, limb: Limb, name: str, pts: np.ndarray) -> bool:
        """Soft tissue follows a limb only if it belongs to it (its centroid
        moves with the limb) or it is a muscle acting across the limb's joints
        (latissimus dorsi, pectoralis major …). Trunk-wall structures that merely
        lie next to the arm — serratus anterior, the obliques, trunk skin — stay."""
        w, _t = self._weights(limb, pts.mean(axis=0, keepdims=True), rigid=True)
        if float(w[0, 0]) < 0.5:
            return True
        from app.anatomy.muscles import MOTIONS, muscle_for_structure
        muscle = muscle_for_structure(name) if name else None
        return bool(muscle and any(MOTIONS[a].limb == limb.kind for a in muscle.actions if a in MOTIONS))

    def _weights(self, limb: Limb, pts: np.ndarray, rigid: bool) -> Tuple[np.ndarray, np.ndarray]:
        j1, j2, j3 = limb.joints
        tip = limb.tip
        u1, u2, u3 = ((b - a) / (np.linalg.norm(b - a) or 1.0) for a, b in ((j1, j2), (j2, j3), (j3, tip)))
        b1, b2, b3 = limb.blend

        def distal(p):
            d1 = smoothstep(-b1, b1, (p - j1) @ u1)
            d2 = d1 * smoothstep(-b2, b2, (p - j2) @ u2)
            d3 = d2 * smoothstep(-b3, b3, (p - j3) @ u3)
            return d1, d2, d3

        def gate(p):
            dist = np.min(np.stack([_dist_to_segment(p, a, b) for a, b in ((j1, j2), (j2, j3), (j3, tip))]), axis=0)
            g = 1.0 - smoothstep(limb.radius[0], limb.radius[1], dist)
            lateral_min = 0.06 if limb.kind == "upper" else 0.0
            return g * smoothstep(lateral_min - 0.02, lateral_min + 0.03, p[:, 0] * limb.s)

        sample = pts.mean(axis=0, keepdims=True) if rigid else pts
        d1, d2, d3 = distal(sample)
        g = gate(sample)
        w = np.stack([1.0 - g * d1, g * (d1 - d2), g * (d2 - d3), g * d3], axis=1)
        if rigid:
            one_hot = np.zeros_like(w)
            one_hot[0, int(np.argmax(w[0]))] = 1.0
            w = np.repeat(one_hot, len(pts), axis=0)
        twist = np.clip(((pts - j2) @ u2) / (np.linalg.norm(j3 - j2) or 1.0), 0.0, 1.0)
        return w, twist

    # ------------------------------------------------------------- posing
    def segment_matrices(self, limb: Limb, pose: Dict[str, float]) -> List[np.ndarray]:
        j1, j2, j3 = limb.joints
        a = limb.axes
        if limb.kind == "upper":
            m1 = (about(j1, rotation(a["shoulder_flex"], pose.get("shoulder_flex", 0.0)))
                  @ about(j1, rotation(a["shoulder_abd"], pose.get("shoulder_abd", 0.0)))
                  @ about(j1, rotation(a["shoulder_rot"], pose.get("shoulder_rot", 0.0))))
            m2 = m1 @ about(j2, rotation(a["elbow_flex"], pose.get("elbow_flex", 0.0)))
            m3 = (m2 @ about(j2, rotation(a["pronation"], pose.get("pronation", 0.0)))
                  @ about(j3, rotation(a["wrist_flex"], pose.get("wrist_flex", 0.0)))
                  @ about(j3, rotation(a["wrist_dev"], pose.get("wrist_dev", 0.0))))
        else:
            m1 = (about(j1, rotation(a["hip_flex"], pose.get("hip_flex", 0.0)))
                  @ about(j1, rotation(a["hip_abd"], pose.get("hip_abd", 0.0)))
                  @ about(j1, rotation(a["hip_rot"], pose.get("hip_rot", 0.0))))
            m2 = m1 @ about(j2, rotation(a["knee_flex"], pose.get("knee_flex", 0.0)))
            m3 = (m2 @ about(j3, rotation(a["ankle_dorsi"], pose.get("ankle_dorsi", 0.0)))
                  @ about(j3, rotation(a["ankle_inv"], pose.get("ankle_inv", 0.0))))
        return [np.eye(4), m1, m2, m3]

    def pose(self, limb_key: str, pose: Dict[str, float]) -> None:
        """Apply joint angles (degrees) to one limb. Unlisted DOFs are neutral."""
        limb = self.limbs.get(limb_key)
        if limb is None:
            return
        mats = self.segment_matrices(limb, pose)
        pron = float(pose.get("pronation", 0.0)) if limb.kind == "upper" else 0.0
        j2 = limb.joints[1]
        for bnd in self._bind(limb):
            pts, nrm = bnd.rest_points, bnd.rest_normals
            w = bnd.weights
            out = w[:, :1] * pts
            out_n = w[:, :1] * nrm if nrm is not None else None
            for k in (1, 2, 3):
                wk = w[:, k:k + 1]
                if not wk.any():
                    continue
                src, src_n = pts, nrm
                m = mats[k]
                if k == 2 and pron:
                    # Distributed twist about the forearm axis before the elbow/arm motion.
                    src, src_n = _twist(pts, nrm, j2, limb.axes["pronation"], pron * bnd.twist_t)
                    m = mats[2]
                out = out + wk * _apply(m, src)
                if out_n is not None:
                    out_n = out_n + wk * (src_n @ m[:3, :3].T)
            _write(bnd.poly, out, out_n)
        self._posed[limb_key] = dict(pose)
        self._viewport_render()

    def reset(self) -> None:
        """Return every posed limb to the anatomical position."""
        for bindings in self._bindings.values():
            for bnd in bindings:
                _write(bnd.poly, bnd.rest_points, bnd.rest_normals)
        self._posed.clear()

    def is_posed(self) -> bool:
        return bool(self._posed)

    def is_bound(self, poly) -> bool:
        """True when *poly* is rewritten from its rest shape by :meth:`pose`."""
        return any(b.poly is poly for bindings in self._bindings.values() for b in bindings)

    def _viewport_render(self) -> None:
        render = getattr(self._viewport, "_render", None)
        if render is not None:
            render()


def _twist(pts: np.ndarray, nrm: Optional[np.ndarray], pivot: np.ndarray, axis: np.ndarray,
           angles_deg: np.ndarray):
    """Rotate each point about *axis* through *pivot* by its own angle."""
    a = axis / (np.linalg.norm(axis) or 1.0)
    th = np.radians(angles_deg)[:, None]
    c, s = np.cos(th), np.sin(th)

    def rot(v):
        return v * c + np.cross(a, v) * s + np.outer(v @ a, a) * (1 - c)

    out = rot(pts - pivot) + pivot
    return out, (rot(nrm) if nrm is not None else None)


def _write(poly, points: np.ndarray, normals: Optional[np.ndarray]) -> None:
    arr = vtk_to_numpy(poly.GetPoints().GetData())
    arr[:] = points
    poly.GetPoints().Modified()
    if normals is not None and poly.GetPointData().GetNormals() is not None:
        n = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12)
        vtk_to_numpy(poly.GetPointData().GetNormals())[:] = n
        poly.GetPointData().GetNormals().Modified()
    poly.Modified()
