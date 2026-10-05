"""
Procedural reference body — anthropometrically proportioned, structure-named.

This is the out-of-the-box anatomy used until a real atlas (BodyParts3D /
Z-Anatomy) is imported with ``tools/fetch_anatomy.py``. It replaces the old
ellipsoid stack with:

* joint centres placed by the Drillis & Contini segment ratios for a 1.75 m
  adult (shoulder 0.818 H, elbow 0.630 H, wrist 0.485 H, hip 0.530 H,
  knee 0.285 H, ankle 0.039 H);
* ~90 named bones (vertebrae, 12 rib pairs, long bones with epiphyses, hand and
  foot rays), ~70 named muscles placed between their real origins and
  insertions, and named vessels, nerves and organs in their real topography;
* every structure written under its anatomical English name, so picking,
  muscle actions, disease effects and the MRI simulator address structures by
  name — the same names BodyParts3D uses ("Right biceps brachii").

It is a *schematic* model: proportions and topography are right, surface detail
is not. Real anatomy comes from real scan-derived meshes; see README.

Frame: metres, Z up (floor at z = 0), anterior = -Y, patient's right = -X.
"""
from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

import numpy as np

from app.anatomy import geometry as g

H = 1.75                     # stature, metres
Named = Tuple[str, object]
V = np.ndarray


def _v(*xyz: float) -> V:
    return np.asarray(xyz, dtype=float)


def _lerp(a: V, b: V, t: float) -> V:
    return a + (b - a) * t


SIDES = (("Right", -1.0), ("Left", 1.0))


# ---------------------------------------------------------------------------
# Landmarks (Drillis & Contini 1966; Winter, Biomechanics of Human Movement)
# ---------------------------------------------------------------------------
def landmarks(s: float) -> Dict[str, V]:
    """Joint centres and bony landmarks for one side (s = -1 right, +1 left)."""
    lat = _v(s, 0, 0)
    med = -lat
    L: Dict[str, V] = {
        "shoulder": _v(s * 0.185, 0.0, 0.800 * H),       # glenohumeral centre
        "elbow": _v(s * 0.222, 0.012, 0.630 * H),
        "wrist": _v(s * 0.240, -0.004, 0.485 * H),
        "hip": _v(s * 0.085, -0.020, 0.530 * H),
        "knee": _v(s * 0.092, 0.000, 0.285 * H),
        "ankle": _v(s * 0.098, 0.022, 0.039 * H),
        "acromion": _v(s * 0.178, 0.004, 0.818 * H),
        "coracoid": _v(s * 0.150, -0.045, 0.802 * H),
        "sternoclavicular": _v(s * 0.022, -0.088, 0.805 * H),
        "asis": _v(s * 0.118, -0.078, 0.567 * H),
        "aiis": _v(s * 0.106, -0.066, 0.548 * H),
        "ischial_tuberosity": _v(s * 0.058, 0.032, 0.478 * H),
        "pubic_tubercle": _v(s * 0.022, -0.078, 0.508 * H),
        "greater_trochanter": _v(s * 0.138, 0.000, 0.528 * H),
        "lesser_trochanter": _v(s * 0.090, 0.006, 0.494 * H),
        "mastoid": _v(s * 0.055, 0.028, 0.906 * H),
        "calcaneus": _v(s * 0.100, 0.068, 0.018),
    }
    sh, el, wr, hp, kn, an = (L[k] for k in ("shoulder", "elbow", "wrist", "hip", "knee", "ankle"))
    L.update({
        "deltoid_tuberosity": _lerp(sh, el, 0.42) + lat * 0.013,
        "radial_tuberosity": _lerp(el, wr, 0.10) + _v(0, -0.008, 0),
        "ulnar_tuberosity": _lerp(el, wr, 0.08) + med * 0.006 + _v(0, -0.006, 0),
        "olecranon": el + _v(0, 0.024, 0.012) + med * 0.004,
        "medial_epicondyle": el + med * 0.030,
        "lateral_epicondyle": el + lat * 0.024,
        "radial_styloid": wr + lat * 0.014,
        "ulnar_styloid": wr + med * 0.012 + _v(0, 0.004, 0),
        "intertubercular": _lerp(sh, el, 0.12) + _v(0, -0.014, 0),
        "tibial_tuberosity": _lerp(kn, an, 0.10) + _v(0, -0.034, 0),
        "fibular_head": kn + lat * 0.034 + _v(0, 0.012, -0.030),
        "pes_anserinus": _lerp(kn, an, 0.12) + med * 0.020 + _v(0, -0.018, 0),
        "patella": kn + _v(0, -0.046, 0.022),
        "medial_femoral_condyle": kn + med * 0.024 + _v(0, 0.012, 0.022),
        "lateral_femoral_condyle": kn + lat * 0.024 + _v(0, 0.012, 0.022),
        "medial_malleolus": an + med * 0.020,
        "lateral_malleolus": an + lat * 0.024 + _v(0, 0.008, -0.012),
    })
    return L


def vertebra_levels() -> List[Tuple[str, V]]:
    """Vertebral body centres C1–L5 with cervical lordosis, thoracic kyphosis,
    lumbar lordosis (sagittal profile from standard radiographic norms)."""
    names = ([f"C{i}" for i in range(1, 8)] + [f"T{i}" for i in range(1, 13)]
             + [f"L{i}" for i in range(1, 6)])
    z_top, z_bottom = 0.893 * H, 0.574 * H
    levels = []
    heights = [0.55] * 7 + [0.85] * 12 + [1.25] * 5        # relative body heights
    total = sum(heights)
    z = z_top
    for name, h in zip(names, heights):
        dz = (z_top - z_bottom) * h / total
        zc = z - dz / 2
        t = (z_top - zc) / (z_top - z_bottom)
        # sagittal curve: + is posterior
        y = 0.030 + 0.030 * math.sin(math.pi * min(1.0, max(0.0, (t - 0.15) / 0.55))) \
            - 0.018 * math.sin(math.pi * max(0.0, (t - 0.72) / 0.28))
        levels.append((name, _v(0.0, y, zc)))
        z -= dz
    return levels


def _vertebra(name: str, c: V, size: float) -> List[Named]:
    region = name[0]
    w = {"C": 0.012, "T": 0.016, "L": 0.022}[region] * size
    h = {"C": 0.006, "T": 0.0095, "L": 0.0125}[region]
    label = {"C": "cervical", "T": "thoracic", "L": "lumbar"}[region]
    body = g.ellipsoid(c, (h, w, w * 0.78), rings=14, segments=18)
    spine_len = {"C": 0.022, "T": 0.034, "L": 0.030}[region]
    droop = {"C": 0.004, "T": 0.020, "L": 0.004}[region]
    arch = g.tube([c + _v(0, w * 0.9, 0), c + _v(0, w * 0.9 + spine_len, -droop)],
                  0.0035 * size, 0.0025, segments=10)
    tp = g.tube([c + _v(-w * 1.7, w * 0.9, 0), c + _v(0, w * 1.1, 0), c + _v(w * 1.7, w * 0.9, 0)],
                0.0028, segments=8)
    ring = g.tube([c + _v(-w * 0.55, w * 0.75, 0), c + _v(0, w * 1.45, 0),
                   c + _v(w * 0.55, w * 0.75, 0)], 0.0022, segments=8)
    return [(f"{name} vertebra ({label})", g.append([body, arch, tp, ring]))]


# ---------------------------------------------------------------------------
# Skeleton
# ---------------------------------------------------------------------------
def build_skeleton() -> List[Named]:
    out: List[Named] = []
    levels = vertebra_levels()
    lv = dict(levels)

    # Skull
    cranium = g.ellipsoid(_v(0, 0.008, 0.948 * H), (0.088, 0.073, 0.098), axis=(0, 0, 1),
                          rings=36, segments=40)
    face = g.ellipsoid(_v(0, -0.060, 0.905 * H), (0.045, 0.055, 0.035), rings=24, segments=28)
    out.append(("Cranium", g.append([cranium, face])))
    jaw = [_v(-0.050, 0.010, 0.908 * H), _v(-0.046, -0.030, 0.876 * H),
           _v(-0.020, -0.076, 0.868 * H), _v(0.0, -0.084, 0.868 * H),
           _v(0.020, -0.076, 0.868 * H), _v(0.046, -0.030, 0.876 * H),
           _v(0.050, 0.010, 0.908 * H)]
    out.append(("Mandible", g.tube(jaw, 0.008, segments=14)))

    # Spine
    for name, c in levels:
        out += _vertebra(name, c, 1.0)
    l5 = lv["L5"]
    sacrum_path = [l5 + _v(0, 0.010, -0.028), _v(0, 0.075, 0.535 * H), _v(0, 0.095, 0.505 * H)]
    out.append(("Sacrum", g.lathe(sacrum_path, g.taper(0.050, 0.016), ratio=0.36,
                                  hint=(1, 0, 0), rings=24, segments=24)))
    out.append(("Coccyx", g.tube([_v(0, 0.096, 0.502 * H), _v(0, 0.088, 0.488 * H),
                                  _v(0, 0.076, 0.480 * H)], 0.006, 0.003, segments=10)))

    # Thorax: sternum and 12 rib pairs
    out.append(("Manubrium of sternum", g.lathe([_v(0, -0.088, 0.803 * H), _v(0, -0.098, 0.775 * H)],
                                               g.taper(0.026, 0.017), ratio=0.30, hint=(1, 0, 0),
                                               rings=12, segments=16)))
    out.append(("Body of sternum", g.lathe([_v(0, -0.098, 0.774 * H), _v(0, -0.110, 0.700 * H)],
                                          g.taper(0.017, 0.014), ratio=0.32, hint=(1, 0, 0),
                                          rings=20, segments=16)))
    out.append(("Xiphoid process", g.tube([_v(0, -0.110, 0.699 * H), _v(0, -0.106, 0.684 * H)],
                                         0.006, 0.002, segments=10)))
    for side, s in SIDES:
        for i in range(1, 13):
            c = lv[f"T{i}"]
            width = 0.060 + 0.090 * math.sin(math.pi * min(1.0, (i + 1.5) / 13.0))
            depth = 0.070 + 0.035 * math.sin(math.pi * min(1.0, i / 11.0))
            if i <= 7:
                end_angle = 262.0
            elif i <= 10:
                end_angle = 238.0 - (i - 8) * 8.0
            else:
                end_angle = 190.0 - (i - 11) * 25.0
            z0 = c[2] + 0.004
            z1 = z0 - 0.050 - 0.0055 * i
            centre = (0.0, c[1] - depth + 0.022, 0)
            pts = g.arc_points(centre, width, depth, 88.0, end_angle, z0, z1, n=28)
            pts = [(s * -p[0], p[1], p[2]) for p in pts]      # right ribs on -X
            r = 0.0045 if i > 1 else 0.0055
            out.append((f"{side} rib {i}", g.tube(pts, r, r * 0.8, segments=10)))

    # Shoulder girdle and limbs
    for side, s in SIDES:
        L = landmarks(s)
        lat = _v(s, 0, 0)
        sc, ac = L["sternoclavicular"], L["acromion"]
        out.append((f"{side} clavicle", g.tube(
            [sc, _lerp(sc, ac, 0.35) + _v(0, -0.018, 0.006), _lerp(sc, ac, 0.75) + _v(0, 0.004, 0.010), ac],
            0.0065, 0.0055, segments=12)))
        scap_c = _v(s * 0.112, 0.098, 0.757 * H)
        blade = g.lathe([_v(s * 0.086, 0.098, 0.808 * H), scap_c, _v(s * 0.102, 0.106, 0.700 * H)],
                        g.blend((0, 0.030), (0.35, 0.048), (0.8, 0.026), (1, 0.006)),
                        ratio=0.16, hint=(s * 1.0, -0.35 * s * s, 0), rings=24, segments=20)
        spine = g.tube([_v(s * 0.078, 0.110, 0.790 * H), _v(s * 0.140, 0.072, 0.812 * H), ac],
                       0.0045, 0.006, segments=10)
        glenoid = g.ellipsoid(L["shoulder"] + lat * -0.022, (0.017, 0.012, 0.007), axis=(0, 0, 1),
                              rings=12, segments=14)
        out.append((f"{side} scapula", g.append([blade, spine, glenoid])))

        sh, el, wr = L["shoulder"], L["elbow"], L["wrist"]
        up = _v(0, 0, 1)
        out.append((f"{side} humerus", g.lathe(
            [sh + up * 0.020, el - up * 0.016], g.long_bone(0.0115, 0.025, 0.026, 0.12, 0.10),
            ratio=0.92, rings=44, segments=24)))
        out.append((f"{side} radius", g.lathe(
            [el + lat * 0.014 - up * 0.008, _lerp(el, wr, 0.5) + lat * 0.016, wr + lat * 0.008],
            g.long_bone(0.0075, 0.011, 0.016, 0.08, 0.14), rings=36, segments=18)))
        out.append((f"{side} ulna", g.lathe(
            [L["olecranon"] + up * 0.010, el - lat * 0.012 - up * 0.010,
             _lerp(el, wr, 0.5) - lat * 0.012, wr - lat * 0.010],
            g.long_bone(0.0068, 0.016, 0.009, 0.14, 0.08), rings=36, segments=18)))
        out += _hand_bones(side, s, L)

        hp, kn, an = L["hip"], L["knee"], L["ankle"]
        out += _hip_bone(side, s, L)
        head = g.ellipsoid(hp, (0.023, 0.023, 0.023), rings=18, segments=20)
        neck = g.tube([hp, L["greater_trochanter"] - lat * 0.010], 0.014, segments=14)
        troch = g.ellipsoid(L["greater_trochanter"], (0.026, 0.017, 0.016), rings=14, segments=16)
        shaft = g.lathe([L["greater_trochanter"] - lat * 0.012 - up * 0.010, kn + up * 0.020],
                        g.long_bone(0.0135, 0.020, 0.040, 0.06, 0.13), ratio=0.95, rings=48, segments=24)
        out.append((f"{side} femur", g.append([head, neck, troch, shaft])))
        out.append((f"{side} patella", g.ellipsoid(L["patella"], (0.022, 0.020, 0.010),
                                                    rings=14, segments=16)))
        out.append((f"{side} tibia", g.lathe(
            [kn - up * 0.008 - lat * 0.004, an + up * 0.006 - lat * 0.006],
            g.long_bone(0.0135, 0.036, 0.021, 0.12, 0.10), ratio=0.92, rings=48, segments=24)))
        out.append((f"{side} fibula", g.lathe(
            [L["fibular_head"], _lerp(kn, an, 0.5) + lat * 0.030 + _v(0, 0.012, 0),
             L["lateral_malleolus"]],
            g.long_bone(0.0055, 0.010, 0.010, 0.06, 0.08), rings=40, segments=14)))
        out += _foot_bones(side, s, L)
    return out


def _hip_bone(side: str, s: float, L: Dict[str, V]) -> List[Named]:
    ilium = g.lathe([L["hip"] + _v(s * 0.010, 0.010, 0.020), _v(s * 0.112, 0.010, 0.575 * H),
                     _v(s * 0.104, 0.030, 0.597 * H)],
                    g.blend((0, 0.022), (0.45, 0.060), (0.85, 0.058), (1, 0.010)),
                    ratio=0.20, hint=(0.35 * s, -1.0, 0.0), rings=24, segments=22)
    crest = g.tube([L["asis"], _v(s * 0.128, -0.030, 0.592 * H), _v(s * 0.115, 0.030, 0.600 * H),
                    _v(s * 0.070, 0.075, 0.585 * H), _v(s * 0.045, 0.080, 0.555 * H)],
                   0.0065, segments=10)
    acetab = g.tube(g.arc_points((L["hip"][0], L["hip"][1], 0), 0.028, 0.028, 0, 330,
                                 L["hip"][2], L["hip"][2], n=20), 0.006, segments=8)
    acetab = g.transform(acetab, rotate=[("y", 70 * s)], about=L["hip"])
    pubic = g.tube([L["hip"] + _v(-s * 0.018, -0.020, -0.004), L["pubic_tubercle"],
                    _v(s * 0.004, -0.070, 0.502 * H)], 0.0075, segments=10)
    ischial = g.tube([L["hip"] + _v(-s * 0.006, 0.016, -0.018), L["ischial_tuberosity"],
                      _v(s * 0.030, -0.040, 0.480 * H), _v(s * 0.006, -0.068, 0.496 * H)],
                     0.0085, 0.006, segments=10)
    return [(f"{side} hip bone", g.append([ilium, crest, acetab, pubic, ischial]))]


def _hand_bones(side: str, s: float, L: Dict[str, V]) -> List[Named]:
    wr = L["wrist"]
    lat = _v(s, 0, 0)
    out = [(f"{side} carpal bones", g.ellipsoid(wr - _v(0, 0, 0.022), (0.016, 0.026, 0.011),
                                                rings=12, segments=16))]
    rays = []
    for k in range(4):                          # index → little finger
        base = wr - _v(0, 0, 0.038) + lat * (0.017 - k * 0.012)
        mc_end = base - _v(0, 0, 0.060 - k * 0.004) + lat * (0.004 - k * 0.003)
        tip = mc_end - _v(0, 0, 0.085 - abs(k - 1.2) * 0.010) + lat * (0.003 - k * 0.003)
        rays.append(g.tube([base, mc_end], 0.0042, 0.0038, segments=8))
        rays.append(g.tube([mc_end, _lerp(mc_end, tip, 0.45), tip], 0.0036, 0.0024, segments=8))
    tb = wr - _v(0, 0, 0.030) + lat * 0.026 + _v(0, -0.010, 0)
    tm = tb + lat * 0.010 - _v(0, 0, 0.038) + _v(0, -0.012, 0)
    ttip = tm + lat * 0.006 - _v(0, 0, 0.050) + _v(0, -0.010, 0)
    rays.append(g.tube([tb, tm], 0.0046, 0.0040, segments=8))
    rays.append(g.tube([tm, ttip], 0.0040, 0.0028, segments=8))
    out.append((f"{side} metacarpals and phalanges of hand", g.append(rays)))
    return out


def _foot_bones(side: str, s: float, L: Dict[str, V]) -> List[Named]:
    an = L["ankle"]
    lat = _v(s, 0, 0)
    out = [(f"{side} talus", g.ellipsoid(an + _v(0, -0.004, -0.022), (0.016, 0.020, 0.026),
                                         axis=(0, 1, 0), rings=12, segments=14)),
           (f"{side} calcaneus", g.lathe([L["calcaneus"] + _v(0, 0.012, 0.010),
                                          an + _v(0, -0.010, -0.040)],
                                         g.blend((0, 0.016), (0.4, 0.022), (1, 0.012)),
                                         rings=16, segments=16))]
    rays = []
    mid = an + _v(0, -0.060, -0.040)
    rays.append(g.ellipsoid(mid, (0.018, 0.024, 0.012), axis=(0, 1, 0), rings=10, segments=12))
    for k in range(5):                          # hallux → 5th toe
        base = mid + _v(0, -0.018, -0.006) + lat * (-0.018 + k * 0.011)
        head = base + _v(0, -0.062 + k * 0.004, -0.018) + lat * (-0.004 + k * 0.004)
        tip = head + _v(0, -0.034 + k * 0.004, -0.004)
        r = 0.0060 if k == 0 else 0.0038
        rays.append(g.tube([base, head], r, r * 0.85, segments=8))
        rays.append(g.tube([head, tip], r * 0.85, r * 0.6, segments=8))
    out.append((f"{side} tarsal, metatarsal and phalanges of foot", g.append(rays)))
    return out


# ---------------------------------------------------------------------------
# Cartilage
# ---------------------------------------------------------------------------
def build_cartilage() -> List[Named]:
    out: List[Named] = []
    levels = vertebra_levels()
    for (n0, c0), (n1, c1) in zip(levels, levels[1:]):
        if n0 in ("C1",):
            continue
        mid = (c0 + c1) / 2
        w = {"C": 0.011, "T": 0.015, "L": 0.021}[n1[0]]
        out.append((f"Intervertebral disc {n0}-{n1}",
                    g.ellipsoid(mid, (0.0025 if n1[0] != "L" else 0.0045, w, w * 0.78),
                                rings=8, segments=16)))
    for side, s in SIDES:
        L = landmarks(s)
        for i in range(1, 11):
            c = dict(levels)[f"T{i}"]
            width = 0.060 + 0.090 * math.sin(math.pi * min(1.0, (i + 1.5) / 13.0))
            depth = 0.070 + 0.035 * math.sin(math.pi * min(1.0, i / 11.0))
            end_angle = 262.0 if i <= 7 else 238.0 - (i - 8) * 8.0
            z0 = c[2] + 0.004
            z1 = z0 - 0.050 - 0.0055 * i
            ang = math.radians(end_angle)
            end = _v(-s * width * math.cos(ang), c[1] - depth + 0.022 + depth * math.sin(ang), z1)
            stern_z = max(0.700 * H, min(0.795 * H, z1 + (0.02 if i <= 7 else 0.05)))
            target = _v(s * 0.014, -0.104, stern_z)
            out.append((f"{side} costal cartilage {i}", g.tube([end, _lerp(end, target, 0.6) + _v(0, -0.004, 0), target],
                                                              0.0042, 0.0038, segments=8)))
        lat = _v(s, 0, 0)
        out.append((f"{side} glenoid labrum and humeral articular cartilage",
                    g.ellipsoid(L["shoulder"] - lat * 0.016, (0.020, 0.010, 0.016), rings=12, segments=14)))
        out.append((f"{side} acetabular labrum", g.ellipsoid(L["hip"] + lat * 0.004, (0.026, 0.010, 0.026),
                                                              axis=(1, 0, 0), rings=12, segments=16)))
        for meniscus, off in (("medial meniscus", -0.020), ("lateral meniscus", 0.020)):
            arc = g.arc_points((L["knee"][0] + s * off, L["knee"][1] + 0.004, 0), 0.016, 0.018,
                               -150 if off < 0 else 30, 150 if off < 0 else 330,
                               L["knee"][2] - 0.004, L["knee"][2] - 0.004, n=18)
            out.append((f"{side} {meniscus}", g.tube(arc, 0.0045, segments=8)))
        out.append((f"{side} articular cartilage of knee", g.ellipsoid(L["knee"] + _v(0, 0.004, 0.012),
                                                                       (0.006, 0.040, 0.026), rings=10, segments=16)))
        out.append((f"{side} articular cartilage of ankle", g.ellipsoid(L["ankle"] + _v(0, -0.004, -0.010),
                                                                        (0.005, 0.017, 0.020), rings=8, segments=12)))
    return out


# ---------------------------------------------------------------------------
# Muscles — origin → insertion paths (Gray's Anatomy, 42nd ed.)
# ---------------------------------------------------------------------------
def muscle_specs(s: float) -> List[dict]:
    """Per-side muscle definitions. ``path`` runs origin → insertion;
    ``normal`` (optional) is the outward surface normal for flat muscles."""
    L = landmarks(s)
    lat, med = _v(s, 0, 0), _v(-s, 0, 0)
    sh, el, wr, hp, kn, an = (L[k] for k in ("shoulder", "elbow", "wrist", "hip", "knee", "ankle"))
    post, ant = _v(0, 1, 0), _v(0, -1, 0)
    lv = dict(vertebra_levels())
    spinous = lambda lvl, extra=0.0: lv[lvl] + _v(0, 0.050 + extra, 0)   # noqa: E731
    M = []

    def add(name, path, r, r_end=0.004, ratio=0.8, normal=None, power=0.75):
        M.append(dict(name=name, path=[np.asarray(p, float) for p in path], r=r, r_end=r_end,
                      ratio=ratio, normal=normal, power=power))

    # -- head & neck -----------------------------------------------------
    add("sternocleidomastoid", [L["mastoid"], _v(s * 0.042, -0.040, 0.858 * H),
                                L["sternoclavicular"] + _v(-s * 0.006, -0.004, 0.010)], 0.012, 0.004, 0.55)
    add("masseter", [_v(s * 0.056, -0.034, 0.912 * H), _v(s * 0.050, -0.030, 0.878 * H)], 0.013, 0.006, 0.45,
        normal=lat)
    add("temporalis", [_v(s * 0.070, 0.010, 0.975 * H), _v(s * 0.075, -0.010, 0.945 * H),
                       _v(s * 0.050, -0.040, 0.905 * H)], 0.028, 0.004, 0.18, normal=lat)

    # -- shoulder & back ---------------------------------------------------
    add("trapezius (descending part)", [_v(s * 0.010, 0.088, 0.915 * H), _v(s * 0.060, 0.072, 0.845 * H),
                                        L["acromion"] + _v(0, 0.018, 0.004)], 0.030, 0.008, 0.25, normal=post)
    add("trapezius (ascending part)", [spinous("T10", 0.012) + lat * 0.010, _v(s * 0.065, 0.122, 0.755 * H),
                                       _v(s * 0.110, 0.100, 0.808 * H)], 0.040, 0.008, 0.18, normal=post)
    add("rhomboid major", [spinous("T3", 0.004) + lat * 0.008, _v(s * 0.084, 0.114, 0.745 * H)],
        0.022, 0.010, 0.25, normal=post)
    add("latissimus dorsi", [_v(s * 0.020, 0.104, 0.600 * H), _v(s * 0.120, 0.104, 0.680 * H),
                             _v(s * 0.150, 0.060, 0.755 * H), L["intertubercular"] + _v(0, 0.006, -0.010)],
        0.060, 0.008, 0.18, normal=post + lat * 0.4)
    add("erector spinae", [_v(s * 0.024, 0.084, 0.548 * H), _v(s * 0.028, 0.104, 0.640 * H),
                           _v(s * 0.026, 0.112, 0.740 * H), _v(s * 0.022, 0.090, 0.820 * H)],
        0.022, 0.008, 0.75)
    add("supraspinatus", [_v(s * 0.084, 0.098, 0.805 * H), _v(s * 0.140, 0.050, 0.812 * H),
                          sh + _v(0, -0.002, 0.024) + lat * 0.010], 0.012, 0.004, 0.7)
    add("infraspinatus", [_v(s * 0.100, 0.112, 0.740 * H), _v(s * 0.150, 0.070, 0.780 * H),
                          sh + _v(0, 0.020, 0.004) + lat * 0.012], 0.026, 0.005, 0.3, normal=post)
    add("teres major", [_v(s * 0.106, 0.110, 0.705 * H), L["intertubercular"] + _v(0, 0.010, -0.018)],
        0.016, 0.005, 0.6)
    add("deltoid", [L["acromion"] + lat * 0.006 + _v(0, 0, 0.012), sh + lat * 0.036 + _v(0, 0, -0.012),
                    L["deltoid_tuberosity"]], 0.032, 0.006, 0.55, normal=lat)
    add("serratus anterior", [_v(s * 0.150, -0.040, 0.700 * H), _v(s * 0.160, 0.020, 0.735 * H),
                              _v(s * 0.112, 0.095, 0.760 * H)], 0.040, 0.008, 0.16, normal=lat)
    add("pectoralis major", [_v(s * 0.020, -0.118, 0.745 * H), _v(s * 0.090, -0.112, 0.765 * H),
                             L["intertubercular"]], 0.058, 0.010, 0.22, normal=ant)
    add("pectoralis minor", [_v(s * 0.080, -0.100, 0.725 * H), L["coracoid"]], 0.026, 0.006, 0.25, normal=ant)

    # -- arm & forearm -----------------------------------------------------
    add("biceps brachii", [L["coracoid"], _lerp(sh, el, 0.50) + _v(0, -0.031, 0),
                           L["radial_tuberosity"]], 0.022, 0.004, 0.80)
    add("brachialis", [_lerp(sh, el, 0.52) + _v(0, -0.012, 0), el + _v(0, -0.022, 0.010),
                       L["ulnar_tuberosity"]], 0.017, 0.004, 0.75)
    add("triceps brachii", [sh + _v(0, 0.020, -0.026), _lerp(sh, el, 0.50) + _v(0, 0.031, 0),
                            L["olecranon"]], 0.026, 0.006, 0.80)
    add("brachioradialis", [_lerp(sh, el, 0.74) + lat * 0.022, _lerp(el, wr, 0.25) + lat * 0.020 + _v(0, -0.012, 0),
                            L["radial_styloid"]], 0.014, 0.003, 0.75)
    add("pronator teres", [L["medial_epicondyle"], _lerp(el, wr, 0.45) + lat * 0.014 + _v(0, -0.006, 0)],
        0.009, 0.003, 0.7)
    add("flexor carpi radialis", [L["medial_epicondyle"] + _v(0, -0.006, 0),
                                  _lerp(el, wr, 0.40) + _v(0, -0.018, 0), wr + _v(0, -0.012, -0.020)],
        0.011, 0.003, 0.7)
    add("flexor carpi ulnaris", [L["medial_epicondyle"] + _v(0, 0.008, 0), _lerp(el, wr, 0.45) + med * 0.016,
                                 L["ulnar_styloid"] + _v(0, -0.004, -0.018)], 0.011, 0.003, 0.7)
    add("flexor digitorum superficialis", [L["medial_epicondyle"] + _v(0, -0.010, -0.008),
                                           _lerp(el, wr, 0.50) + _v(0, -0.016, 0) + med * 0.004,
                                           wr + _v(0, -0.010, -0.030)], 0.013, 0.004, 0.6)
    add("extensor carpi radialis longus", [L["lateral_epicondyle"] + _v(0, 0, 0.018),
                                           _lerp(el, wr, 0.40) + lat * 0.016 + _v(0, 0.006, 0),
                                           wr + lat * 0.010 + _v(0, 0.010, -0.030)], 0.010, 0.003, 0.7)
    add("extensor digitorum", [L["lateral_epicondyle"] + _v(0, 0.010, 0), _lerp(el, wr, 0.50) + _v(0, 0.016, 0),
                               wr + _v(0, 0.012, -0.030)], 0.012, 0.003, 0.65)

    # -- trunk wall --------------------------------------------------------
    add("rectus abdominis", [L["pubic_tubercle"] + _v(0, -0.008, 0.004), _v(s * 0.045, -0.132, 0.600 * H),
                             _v(s * 0.060, -0.118, 0.700 * H)], 0.034, 0.012, 0.22, normal=ant, power=0.4)
    add("external oblique", [_v(s * 0.150, -0.050, 0.705 * H), _v(s * 0.140, -0.095, 0.625 * H),
                             _v(s * 0.075, -0.124, 0.560 * H)], 0.052, 0.010, 0.16, normal=ant + lat)
    add("diaphragm (hemidiaphragm)", [_v(s * 0.010, 0.050, 0.655 * H), _v(s * 0.090, 0.000, 0.700 * H),
                                     _v(s * 0.010, -0.090, 0.690 * H)], 0.085, 0.020, 0.10,
        normal=_v(0, 0, 1), power=0.5)

    # -- hip & thigh -------------------------------------------------------
    add("iliopsoas", [_v(s * 0.032, 0.034, 0.640 * H), _v(s * 0.078, -0.030, 0.550 * H),
                      L["lesser_trochanter"] + _v(0, -0.006, 0)], 0.024, 0.006, 0.8)
    add("gluteus maximus", [_v(s * 0.030, 0.106, 0.555 * H), _v(s * 0.100, 0.104, 0.515 * H),
                            L["greater_trochanter"] + _v(0, 0.034, -0.050)], 0.062, 0.012, 0.42,
        normal=post + lat * 0.3)
    add("gluteus medius", [_v(s * 0.118, 0.030, 0.592 * H), _v(s * 0.140, 0.012, 0.556 * H),
                           L["greater_trochanter"] + lat * 0.010 + _v(0, 0, 0.012)], 0.040, 0.008, 0.40,
        normal=lat)
    add("tensor fasciae latae", [L["asis"] + lat * 0.006, L["greater_trochanter"] + _v(0, -0.030, -0.040) + lat * 0.012],
        0.016, 0.006, 0.45, normal=lat)
    add("sartorius", [L["asis"], _lerp(hp, kn, 0.45) + med * 0.012 + _v(0, -0.050, 0),
                      _lerp(hp, kn, 0.85) + med * 0.044 + _v(0, 0.000, 0), L["pes_anserinus"]],
        0.011, 0.004, 0.5)
    add("rectus femoris", [L["aiis"], _lerp(hp, kn, 0.50) + _v(0, -0.060, 0),
                           L["patella"] + _v(0, 0, 0.026)], 0.028, 0.006, 0.75)
    add("vastus lateralis", [L["greater_trochanter"] + _v(0, -0.010, -0.020),
                             _lerp(hp, kn, 0.50) + lat * 0.046 + _v(0, -0.020, 0),
                             L["patella"] + lat * 0.016 + _v(0, 0, 0.020)], 0.034, 0.006, 0.7)
    add("vastus medialis", [_lerp(hp, kn, 0.25) + med * 0.018 + _v(0, -0.018, 0),
                            _lerp(hp, kn, 0.72) + med * 0.040 + _v(0, -0.030, 0),
                            L["patella"] + med * 0.016 + _v(0, 0, 0.012)], 0.030, 0.006, 0.75, power=0.5)
    add("adductor longus", [L["pubic_tubercle"], _lerp(hp, kn, 0.55) + med * 0.006 + _v(0, 0.004, 0)],
        0.024, 0.006, 0.55)
    add("adductor magnus", [L["ischial_tuberosity"] + med * 0.006, _lerp(hp, kn, 0.50) + med * 0.026 + _v(0, 0.020, 0),
                            L["medial_femoral_condyle"] + _v(0, 0, 0.040)], 0.034, 0.006, 0.6)
    add("biceps femoris", [L["ischial_tuberosity"], _lerp(hp, kn, 0.50) + _v(0, 0.050, 0) + lat * 0.022,
                           L["fibular_head"]], 0.027, 0.005, 0.8)
    add("semitendinosus", [L["ischial_tuberosity"] + lat * 0.004, _lerp(hp, kn, 0.50) + _v(0, 0.052, 0) + med * 0.012,
                           L["pes_anserinus"]], 0.020, 0.003, 0.8)
    add("semimembranosus", [L["ischial_tuberosity"] + _v(0, -0.006, -0.010),
                            _lerp(hp, kn, 0.62) + _v(0, 0.040, 0) + med * 0.022,
                            L["medial_femoral_condyle"] + _v(0, 0.004, -0.040)], 0.024, 0.005, 0.8)

    # -- leg ---------------------------------------------------------------
    add("gastrocnemius (medial head)", [L["medial_femoral_condyle"] + _v(0, 0.018, 0.010),
                                        _lerp(kn, an, 0.28) + _v(0, 0.052, 0) + med * 0.016,
                                        _lerp(kn, an, 0.55) + _v(0, 0.046, 0)], 0.026, 0.006, 0.7)
    add("gastrocnemius (lateral head)", [L["lateral_femoral_condyle"] + _v(0, 0.018, 0.010),
                                         _lerp(kn, an, 0.28) + _v(0, 0.050, 0) + lat * 0.014,
                                         _lerp(kn, an, 0.52) + _v(0, 0.044, 0)], 0.022, 0.006, 0.7)
    add("soleus", [L["fibular_head"] + _v(0, 0.012, -0.020), _lerp(kn, an, 0.50) + _v(0, 0.036, 0),
                   _lerp(kn, an, 0.82) + _v(0, 0.034, 0)], 0.030, 0.008, 0.55)
    add("tibialis anterior", [_lerp(kn, an, 0.10) + lat * 0.014 + _v(0, -0.026, 0),
                              _lerp(kn, an, 0.50) + lat * 0.014 + _v(0, -0.024, 0),
                              an + med * 0.008 + _v(0, -0.030, 0.010), an + med * 0.018 + _v(0, -0.060, -0.050)],
        0.015, 0.003, 0.7)
    add("fibularis longus", [L["fibular_head"] + lat * 0.006, _lerp(kn, an, 0.50) + lat * 0.034,
                             L["lateral_malleolus"] + _v(0, 0.010, -0.014)], 0.012, 0.003, 0.7)
    add("extensor digitorum longus", [_lerp(kn, an, 0.12) + lat * 0.030 + _v(0, -0.018, 0),
                                      _lerp(kn, an, 0.55) + lat * 0.024 + _v(0, -0.022, 0),
                                      an + _v(0, -0.034, 0.006), an + _v(0, -0.090, -0.050)],
        0.011, 0.003, 0.7)
    return M


def build_muscles() -> List[Named]:
    out: List[Named] = []
    for side, s in SIDES:
        for m in muscle_specs(s):
            path = m["path"]
            hint = None
            if m["normal"] is not None:
                axis = path[-1] - path[0]
                hint = np.cross(np.asarray(m["normal"], float), axis)
                if np.linalg.norm(hint) < 1e-9:
                    hint = None
            pts = g.smooth_path(path, samples=40) if len(path) > 2 else path
            poly = g.lathe(pts, g.fusiform(m["r"], m["r_end"], m["power"]), ratio=m["ratio"],
                           hint=hint, rings=40, segments=26)
            out.append((f"{side} {m['name']}", poly))
    return out


def build_tendons() -> List[Named]:
    out: List[Named] = []
    for side, s in SIDES:
        L = landmarks(s)
        kn, an = L["knee"], L["ankle"]
        lat = _v(s, 0, 0)
        out.append((f"{side} calcaneal tendon (Achilles)",
                    g.lathe([_lerp(kn, an, 0.55) + _v(0, 0.044, 0), an + _v(0, 0.044, 0.010),
                             L["calcaneus"] + _v(0, 0.010, 0.012)], g.taper(0.009, 0.006), ratio=0.45,
                            hint=(1, 0, 0), rings=24, segments=14)))
        out.append((f"{side} patellar ligament", g.lathe([L["patella"] - _v(0, 0, 0.018), L["tibial_tuberosity"]],
                                                         g.taper(0.010, 0.008), ratio=0.35, hint=(1, 0, 0),
                                                         rings=14, segments=14)))
        out.append((f"{side} quadriceps tendon", g.lathe([L["patella"] + _v(0, 0.004, 0.060),
                                                          L["patella"] + _v(0, 0, 0.018)],
                                                         g.taper(0.014, 0.012), ratio=0.4, hint=(1, 0, 0),
                                                         rings=10, segments=14)))
        out.append((f"{side} iliotibial tract", g.lathe(
            [L["greater_trochanter"] + lat * 0.016 + _v(0, -0.010, -0.040),
             _lerp(L["hip"], kn, 0.6) + lat * 0.060, kn + lat * 0.040 + _v(0, -0.010, -0.020)],
            g.taper(0.012, 0.008), ratio=0.2, hint=(0, -1, 0), rings=24, segments=12)))
        out.append((f"{side} medial collateral ligament of knee",
                    g.tube([kn + _v(-s * 0.040, 0.004, 0.040), kn + _v(-s * 0.040, 0.000, -0.050)], 0.004,
                           segments=8)))
        out.append((f"{side} lateral collateral ligament of knee",
                    g.tube([kn + lat * 0.042 + _v(0, 0.006, 0.030), L["fibular_head"] + lat * 0.006], 0.003,
                           segments=8)))
        out.append((f"{side} anterior cruciate ligament",
                    g.tube([kn + _v(s * 0.008, -0.010, -0.012), kn + _v(s * 0.014, 0.014, 0.016)], 0.004,
                           segments=8)))
        out.append((f"{side} posterior cruciate ligament",
                    g.tube([kn + _v(-s * 0.004, 0.018, -0.014), kn + _v(-s * 0.012, -0.004, 0.016)], 0.0045,
                           segments=8)))
        out.append((f"{side} distal biceps tendon", g.tube([_lerp(L["shoulder"], L["elbow"], 0.85) + _v(0, -0.026, 0),
                                                             L["radial_tuberosity"]], 0.0035, segments=8)))
        out.append((f"{side} plantar aponeurosis", g.lathe([L["calcaneus"] + _v(0, -0.010, -0.010),
                                                            an + _v(0, -0.130, -0.068)],
                                                           g.taper(0.012, 0.024), ratio=0.12, hint=(1, 0, 0),
                                                           rings=16, segments=12)))
    return out


# ---------------------------------------------------------------------------
# Vessels
# ---------------------------------------------------------------------------
HEART_BASE = _v(0.000, -0.025, 0.755 * H)
HEART_APEX = _v(0.072, -0.082, 0.692 * H)


def build_arteries() -> List[Named]:
    out: List[Named] = []
    aorta = [_v(0.004, -0.040, 0.748 * H), _v(0.000, -0.040, 0.782 * H), _v(0.008, -0.020, 0.796 * H),
             _v(0.022, 0.012, 0.790 * H), _v(0.026, 0.034, 0.765 * H), _v(0.022, 0.040, 0.700 * H),
             _v(0.012, 0.016, 0.640 * H), _v(0.006, -0.004, 0.600 * H)]
    out.append(("Thoracic aorta", g.tube(aorta, 0.0125, 0.0105, segments=20)))
    abd = [_v(0.006, -0.004, 0.600 * H), _v(0.006, -0.012, 0.560 * H), _v(0.004, -0.016, 0.590 * H * 0.98)]
    out.append(("Abdominal aorta", g.tube([abd[0], _v(0.006, -0.012, 0.575 * H), _v(0.004, -0.018, 0.555 * H)],
                                          0.0100, 0.0085, segments=18)))
    bif = _v(0.004, -0.018, 0.555 * H)
    out.append(("Brachiocephalic trunk", g.tube([_v(0.002, -0.030, 0.796 * H), _v(-0.016, -0.036, 0.812 * H)],
                                                0.0060, segments=14)))
    out.append(("Coeliac trunk", g.tube([_v(0.006, -0.004, 0.640 * H) + _v(0, -0.008, 0),
                                         _v(0.010, -0.040, 0.640 * H)], 0.0035, segments=10)))
    out.append(("Superior mesenteric artery", g.tube([_v(0.006, -0.008, 0.625 * H), _v(0.010, -0.050, 0.610 * H),
                                                      _v(-0.010, -0.070, 0.560 * H)], 0.0035, 0.002, segments=10)))
    for side, s in SIDES:
        L = landmarks(s)
        lat = _v(s, 0, 0)
        med = -lat
        root = _v(-0.016, -0.036, 0.812 * H) if s < 0 else _v(0.010, -0.010, 0.797 * H)
        carotid_root = root if s < 0 else _v(0.004, -0.022, 0.798 * H)
        out.append((f"{side} common carotid artery",
                    g.tube([carotid_root, _v(s * 0.024, -0.030, 0.840 * H), _v(s * 0.030, -0.022, 0.885 * H)],
                           0.0036, 0.0032, segments=12)))
        out.append((f"{side} internal carotid artery",
                    g.tube([_v(s * 0.030, -0.022, 0.885 * H), _v(s * 0.028, -0.004, 0.905 * H),
                            _v(s * 0.018, 0.000, 0.925 * H)], 0.0026, segments=10)))
        sub = [root, _v(s * 0.060, -0.030, 0.810 * H), _v(s * 0.120, -0.028, 0.796 * H),
               L["shoulder"] + med * 0.020 + _v(0, -0.012, -0.040)]
        out.append((f"{side} subclavian and axillary artery", g.tube(sub, 0.0045, 0.0038, segments=12)))
        sh, el, wr = L["shoulder"], L["elbow"], L["wrist"]
        brachial = [sub[-1], _lerp(sh, el, 0.5) + med * 0.014 + _v(0, -0.014, 0), el + _v(0, -0.020, 0.004)]
        out.append((f"{side} brachial artery", g.tube(brachial, 0.0028, 0.0025, segments=10)))
        out.append((f"{side} radial artery", g.tube([brachial[-1], _lerp(el, wr, 0.5) + lat * 0.012 + _v(0, -0.016, 0),
                                                     wr + lat * 0.010 + _v(0, -0.012, 0)], 0.0017, 0.0015, segments=8)))
        out.append((f"{side} ulnar artery", g.tube([brachial[-1], _lerp(el, wr, 0.5) + med * 0.010 + _v(0, -0.016, 0),
                                                    wr + med * 0.010 + _v(0, -0.012, 0)], 0.0017, 0.0015, segments=8)))
        out.append((f"{side} renal artery", g.tube([_v(s * 0.008, -0.004, 0.622 * H), _v(s * 0.040, 0.020, 0.624 * H)],
                                                   0.0028, segments=10)))
        hp, kn, an = L["hip"], L["knee"], L["ankle"]
        iliac = [bif, _v(s * 0.040, -0.018, 0.540 * H), _v(s * 0.070, -0.050, 0.522 * H)]
        out.append((f"{side} common and external iliac artery", g.tube(iliac, 0.0055, 0.0045, segments=12)))
        fem = [iliac[-1], hp + med * 0.012 + _v(0, -0.040, -0.020), _lerp(hp, kn, 0.50) + med * 0.020 + _v(0, -0.016, 0),
               _lerp(hp, kn, 0.85) + med * 0.010 + _v(0, 0.020, 0), kn + _v(0, 0.030, 0)]
        out.append((f"{side} femoral artery", g.tube(fem, 0.0042, 0.0036, segments=12)))
        out.append((f"{side} popliteal artery", g.tube([fem[-1], kn + _v(0, 0.026, -0.040)], 0.0034, segments=10)))
        out.append((f"{side} anterior tibial artery",
                    g.tube([kn + _v(0, 0.026, -0.040), _lerp(kn, an, 0.2) + lat * 0.012 + _v(0, -0.014, 0),
                            an + _v(0, -0.028, 0.012), an + _v(0, -0.080, -0.040)], 0.0020, 0.0016, segments=8)))
        out.append((f"{side} posterior tibial artery",
                    g.tube([kn + _v(0, 0.026, -0.040), _lerp(kn, an, 0.5) + med * 0.012 + _v(0, 0.020, 0),
                            L["medial_malleolus"] + _v(0, 0.012, -0.006)], 0.0020, 0.0016, segments=8)))
    return out


def build_veins() -> List[Named]:
    out: List[Named] = []
    out.append(("Superior vena cava", g.tube([_v(-0.022, -0.030, 0.752 * H), _v(-0.024, -0.030, 0.800 * H)],
                                             0.0105, segments=16)))
    out.append(("Inferior vena cava", g.tube([_v(-0.020, -0.012, 0.552 * H), _v(-0.022, -0.004, 0.640 * H),
                                              _v(-0.024, -0.020, 0.700 * H), _v(-0.020, -0.034, 0.736 * H)],
                                             0.0110, 0.0115, segments=16)))
    out.append(("Hepatic portal vein", g.tube([_v(0.010, -0.040, 0.615 * H), _v(-0.025, -0.040, 0.640 * H),
                                               _v(-0.045, -0.030, 0.655 * H)], 0.0065, segments=12)))
    for side, s in SIDES:
        L = landmarks(s)
        lat = _v(s, 0, 0)
        med = -lat
        bc = [_v(-0.024, -0.030, 0.800 * H), _v(s * 0.030, -0.040, 0.810 * H)]
        out.append((f"{side} brachiocephalic vein", g.tube(bc, 0.0070, 0.0065, segments=12)))
        out.append((f"{side} internal jugular vein", g.tube([bc[-1], _v(s * 0.040, -0.026, 0.845 * H),
                                                             _v(s * 0.040, -0.004, 0.895 * H)], 0.0060, 0.0050,
                                                            segments=12)))
        sh, el, wr = L["shoulder"], L["elbow"], L["wrist"]
        out.append((f"{side} subclavian vein", g.tube([bc[-1], _v(s * 0.110, -0.040, 0.790 * H),
                                                       sh + med * 0.020 + _v(0, -0.020, -0.045)], 0.0050, segments=10)))
        out.append((f"{side} cephalic vein", g.tube([_v(s * 0.120, -0.050, 0.795 * H), sh + lat * 0.020 + _v(0, -0.040, -0.040),
                                                     _lerp(sh, el, 0.6) + lat * 0.024 + _v(0, -0.028, 0),
                                                     _lerp(el, wr, 0.5) + lat * 0.022 + _v(0, -0.014, 0),
                                                     wr + lat * 0.016], 0.0022, 0.0018, segments=8)))
        out.append((f"{side} basilic vein", g.tube([sh + med * 0.022 + _v(0, -0.010, -0.060),
                                                    _lerp(sh, el, 0.6) + med * 0.024 + _v(0, -0.010, 0),
                                                    _lerp(el, wr, 0.5) + med * 0.020 + _v(0, -0.010, 0),
                                                    wr + med * 0.016], 0.0022, 0.0018, segments=8)))
        out.append((f"{side} renal vein", g.tube([_v(-0.020, -0.006, 0.620 * H), _v(s * 0.042, 0.016, 0.622 * H)],
                                                 0.0040, segments=10)))
        hp, kn, an = L["hip"], L["knee"], L["ankle"]
        iliac = [_v(-0.020, -0.012, 0.552 * H), _v(s * 0.040, -0.010, 0.536 * H), _v(s * 0.062, -0.050, 0.520 * H)]
        out.append((f"{side} common and external iliac vein", g.tube(iliac, 0.0065, 0.0055, segments=12)))
        fem = [iliac[-1], hp + med * 0.022 + _v(0, -0.036, -0.022), _lerp(hp, kn, 0.5) + med * 0.028 + _v(0, -0.012, 0),
               _lerp(hp, kn, 0.85) + med * 0.014 + _v(0, 0.026, 0), kn + _v(-s * 0.006, 0.036, -0.030)]
        out.append((f"{side} femoral vein", g.tube(fem, 0.0052, 0.0044, segments=12)))
        out.append((f"{side} great saphenous vein",
                    g.tube([hp + med * 0.020 + _v(0, -0.050, -0.040), _lerp(hp, kn, 0.5) + med * 0.060 + _v(0, -0.020, 0),
                            kn + med * 0.050 + _v(0, 0.010, 0), _lerp(kn, an, 0.5) + med * 0.034 + _v(0, -0.010, 0),
                            L["medial_malleolus"] + _v(0, -0.018, 0.010)], 0.0024, 0.0018, segments=8)))
    return out


# ---------------------------------------------------------------------------
# Nerves & lymphatics
# ---------------------------------------------------------------------------
def build_nerves() -> List[Named]:
    out: List[Named] = []
    levels = vertebra_levels()
    lv = dict(levels)
    cord = [lv["C1"] + _v(0, 0.022, 0.018)] + [c + _v(0, 0.022, 0) for n, c in levels
                                               if n[0] in "CT"] + [lv["L1"] + _v(0, 0.020, 0)]
    out.append(("Spinal cord", g.tube(cord, 0.0055, 0.0040, segments=12)))
    cauda = [lv["L1"] + _v(0, 0.020, 0), lv["L5"] + _v(0, 0.018, 0), _v(0, 0.070, 0.540 * H)]
    out.append(("Cauda equina", g.tube(cauda, 0.0040, 0.0030, segments=10)))
    for side, s in SIDES:
        L = landmarks(s)
        lat = _v(s, 0, 0)
        med = -lat
        sh, el, wr = L["shoulder"], L["elbow"], L["wrist"]
        plexus_root = lv["C6"] + _v(s * 0.020, 0.010, 0)
        axilla = sh + med * 0.030 + _v(0, -0.004, -0.040)
        out.append((f"{side} brachial plexus", g.tube([plexus_root, _v(s * 0.070, -0.012, 0.810 * H), axilla],
                                                      0.0040, 0.0035, segments=10)))
        out.append((f"{side} median nerve", g.tube([axilla, _lerp(sh, el, 0.5) + med * 0.012 + _v(0, -0.018, 0),
                                                    el + _v(0, -0.022, 0.006), _lerp(el, wr, 0.5) + _v(0, -0.014, 0),
                                                    wr + _v(0, -0.012, -0.020)], 0.0022, 0.0018, segments=8)))
        out.append((f"{side} ulnar nerve", g.tube([axilla, _lerp(sh, el, 0.6) + med * 0.020 + _v(0, 0.006, 0),
                                                   L["medial_epicondyle"] + _v(0, 0.012, 0),
                                                   _lerp(el, wr, 0.5) + med * 0.014, wr + med * 0.012],
                                                  0.0020, 0.0016, segments=8)))
        out.append((f"{side} radial nerve", g.tube([axilla, _lerp(sh, el, 0.35) + _v(0, 0.020, 0),
                                                    _lerp(sh, el, 0.65) + lat * 0.020 + _v(0, 0.006, 0),
                                                    L["lateral_epicondyle"] + _v(0, -0.010, 0.010),
                                                    _lerp(el, wr, 0.6) + lat * 0.014], 0.0021, 0.0016, segments=8)))
        out.append((f"{side} vagus nerve", g.tube([_v(s * 0.034, -0.010, 0.900 * H), _v(s * 0.034, -0.024, 0.840 * H),
                                                   _v(s * 0.020, 0.010, 0.760 * H), _v(s * 0.010, 0.020, 0.680 * H)],
                                                  0.0016, segments=8)))
        hp, kn, an = L["hip"], L["knee"], L["ankle"]
        out.append((f"{side} femoral nerve", g.tube([lv["L3"] + _v(s * 0.030, 0.0, 0),
                                                     hp + _v(0, -0.044, -0.010),
                                                     _lerp(hp, kn, 0.3) + _v(0, -0.040, 0)], 0.0030, 0.0020, segments=8)))
        sciatic = [_v(s * 0.030, 0.090, 0.540 * H), L["ischial_tuberosity"] + lat * 0.030 + _v(0, 0.010, 0.010),
                   _lerp(hp, kn, 0.5) + _v(0, 0.040, 0), kn + _v(0, 0.036, 0.060)]
        out.append((f"{side} sciatic nerve", g.tube(sciatic, 0.0055, 0.0045, segments=10)))
        out.append((f"{side} tibial nerve", g.tube([sciatic[-1], kn + _v(0, 0.032, -0.030),
                                                    _lerp(kn, an, 0.5) + _v(0, 0.026, 0) + med * 0.006,
                                                    L["medial_malleolus"] + _v(0, 0.016, -0.004)], 0.0030, 0.0022,
                                                   segments=8)))
        out.append((f"{side} common fibular nerve", g.tube([sciatic[-1], kn + lat * 0.034 + _v(0, 0.022, -0.020),
                                                            L["fibular_head"] + lat * 0.010 + _v(0, 0.0, -0.016)],
                                                           0.0024, segments=8)))
    return out


def build_lymphatics() -> List[Named]:
    out: List[Named] = []
    duct = [_v(0.000, 0.012, 0.630 * H), _v(0.004, 0.024, 0.700 * H), _v(0.008, 0.016, 0.770 * H),
            _v(0.030, -0.010, 0.815 * H), _v(0.034, -0.032, 0.808 * H)]
    out.append(("Thoracic duct", g.tube(duct, 0.0022, segments=8)))
    out.append(("Cisterna chyli", g.ellipsoid(_v(0.000, 0.010, 0.625 * H), (0.012, 0.005, 0.004), rings=8, segments=10)))
    out.append(("Spleen", g.ellipsoid(_v(0.100, 0.040, 0.655 * H), (0.055, 0.030, 0.018), axis=(0.3, 0.6, 0.75),
                                      rings=18, segments=20)))
    out.append(("Thymus", g.ellipsoid(_v(0.000, -0.085, 0.780 * H), (0.030, 0.018, 0.008), rings=12, segments=14)))
    for side, s in SIDES:
        L = landmarks(s)
        med = _v(-s, 0, 0)
        groups = {
            "deep cervical lymph nodes": [_v(s * 0.046, -0.020, z * H) for z in (0.880, 0.862, 0.845, 0.828)],
            "axillary lymph nodes": [L["shoulder"] + med * 0.035 + _v(dy, -0.006, dz)
                                     for dy, dz in ((0, -0.050), (0.012, -0.060), (-0.010, -0.066), (0.004, -0.076))],
            "inguinal lymph nodes": [L["hip"] + med * 0.020 + _v(dx, -0.050, dz)
                                     for dx, dz in ((0, -0.030), (s * 0.016, -0.036), (-s * 0.012, -0.044))],
            "para-aortic lymph nodes": [_v(s * 0.022, -0.006, z * H) for z in (0.585, 0.600, 0.615)],
        }
        for label, centres in groups.items():
            nodes = [g.ellipsoid(c, (0.006, 0.0045, 0.0035), rings=8, segments=10) for c in centres]
            out.append((f"{side} {label}", g.append(nodes)))
    return out


# ---------------------------------------------------------------------------
# Viscera
# ---------------------------------------------------------------------------
def build_brain() -> List[Named]:
    out: List[Named] = []
    for side, s in SIDES:
        out.append((f"{side} cerebral hemisphere",
                    g.ellipsoid(_v(s * 0.034, 0.004, 0.954 * H), (0.080, 0.034, 0.058), axis=(0, 1, 0),
                                hint=(1, 0, 0), rings=36, segments=32)))
    out.append(("Cerebellum", g.ellipsoid(_v(0, 0.058, 0.915 * H), (0.026, 0.050, 0.030), axis=(0, 0, 1),
                                          rings=20, segments=24)))
    out.append(("Brainstem", g.lathe([_v(0, 0.012, 0.950 * H), _v(0, 0.026, 0.905 * H), _v(0, 0.030, 0.892 * H)],
                                     g.taper(0.014, 0.009), rings=18, segments=18)))
    out.append(("Corpus callosum", g.lathe([_v(0, -0.034, 0.958 * H), _v(0, 0.000, 0.966 * H),
                                            _v(0, 0.034, 0.958 * H)], g.taper(0.006, 0.007), ratio=0.6,
                                           rings=18, segments=12)))
    return out


def build_lungs() -> List[Named]:
    out: List[Named] = []
    trachea = [_v(0, -0.040, 0.868 * H), _v(0, -0.030, 0.820 * H), _v(0, -0.018, 0.780 * H)]
    out.append(("Larynx", g.lathe([_v(0, -0.046, 0.885 * H), _v(0, -0.044, 0.866 * H)], g.taper(0.018, 0.012),
                                  rings=12, segments=16)))
    out.append(("Trachea", g.tube(trachea, 0.0085, segments=14)))
    for side, s in SIDES:
        bronchus = [trachea[-1], _v(s * 0.030, -0.010, 0.770 * H), _v(s * 0.055, 0.000, 0.760 * H)]
        out.append((f"{side} main bronchus", g.tube(bronchus, 0.0065, 0.0050, segments=12)))
        x = s * (0.078 if s < 0 else 0.085)
        notch = 0.0 if s < 0 else 0.010
        apex, base = _v(x - s * 0.020, 0.005, 0.832 * H), _v(x, 0.006, 0.690 * H)
        profile = lambda r_scale: g.blend((0, 0.010), (0.12, 0.034 * r_scale), (0.55, 0.062 * r_scale),
                                          (0.9, 0.068 * r_scale), (1.0, 0.034 * r_scale))   # noqa: E731
        lobes = ([("superior lobe", 0.00, 0.45), ("middle lobe", 0.45, 0.62), ("inferior lobe", 0.62, 1.0)]
                 if s < 0 else [("superior lobe", 0.00, 0.55), ("inferior lobe", 0.55, 1.0)])
        full = profile(1.0 if s < 0 else 0.92)
        for lobe, t0, t1 in lobes:
            a, b = _lerp(apex, base, t0), _lerp(apex, base, t1)
            prof = lambda t, t0=t0, t1=t1: full(t0 + (t1 - t0) * t) * (  # noqa: E731
                math.sqrt(max(0.05, 1 - (2 * t - 1) ** 8)))
            poly = g.lathe([a + _v(-s * notch, 0, 0), b + _v(-s * notch * 0.5, 0, 0)], prof, ratio=1.45,
                           hint=(1, 0, 0), rings=22, segments=30)
            out.append((f"{lobe.capitalize()} of {side.lower()} lung", poly))
    out.append(("Diaphragm", g.lathe([_v(-0.150, 0.010, 0.680 * H), _v(0, 0.000, 0.708 * H), _v(0.150, 0.010, 0.682 * H)],
                                     g.blend((0, 0.050), (0.5, 0.110), (1, 0.050)), ratio=0.06,
                                     hint=(0, 1, 0), rings=30, segments=26)))
    return out


def build_heart() -> List[Named]:
    out: List[Named] = []
    out.append(("Left ventricle", g.lathe([_v(0.014, -0.026, 0.748 * H), HEART_APEX],
                                          g.blend((0, 0.034), (0.3, 0.040), (0.75, 0.028), (1, 0.004)),
                                          ratio=0.92, rings=28, segments=28)))
    out.append(("Right ventricle", g.lathe([_v(-0.018, -0.054, 0.748 * H), HEART_APEX + _v(-0.022, -0.004, 0.012)],
                                           g.blend((0, 0.026), (0.3, 0.032), (0.8, 0.018), (1, 0.003)),
                                           ratio=0.62, hint=(0.5, -1.0, 0), rings=26, segments=26)))
    out.append(("Right atrium", g.ellipsoid(_v(-0.034, -0.030, 0.752 * H), (0.024, 0.022, 0.020),
                                            rings=16, segments=18)))
    out.append(("Left atrium", g.ellipsoid(_v(0.016, 0.010, 0.760 * H), (0.018, 0.024, 0.016),
                                           rings=16, segments=18)))
    out.append(("Pulmonary trunk", g.tube([_v(-0.004, -0.068, 0.752 * H), _v(0.010, -0.058, 0.775 * H),
                                           _v(0.018, -0.030, 0.784 * H)], 0.0110, 0.0100, segments=16)))
    out.append(("Ascending aorta (aortic root)", g.tube([_v(0.004, -0.034, 0.744 * H), _v(0.002, -0.040, 0.760 * H)],
                                                        0.0130, segments=16)))
    surface = lambda t, a, b: _lerp(a, b, t)   # noqa: E731
    out.append(("Left anterior descending artery", g.tube(
        [_v(0.012, -0.052, 0.752 * H), _v(0.030, -0.074, 0.735 * H), _v(0.052, -0.088, 0.712 * H),
         HEART_APEX + _v(-0.006, -0.010, 0.004)], 0.0019, 0.0012, segments=8)))
    out.append(("Left circumflex artery", g.tube(
        [_v(0.012, -0.040, 0.752 * H), _v(0.042, -0.026, 0.746 * H), _v(0.056, -0.004, 0.728 * H)],
        0.0017, 0.0012, segments=8)))
    out.append(("Right coronary artery", g.tube(
        [_v(-0.006, -0.050, 0.748 * H), _v(-0.040, -0.058, 0.735 * H), _v(-0.050, -0.040, 0.712 * H),
         _v(-0.020, -0.010, 0.700 * H)], 0.0019, 0.0013, segments=8)))
    del surface
    return out


def build_liver() -> List[Named]:
    out: List[Named] = []
    right = g.lathe([_v(-0.150, 0.000, 0.650 * H), _v(-0.070, -0.020, 0.660 * H), _v(-0.015, -0.040, 0.668 * H)],
                    g.blend((0, 0.030), (0.3, 0.072), (0.8, 0.062), (1, 0.040)), ratio=0.72,
                    hint=(0, 0, 1), rings=30, segments=30)
    left = g.lathe([_v(-0.015, -0.040, 0.668 * H), _v(0.040, -0.060, 0.676 * H), _v(0.085, -0.050, 0.680 * H)],
                   g.blend((0, 0.040), (0.4, 0.030), (1, 0.006)), ratio=0.45,
                   hint=(0, 0, 1), rings=22, segments=26)
    out.append(("Right lobe of liver", right))
    out.append(("Left lobe of liver", left))
    out.append(("Gallbladder", g.ellipsoid(_v(-0.060, -0.072, 0.625 * H), (0.032, 0.012, 0.012),
                                           axis=(0.2, -0.5, -0.8), rings=14, segments=14)))
    out.append(("Common bile duct", g.tube([_v(-0.050, -0.060, 0.630 * H), _v(-0.030, -0.040, 0.615 * H),
                                            _v(-0.025, -0.030, 0.600 * H)], 0.0025, segments=8)))
    return out


def build_kidneys() -> List[Named]:
    out: List[Named] = []
    for side, s in SIDES:
        zc = (0.618 if s < 0 else 0.628) * H          # right kidney sits lower (liver)
        c = _v(s * 0.052, 0.040, zc)
        axis = (s * -0.20, 0.15, 1.0)
        kidney = g.lathe([c - np.asarray(axis) / np.linalg.norm(axis) * 0.052,
                          c + np.asarray(axis) / np.linalg.norm(axis) * 0.052],
                         g.blend((0, 0.008), (0.15, 0.026), (0.5, 0.030), (0.85, 0.026), (1, 0.008)),
                         ratio=0.62, hint=(s, -0.4, 0), rings=24, segments=24)
        out.append((f"{side} kidney", kidney))
        out.append((f"{side} adrenal gland", g.ellipsoid(c + _v(-s * 0.008, 0.002, 0.060), (0.010, 0.014, 0.004),
                                                          axis=(0, 0, 1), rings=10, segments=12)))
        out.append((f"{side} ureter", g.tube([c + _v(-s * 0.018, -0.006, -0.010), c + _v(-s * 0.016, 0.000, -0.080),
                                              _v(s * 0.040, 0.000, 0.540 * H), _v(s * 0.018, -0.040, 0.520 * H)],
                                             0.0022, segments=8)))
    out.append(("Urinary bladder", g.ellipsoid(_v(0, -0.050, 0.515 * H), (0.026, 0.032, 0.028),
                                               rings=18, segments=20)))
    return out


def build_digestive() -> List[Named]:
    out: List[Named] = []
    out.append(("Oesophagus", g.tube([_v(0, -0.026, 0.858 * H), _v(0.002, 0.004, 0.800 * H),
                                      _v(0.008, 0.012, 0.720 * H), _v(0.030, -0.010, 0.672 * H)],
                                     0.0075, segments=12)))
    stomach = [_v(0.030, -0.010, 0.672 * H), _v(0.080, -0.010, 0.665 * H), _v(0.100, -0.040, 0.640 * H),
               _v(0.075, -0.075, 0.612 * H), _v(0.025, -0.075, 0.610 * H), _v(-0.012, -0.060, 0.620 * H)]
    out.append(("Stomach", g.lathe(g.smooth_path(stomach, 40), g.blend((0, 0.012), (0.15, 0.040), (0.45, 0.042),
                                                                       (0.85, 0.020), (1, 0.010)),
                                   rings=40, segments=24)))
    duo = [_v(-0.012, -0.060, 0.620 * H), _v(-0.040, -0.040, 0.615 * H), _v(-0.044, -0.030, 0.585 * H),
           _v(-0.010, -0.030, 0.575 * H), _v(0.020, -0.040, 0.582 * H)]
    out.append(("Duodenum", g.tube(duo, 0.0120, segments=12)))
    out.append(("Pancreas", g.lathe([_v(-0.030, -0.030, 0.600 * H), _v(0.030, -0.010, 0.612 * H),
                                     _v(0.085, 0.020, 0.630 * H)], g.taper(0.016, 0.009), ratio=0.6,
                                    hint=(0, 0, 1), rings=24, segments=18)))
    rng = np.random.default_rng(7)
    coil = []
    for k in range(70):
        t = k / 69
        row = int(t * 6)
        x = (-0.065 + 0.130 * ((t * 6) % 1.0)) if row % 2 == 0 else (0.065 - 0.130 * ((t * 6) % 1.0))
        coil.append(_v(x, -0.060 + 0.012 * math.sin(k * 0.9) + rng.normal(0, 0.003),
                       0.590 * H - row * 0.0125 * H * 0.9))
    out.append(("Small intestine (jejunum and ileum)", g.tube(coil, 0.0110, segments=12)))
    colon = [_v(-0.090, -0.050, 0.540 * H), _v(-0.095, -0.050, 0.585 * H), _v(-0.085, -0.060, 0.620 * H),
             _v(0.000, -0.080, 0.612 * H), _v(0.090, -0.050, 0.632 * H), _v(0.098, -0.030, 0.590 * H),
             _v(0.090, -0.030, 0.545 * H), _v(0.040, -0.040, 0.530 * H), _v(0.000, 0.040, 0.528 * H)]
    out.append(("Colon", g.tube(colon, 0.0200, 0.0160, segments=16)))
    out.append(("Caecum and appendix", g.append([g.ellipsoid(_v(-0.090, -0.050, 0.532 * H), (0.022, 0.022, 0.020),
                                                             rings=12, segments=14),
                                                 g.tube([_v(-0.085, -0.050, 0.522 * H), _v(-0.075, -0.040, 0.505 * H)],
                                                        0.0035, segments=8)])))
    out.append(("Rectum and anal canal", g.tube([_v(0.000, 0.040, 0.528 * H), _v(0.000, 0.060, 0.505 * H),
                                                 _v(0.000, 0.050, 0.488 * H)], 0.0160, 0.0080, segments=14)))
    return out


# ---------------------------------------------------------------------------
# Body surface
# ---------------------------------------------------------------------------
def _envelope(scale: float, tag: str) -> List[Named]:
    out: List[Named] = []
    k = scale
    out.append((f"{tag} of head", g.ellipsoid(_v(0, -0.004, 0.940 * H), (0.118 * k, 0.081 * k, 0.103 * k),
                                              rings=30, segments=36)))
    out.append((f"{tag} of neck", g.lathe([_v(0, 0.012, 0.815 * H), _v(0, 0.000, 0.890 * H)],
                                          g.taper(0.060 * k, 0.050 * k, round_ends=False), ratio=0.9,
                                          hint=(1, 0, 0), rings=12, segments=30)))
    torso = g.lathe([_v(0, 0.000, 0.488 * H), _v(0, -0.004, 0.640 * H), _v(0, 0.000, 0.835 * H)],
                    g.blend((0, 0.115 * k), (0.08, 0.172 * k), (0.25, 0.160 * k), (0.45, 0.142 * k),
                            (0.72, 0.168 * k), (0.90, 0.188 * k), (1.0, 0.090 * k)),
                    ratio=0.66, hint=(1, 0, 0), rings=50, segments=44)
    out.append((f"{tag} of trunk", torso))
    for side, s in SIDES:
        L = landmarks(s)
        sh, el, wr, hp, kn, an = (L[x] for x in ("shoulder", "elbow", "wrist", "hip", "knee", "ankle"))
        arm = g.lathe([sh + _v(s * 0.010, 0, 0.038), el, wr],
                      g.blend((0, 0.050 * k), (0.12, 0.054 * k), (0.42, 0.044 * k), (0.52, 0.040 * k),
                              (0.66, 0.042 * k), (1.0, 0.027 * k)), ratio=0.92, rings=40, segments=28)
        out.append((f"{tag} of {side.lower()} upper limb", arm))
        hand = g.lathe([wr + _v(0, 0, 0.004), wr - _v(0, 0, 0.190)],
                       g.blend((0, 0.026 * k), (0.4, 0.040 * k), (0.6, 0.038 * k), (1, 0.010 * k)),
                       ratio=0.36, hint=(s, 0, 0), rings=24, segments=22)
        out.append((f"{tag} of {side.lower()} hand", hand))
        leg = g.lathe([hp + _v(s * 0.004, 0.010, 0.060), kn, an + _v(0, 0, -0.010)],
                      g.blend((0, 0.088 * k), (0.18, 0.084 * k), (0.46, 0.058 * k), (0.50, 0.054 * k),
                              (0.60, 0.058 * k), (0.78, 0.046 * k), (0.96, 0.032 * k), (1.0, 0.030 * k)),
                      ratio=0.95, rings=48, segments=30)
        out.append((f"{tag} of {side.lower()} lower limb", leg))
        foot = g.lathe([L["calcaneus"] + _v(0, 0.010, 0.020), an + _v(0, -0.180, -0.046)],
                       g.blend((0, 0.026 * k), (0.25, 0.034 * k), (0.7, 0.044 * k), (1, 0.016 * k)),
                       ratio=0.55, hint=(1, 0, 0), rings=26, segments=22)
        out.append((f"{tag} of {side.lower()} foot", foot))
    return out


def build_skin() -> List[Named]:
    return _envelope(1.0, "Skin")


def build_fascia() -> List[Named]:
    return _envelope(0.955, "Superficial fascia")


BUILDERS = {
    "skin": build_skin,
    "fascia": build_fascia,
    "muscles": build_muscles,
    "tendons": build_tendons,
    "skeleton": build_skeleton,
    "cartilage": build_cartilage,
    "arteries": build_arteries,
    "veins": build_veins,
    "nerves": build_nerves,
    "lymphatics": build_lymphatics,
    "brain": build_brain,
    "lungs": build_lungs,
    "heart": build_heart,
    "liver": build_liver,
    "kidneys": build_kidneys,
    "digestive": build_digestive,
}


def build_layer(layer_id: str) -> List[Named]:
    builder = BUILDERS.get(layer_id)
    return builder() if builder else []
