"""Structure-level anatomy, atlas normalisation and muscle kinematics."""
from __future__ import annotations

import numpy as np
import pytest

vtk = pytest.importorskip("vtk")

from vtkmodules.util.numpy_support import vtk_to_numpy  # noqa: E402

from app.anatomy import procedural_body as pb  # noqa: E402
from app.anatomy.atlas_frame import apply_matrix, infer_frame  # noqa: E402
from app.anatomy.kinematics import estimate_limbs, rotation  # noqa: E402
from app.anatomy.muscles import MOTIONS, MUSCLES, OPPOSITE, agonists_for, muscle_for_structure  # noqa: E402
from app.anatomy.structures import merge_structures, split_structures, structure_names  # noqa: E402


def _centroid(poly):
    return vtk_to_numpy(poly.GetPoints().GetData()).mean(axis=0)


@pytest.fixture(scope="module")
def skeleton():
    return dict(pb.build_skeleton())


def test_labelled_layer_round_trip():
    items = pb.build_heart()
    merged = merge_structures(items)
    assert structure_names(merged) == [name for name, _ in items]
    split = dict(split_structures(merged))
    assert set(split) == {name for name, _ in items}
    for name, poly in items:
        assert split[name].GetNumberOfCells() == poly.GetNumberOfCells()


def test_reference_body_proportions(skeleton):
    """Joint heights follow Drillis & Contini for a 1.75 m adult (±3 cm)."""
    limbs = estimate_limbs({n.lower(): vtk_to_numpy(p.GetPoints().GetData()).astype(float)
                            for n, p in skeleton.items()})
    upper, lower = limbs["right_upper"], limbs["left_lower"]
    for joint, expected in zip(upper.joints, (0.800, 0.630, 0.485)):
        assert abs(joint[2] - expected * pb.H) < 0.03
    for joint, expected in zip(lower.joints, (0.530, 0.285, 0.039)):
        assert abs(joint[2] - expected * pb.H) < 0.03
    assert upper.joints[0][0] < 0 < limbs["left_upper"].joints[0][0], "patient's right is -X"


def test_flexion_moves_distal_segment_anteriorly(skeleton):
    limbs = estimate_limbs({n.lower(): vtk_to_numpy(p.GetPoints().GetData()).astype(float)
                            for n, p in skeleton.items()})
    for key, dof, probe_index, expected in (("right_upper", "elbow_flex", 2, np.array([0, -1, 0])),
                                            ("left_lower", "hip_flex", 1, np.array([0, -1, 0])),
                                            ("left_lower", "knee_flex", 2, np.array([0, 1, 0]))):
        limb = limbs[key]
        pivot = limb.joints[probe_index - 1]
        probe = limb.joints[probe_index]
        moved = rotation(limb.axes[dof], 30) @ (probe - pivot) + pivot
        assert float((moved - probe) @ expected) > 0, f"{key} {dof} moves the wrong way"


def test_atlas_frame_recovers_millimetre_rotated_dataset(skeleton):
    named = dict(skeleton)
    scramble = np.eye(4)
    scramble[:3, :3] = np.diag([-1.0, -1.0, 1.0]) * 1000.0      # mm, facing +Y
    scramble[:3, 3] = [250.0, -90.0, 40.0]
    scrambled = {n: apply_matrix(p, scramble) for n, p in named.items()}
    report = infer_frame(scrambled)
    restored = {n: apply_matrix(p, report.matrix) for n, p in scrambled.items()}
    assert abs(_centroid(restored["Left humerus"])[0] - _centroid(named["Left humerus"])[0]) < 0.02
    assert _centroid(restored["Body of sternum"])[1] < _centroid(restored["T6 vertebra (thoracic)"])[1]
    assert 1.6 < max(_centroid(p)[2] for p in restored.values()) < 1.8


def test_muscle_knowledge_base_is_consistent():
    for muscle in MUSCLES:
        assert muscle.origin and muscle.insertion and muscle.innervation
        for action in muscle.actions:
            assert action in MOTIONS, f"{muscle.name}: unknown motion {action}"
    for motion_id in MOTIONS:
        assert motion_id in OPPOSITE
    assert muscle_for_structure("Right biceps femoris").key == "biceps femoris"
    assert muscle_for_structure("left biceps brachii").key == "biceps brachii"
    assert {m.key for m in agonists_for("elbow.flexion")} >= {"biceps brachii", "brachialis", "brachioradialis"}
