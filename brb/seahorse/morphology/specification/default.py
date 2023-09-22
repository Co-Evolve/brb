import json
from pathlib import Path

import numpy as np

from brb.seahorse.morphology.specification.specification import JointSpecification, MeshSpecification, \
    SeahorseHMMTendonActuationSpecification, SeahorseMVMTendonActuationSpecification, SeahorseMorphologySpecification, \
    SeahorsePlateSpecification, SeahorseSegmentSpecification, SeahorseTendonActuationSpecification, \
    SeahorseTendonSpineSpecification, SeahorseVertebraeSpecification
from brb.seahorse.morphology.utils import is_inner_plate_x_axis, is_inner_plate_y_axis

PLATE_INDEX_TO_SIDE = ["ventral_dextral", "ventral_sinistral", "dorsal_sinistral", "dorsal_dextral"]

BASE_MESH_PATH = Path(__file__).parent.parent / "assets" / "v1"
PLATE_MESH_NAMES = ["ventral_dextral", "ventral_sinistral", "dorsal_sinistral_even", "dorsal_dextral_even",
                    "dorsal_sinistral_odd", "dorsal_dextral_odd"]
VERTEBRAE_MESH_NAME = "vertebrae.stl"
BALL_BEARING_MESH_NAME = "ball_bearing.stl"
VERTEBRAE_CONNECTOR_MESH_NAME = "connector_vertebrae.stl"
VERTEBRAE_OFFSET_TO_BAR_END = 0.013 + 0.003  # todo: update
VERTEBRAE_CONNECTOR_LENGTH = 0.0036

PLATE_MESH_NAME_TO_OFFSET_FROM_VERTEBRAE = {
        "ventral_dextral.stl": 0.05098, "ventral_sinistral.stl": 0.05119, "dorsal_sinistral_even.stl": 0.0512,
        "dorsal_sinistral_odd.stl": 0.0512, "dorsal_dextral_even.stl": 0.05145, "dorsal_dextral_odd.stl": 0.0512}
PLATE_MESH_NAME_TO_HMM_TAP_OFFSET = {
        "ventral_sinistral.stl": [0.00219, -0.02981], "ventral_dextral.stl": [0.00235, 0.0304],
        "dorsal_sinistral_even.stl": [-0.00224, -0.02986], "dorsal_dextral_even.stl": [-0.00202, 0.03072],
        "dorsal_sinistral_odd.stl": [-0.00219, -0.03061], "dorsal_dextral_odd.stl": [-0.00219, 0.0298]}

OUTER_PLATE_S_TAP_OFFSET_FROM_VERTEBRAE = 0.052
INNER_PLATE_S_TAP_OFFSET_FROM_VERTEBRAE = 0.048

PLATE_CONNECTOR_MESH_NAME = "connector_plate.stl"
PLATE_CONNECTOR_LENGTH = 0.010
VERTEBRAE_TO_PLATE_CONNECTOR_HOLE = 0.04101
VERTEBRAE_S_TAP_OFFSET = 0.015

MESH_NAME_TO_INERTIA_VALUES = json.load(open(f"{BASE_MESH_PATH}/inertia.json", "r"))

PLATE_DEPTH = 0.015
PLATE_GLIDING_JOINT_RANGE = 0.0098
SEGMENT_Z_OFFSET = 0.032
VERTEBRAE_BALL_BEARING_Z_OFFSET = 0.016


def default_mesh_specification(
        *,
        mesh_name: str
        ) -> MeshSpecification:
    inertia_values = MESH_NAME_TO_INERTIA_VALUES[mesh_name]
    indices = [[0, 0], [1, 1], [2, 2], [0, 1], [0, 2], [1, 2]]
    fullinertia = []
    for x, y in indices:
        fullinertia.append(inertia_values["inertia_matrix"][x][y])
    return MeshSpecification(
            mesh_path=str(BASE_MESH_PATH / mesh_name), scale=0.001 * np.ones(3),  # gram to kg
            mass=inertia_values["mass"] / 1000,  # mm to meter
            center_of_mass=np.array(inertia_values["center_of_mass"]) / 1000,  # (g*m^2) to (kg*m^2)
            fullinertia=np.array(fullinertia) / 1e9, )


def default_gliding_joint_specification(
        *,
        plate_index: int,
        axis: str
        ) -> JointSpecification:
    return JointSpecification(
            stiffness=0, damping=1, friction_loss=0.3, armature=0.045, range=PLATE_GLIDING_JOINT_RANGE / 2
            )


def default_seahorse_plate_specification(
        *,
        segment_index: int,
        plate_index: int
        ) -> SeahorsePlateSpecification:

    side = PLATE_INDEX_TO_SIDE[plate_index]

    if side.startswith("ventral"):
        plate_mesh_name = f"{side}.stl"
    else:
        alternator = "even" if segment_index % 2 == 0 else "odd"
        plate_mesh_name = f"{side}_{alternator}.stl"

    plate_mesh_specification = default_mesh_specification(mesh_name=plate_mesh_name)
    connector_mesh_specification = default_mesh_specification(mesh_name=PLATE_CONNECTOR_MESH_NAME)

    x_axis_gliding_joint_specification = default_gliding_joint_specification(plate_index=plate_index, axis='x')
    y_axis_gliding_joint_specification = default_gliding_joint_specification(plate_index=plate_index, axis='y')

    connector_offset_from_vertebrae = (VERTEBRAE_TO_PLATE_CONNECTOR_HOLE - 2 / 5 * PLATE_CONNECTOR_LENGTH)

    if is_inner_plate_x_axis(segment_index=segment_index, plate_index=plate_index):
        s_tap_x_offset = INNER_PLATE_S_TAP_OFFSET_FROM_VERTEBRAE
    else:
        s_tap_x_offset = OUTER_PLATE_S_TAP_OFFSET_FROM_VERTEBRAE
    if is_inner_plate_y_axis(segment_index=segment_index, plate_index=plate_index):
        s_tap_y_offset = INNER_PLATE_S_TAP_OFFSET_FROM_VERTEBRAE
    else:
        s_tap_y_offset = OUTER_PLATE_S_TAP_OFFSET_FROM_VERTEBRAE

    hmm_tap_x_offset, hmm_tap_y_offset = PLATE_MESH_NAME_TO_HMM_TAP_OFFSET[plate_mesh_name]

    plate_specification = SeahorsePlateSpecification(
            plate_mesh_specification=plate_mesh_specification,
            connector_mesh_specification=connector_mesh_specification,
            offset_from_vertebrae=PLATE_MESH_NAME_TO_OFFSET_FROM_VERTEBRAE[plate_mesh_name],
            depth=PLATE_DEPTH,
            s_tap_x_offset_from_vertebrae=s_tap_x_offset,
            s_tap_y_offset_from_vertebrae=s_tap_y_offset,
            hmm_tap_x_offset_from_plate_origin=hmm_tap_x_offset,
            hmm_tap_y_offset_from_plate_origin=hmm_tap_y_offset,
            connector_offset_from_vertebrae=connector_offset_from_vertebrae,
            x_axis_gliding_joint_specification=x_axis_gliding_joint_specification,
            y_axis_gliding_joint_specification=y_axis_gliding_joint_specification
            )
    return plate_specification


def default_seahorse_vertebrae_specification() -> SeahorseVertebraeSpecification:

    bend_joint_specification = JointSpecification(
            stiffness=0.0, damping=1, friction_loss=0.03, armature=0.045, range=15 / 180 * np.pi
            )
    twist_joint_specification = JointSpecification(
            stiffness=0.0, damping=1, friction_loss=0.3, armature=0.045, range=0 / 180 * np.pi  # 5
            )

    vertebrae_specification = SeahorseVertebraeSpecification(
            z_offset_to_ball_bearing=VERTEBRAE_BALL_BEARING_Z_OFFSET,
            offset_to_spine_attachment_point=VERTEBRAE_S_TAP_OFFSET,
            offset_to_bar_end=VERTEBRAE_OFFSET_TO_BAR_END,
            connector_length=VERTEBRAE_CONNECTOR_LENGTH,
            vertebral_mesh_specification=default_mesh_specification(mesh_name=VERTEBRAE_MESH_NAME),
            ball_bearing_mesh_specification=default_mesh_specification(mesh_name=BALL_BEARING_MESH_NAME),
            connector_mesh_specification=default_mesh_specification(mesh_name=VERTEBRAE_CONNECTOR_MESH_NAME),
            bend_joint_specification=bend_joint_specification,
            twist_joint_specification=twist_joint_specification
            )
    return vertebrae_specification


def default_seahorse_tendon_spine_specification() -> SeahorseTendonSpineSpecification:
    return SeahorseTendonSpineSpecification(
            stiffness=0.0, damping=1.0, tendon_width=0.0005
            )


def default_seahorse_segment_specification(
        *,
        segment_index: int
        ) -> SeahorseSegmentSpecification:
    if segment_index == 0:
        z_offset_from_previous_segment = 0
    else:
        z_offset_from_previous_segment = SEGMENT_Z_OFFSET

    vertebrae_specification = default_seahorse_vertebrae_specification()
    plate_specifications = [default_seahorse_plate_specification(segment_index=segment_index, plate_index=plate_index)
                            for plate_index in range(4)]
    tendon_spine_specification = default_seahorse_tendon_spine_specification()
    segment_specification = SeahorseSegmentSpecification(
            z_offset_from_previous_segment=z_offset_from_previous_segment,
            vertebrae_specification=vertebrae_specification,
            plate_specifications=plate_specifications,
            tendon_spine_specification=tendon_spine_specification
            )
    return segment_specification


def default_tendon_actuation_specification(
        hmm_segment_span: int
        ) -> SeahorseTendonActuationSpecification:
    mvm_tendon_actuation_specification = SeahorseMVMTendonActuationSpecification(
            contraction_factor=0.5, relaxation_factor=1.5, tendon_width=0.0005, p_control_kp=100, damping=1
            )
    hmm_tendon_actuation_specification = SeahorseHMMTendonActuationSpecification(
            tendon_strain=0.01, p_control_kp=30, tendon_width=0.0005, segment_span=hmm_segment_span, damping=1
            )

    return SeahorseTendonActuationSpecification(
            hmm_tendon_actuation_specification=hmm_tendon_actuation_specification,
            mvm_tendon_actuation_specification=mvm_tendon_actuation_specification
            )


def default_seahorse_morphology_specification(
        *,
        num_segments: int,
        hmm_segment_span: int
        ) -> SeahorseMorphologySpecification:
    segment_specifications = [default_seahorse_segment_specification(segment_index=segment_index) for segment_index in
                              range(num_segments)]
    tendon_actuation_specification = default_tendon_actuation_specification(hmm_segment_span=hmm_segment_span)

    morphology_specification = SeahorseMorphologySpecification(
            segment_specifications=segment_specifications, tendon_actuation_specification=tendon_actuation_specification
            )
    return morphology_specification
