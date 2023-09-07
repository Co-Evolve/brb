from pathlib import Path

import numpy as np

from brb.seahorse.morphology.specification.specification import JointSpecification, MeshSpecification, \
    SeahorseMorphologySpecification, SeahorsePlateSpecification, SeahorseSegmentSpecification, \
    SeahorseTendonActuationSpecification, SeahorseTendonSpineSpecification, SeahorseVertebraeSpecification

BASE_MESH_PATH = Path(__file__).parent.parent / "assets"
PLATE_INDEX_TO_SIDE = ["ventral_sinistral", "ventral_dextral", "dorsal_dextral", "dorsal_sinistral"]
MESH_NAME_TO_MASS = {
        # in kilograms
        "ventral_dextral.stl": 0.00898, "ventral_sinistral.stl": 0.00901, "dorsal_dextral_even.stl": 0.00821,
        "dorsal_sinistral_even.stl": 0.00883, "dorsal_dextral_odd.stl": 0.00895, "dorsal_sinistral_odd.stl": 0.00926,
        "ball_bearing.stl": 0.00214, "vertebrae.stl": 0.00453, "connector_plate.stl": 0.00014,
        "connector_vertebrae.stl": 0.00007}
PLATE_INDEX_TO_VERTEBRAE_OFFSET = 0.001 * np.array([65.76, 65.03, 64.91, 64.83])
PLATE_INDEX_TO_X_GLIDING_JOINT_DOF = 0.001 * 0.5 * np.array([10.23, 10.23, 10.23, 10.23])
PLATE_INDEX_TO_Y_GLIDING_JOINT_DOF = 0.001 * 0.5 * np.array([3.3 / 2, 3.3 / 2, 11.0, 11.0])
PLATE_HALF_HEIGHT = 0.0075
PLATE_VERTEBRAE_CONNECTOR_OFFSET = 0.036
VERTEBRAE_HALF_HEIGHT = 0.015


def default_mesh_specification(
        *,
        mesh_name: str
        ) -> MeshSpecification:
    return MeshSpecification(
            mesh_path=str(BASE_MESH_PATH / mesh_name), scale_ratio=0.001 * np.ones(3), mass=MESH_NAME_TO_MASS[mesh_name]
            )


def default_gliding_joint_specification(
        *,
        plate_index: int,
        axis: str
        ) -> JointSpecification:
    if axis == 'x':
        dof = PLATE_INDEX_TO_X_GLIDING_JOINT_DOF[plate_index]
    else:
        dof = PLATE_INDEX_TO_Y_GLIDING_JOINT_DOF[plate_index]

    return JointSpecification(
            stiffness=1, damping=0.1, friction_loss=10, dof=dof
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
    connector_mesh_name = "connector_plate.stl"

    plate_mesh_specification = default_mesh_specification(mesh_name=plate_mesh_name)
    connector_mesh_specification = default_mesh_specification(mesh_name=connector_mesh_name)

    x_axis_gliding_joint_specification = default_gliding_joint_specification(plate_index=plate_index, axis='x')
    y_axis_gliding_joint_specification = default_gliding_joint_specification(plate_index=plate_index, axis='y')
    plate_specification = SeahorsePlateSpecification(
            plate_mesh_specification=plate_mesh_specification,
            connector_mesh_specification=connector_mesh_specification,
            offset_from_vertebrae=PLATE_INDEX_TO_VERTEBRAE_OFFSET[plate_index],
            x_axis_gliding_joint_specification=x_axis_gliding_joint_specification,
            y_axis_gliding_joint_specification=y_axis_gliding_joint_specification,
            half_height=PLATE_HALF_HEIGHT,
            vertebrae_connector_offset=PLATE_VERTEBRAE_CONNECTOR_OFFSET
            )
    return plate_specification


def default_seahorse_vertebrae_specification() -> SeahorseVertebraeSpecification:
    vertebrae_mesh_name = "vertebrae.stl"
    ball_bearing_mesh_name = "ball_bearing.stl"
    connector_mesh_name = "connector_vertebrae.stl"

    bend_joint_specification = JointSpecification(
            stiffness=1.0, damping=0.1, friction_loss=10.0, dof=10 / 180 * np.pi
            )
    twist_joint_specification = JointSpecification(
            stiffness=1.0, damping=0.1, friction_loss=10.0, dof=3 / 180 * np.pi
            )

    vertebrae_specification = SeahorseVertebraeSpecification(
            vertebral_mesh_specification=default_mesh_specification(mesh_name=vertebrae_mesh_name),
            ball_bearing_mesh_specification=default_mesh_specification(mesh_name=ball_bearing_mesh_name),
            connector_mesh_specification=default_mesh_specification(mesh_name=connector_mesh_name),
            bend_joint_specification=bend_joint_specification,
            twist_joint_specification=twist_joint_specification,
            vertebrae_half_height=VERTEBRAE_HALF_HEIGHT
            )
    return vertebrae_specification


def default_seahorse_tendon_spine_specification() -> SeahorseTendonSpineSpecification:
    return SeahorseTendonSpineSpecification(
            stiffness=1.0, damping=0.1, tendon_width=0.002
            )


def default_seahorse_segment_specification(
        *,
        segment_index: int
        ) -> SeahorseSegmentSpecification:
    vertebrae_specification = default_seahorse_vertebrae_specification()
    plate_specifications = [default_seahorse_plate_specification(segment_index=segment_index, plate_index=plate_index)
            for plate_index in range(4)]
    tendon_spine_specification = default_seahorse_tendon_spine_specification()
    segment_specification = SeahorseSegmentSpecification(
            vertebrae_specification=vertebrae_specification,
            plate_specifications=plate_specifications,
            tendon_spine_specification=tendon_spine_specification
            )
    return segment_specification


def default_tendon_actuation_specification() -> SeahorseTendonActuationSpecification:
    return SeahorseTendonActuationSpecification(
            contraction_factor=0.5, relaxation_factor=1.5, p_control_kp=50, tendon_width=0.0005
            )


def default_seahorse_morphology_specification(
        *,
        num_segments: int
        ) -> SeahorseMorphologySpecification:
    segment_specifications = [default_seahorse_segment_specification(segment_index=segment_index) for segment_index in
            range(num_segments)]
    tendon_actuation_specification = default_tendon_actuation_specification()

    morphology_specification = SeahorseMorphologySpecification(
            segment_specifications=segment_specifications, tendon_actuation_specification=tendon_actuation_specification
            )
    return morphology_specification
