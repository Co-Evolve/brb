from __future__ import annotations

from typing import List

import numpy as np
from fprs.parameters import FixedParameter
from fprs.specification import MorphologySpecification, Specification


class MeshSpecification(Specification):
    def __init__(
            self,
            *,
            mesh_path: str,
            scale: np.ndarray,
            mass: float,
            center_of_mass: np.ndarray,
            fullinertia: np.ndarray
            ) -> None:
        super().__init__()
        self.mesh_path = FixedParameter(mesh_path)
        self.scale_ratio = FixedParameter(scale)
        self.mass = FixedParameter(mass)
        self.center_of_mass = FixedParameter(center_of_mass)
        self.fullinertia = FixedParameter(fullinertia)


class JointSpecification(Specification):
    def __init__(
            self,
            *,
            stiffness: float,
            damping: float,
            friction_loss: float,
            range: float,
            armature: float
            ) -> None:
        super().__init__()
        self.stiffness = FixedParameter(stiffness)
        self.damping = FixedParameter(damping)
        self.friction_loss = FixedParameter(friction_loss)
        self.range = FixedParameter(range)
        self.armature = FixedParameter(armature)


class SeahorsePlateSpecification(Specification):
    def __init__(
            self,
            *,
            plate_mesh_specification: MeshSpecification,
            connector_mesh_specification: MeshSpecification,
            offset_from_vertebrae: float,
            depth: float,
            connector_offset_from_vertebrae: float,
            s_tap_x_offset_from_vertebrae: float,
            s_tap_y_offset_from_vertebrae: float,
            hmm_ghost_tap_x_offset_from_plate_origin: float,
            hmm_ghost_tap_y_offset_from_plate_origin: float,
            hmm_num_intermediate_taps: int,
            hmm_y_offset_between_intermediate_taps: float,
            hmm_intermediate_first_tap_x_offset_from_plate_origin: float,
            hmm_intermediate_first_tap_y_offset_from_plate_origin: float,
            mvm_tap_x_offset: float,
            mvm_tap_z_offset: float,
            x_axis_gliding_joint_specification: JointSpecification,
            y_axis_gliding_joint_specification: JointSpecification, ) -> None:
        super().__init__()
        self.plate_mesh_specification = plate_mesh_specification
        self.connector_mesh_specification = connector_mesh_specification
        self.offset_from_vertebrae = FixedParameter(offset_from_vertebrae)
        self.hmm_ghost_tap_x_offset_from_plate_origin = FixedParameter(hmm_ghost_tap_x_offset_from_plate_origin)
        self.hmm_ghost_tap_y_offset_from_plate_origin = FixedParameter(hmm_ghost_tap_y_offset_from_plate_origin)
        self.hmm_num_intermediate_taps = FixedParameter(hmm_num_intermediate_taps)
        self.hmm_y_offset_between_intermediate_taps = FixedParameter(hmm_y_offset_between_intermediate_taps)
        self.hmm_intermediate_first_tap_x_offset_from_plate_origin = FixedParameter(
                hmm_intermediate_first_tap_x_offset_from_plate_origin
                )
        self.hmm_intermediate_first_tap_y_offset_from_plate_origin = FixedParameter(
                hmm_intermediate_first_tap_y_offset_from_plate_origin
                )
        self.mvm_tap_x_offset = FixedParameter(mvm_tap_x_offset)
        self.mvm_tap_z_offset = FixedParameter(mvm_tap_z_offset)
        self.connector_offset_from_vertebrae = FixedParameter(connector_offset_from_vertebrae)
        self.s_tap_x_offset_from_vertebrae = FixedParameter(s_tap_x_offset_from_vertebrae)
        self.s_tap_y_offset_from_vertebrae = FixedParameter(s_tap_y_offset_from_vertebrae)
        self.x_axis_gliding_joint_specification = x_axis_gliding_joint_specification
        self.y_axis_gliding_joint_specification = y_axis_gliding_joint_specification
        self.depth = FixedParameter(depth)


class SeahorseVertebraeSpecification(Specification):
    def __init__(
            self,
            *,
            z_offset_to_ball_bearing: float,
            offset_to_spine_attachment_point: float,
            connector_length: float,
            offset_to_bar_end: float,
            vertebral_mesh_specification: MeshSpecification,
            ball_bearing_mesh_specification: MeshSpecification,
            connector_mesh_specification: MeshSpecification,
            yaw_joint_specification: JointSpecification,
            pitch_joint_specification: JointSpecification,
            roll_joint_specification: JointSpecification
            ) -> None:
        super().__init__()
        self.z_offset_to_ball_bearing = FixedParameter(z_offset_to_ball_bearing)
        self.offset_to_spine_attachment_point = FixedParameter(offset_to_spine_attachment_point)
        self.offset_to_bar_end = FixedParameter(offset_to_bar_end)
        self.connector_length = FixedParameter(connector_length)
        self.vertebral_mesh_specification = vertebral_mesh_specification
        self.ball_bearing_mesh_specification = ball_bearing_mesh_specification
        self.connector_mesh_specification = connector_mesh_specification
        self.yaw_joint_specification = yaw_joint_specification
        self.pitch_joint_specification = pitch_joint_specification
        self.roll_joint_specification = roll_joint_specification


class SeahorseTendonSpineSpecification(Specification):
    def __init__(
            self,
            *,
            stiffness: float,
            damping: float,
            tendon_width: float
            ):
        super().__init__()
        self.stiffness = stiffness
        self.damping = damping
        self.tendon_width = tendon_width


class SeahorseSegmentSpecification(Specification):
    def __init__(
            self,
            *,
            z_offset_from_previous_segment: float,
            vertebrae_specification: SeahorseVertebraeSpecification,
            tendon_spine_specification: SeahorseTendonSpineSpecification,
            plate_specifications: List[SeahorsePlateSpecification]
            ) -> None:
        super().__init__()
        self.z_offset_from_previous_segment = FixedParameter(z_offset_from_previous_segment)
        self.vertebrae_specification = vertebrae_specification
        self.tendon_spine_specification = tendon_spine_specification
        self.plate_specifications = plate_specifications


class SeahorseHMMTendonActuationSpecification(Specification):
    def __init__(
            self,
            *,
            tendon_strain: float,
            p_control_kp: float,
            tendon_width: float,
            segment_span: int,
            damping: float
            ) -> None:
        super().__init__()
        self.tendon_strain = FixedParameter(tendon_strain)
        self.p_control_kp = FixedParameter(p_control_kp)
        self.tendon_width = FixedParameter(tendon_width)
        self.segment_span = FixedParameter(segment_span)
        self.damping = FixedParameter(damping)


class SeahorseMVMTendonActuationSpecification(Specification):
    def __init__(
            self,
            *,
            enabled: bool,
            contraction_factor: float,
            relaxation_factor: float,
            p_control_kp: float,
            tendon_width: float,
            damping: float
            ) -> None:
        super().__init__()
        self.enabled = FixedParameter(enabled)
        self.contraction_factor = FixedParameter(contraction_factor)
        self.relaxation_factor = FixedParameter(relaxation_factor)
        self.p_control_kp = FixedParameter(p_control_kp)
        self.tendon_width = FixedParameter(tendon_width)
        self.damping = FixedParameter(damping)


class SeahorseTendonActuationSpecification(Specification):
    def __init__(
            self,
            *,
            hmm_tendon_actuation_specification: SeahorseHMMTendonActuationSpecification,
            mvm_tendon_actuation_specification: SeahorseMVMTendonActuationSpecification
            ) -> None:
        super().__init__()
        self.hmm_tendon_actuation_specification = hmm_tendon_actuation_specification
        self.mvm_tendon_actuation_specification = mvm_tendon_actuation_specification


class SeahorseMorphologySpecification(MorphologySpecification):
    sides = ["ventral", "sinistral", "dorsal", "dextral"]
    corners = ["ventral_dextral", "ventral_sinistral", "dorsal_sinistral", "dorsal_dextral"]

    def __init__(
            self,
            *,
            segment_specifications: List[SeahorseSegmentSpecification],
            tendon_actuation_specification: SeahorseTendonActuationSpecification
            ) -> None:
        super().__init__()
        self.segment_specifications = segment_specifications
        self.tendon_actuation_specification = tendon_actuation_specification

    @property
    def num_segments(
            self
            ) -> int:
        return len(self.segment_specifications)
