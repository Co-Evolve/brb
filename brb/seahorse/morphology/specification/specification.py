from __future__ import annotations

from typing import List

import numpy as np
from fprs.specification import MorphologySpecification, Specification


class MeshSpecification(Specification):
    def __init__(
            self,
            *,
            mesh_path: str,
            scale_ratio: np.ndarray,
            mass: float
            ) -> None:
        super().__init__()
        self.mesh_path = mesh_path
        self.scale_ratio = scale_ratio
        self.mass = mass


class JointSpecification(Specification):
    def __init__(
            self,
            *,
            stiffness: float,
            damping: float,
            friction_loss: float,
            dof: float
            ) -> None:
        super().__init__()
        self.stiffness = stiffness
        self.damping = damping
        self.friction_loss = friction_loss
        self.dof = dof


class SeahorsePlateSpecification(Specification):
    def __init__(
            self,
            *,
            plate_mesh_specification: MeshSpecification,
            connector_mesh_specification: MeshSpecification,
            offset_from_vertebrae: float,
            x_axis_gliding_joint_specification: JointSpecification,
            y_axis_gliding_joint_specification: JointSpecification,
            half_height: float,
            vertebrae_connector_offset: float
            ) -> None:
        super().__init__()
        self.plate_mesh_specification = plate_mesh_specification
        self.connector_mesh_specification = connector_mesh_specification
        self.offset_from_vertebrae = offset_from_vertebrae
        self.x_axis_gliding_joint_specification = x_axis_gliding_joint_specification
        self.y_axis_gliding_joint_specification = y_axis_gliding_joint_specification
        self.half_height = half_height
        self.vertebrae_connector_offset = vertebrae_connector_offset


class SeahorseVertebraeSpecification(Specification):
    def __init__(
            self,
            *,
            vertebral_mesh_specification: MeshSpecification,
            ball_bearing_mesh_specification: MeshSpecification,
            connector_mesh_specification: MeshSpecification,
            bend_joint_specification: JointSpecification,
            twist_joint_specification: JointSpecification,
            vertebrae_half_height: float
            ) -> None:
        super().__init__()
        self.vertebral_mesh_specification = vertebral_mesh_specification
        self.ball_bearing_mesh_specification = ball_bearing_mesh_specification
        self.connector_mesh_specification = connector_mesh_specification
        self.bend_joint_specification = bend_joint_specification
        self.twist_joint_specification = twist_joint_specification
        self.vertebrae_half_height = vertebrae_half_height


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
            vertebrae_specification: SeahorseVertebraeSpecification,
            tendon_spine_specification: SeahorseTendonSpineSpecification,
            plate_specifications: List[SeahorsePlateSpecification]
            ) -> None:
        super().__init__()
        self.vertebrae_specification = vertebrae_specification
        self.tendon_spine_specification = tendon_spine_specification
        self.plate_specifications = plate_specifications


class SeahorseTendonActuationSpecification(Specification):
    def __init__(
            self,
            *,
            contraction_factor: float,
            relaxation_factor: float,
            p_control_kp: float,
            tendon_width: float
            ) -> None:
        super().__init__()
        self.contraction_factor = contraction_factor
        self.relaxation_factor = relaxation_factor
        self.p_control_kp = p_control_kp
        self.tendon_width = tendon_width


class SeahorseMorphologySpecification(MorphologySpecification):
    sides = ["ventral", "dextral", "dorsal", "sinistral"]
    corners = ["ventral_sinistral", "ventral_dextral", "dorsal_dextral", "dorsal_sinistral"]

    def __init__(
            self,
            *,
            segment_specifications: List[SeahorseSegmentSpecification],
            tendon_actuation_specification: SeahorseTendonActuationSpecification, ) -> None:
        super().__init__()
        self.segment_specifications = segment_specifications
        self.tendon_actuation_specification = tendon_actuation_specification

    @property
    def num_segments(
            self
            ) -> int:
        return len(self.segment_specifications)
