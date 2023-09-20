from typing import Union

import numpy as np
from dm_control import mjcf
from mujoco_utils.robot import MJCMorphology, MJCMorphologyPart

from brb.seahorse.morphology.specification.specification import SeahorseMorphologySpecification, \
    SeahorseVertebraeSpecification
from brb.utils import colors

SEAHORSE_VERTEBRAE_COLOR = np.array([158, 38, 212, 255]) / 255


class SeahorseVertebrae(MJCMorphologyPart):
    def __init__(
            self,
            parent: Union[MJCMorphology, MJCMorphologyPart],
            name: str,
            pos: np.array,
            euler: np.array,
            *args,
            **kwargs
            ) -> None:
        super().__init__(parent=parent, name=name, pos=pos, euler=euler, *args, **kwargs)

    @property
    def morphology_specification(
            self
            ) -> SeahorseMorphologySpecification:
        return super().morphology_specification

    @property
    def vertebrae_specification(
            self
            ) -> SeahorseVertebraeSpecification:
        return self.morphology_specification.segment_specifications[self.segment_index].vertebrae_specification

    @property
    def is_first_segment(
            self
            ) -> bool:
        return self.segment_index == 0

    @property
    def is_last_segment(
            self
            ) -> bool:
        return self.segment_index == (self.morphology_specification.num_segments - 1)

    def _build(
            self,
            segment_index: int
            ) -> None:
        self.segment_index = segment_index

        self._configure_mesh_assets()
        self._build_vertebrae()
        self._build_ball_bearing()
        self._build_connectors()
        self._configure_spine_tendon_attachment_points()
        self._configure_vertebral_joints()

    def _configure_mesh_assets(
            self
            ) -> None:
        vertebrae_mesh_specification = self.vertebrae_specification.vertebral_mesh_specification
        ball_bearing_mesh_specification = self.vertebrae_specification.ball_bearing_mesh_specification
        connector_mesh_specification = self.vertebrae_specification.connector_mesh_specification

        self.mjcf_model.asset.add(
                "mesh",
                name=f"{self.base_name}_vertebrae",
                file=vertebrae_mesh_specification.mesh_path.value,
                scale=vertebrae_mesh_specification.scale_ratio.value
                )
        self.mjcf_model.asset.add(
                "mesh",
                name=f"{self.base_name}_ball_bearing",
                file=ball_bearing_mesh_specification.mesh_path.value,
                scale=ball_bearing_mesh_specification.scale_ratio.value
                )
        self.mjcf_model.asset.add(
                "mesh",
                name=f"{self.base_name}_connector",
                file=connector_mesh_specification.mesh_path.value,
                scale=connector_mesh_specification.scale_ratio.value
                )

    def _build_vertebrae(
            self
            ) -> None:
        self.vertebrae = self.mjcf_body.add(
                'geom',
                name=f"{self.base_name}_vertebrae",
                type="mesh",
                mesh=f"{self.base_name}_vertebrae",
                pos=np.zeros(3),
                rgba=SEAHORSE_VERTEBRAE_COLOR,
                group=1,
                mass=self.vertebrae_specification.vertebral_mesh_specification.mass.value
                )

    def _build_ball_bearing(
            self
            ) -> None:
        if not self.is_last_segment:
            self.ball_bearing = self.mjcf_body.add(
                    'geom',
                    name=f"{self.base_name}_ball_bearing",
                    type="mesh",
                    mesh=f"{self.base_name}_ball_bearing",
                    pos=np.array([0, 0, self.vertebrae_specification.z_offset_to_ball_bearing.value]),
                    rgba=colors.rgba_gray,
                    group=1,
                    mass=self.vertebrae_specification.ball_bearing_mesh_specification.mass.value
                    )

    def _build_connectors(
            self
            ) -> None:
        sides = ["ventral", "dextral", "dorsal", "sinistral"]
        radius = (
                    self.vertebrae_specification.offset_to_bar_end.value +
                    self.vertebrae_specification.connector_length.value / 2)

        angles = np.array([np.pi / 2 * side_index for side_index in range(4)])
        positions = radius * np.array([[np.cos(angle), np.sin(angle), 0.0] for angle in angles])
        eulers = np.array([[0, 0, angle] for angle in angles])

        self.connectors = []
        for side, position, euler in zip(sides, positions, eulers):
            connector = self.mjcf_body.add(
                    'geom',
                    name=f"{self.base_name}_connector_{side}",
                    type="mesh",
                    mesh=f"{self.base_name}_connector",
                    pos=position,
                    euler=euler,
                    rgba=colors.rgba_gray,
                    group=1,
                    mass=self.vertebrae_specification.connector_mesh_specification.mass.value
                    )
            self.connectors.append(connector)

    def _configure_spine_tendon_attachment_points(
            self
            ) -> None:
        sides = self.morphology_specification.sides
        angles = [0, np.pi / 2, np.pi, 3 * np.pi / 2]
        radius = self.vertebrae_specification.offset_to_spine_attachment_point.value

        self.s_taps = []
        for side, angle in zip(sides, angles):
            pos = radius * np.array([np.cos(angle), np.sin(angle), 0.0])
            s_tap = self.mjcf_body.add(
                    'site',
                    name=f"{self.base_name}_s_tap_{side}",
                    type="sphere",
                    rgba=colors.rgba_red,
                    pos=pos,
                    size=[0.001]
                    )
            self.s_taps.append(s_tap)

    def _configure_vertebral_joints(
            self
            ) -> None:
        if self.is_first_segment:
            return

        joint_pos = np.array([0.0, 0.0, -self.vertebrae_specification.z_offset_to_ball_bearing.value])

        bend_joint_specification = self.vertebrae_specification.bend_joint_specification
        twist_joint_specification = self.vertebrae_specification.twist_joint_specification

        self.mjcf_body.add(
                'joint',
                name=f'{self.base_name}_vertebrae_joint_coronal',
                type='hinge',
                pos=joint_pos,
                limited=True,
                axis=[0, 1, 0],
                range=(-bend_joint_specification.range.value, bend_joint_specification.range.value),
                damping=bend_joint_specification.damping.value,
                stiffness=bend_joint_specification.stiffness.value,
                frictionloss=bend_joint_specification.friction_loss.value,
                armature=bend_joint_specification.armature.value
                )
        self.mjcf_body.add(
                'joint',
                name=f'{self.base_name}_vertebrae_joint_sagittal',
                type='hinge',
                pos=joint_pos,
                limited=True,
                axis=[1, 0, 0],
                range=(-bend_joint_specification.range.value, bend_joint_specification.range.value),
                damping=bend_joint_specification.damping.value,
                stiffness=bend_joint_specification.stiffness.value,
                frictionloss=bend_joint_specification.friction_loss.value,
                armature=bend_joint_specification.armature.value
                )
        if twist_joint_specification.range.value != 0:
            self.mjcf_body.add(
                    'joint',
                    name=f'{self.base_name}_vertebrae_joint_twist',
                    type='hinge',
                    pos=joint_pos,
                    limited=True,
                    axis=[0, 0, 1],
                    range=(-twist_joint_specification.range.value, twist_joint_specification.range.value),
                    damping=twist_joint_specification.damping.value,
                    stiffness=twist_joint_specification.stiffness.value,
                    frictionloss=twist_joint_specification.friction_loss.value
                    )

    def add_intermediate_a_tap(
            self,
            identifier: str,
            start_world_pos: np.ndarray,
            stop_world_pos: np.ndarray
            ) -> mjcf.Element:
        # linearly interpolate between two world positions x and y based on z
        start_z = start_world_pos[2]
        stop_z = stop_world_pos[2]

        vertebrae_world_z = self.world_coordinates_of_point([0.0, 0.0, 0.0])[2]
        normalized_z = (vertebrae_world_z - start_z) / (stop_z - start_z)
        intermediate_pos = start_world_pos + normalized_z * (stop_world_pos - start_world_pos)
        intermediate_pos[2] = 0

        return self.mjcf_body.add(
                'site',
                name=f"{self.base_name}_a_tap_intermediate_{identifier}",
                type="sphere",
                rgba=colors.rgba_orange,
                pos=np.array(intermediate_pos), size=[0.001]
                )