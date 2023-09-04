from typing import Union

import numpy as np
from mujoco_utils.robot import MJCMorphology, MJCMorphologyPart

from brb.seahorse.morphology.specification.specification import SeahorseMorphologySpecification, \
    SeahorseVertebraeSpecification
from brb.utils import colors

SEAHORSE_VERTEBRAE_COLOR = np.array([158, 38, 212, 255]) / 255


class SeahorseVertebrae(MJCMorphologyPart):
    def __init__(self, parent: Union[MJCMorphology, MJCMorphologyPart], name: str, pos: np.array, euler: np.array,
                 *args, **kwargs) -> None:
        super().__init__(parent=parent, name=name, pos=pos, euler=euler, *args, **kwargs)

    @property
    def morphology_specification(self) -> SeahorseMorphologySpecification:
        return super().morphology_specification

    @property
    def vertebrae_specification(self) -> SeahorseVertebraeSpecification:
        return self.morphology_specification.segment_specifications[self.segment_index].vertebrae_specification

    @property
    def is_last_segment(self) -> bool:
        return self.segment_index == (self.morphology_specification.num_segments - 1)

    def _build(self, segment_index: int) -> None:
        self.segment_index = segment_index

        self._configure_mesh_assets()
        self._build_vertebrae()
        self._build_ball_bearing()
        self._build_connectors()
        self._configure_spine_tendon_attachment_points()
        self._configure_vertebral_joints()

    def _configure_mesh_assets(self) -> None:
        vertebrae_mesh_specification = self.vertebrae_specification.vertebral_mesh_specification
        ball_bearing_mesh_specification = self.vertebrae_specification.ball_bearing_mesh_specification
        connector_mesh_specification = self.vertebrae_specification.connector_mesh_specification

        self.mjcf_model.asset.add("mesh",
                                  name=f"{self.base_name}_vertebrae",
                                  file=vertebrae_mesh_specification.mesh_path,
                                  scale=vertebrae_mesh_specification.scale_ratio)
        self.mjcf_model.asset.add("mesh",
                                  name=f"{self.base_name}_ball_bearing",
                                  file=ball_bearing_mesh_specification.mesh_path,
                                  scale=ball_bearing_mesh_specification.scale_ratio)
        self.mjcf_model.asset.add("mesh",
                                  name=f"{self.base_name}_connector",
                                  file=connector_mesh_specification.mesh_path,
                                  scale=connector_mesh_specification.scale_ratio)

    def _build_vertebrae(self) -> None:
        self.vertebrae = self.mjcf_body.add('geom', name=f"{self.base_name}_vertebrae",
                                            type="mesh",
                                            mesh=f"{self.base_name}_vertebrae",
                                            pos=np.zeros(3),
                                            rgba=SEAHORSE_VERTEBRAE_COLOR,
                                            group=1,
                                            mass=self.vertebrae_specification.vertebral_mesh_specification.mass)

    def _build_ball_bearing(self) -> None:
        if not self.is_last_segment:
            self.ball_bearing = self.mjcf_body.add('geom', name=f"{self.base_name}_ball_bearing",
                                                   type="mesh",
                                                   mesh=f"{self.base_name}_ball_bearing",
                                                   pos=np.array([0.0, 0.0, 0.014]),
                                                   rgba=colors.rgba_gray,
                                                   group=1,
                                                   mass=self.vertebrae_specification.ball_bearing_mesh_specification.mass)

    def _build_connectors(self) -> None:
        sides = ["ventral", "dextral", "dorsal", "sinistral"]
        radius = 0.015

        angles = np.array([np.pi / 2 * side_index for side_index in range(4)])
        positions = radius * np.array([[np.cos(angle), np.sin(angle), 0.0] for angle in angles])
        eulers = np.array([[0, 0, angle] for angle in angles])

        self.connectors = []
        for side, position, euler in zip(sides, positions, eulers):
            connector = self.mjcf_body.add('geom',
                                           name=f"{self.base_name}_connector_{side}",
                                           type="mesh",
                                           mesh=f"{self.base_name}_connector",
                                           pos=position,
                                           euler=euler,
                                           rgba=colors.rgba_gray,
                                           group=1,
                                           mass=self.vertebrae_specification.connector_mesh_specification.mass)
            self.connectors.append(connector)

    def _configure_spine_tendon_attachment_points(self) -> None:
        self.s_taps = []
        for connector in self.connectors:
            s_tap = self.mjcf_body.add('site',
                                       name=f"{connector.name}_s_tap",
                                       type="sphere",
                                       rgba=colors.rgba_red,
                                       pos=connector.pos,
                                       size=[0.001])
            self.s_taps.append(s_tap)

    def _configure_vertebral_joints(self) -> None:
        # specify this joint at location of previous ball joint
        joint_pos = np.array([0.0, 0.0, -self.vertebrae_specification.vertebrae_half_height])

        bend_joint_specification = self.vertebrae_specification.bend_joint_specification
        twist_joint_specification = self.vertebrae_specification.twist_joint_specification

        self.mjcf_body.add('joint',
                           name=f'{self.base_name}_vertebrae_joint_coronal',
                           type='hinge',
                           pos=joint_pos,
                           limited=True,
                           axis=[0, 1, 0],
                           range=(-bend_joint_specification.dof, bend_joint_specification.dof),
                           damping=bend_joint_specification.damping,
                           stiffness=bend_joint_specification.stiffness,
                           frictionloss=bend_joint_specification.friction_loss
                           )
        self.mjcf_body.add('joint',
                           name=f'{self.base_name}_vertebrae_joint_sagittal',
                           type='hinge',
                           pos=joint_pos,
                           limited=True,
                           axis=[1, 0, 0],
                           range=(-bend_joint_specification.dof, bend_joint_specification.dof),
                           damping=bend_joint_specification.damping,
                           stiffness=bend_joint_specification.stiffness,
                           frictionloss=bend_joint_specification.friction_loss
                           )
        self.mjcf_body.add('joint',
                           name=f'{self.base_name}_vertebrae_joint_twist',
                           type='hinge',
                           pos=joint_pos,
                           limited=True,
                           axis=[0, 0, 1],
                           range=(-twist_joint_specification.dof, twist_joint_specification.dof),
                           damping=twist_joint_specification.damping,
                           stiffness=twist_joint_specification.stiffness,
                           frictionloss=twist_joint_specification.friction_loss
                           )