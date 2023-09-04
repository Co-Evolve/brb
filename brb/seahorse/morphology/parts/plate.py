from typing import Union

import numpy as np
from mujoco_utils.robot import MJCMorphology, MJCMorphologyPart

from brb.seahorse.morphology.specification.specification import SeahorseMorphologySpecification, \
    SeahorsePlateSpecification
from brb.utils import colors


class SeahorsePlate(MJCMorphologyPart):
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
    def plate_specification(
            self
            ) -> SeahorsePlateSpecification:
        return self.morphology_specification.segment_specifications[self.segment_index].plate_specifications[
            self.plate_index]

    def _build(
            self,
            segment_index: int,
            plate_index: int
            ) -> None:
        self.segment_index = segment_index
        self.plate_index = plate_index

        self._configure_mesh_assets()
        self._build_plate()
        self._build_connectors()
        self._configure_gliding_joints()
        self._configure_actuator_tendon_attachment_points()
        self._configure_spine_tendon_attachment_points()

    def _configure_mesh_assets(
            self
            ) -> None:
        plate_mesh_specification = self.plate_specification.plate_mesh_specification
        connector_mesh_specification = self.plate_specification.connector_mesh_specification

        self.mjcf_model.asset.add(
            "mesh",
            name=f"{self.base_name}_plate",
            file=plate_mesh_specification.mesh_path,
            scale=plate_mesh_specification.scale_ratio
            )
        self.mjcf_model.asset.add(
            "mesh",
            name=f"{self.base_name}_connector",
            file=connector_mesh_specification.mesh_path,
            scale=connector_mesh_specification.scale_ratio
            )

    def _build_plate(
            self
            ) -> None:
        self.plate = self.mjcf_body.add(
            'geom',
            name=f"{self.base_name}_plate",
            type="mesh",
            mesh=f"{self.base_name}_plate",
            pos=np.array([0.0, 0.0, 0.0]),
            rgba=colors.rgba_green,
            group=0,
            mass=self.plate_specification.plate_mesh_specification.mass
            )

    def _build_connectors(
            self
            ) -> None:
        self.connectors = []

        if self.plate_index == 0:
            return

        connector_mass = self.plate_specification.connector_mesh_specification.mass
        vertebrae_offset = self.plate_specification.vertebrae_connector_offset
        plate_pos = np.array(self.mjcf_body.pos)

        if self.plate_index == 1:
            connector_pos = np.array([-plate_pos[0] + vertebrae_offset, -plate_pos[1], 0.0])
            connector = self.mjcf_body.add(
                'geom',
                name=f"{self.base_name}_connector_ventral",
                type="mesh",
                mesh=f"{self.base_name}_connector",
                pos=connector_pos,
                rgba=colors.rgba_gray,
                group=0,
                mass=connector_mass
                )
            self.connectors.append(connector)
        if self.plate_index == 2:
            connector_pos = np.array([-plate_pos[0], -plate_pos[1] + vertebrae_offset, 0.0])
            connector_euler = np.array([0.0, 0.0, np.pi / 2])
            connector = self.mjcf_body.add(
                'geom',
                name=f"{self.base_name}_connector_dextral",
                type="mesh",
                mesh=f"{self.base_name}_connector",
                pos=connector_pos,
                euler=connector_euler,
                rgba=colors.rgba_gray,
                group=0,
                mass=connector_mass
                )
            self.connectors.append(connector)

        if self.plate_index == 2 and (self.segment_index % 2 == 1) or self.plate_index == 3 and (
                self.segment_index % 2 == 0):
            connector_pos = np.array([-plate_pos[0] - vertebrae_offset, -plate_pos[1], 0.0])
            connector_euler = np.zeros(3)
            connector = self.mjcf_body.add(
                'geom',
                name=f"{self.base_name}_connector_dorsal",
                type="mesh",
                mesh=f"{self.base_name}_connector",
                pos=connector_pos,
                euler=connector_euler,
                rgba=colors.rgba_gray,
                group=0,
                mass=connector_mass
                )
            self.connectors.append(connector)
        if self.plate_index == 3:
            connector_pos = np.array([-plate_pos[0], -plate_pos[1] - vertebrae_offset, 0.0])
            connector_euler = np.array([0.0, 0.0, np.pi / 2])
            connector = self.mjcf_body.add(
                'geom',
                name=f"{self.base_name}_connector_sinistral",
                type="mesh",
                mesh=f"{self.base_name}_connector",
                pos=connector_pos,
                euler=connector_euler,
                rgba=colors.rgba_gray,
                group=0,
                mass=connector_mass
                )
            self.connectors.append(connector)

    def _configure_actuator_tendon_attachment_points(
            self
            ) -> None:
        plate_half_height = self.plate_specification.half_height
        self.a_tap_bottom = self.mjcf_body.add(
            'site',
            name=f"{self.base_name}_a_tap_bottom",
            type="sphere",
            rgba=colors.rgba_red,
            pos=np.array([0.0, 0.0, -plate_half_height]),
            size=[0.001]
            )
        self.a_tap_top = self.mjcf_body.add(
            'site',
            name=f"{self.base_name}_a_tap_top",
            type="sphere",
            rgba=colors.rgba_red,
            pos=np.array([0.0, 0.0, plate_half_height]),
            size=[0.001]
            )

    def _configure_spine_tendon_attachment_points(
            self
            ) -> None:
        self.s_taps = []
        for connector in self.connectors:
            s_tap = self.mjcf_body.add(
                'site',
                name=f"{connector.name}_s_tap",
                type="sphere",
                rgba=colors.rgba_red,
                pos=connector.pos,
                size=[0.001]
                )
            self.s_taps.append(s_tap)

    def _configure_gliding_joints(
            self
            ) -> None:
        x_axis_joint_specification = self.plate_specification.x_axis_gliding_joint_specification
        y_axis_joint_specification = self.plate_specification.y_axis_gliding_joint_specification

        if self.plate_index in [0, 1]:
            x_axis_range = (-x_axis_joint_specification.dof, 0)
        else:
            x_axis_range = (0, x_axis_joint_specification.dof)

        if self.plate_index in [0, 1]:
            y_axis_range = (-y_axis_joint_specification.dof, y_axis_joint_specification.dof)
        elif self.plate_index == 2:
            y_axis_range = (-y_axis_joint_specification.dof, 0)
        else:
            y_axis_range = (0, y_axis_joint_specification.dof)

        self.x_axis_gliding_joint = self.mjcf_body.add(
            'joint',
            name=f'{self.base_name}_x_axis',
            type='slide',
            pos=np.zeros(3),
            limited=True,
            axis=[1, 0, 0],
            range=x_axis_range,
            damping=x_axis_joint_specification.damping,
            stiffness=x_axis_joint_specification.stiffness, )
        self.y_axis_gliding_joint = self.mjcf_body.add(
            'joint',
            name=f'{self.base_name}_y_axis',
            type='slide',
            pos=np.zeros(3),
            limited=True,
            axis=[0, 1, 0],
            range=y_axis_range,
            damping=y_axis_joint_specification.damping,
            stiffness=y_axis_joint_specification.stiffness, )
