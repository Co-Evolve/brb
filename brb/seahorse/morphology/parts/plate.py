from typing import List, Union

import numpy as np
from dm_control import mjcf
from mujoco_utils.robot import MJCMorphology, MJCMorphologyPart

from brb.seahorse.morphology.specification.specification import SeahorseMorphologySpecification, \
    SeahorsePlateSpecification
from brb.seahorse.morphology.utils import (get_actuator_tendon_plate_indices, get_plate_position, is_inner_plate_x_axis, \
                                           is_inner_plate_y_axis)
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
        self._configure_mvm_tendon_attachment_points()
        self._configure_spine_tendon_attachment_points()

    def _configure_mesh_assets(
            self
            ) -> None:
        plate_mesh_specification = self.plate_specification.plate_mesh_specification
        connector_mesh_specification = self.plate_specification.connector_mesh_specification

        self.mjcf_model.asset.add(
                "mesh",
                name=f"{self.base_name}_plate",
                file=plate_mesh_specification.mesh_path.value,
                scale=plate_mesh_specification.scale_ratio.value
                )
        self.mjcf_model.asset.add(
                "mesh",
                name=f"{self.base_name}_connector",
                file=connector_mesh_specification.mesh_path.value,
                scale=connector_mesh_specification.scale_ratio.value
                )

    def _build_plate(
            self
            ) -> None:
        position = get_plate_position(plate_index=self.plate_index, plate_specification=self.plate_specification)
        self.plate = self.mjcf_body.add(
                'geom',
                name=f"{self.base_name}_plate",
                type="mesh",
                mesh=f"{self.base_name}_plate",
                pos=position,
                rgba=colors.rgba_green,
                group=0,
                mass=self.plate_specification.plate_mesh_specification.mass.value
                )

    def _build_connectors(
            self
            ) -> None:
        self.connectors = {}

        is_inner_plate_x = is_inner_plate_x_axis(
                segment_index=self.segment_index, plate_index=self.plate_index
                )
        is_inner_plate_y = is_inner_plate_y_axis(
                segment_index=self.segment_index, plate_index=self.plate_index
                )

        connector_mass = self.plate_specification.connector_mesh_specification.mass.value
        offset_from_vertebrae = self.plate_specification.connector_offset_from_vertebrae.value

        if is_inner_plate_x:
            direction = 1 if self.plate_index < 2 else -1
            connector_pos = np.array([direction * offset_from_vertebrae, 0.0, 0.0])
            self.connectors["x"] = self.mjcf_body.add(
                    'geom',
                    name=f"{self.base_name}_connector_x",
                    type="mesh",
                    mesh=f"{self.base_name}_connector",
                    pos=connector_pos,
                    rgba=colors.rgba_gray,
                    group=0,
                    mass=connector_mass
                    )

        if is_inner_plate_y:
            direction = 1 if 1 <= self.plate_index <= 2 else -1
            connector_pos = np.array([0.0, direction * offset_from_vertebrae, 0.0])
            connector_euler = np.array([0.0, 0.0, np.pi / 2])
            self.connectors["y"] = self.mjcf_body.add(
                    'geom',
                    name=f"{self.base_name}_connector_y",
                    type="mesh",
                    mesh=f"{self.base_name}_connector",
                    pos=connector_pos,
                    euler=connector_euler,
                    rgba=colors.rgba_gray,
                    group=0,
                    mass=connector_mass
                    )

    def _add_actuator_tendon_attachment_points(
            self,
            side: str,
            plate_index, ) -> List[mjcf.Element]:
        plate_specification = \
            self.morphology_specification.segment_specifications[self.segment_index].plate_specifications[plate_index]
        plate_position = get_plate_position(
                plate_index=plate_index, plate_specification=plate_specification
                )

        bottom_site = self.mjcf_body.add(
                'site',
                name=f"{self.base_name}_a_tap_{side}_bottom",
                type="sphere",
                rgba=colors.rgba_red,
                pos=plate_position + np.array(
                        [plate_specification.a_tap_x_offset_from_plate_origin.value,
                         plate_specification.a_tap_y_offset_from_plate_origin.value, -plate_specification.depth.value]
                        ),
                size=[0.001]
                )
        top_site = self.mjcf_body.add(
                'site',
                name=f"{self.base_name}_a_tap_{side}_top",
                type="sphere",
                rgba=colors.rgba_red,
                pos=plate_position + np.array(
                        [plate_specification.a_tap_x_offset_from_plate_origin.value,
                         plate_specification.a_tap_y_offset_from_plate_origin.value, 0.0]
                        ),
                size=[0.001]
                )
        return [bottom_site, top_site]

    def _configure_actuator_tendon_attachment_points(
            self
            ) -> None:
        self.a_tap_end = self.mjcf_body.add(
                'site',
                name=f"{self.base_name}_a_tap_end",
                type="sphere",
                rgba=colors.rgba_red,
                pos=self.plate.pos,
                size=[0.001]
                )

        for side in ["ventral", "dorsal"]:
            main_plate_index, other_plate_index = get_actuator_tendon_plate_indices(
                    side=side, segment_index=self.segment_index
                    )
            if self.plate_index == main_plate_index:
                if 1 <= main_plate_index <= 2:
                    sinistral_plate_index = main_plate_index
                    dextral_plate_index = other_plate_index
                else:
                    sinistral_plate_index = other_plate_index
                    dextral_plate_index = main_plate_index

                self.a_taps_sinistral = self._add_actuator_tendon_attachment_points(
                        side=f"{side}_sinistral", plate_index=sinistral_plate_index
                        )
                self.a_taps_dextral = self._add_actuator_tendon_attachment_points(
                        side=f"{side}_dextral", plate_index=dextral_plate_index
                        )

    def _configure_mvm_tendon_attachment_points(
            self
            ) -> None:
        if self.plate_index == 0:
            self.mvm_a_taps = []

            # todo: move to specification
            x = 0.0435
            y = 0
            z = -self.plate_specification.depth.value / 2 + 0.004
            self.mvm_a_taps.append(
                    self.mjcf_body.add(
                            'site',
                            name=f"{self.base_name}_mvm_a_tap_proximal",
                            type="sphere",
                            rgba=colors.rgba_red,
                            pos=[x, y, z],
                            size=[0.001]
                            )
                    )
            x = 0.0435
            y = 0
            z = self.plate_specification.depth.value / 2 - 0.004
            self.mvm_a_taps.append(
                    self.mjcf_body.add(
                            'site',
                            name=f"{self.base_name}_mvm_a_tap_distal",
                            type="sphere",
                            rgba=colors.rgba_red,
                            pos=[x, y, z],
                            size=[0.001]
                            )
                    )

    def _configure_spine_tendon_attachment_points(
            self
            ) -> None:
        self.s_taps = {}

        s_tap_x_offset = self.plate_specification.s_tap_x_offset_from_vertebrae.value
        x_direction = 1 if self.plate_index < 2 else -1
        s_tap_y_offset = self.plate_specification.s_tap_y_offset_from_vertebrae.value
        y_direction = 1 if 1 <= self.plate_index <= 2 else -1

        self.s_taps['plate-x'] = self.mjcf_body.add(
                'site',
                name=f"{self.base_name}_s_tap_x",
                type="sphere",
                rgba=colors.rgba_red,
                pos=np.array([x_direction * s_tap_x_offset, 0.0, 0.0]),
                size=[0.001]
                )
        self.s_taps["plate-y"] = self.mjcf_body.add(
                'site',
                name=f"{self.base_name}_s_tap_y",
                type="sphere",
                rgba=colors.rgba_red,
                pos=np.array([0.0, y_direction * s_tap_y_offset, 0.0]),
                size=[0.001]
                )

        for axis, connector in self.connectors.items():
            self.s_taps[f"connector-{axis}"] = self.mjcf_body.add(
                    'site',
                    name=f"{connector.name}_s_tap",
                    type='sphere',
                    rgba=colors.rgba_red,
                    pos=0.9 * connector.pos,
                    # factor to bring it a bit closer to the vertebrae
                    size=[0.001]
                    )

    def _configure_gliding_joints(
            self
            ) -> None:

        x_axis_joint_specification = self.plate_specification.x_axis_gliding_joint_specification
        y_axis_joint_specification = self.plate_specification.y_axis_gliding_joint_specification

        if self.plate_index in [0, 1]:
            x_axis_range = (-x_axis_joint_specification.range.value, 0)
        else:
            x_axis_range = (0, x_axis_joint_specification.range.value)

        if self.plate_index in [0, 3]:
            y_axis_range = (0, y_axis_joint_specification.range.value)
        else:
            y_axis_range = (-y_axis_joint_specification.range.value, 0)

        if x_axis_joint_specification.range.value != 0:
            self.x_axis_gliding_joint = self.mjcf_body.add(
                    'joint',
                    name=f'{self.base_name}_x_axis',
                    type='slide',
                    pos=np.zeros(3),
                    limited=True,
                    axis=[1, 0, 0],
                    range=x_axis_range,
                    damping=x_axis_joint_specification.damping.value,
                    stiffness=x_axis_joint_specification.stiffness.value
                    )
        if y_axis_joint_specification.range.value != 0:
            self.y_axis_gliding_joint = self.mjcf_body.add(
                    'joint',
                    name=f'{self.base_name}_y_axis',
                    type='slide',
                    pos=np.zeros(3),
                    limited=True,
                    axis=[0, 1, 0],
                    range=y_axis_range,
                    damping=y_axis_joint_specification.damping.value,
                    stiffness=y_axis_joint_specification.stiffness.value
                    )
