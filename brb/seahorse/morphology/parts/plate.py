from typing import List, Union

import numpy as np
from dm_control import mjcf
from mujoco_utils.robot import MJCMorphology, MJCMorphologyPart

from brb.seahorse.morphology.specification.specification import SeahorseMorphologySpecification, \
    SeahorsePlateSpecification
from brb.seahorse.morphology.utils import (add_mesh_to_body, get_actuator_tendon_plate_indices, get_plate_position,
                                           is_inner_plate_x_axis, is_inner_plate_y_axis)
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
        self._configure_hmm_tendon_attachment_points()
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
        self.plate = add_mesh_to_body(
                body=self.mjcf_body,
                name=f"{self.base_name}_plate",
                mesh_name=f"{self.base_name}_plate",
                position=position,
                euler=np.zeros(3),
                rgba=colors.rgba_green,
                group=0,
                mesh_specification=self.plate_specification.plate_mesh_specification
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

        offset_from_vertebrae = self.plate_specification.connector_offset_from_vertebrae.value

        if is_inner_plate_x:
            direction = 1 if self.plate_index < 2 else -1
            connector_pos = np.array([direction * offset_from_vertebrae, 0.0, 0.0])
            self.connectors["x"] = add_mesh_to_body(
                    body=self.mjcf_body,
                    name=f"{self.base_name}_connector_x",
                    mesh_name=f"{self.base_name}_connector",
                    position=connector_pos,
                    euler=np.zeros(3),
                    rgba=colors.rgba_gray,
                    group=0,
                    mesh_specification=self.plate_specification.connector_mesh_specification
                    )

        if is_inner_plate_y:
            direction = 1 if 1 <= self.plate_index <= 2 else -1
            connector_pos = np.array([0.0, direction * offset_from_vertebrae, 0.0])
            connector_euler = np.array([0.0, 0.0, np.pi / 2])

            self.connectors["y"] = add_mesh_to_body(
                    body=self.mjcf_body,
                    name=f"{self.base_name}_connector_y",
                    mesh_name=f"{self.base_name}_connector",
                    position=connector_pos,
                    euler=connector_euler,
                    rgba=colors.rgba_gray,
                    group=0,
                    mesh_specification=self.plate_specification.connector_mesh_specification
                    )

    def _add_hmm_ghost_tendon_attachment_points(
            self,
            side: str, ) -> List[mjcf.Element]:
        plate_specification = \
            self.morphology_specification.segment_specifications[self.segment_index].plate_specifications[
                self.plate_index]

        bottom_site = self.mjcf_body.add(
                'site',
                name=f"{self.base_name}_ghost_tap_{side}_bottom",
                type="sphere",
                rgba=colors.rgba_red,
                pos=self.plate.pos + np.array(
                        [plate_specification.hmm_ghost_tap_x_offset_from_plate_origin.value,
                         plate_specification.hmm_ghost_tap_y_offset_from_plate_origin.value,
                         -plate_specification.depth.value]
                        ),
                size=[0.001]
                )
        top_site = self.mjcf_body.add(
                'site',
                name=f"{self.base_name}_hmm_ghost_tap_{side}_top",
                type="sphere",
                rgba=colors.rgba_red,
                pos=self.plate.pos + np.array(
                        [plate_specification.hmm_ghost_tap_x_offset_from_plate_origin.value,
                         plate_specification.hmm_ghost_tap_y_offset_from_plate_origin.value, 0.0]
                        ),
                size=[0.001]
                )
        return [bottom_site, top_site]

    def _configure_hmm_tendon_attachment_points(
            self
            ) -> None:
        self._configure_intermediate_hmm_taps()
        self._configure_ghost_hmm_taps()
        self._configure_end_hmm_taps()

    def _configure_intermediate_hmm_taps(
            self
            ) -> None:
        first_tap_pos = np.array(
                [self.plate_specification.hmm_intermediate_first_tap_x_offset_from_plate_origin.value,
                 self.plate_specification.hmm_intermediate_first_tap_y_offset_from_plate_origin.value, 0.0]
                ) + self.plate.pos
        translation = np.array([0.0, self.plate_specification.hmm_y_offset_between_intermediate_taps.value, 0.0])
        self.intermediate_hmm_taps = []
        z_offsets = [-0.0092, -0.0057]
        for tap_index in range(self.plate_specification.hmm_num_intermediate_taps.value):
            pos = first_tap_pos + -np.sign(first_tap_pos[1]) * tap_index * translation
            taps = []
            for sub_index, z_offset in enumerate(z_offsets):
                pos[2] += z_offset
                taps.append(
                        self.mjcf_body.add(
                                'site',
                                pos=pos,
                                type="sphere",
                                size=[0.0005],
                                rgba=colors.rgba_red,
                                name=f"{self.base_name}_intermediate_hmm_tap_{tap_index}_{sub_index}"
                                )
                        )
                pos[2] -= z_offset
            self.intermediate_hmm_taps.append(taps)

    def _configure_ghost_hmm_taps(
            self
            ) -> None:
        self.upper_ghost_hmm_taps = []
        self.lower_ghost_hmm_taps = []
        side = self.morphology_specification.corners[self.plate_index].split("_")[0]
        if self.plate_index != get_actuator_tendon_plate_indices(side=side,
                                                                 segment_index=self.segment_index)[0]:
            return

        pos = np.array(
                [self.plate_specification.hmm_ghost_tap_x_offset_from_plate_origin.value,
                 self.plate_specification.hmm_ghost_tap_y_offset_from_plate_origin.value, 0.0]
                ) + self.plate.pos

        y_offsets = [0.0, -np.sign(pos[1]) * 0.012]
        z_offsets = [-0.0092, -0.0057]
        x_offsets = [-0.002, 0.002][:: -1 if side == "dorsal" else 1]
        for tap_index, (taps, x_offset) in enumerate(
                zip(
                        [self.upper_ghost_hmm_taps, self.lower_ghost_hmm_taps], x_offsets
                        )
                ):
            pos[0] += x_offset
            for side, y_offset in zip(["a", "b"], y_offsets):
                side_taps = []
                pos[1] += y_offset
                for sub_index, z_offset in enumerate(z_offsets):
                    pos[2] += z_offset
                    side_taps.append(
                            self.mjcf_body.add(
                                    'site',
                                    pos=pos,
                                    type="sphere",
                                    size=[0.0005],
                                    rgba=colors.rgba_red,
                                    name=f"{self.base_name}_ghost_hmm_tap_{side}_{tap_index}_{sub_index}"
                                    )
                            )
                    pos[2] -= z_offset
                taps.append(side_taps)
                pos[1] -= y_offset
            pos[0] -= x_offset

    def _configure_end_hmm_taps(
            self
            ) -> None:
        self.hmm_tap_end = []
        z_offsets = [-0.002, 0.0]
        for tap_index, z_offset in enumerate(z_offsets):
            self.hmm_tap_end.append(
                    self.mjcf_body.add(
                            'site',
                            name=f"{self.base_name}_end_hmm_tap_{tap_index}",
                            type="sphere",
                            rgba=colors.rgba_red,
                            pos=self.plate.pos + np.array([0.0, 0.0, z_offset]),
                            size=[0.001]
                            )
                    )

    def _configure_mvm_tendon_attachment_points(
            self
            ) -> None:
        if not (self.morphology_specification.tendon_actuation_specification.mvm_tendon_actuation_specification
                .enabled.value):
            return
        if self.plate_index == 1:
            self.mvm_taps = []

            x = self.plate_specification.mvm_tap_x_offset.value
            z = -self.plate_specification.depth.value / 2 + self.plate_specification.mvm_tap_z_offset.value
            self.mvm_taps.append(
                    self.mjcf_body.add(
                            'site',
                            name=f"{self.base_name}_mvm_tap_proximal",
                            type="sphere",
                            rgba=colors.rgba_red,
                            pos=[x, 0, z],
                            size=[0.001]
                            )
                    )

            z = self.plate_specification.depth.value / 2 - self.plate_specification.mvm_tap_z_offset.value
            self.mvm_taps.append(
                    self.mjcf_body.add(
                            'site',
                            name=f"{self.base_name}_mvm_tap_distal",
                            type="sphere",
                            rgba=colors.rgba_red,
                            pos=[x, 0, z],
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
