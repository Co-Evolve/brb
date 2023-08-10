from typing import Union, List, Tuple

import numpy as np
from dm_control import mjcf
from dm_control.mjcf.element import _ElementImpl
from mujoco_utils.robot import MJCMorphology, MJCMorphologyPart

from brb.brittle_star.morphology.parts.arm_segment import MJCBrittleStarArmSegment
from brb.brittle_star.morphology.specification.specification import BrittleStarMorphologySpecification
from brb.utils import colors


class MJCBrittleStarArm(MJCMorphologyPart):
    def __init__(self, parent: Union[MJCMorphology, MJCMorphologyPart], name: str, pos: np.array, euler: np.array,
                 *args, **kwargs):
        super().__init__(parent, name, pos, euler, *args, **kwargs)

    @property
    def morphology_specification(self) -> BrittleStarMorphologySpecification:
        return super().morphology_specification

    def _build(self, arm_index: int) -> None:
        self._arm_index = arm_index
        self._arm_specification = self.morphology_specification.arm_specifications[self._arm_index]

        self._build_segments()
        self._configure_actuators()

    def _build_segments(self) -> None:
        self._segments = []

        number_of_segments = self._arm_specification.number_of_segments

        for segment_index in range(number_of_segments):
            try:
                parent = self._segments[-1]
                position = 2 * self._segments[-1].center_of_capsule
                # position[0] -= self._segments[-1]._segment_specification.radius.value
            except IndexError:
                position = np.zeros(3)
                parent = self

            segment = MJCBrittleStarArmSegment(parent=parent,
                                               name=f"{self.base_name}_segment_{segment_index}",
                                               pos=position,
                                               euler=np.zeros(3),
                                               arm_index=self._arm_index,
                                               segment_index=segment_index)
            self._segments.append(segment)

    def _configure_tendon_attachment_points(self) -> None:
        first_segment = self._segments[0]

        attachment_points = first_segment.tendon_attachment_points["proximal"]
        attachment_positions = [np.array(attachment_point[0].pos) for attachment_point in attachment_points]

        self._tendon_attachment_points = []
        for tendon_index, attachment_position in enumerate(attachment_positions):
            attachment_position[0] = 0.0
            attachment_point = self.mjcf_body.add("site",
                                                  name=f"{self.base_name}_muscle_attachment_point_{tendon_index}",
                                                  pos=attachment_position,
                                                  rgba=colors.rgba_red)
            self._tendon_attachment_points.append(attachment_point)

    def _get_tendon_morphology_parts_and_attachment_points(self, tendon_side: int, start_joint_index: int) -> Tuple[
        List[MJCMorphologyPart], List[mjcf.Element]]:
        attachment_points = []
        morphology_parts = []

        is_ventral_tendon = tendon_side < 2

        if start_joint_index == 0:
            # Attach to disc
            attachment_points.append(self._tendon_attachment_points[tendon_side])
            morphology_parts.append(self)
        else:
            previous_segment = self._segments[start_joint_index - 1]
            if is_ventral_tendon:
                sides = ["distal"]
                sub_side_indices = [[2]]
            else:
                sides = ["proximal", "distal"]
                sub_side_indices = [[2], [1]]

            for side, indices in zip(sides, sub_side_indices):
                for index in indices:
                    attachment_point = previous_segment.tendon_attachment_points[side][tendon_side][index]
                    attachment_points.append(attachment_point)
                    morphology_parts.append(previous_segment)

        current_segment = self._segments[start_joint_index]
        if is_ventral_tendon:
            sides = ["proximal", "distal"]
            sub_side_indices = [[1], [0]]
        else:
            sides = ["proximal"]
            sub_side_indices = [[0]]

        for side, indices in zip(sides, sub_side_indices):
            for index in indices:
                attachment_point = current_segment.tendon_attachment_points[side][tendon_side][index]
                attachment_points.append(attachment_point)
                morphology_parts.append(current_segment)

        return morphology_parts, attachment_points

    def _build_tendons(self) -> None:
        num_tendon_sides = len(self._tendon_attachment_points)
        self._tendons = []
        for start_joint_index in range(0, self._arm_specification.number_of_segments):
            for tendon_side in range(num_tendon_sides):
                morphology_parts, attachment_points = self._get_tendon_morphology_parts_and_attachment_points(
                    tendon_side=tendon_side, start_joint_index=start_joint_index)

                tendon = self.mjcf_model.tendon.add("spatial",
                                                    name=f"{self.base_name}_tendon_{tendon_side}_"
                                                         f"{start_joint_index}",
                                                    width=0.004,
                                                    rgba=colors.rgba_tendon_relaxed)
                for attachment_point in attachment_points:
                    tendon.add('site', site=attachment_point)

                self._tendons.append(tendon)

    def _get_tendon_gear(self, tendon: _ElementImpl) -> float:
        first_segment = int(tendon.site[0].site.name.split('_')[3]) if 'segment' in tendon.site[0].site.name else 0
        last_segment = int(tendon.site[-1].site.name.split('_')[3]) + 1

        segment_specifications = [self._arm_specification.segment_specifications[i] for i in
                                  range(first_segment, last_segment)]
        raddii = np.array([segment_spec.radius.value for segment_spec in segment_specifications])
        areas = np.pi * np.power(raddii, 2)
        lengths = np.array([segment_spec.length.value for segment_spec in segment_specifications])
        volumes = areas * lengths
        total_volume = np.sum(volumes)

        gear = total_volume * (1000 ** 2) * 2
        return gear

    def _configure_tendon_actuators(self) -> None:
        if self.morphology_specification.actuation_specification.use_tendons.value:
            self._configure_tendon_attachment_points()
            self._build_tendons()
            for tendon in self._tendons:
                gear = self._get_tendon_gear(tendon)
                self.mjcf_model.actuator.add('motor',
                                             name=f"{tendon.name}_motor",
                                             tendon=tendon,
                                             gear=[gear],
                                             forcelimited=True,
                                             forcerange=[-gear, 0],
                                             ctrllimited=True,
                                             ctrlrange=[-1, 0])

    def _configure_actuators(self) -> None:
        self._configure_tendon_actuators()
