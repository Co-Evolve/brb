from __future__ import annotations

from typing import List

import numpy as np
from mujoco_utils.robot import MJCMorphology

from brb.seahorse.morphology.parts.segment import SeahorseSegment
from brb.seahorse.morphology.specification.default import default_seahorse_morphology_specification
from brb.seahorse.morphology.specification.specification import SeahorseMorphologySpecification, \
    SeahorseTendonActuationSpecification
from brb.utils import colors


class MJCSeahorseMorphology(MJCMorphology):
    def __init__(
            self,
            specification: SeahorseMorphologySpecification
            ) -> None:
        super().__init__(specification)

    @property
    def morphology_specification(
            self
            ) -> SeahorseMorphologySpecification:
        return super().morphology_specification

    @property
    def tendon_actuation_specification(
            self
            ) -> SeahorseTendonActuationSpecification:
        return self.morphology_specification.tendon_actuation_specification

    def _build(
            self
            ) -> None:
        self._configure_compiler()
        self._build_tail()
        self._build_tendons()
        self._configure_actuators()

    def _configure_compiler(
            self
            ) -> None:
        self.mjcf_model.compiler.angle = 'radian'  # Use radians
        self.mjcf_model.option.flag.contact = 'disable'

    def _build_tail(
            self
            ) -> None:
        self._segments: List[SeahorseSegment] = []
        num_segments = self.morphology_specification.num_segments
        next_parent = self
        next_pos = [0.0, 0.0, 0.0]
        for segment_index in range(num_segments):
            segment = SeahorseSegment(
                    parent=next_parent,
                    name=f"segment_{segment_index}",
                    pos=next_pos,
                    euler=np.zeros(3),
                    segment_index=segment_index
                    )
            self._segments.append(segment)

            if segment_index < num_segments - 1:
                next_parent = segment.vertebrae
                next_pos = segment.get_next_segment_position()

    def _build_tendons(
            self
            ) -> None:
        self.tendons = []
        sides = self.morphology_specification.sides
        contraction_factor = self.tendon_actuation_specification.contraction_factor
        relaxation_factor = self.tendon_actuation_specification.relaxation_factor
        tendon_width = self.tendon_actuation_specification.tendon_width

        for side_index, side in enumerate(sides):
            plates = [segment.plates[side_index] for segment in self._segments]

            taps = []
            tap_coords = []
            for plate in plates:
                taps.append(plate.a_tap_bottom)
                taps.append(plate.a_tap_top)
                tap_coords.append(plate.world_coordinates_of_point(plate.a_tap_bottom.pos))
                tap_coords.append(plate.world_coordinates_of_point(plate.a_tap_top.pos))

            base_length = 0
            for previous_coord, next_coord in zip(tap_coords, tap_coords[1:]):
                base_length += np.linalg.norm(next_coord - previous_coord)

            tendon = self.mjcf_model.tendon.add(
                    'spatial',
                    name=f"tendon_{side}",
                    width=tendon_width,
                    rgba=colors.rgba_blue,
                    limited=True,
                    range=[base_length * contraction_factor, base_length * relaxation_factor], )
            for tap in taps:
                tendon.add('site', site=tap)

            self.tendons.append(tendon)

    def _configure_actuators(
            self
            ) -> None:
        kp = self.tendon_actuation_specification.p_control_kp
        for tendon in self.tendons:
            self.mjcf_model.actuator.add(
                    'position',
                    tendon=tendon,
                    name=tendon.name,
                    forcelimited=True,
                    forcerange=[-kp, 0],
                    ctrllimited=True,
                    ctrlrange=tendon.range,
                    kp=kp
                    )


if __name__ == '__main__':
    morphology_specification = default_seahorse_morphology_specification(num_segments=7)
    morphology = MJCSeahorseMorphology(specification=morphology_specification)
    morphology.export_to_xml_with_assets("./mjcf")
