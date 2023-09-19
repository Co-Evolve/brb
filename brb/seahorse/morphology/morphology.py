from __future__ import annotations

from typing import List

import numpy as np
from dm_control import mjcf
from mujoco_utils.robot import MJCMorphology

from brb.seahorse.morphology.parts.segment import SeahorseSegment
from brb.seahorse.morphology.specification.default import default_seahorse_morphology_specification
from brb.seahorse.morphology.specification.specification import SeahorseMorphologySpecification, \
    SeahorseTendonActuationSpecification
from brb.seahorse.morphology.utils import calculate_relaxed_tendon_length, get_actuator_tendon_plate_indices, \
    get_all_tendon_start_and_stop_segment_indices
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
        self._prepare_tendon_coloring()

    def _configure_compiler(
            self
            ) -> None:
        self.mjcf_model.compiler.angle = 'radian'
        self.mjcf_model.option.flag.contact = 'disable'

    def _build_tail(
            self
            ) -> None:
        self._segments: List[SeahorseSegment] = []
        num_segments = self.morphology_specification.num_segments
        next_parent = self
        for segment_index in range(num_segments):
            position = np.array(
                    [0.0, 0.0, self.morphology_specification.segment_specifications[
                        segment_index].z_offset_from_previous_segment.value]
                    )
            segment = SeahorseSegment(
                    parent=next_parent,
                    name=f"segment_{segment_index}",
                    pos=position,
                    euler=np.zeros(3),
                    segment_index=segment_index
                    )
            self._segments.append(segment)

            next_parent = segment.vertebrae

    def _build_tendons(
            self
            ) -> None:
        self._tendons = []
        self._tendon_to_base_length = {}

        start_and_stop_indices = get_all_tendon_start_and_stop_segment_indices(
                total_num_segments=self.morphology_specification.num_segments,
                segment_span=self.tendon_actuation_specification.segment_span.value
                )

        start_and_stop_indices_to_length = {}
        for x_side in ["ventral", "dorsal"]:
            for y_side in ["dextral", "sinistral"]:
                for start_index, stop_index in start_and_stop_indices:
                    a_taps = []
                    morphology_parts = []

                    # todo: add a start a_tap that is outside of manipulator (i.e. at motor position)
                    # Route through previous segments
                    for segment_index in range(start_index + 1):
                        plate_index, _ = get_actuator_tendon_plate_indices(side=x_side, segment_index=segment_index)
                        plate = self._segments[segment_index].plates[plate_index]
                        if y_side == "sinistral":
                            a_taps += plate.a_taps_sinistral
                        else:
                            a_taps += plate.a_taps_dextral
                        morphology_parts.append(plate)
                        morphology_parts.append(plate)

                    start_world_pos = plate.world_coordinates_of_point(a_taps[-1].pos)

                    end_plate_index = self.morphology_specification.corners.index(f"{x_side}_{y_side}")
                    end_plate = self._segments[stop_index].plates[end_plate_index]
                    end_a_tap = end_plate.a_tap_end
                    end_world_pos = end_plate.world_coordinates_of_point(end_a_tap.pos)

                    # Route through intermediate segments
                    for segment_index in range(start_index + 1, stop_index):
                        vertebrae = self._segments[segment_index].vertebrae
                        a_taps.append(
                                vertebrae.add_intermediate_a_tap(
                                        identifier=f"{x_side}-{y_side}_{start_index}-{stop_index}",
                                        start_world_pos=start_world_pos,
                                        stop_world_pos=end_world_pos
                                        )
                                )
                        morphology_parts.append(vertebrae)

                    # Set end point
                    a_taps.append(end_plate.a_tap_end)
                    morphology_parts.append(end_plate)
                    if (start_index, stop_index) not in start_and_stop_indices_to_length:
                        start_and_stop_indices_to_length[(start_index, stop_index)] = calculate_relaxed_tendon_length(
                                morphology_parts=morphology_parts, attachment_points=a_taps
                                )
                    base_length = start_and_stop_indices_to_length[(start_index, stop_index)]

                    tendon = self.mjcf_model.tendon.add(
                            'spatial',
                            name=f"tendon_{x_side}_{y_side}_{start_index}-{stop_index}",
                            width=self.tendon_actuation_specification.tendon_width.value,
                            rgba=colors.rgba_blue,
                            limited=True,
                            range=[base_length * self.tendon_actuation_specification.contraction_factor.value,
                                   base_length * self.tendon_actuation_specification.relaxation_factor.value],
                            damping=self.tendon_actuation_specification.damping.value
                            )
                    self._tendon_to_base_length[tendon.name] = base_length

                    for i, tap in enumerate(a_taps):
                        tendon.add('site', site=tap)

                    self._tendons.append(tendon)
                    print(f"Tendon {tendon.name}\t->\tlength: {base_length}")

    def _configure_actuators(
            self
            ) -> None:
        kp = self.tendon_actuation_specification.p_control_kp.value
        self._tendon_actuators = []
        for tendon in self._tendons:
            self._tendon_actuators.append(
                    self.mjcf_model.actuator.add(
                            'position',
                            tendon=tendon,
                            name=tendon.name,
                            forcelimited=True,
                            forcerange=[-kp, 0],    # only allow contraction forces
                            ctrllimited=True,
                            ctrlrange=tendon.range,
                            kp=kp
                            )
                    )

    def _prepare_tendon_coloring(
            self
            ) -> None:
        self._contracted_rgbas = np.ones((len(self._tendons), 4))
        self._contracted_rgbas[:] = colors.rgba_tendon_contracted

        self._color_changes = np.ones((len(self._tendons), 4))
        self._color_changes[:] = colors.rgba_tendon_relaxed - colors.rgba_tendon_contracted
        self._color_changes = self._color_changes.T
        self._control_ranges = np.array([act.ctrlrange for act in self._tendon_actuators])

    def _color_muscles(
            self,
            physics: mjcf.Physics
            ) -> None:
        tendon_control = np.array(physics.bind(self._tendon_actuators).ctrl)
        # Normalize to 0, 1
        min_control, max_control = self._control_ranges[:, 0], self._control_ranges[:, 1]
        tendon_control = (tendon_control - min_control) / (max_control - min_control)

        physics.bind(self._tendons).rgba = self._contracted_rgbas + (tendon_control * self._color_changes).T

    def after_step(
            self,
            physics,
            random_state
            ) -> None:
        self._color_muscles(physics=physics)


if __name__ == '__main__':
    morphology_specification = default_seahorse_morphology_specification(num_segments=15)
    morphology = MJCSeahorseMorphology(specification=morphology_specification)
    morphology.export_to_xml_with_assets("./mjcf")
