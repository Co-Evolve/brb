from __future__ import annotations

import numpy as np
from dm_control import mjcf
from mujoco_utils.robot import MJCMorphology

from brb.seahorse.morphology.observables import SeahorseObservables
from brb.seahorse.morphology.parts.segment import SeahorseSegment
from brb.seahorse.morphology.specification.default import default_seahorse_morphology_specification
from brb.seahorse.morphology.specification.specification import SeahorseMorphologySpecification, \
    SeahorseTendonActuationSpecification
from brb.seahorse.morphology.utils import calculate_relaxed_tendon_length, get_actuator_ghost_taps_index, \
    get_actuator_tendon_plate_indices, get_all_tendon_start_and_stop_segment_indices
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
        self._configure_extensions()
        self._build_tail()
        self._configure_gliding_joint_equality_constraints()
        self._build_hmm_tendons()
        self._build_mvm_tendons()
        self._configure_actuators()
        self._prepare_tendon_coloring()
        self._configure_sensors()

    def _build_observables(
            self
            ) -> None:
        return SeahorseObservables(self)

    def _configure_compiler(
            self
            ) -> None:
        self.mjcf_model.compiler.angle = 'radian'
        # self.mjcf_model.option.flag.contact = 'disable'

    def _configure_extensions(self) -> None:
        sdf_plugin = self.mjcf_model.extension.add("plugin", plugin="mujoco.sdf.sdflib")
        sdf_instance = sdf_plugin.add("instance", name="sdf")
        sdf_instance.add("config", key="aabb", value="0")

    def _build_tail(
            self
            ) -> None:
        self._segments = []
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

    def _configure_gliding_joint_equality_constraints(
            self
            ) -> None:
        for segment, next_segment in zip(self._segments, self._segments[1:]):
            for plate, next_plate in zip(segment.plates, next_segment.plates):
                try:
                    self.mjcf_model.equality.add(
                            'joint',
                            joint1=plate.x_axis_gliding_joint,
                            joint2=next_plate.x_axis_gliding_joint,
                            polycoef=[0, 1, 0, 0, 0]
                            )
                except AttributeError:
                    pass
                try:
                    self.mjcf_model.equality.add(
                            'joint',
                            joint1=plate.y_axis_gliding_joint,
                            joint2=next_plate.y_axis_gliding_joint,
                            polycoef=[0, 1, 0, 0, 0]
                            )
                except AttributeError:
                    pass

    def _build_hmm_tendons(
            self
            ) -> None:
        self._hmm_tendons = []

        hmm_tendon_actuation_specification = self.tendon_actuation_specification.hmm_tendon_actuation_specification
        tendon_segment_span = hmm_tendon_actuation_specification.segment_span.value
        start_and_stop_indices = get_all_tendon_start_and_stop_segment_indices(
                total_num_segments=self.morphology_specification.num_segments, segment_span=tendon_segment_span
                )

        for x_side in ["ventral", "dorsal"]:
            for y_side in ["dextral", "sinistral"]:
                for start_index, stop_index in start_and_stop_indices:
                    taps = []
                    morphology_parts = []

                    # todo: add a start tap that is outside of manipulator (i.e. at motor position)
                    # Route through previous segments
                    for segment_index in range(start_index + 1):
                        plate_index, _ = get_actuator_tendon_plate_indices(side=x_side, segment_index=segment_index)
                        plate = self._segments[segment_index].plates[plate_index]
                        if segment_index < start_index:
                            ghost_hmm_taps = plate.lower_ghost_hmm_taps
                        else:
                            ghost_hmm_taps = plate.upper_ghost_hmm_taps

                        ghost_taps_index = get_actuator_ghost_taps_index(
                                x_side=x_side, y_side=y_side, segment_index=segment_index
                                )
                        taps += ghost_hmm_taps[ghost_taps_index]
                        morphology_parts += [plate, plate]

                    end_plate_index = self.morphology_specification.corners.index(f"{x_side}_{y_side}")
                    end_plate = self._segments[stop_index].plates[end_plate_index]

                    # Route through intermediate segments
                    for intermediate_tap_index, segment_index in enumerate(range(start_index + 1, stop_index)):
                        plate = self._segments[segment_index].plates[end_plate_index]
                        taps += plate.intermediate_hmm_taps[tendon_segment_span - 2 - intermediate_tap_index]
                        morphology_parts += [plate, plate]

                    # Set end point
                    taps += end_plate.hmm_tap_end
                    morphology_parts += [end_plate, end_plate]
                    base_length = calculate_relaxed_tendon_length(
                            morphology_parts=morphology_parts, attachment_points=taps
                            )

                    num_outer_tendons = 1 + min(start_index, hmm_tendon_actuation_specification.segment_span.value)
                    tendon_translation = num_outer_tendons * hmm_tendon_actuation_specification.tendon_strain.value

                    tendon = self.mjcf_model.tendon.add(
                            'spatial',
                            name=f"hmm_tendon_{x_side}_{y_side}_{start_index}-{stop_index}",
                            width=hmm_tendon_actuation_specification.tendon_width.value,
                            rgba=colors.rgba_blue,
                            limited=True,
                            range=[base_length - tendon_translation, 10 * base_length],
                            damping=hmm_tendon_actuation_specification.damping.value
                            )

                    for i, tap in enumerate(taps):
                        tendon.add('site', site=tap)

                    self._hmm_tendons.append(tendon)

    def _build_mvm_tendons(
            self
            ) -> None:
        self._mvm_tendons = []
        if not self.tendon_actuation_specification.mvm_tendon_actuation_specification.enabled.value:
            return

        mvm_tendon_actuation_specification = self.tendon_actuation_specification.mvm_tendon_actuation_specification
        for segment, next_segment in zip(self._segments, self._segments[1:]):
            start_plate = segment.plates[1]
            end_plate = next_segment.plates[1]

            start_tap = start_plate.mvm_taps[1]
            end_tap = end_plate.mvm_taps[0]
            taps = [start_tap, end_tap]

            base_length = calculate_relaxed_tendon_length(
                    morphology_parts=[start_plate, end_plate], attachment_points=taps
                    )

            tendon = self.mjcf_model.tendon.add(
                    'spatial',
                    name=f"mvm_tendon_{segment.segment_index, next_segment.segment_index}",
                    width=mvm_tendon_actuation_specification.tendon_width.value,
                    rgba=colors.rgba_blue,
                    limited=True,
                    range=[base_length * mvm_tendon_actuation_specification.contraction_factor.value,
                           base_length * mvm_tendon_actuation_specification.relaxation_factor.value],
                    damping=mvm_tendon_actuation_specification.damping.value
                    )
            for tap in taps:
                tendon.add('site', site=tap)

            self._mvm_tendons.append(tendon)

    def _configure_actuators(
            self
            ) -> None:
        self._tendon_actuators = []
        for tendon in self._hmm_tendons + self._mvm_tendons:
            if "mvm" in tendon.name:
                kp = self.tendon_actuation_specification.mvm_tendon_actuation_specification.p_control_kp.value
            else:
                kp = self.tendon_actuation_specification.hmm_tendon_actuation_specification.p_control_kp.value
            self._tendon_actuators.append(
                    self.mjcf_model.actuator.add(
                            'position',
                            tendon=tendon,
                            name=tendon.name,
                            forcelimited=True,
                            # only allow contraction forces
                            forcerange=[-kp, 0],
                            ctrllimited=True,
                            ctrlrange=tendon.range,
                            kp=kp
                            )
                    )

    def _configure_sensors(
            self
            ) -> None:
        for tendon in self._hmm_tendons + self._mvm_tendons:
            self.mjcf_model.sensor.add("tendonpos", name=f"{tendon.name}_position_sensor", tendon=tendon)

    def _prepare_tendon_coloring(
            self
            ) -> None:
        self._contracted_rgbas = np.ones((len(self._tendon_actuators), 4))
        self._contracted_rgbas[:] = colors.rgba_tendon_contracted

        self._color_changes = np.ones((len(self._tendon_actuators), 4))
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

        actuated_tendons = [tendon_actuator.tendon for tendon_actuator in self._tendon_actuators]
        physics.bind(actuated_tendons).rgba = self._contracted_rgbas + (tendon_control * self._color_changes).T

    def after_step(
            self,
            physics,
            random_state
            ) -> None:
        self._color_muscles(physics=physics)


if __name__ == '__main__':
    morphology_specification = default_seahorse_morphology_specification(
        num_segments=10, hmm_segment_span=1, mvm_enabled=True
        )
    morphology = MJCSeahorseMorphology(specification=morphology_specification)
    morphology.mjcf_body.euler = [np.pi, 0.0, 0.0]
    morphology.export_to_xml_with_assets("./mjcf")
