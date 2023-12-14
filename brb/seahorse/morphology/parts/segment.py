from typing import List, Union, cast

import numpy as np
from mujoco_utils.robot import MJCMorphology, MJCMorphologyPart

from brb.seahorse.morphology.parts.plate import SeahorsePlate
from brb.seahorse.morphology.parts.vertebrae import SeahorseVertebrae
from brb.seahorse.morphology.specification.specification import SeahorseMorphologySpecification, \
    SeahorseSegmentSpecification, SeahorseTendonSpineSpecification, SeahorseVertebraeSpecification
from brb.seahorse.morphology.utils import get_inner_and_outer_plate_indices_per_side
from brb.utils import colors


class SeahorseSegment(MJCMorphologyPart):
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
        return cast(SeahorseMorphologySpecification, super().morphology_specification)

    @property
    def segment_specification(
            self
            ) -> SeahorseSegmentSpecification:
        return self.morphology_specification.segment_specifications[self.segment_index]

    @property
    def vertebrae_specification(
            self
            ) -> SeahorseVertebraeSpecification:
        return self.segment_specification.vertebrae_specification

    @property
    def tendon_spine_specification(
            self
            ) -> SeahorseTendonSpineSpecification:
        return self.segment_specification.tendon_spine_specification

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

        self._build_vertebrae()
        self._build_plates()
        self._build_spines()
        self._configure_plate_gliding_joints_equality_constraints()

    def _build_vertebrae(
            self
            ) -> None:
        self.vertebrae = SeahorseVertebrae(
                parent=self,
                name=f"{self.base_name}_vertebrae",
                pos=np.zeros(3),
                euler=np.zeros(3),
                segment_index=self.segment_index
                )

    def _build_plates(
            self
            ) -> None:
        self.plates: List[SeahorsePlate] = []

        for plate_index in range(4):
            plate = SeahorsePlate(
                    parent=self.vertebrae,
                    name=f"{self.base_name}_plate_{plate_index}",
                    pos=np.zeros(3),
                    euler=np.zeros(3),
                    segment_index=self.segment_index,
                    plate_index=plate_index
                    )
            self.plates.append(plate)

        for plate1 in self.plates:
            for plate2 in self.plates[1:]:
                self.mjcf_model.contact.add("exclude", name=f"{self.base_name}_{plate1.base_name}_"
                                                            f"{plate2.base_name}",
                                            body1=plate1.plate,
                                            body2=plate2.plate),

    def _build_spines(
            self
            ) -> None:
        sides = self.morphology_specification.sides

        self.spines = []

        for side, vertebrae_s_tap in zip(sides, self.vertebrae.s_taps):
            inner_plate_index, outer_plate_index = get_inner_and_outer_plate_indices_per_side(
                    segment_index=self.segment_index,
                    side=side)
            inner_plate = self.plates[inner_plate_index]
            outer_plate = self.plates[outer_plate_index]
            if side == "ventral" or side == "dorsal":
                axis = "x"
            else:
                axis = "y"

            inner_plate_s_tap = inner_plate.s_taps[f"plate-{axis}"]
            outer_plate_s_tap = outer_plate.s_taps[f"plate-{axis}"]
            connector_s_tap = inner_plate.s_taps[f"connector-{axis}"]

            tendon = self.mjcf_model.tendon.add(
                    'spatial',
                    name=f"{self.base_name}_vertebral_spine_{side}",
                    width=self.tendon_spine_specification.tendon_width,
                    rgba=colors.rgba_gray,
                    stiffness=self.tendon_spine_specification.stiffness,
                    damping=self.tendon_spine_specification.damping
                    )
            tendon.add(
                    'site', site=vertebrae_s_tap
                    )
            tendon.add(
                    'site', site=connector_s_tap
                    )
            tendon.add(
                    'site', site=inner_plate_s_tap
                    )
            tendon.add(
                    'site', site=outer_plate_s_tap
                    )

    def _configure_plate_gliding_joints_equality_constraints(
            self
            ) -> None:
        x_aligned_plate_neighbours = [(0, 1), (2, 3)]
        y_aligned_plate_neighbours = [(0, 3), (1, 2)]

        for x_aligned_plates in x_aligned_plate_neighbours:
            plate_index_1, plate_index_2 = x_aligned_plates
            try:
                self.mjcf_model.equality.add(
                        'joint',
                        joint1=self.plates[plate_index_1].x_axis_gliding_joint,
                        joint2=self.plates[plate_index_2].x_axis_gliding_joint,
                        polycoef=[0, 1, 0, 0, 0]
                        )
            except AttributeError:
                pass
            try:
                self.mjcf_model.equality.add(
                        'joint',
                        joint1=self.plates[plate_index_1].y_axis_gliding_joint,
                        joint2=self.plates[plate_index_2].y_axis_gliding_joint,
                        polycoef=[0, -1, 0, 0, 0]
                        )
            except AttributeError:
                pass
        for y_aligned_plates in y_aligned_plate_neighbours:
            plate_index_1, plate_index_2 = y_aligned_plates
            try:
                self.mjcf_model.equality.add(
                        'joint',
                        joint1=self.plates[plate_index_1].x_axis_gliding_joint,
                        joint2=self.plates[plate_index_2].x_axis_gliding_joint,
                        polycoef=[0, -1, 0, 0, 0]
                        )
            except AttributeError:
                pass
            try:
                self.mjcf_model.equality.add(
                        'joint',
                        joint1=self.plates[plate_index_1].y_axis_gliding_joint,
                        joint2=self.plates[plate_index_2].y_axis_gliding_joint,
                        polycoef=[0, 1, 0, 0, 0]
                        )
            except AttributeError:
                pass
