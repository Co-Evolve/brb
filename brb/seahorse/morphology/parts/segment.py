from typing import List, Union, cast

import numpy as np
from mujoco_utils.robot import MJCMorphology, MJCMorphologyPart

from brb.seahorse.morphology.parts.plate import SeahorsePlate
from brb.seahorse.morphology.parts.vertebrae import SeahorseVertebrae
from brb.seahorse.morphology.specification.specification import SeahorseMorphologySpecification, \
    SeahorseSegmentSpecification, SeahorseTendonSpineSpecification, SeahorseVertebraeSpecification
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
            ):
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
            angle = -np.pi / 4 + (plate_index * np.pi / 2)
            offset_from_vertebrae = self.segment_specification.plate_specifications[plate_index].offset_from_vertebrae
            x = offset_from_vertebrae * np.cos(angle)
            y = offset_from_vertebrae * np.sin(angle)

            plate = SeahorsePlate(
                    parent=self.vertebrae,
                    name=f"{self.base_name}_plate_{plate_index}",
                    pos=np.array([x, y, 0.0]),
                    euler=np.zeros(3),
                    segment_index=self.segment_index,
                    plate_index=plate_index
                    )
            self.plates.append(plate)

    def _build_spines(
            self
            ) -> None:
        sides = self.morphology_specification.sides

        self.spines = []
        vertebrae_s_taps = self.vertebrae.s_taps
        if self.segment_index % 2 == 0:
            plate_s_taps = [self.plates[1].s_taps[0], self.plates[2].s_taps[0], self.plates[3].s_taps[0],
                            self.plates[3].s_taps[1]]
        else:
            plate_s_taps = [self.plates[1].s_taps[0], self.plates[2].s_taps[0], self.plates[2].s_taps[1],
                            self.plates[3].s_taps[0]]

        for side, vertebrae_s_tap, plate_s_tap in zip(sides, vertebrae_s_taps, plate_s_taps):
            tendon = self.mjcf_model.tendon.add(
                    'spatial',
                    name=f"{self.base_name}_vertebral_spine_{side}",
                    width=self.tendon_spine_specification.tendon_width,
                    rgba=colors.rgba_red,
                    stiffness=self.tendon_spine_specification.stiffness,
                    damping=self.tendon_spine_specification.damping
                    )
            tendon.add(
                'site', site=vertebrae_s_tap
                )
            tendon.add(
                'site', site=plate_s_tap
                )

    def _configure_plate_gliding_joints_equality_constraints(
            self
            ) -> None:
        x_aligned_plate_neighbours = [(0, 1), (2, 3)]
        y_aligned_plate_neighbours = [(0, 3), (1, 2)]

        for x_aligned_plates in x_aligned_plate_neighbours:
            plate_index_1, plate_index_2 = x_aligned_plates
            self.mjcf_model.equality.add(
                    'joint',
                    joint1=self.plates[plate_index_1].x_axis_gliding_joint,
                    joint2=self.plates[plate_index_2].x_axis_gliding_joint,
                    polycoef=[0, 1, 0, 0, 0]
                    )
            self.mjcf_model.equality.add(
                    'joint',
                    joint1=self.plates[plate_index_1].y_axis_gliding_joint,
                    joint2=self.plates[plate_index_2].y_axis_gliding_joint,
                    polycoef=[0, -1, 0, 0, 0]
                    )
        for y_aligned_plates in y_aligned_plate_neighbours:
            plate_index_1, plate_index_2 = y_aligned_plates
            self.mjcf_model.equality.add(
                    'joint',
                    joint1=self.plates[plate_index_1].x_axis_gliding_joint,
                    joint2=self.plates[plate_index_2].x_axis_gliding_joint,
                    polycoef=[0, -1, 0, 0, 0]
                    )
            self.mjcf_model.equality.add(
                    'joint',
                    joint1=self.plates[plate_index_1].y_axis_gliding_joint,
                    joint2=self.plates[plate_index_2].y_axis_gliding_joint,
                    polycoef=[0, 1, 0, 0, 0]
                    )

    def get_next_segment_position(
            self
            ) -> np.ndarray:
        return np.array([0.0, 0.0, 2 * self.vertebrae_specification.vertebrae_half_height + 0.001])
