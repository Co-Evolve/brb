from typing import Union

import numpy as np
from mujoco_utils.morphology import MJCFMorphology, MJCFMorphologyPart

from brb.toy_example.morphology.parts.arm_segment import MJCFToyExampleArmSegment
from brb.toy_example.morphology.specification.specification import ToyExampleMorphologySpecification


class MJCFToyExampleArm(MJCFMorphologyPart):
    def __init__(
            self,
            parent: Union[MJCFMorphology, MJCFMorphologyPart],
            name: str,
            pos: np.array,
            euler: np.array,
            *args,
            **kwargs
            ) -> None:
        super().__init__(parent, name, pos, euler, *args, **kwargs)

    @property
    def morphology_specification(
            self
            ) -> ToyExampleMorphologySpecification:
        return super().morphology_specification

    def _build(
            self,
            arm_index: int
            ) -> None:
        self._arm_index = arm_index
        self._arm_specification = self.morphology_specification.arm_specifications[self._arm_index]

        self._build_segments()

    def _build_segments(
            self
            ) -> None:
        self._segments = []

        number_of_segments = len(self._arm_specification.segment_specifications)

        for segment_index in range(number_of_segments):
            try:
                parent = self._segments[-1]
                position = 2 * self._segments[-1].center_of_capsule
            except IndexError:
                position = np.zeros(3)
                parent = self

            segment = MJCFToyExampleArmSegment(
                    parent=parent,
                    name=f"{self.base_name}_segment_{segment_index}",
                    pos=position,
                    euler=np.zeros(3),
                    arm_index=self._arm_index,
                    segment_index=segment_index
                    )
            self._segments.append(segment)
