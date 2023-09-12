import copy

import numpy as np

import brb
from brb.brittle_star.morphology.specification.specification import BrittleStarMorphologySpecification


def remove_random_segment_groups(
        base_specification: BrittleStarMorphologySpecification,
        arm_shortening_probability: float = 0.1,
        min_num_segments_per_arm: int = 0,
        max_num_segments_per_arm: int = 12,
        num_segments_per_group: int = 3
        ) -> BrittleStarMorphologySpecification:
    adapted_specification = copy.deepcopy(base_specification)

    num_segments_per_arm_possibilities = [min_num_segments_per_arm + num_segments_per_group * i for i in range(
            (max_num_segments_per_arm - min_num_segments_per_arm) // num_segments_per_group + 1
            )]

    for arm_index in range(base_specification.number_of_arms):
        # If we're at the last arm and all previous arms have been completely removed -> do not allow 0 segments on
        # this last arm
        if arm_index == (base_specification.number_of_arms - 1) and np.sum(
                adapted_specification.number_of_segments_per_arm[:-1]
                ) == 0:
            num_segments_per_arm_possibilities = num_segments_per_arm_possibilities[1:]

        if brb.brb_random_state.rand() < arm_shortening_probability:
            new_num_segments = brb.brb_random_state.choice(num_segments_per_arm_possibilities)
            adapted_specification.arm_specifications[arm_index].segment_specifications = (
                    base_specification.arm_specifications[arm_index].segment_specifications[:new_num_segments])

    return adapted_specification
