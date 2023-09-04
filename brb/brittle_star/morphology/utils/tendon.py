from typing import List

import numpy as np
from dm_control import mjcf
from mujoco_utils.robot import MJCMorphologyPart


def calculate_relaxed_tendon_length(
        morphology_parts: List[MJCMorphologyPart],
        attachment_points: List[mjcf.Element]
        ) -> float:
    relaxed_tendon_length = 0
    for current_index in range(len(attachment_points) - 1):
        next_index = current_index + 1
        current_part = morphology_parts[current_index]
        next_part = morphology_parts[next_index]

        current_attachment_point = attachment_points[current_index]
        next_attachment_point = attachment_points[next_index]

        current_position = current_part.world_coordinates_of_point(current_attachment_point.pos)
        next_position = next_part.world_coordinates_of_point(next_attachment_point.pos)

        distance_between_points = np.linalg.norm(next_position - current_position)
        relaxed_tendon_length += distance_between_points

    return relaxed_tendon_length
