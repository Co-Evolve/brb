from typing import List, Tuple

import numpy as np
from dm_control import mjcf
from mujoco_utils.robot import MJCMorphologyPart

from brb.seahorse.morphology.specification.specification import SeahorsePlateSpecification


def is_inner_plate_x_axis(
        segment_index: int,
        plate_index: int
        ) -> bool:
    if plate_index == 0:
        return False
    if plate_index == 1:
        return True
    if segment_index % 2 == 0:
        if plate_index == 2:
            return True
        else:
            return False
    else:
        if plate_index == 2:
            return False
        else:
            return True


def is_inner_plate_y_axis(
        segment_index: int,
        plate_index: int
        ) -> bool:
    if plate_index < 2:
        return False
    else:
        return True


def get_plate_indices_per_side(
        side: str
        ) -> Tuple[int, int]:
    if side == "ventral":
        return (0, 1)
    elif side == "dextral":
        return (0, 3)
    elif side == "sinistral":
        return (1, 2)
    elif side == "dorsal":
        return (2, 3)


def get_inner_and_outer_plate_indices_per_side(
        segment_index: int,
        side: str
        ) -> Tuple[int, int]:
    plates = get_plate_indices_per_side(side=side)

    if side == "ventral" or side == "dorsal":
        is_inner_check = is_inner_plate_x_axis
    else:
        is_inner_check = is_inner_plate_y_axis

    inner_plate, outer_plate = sorted(
            plates,
            key=lambda
                plate_index: not is_inner_check(
                    segment_index=segment_index, plate_index=plate_index
                    )
            )
    return inner_plate, outer_plate


def get_actuator_tendon_plate_indices(
        side: str,
        segment_index: str
        ) -> Tuple[int, int]:
    if side == "ventral":
        return 1, 0
    elif side == "dorsal":
        if segment_index % 2 == 0:
            return 2, 3
        else:
            return 3, 2


def get_plate_position(
        plate_index: int,
        plate_specification: SeahorsePlateSpecification
        ) -> np.ndarray:
    angles = [5 * np.pi / 4, 7 * np.pi / 4, np.pi / 4, 3 * np.pi / 4]
    angle = angles[plate_index]
    offset_from_vertebrae = plate_specification.offset_from_vertebrae.value
    position = offset_from_vertebrae * np.array([-np.sin(angle), np.cos(angle), 0.0])
    position[2] = plate_specification.depth.value / 2
    return position


def get_all_tendon_start_and_stop_segment_indices(
        total_num_segments: int,
        segment_span: int
        ) -> List[Tuple[int, int]]:
    start_and_stop_indices = []

    for segment_index in reversed(range(total_num_segments)):
        if segment_index - segment_span >= 0:
            start_and_stop_indices.append((segment_index - segment_span, segment_index))

    return start_and_stop_indices


def calculate_relaxed_tendon_length(morphology_parts: List[MJCMorphologyPart],
                                    attachment_points: List[mjcf.Element]) -> float:
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