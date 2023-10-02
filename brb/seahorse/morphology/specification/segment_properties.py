from typing import Tuple

import numpy as np

# T. Praet's measurements
ORIGINAL_TAIL_LENGTH = 63.080  # in mm
ORIGINAL_NUM_SEGMENTS = 29


def segment_properties(
        skip_n: int = 0
        ) -> Tuple[np.ndarray, ...]:
    tail_length = ORIGINAL_TAIL_LENGTH
    orig_num_segments = ORIGINAL_NUM_SEGMENTS

    segment_pos_to_height = lambda \
        l: (-2.086 * l ** 2 - 1.896 * l + 5.233)
    segment_pos_to_length = lambda \
        l: (-2.619 * l ** 3 + 1.613 * l ** 2 - 1.13 * l + 2.851)
    segment_pos_to_inclination = lambda \
        x: (28.396 * x + 4.048) / 180 * np.pi

    segment_heights = np.zeros(orig_num_segments)
    segment_lengths = np.zeros(orig_num_segments)
    segment_inclinations = np.zeros(orig_num_segments)

    current_total_l = 0
    for segment_index in range(orig_num_segments):
        relative_l = current_total_l / tail_length

        segment_heights[segment_index] = segment_pos_to_height(relative_l)
        segment_lengths[segment_index] = segment_pos_to_length(relative_l)
        segment_inclinations[segment_index] = segment_pos_to_inclination(relative_l)

        current_total_l += segment_lengths[segment_index]

    segment_widths = segment_heights

    skip_n += 1
    return segment_heights[::skip_n], segment_widths[::skip_n], segment_lengths[::skip_n], segment_inclinations[
                                                                                           ::skip_n]
