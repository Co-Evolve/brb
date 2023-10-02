from typing import List

import numpy as np


def get_max_number_of_routing_holes(
        num_segments: int,
        tendon_segment_span: int
        ) -> List[int]:
    return num_segments - tendon_segment_span


def get_intermediate_tendon_point_locations(num_segments: int,
                                            tendon_segment_span: int) -> np.ndarray:
    pass


if __name__ == '__main__':
    print(get_max_number_of_routing_holes(
        num_segments=20, tendon_segment_span=3
        ))
