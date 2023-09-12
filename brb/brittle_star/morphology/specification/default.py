import numpy as np

from brb.brittle_star.morphology.morphology import MJCBrittleStarMorphology
from brb.brittle_star.morphology.specification.specification import BrittleStarActuationSpecification, \
    BrittleStarArmSegmentSpecification, BrittleStarArmSpecification, BrittleStarDiscSpecification, \
    BrittleStarJointSpecification, BrittleStarMorphologySpecification

START_SEGMENT_RADIUS = 0.025
STOP_SEGMENT_RADIUS = 0.0125
START_SEGMENT_LENGTH = 0.075
STOP_SEGMENT_LENGTH = 0.025
DISC_DIAMETER = 0.25
DISC_HEIGHT = 0.025


def linear_interpolation(
        alpha: float,
        start: float,
        stop: float
        ) -> float:
    return start + alpha * (stop - start)


def default_joint_specification(
        range: float
        ) -> BrittleStarJointSpecification:
    joint_specification = BrittleStarJointSpecification(
            range=range, stiffness=10.0, damping_factor=0.25
            )

    return joint_specification


def default_arm_segment_specification(
        alpha: float
        ) -> BrittleStarArmSegmentSpecification:
    in_plane_joint_specification = default_joint_specification(range=30 / 180 * np.pi)  # 30
    out_of_plane_joint_specification = default_joint_specification(range=30 / 180 * np.pi)  # 5

    radius = linear_interpolation(alpha=alpha, start=START_SEGMENT_RADIUS, stop=STOP_SEGMENT_RADIUS)
    length = linear_interpolation(alpha=alpha, start=START_SEGMENT_LENGTH, stop=STOP_SEGMENT_RADIUS)
    tendon_offset = 1.2
    num_touch_sensors = 12

    segment_specification = BrittleStarArmSegmentSpecification(
            radius=radius,
            length=length,
            tendon_offset=tendon_offset,
            num_touch_sensors=num_touch_sensors,
            in_plane_joint_specification=in_plane_joint_specification,
            out_of_plane_joint_specification=out_of_plane_joint_specification
            )
    return segment_specification


def default_arm_specification(
        num_segments_per_arm: int
        ) -> BrittleStarArmSpecification:
    segment_specifications = list()
    for segment_index in range(num_segments_per_arm):
        segment_specification = default_arm_segment_specification(alpha=segment_index / num_segments_per_arm)
        segment_specifications.append(segment_specification)

    arm_specification = BrittleStarArmSpecification(
            segment_specifications=segment_specifications
            )
    return arm_specification


def default_brittle_star_morphology_specification(
        num_arms: int = 5,
        num_segments_per_arm: int = 5,
        use_p_control: bool = False
        ) -> BrittleStarMorphologySpecification:
    disc_specification = BrittleStarDiscSpecification(diameter=DISC_DIAMETER, height=DISC_HEIGHT, pentagon=True)

    arm_specifications = list()
    for arm_index in range(num_arms):
        arm_specification = default_arm_specification(num_segments_per_arm=num_segments_per_arm)
        arm_specifications.append(arm_specification)

    actuation_specification = BrittleStarActuationSpecification(
        use_tendons=not use_p_control,
        use_p_control=use_p_control, )
    specification = BrittleStarMorphologySpecification(
        disc_specification=disc_specification,
        arm_specifications=arm_specifications,
        actuation_specification=actuation_specification
        )

    return specification


def default_arm_length_based_brittle_star_morphology_specification(
        num_arms: int = 5,
        arm_length_in_disc_diameters: float = 3,
        use_p_control: bool = False
        ) -> BrittleStarMorphologySpecification:
    target_arm_length = DISC_DIAMETER * arm_length_in_disc_diameters

    length_per_num_segments = []
    for num_segments in range(3, 100, 3):
        total_length = 0
        for segment_index in range(num_segments):
            alpha = segment_index / num_segments
            radius = linear_interpolation(alpha=alpha, start=START_SEGMENT_RADIUS, stop=STOP_SEGMENT_RADIUS)
            length = linear_interpolation(alpha=alpha, start=START_SEGMENT_LENGTH, stop=STOP_SEGMENT_RADIUS)
            total_length += length + 2 * radius

        length_per_num_segments.append(total_length)
    length_per_num_segments = np.array(length_per_num_segments)

    num_segments = np.argmin(np.abs(length_per_num_segments - target_arm_length)) * 3 + 3

    return default_brittle_star_morphology_specification(
        num_arms=num_arms, num_segments_per_arm=num_segments, use_p_control=use_p_control
        )


if __name__ == '__main__':
    morph_spec = default_arm_length_based_brittle_star_morphology_specification(
        num_arms=5, arm_length_in_disc_diameters=4, use_p_control=False
        )
    MJCBrittleStarMorphology(specification=morph_spec).export_to_xml_with_assets("./shifted")
