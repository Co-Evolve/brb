from brb.brittle_star.morphology.specification.specification import BrittleStarMorphologySpecification


def calculate_arm_length(specification: BrittleStarMorphologySpecification) -> float:
    segment_specifications = specification.arm_specifications[0].segment_specifications

    arm_length = 0
    for segment_specification in segment_specifications:
        arm_length += 2 * segment_specification.radius.value + segment_specification.length.value
    return arm_length


def calculate_arm_length_in_disc_diameter(specification: BrittleStarMorphologySpecification) -> float:
    arm_length = calculate_arm_length(specification)
    disc_diameter = 2 * specification.disc_specification.radius.value

    return arm_length / disc_diameter
