from fprs.parameters import ContinuousParameter, FixedParameter

from brb.toy_example.morphology.specification.default import default_toy_example_morphology_specification

if __name__ == '__main__':
    morphology_specification = default_toy_example_morphology_specification(
            num_arms=4, num_segments_per_arm=3
            )

    # Passing None as value will set the parameter to a random value within the given range
    morphology_specification.torso_specification.radius = ContinuousParameter(
            low=0.1, high=0.6, value=None
            )

    all_parameters = morphology_specification.parameters
    non_fixed_parameters = list(
            filter(
                    lambda
                        parameter: not isinstance(parameter, FixedParameter), all_parameters
                    )
            )
    print("All parameters:")
    print(f"\t{all_parameters}")
    print()
    print("Parameters to optimise:")
    print(f"\t{non_fixed_parameters}")
