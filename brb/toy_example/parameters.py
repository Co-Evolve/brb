from typing import List

from fprs.parameters import ContinuousParameter
from fprs.specification import MorphologySpecificationParameterizer

from brb.toy_example.morphology.specification.default import default_toy_example_morphology_specification
from brb.toy_example.morphology.specification.specification import ToyExampleMorphologySpecification


class ToyExampleMorphologySpecificationParameterizer(MorphologySpecificationParameterizer):
    def __init__(
            self,
            torso_radius_range: tuple[float, float]
            ) -> None:
        super().__init__()
        self._torso_radius_range = torso_radius_range

    def parameterize_specification(
            self,
            specification: ToyExampleMorphologySpecification
            ) -> None:
        # Passing None as value will set the parameter to a random value within the given range
        morphology_specification.torso_specification.radius = ContinuousParameter(
                low=self._torso_radius_range[0], high=self._torso_radius_range[1], value=None
                )

    def get_parameter_labels(
            self,
            specification: ToyExampleMorphologySpecification
            ) -> List[str]:
        return ["torso-radius"]


if __name__ == '__main__':
    morphology_specification = default_toy_example_morphology_specification(
            num_arms=4, num_segments_per_arm=3
            )

    parameterizer = ToyExampleMorphologySpecificationParameterizer(torso_radius_range=(0.05, 0.15))
    parameterizer.parameterize_specification(specification=morphology_specification)

    print("All parameters:")
    print(f"\t{morphology_specification.parameters}")
    print()
    print("Parameters to optimise:")
    for parameter, label in zip(
            parameterizer.get_target_parameters(specification=morphology_specification),
            parameterizer.get_parameter_labels(specification=morphology_specification)
            ):
        print(f"\t{label}\t->\t{parameter}")
