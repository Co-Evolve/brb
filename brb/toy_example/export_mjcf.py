from brb.toy_example.morphology.morphology import MJCToyExampleMorphology
from brb.toy_example.morphology.specification.default import default_toy_example_morphology_specification

if __name__ == '__main__':
    morphology_specification = default_toy_example_morphology_specification(
            num_arms=4, num_segments_per_arm=3
            )

    morphology = MJCToyExampleMorphology(specification=morphology_specification)

    morphology.export_to_xml_with_assets(output_directory="./mjcf")
