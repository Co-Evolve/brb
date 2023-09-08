from __future__ import annotations

import numpy as np
from mujoco_utils.robot import MJCMorphology
from transforms3d.euler import euler2quat

from brb.seahorse.morphology.specification.default import default_seahorse_morphology_specification
from brb.seahorse.morphology.specification.specification import SeahorseMorphologySpecification, \
    SeahorseTendonActuationSpecification


class MJCSeahorseMorphology(MJCMorphology):
    def __init__(
            self,
            specification: SeahorseMorphologySpecification
            ) -> None:
        super().__init__(specification)

    @property
    def morphology_specification(
            self
            ) -> SeahorseMorphologySpecification:
        return super().morphology_specification

    @property
    def tendon_actuation_specification(
            self
            ) -> SeahorseTendonActuationSpecification:
        return self.morphology_specification.tendon_actuation_specification

    def _build(
            self
            ) -> None:
        self._configure_compiler()
        self._build_tail()

    def _configure_compiler(
            self
            ) -> None:
        self.mjcf_model.compiler.angle = 'radian'  # Use radians
        self.mjcf_model.option.flag.contact = 'disable'

    def _build_tail(
            self
            ) -> None:

        # configure mesh assets
        plates = ["ventral_sinistral", "ventral_dextral", "dorsal_dextral_even", "dorsal_sinistral_even"]
        self.mjcf_model.compiler.meshdir = ("/Users/driesmarzougui/phd/experiments/brb/brb/seahorse/morphology/assets"
                                            "/v1"
                                            "/original")



        vs_height = 0.052
        vs_width = 0.059
        vd_height = 0.055
        vd_width = 0.056
        ds_height = 0.052
        ds_width = 0.055
        dd_height = 0.056
        dd_width = 0.052

        scales = 0.001 * np.array(
                [[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]]
                )
        rotations = [
                euler2quat(*[np.pi / 2, 0.0, 0.0]),
                euler2quat(*[-np.pi/2, np.pi/2, 0], axes='syxz'),
                euler2quat(*[np.pi/2, 0.0, np.pi]),
                euler2quat(*[np.pi/2, 0, np.pi/2])
                ]
        positions = np.array([
                [-vs_height, 0.002, 0],
                [0, vd_width, 0],
                [dd_height, 0, 0],
                [0, -ds_width, 0]
                ])

        # Shift z dim of plates such that their middle is at z = 0
        #   todo: need real measurement
        plate_depth = 0.015
        positions[:, 2] = -plate_depth / 2
        #
        # # Shift x and y dim of plates such that they are locked in gliding joints
        # #   todo: need real measurements
        plate_y_gliding_shift = 0.0085 / 2
        plate_y_gliding_shift_signed = np.array([1, -1, -1, 1]) * plate_y_gliding_shift
        plate_x_gliding_shift = 0.0085 / 2
        plate_x_gliding_shift_signed = np.array([1, 1, -1, -1]) * plate_x_gliding_shift
        positions[:, 1] += plate_y_gliding_shift_signed
        positions[:, 0] += plate_x_gliding_shift_signed


        targets = [0, 1]
        for plate_index, plate in enumerate(plates):
            if plate_index not in targets:
                continue
            self.mjcf_model.asset.add(
                    "mesh", name=f"{plate}_plate", file=f"{plate}.STL", scale=scales[plate_index]
                    )

        for plate_index, plate in enumerate(plates):
            if plate_index not in targets:
                continue

            self.mjcf_body.add(
                    'geom',
                    name=f"{plate}_plate",
                    type="mesh",
                    mesh=f"{plate}_plate",
                    pos=positions[plate_index],
                    quat=rotations[plate_index]
                    )

        self.mjcf_model.asset.add(
                "mesh", name="vertebrae", file=f"vertebrae.STL", scale=[0.001, 0.001, 0.001]
                )
        self.mjcf_body.add(
                'geom',
                name=f"vertebrae",
                type="mesh",
                mesh=f"vertebrae",
                pos=[0, 0, 0],
                )

        # self.mjcf_body.add('geom',
        #                    type='box',
        #                    size=[10, 0.0001, 0.0001],
        #                    rgba=[1, 0, 0, 0.5])
        # self.mjcf_body.add('geom',
        #                    type='box',
        #                    size=[0.0001, 10, 0.0001],
        #                    rgba=[0, 1, 0, 0.5])
        #

if __name__ == '__main__':
    morphology_specification = default_seahorse_morphology_specification(num_segments=5)
morphology = MJCSeahorseMorphology(specification=morphology_specification)
morphology.export_to_xml_with_assets("./mjcf_test")
