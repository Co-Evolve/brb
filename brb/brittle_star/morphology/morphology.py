import numpy as np
from dm_control import mjcf
from transforms3d.euler import euler2quat

from brb.brittle_star.morphology.observables import BrittleStarObservables
from brb.brittle_star.morphology.parts.arm import MJCBrittleStarArm
from brb.brittle_star.morphology.parts.disc import MJCBrittleStarDisc
from brb.brittle_star.morphology.specification.specification import BrittleStarMorphologySpecification
from erpy.framework.specification import RobotSpecification
from erpy.interfaces.mujoco.phenome import MJCMorphology
from erpy.utils import colors


class MJCBrittleStarMorphology(MJCMorphology):
    def __init__(self, specification: RobotSpecification) -> None:
        super().__init__(specification)
        self.touch_coloring = False

    @property
    def morphology_specification(self) -> BrittleStarMorphologySpecification:
        return super().morphology_specification

    def _build(self, *args, **kwargs):
        self._configure_compiler()
        self._build_disc()
        self._build_arms()

        self._prepare_coloring()
        self._configure_camera()

    def _build_observables(self) -> None:
        return BrittleStarObservables(self)

    def _configure_compiler(self) -> None:
        self.mjcf_model.compiler.angle = "radian"

    def _build_disc(self) -> None:
        self._disc = MJCBrittleStarDisc(parent=self,
                                        name="central_disc",
                                        pos=np.zeros(3),
                                        euler=np.zeros(3))

    def _build_arms(self) -> None:
        # Equally spaced over the disc
        self.arms = []

        disc_radius = self.morphology_specification.disc_specification.radius.value
        arm_angles = np.linspace(0, 2 * np.pi, 6)[:-1]
        number_of_arms = self.morphology_specification.number_of_arms

        for arm_index in range(number_of_arms):
            angle = arm_angles[arm_index]
            position = disc_radius * np.array([np.cos(angle), np.sin(angle), 0])
            arm = MJCBrittleStarArm(parent=self._disc,
                                    name=f"arm_{arm_index}",
                                    pos=position,
                                    euler=[0, 0, angle],
                                    arm_index=arm_index)
            self.arms.append(arm)

    def _prepare_tendon_coloring(self) -> None:
        if self.morphology_specification.actuation_specification.use_tendons.value:
            self._tendon_actuators = list(filter(lambda actuator: actuator.tendon is not None, self.actuators))
            self._tendons = [actuator.tendon for actuator in self._tendon_actuators]

            self._contracted_rgbas = np.ones((len(self._tendons), 4))
            self._contracted_rgbas[:] = colors.rgba_tendon_contracted

            self._color_changes = np.ones((len(self._tendons), 4))
            self._color_changes[:] = colors.rgba_tendon_relaxed - colors.rgba_tendon_contracted
            self._color_changes = self._color_changes.T

    def _color_muscles(self, physics: mjcf.Physics) -> None:
        if self.morphology_specification.actuation_specification.use_tendons.value:
            # Called often -> need high performance -> cache stuff as much as possible -> _prepare_tendon_coloring()!

            tendon_control = np.array(physics.bind(self._tendon_actuators).ctrl)
            # [-1, 0] to [0, 1] (relaxation because 0 -> fully contracted and 1 means relaxed
            tendon_relaxation = tendon_control + 1

            physics.bind(self._tendons).rgba = self._contracted_rgbas + (tendon_relaxation * self._color_changes).T

    def _prepare_coloring(self) -> None:
        self._segment_capsules = list(filter(lambda geom: "_segment_" in geom.name and geom.name.endswith("capsule"),
                                             self.mjcf_model.find_all('geom')))
        self._tendon_plates = list(filter(lambda geom: "_tendon_plate" in geom.name and geom.name.endswith("capsule"),
                                             self.mjcf_model.find_all('geom')))

        sensors = self.mjcf_model.find_all('sensor')
        self.touch_sensors = list(filter(lambda sensor: sensor.tag == "touch", sensors))

        self._prepare_tendon_coloring()

    def _color_touch(self, physics: mjcf.Physics) -> None:
        if self.touch_coloring:
            # Get values
            touch_per_tendon_plate = np.array(physics.bind(self.touch_sensors).sensordata)

            touch_per_tendon_plate[touch_per_tendon_plate > 0.0] = 1
            touch_per_tendon_plate[touch_per_tendon_plate == 0.0] = -1

            touch_per_segment = touch_per_tendon_plate.reshape(-1, 2)

            aggregated_touch_per_segment = np.sum(touch_per_segment, axis=1)

            for segment, touch_value in zip(self._segment_capsules, aggregated_touch_per_segment):
                segment_physics = physics.bind(segment)
                if touch_value > 0:
                    segment_physics.rgba = colors.rgba_red
                elif touch_value == 0.0:
                    segment_physics.rgba = colors.rgba_orange
                else:
                    segment_physics.rgba = colors.rgba_green

    def color_light(self, physics: mjcf.Physics, light_value_per_segment: np.ndarray) -> None:
        for segment, light_value in zip(self._segment_capsules, light_value_per_segment):
            segment_physics = physics.bind(segment)
            segment_physics.rgba = colors.rgba_gray + light_value * (colors.rgba_bright_green - colors.rgba_gray)

    def _configure_camera(self) -> None:
         self._disc.mjcf_body.add(
            'camera',
            name='side_camera',
            pos=[0.0, -5.0, 6.0],
            quat=euler2quat(40 / 180 * np.pi, 0, 0),
            mode="track",
        )

    def after_step(self, physics, random_state) -> None:
        self._color_muscles(physics=physics)
        self._color_touch(physics=physics)
        super().after_step(physics, random_state)
