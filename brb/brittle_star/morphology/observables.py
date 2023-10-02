from typing import List

import numpy as np
from dm_control import composer, mjcf
from dm_control.composer.observation.observable import MJCFFeature
from dm_control.mjcf.physics import SynchronizingArrayWrapper
from mujoco_utils.observables import ConfinedMJCFFeature

from brb.brittle_star.morphology.utils.morphology import calculate_arm_length


def normalizer_factory(
        original_range: np.ndarray,
        target_range: np.ndarray = np.array([-1, 1])
        ):
    def normalizer(
            observations: SynchronizingArrayWrapper,
            *args,
            **kwargs
            ) -> np.ndarray:
        data = np.array(observations)

        delta1 = original_range[:, 1] - original_range[:, 0]
        delta2 = target_range[1] - target_range[0]

        return (delta2 * (data - original_range[:, 0]) / delta1) + target_range[0]

    return normalizer


class BrittleStarObservables(composer.Observables):
    _number_of_arms = None
    _arm_length = None
    _touch_sensors = None
    _in_plane_joint_pos_sensors = None
    _out_of_plane_joint_pos_sensors = None
    _in_plane_joint_vel_sensors = None
    _out_of_plane_joint_vel_sensors = None
    _actuator_force_sensors = None
    _linear_velocity_sensor = None
    _angular_velocity_sensor = None

    @property
    def number_of_arms(
            self
            ) -> int:
        if self._number_of_arms is None:
            self._number_of_arms = self._entity.morphology_specification.number_of_arms
        return self._number_of_arms

    @property
    def arm_length(
            self
            ) -> float:
        if self._arm_length is None:
            self._arm_length = calculate_arm_length(specification=self._entity.morphology_specification)
        return self._arm_length

    @property
    def touch_sensors(
            self
            ) -> List[mjcf.Element]:
        if self._touch_sensors is None:
            sensors = self._entity.mjcf_model.find_all('sensor')
            self._touch_sensors = list(
                    filter(
                            lambda
                                sensor: sensor.tag == "touch", sensors
                            )
                    )
        return self._touch_sensors

    @property
    def in_plane_joint_pos_sensors(
            self
            ) -> List[mjcf.Element]:
        if self._in_plane_joint_pos_sensors is None:
            sensors = self._entity.mjcf_model.find_all('sensor')
            self._in_plane_joint_pos_sensors = list(
                    filter(
                            lambda
                                sensor: sensor.tag == "jointpos" and "in_plane_joint" in sensor.name, sensors
                            )
                    )
        return self._in_plane_joint_pos_sensors

    @property
    def out_of_plane_joint_pos_sensors(
            self
            ) -> List[mjcf.Element]:
        if self._out_of_plane_joint_pos_sensors is None:
            sensors = self._entity.mjcf_model.find_all('sensor')
            self._out_of_plane_joint_pos_sensors = list(
                    filter(
                            lambda
                                sensor: sensor.tag == "jointpos" and "out_of_plane_joint" in sensor.name, sensors
                            )
                    )
        return self._out_of_plane_joint_pos_sensors


    @property
    def in_plane_joint_vel_sensors(
            self
            ) -> List[mjcf.Element]:
        if self._in_plane_joint_vel_sensors is None:
            sensors = self._entity.mjcf_model.find_all('sensor')
            self._in_plane_joint_vel_sensors = list(
                    filter(
                            lambda
                                sensor: sensor.tag == "jointvel" and "in_plane_joint" in sensor.name, sensors
                            )
                    )
        return self._in_plane_joint_vel_sensors

    @property
    def out_of_plane_joint_vel_sensors(
            self
            ) -> List[mjcf.Element]:
        if self._out_of_plane_joint_vel_sensors is None:
            sensors = self._entity.mjcf_model.find_all('sensor')
            self._out_of_plane_joint_vel_sensors = list(
                    filter(
                            lambda
                                sensor: sensor.tag == "jointvel" and "out_of_plane_joint" in sensor.name, sensors
                            )
                    )
        return self._out_of_plane_joint_vel_sensors


    @property
    def linear_velocity_sensor(
            self
            ) -> mjcf.Element:
        if self._linear_velocity_sensor is None:
            sensors = self._entity.mjcf_model.find_all('sensor')
            self._linear_velocity_sensor = list(
                    filter(
                            lambda
                                sensor: sensor.tag == "framelinvel" and "disc" in sensor.name, sensors
                            )
                    )[0]
        return self._linear_velocity_sensor

    @property
    def angular_velocity_sensor(
            self
            ) -> mjcf.Element:
        if self._angular_velocity_sensor is None:
            sensors = self._entity.mjcf_model.find_all('sensor')
            self._angular_velocity_sensor = list(
                    filter(
                            lambda
                                sensor: sensor.tag == "frameangvel" and "disc" in sensor.name, sensors
                            )
                    )[0]
        return self._angular_velocity_sensor

    @composer.observable
    def touch_per_tendon_plate(
            self
            ) -> MJCFFeature:
        def convert_to_booleans(
                observations: SynchronizingArrayWrapper,
                *args,
                **kwargs
                ) -> np.ndarray:
            data = np.array(observations)

            data[data > 0] = 1
            data[data == 0] = -1
            return data

        return ConfinedMJCFFeature(
                low=-1,
                high=1,
                shape=[len(self.touch_sensors)],
                kind="sensordata",
                mjcf_element=self.touch_sensors,
                corruptor=convert_to_booleans
                )

    @composer.observable
    def linear_velocity(
            self
            ) -> MJCFFeature:
        return ConfinedMJCFFeature(
                low=-5.0, high=5.0, shape=[3], kind="sensordata", mjcf_element=self.linear_velocity_sensor
                )

    @composer.observable
    def angular_velocity(
            self
            ) -> MJCFFeature:
        return ConfinedMJCFFeature(
                low=-5.0, high=5.0, shape=[3], kind="sensordata", mjcf_element=self.angular_velocity_sensor
                )

    @composer.observable
    def in_plane_joint_pos(
            self
            ) -> MJCFFeature:
        low, high = list(zip(*[sensor.joint.range for sensor in self.in_plane_joint_pos_sensors]))
        return ConfinedMJCFFeature(
                low=low,
                high=high,
                shape=[len(self.in_plane_joint_pos_sensors)],
                kind="sensordata",
                mjcf_element=self.in_plane_joint_pos_sensors
                )

    @composer.observable
    def out_of_plane_joint_pos(
            self
            ) -> MJCFFeature:
        low, high = list(zip(*[sensor.joint.range for sensor in self.out_of_plane_joint_pos_sensors]))
        return ConfinedMJCFFeature(
                low=low,
                high=high,
                shape=[len(self.out_of_plane_joint_pos_sensors)],
                kind="sensordata",
                mjcf_element=self.out_of_plane_joint_pos_sensors
                )

    @composer.observable
    def in_plane_joint_vel(
            self
            ) -> MJCFFeature:
        return ConfinedMJCFFeature(
                low=-5,
                high=5,
                shape=[len(self.in_plane_joint_vel_sensors)],
                kind="sensordata",
                mjcf_element=self.in_plane_joint_vel_sensors
                )

    @composer.observable
    def out_of_plane_joint_vel(
            self
            ) -> MJCFFeature:
        return ConfinedMJCFFeature(
                low=-5,
                high=5,
                shape=[len(self.out_of_plane_joint_vel_sensors)],
                kind="sensordata",
                mjcf_element=self.out_of_plane_joint_vel_sensors
                )

