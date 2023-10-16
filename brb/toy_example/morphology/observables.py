from typing import List

from dm_control import composer, mjcf
from dm_control.composer.observation.observable import MJCFFeature
from mujoco_utils.observables import ConfinedMJCFFeature


class ToyExampleObservables(composer.Observables):
    _in_plane_joint_pos_sensors = None
    _out_of_plane_joint_pos_sensors = None
    _in_plane_joint_vel_sensors = None
    _out_of_plane_joint_vel_sensors = None
    _torso_linear_velocity_sensor = None
    _torso_angular_velocity_sensor = None

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
    def torso_linear_velocity_sensor(
            self
            ) -> mjcf.Element:
        if self._torso_linear_velocity_sensor is None:
            sensors = self._entity.mjcf_model.find_all('sensor')
            self._torso_linear_velocity_sensor = list(
                    filter(
                            lambda
                                sensor: sensor.tag == "framelinvel" and "torso" in sensor.name, sensors
                            )
                    )[0]
        return self._torso_linear_velocity_sensor

    @property
    def torso_angular_velocity_sensor(
            self
            ) -> mjcf.Element:
        if self._torso_angular_velocity_sensor is None:
            sensors = self._entity.mjcf_model.find_all('sensor')
            self._torso_angular_velocity_sensor = list(
                    filter(
                            lambda
                                sensor: sensor.tag == "frameangvel" and "torso" in sensor.name, sensors
                            )
                    )[0]
        return self._torso_angular_velocity_sensor

    @composer.observable
    def torso_linear_velocity(
            self
            ) -> MJCFFeature:
        return ConfinedMJCFFeature(
                low=-5.0, high=5.0, shape=[3], kind="sensordata", mjcf_element=self.torso_linear_velocity_sensor
                )

    @composer.observable
    def torso_angular_velocity(
            self
            ) -> MJCFFeature:
        return ConfinedMJCFFeature(
                low=-5.0, high=5.0, shape=[3], kind="sensordata", mjcf_element=self.torso_angular_velocity_sensor
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
