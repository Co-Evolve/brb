import numpy as np
from dm_control import composer, mjcf
from dm_control.composer.observation.observable import MJCFFeature
from dm_control.mjcf.physics import SynchronizingArrayWrapper
from mujoco_utils.observables import ConfinedMJCFFeature

from brb.seahorse.morphology.specification.specification import SeahorseMorphologySpecification


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


def get_joint_range_from_sensor(
        sensor: mjcf.Element
        ) -> np.ndarray:
    return


class SeahorseObservables(composer.Observables):
    _vertebrae_pitch_joint_pos_sensors = None
    _vertebrae_roll_joint_pos_sensors = None
    _hmm_tendon_pos_sensors = None
    _mvm_tendon_pos_sensors = None

    @property
    def morphology_specification(
            self
            ) -> SeahorseMorphologySpecification:
        return self._entity.morphology_specification

    @property
    def vertebrae_pitch_joint_pos_sensors(
            self
            ) -> mjcf.Element:
        if self._vertebrae_pitch_joint_pos_sensors is None:
            sensors = self._entity.mjcf_model.find_all('sensor')
            self._vertebrae_pitch_joint_pos_sensors = list(
                    filter(
                            lambda
                                sensor: sensor.tag == "jointpos" and "vertebrae" in sensor.name and "pitch" in
                                        sensor.name,
                            sensors
                            )
                    )
        return self._vertebrae_pitch_joint_pos_sensors

    @property
    def vertebrae_roll_joint_pos_sensors(
            self
            ) -> mjcf.Element:
        if self._vertebrae_roll_joint_pos_sensors is None:
            sensors = self._entity.mjcf_model.find_all('sensor')
            self._vertebrae_roll_joint_pos_sensors = list(
                    filter(
                            lambda
                                sensor: sensor.tag == "jointpos" and "vertebrae" in sensor.name and "roll" in
                                        sensor.name,
                            sensors
                            )
                    )
        return self._vertebrae_roll_joint_pos_sensors

    @property
    def hmm_tendon_pos_sensors(
            self
            ) -> mjcf.Element:
        if self._hmm_tendon_pos_sensors is None:
            sensors = self._entity.mjcf_model.find_all('sensor')
            self._hmm_tendon_pos_sensors = list(
                    filter(
                            lambda
                                sensor: sensor.tag == "tendonpos" and "hmm" in sensor.name, sensors
                            )
                    )
        return self._hmm_tendon_pos_sensors

    @property
    def mvm_tendon_pos_sensors(
            self
            ) -> mjcf.Element:
        if self._mvm_tendon_pos_sensors is None:
            sensors = self._entity.mjcf_model.find_all('sensor')
            self._mvm_tendon_pos_sensors = list(
                    filter(
                            lambda
                                sensor: sensor.tag == "tendonpos" and "mvm" in sensor.name, sensors
                            )
                    )
        return self._mvm_tendon_pos_sensors

    @composer.observable
    def vertebrae_pitch_joint_pos(
            self
            ) -> MJCFFeature:
        low, high = np.array([sensor.joint.range for sensor in self.vertebrae_pitch_joint_pos_sensors]).T
        return ConfinedMJCFFeature(
                low=low,
                high=high,
                shape=[len(self.vertebrae_pitch_joint_pos_sensors)],
                kind="sensordata",
                mjcf_element=self.vertebrae_pitch_joint_pos_sensors
                )

    @composer.observable
    def vertebrae_roll_joint_pos(
            self
            ) -> MJCFFeature:
        low, high = np.array([sensor.joint.range for sensor in self.vertebrae_roll_joint_pos_sensors]).T
        return ConfinedMJCFFeature(
                low=low,
                high=high,
                shape=[len(self.vertebrae_roll_joint_pos_sensors)],
                kind="sensordata",
                mjcf_element=self.vertebrae_roll_joint_pos_sensors
                )

    @composer.observable
    def hmm_tendon_pos(
            self
            ) -> MJCFFeature:
        low, high = np.array([sensor.tendon.range for sensor in self.hmm_tendon_pos_sensors]).T
        return ConfinedMJCFFeature(
                low=low,
                high=high,
                shape=[len(self.hmm_tendon_pos_sensors)],
                kind="sensordata",
                mjcf_element=self.hmm_tendon_pos_sensors
                )

    @composer.observable
    def mvm_tendon_pos(
            self
            ) -> MJCFFeature:
        if len(self.mvm_tendon_pos_sensors) > 0:
            low, high = np.array([sensor.tendon.range for sensor in self.mvm_tendon_pos_sensors]).T
        else:
            low, high = -1, 1
        return ConfinedMJCFFeature(
                low=low,
                high=high,
                shape=[len(self.mvm_tendon_pos_sensors)],
                kind="sensordata",
                mjcf_element=self.mvm_tendon_pos_sensors
                )
