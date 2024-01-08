from itertools import count
from typing import Any, Dict, List, SupportsFloat

import mujoco
import numpy as np
from gymnasium.core import ActType, ObsType
from mujoco_utils.environment.base import MuJoCoEnvironmentConfiguration
from mujoco_utils.environment.mjc_env import MJCEnv, MJCObservable
from mujoco_utils.morphology import MJCFMorphology
from transforms3d.euler import quat2euler

from brb.toy_example.arena.arena import MJCFPlaneWithTargetArena
from brb.toy_example.morphology.morphology import MJCFToyExampleMorphology


class ToyExampleEnvironmentConfiguration(MuJoCoEnvironmentConfiguration):
    def __init__(
            self,
            target_distance: float = 3,
            randomization_noise_scale: float = 0.0,
            *args,
            **kwargs
            ) -> None:
        super().__init__(*args, **kwargs)
        self.target_distance = target_distance
        self.randomization_noise_scale = randomization_noise_scale


class ToyExampleMJCEnvironment(MJCEnv):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
            self,
            morphology: MJCFToyExampleMorphology,
            arena: MJCFPlaneWithTargetArena,
            configuration: ToyExampleEnvironmentConfiguration
            ) -> None:
        super().__init__(morphology=morphology, arena=arena, configuration=configuration)

        self._previous_distance_to_target = 0

    @property
    def morphology(
            self
            ) -> MJCFMorphology:
        return super().morphology

    @property
    def arena(
            self
            ) -> MJCFPlaneWithTargetArena:
        return super().arena

    @property
    def environment_configuration(
            self
            ) -> ToyExampleEnvironmentConfiguration:
        return super().environment_configuration

    @staticmethod
    def get_xy_direction_to_target(
            model: mujoco.MjModel,
            data: mujoco.MjData
            ) -> np.ndarray:
        target_position = data.body("target").xpos
        torso_position = data.body("ToyExampleMorphology/torso").xpos
        direction_to_target = target_position - torso_position
        return direction_to_target[:2]

    @staticmethod
    def get_xy_distance_to_target(
            model: mujoco.MjModel,
            data: mujoco.MjData
            ) -> float:
        xy_direction_to_target = ToyExampleMJCEnvironment.get_xy_direction_to_target(model=model, data=data)
        xy_distance_to_target = np.linalg.norm(xy_direction_to_target)
        return xy_distance_to_target

    def _get_observables(
            self
            ) -> List[MJCObservable]:
        sensors = [self.mj_model.sensor(i) for i in range(self.mj_model.nsensor)]

        # All joint positions
        joint_pos_sensors = [sensor for sensor in sensors if sensor.type[0] == mujoco.mjtSensor.mjSENS_JOINTPOS]
        in_plane_joint_pos_sensors = [sensor for sensor in joint_pos_sensors if "in_plane_joint" in sensor.name]
        in_plane_joints = [self.mj_model.joint(sensor.objid[0]) for sensor in in_plane_joint_pos_sensors]
        out_of_plane_joint_pos_sensors = [sensor for sensor in joint_pos_sensors if "out_of_plane_joint" in sensor.name]
        out_of_plane_joints = [self.mj_model.joint(sensor.objid[0]) for sensor in out_of_plane_joint_pos_sensors]

        in_plane_joint_pos_observable = MJCObservable(
                name="in_plane_joint_position",
                low=np.array([joint.range[0] for joint in in_plane_joints]),
                high=np.array([joint.range[1] for joint in in_plane_joints]),
                retriever=lambda
                    model,
                    data: np.array(
                        [data.sensordata[sensor.adr[0]: sensor.adr[0] + sensor.dim[0]] for sensor in
                         in_plane_joint_pos_sensors]
                        ).flatten()
                )
        out_of_plane_joint_pos_observable = MJCObservable(
                name="out_of_plane_joint_position",
                low=np.array([joint.range[0] for joint in out_of_plane_joints]),
                high=np.array([joint.range[1] for joint in out_of_plane_joints]),
                retriever=lambda
                    model,
                    data: np.array(
                        [data.sensordata[sensor.adr[0]: sensor.adr[0] + sensor.dim[0]] for sensor in
                         out_of_plane_joint_pos_sensors]
                        ).flatten()
                )

        # All joint velocities
        joint_vel_sensors = [sensor for sensor in sensors if sensor.type[0] == mujoco.mjtSensor.mjSENS_JOINTVEL]
        in_plane_joint_velocity_sensors = [sensor for sensor in joint_vel_sensors if "in_plane_joint" in sensor.name]
        out_of_plane_joint_velocity_sensors = [sensor for sensor in joint_vel_sensors if
                                               "out_of_plane_joint" in sensor.name]
        in_plane_joint_velocity_observable = MJCObservable(
                name="in_plane_joint_velocity",
                low=-np.inf * np.ones(len(in_plane_joint_velocity_sensors)),
                high=np.inf * np.ones(len(in_plane_joint_velocity_sensors)),
                retriever=lambda
                    model,
                    data: np.array(
                        [data.sensordata[sensor.adr[0]: sensor.adr[0] + sensor.dim[0]] for sensor in
                         in_plane_joint_velocity_sensors]
                        ).flatten()
                )
        out_of_plane_joint_velocity_observable = MJCObservable(
                name="out_of_plane_joint_velocity",
                low=-np.inf * np.ones(len(out_of_plane_joint_velocity_sensors)),
                high=np.inf * np.ones(len(out_of_plane_joint_velocity_sensors)),
                retriever=lambda
                    model,
                    data: np.array(
                        [data.sensordata[sensor.adr[0]: sensor.adr[0] + sensor.dim[0]] for sensor in
                         out_of_plane_joint_velocity_sensors]
                        ).flatten()
                )

        # segment touch values
        indexer = count(0)
        segment_capsule_geom_ids_to_contact_idx = {geom_id: next(indexer) for geom_id in range(self.mj_model.ngeom) if
                                                   "segment" in self.mj_model.geom(
                                                           geom_id
                                                           ).name and "capsule" in self.mj_model.geom(geom_id).name}
        ground_floor_geom_id = self.mj_model.geom("groundplane").id

        def get_segment_ground_contacts(
                model: mujoco.MjModel,
                data: mujoco.MjData
                ) -> np.ndarray:
            ground_contacts = np.zeros(len(segment_capsule_geom_ids_to_contact_idx))
            # based on https://gist.github.com/WuXinyang2012/b6649817101dfcb061eff901e9942057
            for contact_id in range(data.ncon):
                contact = self.mj_data.contact[contact_id]
                if contact.geom1 == ground_floor_geom_id:
                    if contact.geom2 in segment_capsule_geom_ids_to_contact_idx:
                        c_array = np.zeros(6, dtype=np.float64)
                        mujoco.mj_contactForce(m=self.mj_model, d=self.mj_data, id=contact_id, result=c_array)

                        # Convert the contact force from contact frame to world frame
                        ref = np.reshape(contact.frame, (3, 3))
                        c_force = np.dot(np.linalg.inv(ref), c_array[0:3])

                        index = segment_capsule_geom_ids_to_contact_idx[contact.geom2]
                        ground_contacts[index] = max(np.linalg.norm(c_force), ground_contacts[index])
            # noinspection PyUnresolvedReferences
            ground_contacts = (ground_contacts > 0).astype(int)
            return ground_contacts

        touch_observable = MJCObservable(
                name="segment_ground_contact",
                low=np.zeros(len(segment_capsule_geom_ids_to_contact_idx)),
                high=np.inf * np.ones(len(segment_capsule_geom_ids_to_contact_idx)),
                retriever=get_segment_ground_contacts
                )

        # torso framequat
        framequat_sensor = [sensor for sensor in sensors if sensor.type[0] == mujoco.mjtSensor.mjSENS_FRAMEQUAT][0]
        torso_rotation_observable = MJCObservable(
                name="torso_rotation",
                low=-np.pi * np.ones(3),
                high=np.pi * np.ones(3),
                retriever=lambda
                    model,
                    data: np.array(
                        quat2euler(
                                data.sensordata[framequat_sensor.adr[0]:
                                                framequat_sensor.adr[0] + framequat_sensor.dim[0]]
                                )
                        )
                )

        # torso framelinvel
        framelinvel_sensor = [sensor for sensor in sensors if sensor.type[0] == mujoco.mjtSensor.mjSENS_FRAMELINVEL][0]
        torso_linvel_observable = MJCObservable(
                name="torso_linear_velocity",
                low=-np.inf * np.ones(3),
                high=np.inf * np.ones(3),
                retriever=lambda
                    model,
                    data: np.array(
                        data.sensordata[
                        framelinvel_sensor.adr[0]: framelinvel_sensor.adr[0] + framelinvel_sensor.dim[0]]
                        )
                )

        # torso frameangvel
        frameangvel_sensor = [sensor for sensor in sensors if sensor.type[0] == mujoco.mjtSensor.mjSENS_FRAMEANGVEL][0]
        torso_angvel_observable = MJCObservable(
                name="torso_angular_velocity",
                low=-np.inf * np.ones(3),
                high=np.inf * np.ones(3),
                retriever=lambda
                    model,
                    data: np.array(
                        data.sensordata[
                        frameangvel_sensor.adr[0]: frameangvel_sensor.adr[0] + frameangvel_sensor.dim[0]]
                        )
                )

        # direction to target
        unit_xy_direction_to_target_observable = MJCObservable(
                name="unit_xy_direction_to_target",
                low=-np.ones(2),
                high=np.ones(2),
                retriever=lambda
                    model,
                    data: self.get_xy_direction_to_target(model=model, data=data) / self.get_xy_distance_to_target(
                        model=model, data=data
                        )
                )

        # distance to target
        xy_distance_to_target_observable = MJCObservable(
                name="xy_distance_to_target",
                low=np.zeros(1),
                high=np.inf * np.ones(1),
                retriever=lambda
                    model,
                    data: np.array([self.get_xy_distance_to_target(model=model, data=data)])
                )

        return [in_plane_joint_pos_observable, out_of_plane_joint_pos_observable, in_plane_joint_velocity_observable,
                out_of_plane_joint_velocity_observable, touch_observable, torso_rotation_observable,
                torso_linvel_observable, torso_angvel_observable, unit_xy_direction_to_target_observable,
                xy_distance_to_target_observable]

    def _get_info(
            self
            ) -> Dict[str, Any]:
        return {"time": self.mj_data.time}

    def _get_reward(
            self
            ) -> float:
        # Previous distance to target - current distance to target
        return self._previous_distance_to_target - self.get_xy_distance_to_target(
                model=self.mj_model, data=self.mj_data
                )

    def _should_terminate(
            self
            ) -> bool:
        return self.get_xy_distance_to_target(model=self.mj_model, data=self.mj_data) < 0.2

    def _should_truncate(
            self
            ) -> bool:
        return self.mj_data.time > self.environment_configuration.simulation_time

    def step(
            self,
            action: ActType
            ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        self._take_n_steps(ctrl=action)

        observations = self._get_observations()
        reward = self._get_reward()
        terminated = self._should_terminate()
        truncated = self._should_truncate()
        info = self._get_info()

        self._previous_distance_to_target = observations["xy_distance_to_target"]

        return observations, reward, terminated, truncated, info

    def reset(
            self,
            *,
            seed: int | None = None,
            options: dict[str, Any] | None = None
            ) -> tuple[ObsType, dict[str, Any]]:
        mujoco.mj_resetData(m=self.mj_model, d=self.mj_data)

        random_state = np.random.RandomState(seed=seed)

        # Set random target position
        angle = random_state.uniform(0, 2 * np.pi)
        radius = self.environment_configuration.target_distance
        self.mj_model.body("target").pos = [radius * np.cos(angle), radius * np.sin(angle), 0.05]

        # Set morphology position
        self.mj_model.body("ToyExampleMorphology/torso").pos[2] = 0.11

        # Add noise to initial qpos and qvel of segment joints
        segment_joints = [self.mj_model.joint(joint_id) for joint_id in range(self.mj_model.njnt) if
                          "segment" in self.mj_model.joint(joint_id).name]
        num_segment_joints = len(segment_joints)
        joint_qpos_adrs = [joint.qposadr[0] for joint in segment_joints]
        self.mj_data.qpos[joint_qpos_adrs] = self.mj_model.qpos0[joint_qpos_adrs] + random_state.uniform(
                low=-self.environment_configuration.randomization_noise_scale,
                high=self.environment_configuration.randomization_noise_scale,
                size=num_segment_joints
                )
        joint_qvel_adrs = [joint.dofadr[0] for joint in segment_joints]
        self.mj_data.qvel[joint_qvel_adrs] = random_state.uniform(
                low=-self.environment_configuration.randomization_noise_scale,
                high=self.environment_configuration.randomization_noise_scale,
                size=num_segment_joints
                )

        # Flush changes in mj_model to mj_data
        mujoco.mj_forward(self.mj_model, self.mj_data)

        self._previous_distance_to_target = self.get_xy_distance_to_target(model=self.mj_model, data=self.mj_data)
        return self._get_observations(), self._get_info()
