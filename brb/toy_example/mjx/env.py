from typing import Any, Dict, List

import jax.random
from jax import numpy as jnp
from jax._src.scipy.spatial.transform import Rotation
from mujoco import mjx
from mujoco._structs import _MjModelJointViews
from mujoco_utils.environment.mjx_env import MJXEnv, MJXObservable, MJXState

from brb.toy_example.arena.arena import MJCFPlaneWithTargetArena
from brb.toy_example.mjc.env import ToyExampleEnvironmentConfiguration
from brb.toy_example.morphology.morphology import MJCFToyExampleMorphology


class ToyExampleMJXEnvironment(MJXEnv):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
            self,
            morphology: MJCFToyExampleMorphology,
            arena: MJCFPlaneWithTargetArena,
            configuration: ToyExampleEnvironmentConfiguration
            ) -> None:
        super().__init__(morphology=morphology, arena=arena, configuration=configuration)
        self._segment_joints: List[_MjModelJointViews] = [self.mj_model.joint(joint_id) for joint_id in range(
                self.mj_model.njnt
                ) if "segment" in self.mj_model.joint(joint_id).name]
        self._segment_joint_qpos_adrs = jnp.array([joint.qposadr[0] for joint in self._segment_joints])
        self._segment_joint_qvel_adrs = jnp.array([joint.dofadr[0] for joint in self._segment_joints])

    @property
    def morphology(
            self
            ) -> MJCFToyExampleMorphology:
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

    def _get_xy_direction_to_target(
            self,
            mjx_model: mjx.Model,
            mjx_data: mjx.Data
            ) -> jnp.ndarray:
        torso_body_id = self.mj_model.body("ToyExampleMorphology/torso").id
        target_body_id = self.mj_model.body("target").id

        torso_body_pos = mjx_data.xpos[torso_body_id]
        target_pos = mjx_data.xpos[target_body_id]

        return (target_pos - torso_body_pos)[:2]

    def _get_xy_distance_to_target(
            self,
            mjx_model: mjx.Model,
            mjx_data: mjx.Data
            ) -> jnp.ndarray:
        xy_direction_to_target = self._get_xy_direction_to_target(
                mjx_model=mjx_model, mjx_data=mjx_data
                )
        return jnp.linalg.norm(xy_direction_to_target)

    def _get_observables(
            self
            ) -> List[MJXObservable]:
        in_plane_joints = [self.mj_model.joint(joint_id) for joint_id in range(self.mj_model.njnt) if
                           "in_plane_joint" in self.mj_model.joint(joint_id).name]
        out_of_plane_joints = [self.mj_model.joint(joint_id) for joint_id in range(self.mj_model.njnt) if
                               "out_of_plane_joint" in self.mj_model.joint(joint_id).name]

        in_plane_joint_qpos_adr = jnp.array([joint.qposadr[0] for joint in in_plane_joints])
        in_plane_joint_range = jnp.array([joint.range for joint in in_plane_joints]).T
        out_of_plane_joint_qpos_adr = jnp.array([joint.qposadr[0] for joint in out_of_plane_joints])
        out_of_plane_joint_range = jnp.array([joint.range for joint in out_of_plane_joints]).T
        in_plane_joint_qvel_adr = jnp.array([joint.dofadr[0] for joint in in_plane_joints])
        out_of_plane_joint_qvel_adr = jnp.array([joint.dofadr[0] for joint in out_of_plane_joints])

        in_plane_joint_position_observable = MJXObservable(
                name="in_plane_joint_position",
                low=in_plane_joint_range[0],
                high=in_plane_joint_range[1],
                retriever=lambda
                    mjx_model,
                    mjx_data: mjx_data.qpos[in_plane_joint_qpos_adr]
                )

        out_of_plane_joint_position_observable = MJXObservable(
                name="out_of_plane_joint_position",
                low=out_of_plane_joint_range[0],
                high=out_of_plane_joint_range[1],
                retriever=lambda
                    mjx_model,
                    mjx_data: mjx_data.qpos[out_of_plane_joint_qpos_adr]
                )

        in_plane_joint_velocity_observable = MJXObservable(
                name="in_plane_joint_velocity",
                low=-jnp.inf * jnp.ones(len(in_plane_joints)),
                high=jnp.inf * jnp.ones(len(in_plane_joints)),
                retriever=lambda
                    mjx_model,
                    mjx_data: mjx_data.qvel[in_plane_joint_qvel_adr]
                )
        out_of_plane_joint_velocity_observable = MJXObservable(
                name="out_of_plane_joint_velocity",
                low=-jnp.inf * jnp.ones(len(out_of_plane_joints)),
                high=jnp.inf * jnp.ones(len(out_of_plane_joints)),
                retriever=lambda
                    mjx_model,
                    mjx_data: mjx_data.qvel[out_of_plane_joint_qvel_adr]
                )

        segment_capsule_geom_ids = jnp.array(
                [geom_id for geom_id in range(self.mj_model.ngeom) if
                 "segment" in self.mj_model.geom(geom_id).name and "capsule" in self.mj_model.geom(
                         geom_id
                         ).name]
                )
        ground_floor_geom_id = self.mj_model.geom("groundplane").id

        def get_segment_ground_contacts(
                mjx_model: mjx.Model,
                mjx_data: mjx.Data
                ) -> jnp.ndarray:
            contact_data = mjx_data.contact
            contacts = contact_data.dist <= 0
            valid_geom1s = contact_data.geom1 == ground_floor_geom_id

            def solve_contact(
                    geom_id: int
                    ) -> jnp.ndarray:
                return (jnp.sum(contacts * valid_geom1s * (contact_data.geom2 == geom_id)) > 0).astype(int)

            return jax.vmap(solve_contact)(segment_capsule_geom_ids)

        touch_observable = MJXObservable(
                name="segment_ground_contact",
                low=jnp.zeros(len(segment_capsule_geom_ids)),
                high=jnp.ones(len(segment_capsule_geom_ids)),
                retriever=get_segment_ground_contacts
                )

        # torso framequat
        torso_id = self.mj_model.body("ToyExampleMorphology/torso").id
        torso_rotation_observable = MJXObservable(
                name="torso_rotation",
                low=-jnp.pi * jnp.ones(3),
                high=jnp.pi * jnp.ones(3),
                retriever=lambda
                    mjx_model,
                    mjx_data: Rotation.from_quat(quat=mjx_data.xquat[torso_id]).as_euler(seq="xyz")
                )

        # torso framelinvel
        morphology_freejoint_adr = self.mj_model.joint('ToyExampleMorphology/freejoint/').dofadr[0]
        torso_linvel_observable = MJXObservable(
                name="torso_linear_velocity",
                low=-jnp.inf * jnp.ones(3),
                high=jnp.inf * jnp.ones(3),
                retriever=lambda
                    mjx_model,
                    mjx_data: mjx_data.qvel[morphology_freejoint_adr: morphology_freejoint_adr + 3]
                )
        # torso frameangvel
        torso_angvel_observable = MJXObservable(
                name="torso_angular_velocity",
                low=-jnp.inf * jnp.ones(3),
                high=jnp.inf * jnp.ones(3),
                retriever=lambda
                    mjx_model,
                    mjx_data: mjx_data.qvel[morphology_freejoint_adr + 3: morphology_freejoint_adr + 6]
                )

        # direction to target
        unit_xy_direction_to_target_observable = MJXObservable(
                name="unit_xy_direction_to_target",
                low=-jnp.ones(2),
                high=jnp.ones(2),
                retriever=lambda
                    mjx_model,
                    mjx_data: self._get_xy_direction_to_target(
                        mjx_model=mjx_model, mjx_data=mjx_data
                        ) / self._get_xy_distance_to_target(
                        mjx_model=mjx_model, mjx_data=mjx_data
                        )
                )
        # distance to target
        xy_distance_to_target_observable = MJXObservable(
                name="xy_distance_to_target",
                low=jnp.zeros(1),
                high=jnp.inf * jnp.ones(1),
                retriever=lambda
                    mjx_model,
                    mjx_data: self._get_xy_distance_to_target(
                        mjx_model=mjx_model, mjx_data=mjx_data
                        )
                )

        return [in_plane_joint_position_observable, out_of_plane_joint_position_observable,
                in_plane_joint_velocity_observable, out_of_plane_joint_velocity_observable, touch_observable,
                torso_rotation_observable, torso_linvel_observable, torso_angvel_observable,
                unit_xy_direction_to_target_observable, xy_distance_to_target_observable]

    def step(
            self,
            state: MJXState,
            actions: jnp.ndarray
            ) -> MJXState:
        mjx_data0 = state.mjx_data
        mjx_data = self._take_n_steps(mjx_model=state.mjx_model, mjx_data=state.mjx_data, ctrl=actions)

        observations = self._get_observations(mjx_model=state.mjx_model, mjx_data=mjx_data)
        reward = self._get_reward(mjx_model=state.mjx_model, mjx_data=mjx_data, mjx_data0=mjx_data0)
        terminated = self._should_terminate(mjx_model=state.mjx_model, mjx_data=mjx_data)
        truncated = self._should_truncate(mjx_model=state.mjx_model, mjx_data=mjx_data)
        info = self._get_info(mjx_model=state.mjx_model, mjx_data=mjx_data)

        return state.replace(
                mjx_data=mjx_data,
                observations=observations,
                reward=reward,
                terminated=terminated,
                truncated=truncated,
                info=info
                )

    def _get_reward(
            self,
            mjx_model: mjx.Model,
            mjx_data: mjx.Data,
            mjx_data0: mjx.Data
            ) -> jnp.ndarray:
        previous_distance_to_target = self._get_xy_distance_to_target(mjx_model=mjx_model, mjx_data=mjx_data0)
        current_distance_to_target = self._get_xy_distance_to_target(mjx_model=mjx_model, mjx_data=mjx_data)
        return previous_distance_to_target - current_distance_to_target

    def _should_terminate(
            self,
            mjx_model: mjx.Model,
            mjx_data: mjx.Data
            ) -> jnp.ndarray:
        distance_to_target = self._get_xy_distance_to_target(mjx_model=mjx_model, mjx_data=mjx_data)
        return (distance_to_target < 0.2).astype(bool)

    def _should_truncate(
            self,
            mjx_model: mjx.Model,
            mjx_data: mjx.Data
            ) -> jnp.ndarray:
        return (mjx_data.time > self.environment_configuration.simulation_time).astype(bool)

    def _get_info(
            self,
            mjx_model: mjx.Model,
            mjx_data: mjx.Data
            ) -> Dict[str, Any]:
        return {"time": mjx_data.time}

    def _get_random_target_position(
            self,
            rng: jnp.ndarray
            ) -> jnp.ndarray:
        angle = jax.random.uniform(key=rng, shape=(), minval=0, maxval=jnp.pi * 2)
        radius = self.environment_configuration.target_distance
        target_pos = jnp.array([radius * jnp.cos(angle), radius * jnp.sin(angle), 0.05])
        return target_pos

    def reset(
            self,
            rng: jnp.ndarray
            ) -> MJXState:
        target_pos_rng, qpos_rng, qvel_rng = jax.random.split(key=rng, num=3)

        target_body_id = self.mj_model.body("target").id
        torso_body_id = self.mj_model.body("ToyExampleMorphology/torso").id

        # Set random target position
        target_pos = self._get_random_target_position(rng=rng)
        mjx_model = self.mjx_model.replace(body_pos=self.mjx_model.body_pos.at[target_body_id].set(target_pos))

        # Set morphology position
        morphology_pos = jnp.array([0.0, 0.0, 0.11])
        mjx_model = mjx_model.replace(body_pos=mjx_model.body_pos.at[torso_body_id].set(morphology_pos))

        # Add noise to initial qpos and qvel of segment joints
        qpos = jnp.copy(self.mjx_model.qpos0)
        qvel = jnp.zeros(self.mjx_model.nv)

        qpos = qpos.at[self._segment_joint_qpos_adrs].set(
                qpos[self._segment_joint_qpos_adrs] + jax.random.uniform(
                        key=qpos_rng,
                        shape=(len(self._segment_joints),),
                        minval=-self.environment_configuration.randomization_noise_scale,
                        maxval=self.environment_configuration.randomization_noise_scale
                        )
                )
        qvel = qvel.at[self._segment_joint_qvel_adrs].set(
                jax.random.uniform(
                        key=qvel_rng,
                        shape=(len(self._segment_joints),),
                        minval=-self.environment_configuration.randomization_noise_scale,
                        maxval=self.environment_configuration.randomization_noise_scale
                        )
                )

        mjx_data = self._initialize_mjx_data(mjx_model=mjx_model, mjx_data=self.mjx_data, qpos=qpos, qvel=qvel)

        obs = self._get_observations(mjx_model=mjx_model, mjx_data=mjx_data)
        reward, terminated, truncated = jnp.zeros(1), jnp.zeros(1).astype(bool), jnp.zeros(1).astype(bool)
        info = self._get_info(mjx_model=mjx_model, mjx_data=mjx_data)

        # noinspection PyArgumentList
        return MJXState(
                mjx_model=mjx_model,
                mjx_data=mjx_data,
                observations=obs,
                reward=reward,
                terminated=terminated.astype(bool),
                truncated=truncated.astype(bool),
                info=info
                )
