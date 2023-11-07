import functools
from datetime import datetime
from typing import List

import jax
import mujoco
import mujoco.viewer
import numpy as np
from brax.training.agents.ppo import train as ppo

from dm_control.mjcf import export_with_assets
from jax import numpy as jnp
from jax._src.scipy.spatial.transform import Rotation
from mujoco import mjx

from brb.toy_example.mjx.env import MjxEnv
from brb.toy_example.mjx.state import State
from brb.toy_example.mjx.utils import run_fixed_viewer
from brb.toy_example.morphology.morphology import MJCToyExampleMorphology
from brb.toy_example.morphology.specification.default import default_toy_example_morphology_specification
from brb.toy_example.task.move_to_target import MoveToTargetTask, MoveToTargetTaskConfiguration


class ToyExampleEnv(MjxEnv):
    def __init__(
            self
            ) -> None:
        self._morphology = None
        super().__init__(self._get_mjModel())

    def _get_mjModel(
            self
            ) -> str:
        morphology_specification = default_toy_example_morphology_specification(
                num_arms=4, num_segments_per_arm=3
                )
        self._morphology = MJCToyExampleMorphology(specification=morphology_specification)
        task_config = MoveToTargetTaskConfiguration()
        self.task: MoveToTargetTask = task_config.task(task_config, self._morphology)
        mjcf_model = self.task.root_entity.mjcf_model

        export_with_assets(mjcf_model=mjcf_model, out_dir="./mjcf")
        mjcf_model.option.cone = "pyramidal"
        xml_string = mjcf_model.to_xml_string()
        mj_model = mujoco.MjModel.from_xml_string(xml=xml_string)
        return mj_model

    def _get_random_target_position(
            self,
            rng: jnp.ndarray
            ) -> jnp.ndarray:
        angle = jax.random.uniform(key=rng, shape=(), minval=0, maxval=jnp.pi * 2)
        radius = 2.0
        target_pos = radius * jnp.array([jnp.cos(angle), jnp.sin(angle), 0.0])
        return target_pos

    def reset(
            self,
            rng: jnp.ndarray
            ) -> State:
        qpos = jnp.copy(self.mjx_model.qpos0)
        qvel = jnp.zeros(self.mjx_model.nv)

        # set robot pos
        morphology_joint = self.mj_model.body("morphology/").jntadr[0]
        qpos = qpos.at[morphology_joint + 2].set(1.0)

        # set random target pos
        target_id = self.mj_model.body("target/").id
        target_pos = self._get_random_target_position(rng=rng)
        mjx_model = self.mjx_model.replace(body_pos=self.mjx_model.body_pos.at[target_id].set(target_pos))

        data = self.pipeline_init(model=mjx_model, qpos=qpos, qvel=qvel)

        obs = self._get_obs(data=data)
        reward = jnp.zeros(1)
        done = self._get_done(data=data)
        metrics = {}

        return State(model=mjx_model, data=data, obs=obs, reward=reward, done=done, metrics=metrics)

    def step(
            self,
            state: State,
            action: jnp.ndarray
            ) -> State:
        data0 = state.data
        data = self.pipeline_step(model=state.model, data=data0, ctrl=action)

        obs = self._get_obs(data=data)
        reward = self._get_reward(data=data, data0=data0)
        done = self._get_done(data=data)

        return state.replace(data=data, obs=obs, reward=reward, done=done)

    def _get_target_position(
            self,
            data: mjx.Data
            ) -> jnp.ndarray:
        target_id = self.mj_model.body("target/").id
        target_pos = data.xpos[target_id]
        return target_pos

    def _direction_to_target(
            self,
            data: mjx.Data
            ) -> jnp.ndarray:
        target_position = self._get_target_position(data=data)
        robot_position = self._get_robot_position(data=data)

        direction = target_position - robot_position
        distance = self._distance_to_target(data=data)
        unit_direction = direction / distance

        return unit_direction

    def _distance_to_target(
            self,
            data: mjx.Data
            ) -> jnp.ndarray:
        target_position = self._get_target_position(data=data)
        robot_position = self._get_robot_position(data=data)

        return jnp.linalg.norm(target_position - robot_position)

    def _get_reward(
            self,
            data: mjx.Data,
            data0: mjx.Data
            ) -> jnp.ndarray:
        current_distance_to_target = self._distance_to_target(data=data)
        previous_distance_to_target = self._distance_to_target(data=data0)
        distance_delta = previous_distance_to_target - current_distance_to_target
        return distance_delta

    def _get_robot_position(
            self,
            data: mjx.Data
            ) -> jnp.ndarray:
        torso_id = self.mj_model.body("morphology/torso").id
        torso_pos = data.xpos[torso_id]
        return torso_pos

    def _get_robot_rotation(
            self,
            data: mjx.Data
            ) -> jnp.ndarray:
        torso_id = self.mj_model.body("morphology/torso").id
        torso_quat = data.xquat[torso_id]
        euler = Rotation.from_quat(quat=torso_quat).as_euler(seq="xyz")
        return euler

    def _get_robot_vel(
            self,
            data: mjx.Data
            ) -> jnp.ndarray:
        torso_joint = self.mj_model.body("morphology/").jntadr[0]
        torso_vel = data.qvel[torso_joint: torso_joint + 6]
        return torso_vel

    def _get_segment_joint_ids(
            self
            ) -> List[int]:
        joints = [joint.full_identifier for joint in self._morphology.mjcf_body.find_all("joint") if "segment" in
                  joint.name]
        return jnp.array([self.mj_model.joint(joint).id for joint in joints])

    def _get_segment_joint_pos(
            self,
            data: mjx.Data
            ) -> jnp.ndarray:
        joint_ids = self._get_segment_joint_ids()
        joint_pos = data.qpos[joint_ids]
        return joint_pos

    def _get_segment_joint_vel(
            self,
            data: mjx.Data
            ) -> jnp.ndarray:
        joint_ids = self._get_segment_joint_ids()
        joint_vel = data.qvel[joint_ids]
        return joint_vel

    def _get_obs(
            self,
            data: mjx.Data
            ) -> jnp.ndarray:
        robot_rotation = self._get_robot_rotation(data=data)
        robot_vel = self._get_robot_vel(data=data)
        segment_joint_pos = self._get_segment_joint_pos(data=data)
        segment_joint_vel = self._get_segment_joint_vel(data=data)
        direction_to_target = self._direction_to_target(data=data)
        return jnp.concatenate([robot_rotation, robot_vel, segment_joint_pos, segment_joint_vel, direction_to_target])

    def _get_done(
            self,
            data: mjx.Data
            ) -> bool:
        return data.time > 10


if __name__ == '__main__':
    env = ToyExampleEnv()
    # policy_fn = lambda \
    #         _: jnp.array(np.array(np.random.uniform(low=-0.349, high=0.349, size=env.action_size)))
    # run_fixed_viewer(env=env, policy_fn=policy_fn, seed=42)  # run_passive_viewer(env=env, policy_fn=policy_fn, seed=42)
