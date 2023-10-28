import jax
import mujoco
import mujoco.viewer
import numpy as np
from dm_control.mjcf import export_with_assets
from jax import numpy as jnp
from mujoco import mjx

from brb.toy_example.mjx.env import MjxEnv
from brb.toy_example.mjx.state import State
from brb.toy_example.mjx.utils import run_passive_viewer
from brb.toy_example.morphology.morphology import MJCToyExampleMorphology
from brb.toy_example.morphology.specification.default import default_toy_example_morphology_specification
from brb.toy_example.task.move_to_target import MoveToTargetTask, MoveToTargetTaskConfiguration


class ToyExampleEnv(MjxEnv):
    def __init__(
            self
            ) -> None:
        super().__init__(self._get_mjModel())

        self._previous_distance_to_target = 0

    def _get_mjModel(
            self
            ) -> str:
        morphology_specification = default_toy_example_morphology_specification(
                num_arms=4, num_segments_per_arm=3
                )
        morphology = MJCToyExampleMorphology(specification=morphology_specification)
        task_config = MoveToTargetTaskConfiguration()
        self.task: MoveToTargetTask = task_config.task(task_config, morphology)
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
        reward = self._get_reward(data=data)
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
        reward = self._get_reward(data=data)
        done = self._get_done(data=data)

        return state.replace(data=data, obs=obs, reward=reward, done=done)


    def _direction_to_target(self, data: mjx.Data) -> jnp.ndarray:
        pass

    def _distance_to_target(self, data: mjx.Data) -> float:
        pass

    def _get_reward(
            self,
            data: mjx.Data
            ) -> float:
        # Calculate distance between com of robot and target
        return 0

    def _get_obs(
            self,
            data: mjx.Data
            ) -> jnp.ndarray:
        # joint positions and velocities
        return jnp.zeros(10)

    def _get_done(
            self,
            data: mjx.Data
            ) -> bool:
        return data.time > 10


if __name__ == '__main__':
    env = ToyExampleEnv()
    policy_fn = lambda \
            _: jnp.array(np.array(np.random.uniform(low=-0.349, high=0.349, size=env.action_size)))
    run_passive_viewer(env=env, policy_fn=policy_fn, seed=42)
