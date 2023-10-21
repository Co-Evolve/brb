import time

import jax
import mujoco
import mujoco.viewer
import numpy as np
from dm_control import mjcf
from dm_control.mjcf import export_with_assets
from jax import numpy as jnp
from mujoco import mjx

from brb.toy_example.mjx.env import MjxEnv
from brb.toy_example.mjx.state import State
from brb.toy_example.morphology.morphology import MJCToyExampleMorphology
from brb.toy_example.morphology.specification.default import default_toy_example_morphology_specification
from brb.toy_example.task.move_to_target import MoveToTargetTask, MoveToTargetTaskConfiguration


class ToyExampleEnv(MjxEnv):
    def __init__(
            self, ) -> None:
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
        super().__init__(mj_model)

    def reset(
            self,
            rng: jnp.ndarray
            ) -> State:
        morphology_joint = self.model.body("morphology/").jntadr[0]

        qpos = jnp.copy(self.sys.qpos0)
        qvel = jnp.zeros(self.sys.nv)

        # set robot pos
        qpos = qpos.at[morphology_joint + 2].set(1.0)

        # set random target pos
        rng, rng1 = jax.random.split(rng, 2)
        angle = jax.random.uniform(key=rng1, shape=(), minval=0, maxval=jnp.pi * 2)
        radius = 2.0
        target_pos = radius * jnp.array([jnp.cos(angle), jnp.sin(angle), 0.0])
        target_id = self.model.body("target/").id

        sys = self.sys.replace(body_pos=self.sys.body_pos.at[target_id].set(target_pos))

        sys, data = self.pipeline_init(sys=sys, qpos=qpos, qvel=qvel)

        obs = self._get_obs(data=data, action=jnp.zeros(self.sys.nu))
        reward = 0
        done = 0
        metrics = {}

        print("reset called!")
        return State(model=sys, data=data, obs=obs, reward=reward, done=done, metrics=metrics)

    def step(
            self,
            state: State,
            action: jnp.ndarray
            ) -> State:
        data0 = state.data
        data = self.pipeline_step(sys=state.model, data=data0, ctrl=action)

        obs = self._get_obs(data=data, action=jnp.zeros(self.sys.nu))
        reward = 0
        done = 0

        return state.replace(data=data, obs=obs, reward=reward, done=done, metrics={})

    def _get_reward(
            self,
            data: mjx.Data
            ) -> jnp.ndarray:
        pass

    def _get_obs(
            self,
            data: mjx.Data,
            action: jnp.ndarray
            ) -> jnp.ndarray:
        return jnp.zeros(10)


if __name__ == '__main__':
    env = ToyExampleEnv()

    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)

    state = jit_reset(jax.random.PRNGKey(seed=42))
    rollout = [state]


    # need to update the env.model as well in order to visualise correctly
    # So basically in order to dynamically change target position upon reset,
    #   we need to either do this or add joints to targets and set those
    #   problem when adding joints to targets is that the target starts moving towards origin for some reason
    env.model.body("target/").pos = state.data.xpos[1]
    mjx.device_get_into(result=env.data, value=state.data)
    print(state.data.xpos[1])
    print(env.data.xpos[1])
    steps = 0

    time.sleep(3)
    # todo: bugfix -> after one step the target starts moving to origin
    #   this only happens when we jit compile!

    with mujoco.viewer.launch_passive(model=env.model, data=env.data) as v:
        while True:
            # if steps % 500 == 0:
            #     state = jit_reset(jax.random.PRNGKey(seed=42))
            #
            state = jit_step(
                state=state,
                action=jnp.array(np.random.uniform(low=-0.349, high=0.349, size=env.action_size))
                )
            mjx.device_get_into(result=env.data, value=state.data)

            steps += 1

            print(f"Steps: {steps}\t| Time: {state.data.time}")
            print()
            v.sync()
