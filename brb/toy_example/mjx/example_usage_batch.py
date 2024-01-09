from typing import Callable

import gymnasium
import jax
import jax.numpy as jnp

from brb.toy_example.mjc.env import ToyExampleEnvironmentConfiguration
from brb.toy_example.mjc.example_usage_single import post_render
from brb.toy_example.mjx.example_usage_single import create_mjx_environment, create_mjx_open_loop_controller

if __name__ == '__main__':
    environment_configuration = ToyExampleEnvironmentConfiguration(
            render_mode="rgb_array", camera_ids=[0, 1], randomization_noise_scale=0.1
            )
    env = create_mjx_environment(environment_configuration=environment_configuration)

    num_envs = 2
    rng = jax.random.PRNGKey(0)
    rng, *subrngs = jax.random.split(key=rng, num=num_envs + 1)

    jit_batched_reset = jax.jit(jax.vmap(env.reset))
    jit_batched_step = jax.jit(jax.vmap(env.step))
    batched_state = jit_batched_reset(rng=jnp.array(subrngs))

    controller = create_mjx_open_loop_controller(
            single_action_space=env.action_space, num_envs=num_envs
            )

    done = False
    steps = 0
    fps = 60
    while not done:
        ts = batched_state.info["time"]
        actions = controller(ts)

        batched_state = jit_batched_step(state=batched_state, actions=actions)
        done = jnp.any((batched_state.terminated | batched_state.truncated))
        if steps % int((1 / fps) / environment_configuration.control_timestep) == 0:
            post_render(
                env.render(batched_state), environment_configuration=environment_configuration
                )
            print(batched_state.observations["segment_ground_contact"])

        steps += 1
        if done:
            rng, *subrngs = jax.random.split(key=rng, num=num_envs + 1)
            batched_state = jit_batched_reset(rng=jnp.array(subrngs))
            post_render(env.render(state=batched_state), environment_configuration=environment_configuration)
            done = False
            steps = 0
    env.close()
