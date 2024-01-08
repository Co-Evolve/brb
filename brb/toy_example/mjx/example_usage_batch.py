from typing import Callable

import gymnasium
import jax
import jax.numpy as jnp

from brb.toy_example.mjc.env import ToyExampleEnvironmentConfiguration
from brb.toy_example.mjc.example_usage_single import post_render
from brb.toy_example.mjx.example_usage_single import create_mjx_environment


def get_action_generator(
        action_space: gymnasium.spaces.Box
        ) -> Callable[[float], jnp.ndarray]:
    def action_generator(
            t: float
            ) -> jnp.ndarray:
        action = jnp.ones(action_space.shape)
        action = action.at[jnp.arange(0, len(action), 2)].set(jnp.cos(5 * t))
        action = action.at[jnp.arange(1, len(action), 2)].set(jnp.sin(5 * t))
        action = action.at[jnp.arange(len(action) // 2, len(action), 2)].set(
                action[jnp.arange(len(action) // 2, len(action), 2)] * -1
                )
        return action

    return action_generator


if __name__ == '__main__':
    environment_configuration = ToyExampleEnvironmentConfiguration(
            render_mode="rgb_array", camera_ids=[0, 1], randomization_noise_scale=0.1
            )
    env = create_mjx_environment(environment_configuration=environment_configuration)

    n_envs = 2
    rng = jax.random.PRNGKey(0)
    rng, *subrngs = jax.random.split(key=rng, num=n_envs + 1)

    jit_batched_reset = jax.jit(jax.vmap(env.reset))
    jit_batched_step = jax.jit(jax.vmap(env.step))

    jit_batched_get_actions = jax.jit(jax.vmap(get_action_generator(env.action_space)))

    batched_state = jit_batched_reset(rng=jnp.array(subrngs))

    done = False
    steps = 0
    fps = 60
    while not done:
        ts = batched_state.info["time"]
        batched_action = jit_batched_get_actions(ts)

        batched_state = jit_batched_step(state=batched_state, action=batched_action)
        done = jnp.any((batched_state.terminated | batched_state.truncated))
        if steps % int((1 / fps) / environment_configuration.control_timestep) == 0:
            post_render(env.render(batched_state), environment_configuration=environment_configuration)
            print(batched_state.observations["segment_ground_contact"])

        steps += 1
        if done:
            rng, *subrngs = jax.random.split(key=rng, num=n_envs + 1)
            batched_state = jit_batched_reset(rng=jnp.array(subrngs))
            post_render(env.render(state=batched_state), environment_configuration=environment_configuration)
            done = False
            steps = 0
    env.close()
