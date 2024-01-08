import jax
import jax.numpy as jnp
from mujoco_utils.environment.mjx_env import MJXGymEnvWrapper

from brb.toy_example.mjc.env import ToyExampleEnvironmentConfiguration
from brb.toy_example.mjc.example_usage_single import post_render
from brb.toy_example.mjx.example_usage_batch import get_action_generator
from brb.toy_example.mjx.example_usage_single import create_mjx_environment

if __name__ == '__main__':
    num_envs = 2
    environment_configuration = ToyExampleEnvironmentConfiguration(
            render_mode="rgb_array", camera_ids=[0, 1], randomization_noise_scale=0.01
            )
    mjx_env = create_mjx_environment(environment_configuration=environment_configuration)
    gym_env = MJXGymEnvWrapper(env=mjx_env, num_envs=num_envs)

    get_actions = get_action_generator(action_space=gym_env.single_action_space)
    if num_envs > 1:
        get_actions = jax.jit(jax.vmap(get_actions))

    obs, info = gym_env.reset()

    done = False
    steps = 0
    fps = 60
    while not done:
        t = info["time"]
        actions = get_actions(t)
        obs, reward, terminated, truncated, info = gym_env.step(actions=actions)

        done = jnp.any((terminated | truncated))
        if steps % int((1 / fps) / environment_configuration.control_timestep) == 0:
            post_render(gym_env.render(), environment_configuration=environment_configuration)
            print(obs["segment_ground_contact"])

        steps += 1
        if done:
            gym_env.reset()
            post_render(gym_env.render(), environment_configuration=environment_configuration)
            done = False
            steps = 0
    gym_env.close()
