import functools

import gymnasium.vector
import numpy as np

from brb.toy_example.mjc.env import ToyExampleEnvironmentConfiguration
from brb.toy_example.mjc.example_usage_single import create_mjc_environment, post_render


def create_batch_mjc_environment(
        n_envs: int,
        environment_configuration: ToyExampleEnvironmentConfiguration
        ) -> gymnasium.vector.AsyncVectorEnv:

    batched_env = gymnasium.vector.AsyncVectorEnv(
            env_fns=[functools.partial(create_mjc_environment, environment_configuration=environment_configuration) for
                     _ in range(n_envs)]
            )
    return batched_env


if __name__ == '__main__':
    n_envs = 2
    environment_configuration = ToyExampleEnvironmentConfiguration(
            render_mode="human", camera_ids=[0, 1], randomization_noise_scale=0.01
            )
    batched_env = create_batch_mjc_environment(n_envs=n_envs, environment_configuration=environment_configuration)
    obs, info = batched_env.reset()

    done = False
    steps = 0
    fps = 60
    while not done:
        ts = np.array(info["time"])
        batched_action = np.ones(batched_env.action_space.shape)
        batched_action[:, ::2] = np.cos(5 * ts).reshape((n_envs, 1))
        batched_action[:, 1::2] = np.sin(5 * ts).reshape((n_envs, 1))

        batched_action[:, -batched_action.shape[1] // 2::2] *= -1
        obs, reward, terminated, truncated, info = batched_env.step(actions=batched_action)
        done = np.any(terminated | truncated)
        if steps % int((1 / fps) / environment_configuration.control_timestep) == 0:
            post_render(batched_env.call(name="render"), environment_configuration=environment_configuration)
            print(obs["segment_ground_contact"])

        steps += 1
        if done:
            batched_env.reset()
            post_render(batched_env.call(name="render"), environment_configuration=environment_configuration)
            done = False
            steps = 0
    batched_env.close()
