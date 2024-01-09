import functools

import gymnasium.vector
import numpy as np

from brb.toy_example.mjc.env import ToyExampleEnvironmentConfiguration
from brb.toy_example.mjc.example_usage_single import create_mjc_environment, create_mjc_open_loop_controller, \
    post_render


def create_batch_mjc_environment(
        num_envs: int,
        environment_configuration: ToyExampleEnvironmentConfiguration
        ) -> gymnasium.vector.AsyncVectorEnv:

    batched_env = gymnasium.vector.AsyncVectorEnv(
            env_fns=[functools.partial(create_mjc_environment, environment_configuration=environment_configuration) for
                     _ in range(num_envs)]
            )
    return batched_env


if __name__ == '__main__':
    num_envs = 2
    environment_configuration = ToyExampleEnvironmentConfiguration(
            render_mode="human", camera_ids=[0, 1], randomization_noise_scale=0.01
            )
    batched_env = create_batch_mjc_environment(num_envs=num_envs, environment_configuration=environment_configuration)
    obs, info = batched_env.reset()

    controller = create_mjc_open_loop_controller(
            single_action_space=batched_env.single_action_space, num_envs=num_envs
            )

    done = False
    steps = 0
    fps = 60
    while not done:
        ts = info["time"]
        print(ts)
        batched_actions = controller(ts)

        obs, reward, terminated, truncated, info = batched_env.step(actions=batched_actions)
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
