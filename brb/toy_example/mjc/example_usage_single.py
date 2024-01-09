from typing import Callable, Tuple, Union

import cv2
import gymnasium
import numpy as np

from brb.toy_example.arena.arena import MJCFPlaneWithTargetArena, PlaneWithTargetArenaConfiguration
from brb.toy_example.mjc.env import ToyExampleEnvironmentConfiguration, ToyExampleMJCEnvironment
from brb.toy_example.morphology.morphology import MJCFToyExampleMorphology
from brb.toy_example.morphology.specification.default import default_toy_example_morphology_specification


def create_mjc_environment(
        environment_configuration: ToyExampleEnvironmentConfiguration
        ) -> ToyExampleMJCEnvironment:
    morphology_specification = default_toy_example_morphology_specification(num_arms=4, num_segments_per_arm=2)
    morphology = MJCFToyExampleMorphology(specification=morphology_specification)
    arena_configuration = PlaneWithTargetArenaConfiguration()
    arena = MJCFPlaneWithTargetArena(configuration=arena_configuration)

    env = ToyExampleMJCEnvironment(
            morphology=morphology, arena=arena, configuration=environment_configuration
            )
    return env


def post_render(
        render_output: Union[None, Tuple[np.ndarray]],
        environment_configuration: ToyExampleEnvironmentConfiguration
        ) -> None:
    if environment_configuration.render_mode == "human":
        return

    if len(environment_configuration.camera_ids) > 1:
        render_output = [np.concatenate(env_frames, axis=1) for env_frames in render_output]

    stacked_frames = np.concatenate(render_output, axis=0)
    cv2.imshow("render", stacked_frames)
    cv2.waitKey(1)


def create_mjc_open_loop_controller(
        single_action_space: gymnasium.spaces.Box,
        num_envs: int
        ) -> Callable[[float], np.ndarray]:
    def open_loop_controller(
            t: float
            ) -> np.ndarray:
        actions = np.ones(single_action_space.shape)
        actions[::2] = np.cos(5 * t)
        actions[1::2] = np.sin(5 * t)
        actions[-actions.shape[0] // 2::2] *= -1
        return actions

    if num_envs > 1:
        batched_open_loop_controller = lambda \
                t: np.stack([open_loop_controller(tt) for tt in t])
        return batched_open_loop_controller
    return open_loop_controller


if __name__ == '__main__':
    environment_configuration = ToyExampleEnvironmentConfiguration(
            render_mode="human", camera_ids=[0, 1]
            )
    env = create_mjc_environment(environment_configuration=environment_configuration)
    obs, info = env.reset()

    controller = create_mjc_open_loop_controller(
            single_action_space=env.action_space, num_envs=1
            )
    done = False
    steps = 0
    fps = 60
    while not done:
        t = info["time"]
        actions = controller(t)

        obs, reward, terminated, truncated, info = env.step(actions=actions)
        done = terminated or truncated
        if steps % int((1 / fps) / environment_configuration.control_timestep) == 0:
            post_render(env.render(), environment_configuration=environment_configuration)
            print(obs["segment_ground_contact"])

        steps += 1
        if done:
            env.reset()
            post_render(env.render(), environment_configuration=environment_configuration)
            done = False
            steps = 0
    env.close()
