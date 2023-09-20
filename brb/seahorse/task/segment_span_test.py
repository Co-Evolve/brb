import copy
from typing import List

import cv2
import gymnasium as gym
import numpy as np
import ray
from dm_control.rl.control import PhysicsError
from fprs.parameters import FixedParameter

from brb.seahorse.morphology.morphology import MJCSeahorseMorphology
from brb.seahorse.morphology.specification.default import default_seahorse_morphology_specification
from brb.seahorse.task.grasping import GraspingTaskConfiguration


def create_env(
        segment_span: int
        ) -> gym.Env:
    env_config = GraspingTaskConfiguration(with_object=False)

    morphology_specification = default_seahorse_morphology_specification(num_segments=30)
    morphology_specification.tendon_actuation_specification.segment_span = FixedParameter(segment_span)
    morphology = MJCSeahorseMorphology(specification=morphology_specification)
    env = env_config.environment(
            morphology=morphology, wrap2gym=True
            )
    return env


def create_video(
        frames: List[np.ndarray],
        framerate: float,
        out_path: str
        ) -> None:
    height, width, _ = frames[0].shape
    size = (width, height)

    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), framerate, size)
    for frame in frames:
        writer.write(frame)
    writer.release()


def grasping_policy_fn(
        time: float,
        action_spec
        ) -> np.ndarray:
    num_tendons_per_side = int(action_spec.shape[0] / 2)
    num_tendons_per_corner = int(num_tendons_per_side / 2)

    SECONDS_TO_FULL_CONTRACTION = 3

    rel_time = np.clip(time / SECONDS_TO_FULL_CONTRACTION, 0, 1)

    # ventral-dorsal, dextral-sinistral
    actions = copy.deepcopy(action_spec.maximum.reshape(4, num_tendons_per_corner))

    num_tendons_per_corner_to_contract = int(num_tendons_per_corner * rel_time)
    actions[:2, :num_tendons_per_corner_to_contract] = action_spec.minimum.reshape(4, num_tendons_per_corner)[:2,
                                                       :num_tendons_per_corner_to_contract]

    return actions.flatten()


@ray.remote(num_cpus=1)
def evaluate_segment_span(
        segment_span: int
        ) -> None:
    env = create_env(segment_span)
    obs, _ = env.reset()

    done = False
    frames = []
    while not done:
        time = obs["task/time"][0]
        print(f"Percentage: {time / 5}")

        actions = grasping_policy_fn(time=time, action_spec=env._env.action_spec())
        try:
            obs, reward, terminated, truncated, _ = env.step(actions)
        except PhysicsError:
            return
        frames.append(env.render(camera_ids=[0]))
        done = terminated or truncated

    create_video(frames=frames, framerate=int(len(frames) / 5), out_path=f"segment_span_{segment_span}.mp4")


if __name__ == '__main__':
    ray.init(num_cpus=10)
    ray.get([evaluate_segment_span.remote(i) for i in range(1, 12)])
