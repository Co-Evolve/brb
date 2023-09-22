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
        num_segments: int,
        segment_span: int
        ) -> gym.Env:
    env_config = GraspingTaskConfiguration(with_object=False)

    morphology_specification = default_seahorse_morphology_specification(num_segments=num_segments)
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
        action_spec,
        num_segments: int,
        use_mvm: bool
        ) -> np.ndarray:
    SECONDS_TO_FULL_CONTRACTION = 4

    num_mvm_tendons = num_segments - 1
    num_hmm_tendons = action_spec.shape[0] - num_mvm_tendons
    num_hmm_tendons_per_side = int(num_hmm_tendons / 2)
    num_hmm_tendons_per_corner = int(num_hmm_tendons_per_side / 2)

    hmm_minimum, hmm_maximum = action_spec.minimum[:num_hmm_tendons], action_spec.maximum[:num_hmm_tendons]
    mvm_minimum, mvm_maximum = action_spec.minimum[-num_mvm_tendons:], action_spec.maximum[-num_mvm_tendons:]

    rel_time = np.clip(time / SECONDS_TO_FULL_CONTRACTION, 0, 1)

    # ventral-dorsal, dextral-sinistral
    hmm_actions = copy.deepcopy(hmm_maximum.reshape(4, num_hmm_tendons_per_corner))

    num_hmm_tendons_per_corner_to_contract = int(num_hmm_tendons_per_corner * rel_time)
    hmm_actions[:2, :num_hmm_tendons_per_corner_to_contract] = hmm_minimum.reshape(
            4, num_hmm_tendons_per_corner
            )[:2, :num_hmm_tendons_per_corner_to_contract]
    hmm_actions = hmm_actions.flatten()

    if use_mvm:
        num_mvm_tendons_to_contract = int(num_mvm_tendons * rel_time)
        mvm_actions = np.concatenate(
                (mvm_maximum[:-num_mvm_tendons_to_contract], mvm_minimum[-num_mvm_tendons_to_contract:])
                )
    else:
        mvm_actions = mvm_maximum

    actions = np.concatenate((hmm_actions, mvm_actions))
    return actions


@ray.remote(num_cpus=1)
def evaluate_segment_span(
        num_segments: int,
        segment_span: int,
        use_mvm: bool
        ) -> None:
    env = create_env(num_segments, segment_span)
    obs, _ = env.reset()

    done = False
    frames = []
    while not done:
        time = obs["task/time"][0]
        print(f"Percentage: {time / 5}")

        actions = grasping_policy_fn(
            time=time, action_spec=env._env.action_spec(), num_segments=num_segments, use_mvm=use_mvm
            )
        try:
            obs, reward, terminated, truncated, _ = env.step(actions)
        except PhysicsError:
            return
        frames.append(env.render(camera_ids=[0]))
        done = terminated or truncated

    create_video(
        frames=frames, framerate=int(len(frames) / 5), out_path=f"num_segments_{num_segments}_segment_span"
                                                                f"_{segment_span}_mvm_"
                                                                f"{'enabled' if use_mvm else 'disabled'}.mp4"
        )


if __name__ == '__main__':
    ray.init(num_cpus=10)
    num_segments = 15
    use_mvm = True
    ray.get([evaluate_segment_span.remote(num_segments, segment_span, use_mvm) for segment_span in range(1, 12)])
