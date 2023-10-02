import copy
from pathlib import Path

import cv2
import gymnasium as gym
import numpy as np
import pandas as pd
import ray
import seaborn as sns
from dm_control.rl.control import PhysicsError
from fprs.parameters import FixedParameter
from matplotlib import pyplot as plt
from tqdm import tqdm

from brb.seahorse.morphology.morphology import MJCSeahorseMorphology
from brb.seahorse.morphology.specification.default import default_seahorse_morphology_specification
from brb.seahorse.task.grasping import GraspingTaskConfiguration
from brb.utils.video import create_video


def create_env(
        num_segments: int,
        segment_span: int,
        tendon_translation: float
        ) -> gym.Env:
    env_config = GraspingTaskConfiguration(with_object=False)

    morphology_specification = default_seahorse_morphology_specification(
            num_segments=num_segments, hmm_segment_span=segment_span
            )
    morphology_specification.tendon_actuation_specification.hmm_tendon_actuation_specification.tendon_strain = (
            FixedParameter(tendon_translation))
    morphology = MJCSeahorseMorphology(specification=morphology_specification)
    env = env_config.environment(
            morphology=morphology, wrap2gym=True
            )
    return env


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
    if num_hmm_tendons_per_corner_to_contract > 0:
        hmm_actions[:2, -num_hmm_tendons_per_corner_to_contract:] = hmm_minimum.reshape(
                4, num_hmm_tendons_per_corner
                )[:2, -num_hmm_tendons_per_corner_to_contract:]
    hmm_actions = hmm_actions.flatten()

    if use_mvm:
        num_mvm_tendons_to_contract = int(num_mvm_tendons * rel_time)
        mvm_actions = copy.deepcopy(mvm_maximum)
        if num_mvm_tendons_to_contract > 0:
            mvm_actions[-num_mvm_tendons_to_contract:] = mvm_minimum[-num_mvm_tendons_to_contract:]
    else:
        mvm_actions = mvm_maximum

    actions = np.concatenate((hmm_actions, mvm_actions))
    return actions


@ray.remote(num_cpus=1)
def evaluate_ventral_curvature(
        tendon_translation: float,
        num_segments: int,
        segment_span: int,
        use_mvm: bool,
        video: bool,
        save_last_frame: bool,
        output_path: str
        ) -> None:
    env = create_env(num_segments, segment_span, tendon_translation)
    obs, _ = env.reset()

    done = False
    frames = []
    max_ventral_curvature = -1

    while not done:
        time = obs["task/time"][0]

        actions = grasping_policy_fn(
                time=time, action_spec=env._env.action_spec(), num_segments=num_segments, use_mvm=use_mvm
                )
        try:
            obs, reward, terminated, truncated, _ = env.step(actions)
        except PhysicsError:
            return

        ventral_curvature = np.sum(obs["morphology/vertebrae_coronal_joint_pos"]) / np.pi * 180
        if ventral_curvature > max_ventral_curvature:
            max_ventral_curvature = ventral_curvature

        if video:
            frames.append(env.render(camera_ids=[0]))
        done = terminated or truncated

    output_file_name = (f"num_segments_{num_segments}"
                        f"_segment_span_{segment_span}"
                        f"_mvm_{'enabled' if use_mvm else 'disabled'}"
                        f"_ventral_curvature_{max_ventral_curvature:.2f}")

    if save_last_frame:
        if video:
            frame = frames[-1]
        else:
            frame = env.render(camera_ids=[0]).astype(np.uint8)
        cv2.imwrite(f"{output_path}/{output_file_name}.png", frame)
    if video:
        create_video(
                frames=frames, framerate=int(len(frames) / 5), out_path=f"{output_path}/{output_file_name}.mp4"
                )

    return num_segments, segment_span, max_ventral_curvature


if __name__ == '__main__':
    redo_simulation = False
    use_mvm = False
    render_last_frame = True
    render_video = False
    min_num_segments = 15
    max_num_segments = 30 + 1
    min_segment_span = 1
    max_segment_span = 11 + 1

    tendon_translations = [0.005, 0.01, 0.015, 0.02]

    grids = []
    for tendon_translation in tendon_translations:
        output_path = f"./output/tendon_translation_{tendon_translation}"
        Path(output_path).mkdir(exist_ok=True, parents=True)

        if redo_simulation:
            ray.init(num_cpus=10, ignore_reinit_error=True)
            result_refs = []
            for num_segments in range(min_num_segments, max_num_segments):
                for segment_span in range(min_segment_span, max_segment_span):
                    result_refs.append(
                            evaluate_ventral_curvature.remote(tendon_translation,
                                                              num_segments, segment_span, use_mvm,
                                                              render_video,
                                                              render_last_frame,
                                                              output_path)
                            )

            num_segments, segment_span, ventral_curvature = [], [], []
            for i in tqdm(range(len(result_refs))):
                result = ray.get(result_refs[i])
                if result is None:
                    continue
                ns, ss, vc = result
                num_segments.append(ns)
                segment_span.append(ss)
                ventral_curvature.append(vc)

            # Save data as csv
            data = {
                    "number of segments": num_segments, "tendon segment span": segment_span,
                    "ventral curvature": ventral_curvature}
            df = pd.DataFrame(data)
            df.to_csv(f"{output_path}/ventral_curvature.csv", index=False)

        data = pd.read_csv(f"{output_path}/ventral_curvature.csv").to_dict(orient="list")

        # Visualise as matrix
        x_dim = max_segment_span - min_segment_span
        y_dim = max_num_segments - min_num_segments
        grid = np.ones((y_dim, x_dim)) * np.inf

        for num_segments, segment_span, ventral_curvature in zip(
                data["number of segments"], data["tendon segment span"], data["ventral curvature"]
                ):
            grid[num_segments - min_num_segments, segment_span - min_segment_span] = ventral_curvature

        valid_mask = grid != np.inf
        invalid_mask = grid == np.inf

        ax = sns.heatmap(
                grid,
                mask=invalid_mask,
                linewidth=0.,
                xticklabels=list(range(min_segment_span, max_segment_span)),
                yticklabels=list(range(min_num_segments, max_num_segments)),
                vmin=np.min(grid[valid_mask]),
                vmax=np.max(grid[valid_mask])
                )
        ax.set_xlabel("tendon segment span")
        ax.set_ylabel("number of segments")
        ax.invert_yaxis()

        plt.savefig(f"{output_path}/heatmap.png")
        plt.close()

        grids.append(grid)

    segment_index = 20 - min_num_segments
    rows = []
    for grid in grids:
        row = grid[segment_index]
        mask = row != np.inf
        valid = row[mask]
        row[mask] = (valid - np.min(valid)) / (np.max(valid) - np.min(valid))
        rows.append(row)
    grid = np.stack(rows)

    ax = sns.heatmap(
            grid,
            mask=grid == np.inf,
            linewidth=0.,
            xticklabels=list(range(min_segment_span, max_segment_span)),
            yticklabels=tendon_translations,
            vmin=np.min(grid[grid != np.inf]),
            vmax=np.max(grid[grid != np.inf])
            )
    ax.set_xlabel("tendon segment span")
    ax.set_ylabel("tendon translation")
    ax.invert_yaxis()

    plt.savefig(f"20segments.png")
    plt.close()