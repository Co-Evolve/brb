from typing import Generator

from brb.toy_example.morphology.morphology import MJCToyExampleMorphology
from brb.toy_example.morphology.specification.default import default_toy_example_morphology_specification
from brb.toy_example.task.move_to_target import MoveToTargetTaskConfiguration
from brb.utils.video import show_video

if __name__ == '__main__':
    morphology_specification = default_toy_example_morphology_specification(
            num_arms=4, num_segments_per_arm=3
            )

    morphology = MJCToyExampleMorphology(specification=morphology_specification)

    task_config = MoveToTargetTaskConfiguration()
    gym_env = task_config.environment(
            morphology=morphology, wrap2gym=True
            )

    observation_space = gym_env.observation_space
    action_space = gym_env.action_space


    def run_episode() -> Generator:
        global gym_env, action_space

        done = False
        obs, _ = gym_env.reset()

        while not done:
            action = action_space.sample()
            obs, reward, terminated, truncated, info = gym_env.step(action=action)
            done = terminated or truncated
            yield gym_env.render(camera_ids=[1])

    show_video(frame_generator=run_episode())
