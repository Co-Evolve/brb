import gymnasium as gym

from brb.brittle_star.morphology.morphology import MJCBrittleStarMorphology
from brb.brittle_star.morphology.specification.default import default_brittle_star_morphology_specification
from brb.brittle_star.task.light_evasion import LightEvasionTaskConfiguration


def create_brittle_star_environment(
        env_id: int = 0,
        wrap2gym: bool = True
        ) -> gym.Env:

    # Define the task
    task_config = LightEvasionTaskConfiguration(
            env_id=env_id,
            hilly_terrain=False,
            light_noise=False,
            random_friction=False,
            random_current=False,
            simulation_time=5
            )

    # Define the morphology
    morphology_specification = default_brittle_star_morphology_specification(
            num_arms=5, num_segments_per_arm=5, use_p_control=False
            )

    morphology = MJCBrittleStarMorphology(
            morphology_specification
            )

    # Create the environment
    env = task_config.environment(
            morphology=morphology, wrap2gym=wrap2gym
            )

    return env


if __name__ == '__main__':
    env = create_brittle_star_environment()
    obs = env.reset()
    print('here')
