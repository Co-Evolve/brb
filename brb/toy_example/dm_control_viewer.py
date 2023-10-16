import numpy as np
from dm_control import viewer
from dm_env import TimeStep

from brb.toy_example.morphology.morphology import MJCToyExampleMorphology
from brb.toy_example.morphology.specification.default import default_toy_example_morphology_specification
from brb.toy_example.task.move_to_target import MoveToTargetTaskConfiguration

if __name__ == '__main__':
    morphology_specification = default_toy_example_morphology_specification(
            num_arms=4, num_segments_per_arm=3
            )

    morphology = MJCToyExampleMorphology(specification=morphology_specification)

    task_config = MoveToTargetTaskConfiguration()
    dm_env = task_config.environment(
            morphology=morphology, wrap2gym=False
            )

    observation_spec = dm_env.observation_spec()
    action_spec = dm_env.action_spec()


    def oscillator_policy_fn(
            timestep: TimeStep
            ) -> np.ndarray:
        global action_spec
        time = timestep.observation["task/time"][0]

        num_actuators = action_spec.shape[0]
        actions = np.zeros(num_actuators)

        in_plane_actions = np.cos(time)
        out_of_plane_actions = np.sin(time)

        actions[0::2] = in_plane_actions
        actions[1::2] = out_of_plane_actions

        # rescale from [-1, 1] to actual joint range
        minimum, maximum = action_spec.minimum, action_spec.maximum

        normalised_actions = (actions + 1) / 2

        scaled_actions = minimum + normalised_actions * (maximum - minimum)

        return scaled_actions


    viewer.launch(
            environment_loader=dm_env, policy=oscillator_policy_fn
            )
