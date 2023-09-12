from __future__ import annotations

from typing import Callable, Tuple, cast

from dm_control import viewer

from brb.brittle_star.morphology.morphology import MJCBrittleStarMorphology
from brb.brittle_star.morphology.specification.damage import remove_random_segment_groups
from brb.brittle_star.morphology.specification.default import default_brittle_star_morphology_specification
from brb.brittle_star.morphology.specification.specification import BrittleStarMorphologySpecification
from brb.brittle_star.task.light_evasion import LightEvasionTask, LightEvasionTaskConfiguration


class LightEvasionWithDamageTask(LightEvasionTask):
    def __init__(
            self,
            config: LightEvasionWithDamageTaskConfiguration,
            morphology: MJCBrittleStarMorphology
            ) -> None:
        super().__init__(config=config, morphology=morphology)
        self._base_specification = morphology.morphology_specification
        self.config = cast(LightEvasionWithDamageTaskConfiguration, self.config)

    def _apply_damage_to_morphology(
            self
            ) -> None:
        if not self.config.damage_fn:
            return

        # Detach morphology
        self._morphology.detach()

        # Create a new one with random damage
        morphology_specification = self.config.damage_fn(self._base_specification)
        morphology = MJCBrittleStarMorphology(
                morphology_specification
                )

        # Attach the morphology
        self._morphology = self._attach_morphology(morphology)

        # Configure observables
        self._segment_position_sensors = None
        self._configure_morphology_observables()
        self._task_observables = self._configure_task_observables()

    def initialize_episode_mjcf(
            self,
            random_state
            ):
        self._apply_damage_to_morphology()


class LightEvasionWithDamageTaskConfiguration(LightEvasionTaskConfiguration):
    def __init__(
            self,
            damage_fn: Callable[[BrittleStarMorphologySpecification], BrittleStarMorphologySpecification] | None = None,
            env_id: int = 0,
            arena_size: Tuple[int, int] = (10, 5),
            light_noise: bool = True,
            hilly_terrain: bool = True,
            random_current: bool = True,
            random_friction: bool = True,
            starting_position: Tuple[int, int] = (-8.0, 0.0),
            touch_coloring: bool = False,
            light_coloring: bool = False,
            time_scale: float = 0.5,
            control_substeps: int = 1,
            simulation_time: int = 20, ) -> None:
        super().__init__(
                env_id=env_id,
                arena_size=arena_size,
                light_noise=light_noise,
                hilly_terrain=hilly_terrain,
                random_current=random_current,
                random_friction=random_friction,
                starting_position=starting_position,
                touch_coloring=touch_coloring,
                light_coloring=light_coloring,
                time_scale=time_scale,
                control_substeps=control_substeps,
                simulation_time=simulation_time
                )
        self.damage_fn = damage_fn
        self.task = LightEvasionWithDamageTask


if __name__ == '__main__':
    def damage_fn(
            morph_spec
            ):
        return remove_random_segment_groups(morph_spec, arm_shortening_probability=0.9)


    # Define the task
    task_config = LightEvasionWithDamageTaskConfiguration(
            damage_fn=damage_fn,
            hilly_terrain=False,
            light_noise=False,
            random_friction=False,
            random_current=False,
            simulation_time=5
            )

    # Define the morphology
    morphology_specification = default_brittle_star_morphology_specification(
            num_arms=5, num_segments_per_arm=12, use_p_control=True
            )

    morphology = MJCBrittleStarMorphology(
            morphology_specification
            )

    # Create the environment
    env = task_config.environment(
            morphology=morphology, wrap2gym=False
            )
    viewer.launch(env)
