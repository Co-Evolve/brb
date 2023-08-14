from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from dm_control import composer
from dm_control import mjcf
from dm_control import viewer
from dm_control.composer.observation import observable
from dm_control.mujoco.math import euler2quat
from mujoco_utils.environment import MJCEnvironmentConfig
from mujoco_utils.observables import ConfinedObservable

from brb.brittle_star.arena.hilly_light_aquarium import HillyLightAquarium
from brb.brittle_star.morphology.morphology import MJCBrittleStarMorphology
from brb.brittle_star.morphology.specification.default import \
    default_arm_length_based_brittle_star_morphology_specification


class DirectedLocomotionTask(composer.Task):
    def __init__(
            self,
            config: DirectedLocomotionTaskConfiguration,
            morphology: MJCBrittleStarMorphology, ) -> None:
        self.config = config

        self._arena = self._build_arena()
        self._morphology = self._attach_morphology(morphology=morphology)
        self._task_observables = self._configure_observables()

        self._configure_contacts()
        self._configure_memory()

        self._previous_distance_travelled = 0

    @property
    def root_entity(
            self
            ) -> HillyLightAquarium:
        return self._arena

    @property
    def task_observables(
            self
            ) -> Dict:
        return self._task_observables

    def _build_arena(
            self
            ) -> HillyLightAquarium:
        arena = HillyLightAquarium(
                size=self.config.arena_size,
                env_id=self.config.env_id,
                light_texture=False,
                light_noise=False,
                hilly_terrain=self.config.hilly_terrain,
                random_current=self.config.random_current,
                random_friction=self.config.random_friction
                )
        return arena

    def _configure_contacts(
            self
            ) -> None:
        defaults = [self.root_entity.mjcf_model.default.geom, self._morphology.mjcf_model.default.geom]

        # Disable all collisions between geoms by default
        for geom_default in defaults:
            geom_default.contype = 1
            geom_default.conaffinity = 0

        for geom in self._arena.needs_collision:
            geom.conaffinity = 1

    def _configure_memory(
            self
            ) -> None:
        self.root_entity.mjcf_model.size.memory = "64M"

    def _attach_morphology(
            self,
            morphology: MJCBrittleStarMorphology
            ) -> MJCBrittleStarMorphology:
        self._arena.add_free_entity(
                entity=morphology
                )
        morphology.touch_coloring = self.config.touch_coloring
        return morphology

    def _configure_morphology_observables(
            self
            ) -> None:
        self._morphology.observables.enable_all()

    def _get_morphology_xy_position(
            self,
            physics: mjcf.Physics
            ) -> np.ndarray:
        position, _ = self._morphology.get_pose(
                physics=physics
                )
        return np.array(
                position
                )[:2]

    def _get_distance_travelled_along_x_axis(
            self,
            physics: mjcf.Physics
            ) -> float:
        x_position = self._get_morphology_xy_position(physics=physics)[0]
        starting_x_position = self.config.starting_position[0]
        return x_position - starting_x_position

    def _configure_task_observables(
            self
            ) -> Dict[str, observable.Observable]:
        task_observables = dict()

        task_observables["task/time"] = ConfinedObservable(
                low=0,
                high=self.config.simulation_time,
                shape=[1],
                raw_observation_callable=lambda
                    physics: physics.time()
                )
        task_observables["task/delta_time"] = ConfinedObservable(
                low=0,
                high=1,
                shape=[1],
                raw_observation_callable=lambda
                    _: self.config.control_timestep
                )

        task_observables["task/distance_travelled_along_x_axis"] = ConfinedObservable(
                low=-self.config.arena_size[0] - self.config.starting_position[0],
                high=self.config.arena_size[0] - self.config.starting_position[0],
                shape=[1],
                raw_observation_callable=self._get_distance_travelled_along_x_axis
                )

        for obs in task_observables.values():
            obs.enabled = True

        return task_observables

    def _configure_observables(
            self
            ) -> Dict[str, observable.Observable]:
        self._configure_morphology_observables()
        task_observables = self._configure_task_observables()
        return task_observables

    def get_reward(
            self,
            physics: mjcf.Physics
            ) -> float:
        current_distance_travelled = self._get_distance_travelled_along_x_axis(physics=physics)
        reward = current_distance_travelled - self._previous_distance_travelled
        self._previous_distance_travelled = current_distance_travelled
        return reward

    def _initialize_morphology_pose(
            self,
            physics: mjcf.Physics
            ) -> None:
        disc_height = self._morphology.morphology_specification.disc_specification.height.value
        initial_position = np.array(self.config.starting_position + (2 * disc_height,))

        if self.config.random_initial_rotation:
            random_rotation = np.random.uniform(0, 360)
            initial_quaternion = euler2quat(0, 0, random_rotation)
        else:
            initial_quaternion = None

        self._morphology.set_pose(
                physics=physics, position=initial_position, quaternion=initial_quaternion
                )

    def initialize_episode(
            self,
            physics: mjcf.Physics,
            random_state: np.random.RandomState
            ) -> None:
        self._arena.randomize(physics=physics, random_state=random_state)
        self._initialize_morphology_pose(physics)
        self._previous_distance_travelled = self._get_distance_travelled_along_x_axis(physics=physics)


class DirectedLocomotionTaskConfiguration(
        MJCEnvironmentConfig
        ):
    def __init__(
            self,
            env_id: int = 0,
            arena_size: Tuple[int, int] = (10, 5),
            random_initial_rotation: bool = True,
            hilly_terrain: bool = True,
            random_current: bool = True,
            random_friction: bool = True,
            starting_position: Tuple[int, int] = (-8.0, 0.0),
            touch_coloring: bool = False,
            time_scale: float = 0.5,
            control_substeps: int = 1,
            simulation_time: int = 20
            ) -> None:
        super().__init__(
                task=DirectedLocomotionTask,
                time_scale=time_scale,
                control_substeps=control_substeps,
                simulation_time=simulation_time,
                camera_ids=[0, 1]
                )
        self.env_id = env_id
        self.arena_size = arena_size
        self.random_initial_rotation = random_initial_rotation
        self.hilly_terrain = hilly_terrain
        self.random_current = random_current
        self.random_friction = random_friction
        self.starting_position = starting_position
        self.touch_coloring = touch_coloring


if __name__ == '__main__':
    env_config = DirectedLocomotionTaskConfiguration(
            touch_coloring=False,
            hilly_terrain=False,
            random_current=False,
            random_friction=False
            )

    print(f"Steps per episode: {env_config.total_num_timesteps}")

    morphology_specification = default_arm_length_based_brittle_star_morphology_specification(
            num_arms=5, arm_length_in_disc_diameters=4, use_p_control=True
            )
    morphology = MJCBrittleStarMorphology(
            specification=morphology_specification
            )

    dm_env = env_config.environment(
            morphology=morphology, wrap2gym=False
            )

    observation_spec = dm_env.observation_spec()
    action_spec = dm_env.action_spec()

    num_actions = dm_env.action_spec().shape[0]


    def policy_fn(
            timestep
            ) -> np.ndarray:
        return 0.1 * np.random.uniform(
                low=action_spec.minimum, high=action_spec.maximum, size=action_spec.shape
                )


    viewer.launch(dm_env, policy=policy_fn)
