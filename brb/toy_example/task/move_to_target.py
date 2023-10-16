from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from dm_control import composer
from dm_control import mjcf
from dm_control.composer import Entity
from dm_control.composer.observation import observable
from dm_control.mujoco.math import euler2quat
from mujoco_utils.environment import MJCEnvironmentConfig
from mujoco_utils.observables import ConfinedObservable

from brb.brittle_star.arena.hilly_light_aquarium import HillyLightAquarium
from brb.brittle_star.morphology.morphology import MJCBrittleStarMorphology
from brb.toy_example.arena.plane_with_target import PlaneWithTargetArena
from brb.toy_example.morphology.morphology import MJCToyExampleMorphology


class MoveToTargetTask(composer.Task):
    def __init__(
            self,
            config: MoveToTargetTaskConfiguration,
            morphology: MJCBrittleStarMorphology
            ) -> None:
        self.config = config

        self._arena = self._build_arena()
        self._morphology = self._attach_morphology(morphology=morphology)
        self._task_observables = self._configure_observables()

        self._previous_distance_to_target = None

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
            ) -> PlaneWithTargetArena:
        arena = PlaneWithTargetArena(
                size=self.config.arena_size, )
        return arena

    def _attach_morphology(
            self,
            morphology: MJCToyExampleMorphology
            ) -> MJCToyExampleMorphology:
        self._arena.add_free_entity(
                entity=morphology
                )
        return morphology

    def _configure_morphology_observables(
            self
            ) -> None:
        self._morphology.observables.enable_all()

    @staticmethod
    def _get_entity_xy_position(
            entity: Entity,
            physics: mjcf.Physics
            ) -> np.ndarray:
        position, _ = entity.get_pose(
                physics=physics
                )
        return np.array(
                position
                )[:2]

    def _get_xy_distance_to_target(
            self,
            physics: mjcf.Physics
            ) -> float:
        morphology_position = self._get_entity_xy_position(entity=self._morphology, physics=physics)
        target_position = self._get_entity_xy_position(
                entity=self._arena.target, physics=physics
                )
        distance = np.linalg.norm(morphology_position - target_position)
        return distance

    def _get_xy_direction_to_target(
            self,
            physics: mjcf.Physics
            ) -> float:
        morphology_position = self._get_entity_xy_position(entity=self._morphology, physics=physics)
        target_position = self._get_entity_xy_position(
                entity=self._arena.target, physics=physics
                )

        direction = target_position - morphology_position
        distance = self._get_xy_distance_to_target(physics=physics)
        unit_direction = direction / distance
        return unit_direction

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
        task_observables["task/xy-distance-to-target"] = ConfinedObservable(
                low=0, high=np.inf, shape=[1], raw_observation_callable=self._get_xy_distance_to_target
                )
        task_observables["task/xy-direction-to-target"] = ConfinedObservable(
                low=-1, high=1, shape=[2], raw_observation_callable=self._get_xy_direction_to_target
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
        current_distance_to_target = self._get_xy_distance_to_target(physics=physics)
        reward = self._previous_distance_to_target - current_distance_to_target
        self._previous_distance_to_target = current_distance_to_target
        return reward

    def _initialize_morphology_pose(
            self,
            physics: mjcf.Physics
            ) -> None:

        torso_radius = self._morphology.morphology_specification.torso_specification.radius.value
        initial_position = np.array([0.0, 0.0, 1.5 * torso_radius])

        initial_quaternion = euler2quat(0, 0, 0)

        self._morphology.set_pose(
                physics=physics, position=initial_position, quaternion=initial_quaternion
                )

    def initialize_episode(
            self,
            physics: mjcf.Physics,
            random_state: np.random.RandomState
            ) -> None:
        self._arena.randomize_target_location(physics=physics,
                                              distance_from_origin=self.config.target_distance_from_origin)
        self._initialize_morphology_pose(physics)
        self._previous_distance_to_target = self._get_xy_distance_to_target(physics=physics)


class MoveToTargetTaskConfiguration(
        MJCEnvironmentConfig
        ):
    def __init__(
            self,
            env_id: int = 0,
            arena_size: Tuple[int, int] = (10, 10),
            time_scale: float = 0.5,
            control_substeps: int = 1,
            simulation_time: int = 20,
            target_distance_from_origin: float = 5) -> None:
        super().__init__(
                task=MoveToTargetTask,
                time_scale=time_scale,
                control_substeps=control_substeps,
                simulation_time=simulation_time,
                camera_ids=[0, 1],
                )
        self.env_id = env_id
        self.arena_size = arena_size
        self.target_distance_from_origin = target_distance_from_origin
