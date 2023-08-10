from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np
from dm_control import composer
from dm_control import mjcf
from dm_control import viewer
from dm_control.composer.observation import observable
from dm_control.mujoco.math import euler2quat
from scipy.interpolate import RegularGridInterpolator

from brb.brittle_star.environment.arenas.hilly_light_aquarium import HillyLightAquarium
from brb.brittle_star.morphology.morphology import MJCBrittleStarMorphology
from brb.brittle_star.morphology.specification.default import default_arm_length_based_brittle_star_morphology_specification
from erpy.framework.specification import RobotSpecification
from erpy.interfaces.mujoco.environment import MJCEnvironmentConfig
from erpy.interfaces.mujoco.observables import ConfinedObservable
from erpy.interfaces.mujoco.phenome import MJCMorphology


class LightEvasionTask(composer.Task):
    def __init__(
            self,
            config: LightEvasionTaskConfiguration,
            morphology: MJCBrittleStarMorphology,
            ) -> None:
        self.config = config

        self._arena = self._build_arena()
        self._light_extractor = self._create_light_extractor()
        self._morphology = self._attach_morphology(morphology=morphology)
        self._task_observables = self._configure_observables()

        self._configure_contacts()
        self._configure_memory()

        self._segment_position_sensors = None
        self._previous_normalised_light_income = None

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
                light_noise=self.config.light_noise,
                hilly_terrain=self.config.hilly_terrain,
                random_current=self.config.random_current,
                random_friction=self.config.random_friction
                )
        return arena

    def _create_light_extractor(
            self
            ) -> RegularGridInterpolator:
        x = np.arange(0, self._arena.lightmap.shape[1])
        y = np.arange(0, self._arena.lightmap.shape[0])
        normalized_lightmap = (self._arena.lightmap - np.min(self._arena.lightmap)) / (
                np.max(self._arena.lightmap) - np.min(self._arena.lightmap))
        interp = RegularGridInterpolator((y, x), normalized_lightmap, bounds_error=False, fill_value=None)
        return interp

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
        morphology.after_attachment()
        morphology.touch_coloring = self.config.touch_coloring
        return morphology

    def _configure_morphology_observables(
            self
            ) -> None:
        self._morphology.observables.enable_all()

    @property
    def segment_position_sensors(
            self
            ) -> List[mjcf.Element]:
        if not self._segment_position_sensors:
            self._segment_position_sensors = list(
                    filter(
                            lambda
                                sensor: sensor.tag == "framepos" and "segment" in sensor.name,
                            self._morphology.mjcf_model.find_all(
                                    "sensor"
                                    )
                            )
                    )
        return self._segment_position_sensors

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

    def _xy_world_positions_to_light_values(
            self,
            world_positions: np.ndarray
            ) -> np.ndarray:
        world_positions = np.array(world_positions)

        # Get normalised positions
        norm_world_pos = (world_positions + self._arena.size) / (2 * self._arena.size)
        # Positive Y axis in light map and in world are inverted
        norm_world_pos[:, 1] = 1 - norm_world_pos[:, 1]

        # Transform into indices of lightmap
        continuous_light_map_indices = self._arena.lightmap.shape * norm_world_pos[:, ::-1]

        light_values = self._light_extractor(continuous_light_map_indices)

        return light_values

    def _get_light_per_segment(
            self,
            physics: mjcf.Physics
            ) -> np.ndarray:
        # Get positions of all segments
        segment_positions = np.array(physics.bind(self.segment_position_sensors).sensordata)
        segment_xy_positions = segment_positions.reshape((-1, 3))[:, :2]

        light_per_segment = self._xy_world_positions_to_light_values(world_positions=segment_xy_positions)

        return light_per_segment

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

        task_observables["task/light_per_segment"] = ConfinedObservable(
                low=0,
                high=1,
                shape=[self._morphology.morphology_specification.total_number_of_segments],
                raw_observation_callable=self._get_light_per_segment
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

    def _get_normalised_light_income(
            self,
            physics: mjcf.Physics
            ) -> float:
        light_per_segment = self._get_light_per_segment(physics=physics)
        light_income = np.sum(np.power(light_per_segment, 2))
        normalised_light_income = light_income / self._morphology.morphology_specification.total_number_of_segments
        return normalised_light_income

    def get_reward(
            self,
            physics: mjcf.Physics
            ) -> float:
        current_normalised_light_income = self._get_normalised_light_income(physics=physics)
        reward = self._previous_normalised_light_income - current_normalised_light_income
        self._previous_normalised_light_income = current_normalised_light_income
        return reward

    def _initialize_morphology_pose(
            self,
            physics: mjcf.Physics
            ) -> None:
        disc_height = self._morphology.morphology_specification.disc_specification.height.value
        initial_position = np.array(self.config.starting_position + (2 * disc_height,))

        random_rotation = np.random.uniform(0, 360)
        initial_quaternion = euler2quat(0, 0, random_rotation)

        self._morphology.set_pose(
                physics=physics, position=initial_position, quaternion=initial_quaternion
                )

    def initialize_episode(
            self,
            physics: mjcf.Physics,
            random_state: np.random.RandomState
            ) -> None:
        self._arena.randomize(physics=physics, random_state=random_state)
        self._light_extractor = self._create_light_extractor()
        self._initialize_morphology_pose(physics)
        self._previous_normalised_light_income = self._get_normalised_light_income(physics=physics)

    def before_step(
            self,
            physics,
            action,
            random_state
            ):
        super().before_step(
                physics=physics, action=action, random_state=random_state
                )

    def after_step(
            self,
            physics: mjcf.Physics,
            random_state: np.random.RandomState
            ) -> None:
        if self.config.light_coloring:
            self._morphology.color_light(physics=physics, light_value_per_segment=self._get_light_per_segment(physics))
        super().after_step(physics=physics, random_state=random_state)


@dataclass
class LightEvasionTaskConfiguration(
        MJCEnvironmentConfig
        ):
    env_id: int = 0
    arena_size: Tuple[int, int] = (10, 5)
    light_noise: bool = True
    hilly_terrain: bool = True
    random_current: bool = True
    random_friction: bool = True
    starting_position: Tuple[int, int] = (-8.0, 0.0)
    touch_coloring: bool = False
    light_coloring: bool = False

    @property
    def task(
            self
            ) -> Callable[[MJCEnvironmentConfig, MJCMorphology], composer.Task]:
        return LightEvasionTask

    @property
    def simulation_time(
            self
            ) -> float:
        return 20.0

    @property
    def num_substeps(
            self
            ) -> int:
        return 1

    @property
    def time_scale(
            self
            ) -> float:
        return 0.5

    @property
    def camera_ids(
            self
            ) -> List[int]:
        return [0, 1]


if __name__ == '__main__':
    env_config = LightEvasionTaskConfiguration(
            touch_coloring=False,
            light_coloring=True,
            hilly_terrain=True,
            light_noise=True,
            random_current=True,
            random_friction=True
            )

    print(f"Steps per episode: {env_config.num_timesteps}")
    print(f"True steps per episode: {env_config.num_timesteps / 5}")
    morphology_specification = default_arm_length_based_brittle_star_morphology_specification(
            num_arms=5, arm_length_in_disc_diameters=4, use_p_control=True
            )
    robot_specification = RobotSpecification(
            morphology_specification=morphology_specification, controller_specification=None
            )
    morphology = MJCBrittleStarMorphology(
            specification=robot_specification
            )

    dm_env = env_config.environment(
            morphology=morphology, wrap2gym=False
            )

    observation_spec = env_config.observation_specification
    action_spec = env_config.action_specification

    num_actions = action_spec.shape[0]


    def policy_fn(
            timestep
            ) -> np.ndarray:
        return 0.1 * np.random.uniform(
                low=action_spec.minimum, high=action_spec.maximum, size=action_spec.shape
                )


    viewer.launch(dm_env, policy=policy_fn)
