from __future__ import annotations

import copy
from typing import Dict, Tuple

import numpy as np
from dm_control import composer
from dm_control import mjcf
from dm_control import viewer
from dm_control.composer.observation import observable
from dm_env import TimeStep
from mujoco_utils.environment import MJCEnvironmentConfig
from mujoco_utils.observables import ConfinedObservable
from transforms3d.euler import euler2quat

from brb.seahorse.arena.entities.grasping_object import GraspingCylinder
from brb.seahorse.arena.grasping_arena import GraspingArena
from brb.seahorse.morphology.morphology import MJCSeahorseMorphology
from brb.seahorse.morphology.specification.default import default_seahorse_morphology_specification


class GraspingTask(composer.Task):
    def __init__(
            self,
            config: GraspingTaskConfiguration,
            morphology: MJCSeahorseMorphology
            ) -> None:
        self.config = config

        self._arena = self._build_arena()
        self._morphology = self._attach_morphology(morphology=morphology)
        self._grasping_object = self._attach_grasping_object()
        self._task_observables = self._configure_observables()

        self._configure_contacts()
        self._configure_memory()

    @property
    def root_entity(
            self
            ) -> GraspingArena:
        return self._arena

    @property
    def task_observables(
            self
            ) -> Dict:
        return self._task_observables

    def _build_arena(
            self
            ) -> GraspingArena:
        arena = GraspingArena(env_id=self.config.env_id)
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
            morphology: MJCSeahorseMorphology
            ) -> MJCSeahorseMorphology:
        self._arena.attach(entity=morphology)
        return morphology

    def _attach_grasping_object(
            self
            ) -> GraspingCylinder:
        if self.config.with_object:
            grasping_cylinder = GraspingCylinder()
            self._arena.attach(entity=grasping_cylinder)
            # todo: add x and z plane joints
            return grasping_cylinder

    def _configure_morphology_observables(
            self
            ) -> None:
        self._morphology.observables.enable_all()

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
        return 0.0

    def _set_morphology_pose(
            self,
            physics: mjcf.Physics
            ) -> None:
        # self._morphology.set_pose(
        #         physics=physics, position=np.array([0.0, 0.0, 0.5]), quaternion=euler2quat(
        #             *[np.pi, np.pi / 2, 0.0], axes="szyx"
        #             )
        #         )
        self._morphology.set_pose(
                physics=physics, position=np.array([0.0, 0.0, 0.5]), quaternion=euler2quat(
                        *[np.pi, 0.0, 0.0], )
                )

    def _set_object_pose(
            self,
            physics: mjcf.Physics
            ) -> None:
        if self.config.with_object:
            self._grasping_object.set_pose(
                    physics=physics, position=np.array([0.3, 0.0, 0.6]), quaternion=euler2quat(*[0.0, 0.0, 0.0])
                    )

    def initialize_episode(
            self,
            physics: mjcf.Physics,
            random_state: np.random.RandomState
            ) -> None:
        self._set_morphology_pose(physics=physics)
        self._set_object_pose(physics=physics)


class GraspingTaskConfiguration(
        MJCEnvironmentConfig
        ):
    def __init__(
            self,
            env_id: int = 0,
            arena_size: Tuple[int, int] = (10, 5, 0),
            time_scale: float = 1.0,
            control_substeps: int = 1,
            simulation_time: int = 5,
            with_object: bool = True
            ) -> None:
        super().__init__(
                task=GraspingTask,
                time_scale=time_scale,
                control_substeps=control_substeps,
                simulation_time=simulation_time,
                camera_ids=[0, 1]
                )
        self.env_id = env_id
        self.arena_size = arena_size
        self.with_object = with_object


if __name__ == '__main__':
    env_config = GraspingTaskConfiguration(with_object=False)

    morphology_specification = default_seahorse_morphology_specification(num_segments=30)
    morphology = MJCSeahorseMorphology(specification=morphology_specification)
    dm_env = env_config.environment(
            morphology=morphology, wrap2gym=False
            )
    action_spec = dm_env.action_spec()
    num_tendons_per_side = int(action_spec.shape[0] / 2)
    num_tendons_per_corner = int(num_tendons_per_side / 2)


    def alternating_policy_fn(
            timestep: TimeStep
            ) -> np.ndarray:
        global action_spec
        time = timestep.observation["task/time"][0][0]

        # ventral_actions = [np.sin(time)] * num_tendons_per_side
        # dorsal_actions = [np.cos(time)] * num_tendons_per_side
        # actions = np.array(ventral_actions + dorsal_actions)
        # actions = (actions + 1) / 2
        #
        # actions = actions * (action_spec.maximum - action_spec.minimum) + action_spec.minimum
        ventral_bending = np.concatenate(
                (action_spec.minimum[:num_tendons_per_side], action_spec.maximum[-num_tendons_per_side:])
                )
        dorsal_bending = np.concatenate(
                (action_spec.maximum[:num_tendons_per_side], action_spec.minimum[-num_tendons_per_side:])
                )
        if time < 4.75:
            print("ventral")
            # Ventral bending
            return ventral_bending
        elif time < 5.25:
            return ventral_bending + ((time - 2) / 0.5) * (dorsal_bending - ventral_bending)
        else:
            print("dorsal")
            # dorsal bending
            return dorsal_bending


    def grasping_policy_fn(
            timestep: TimeStep
            ) -> np.ndarray:
        global action_spec
        SECONDS_TO_FULL_CONTRACTION = 4
        time = timestep.observation["task/time"][0][0]

        rel_time = np.clip(time / SECONDS_TO_FULL_CONTRACTION, 0, 1)

        # ventral-dorsal, dextral-sinistral
        actions = copy.deepcopy(action_spec.maximum.reshape(4, num_tendons_per_corner))

        num_tendons_per_corner_to_contract = int(num_tendons_per_corner * rel_time)
        actions[:2, :num_tendons_per_corner_to_contract] = action_spec.minimum.reshape(4, num_tendons_per_corner)[:2,
                                                           :num_tendons_per_corner_to_contract]

        return actions.flatten()


    viewer.launch(dm_env, grasping_policy_fn)
