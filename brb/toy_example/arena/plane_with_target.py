import numpy as np
from dm_control import mjcf
from dm_control.locomotion.arenas import Floor

import brb
from brb.toy_example.arena.entities.target import Target


class PlaneWithTargetArena(Floor):
    def _build(
            self,
            size=(8, 8),
            reflectance=.2,
            aesthetic='default',
            name='floor',
            top_camera_y_padding_factor=1.1,
            top_camera_distance=100
            ) -> None:
        super()._build(name=name)

        self.target = self._attach_target()

    def _attach_target(
            self
            ) -> Target:
        target = Target()
        attachment_frame = self.attach(target)

        # MJX compatability
        attachment_frame.add(
                "joint",
                name="target_joint_x",
                type="slide",
                axis=[1, 0, 0])
        attachment_frame.add(
                "joint", name="target_joint_y", type="slide", axis=[0, 1, 0]
                )

        return target

    def randomize_target_location(
            self,
            physics: mjcf.Physics,
            distance_from_origin: float
            ) -> None:
        angle = brb.brb_random_state.uniform(low=0, high=2 * np.pi)
        position = distance_from_origin * np.array([np.cos(angle), np.sin(angle), 0.0])
        self.target.set_pose(
            physics=physics, position=position
            )
