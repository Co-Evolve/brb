from collections import defaultdict
from typing import Dict, List, Union

import numpy as np
from dm_control import mjcf
from dm_control.mjcf.element import _ElementImpl
from mujoco_utils.robot import MJCMorphology, MJCMorphologyPart

from brb.brittle_star.morphology.specification.specification import BrittleStarJointSpecification, \
    BrittleStarMorphologySpecification
from brb.utils import colors


class MJCBrittleStarArmSegment(MJCMorphologyPart):
    def __init__(
            self,
            parent: Union[MJCMorphology, MJCMorphologyPart],
            name: str,
            pos: np.array,
            euler: np.array,
            *args,
            **kwargs
            ) -> None:
        super().__init__(parent, name, pos, euler, *args, **kwargs)

    @property
    def morphology_specification(
            self
            ) -> BrittleStarMorphologySpecification:
        return super().morphology_specification

    def _build(
            self,
            arm_index: int,
            segment_index: int
            ) -> None:
        self._arm_index = arm_index
        self._segment_index = segment_index

        self._arm_specification = self.morphology_specification.arm_specifications[self._arm_index]
        self._segment_specification = self._arm_specification.segment_specifications[self._segment_index]

        self._tendon_attachment_points = defaultdict(list)
        self._tendon_plates = []

        self._build_capsule()
        self._build_connector()
        self._configure_joints()
        self._build_tendon_plates()
        self._configure_actuators()
        self._configure_sensors()

    def _build_capsule(
            self
            ) -> None:
        radius = self._segment_specification.radius.value
        length = self._segment_specification.length.value

        self._capsule = self.mjcf_body.add(
                "geom",
                name=f"{self.base_name}_capsule",
                type="capsule",
                pos=self.center_of_capsule,
                euler=[0, np.pi / 2, 0],
                size=[radius, length / 2],
                rgba=colors.rgba_green,
                contype=0,
                conaffinity=0
                )

    def _build_connector(
            self
            ) -> None:
        radius = self._segment_specification.radius.value
        self._connector = self.mjcf_body.add(
                "geom",
                name=f"{self.base_name}_connector",
                type="sphere",
                pos=np.zeros(3),
                size=[0.5 * radius],
                rgba=colors.rgba_gray,
                contype=0,
                conaffinity=0
                )

    @property
    def center_of_capsule(
            self
            ) -> np.ndarray:
        radius = self._segment_specification.radius.value
        length = self._segment_specification.length.value
        x_offset = radius + length / 2
        return np.array([x_offset, 0, 0])

    def _configure_joint(
            self,
            name: str,
            axis: np.ndarray,
            joint_specification: BrittleStarJointSpecification
            ) -> _ElementImpl:
        joint = self.mjcf_body.add(
                "joint",
                name=name,
                type="hinge",
                limited=True,
                range=[-joint_specification.range.value, joint_specification.range.value],
                axis=axis,
                stiffness=joint_specification.stiffness.value,
                damping=joint_specification.damping.value,
                armature=joint_specification.armature.value,
                frictionloss=joint_specification.frictionloss.value
                )
        return joint

    def _configure_joints(
            self
            ) -> None:
        self._in_plane_joint = self._configure_joint(
                name=f"{self.base_name}_in_plane_joint",
                axis=[0, 0, 1],
                joint_specification=self._segment_specification.in_plane_joint_specification
                )
        self._out_of_plane_joint = self._configure_joint(
                name=f"{self.base_name}_out_of_plane_joint",
                axis=[0, 1, 0],
                joint_specification=self._segment_specification.out_of_plane_joint_specification
                )

    @property
    def tendon_plate_radius(
            self
            ) -> float:
        capsule_radius = self._segment_specification.radius.value
        tendon_offset = self._segment_specification.tendon_offset.value

        tendon_plate_radius = capsule_radius * tendon_offset
        return tendon_plate_radius

    def _build_tendon_plate(
            self,
            side: str,
            position: np.ndarray
            ) -> None:
        tendon_plate = self.mjcf_body.add(
                "geom",
                name=f"{self.base_name}_tendon_plate_{side}",
                type="cylinder",
                size=[self.tendon_plate_radius, self._tendon_plate_thickness],
                pos=position,
                euler=[0, np.pi / 2, 0],
                rgba=colors.rgba_green
                )
        self._tendon_plates.append(tendon_plate)

        self._configure_tendon_attachment_points(side=side, x_offset=position[0])

    def _build_tendon_plates(
            self
            ) -> None:
        self._tendon_plate_thickness = 0.1 * self._segment_specification.length.value
        self._build_tendon_plate(
                side="proximal", position=0.5 * self.center_of_capsule
                )
        self._build_tendon_plate(
                side="distal", position=1.5 * self.center_of_capsule
                )

    @property
    def tendon_attachment_points(
            self
            ) -> Dict[str, List[List[mjcf.Element]]]:
        return self._tendon_attachment_points

    def _configure_tendon_attachment_points(
            self,
            side: str,
            x_offset: float
            ) -> None:
        # 4 equally spaced tendon attachment points
        # order: Ventral-sinistral, Ventral-dextral, Dorsal-dextral, Dorsal-sinistral
        angles = [-np.pi / 4, - 3 * np.pi / 4, 3 * np.pi / 4, np.pi / 4]

        tendon_plate_radius = self.tendon_plate_radius
        for tendon_index, angle in enumerate(angles):
            attachment_points = []
            for sub_side, x_offset_factor in zip(["proximal", "center", "distal"], [-1, 0, 1]):
                position = 0.9 * tendon_plate_radius * np.array([0, np.cos(angle), np.sin(angle)])
                position[0] = x_offset + x_offset_factor * self._tendon_plate_thickness

                name = f"{self.base_name}_muscle_attachment_point_{side}_{tendon_index}_{sub_side}"

                attachment_point = self.mjcf_body.add(
                        "site", name=name, pos=position, size=[0.0001]
                        )
                attachment_points.append(attachment_point)

            self.tendon_attachment_points[side].append(attachment_points)

    def _is_first_segment(
            self
            ) -> bool:
        return self._segment_index == 0

    def _is_last_segment(
            self
            ) -> bool:
        number_of_segments = len(self._arm_specification.segment_specifications)
        return self._segment_index == number_of_segments - 1

    @property
    def volume(
            self
            ) -> float:
        radius = self._segment_specification.radius.value
        length = self._segment_specification.length.value

        sphere_volume = 4 / 3 * np.pi * radius ** 3
        cylinder_volume = np.pi * radius ** 2 * length

        return sphere_volume + cylinder_volume

    def _get_p_control_kp(
            self,
            joint: _ElementImpl
            ) -> float:
        kp = self.volume * 500_000
        return kp

    def _configure_p_control_actuator(
            self,
            joint: _ElementImpl
            ) -> None:
        self.mjcf_model.actuator.add(
                'position',
                name=f"{joint.name}_p_control",
                joint=joint,
                kp=self._get_p_control_kp(joint),
                ctrllimited=True,
                ctrlrange=joint.range
                )

    def _configure_p_control_actuators(
            self
            ) -> None:
        if self.morphology_specification.actuation_specification.use_p_control.value:
            self._configure_p_control_actuator(self._in_plane_joint)
            self._configure_p_control_actuator(self._out_of_plane_joint)

    def _get_torque_control_gear(
            self,
            joint: _ElementImpl
            ) -> float:
        gear = self.volume * 200_000
        return gear

    def _configure_torque_control_actuator(
            self,
            joint: _ElementImpl
            ) -> None:
        self.mjcf_model.actuator.add(
                'motor',
                name=f"{joint.name}_torque_control",
                joint=joint,
                gear=[self._get_torque_control_gear(joint)],
                ctrllimited=True,
                ctrlrange=[-1, 1]
                )

    def _configure_torque_control_actuators(
            self
            ) -> None:
        if self.morphology_specification.actuation_specification.use_torque_control.value:
            self._configure_torque_control_actuator(self._in_plane_joint)
            self._configure_torque_control_actuator(self._out_of_plane_joint)

    def _configure_actuators(
            self
            ) -> None:
        self._configure_p_control_actuators()
        self._configure_torque_control_actuators()

    def _configure_touch_sensors(
            self
            ) -> None:
        num_touch_sensors = self._segment_specification.num_touch_sensors.value
        angles = np.linspace(0, 2 * np.pi, num_touch_sensors + 1)[:-1]
        circumference = 2 * np.pi * self.tendon_plate_radius
        sensor_height = 0.9 * circumference / num_touch_sensors / 2

        x_distance_between_tendon_plates = abs(self._tendon_plates[0].pos[0] - self._tendon_plates[1].pos[1])
        sensor_depth = x_distance_between_tendon_plates + self._tendon_plate_thickness
        for sensor_index, angle in enumerate(angles):
            tendon_plate_touch_site = self.mjcf_body.add(
                    "site",
                    name=f"{self.base_name}_touch_site_{angle}",
                    type="box",
                    pos=self.center_of_capsule + self.tendon_plate_radius * np.array([0, np.cos(angle), np.sin(angle)]),
                    euler=[angle, 0, 0],
                    rgba=colors.rgba_red * [1, 1, 1, 0.1],
                    size=[sensor_depth, 0.001, sensor_height]
                    )
            self.mjcf_model.sensor.add(
                    "touch", name=f"{self.base_name}_touch_sensor_{sensor_index}", site=tendon_plate_touch_site
                    )

    def _configure_position_sensor(
            self
            ) -> None:
        self.mjcf_model.sensor.add(
                "framepos", name=f"{self.base_name}_position_sensor", objtype="geom", objname=self._capsule.name
                )

    def _configure_joint_sensors(
            self
            ) -> None:
        self.mjcf_model.sensor.add(
                "jointpos", joint=self._in_plane_joint, name=f"{self._in_plane_joint.name}"
                                                             f"_jointpos_sensor"
                )
        self.mjcf_model.sensor.add(
                "jointvel", joint=self._in_plane_joint, name=f"{self._in_plane_joint.name}"
                                                             f"_jointvel_sensor"
                )
        self.mjcf_model.sensor.add(
                "jointpos", joint=self._out_of_plane_joint, name=f"{self._out_of_plane_joint.name}"
                                                                 f"_jointpos_sensor"
                )
        self.mjcf_model.sensor.add(
                "jointvel", joint=self._out_of_plane_joint, name=f"{self._out_of_plane_joint.name}"
                                                                 f"_jointvel_sensor"
                )

    def _configure_sensors(
            self
            ):
        self._configure_touch_sensors()
        self._configure_position_sensor()
        self._configure_joint_sensors()
