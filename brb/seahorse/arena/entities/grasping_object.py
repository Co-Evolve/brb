import numpy as np
from dm_control import composer, mjcf

from brb.utils import colors


class GraspingCylinder(composer.Entity):
    def _build(
            self
            ) -> None:
        self._mjcf_model = mjcf.RootElement()
        self._geom = self._mjcf_model.worldbody.add(
                "geom", type="cylinder", size=[0.05, 0.1], rgba=colors.rgba_gray, euler=[np.pi / 2, 0.0, 0.0]
                )

    @property
    def mjcf_model(
            self
            ) -> mjcf.RootElement:
        return self._mjcf_model
