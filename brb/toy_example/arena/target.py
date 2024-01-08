from mujoco_utils.mjcf_utils import MJCFSubComponent


class Target(MJCFSubComponent):
    def _build(
            self
            ):
        self._target = self.mjcf_body.add(
                'site', name="target_site", type='sphere', size=[0.2], rgba=[1.0, 0.0, 0.0, 1.0]
                )
