from dm_control import composer, mjcf

from brb.utils import colors


class Target(composer.Entity):
    def _build(
            self
            ):
        self._mjcf_model = mjcf.RootElement("target")
        self._target = self._mjcf_model.worldbody.add(
                'site', name="target_site", type='sphere', size=[0.2], rgba=colors.rgba_red)


    @property
    def mjcf_model(
            self
            ):
        return self._mjcf_model
