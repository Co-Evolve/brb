import numpy as np
from dm_control.composer import Arena


class GraspingArena(Arena):
    needs_collision = []

    def _build(
            self,
            name='grasping_arena',
            env_id: int = 0,
            size=(1, 1, 0.1),
            aquatic: bool = False
            ) -> None:
        super()._build(name=name)

        self._dynamic_assets_identifier = env_id
        self.size = np.array(size)
        self._aquatic = aquatic

        self._configure_cameras()
        self._configure_lights()
        self._configure_sky()
        self._build_ground()
        self._configure_water()

    def _configure_cameras(
            self
            ):
        self._mjcf_root.worldbody.add(
                'camera', name='top_camera', pos=[0, 0, 20], quat=[1, 0, 0, 0], )
        # Always initialize the free camera so that it points at the origin.
        self.mjcf_model.statistic.center = (0., 0., 0.)

    def _configure_lights(
            self
            ):
        self.mjcf_model.worldbody.add(
                'light',
                pos=(0, 0, 5),
                dir=(0, 0, -1),
                diffuse=(0.7, 0.7, 0.7),
                specular=(.3, .3, .3),
                directional='false',
                castshadow='true'
                )

    def _configure_sky(
            self
            ) -> None:
        # white sky
        self._mjcf_root.asset.add(
                'texture', type='skybox', builtin='flat', rgb1='1.0 1.0 1.0', rgb2='1.0 1.0 1.0', width=200, height=200
                )

    def _build_ground(
            self
            ) -> None:
        ground_texture = self._mjcf_root.asset.add(
                'texture',
                rgb1=[.2, .3, .4],
                rgb2=[.1, .2, .3],
                type='2d',
                builtin='checker',
                name='groundplane',
                width=200,
                height=200,
                mark='edge',
                markrgb=[0.8, 0.8, 0.8]
                )
        ground_material = self._mjcf_root.asset.add(
                'material', name='groundplane', texrepeat=[2, 2],  # Makes white squares exactly 1x1 length units.
                texuniform=True, reflectance=0.0, texture=ground_texture
                )

        # Build groundplane.
        self._ground_geom = self._mjcf_root.worldbody.add(
                'geom', type='plane', name='ground', material=ground_material, size=self.size
                )
        self.needs_collision.append(self._ground_geom)

    def _configure_water(
            self
            ) -> None:
        if self._aquatic:
            self.mjcf_model.option.density = 1000
            self.mjcf_model.option.viscosity = 0.0009
