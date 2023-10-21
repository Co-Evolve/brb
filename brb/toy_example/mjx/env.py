import time
from typing import Tuple

import jax
from jax import numpy as jnp
import mujoco
import numpy as np
from brax import Motion, Transform
from brax.envs import Env
from mujoco import mjx


class MjxEnv(Env):
    """API for driving an MJX system for training and inference in brax."""

    def __init__(
            self,
            mj_model: mujoco.MjModel,
            physics_steps_per_control_step: int = 1, ):
        """Initializes MjxEnv.

        Args:
          mj_model: mujoco.MjModel
          physics_steps_per_control_step: the number of times to step the physics
            pipeline for each environment step
        """
        self.model = mj_model
        self.data = mujoco.MjData(mj_model)
        self.sys = mjx.device_put(mj_model)
        self._physics_steps_per_control_step = physics_steps_per_control_step

    def pipeline_init(
            self,
            sys: mjx.Model,
            qpos: jax.Array,
            qvel: jax.Array
            ) -> mjx.Data:
        """Initializes the physics state."""
        data = mjx.device_put(self.data)
        data = data.replace(qpos=qpos, qvel=qvel, ctrl=jnp.zeros(sys.nu))
        data = mjx.forward(sys, data)
        return sys, data

    def pipeline_step(
            self,
            sys: mjx.Model,
            data: mjx.Data,
            ctrl: jax.Array
            ) -> mjx.Data:
        """Takes a physics step using the physics pipeline."""

        def f(
                data,
                _
                ) -> Tuple[mjx.Data, None]:
            data = data.replace(ctrl=ctrl)
            return (mjx.step(sys, data), None,)

        data, _ = jax.lax.scan(f, data, (), self._physics_steps_per_control_step)
        return data

    @property
    def dt(
            self
            ) -> jax.Array:
        """The timestep used for each env step."""
        return self.sys.opt.timestep * self._physics_steps_per_control_step

    @property
    def observation_size(
            self
            ) -> int:
        rng = jax.random.PRNGKey(0)
        reset_state = self.unwrapped.reset(rng)
        return reset_state.obs.shape[-1]

    @property
    def action_size(
            self
            ) -> int:
        return self.sys.nu

    @property
    def backend(
            self
            ) -> str:
        return 'mjx'

    def _pos_vel(
            self,
            data: mjx.Data
            ) -> Tuple[Transform, Motion]:
        """Returns 6d spatial transform and 6d velocity for all bodies."""
        x = Transform(pos=data.xpos[1:, :], rot=data.xquat[1:, :])
        cvel = Motion(vel=data.cvel[1:, 3:], ang=data.cvel[1:, :3])
        offset = data.xipos[1:, :] - data.subtree_com[self.model.body_rootid[np.arange(1, self.model.nbody)]]
        xd = Transform.create(pos=offset).vmap().do(cvel)
        return x, xd
