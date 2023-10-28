from typing import Tuple

import jax
import mujoco
from brax.envs import Env
from jax import numpy as jnp
from mujoco import mjx


class MjxEnv(Env):
    """API for driving an MJX system for training and inference in brax."""

    def __init__(
            self,
            mj_model: mujoco.MjModel,
            physics_steps_per_control_step: int = 1
            ) -> None:
        """Initializes MjxEnv.

        Args:
          mj_model: mujoco.MjModel
          physics_steps_per_control_step: the number of times to step the physics
            pipeline for each environment step
        """
        self.mj_model = mj_model
        self.data = mujoco.MjData(mj_model)
        self.mjx_model = mjx.device_put(mj_model)
        self._physics_steps_per_control_step = physics_steps_per_control_step

    def pipeline_init(
            self,
            model: mjx.Model,
            qpos: jax.Array,
            qvel: jax.Array
            ) -> mjx.Data:
        """Initializes the physics state."""
        data = mjx.device_put(self.data)
        data = data.replace(qpos=qpos, qvel=qvel, ctrl=jnp.zeros(model.nu))
        data = mjx.forward(model, data)
        return data

    def pipeline_step(
            self,
            model: mjx.Model,
            data: mjx.Data,
            ctrl: jax.Array
            ) -> mjx.Data:
        """Takes a physics step using the physics pipeline."""

        def f(
                data,
                _
                ) -> Tuple[mjx.Data, None]:
            data = data.replace(ctrl=ctrl)
            return (mjx.step(model, data), None,)

        data, _ = jax.lax.scan(f, data, (), self._physics_steps_per_control_step)
        return data

    @property
    def dt(
            self
            ) -> jax.Array:
        """The timestep used for each env step."""
        return self.mjx_model.opt.timestep * self._physics_steps_per_control_step

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
        return self.mjx_model.nu

    @property
    def backend(
            self
            ) -> str:
        return 'mjx'
