from typing import Callable

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from mujoco import mjx

from brb.toy_example.mjx.env import MjxEnv


def device_model_get_into(
        result: mujoco.MjModel,
        value: mjx.Model
        ) -> None:
    """
    Loads
    :param result:
    :param value:
    :return:
    """
    value = jax.device_get(value)
    for key, v in vars(value).items():
        try:
            previous_value = getattr(result, key)
            if isinstance(previous_value, np.ndarray):
                previous_value[:] = v
            else:
                setattr(result, key, v)
        except AttributeError:
            pass
        except ValueError:
            pass


def run_passive_viewer(
        env: MjxEnv,
        policy_fn: Callable[[jnp.ndarray], jnp.ndarray],
        seed: int = 42
        ) -> None:
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)

    rng = jax.random.PRNGKey(seed=seed)
    rng, reset_rng = jax.random.split(key=rng, num=2)
    state = jit_reset(reset_rng)

    device_model_get_into(result=env.mj_model, value=state.model)
    mjx.device_get_into(result=env.data, value=state.data)

    with mujoco.viewer.launch_passive(model=env.mj_model, data=env.data) as v:
        while True:
            if state.done:
                rng, reset_rng = jax.random.split(key=rng, num=2)
                state = jit_reset(reset_rng)
                device_model_get_into(result=env.mj_model, value=state.model)
                mjx.device_get_into(result=env.data, value=state.data)
                mujoco.mj_forward(env.mj_model, env.data)
                v.sync()

            actions = policy_fn(state.obs)
            state = jit_step(
                    state=state, action=actions
                    )
            mjx.device_get_into(result=env.data, value=state.data)
            v.sync()
