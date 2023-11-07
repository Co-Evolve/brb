from typing import Any, Dict

import jax
from brax.base import Base
from flax import struct
from mujoco import mjx


@struct.dataclass
class State(Base):
    """Environment state for training and inference with brax.

    Args:
      model: the current Model, mjx.Model
      data: the physics state, mjx.Data
      obs: environment observations
      reward: environment reward
      done: boolean, True if the current episode has terminated
      metrics: metrics that get tracked per environment step
      info: environment variables defined and updated by the environment reset
        and step functions
    """
    model: mjx.Model
    data: mjx.Data
    obs: jax.Array
    reward: jax.Array
    done: jax.Array
    metrics: Dict[str, jax.Array] = struct.field(default_factory=dict)
    info: Dict[str, Any] = struct.field(default_factory=dict)

    @property
    def pipeline_state(self) -> mjx.Data:
        return self.data