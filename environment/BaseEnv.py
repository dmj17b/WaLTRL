import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # Add parent directory to path
from typing import Any, Dict, Optional, Union  # Import type hints for function signatures.
import warnings  # Import warnings module (not used in this snippet).

import jax  # Import JAX for numerical computing and random number generation.
import jax.numpy as jp  # Import JAX's numpy as jp for array operations.
from ml_collections import config_dict  # Import config_dict for configuration management.
import mujoco  # Import mujoco for physics simulation.
from mujoco import mjx  # Import mjx, a JAX-based Mujoco wrapper.

from mujoco_playground._src import mjx_env  # Import custom environment base class.
from mujoco_playground._src import reward  # Import reward utilities (not used in this snippet).
from mujoco_playground._src.dm_control_suite import common  # Import common utilities for dm_control_suite.


class BaseEnv(mjx_env.MjxEnv):
    """Base Environment class for WaLTER DRL Training."""

    def __init__(
            self,
            sim_config,
            reward_config,
            command_config,
    ):
        
        super().__init__(config=sim_config)

    
    def reset(self, rng: jax.Array) -> mjx_env.State:
        pass


    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        pass

    def _get_obs(self, state: mjx_env.State) -> jax.Array:
        pass

    def _define_joint_addresses(self):
        pass

    def _define_actuator_addresses(self):
        pass


    @property
    def xml_path(self) -> str:
        return "modeling/WaLTER.xml"
    

    @property
    def action_size(self) -> int:
        return self.mjx_model.nu
    
    @property
    def mj_model(self) -> mujoco.MjModel:
        return self.mj_model
    
    @property
    def mjx_model(self) -> mjx.Model:
        return self.mjx_model
    
def main():
    env = BaseEnv()
    state = env.reset(rng = jax.random.PRNGKey(0))