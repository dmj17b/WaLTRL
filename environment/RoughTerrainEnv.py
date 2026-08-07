import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # Add parent directory to path
from typing import Any, Dict, Optional, Tuple, Union  # Import type hints for function signatures.
import warnings  # Import warnings module (not used in this snippet).

import jax  # Import JAX for numerical computing and random number generation.
import jax.numpy as jp  # Import JAX's numpy as jp for array operations.
import numpy as np
from ml_collections import config_dict  # Import config_dict for configuration management.
import mujoco  # Import mujoco for physics simulation.
from mujoco import mjx 

from mujoco_playground._src import mjx_env  # Import custom environment base class.
from mujoco_playground._src import reward  # Import reward utilities (not used in this snippet).
from mujoco_playground._src.dm_control_suite import common  # Import common utilities for dm_control_suite.

from modeling import GenModel
import yaml
from pathlib import Path

from configs.base_config import SimConfig, RewardConfig, CommandConfig, NoiseConfig  # Import configuration dataclasses.

from environment.BaseEnv import BaseEnv  # Import the BaseEnv class from the environment module.

class RoughTerrainEnv(BaseEnv):
    """Base Environment class for WaLTER DRL Training."""

    def __init__(
            self,
            difficulty = 0.5,  # Difficulty level for terrain generation (0.0 to 1.0).
            smoothing = 0.5,
            sim_config = SimConfig(),
            reward_config = RewardConfig(),
            command_config = CommandConfig(),
    ):
        self.difficulty = difficulty
        self.smoothing = smoothing
        super().__init__(sim_config, reward_config, command_config)  # Initialize the BaseEnv with the provided configurations.
        

    def _reset_model_qpos(self, pos_rng: jax.Array) -> jax.Array:
        '''Randomize the initial position of the model within a specified range.'''
        qpos = jp.zeros(self.mj_model.nq)  # Start with default qpos
        qpos = qpos.at[3].set(1.0) # Initial rotation quaternion (w component)
        qpos = qpos.at[2].set(self.difficulty+0.2) # Initial height of the torso above the ground
        [xpos, ypos] = jax.random.uniform(pos_rng, shape=(2,), minval=-10, maxval=10)  # Randomize x and y positions within [-10, 10]
        qpos = qpos.at[0].set(xpos)  # Set randomized x position
        qpos = qpos.at[1].set(ypos)  # Set randomized y position
        return qpos


    def _reset_model_qvel(self, vel_rng: jax.Array) -> jax.Array:
        '''Randomize the initial velocity of the model within a specified range.'''
        qvel = jp.zeros(self.mj_model.nv)  # Start with default qvel
        return qvel

    
    def _add_terrain(self):
        '''Defines the terrain for the environment and then applies necessary contact pairs.
        Make sure to update this function and contact pairs for every new environment.'''
        self.model_spec.add_hfield(height = self.difficulty, sigma = self.smoothing)
        self.model_spec.add_contact_pairs()  # Add contact pairs for wheel-ground interactions


    def _reset_terrain(self, terrain_rng: jax.Array) -> jax.Array:
        ''' Placeholder for randomizing terrain features.'''
        return None  # No terrain randomization implemented yet
    
    def _get_mocap_ids(self):
        """Get the mocap body ID for the terrain"""
        hfield_body_id = self.mj_model.body("terrain_body").id  # Get the body ID for the heightfield terrain.
        hfield_mocap_id = self.mj_model.body_mocapid[hfield_body_id]  # Get the mocap ID for the heightfield body to control its position during terrain randomization.
        return jp.array([hfield_mocap_id])  # Return the mocap ID as a JAX array for use in the environment.

    @property
    def xml_path(self) -> str:
        return "modeling/WaLTER.xml"
    

    @property
    def action_size(self) -> int:
        return self.mjx_model.nu
    

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model
    
    @property
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model
    
def main():
    env = BaseEnv()
    state = env.reset(rng = jax.random.PRNGKey(0))