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

from modeling import GenModel


class BaseEnv(mjx_env.MjxEnv):
    """Base Environment class for WaLTER DRL Training."""

    def __init__(
            self,
            sim_config,
            reward_config,
            command_config,
    ):
        super().__init__(config=sim_config)

        # Generate WaLTER model spec, then add appropriate terrain/lighting elements
        self.model_spec = GenModel.GenModel()
        self.model_spec.add_scene()
        self._add_terrain()

        # Define addresses for joints and actuators for easy access later
        self._define_actuator_addresses()
    
    def reset(self, rng: jax.Array) -> mjx_env.State:
        '''Reset the environment to an initial state.'''

        # Randomize initial command and time until command change
        command = self.sample_command(command_rng)
        steps_until_cmd_change = self.sample_command_duration(command_duration_rng)
        info = {
            "rng": rng,
            "command": command,
            "prev_action": jp.zeros(self.action_size),
            "steps_since_cmd_change": 0,
            "steps_until_cmd_change": steps_until_cmd_change,
        }

        # Get initial observation:
        obs = self._get_policy_obs(data, info)

        return mjx_env.State(data, obs, reward, done, metrics, info)

    def _reset_model_pos(pos_rng: jax.Array) -> jax.Array:
        '''Randomize the initial position of the model within a specified range.'''
        pass

    def _reset_joint_pos(pos_rng: jax.Array) -> jax.Array:
        '''Randomize the initial joint positions of the model within a specified range.'''
        pass

    def _reset_model_vel(vel_rng: jax.Array) -> jax.Array:
        '''Randomize the initial velocity of the model within a specified range.'''
        pass

    def _reset_joint_vel(vel_rng: jax.Array) -> jax.Array:
        '''Randomize the initial joint velocities of the model within a specified range.'''
        pass

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        pass

    def _get_policy_obs(self, data: mjx.Data, info: Dict[str, Any]) -> jax.Array:
        '''Extract policy network observation from the simulation state'''
        pass

    def _get_value_obs(self, data: mjx.Data, info: Dict[str, Any]) -> jax.Array:
        '''Extract value network observation from the simulation state'''
        pass


    def _define_joint_addresses(self):
        pass

    def _define_actuator_addresses(self):
        # motor_names = ['fl_hip', 'fl_knee', 'fl_wheel1', 'fl_wheel2',
        #                'fr_hip', 'fr_knee', 'fr_wheel1', 'fr_wheel2',
        #                'bl_hip', 'bl_knee', 'bl_wheel1', 'bl_wheel2',
        #                'br_hip', 'br_knee', 'br_wheel1', 'br_wheel2']

        # Front left leg actuator addresses:
        self.fl_hip_act_id = self._mj_model.actuator("fl_hip").id
        self.fl_knee_act_id = self._mj_model.actuator("fl_knee").id
        self.fl_wheel1_act_id = self._mj_model.actuator("fl_wheel1").id
        self.fl_wheel2_act_id = self._mj_model.actuator("fl_wheel2").id

        # Front right leg actuator addresses:
        self.fr_hip_act_id = self._mj_model.actuator("fr_hip").id
        self.fr_knee_act_id = self._mj_model.actuator("fr_knee").id
        self.fr_wheel1_act_id = self._mj_model.actuator("fr_wheel1").id
        self.fr_wheel2_act_id = self._mj_model.actuator("fr_wheel2").id

        # Back left leg actuator addresses:
        self.bl_hip_act_id = self._mj_model.actuator("bl_hip").id
        self.bl_knee_act_id = self._mj_model.actuator("bl_knee").id
        self.bl_wheel1_act_id = self._mj_model.actuator("bl_wheel1").id
        self.bl_wheel2_act_id = self._mj_model.actuator("bl_wheel2").id

        # Back right leg actuator addresses:
        self.br_hip_act_id = self._mj_model.actuator("br_hip").id
        self.br_knee_act_id = self._mj_model.actuator("br_knee").id
        self.br_wheel1_act_id = self._mj_model.actuator("br_wheel1").id
        self.br_wheel2_act_id = self._mj_model.actuator("br_wheel2").id 

    def _add_terrain(self):
        self.model_spec.add_groundplane()


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