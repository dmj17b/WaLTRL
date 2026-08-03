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
            sim_config = SimConfig(),
            reward_config = RewardConfig(),
            command_config = CommandConfig(),
    ):
        self.difficulty = difficulty

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

    # REWARD FUNCTION
    def _get_reward(self, 
                    data: mjx.Data, 
                    action: jax.Array,
                    info: Dict[str,Any],
                    metrics: Dict[str, Any]) -> jax.Array:
        '''Basic reward function. This will likely be overwritten for most other environments, but common reward components can be implemented here.'''
        # Tracking rewards:
        # Calculate linear velocity tracking reward based on the current command and the body linear velocity from the sensor data
        lin_vel_tracking_reward = self.reward_config.lin_vel_tracking * self.tracking_reward(info["command"][0], 
                                                                                             data.sensordata[self.body_lin_vel_adrs],
                                                                                             sigma = self.reward_config.tracking_sigma)  
        ang_vel_tracking_reward = self.reward_config.ang_vel_tracking * self.tracking_reward(info["command"][1], 
                                                                                             data.qvel[self.torso_qveladr+5],
                                                                                             sigma = self.reward_config.tracking_sigma)  # Calculate angular velocity tracking reward based on the current command and the body angular velocity from the sensor data

        # Orientation Penalty:
        torso_rot_mat = data.xmat[self.torso_body_id]  # Extract the torso rotation matrix from the simulation data
        up_vector = torso_rot_mat[:, 2]  # Extract the up vector from the torso rotation matrix
        orientation_penalty = self.reward_config.orientation * jp.sum(jp.square(up_vector - jp.array([0.0, 0.0, 1.0])))  # Calculate the squared distance of the up vector from the ideal up vector (0, 0, 1) for orientation penalty

        # Roll and Pitch Velocity Penalties:
        roll_vel = data.qvel[self.torso_qveladr+3]  # Extract the roll velocity from the body angular velocity
        pitch_vel = data.qvel[self.torso_qveladr+4]  # Extract the pitch velocity from the body angular velocity
        roll_penalty = self.reward_config.body_roll_vel * jp.square(roll_vel)  # Calculate the squared roll velocity for penalty
        pitch_penalty = self.reward_config.body_pitch_vel * jp.square(pitch_vel)  # Calculate the squared pitch velocity for penalty

        # Z Velocity Penalty:
        body_z_vel = data.sensordata[self.body_lin_vel_adrs+2]  # Extract the body Z velocity from the sensor data
        body_z_vel_penalty = self.reward_config.body_z_vel * jp.square(body_z_vel)  # Calculate the squared body Z velocity for penalty

        # Torque Penalty:
        torque_penalty = self.reward_config.low_torques * jp.sum(jp.square(data.qfrc_actuator[self.act_ids]))

        # Rollover and Pitchover Penalties:
        flipped = jp.where(up_vector[2] < 0.1, 1.0, 0.0)  # Check if the up vector's Z component is less than 0 (indicating a fallover)
        flip_penalty = flipped*self.reward_config.flipped  # Apply a penalty for falling over


        # Zero Velocity Penalties:
        tol = 0.05  # Tolerance for considering a command as "zero"
        zero_vel_penalty = self.reward_config.zero_joint_vel * jp.sum(jp.square(data.qvel[self.qvel_adrs]))  # Calculate the squared joint velocities for penalty
        zero_cmd = jp.logical_and(jp.abs(info["command"][0]) < tol, jp.abs(info["command"][1]) < tol)
        zero_vel_penalty = jp.where(zero_cmd, zero_vel_penalty, 0.0)  # Only apply the penalty if the linear and angular velocity commands are zero
        
        # Action smoothing penalty:
        action_smoothing_penalty = self.reward_config.action_smoothing * jp.sum(jp.square(action - info["prev_action"]))  # Calculate the squared difference between the current action and the previous action for smoothing penalty
        
        # If commanded velocity is zero, scale action smoothing penalty to encourage less jitter
        # action_smoothing_penalty = jp.where(zero_cmd,
        #                                     action_smoothing_penalty * self.reward_config.zero_vel_smoothing_multiplier,
        #                                     action_smoothing_penalty)  

        # Wheel Collision Penalty:
        wheel_touch_penalty = self.wheel_touch_penalty(data)  # Calculate the penalty for wheel collisions based on the current simulation data

        # T-Pose Deviation Penalty:
        t_pose_deviation_penalty = self.t_pose_deviation_penalty(data)  

        # Calculate the total episode reward by summing rewards and subtracting penalties
        episode_reward = lin_vel_tracking_reward + ang_vel_tracking_reward - orientation_penalty - torque_penalty \
              - flip_penalty - body_z_vel_penalty - roll_penalty - pitch_penalty - action_smoothing_penalty - zero_vel_penalty \
             - wheel_touch_penalty  - t_pose_deviation_penalty


        metrics["reward/lin_vel_tracking"] = lin_vel_tracking_reward  # Log the linear velocity tracking reward in the metrics dictionary
        metrics["reward/ang_vel_tracking"] = ang_vel_tracking_reward  # Log the angular velocity tracking reward in the metrics dictionary
        metrics["penalty/orientation"] = orientation_penalty  # Log the orientation penalty in the metrics dictionary
        metrics["penalty/roll_velocity"] = roll_penalty  # Log the roll velocity penalty in the metrics dictionary
        metrics["penalty/pitch_velocity"] = pitch_penalty  # Log the pitch velocity penalty in the metrics dictionary
        metrics["penalty/z_velocity"] = body_z_vel_penalty  # Log the Z velocity penalty in the metrics dictionary
        metrics["penalty/torques"] = torque_penalty  # Log the torque penalty in the metrics dictionary
        metrics["penalty/flipped"] = flip_penalty  # Log the fallover penalty in the metrics dictionary
        metrics["penalty/zero_vel"] = zero_vel_penalty  # Log the zero velocity penalty in the metrics dictionary
        metrics["penalty/action_smoothing"] = action_smoothing_penalty  # Log the action smoothing penalty in the metrics dictionary
        metrics["penalty/wheel_collisions"] = wheel_touch_penalty  # Log the wheel collision penalty in the metrics dictionary
        metrics["penalty/t_pose_deviation"] = t_pose_deviation_penalty  # Log the T-pose deviation penalty in the metrics dictionary
        metrics["train/episode_reward"] = episode_reward  # Accumulate the episode reward in the metrics dictionary

        return episode_reward


    def _add_terrain(self):
        '''Defines the terrain for the environment and then applies necessary contact pairs.
        Make sure to update this function and contact pairs for every new environment.'''
        self.model_spec.add_hfield(height = self.difficulty, sigma = 0.3)
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