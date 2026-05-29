import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # Add parent directory to path
from typing import Any, Dict, Optional, Union  # Import type hints for function signatures.
import warnings  # Import warnings module (not used in this snippet).

import jax  # Import JAX for numerical computing and random number generation.
import jax.numpy as jp  # Import JAX's numpy as jp for array operations.
from ml_collections import config_dict  # Import config_dict for configuration management.
import mujoco  # Import mujoco for physics simulation.
from mujoco import mjx 

from mujoco_playground._src import mjx_env  # Import custom environment base class.
from mujoco_playground._src import reward  # Import reward utilities (not used in this snippet).
from mujoco_playground._src.dm_control_suite import common  # Import common utilities for dm_control_suite.

from modeling import GenModel

from configs.base_config import SimConfig, RewardConfig, CommandConfig  # Import configuration dataclasses.

class BaseEnv(mjx_env.MjxEnv):
    """Base Environment class for WaLTER DRL Training."""

    def __init__(
            self,
            sim_config = SimConfig(),
            reward_config = RewardConfig(),
            command_config = CommandConfig(),
    ):
        super().__init__(config=sim_config)
        
        # Load configurations
        self.config = sim_config  # Store the configuration
        self.reward_config = reward_config  # Store the reward configuration
        self.command_config = command_config  # Store the command configuration

        # Generate WaLTER model spec, then add appropriate terrain/lighting elements
        model_config = 'modeling/model_configs/2_7_Scale/model_config.yaml'
        motor_config = 'modeling/model_configs/2_7_Scale/motor_config.yaml'
        self.model_spec = GenModel.GenModel(model_config, motor_config)
        self.model_spec.add_scene()
        self._add_terrain()
        self._mj_model = self.model_spec.spec.compile()
        self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.impl)

        # Select the appropriate model based on the implementation specified in the configuration
        if(self._config.impl == 'warp'):
            self.model = self._mj_model
        else:
            self.model = self._mjx_model

        # Define addresses for joints and actuators for easy access later
        self._define_addresses()
    
    def reset(self, rng: jax.Array) -> mjx_env.State:
        '''Reset the environment to an initial state.'''

        # Split the input RNG into multiple RNGs for different randomization purposes
        rng, terrain_rng, pos_rng, vel_rng, command_rng, command_duration_rng = jax.random.split(rng, 6)

        # Randomize initial position and velocity of the model
        qpos = self._reset_model_qpos(pos_rng)
        qvel = self._reset_model_qvel(vel_rng)
        mocap_pos = self._reset_terrain(terrain_rng)

        # Create new data object:
        data = mjx_env.make_data(
            self.model,
            qpos = qpos,
            qvel = qvel,
            mocap_pos = mocap_pos,
            impl = self._config.impl,
            naconmax = self._config.naconmax,
        )

        # Randomize initial command and time until command change
        command = self.sample_command(command_rng)
        steps_until_cmd_change = jax.random.randint(command_duration_rng,
                                                    shape = (), 
                                                    minval=int(self.command_config.min_cmd_duration / self.config.ctrl_dt), 
                                                    maxval=int(self.command_config.max_cmd_duration / self.config.ctrl_dt))
        info = {
            "rng": rng,
            "command": command,
            "prev_action": jp.zeros(self.action_size),
            "steps_since_cmd_change": 0,
            "steps_until_cmd_change": steps_until_cmd_change,
        }

        # Initialize metrics:
        metrics = {
            "reward/lin_vel_tracking": 0.0,
            "reward/ang_vel_tracking": 0.0,
            "penalty/action_smoothness": 0.0,
            "penalty/torques": 0.0,
            "penalty/pitchover_failure": 0.0,
            "penalty/pitch_angle_deviation": 0.0,
            "penalty/pitch_angle_velocity": 0.0,
            "penalty/rollover_failure": 0.0,
            "penalty/roll_angle_deviation": 0.0,
            "penalty/roll_angle_velocity": 0.0,
            "penalty/body_z_velocity": 0.0,
            "train/episode_reward": 0.0,
        }

        done = jp.zeros(())  # Episode termination flag, initialized to False (0.0)
        reward = jp.zeros(())  # Reward, initialized to 0.0


        # Get initial observation:
        obs = self._get_policy_obs(data, info)

        return mjx_env.State(data, obs, reward, done, metrics, info)

    def _reset_model_qpos(self, pos_rng: jax.Array) -> jax.Array:
        '''Randomize the initial position of the model within a specified range.'''
        qpos = jp.zeros(self.mj_model.nq)  # Start with default qpos
        return qpos


    def _reset_model_qvel(self, vel_rng: jax.Array) -> jax.Array:
        '''Randomize the initial velocity of the model within a specified range.'''
        qvel = jp.zeros(self.mj_model.nv)  # Start with default qvel
        return qvel



    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        '''Apply the given action to the environment and step the simulation forward.'''
        pass


    def calculate_motor_targets(self, state: mjx_env.State, action: jax.Array, rng: jax.Array) -> jax.Array:
        '''Calculate the target positions for the motors based on the current state, action, and command.'''
        return jp.zeros_like(action)  # Placeholder implementation

    def _get_policy_obs(self, data: mjx.Data, info: Dict[str, Any]) -> jax.Array:
        '''Extract policy network observation from the simulation state'''
        vel_commands = info["command"]
        prev_action = info["prev_action"]
        hip_joint_positions = data.qpos[self.hip_qposadrs]
        knee_joint_positions = data.qpos[self.knee_qposadrs]
        knee_pos_sins = jp.sin(knee_joint_positions)
        knee_pos_coss = jp.cos(knee_joint_positions)
        joint_velocities = data.qvel[self.qvel_idx]
        joint_torques = data.qfrc_actuator[self.ctrl_idx]

        obs = jp.concatenate([
            vel_commands,
            prev_action,
            hip_joint_positions,
            knee_pos_sins,
            knee_pos_coss,
            joint_velocities,
            joint_torques
            ])
        return obs

    def _get_value_obs(self, data: mjx.Data, info: Dict[str, Any]) -> jax.Array:
        '''Extract value network observation from the simulation state'''
        pass

    def sample_command(self, rng: jax.Array) -> jax.Array:

        # Split RNG keys
        rng, lin_vel_rng, ang_vel_rng, zero_rng = jax.random.split(rng, num=4)
        # Determine whether to sample a zero command based on the specified probability
        is_zero_command = jax.random.uniform(zero_rng) < self.command_config.zero_lin_prob

        # Sample linear and angular velocity commands uniformly within the specified ranges
        lin_vel_command = jax.random.uniform(lin_vel_rng, minval=-self.command_config.max_lin_vel, maxval=self.command_config.max_lin_vel)
        ang_vel_command = jax.random.uniform(ang_vel_rng, minval=-self.command_config.max_ang_vel, maxval=self.command_config.max_ang_vel)

        lin_vel_command = jp.where(is_zero_command, 0.0, lin_vel_command)
        ang_vel_command = jp.where(is_zero_command, 0.0, ang_vel_command)
        command = jp.array([lin_vel_command, ang_vel_command])
        return command
    
    def _maybe_update_cmd(self, info: dict[str, Any]) -> dict[str, Any]:
        """Checks if it's time to update the command and samples a new one if necessary."""
        new_info = dict(info)
        new_info["steps_since_cmd_change"] = info["steps_since_cmd_change"] + 1  # Increment the steps since last command change
        command_key, time_key, rng = jax.random.split(info["rng"], num=3)  # Split the RNG for command sampling and time sampling

        # Check if it's time to sample a new command:
        new_cmd = jp.where(
            new_info["steps_since_cmd_change"] >= new_info["steps_until_cmd_change"],
            self.sample_command(command_key),  # Sample a new command if the counter has reached the threshold
            info["command"]  # Otherwise, keep the current command
        )
        steps_since_cmd_change = jp.where(
            new_info["steps_since_cmd_change"] >= new_info["steps_until_cmd_change"],
            0,  # Reset the counter if a new command is sampled
            new_info["steps_since_cmd_change"]  # Otherwise, keep the current counter
        )
        steps_until_cmd_change = jp.where(
            new_info["steps_since_cmd_change"] >= new_info["steps_until_cmd_change"],
            jax.random.randint(time_key, (), self.min_steps_per_command, self.max_steps_per_command + 1),  # Sample a new duration for the next command if the counter has reached the threshold
            new_info["steps_until_cmd_change"]  # Otherwise, keep the current duration
        )
        new_info["steps_until_cmd_change"] = steps_until_cmd_change  # Update the steps until command change in the info dictionary
        new_info["command"] = new_cmd  # Update the command in the info dictionary
        new_info["steps_since_cmd_change"] = steps_since_cmd_change  # Update the counter in the info dictionary
        new_info["rng"] = rng  # Update the RNG in the info dictionary for the next step

        return new_info
    

    def _define_addresses(self):
        '''Define convenient references to joint IDs and qpos addresses for easy access later.'''
        self._joint_names = [
            "head_left_thigh_joint", "head_left_thigh_shin_joint", "head_left_shin_front_wheel_joint", "head_left_shin_back_wheel_joint",
            "head_right_thigh_joint", "head_right_thigh_shin_joint", "head_right_shin_front_wheel_joint", "head_right_shin_back_wheel_joint",
            "torso_left_thigh_joint", "torso_left_thigh_shin_joint", "torso_left_shin_front_wheel_joint", "torso_left_shin_back_wheel_joint",
            "torso_right_thigh_joint", "torso_right_thigh_shin_joint", "torso_right_shin_front_wheel_joint", "torso_right_shin_back_wheel_joint"
        ]
        self._knee_joint_names = [
            "head_left_thigh_shin_joint", "head_right_thigh_shin_joint",
            "torso_left_thigh_shin_joint", "torso_right_thigh_shin_joint"
        ]
        self._hip_joint_names = [
            "head_left_thigh_joint", "head_right_thigh_joint",
            "torso_left_thigh_joint", "torso_right_thigh_joint"
        ]
        self._wheel_joint_names = [
            "head_left_shin_front_wheel_joint", "head_left_shin_back_wheel_joint",
            "head_right_shin_front_wheel_joint", "head_right_shin_back_wheel_joint",
            "torso_left_shin_front_wheel_joint", "torso_left_shin_back_wheel_joint",
            "torso_right_shin_front_wheel_joint", "torso_right_shin_back_wheel_joint"
        ]
        self._actuator_names = [
            "fl_hip", "fl_knee", "fl_wheel1", "fl_wheel2",
            "fr_hip", "fr_knee", "fr_wheel1", "fr_wheel2",
            "bl_hip", "bl_knee", "bl_wheel1", "bl_wheel2",
            "br_hip", "br_knee", "br_wheel1", "br_wheel2"
        ]

        joint_ids = [mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in self._joint_names]
        knee_joint_ids = [mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in self._knee_joint_names]
        hip_joint_ids = [mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in self._hip_joint_names]
        wheel_joint_ids = [mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in self._wheel_joint_names]

        self.joint_qposadr = jp.array([self.mj_model.jnt_qposadr[jid] for jid in joint_ids])
        self.knee_qposadrs = jp.array([self.mj_model.jnt_qposadr[jid] for jid in knee_joint_ids])
        self.hip_qposadrs = jp.array([self.mj_model.jnt_qposadr[jid] for jid in hip_joint_ids])
        self.wheel_qposadrs = jp.array([self.mj_model.jnt_qposadr[jid] for jid in wheel_joint_ids])

        # Note: qvel uses dofadr, not qposadr. For 1-DOF joints, dofadr is the qvel index.
        self.qvel_idx = jp.array([self.mj_model.jnt_dofadr[jid] for jid in joint_ids]) 
        self.knee_qveladrs = jp.array([self.mj_model.jnt_dofadr[jid] for jid in knee_joint_ids])
        self.hip_qveladrs = jp.array([self.mj_model.jnt_dofadr[jid] for jid in hip_joint_ids])
        self.wheel_qveladrs = jp.array([self.mj_model.jnt_dofadr[jid] for jid in wheel_joint_ids])

        self.ctrl_idx = jp.array([mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in self._actuator_names])


        # Front left leg actuator addresses:
        self.fl_hip_act_id = self.mj_model.actuator("fl_hip").id
        self.fl_knee_act_id = self.mj_model.actuator("fl_knee").id
        self.fl_wheel1_act_id = self.mj_model.actuator("fl_wheel1").id
        self.fl_wheel2_act_id = self.mj_model.actuator("fl_wheel2").id

        # Front right leg actuator addresses:
        self.fr_hip_act_id = self.mj_model.actuator("fr_hip").id
        self.fr_knee_act_id = self.mj_model.actuator("fr_knee").id
        self.fr_wheel1_act_id = self.mj_model.actuator("fr_wheel1").id
        self.fr_wheel2_act_id = self.mj_model.actuator("fr_wheel2").id

        # Back left leg actuator addresses:
        self.bl_hip_act_id = self.mj_model.actuator("bl_hip").id
        self.bl_knee_act_id = self.mj_model.actuator("bl_knee").id
        self.bl_wheel1_act_id = self.mj_model.actuator("bl_wheel1").id
        self.bl_wheel2_act_id = self.mj_model.actuator("bl_wheel2").id

        # Back right leg actuator addresses:
        self.br_hip_act_id = self.mj_model.actuator("br_hip").id
        self.br_knee_act_id = self.mj_model.actuator("br_knee").id
        self.br_wheel1_act_id = self.mj_model.actuator("br_wheel1").id
        self.br_wheel2_act_id = self.mj_model.actuator("br_wheel2").id 
        

    def _add_terrain(self):
        self.model_spec.add_groundplane()

    def _reset_terrain(self, terrain_rng: jax.Array) -> jax.Array:
        ''' Placeholder for randomizing terrain features.'''
        return None  # No terrain randomization implemented yet

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