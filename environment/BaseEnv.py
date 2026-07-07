import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # Add parent directory to path
from typing import Any, Dict, Optional, Tuple, Union  # Import type hints for function signatures.
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
import yaml
from pathlib import Path

from configs.base_config import SimConfig, RewardConfig, CommandConfig, NoiseConfig  # Import configuration dataclasses.

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
        self.model_spec = GenModel.GenModel(model_config, motor_config, include_waist_joint=False)  # Generate the model specification
        self.model_spec.add_scene()
        self._add_terrain()
        self._mj_model = self.model_spec.spec.compile()
        self._mjx_model = mjx.put_model(m = self._mj_model, impl=self._config.impl)

        # Select the appropriate model based on the implementation specified in the configuration
        if(self._config.impl == 'warp'):
            self.model = self._mj_model
        else:
            self.model = self._mjx_model

        # Define addresses for joints and actuators for easy access later
        self._define_addresses()

        # Action scaling variables:
        self.hip_action_scale = 1.25  # Scaling factor for hip joint actions to limit the range of motion
        self.knee_action_scale = 2.5  # Scaling factor for knee joint actions to limit the range of motion
        self.wheel_action_scale = 33.0  # Scaling factor for wheel joint velocities to limit the maximum speed

        # Motor parameters:
        self.motor_params = yaml.safe_load(Path(motor_config).read_text())  # Load motor parameters from the motor configuration file
        self.hip_motor_params = self.motor_params["hip_params"]  # Extract hip motor parameters
        self.knee_motor_params = self.motor_params["knee_params"]  # Extract knee motor parameters
        self.wheel_motor_params = self.motor_params["wheel_params"]  # Extract

        # Noise config:
        self.noise_config = NoiseConfig()  # Initialize noise configuration with default values

        # Velocity command parameters:
        self.min_steps_per_command = int(self.command_config.min_cmd_duration / self.config.ctrl_dt)  # Minimum number of steps before changing command 
        self.max_steps_per_command = int(self.command_config.max_cmd_duration / self.config.ctrl_dt)  # Maximum number of steps before changing command
    
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
            njmax = self._config.njmax,
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
            "penalty/action_smoothing": 0.0,
            "penalty/torques": 0.0,
            "penalty/orientation": 0.0,
            "penalty/flipped": 0.0,
            "penalty/roll_velocity": 0.0,
            "penalty/pitch_velocity": 0.0,
            "penalty/z_velocity": 0.0,
            "train/episode_reward": 0.0,
        }

        done = jp.zeros(())  # Episode termination flag, initialized to False (0.0)
        reward = jp.zeros(())  # Reward, initialized to 0.0


        # Get initial observation:
        obs, info = self._get_policy_obs(data, info)

        return mjx_env.State(data, obs, reward, done, metrics, info)

    def _reset_model_qpos(self, pos_rng: jax.Array) -> jax.Array:
        '''Randomize the initial position of the model within a specified range.'''
        qpos = jp.zeros(self.mj_model.nq)  # Start with default qpos
        qpos = qpos.at[3].set(1.0) # Initial rotation quaternion (w component)
        qpos = qpos.at[2].set(0.3) # Initial height of the torso above the ground
        return qpos


    def _reset_model_qvel(self, vel_rng: jax.Array) -> jax.Array:
        '''Randomize the initial velocity of the model within a specified range.'''
        qvel = jp.zeros(self.mj_model.nv)  # Start with default qvel
        return qvel

    # SIMULATION Step - applies low level control loop to each simulation step
    def _simulation_step(self, state: mjx_env.State, action: jax.Array, n_substeps: int) -> mjx.Data:

        data = state.data
        hip_actions = action[self.hip_act_ids]
        knee_actions = action[self.knee_act_ids]
        wheel_actions = action[self.wheel_act_ids]

        def _substep(carry: mjx.Data, unused_t) -> Tuple[mjx.Data, None]:
            # Apply PD Control laws
            hip_torques = self.hip_controller(state, hip_actions, state.info["rng"])
            knee_torques = self.knee_controller(state, knee_actions, state.info["rng"])
            wheel_torques = self.wheel_controller(state, wheel_actions, state.info["rng"])
            
            # Combine torques into a single vector for all actuators:
            torques = jp.zeros(self.mj_model.nu)
            torques = torques.at[self.hip_act_ids].set(hip_torques)
            torques = torques.at[self.knee_act_ids].set(knee_torques)
            torques = torques.at[self.wheel_act_ids].set(wheel_torques)

            data = carry.replace(ctrl = torques)
            data = mjx.step(self._mjx_model, data)
            return data, None
        data, _ = jax.lax.scan(_substep, data, None, length=n_substeps)
        return data

    # MAIN STEP FUNCTION
    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        '''Apply the given action to the environment and step the simulation forward.'''
        info = self._maybe_update_cmd(dict(state.info))
        data = self._simulation_step(state, action, self.n_substeps)

        obs, info = self._get_policy_obs(data, info)


        reward = self._get_reward(data, action, info, state.metrics)
        done = jp.zeros(())  # Placeholder for episode termination condition
        metrics = state.metrics
        
        info["prev_action"] = action

        # Check for rollover/pitchover failure based on the body up vector:
        upvec = data.sensordata[self.body_upvec_adrs:self.body_upvec_adrs + self.body_upvec_dim]
        done = jp.where(upvec[2] < -0.1, 1.0, done)  # Terminate episode if the up vector's Z component is less than -0.1 (indicating a fallover)

        return mjx_env.State(data, obs, reward, done, metrics, info)
    
    # REWARD FUNCTION
    def _get_reward(self, 
                    data: mjx.Data, 
                    action: jax.Array,
                    info: Dict[str,Any],
                    metrics: Dict[str, Any]) -> jax.Array:
        '''Basic reward function. This will likely be overwritten for most other environments, but common reward components can be implemented here.'''
        # Tracking rewards:
        lin_vel_tracking_reward = self.reward_config.lin_vel_tracking * self.tracking_reward(info["command"][0], data.sensordata[self.body_lin_vel_adrs])  # Calculate linear velocity tracking reward based on the current command and the body linear velocity from the sensor data
        ang_vel_tracking_reward = self.reward_config.ang_vel_tracking * self.tracking_reward(info["command"][1], data.qvel[self.torso_qveladr+5])  # Calculate angular velocity tracking reward based on the current command and the body angular velocity from the sensor data

        # Orientation Penalty:
        up_vector = data.sensordata[self.body_upvec_adrs:self.body_upvec_adrs+self.body_upvec_dim]  # Extract the body up vector from the sensor data
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
        flipped = jp.where(up_vector[2] < 0.0, 1.0, 0.0)  # Check if the up vector's Z component is less than 0 (indicating a fallover)
        flip_penalty = flipped*self.reward_config.flipped  # Apply a penalty for falling over

        # Action smoothing penalty:
        action_smoothing_penalty = self.reward_config.action_smoothing * jp.sum(jp.square(action - info["prev_action"]))  # Calculate the squared difference between the current action and the previous action for smoothing penalty

        episode_reward = lin_vel_tracking_reward + ang_vel_tracking_reward - orientation_penalty - torque_penalty - flip_penalty - body_z_vel_penalty - roll_penalty - pitch_penalty - action_smoothing_penalty  # Calculate the total episode reward by summing rewards and subtracting penalties


        metrics["reward/lin_vel_tracking"] = lin_vel_tracking_reward  # Log the linear velocity tracking reward in the metrics dictionary
        metrics["reward/ang_vel_tracking"] = ang_vel_tracking_reward  # Log the angular velocity tracking reward in the metrics dictionary
        metrics["penalty/orientation"] = orientation_penalty  # Log the orientation penalty in the metrics dictionary
        metrics["penalty/roll_velocity"] = roll_penalty  # Log the roll velocity penalty in the metrics dictionary
        metrics["penalty/pitch_velocity"] = pitch_penalty  # Log the pitch velocity penalty in the metrics dictionary
        metrics["penalty/z_velocity"] = body_z_vel_penalty  # Log the Z velocity penalty in the metrics dictionary
        metrics["penalty/torques"] = torque_penalty  # Log the torque penalty in the metrics dictionary
        metrics["penalty/flipped"] = flip_penalty  # Log the fallover penalty in the metrics dictionary
        metrics["penalty/action_smoothing"] = action_smoothing_penalty  # Log the action smoothing penalty in the metrics dictionary
        metrics["train/episode_reward"] = episode_reward  # Accumulate the episode reward in the metrics dictionary

        return episode_reward

    def tracking_reward(self, desired, actual, sigma=0.25):
        error = desired - actual
        reward = jp.exp(-0.5 * (error / sigma) ** 2)
        return reward

    def hip_controller(self, state: mjx_env.State, hip_actions: jax.Array, rng: jax.Array) -> jax.Array:
        '''Calculate the control signals for the hip motors based on the current state, action, and command.'''
        hip_targets = self.hip_action_scale*hip_actions
        des_torques = self.hip_motor_params["Kp"] * (hip_targets - state.data.qpos[self.hip_qposadrs]) + self.hip_motor_params["Kd"] * (0 - state.data.qvel[self.hip_qveladrs])  # PD control law to calculate desired torques based on position error and velocity error
        actual_torques = self.motor_model(des_torques, state.data.qvel[self.hip_qveladrs], self.hip_motor_params)  # Apply motor model to limit torques based on speed-torque curves
        return des_torques

    def knee_controller(self, state: mjx_env.State, knee_actions: jax.Array, rng: jax.Array) -> jax.Array:
        '''Calculate the control signals for the knee motors based on the current state, action, and command.'''
        knee_targets = state.data.qpos[self.knee_qposadrs] + self.knee_action_scale*knee_actions  # Calculate target knee positions by adding scaled actions to current positions
        des_torques = self.knee_motor_params["Kp"] * (knee_targets - state.data.qpos[self.knee_qposadrs]) + self.knee_motor_params["Kd"] * (0 - state.data.qvel[self.knee_qveladrs])  # PD control law to calculate desired torques based on position error and velocity error
        actual_torques = self.motor_model(des_torques, state.data.qvel[self.knee_qveladrs], self.knee_motor_params)  # Apply motor model to limit torques based on speed-torque curves
        return des_torques

    def wheel_controller(self, state: mjx_env.State, wheel_actions: jax.Array, rng: jax.Array) -> jax.Array:
        '''Calculate the control signals for the wheel motors based on the current state, action, and command.'''
        wheel_targets = self.wheel_action_scale*wheel_actions  # Calculate target wheel velocities by scaling the actions
        des_torques = self.wheel_motor_params["Kd"] * (wheel_targets - state.data.qvel[self.wheel_qveladrs])  # Velocity control law to calculate desired torques based on velocity error
        actual_torques = self.motor_model(des_torques, state.data.qvel[self.wheel_qveladrs], self.wheel_motor_params)  # Apply motor model to limit torques based on speed-torque curves
        return des_torques
    
    def motor_model(self, des_torques: jax.Array, curr_vel: jax.Array, motor_params: dict) -> jax.Array:
        '''NOT IMPLEMENTED: Placeholder to restrict desired torques based on motor speed-torque curves.'''
        return des_torques  # Placeholder for motor model implementation that limits torques based on speed-torque curves



    def _get_policy_obs(self, data: mjx.Data, info: Dict[str, Any]) -> jax.Array:
        '''Extract policy network observation from the simulation state'''
        info = dict(info)
        
        # Forward and angular velocity commands
        vel_commands = info["command"]

        # Previous action (for action smoothing penalty)
        prev_action = info["prev_action"]

        # Body accelerometer and gyroscope readings:
        accel_rng, rng = jax.random.split(info["rng"])
        accel_noise = jax.random.normal(accel_rng, shape=(self.body_accel_dim,)) * self.noise_config.body_accel_std  # Generate noise for body accelerometer and gyroscope readings based on the specified standard deviation in the noise configuration
        body_accel = data.sensordata[self.body_accel_adrs:self.body_accel_adrs+self.body_accel_dim] + accel_noise  # Add noise to the body accelerometer and gyroscope readings for observation

        # Body gyroscope readings:
        gyro_rng, rng = jax.random.split(rng)
        gyro_noise = jax.random.normal(gyro_rng, shape=(self.body_gyro_dim,)) * self.noise_config.body_gyro_std  # Generate noise for body gyroscope readings based on the specified standard deviation in the noise configuration
        body_gyro = data.sensordata[self.body_gyro_adrs:self.body_gyro_adrs+self.body_gyro_dim] + gyro_noise  # Add noise to the body gyroscope readings for observation

        # Hip Joint Positions and Velocities:
        hip_pos_rng, rng = jax.random.split(rng)
        hip_pos_noise = jax.random.normal(hip_pos_rng, shape=(4,)) * self.noise_config.joint_pos_std  # Generate noise for hip joint positions based on the specified standard deviation in the noise configuration
        hip_joint_positions = data.qpos[self.hip_qposadrs] + hip_pos_noise  # Add noise to the hip joint positions for observation

        # Knee Joint Positions:
        knee_pos_rng, rng = jax.random.split(rng)
        knee_pos_noise = jax.random.normal(knee_pos_rng, shape=(4,)) * self.noise_config.joint_pos_std  # Generate noise for knee joint positions based on the specified standard deviation in the noise configuration
        knee_joint_positions = data.qpos[self.knee_qposadrs] + knee_pos_noise  # Add noise to the knee joint positions for observation
        knee_pos_sins = jp.sin(knee_joint_positions)
        knee_pos_coss = jp.cos(knee_joint_positions)
        
        # Joint Velocities"
        joint_vel_rng, rng = jax.random.split(rng)
        joint_vel_noise = jax.random.normal(joint_vel_rng, shape=(self.mj_model.nu,)) * self.noise_config.joint_vel_std  # Generate noise for joint velocities based on the specified standard deviation in the noise configuration
        joint_velocities = data.qvel[self.qvel_adrs] + joint_vel_noise  # Add noise to the joint velocities for observation

        # Torque noise:
        torque_rng, rng = jax.random.split(rng)
        torque_noise = jax.random.normal(torque_rng, shape=(self.mj_model.nu,)) * self.noise_config.torque_std  # Generate noise for torques based on the specified standard deviation in the noise configuration
        joint_torques = data.qfrc_actuator[self.act_ids] + torque_noise  # Add noise to the joint torques for observation

        # Concatenate everything into observation vector:
        obs = jp.concatenate([
            vel_commands,
            prev_action,
            body_accel,
            body_gyro,
            hip_joint_positions,
            knee_pos_sins,
            knee_pos_coss,
            joint_velocities,
            joint_torques
            ])
        
        info["rng"] = rng  # Update the RNG in the info dictionary for the next step
        return obs, info
    
    def _get_value_obs(self, data: mjx.Data, info: Dict[str, Any]) -> jax.Array:
        '''Extract value network observation from the simulation state'''
        pass


    def sample_command(self, rng: jax.Array) -> jax.Array:
        # Split RNG keys
        rng, lin_vel_rng, ang_vel_rng, zero_rng, zero_lin_rng, zero_ang_rng = jax.random.split(rng, num=6)
        # Determine whether to sample a zero command based on the specified probability
        is_zero_lin_command = jax.random.uniform(zero_lin_rng) < self.command_config.zero_lin_prob
        is_zero_ang_command = jax.random.uniform(zero_ang_rng) < self.command_config.zero_ang_prob
        is_all_zero_command = jax.random.uniform(zero_rng) < self.command_config.zero_all_prob

        # Sample linear and angular velocity commands uniformly within the specified ranges
        lin_vel_command = jax.random.uniform(lin_vel_rng, minval=-self.command_config.max_lin_vel, maxval=self.command_config.max_lin_vel)
        ang_vel_command = jax.random.uniform(ang_vel_rng, minval=-self.command_config.max_ang_vel, maxval=self.command_config.max_ang_vel)

        # If the sampled command is determined to be zero, set the corresponding command to zero
        # Either linear, angular, neither, or both can be set to zero based on configured probabilities
        lin_vel_command = jp.where(is_zero_lin_command, 0.0, lin_vel_command)
        ang_vel_command = jp.where(is_zero_ang_command, 0.0, ang_vel_command)
        lin_vel_command = jp.where(is_all_zero_command, 0.0, lin_vel_command)
        ang_vel_command = jp.where(is_all_zero_command, 0.0, ang_vel_command)
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
        self.hip_act_ids = jp.array([mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in ["fl_hip", "fr_hip", "bl_hip", "br_hip"]])
        self.knee_act_ids = jp.array([mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in ["fl_knee", "fr_knee", "bl_knee", "br_knee"]])
        self.wheel_act_ids = jp.array([mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in [
            "fl_wheel1", "fl_wheel2", "fr_wheel1", "fr_wheel2",
            "bl_wheel1", "bl_wheel2", "br_wheel1", "br_wheel2"
        ]])  
        self.fl_hip_act_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, "fl_hip")
        self.fr_hip_act_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, "fr_hip")
        self.bl_hip_act_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, "bl_hip")
        self.br_hip_act_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, "br_hip")
        self.fl_knee_act_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, "fl_knee")
        self.fr_knee_act_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, "fr_knee")
        self.bl_knee_act_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, "bl_knee")
        self.br_knee_act_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, "br_knee")
        self.fl_wheel1_act_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, "fl_wheel1")
        self.fl_wheel2_act_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, "fl_wheel2")
        self.fr_wheel1_act_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, "fr_wheel1")
        self.fr_wheel2_act_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, "fr_wheel2")
        self.bl_wheel1_act_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, "bl_wheel1")
        self.bl_wheel2_act_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, "bl_wheel2")
        self.br_wheel1_act_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, "br_wheel1")
        self.br_wheel2_act_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, "br_wheel2")

        self.act_ids = jp.concatenate([self.hip_act_ids, self.knee_act_ids, self.wheel_act_ids])  # Concatenate all actuator IDs for easy indexing

        hip_jids = self.mj_model.actuator_trnid[self.hip_act_ids, 0]  # Get the joint IDs for the hip actuators
        knee_jids = self.mj_model.actuator_trnid[self.knee_act_ids, 0]  # Get the joint IDs for the knee actuators
        wheel_jids = self.mj_model.actuator_trnid[self.wheel_act_ids, 0]  # Get the joint IDs for the wheel actuators

        self.hip_qposadrs = self.mj_model.jnt_qposadr[hip_jids]  # Get the qpos addresses for the hip joints
        self.knee_qposadrs = self.mj_model.jnt_qposadr[knee_jids]  # Get the qpos addresses for the knee joints
        self.wheel_qposadrs = self.mj_model.jnt_qposadr[wheel_jids]  # Get the qpos addresses for the wheel joints
        self.qpos_adrs = jp.concatenate([self.hip_qposadrs, self.knee_qposadrs, self.wheel_qposadrs])  # Concatenate all qpos addresses for easy indexing

        self.hip_qveladrs = self.mj_model.jnt_dofadr[hip_jids]  # Get the qvel addresses for the hip joints
        self.knee_qveladrs = self.mj_model.jnt_dofadr[knee_jids]  # Get the qvel addresses for the knee joints
        self.wheel_qveladrs = self.mj_model.jnt_dofadr[wheel_jids]  # Get the qvel addresses for the wheel joints
        self.qvel_adrs = jp.concatenate([self.hip_qveladrs, self.knee_qveladrs, self.wheel_qveladrs])  # Concatenate all qvel addresses for easy indexing
        
        # Torso and Head IDs:
        self.torso_body_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "torso")  # Get the body ID for the torso
        self.torso_qveladr = self.mj_model.body_dofadr[self.torso_body_id]  # Get the qvel address for the torso body

        self.head_body_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "head")  # Get the body ID for the head
        self.head_qveladr = self.mj_model.body_dofadr[self.head_body_id]  # Get the qvel address for the head body

        # Sensor addresses:
        self.body_lin_vel_sensor_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "torso_lin_vel")  # Get the sensor ID for the body linear velocity sensor
        self.body_lin_vel_adrs = self.mj_model.sensor_adr[self.body_lin_vel_sensor_id]  # Get the addresses for the body linear velocity sensor
        self.body_lin_vel_dim = self.mj_model.sensor_dim[self.body_lin_vel_sensor_id]  # Get the dimensionality of the body linear velocity sensor
        self.body_accel_sensor_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "torso_accel")  # Get the sensor ID for the body accelerometer
        self.body_accel_adrs = self.mj_model.sensor_adr[self.body_accel_sensor_id]  # Get the addresses for the body accelerometer
        self.body_accel_dim = self.mj_model.sensor_dim[self.body_accel_sensor_id]  # Get the dimensionality of the body accelerometer
        self.body_gyro_sensor_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "torso_gyro")  # Get the sensor ID for the body gyroscope
        self.body_gyro_adrs = self.mj_model.sensor_adr[self.body_gyro_sensor_id]  # Get the addresses for the body gyroscope
        self.body_gyro_dim = self.mj_model.sensor_dim[self.body_gyro_sensor_id]  # Get the dimensionality of the body gyroscope

        self.body_upvec_sensor_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "torso_upvec")  # Get the sensor ID for the body up vector sensor
        self.body_upvec_adrs = self.mj_model.sensor_adr[self.body_upvec_sensor_id]  # Get the addresses for the body up vector sensor
        self.body_upvec_dim = self.mj_model.sensor_dim[self.body_upvec_sensor_id]  # Get the dimensionality of the body up vector sensor



    def _add_terrain(self):
        '''Defines the terrain for the environment and then applies necessary contact pairs.
        Make sure to update this function and contact pairs for every new environment.'''
        self.model_spec.add_groundplane()
        self.model_spec.add_contact_pairs()  # Add contact pairs for wheel-ground interactions


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