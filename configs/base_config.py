import jax
import jax.numpy as jp
import flax.struct
import os
import sys
from ml_collections import config_dict  # Import config_dict for configuration management.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # Add parent directory to path

@flax.struct.dataclass
class RewardConfig:
    lin_vel_tracking: float = 100.0
    ang_vel_tracking: float = 100.0
    body_pitch_vel: float = 1.0
    body_roll_vel: float = 1.0
    orientation: float = 0.5
    low_torques: float = 0.0001
    body_z_vel: float = 0.5
    action_smoothing: float = 0.01
    flipped: float = 1000.0
    joint_vel: float = 10.0
    success_bonus: float = 1000.0


@flax.struct.dataclass
class CommandConfig:
    max_lin_vel: float = 2.0
    max_ang_vel: float = 1.0
    min_cmd_duration: float = 2.5
    max_cmd_duration: float = 5.5
    zero_lin_prob: float = 0.1
    zero_ang_prob: float = 0.1
    zero_all_prob: float = 0.1


@flax.struct.dataclass
class NoiseConfig:
    joint_pos_std: float = 0.001
    joint_vel_std: float = 0.1
    torque_std: float = 0.1
    body_accel_std: float = 0.01
    body_gyro_std: float = 0.01


def SimConfig() -> config_dict.ConfigDict:
    return config_dict.create(
        ctrl_dt = 0.02,
        sim_dt = 0.004,
        episode_length = 1000,
        action_repeat = 1,
        impl = 'warp',
        naconmax = 512*12,
        njmax = 44,
    )
