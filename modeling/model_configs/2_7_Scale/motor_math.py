import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import yaml

motor_config_path = "modeling/model_configs/2_7_Scale/motor_config.yaml"
motor_params = yaml.safe_load(open(motor_config_path, 'r'))

def rpm_to_rad_per_sec(rpm):
    return rpm * 2 * np.pi / 60.0

# Hip Motor Parameters:
ak70_10_params = {
    "stall_torque": 24.8,
    "no_load_speed_rpm": 480.0,
    "built_in_gear": 10.0,
}
ak70_10_params["rotor_stall_torque"] = ak70_10_params["stall_torque"] / ak70_10_params["built_in_gear"]
ak70_10_params["rotor_no_load_speed"] = rpm_to_rad_per_sec(ak70_10_params["no_load_speed_rpm"] * ak70_10_params["built_in_gear"])

hip_params = {}
hip_params["rotor_stall_torque"] = ak70_10_params["rotor_stall_torque"]
hip_params["rotor_no_load_speed"] = ak70_10_params["rotor_no_load_speed"]
hip_params["additional_gear"] = 33.0
hip_params["total_gear"] = hip_params["additional_gear"] * ak70_10_params["built_in_gear"]
hip_params["joint_stall_torque"] = hip_params["rotor_stall_torque"] * hip_params["total_gear"]
hip_params["joint_no_load_speed"] = hip_params["rotor_no_load_speed"] * hip_params["total_gear"]

# Knee Motor Parameters:
ak80_64_params = {
    "stall_torque": 120.0,
    "no_load_speed_rpm": 75.0,
    "built_in_gear": 64.0,
}
ak80_64_params["rotor_stall_torque"] = ak80_64_params["stall_torque"] / ak80_64_params["built_in_gear"]
ak80_64_params["rotor_no_load_speed"] = rpm_to_rad_per_sec(ak80_64_params["no_load_speed_rpm"] * ak80_64_params["built_in_gear"])

knee_params = {}
knee_params["rotor_stall_torque"] = ak80_64_params["rotor_stall_torque"]
knee_params["rotor_no_load_speed"] = ak80_64_params["rotor_no_load_speed"]
knee_params["additional_gear"] = 36.0/22.0
knee_params["total_gear"] = knee_params["additional_gear"] * ak80_64_params["built_in_gear"]
knee_params["joint_stall_torque"] = ak80_64_params["rotor_stall_torque"] * knee_params["total_gear"]
knee_params["joint_no_load_speed"] = ak80_64_params["rotor_no_load_speed"] * knee_params["total_gear"]

# Wheel Motor Parameters:
ak10_9_params = {
    "stall_torque": 48.0,
    "no_load_speed_rpm": 320.0,
    "built_in_gear": 9.0,
}
ak10_9_params["rotor_stall_torque"] = ak10_9_params["stall_torque"] / ak10_9_params["built_in_gear"]
ak10_9_params["rotor_no_load_speed"] = rpm_to_rad_per_sec(ak10_9_params["no_load_speed_rpm"] * ak10_9_params["built_in_gear"])

wheel_params = {}
wheel_params["rotor_stall_torque"] = ak10_9_params["rotor_stall_torque"]
wheel_params["rotor_no_load_speed"] = ak10_9_params["rotor_no_load_speed"]
wheel_params["additional_gear"] = 1.0
wheel_params["total_gear"] = wheel_params["additional_gear"] * ak10_9_params["built_in_gear"]
wheel_params["joint_stall_torque"] = ak10_9_params["rotor_stall_torque"] * wheel_params["total_gear"]
wheel_params["joint_no_load_speed"] = ak10_9_params["rotor_no_load_speed"] * wheel_params["total_gear"]

# Hip Parameters
# print("AK70-10 Params:", ak70_10_params)
print("Hip Joint Params:", hip_params)
# Knee Parameters
# print("AK80-64 Params:", ak80_64_params)
print("Knee Joint Params:", knee_params)

print("Wheel Joint Params:", wheel_params)