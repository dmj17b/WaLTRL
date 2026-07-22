import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# os.environ["JAX_PLATFORMS"] = "cpu"  # Force JAX to use CPU for this test
import mujoco
from mujoco import mjx
import mujoco.viewer
import time
import jax.numpy as jp
import jax
import numpy as np
from environment.RoughTerrainEnv import RoughTerrainEnv
import pygame
from pygame import joystick
import matplotlib.pyplot as plt



def main():
    key = jax.random.PRNGKey(0)
    env = RoughTerrainEnv()
    reset_fn = jax.jit(env.reset)
    step_fn = jax.jit(env.step)

    state = reset_fn(key)
    mj_data = mujoco.MjData(env.mj_model)
    # mj_data.qpos = state.data.qpos
    dt = env.config.ctrl_dt
    n_steps = 0

    
    # Initialize Pygame and the joystick
    pygame.init()
    js = joystick.Joystick(0)
    hip_delta = 0.05

    def build_action():
        pygame.event.pump()  # Process event queue to update joystick state
        action = np.zeros(env.action_size) # Initialize action vector with zeros
        # Knee actions:
        fwd_knee_vel = js.get_axis(3)  # Assuming axis 1 controls forward/backward knee movement
        ang_knee_vel = js.get_axis(2)  # Assuming axis 0 controls angular knee movement
        left_knee_actions = -fwd_knee_vel + ang_knee_vel
        right_knee_actions = -fwd_knee_vel - ang_knee_vel
        left_knee_actions = np.clip(left_knee_actions, -1.0, 1.0)
        right_knee_actions = np.clip(right_knee_actions, -1.0, 1.0)

        action[env.fr_knee_act_id] = right_knee_actions
        action[env.fl_knee_act_id] = left_knee_actions
        action[env.br_knee_act_id] = right_knee_actions
        action[env.bl_knee_act_id] = left_knee_actions

        # Wheel actions:
        fwd_wheel_vel = js.get_axis(1)  # Assuming axis 3 controls forward/backward wheel movement
        ang_wheel_vel = js.get_axis(0)  # Assuming axis 2 controls angular wheel movement
        left_wheel_actions = -fwd_wheel_vel + ang_wheel_vel
        right_wheel_actions = -fwd_wheel_vel - ang_wheel_vel
        left_wheel_actions = np.clip(left_wheel_actions, -1.0, 1.0)
        right_wheel_actions = np.clip(right_wheel_actions, -1.0, 1.0)

        action[env.fr_wheel1_act_id] = right_wheel_actions
        action[env.fr_wheel2_act_id] = right_wheel_actions
        action[env.fl_wheel1_act_id] = left_wheel_actions
        action[env.fl_wheel2_act_id] = left_wheel_actions
        action[env.br_wheel1_act_id] = right_wheel_actions
        action[env.br_wheel2_act_id] = right_wheel_actions
        action[env.bl_wheel1_act_id] = left_wheel_actions
        action[env.bl_wheel2_act_id] = left_wheel_actions

        # Hip splay actions:
        if not hasattr(build_action, "front_splay"):
            build_action.front_splay = 0.0
        if not hasattr(build_action, "rear_splay"):
            build_action.rear_splay = 0.0
        build_action.front_splay = build_action.front_splay - hip_delta*js.get_button(5) + hip_delta*js.get_button(7)  # Assuming buttons 4 and 5 control hip splay
        build_action.rear_splay = build_action.rear_splay + hip_delta*js.get_button(4) - hip_delta*js.get_button(6)  # Assuming buttons 6 and 7 control hip splay
        build_action.front_splay = np.clip(build_action.front_splay, -1.0, 1.0)
        build_action.rear_splay = np.clip(build_action.rear_splay, -1.0, 1.0)

        action[env.fl_hip_act_id] = build_action.front_splay
        action[env.fr_hip_act_id] = build_action.front_splay
        action[env.bl_hip_act_id] = build_action.rear_splay
        action[env.br_hip_act_id] = build_action.rear_splay
        return action
    
    print("Initial command: ", state.info["command"])

    with mujoco.viewer.launch_passive(env.mj_model, mj_data) as viewer:
        while viewer.is_running():
            # Keep track of step time
            start_time = time.time()

            viewer.sync()  # Sync the viewer to update the visualization

            # Update the MJX state with any changes from viewer interactions (e.g., user dragging the model)
            state = state.replace(
                data=state.data.replace(
                    qpos=jp.array(mj_data.qpos),
                    qvel=jp.array(mj_data.qvel),
                    qfrc_applied=jp.array(mj_data.qfrc_applied),
                    xfrc_applied=jp.array(mj_data.xfrc_applied),
                    ctrl=jp.array(mj_data.ctrl),
                )
            )



            # Substitute joystick actions for testing:
            action = build_action()

            # Test wheel collision detection:
            print(f"T Pose Deviation Penalty: {state.metrics['penalty/t_pose_deviation']}, Wheel Collision Penalty: {state.metrics['penalty/wheel_collisions']}")
            # Reset conditions:
            if state.done:
                print("Episode done. Resetting environment.")
                state = reset_fn(key)  # Reset the environment if the episode is done
                n_steps = 0

            if n_steps > env.config.episode_length:
                key, subkey = jax.random.split(key)
                state = reset_fn(subkey)  # Reset the environment after 2000 steps for testing purposes
                n_steps = 0

            state = step_fn(state, action)  # Step the environment
            # print(f"Reward: {state.reward}, Command: {state.info['command']}, Done: {state.done}")

            # Update the standard CPU mj_data with the new MJX state
            mjx.get_data_into(mj_data, env.mj_model, state.data)
            n_steps += 1



            elapsed = time.time()-start_time
            if elapsed < dt:
                time.sleep(dt - elapsed)  # Sleep to maintain real-time simulation



if __name__ == "__main__":
    main()