import os
import sys
import glob
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # Add parent directory to path
# os.environ["JAX_PLATFORMS"] = "cpu"  # Force JAX to use CPU for this test
import mujoco
import mujoco.viewer
from typing import Optional, Dict, Union
from mujoco import mjx
import numpy as np
import time
import jax
import jax.numpy as jp
from brax.io import model
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import networks_vision as ppo_networks_vision
from brax.training.agents.ppo import train as ppo
from brax.training.acme import running_statistics
from pygame import joystick
import pygame
import environment.BaseEnv as BaseEnv
import environment.RoughTerrainEnv as RoughTerrainEnv

print(jax.devices())

def main():
    model_path = None # Path to the saved PPO model parameters

    # If model_path is none, select most recent model in policies directory
    if model_path is None:
        policies_dir = "policies"

        # Sort files in policies by modification time and select the most recent one
        files = filter(os.path.isfile, glob.glob('policies/*'))
        model_path = max(files, key=os.path.getmtime, default=None)
        if model_path is None:
            raise FileNotFoundError("No model files found in policies directory")
        print(f"Selected most recent model: {model_path}")

    # Initialize joystick
    joystick.init()
    js = joystick.Joystick(0)  # Initialize the first joystick
    js.init()
    pygame.init()
    

    # Initialize the environment
    env = RoughTerrainEnv.RoughTerrainEnv(difficulty=0.5)  # Create an instance of the BaseEnv environment
    # env = BaseEnv.BaseEnv()
    key = jax.random.PRNGKey(2)  # Initialize a random key for JAX

    # JIT compile the reset and step functions
    reset_fn = jax.jit(env.reset)
    step_fn = jax.jit(env.step)

    # Reset the environment to get initial state
    key, subkey = jax.random.split(key)
    state = reset_fn(subkey)

    # Create standard CPU mj_data
    mj_data = mujoco.MjData(env._mj_model)

    # Pull the initial MJX state into standard CPU mj_data
    mjx.get_data_into(mj_data, env.mj_model, state.data)
    dt = env._mj_model.opt.timestep  # Get the simulation timestep
    
    # Load the PPO model:
    params = model.load_params(model_path)
    inference_fn = ppo_networks.make_inference_fn(
        ppo_networks.make_ppo_networks(
            observation_size=env.observation_size,
            action_size=env.action_size,
            preprocess_observations_fn=running_statistics.normalize,
            policy_obs_key = "policy",
            value_obs_key = "value",
            policy_hidden_layer_sizes = [512, 256, 256, 128],
            value_hidden_layer_sizes = [512, 256, 256, 128],
        )
    )(params, deterministic=True)

    jit_inference_fn = jax.jit(inference_fn)


    # Launch standard MuJoCo viewer
    n_steps = 0
    with mujoco.viewer.launch_passive(env.mj_model, mj_data) as viewer:
        while viewer.is_running():
            # Keep track of step time
            start_time = time.time()
            # Update the standard CPU mj_data with the new MJX state
            mjx.get_data_into(mj_data, env.mj_model, state.data)

            viewer.sync()  # Sync the viewer to update the visualization

            # Get velocity command from joystick input (for testing purposes)
            pygame.event.pump()  # Process event queue to update joystick state
            
            reset_button = js.get_button(1)
            if reset_button:
                key, subkey = jax.random.split(key)
                state = reset_fn(subkey)  # Reset the environment if the reset button is pressed

            velocity_command = jp.array([
                -js.get_axis(1)*env.command_config.max_lin_vel,  # Linear velocity command from joystick axis 0
                -js.get_axis(0)*env.command_config.max_ang_vel,  # Linear velocity command from joystick axis 1
            ])
            info = dict(state.info)
            info['command'] = jp.asarray(velocity_command)
            state = state.replace(info=info)  # Update the state info with the new command

            


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
        

            # Sample a random action (for testing purposes) every 5 sim steps:
            if n_steps % 5 == 0:
                action = jit_inference_fn(state.obs, key)  # Get action from the PPO policy



            state = step_fn(state, action[0])  # Step the environment
            n_steps += 1

            elapsed = time.time()-start_time
            if elapsed < dt:
                time.sleep(dt - elapsed)  # Sleep to maintain real-time simulation
            

if __name__ == "__main__":
    main()



