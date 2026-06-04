import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mujoco
from mujoco import mjx
import mujoco.viewer
import time
import jax.numpy as jp
import jax
from environment.BaseEnv import BaseEnv


def main():
    key = jax.random.PRNGKey(0)
    env = BaseEnv()
    reset_fn = jax.jit(env.reset)
    step_fn = jax.jit(env.step)

    state = reset_fn(key)
    mj_data = mujoco.MjData(env.mj_model)

    dt = env.config.sim_dt
    n_steps = 0

    mj_data.qpos[2] = 0.5
    
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

            action = jp.zeros(env.mj_model.nu)  # Dummy action for testing

            state = step_fn(state, action)  # Step the environment

            # Update the standard CPU mj_data with the new MJX state
            mjx.get_data_into(mj_data, env.mj_model, state.data)
            n_steps += 1

            if n_steps > env.config.episode_length:
                key, subkey = jax.random.split(key)
                state = reset_fn(subkey)  # Reset the environment after 2000 steps for testing purposes
                n_steps = 0

            elapsed = time.time()-start_time
            if elapsed < dt:
                time.sleep(dt - elapsed)  # Sleep to maintain real-time simulation

if __name__ == "__main__":
    main()