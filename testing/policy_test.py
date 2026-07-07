import os
import sys
import time
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import jax
import jax.numpy as jp
import mujoco
import mujoco.viewer
from brax.io import model
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.acme import running_statistics
from mujoco import mjx
from mujoco_playground import wrapper

from environment.BaseEnv import BaseEnv


def main():
    model_path = Path("policies/test3")
    if not model_path.exists():
        raise FileNotFoundError(f"Missing PPO checkpoint directory: {model_path}")

    env = BaseEnv()
    env = wrapper.wrap_for_brax_training(
        env,
        episode_length=env.config.episode_length,
        action_repeat=env.config.action_repeat,
    )

    reset_fn = jax.jit(env.reset)
    step_fn = jax.jit(env.step)

    params = model.load_params(str(model_path))
    inference_fn = ppo_networks.make_inference_fn(
        ppo_networks.make_ppo_networks(
            observation_size=env.observation_size,
            action_size=env.action_size,
            preprocess_observations_fn=running_statistics.normalize,
        )
    )(params, deterministic=True)
    policy_fn = jax.jit(inference_fn)

    rng = jax.random.PRNGKey(2)
    rng, reset_rng = jax.random.split(rng)
    state = reset_fn(reset_rng[None])

    mj_data = mujoco.MjData(env.mj_model)
    mjx.get_data_into([mj_data], env.mj_model, state.data)

    dt = env.mj_model.opt.timestep
    previous_done = False

    with mujoco.viewer.launch_passive(env.mj_model, mj_data) as viewer:
        while viewer.is_running():
            start_time = time.time()

            if bool(state.done[0]):
                if not previous_done:
                    print(f"episode finished at wrapped step {int(state.info['steps'][0])}")
                previous_done = True
            else:
                previous_done = False

            action, extras = policy_fn(state.obs, rng)
            del extras

            state = step_fn(state, action)

            mjx.get_data_into([mj_data], env.mj_model, state.data)
            viewer.sync()

            command = jp.asarray(state.info["command"][0])
            reward = float(state.reward[0])
            done = bool(state.done[0])
            print(
                f"step={int(state.info['steps'][0])} command={command.tolist()} "
                f"reward={reward:.3f} done={done}"
            )

            elapsed = time.time() - start_time
            if elapsed < dt:
                time.sleep(dt - elapsed)


if __name__ == "__main__":
    main()



