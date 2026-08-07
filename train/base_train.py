import os
import sys
os.environ['XLA_FLAGS'] = (
    '--xla_gpu_triton_gemm_any=True '
    '--xla_gpu_enable_latency_hiding_scheduler=true '
    '--xla_gpu_enable_highest_priority_async_stream=true '
)
from wandb.util import np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # Add parent directory to path
import jax
import jax.numpy as jp
import mujoco as mj
import mujoco.viewer
import time
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import networks_vision as ppo_networks_vision
from brax.training.agents.ppo import train as ppo
from brax.io import model
from brax.io import html
from ml_collections import config_dict
import functools
from mujoco_playground import wrapper
import inspect
from pathlib import Path
import wandb
from environment.BaseEnv import BaseEnv
from environment.RoughTerrainEnv import RoughTerrainEnv
from configs.base_config import SimConfig, RewardConfig, CommandConfig, NoiseConfig  # Import configuration dataclasses.


def main():
    resume_path = None  # Path to the saved PPO model parameters

    notes = "Messed up thinking I was changing the correct reward function. Now formulated with task reward"

    # env = BaseEnv(sim_config = SimConfig(),
    #         reward_config = RewardConfig(),
    #         command_config = CommandConfig(),)  # Create an instance of the BaseEnv environment

    env = RoughTerrainEnv(sim_config = SimConfig(),
            reward_config = RewardConfig(),
            command_config = CommandConfig(),
            difficulty = 0.8,
            smoothing = 0.6)  # Create an instance of the RoughTerrainEnv environment

    wrapper_fn = wrapper.wrap_for_brax_training  # Use the standard Brax wrapper for training
    
    env_cfg = env.config  # Retrieve the environment configuration
    ppo_params = {
        'action_repeat': 1,
        'batch_size': 256,  
        'discounting': 0.995,
        'entropy_cost': 0.01,
        'episode_length': env_cfg.episode_length,
        'learning_rate': 3e-4,
        'num_envs': 4096,
        'num_evals': 20,
        'num_minibatches': 16,
        'num_updates_per_batch': 4,
        'num_timesteps': 500_000_000,
        'normalize_observations': True,
        'reward_scaling': 1.0,
        'unroll_length': 256,
        'deterministic_eval': True,
        'seed': 42,
        }

    #---------- WandB logging setup ------------#
    wandb.login()
    project = "WaLTRL"
    reward_config = env.reward_config
    command_config = env.command_config
    wandb_config = {
        "ppo_params": ppo_params,
        "reward_config": reward_config,
        "command_config": command_config,
        "env_config": env_cfg,
        "notes": notes,
        "resume_path": resume_path,
    }
    run = wandb.init(project=project, config=wandb_config)
    # Use WandB run name for saving policy parameters
    save_path = "policies/" + wandb.run.name

    def _to_float(value):
        """Safely converts JAX/NumPy scalars to plain Python floats for logging."""
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
        
   
    # Progress callback
    def progress(num_steps, metrics):
        if not hasattr(progress, "eval_counter"):
            progress.eval_counter = 0

        # FIX: Print the available keys on the first run so you know exactly what Brax named them
        if progress.eval_counter == 0:
            print("Available Metric Keys:", list(metrics.keys()))

        print(f"\nEvaluation #{progress.eval_counter}:")
        # Print all metrics for debugging:
        for key, value in metrics.items():
            print(f"{key}: {value:.4f}")

        wandb_metrics = {k: _to_float(v) for k, v in metrics.items()}
        wandb_metrics = {k: v for k, v in wandb_metrics.items() if v is not None}
        wandb_metrics["num_steps"] = int(num_steps)
        run.log(wandb_metrics, step=int(num_steps))
        
        progress.eval_counter += 1
        

    
    ppo_training_params = ppo_params
    
    network_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes = [512, 256, 256, 128],
        value_hidden_layer_sizes = [512, 256, 256, 128],
        policy_obs_key = "policy",
        value_obs_key = "value",
    )

    train_fn = functools.partial(
        ppo.train,
        **dict(ppo_training_params),
        network_factory = network_factory,
        progress_fn=progress,
    )
    
    train_kwargs = dict(
        environment=env,
        wrap_env_fn=wrapper_fn,  # Use the appropriate wrapper function for training
    )

    # If a resume path is provided, load the parameters and pass them, otherwise start training from scratch
    if resume_path is not None and Path(resume_path).exists():
        print(f"Resuming PPO training from {resume_path}...")
        resume_params = model.load_params(resume_path)
        make_inference_fn, params, metrics = train_fn(
           **train_kwargs,
            restore_params=resume_params
        )
    else:
        print("Starting PPO training from scratch...")
        make_inference_fn, params, metrics = train_fn(
            **train_kwargs
        )



    # Save the trained policy parameters and metrics:
    model.save_params(save_path, params)

    run.finish()  # Finish the WandB run after training is complete
    print(f"\nFinal metrics: {metrics}")

if __name__ == "__main__":
    main()  
