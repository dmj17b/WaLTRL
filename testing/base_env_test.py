import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mujoco
import mujoco.viewer
import time
import jax.numpy as jp
import jax
from environment.BaseEnv import BaseEnv


def main():
    env = BaseEnv()
    state = env.reset(rng=jax.random.PRNGKey(0))

if __name__ == "__main__":
    main()