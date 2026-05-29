import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # Add parent directory to pathimport mujoco
import mujoco.viewer
import jax
import jax.numpy as jp
import time
import modeling.GenModel as GenModel
import numpy as np
import pygame
from pygame import joystick
pygame.init()

model_config = 'modeling/model_configs/2_7_Scale/model_config.yaml'
motor_config = 'modeling/model_configs/2_7_Scale/motor_config.yaml'
model_spec = GenModel.GenModel(model_config, motor_config)  # Create an instance of the model generator
model_spec.add_scene()  # Add the scene to the model
model_spec.add_groundplane()  # Add a ground plane to the model

m = model_spec.spec.compile()
d = mujoco.MjData(m)

js = joystick.Joystick(0)  # Initialize the first joystick
js.init()

def deadzone(value, threshold=0.1):
    """Applies a deadzone to joystick input to prevent drift."""
    if abs(value) < threshold:
        return 0.0
    return value

# Launch standard MuJoCo viewer
with mujoco.viewer.launch_passive(m, d) as viewer:
    while viewer.is_running():
        # Keep track of step time
        start_time = time.time()

        viewer.sync()  # Sync the viewer to update the visualization


        mujoco.mj_step(m, d)  # Step the simulation forward

        # Change qpos based on joystick states
        pygame.event.pump()  # Process event queue to update joystick state

        d.qpos[0] += 0.01*deadzone(js.get_axis(0))  # Use the horizontal axis of the joystick to control x position
        d.qpos[1] -= 0.01*deadzone(js.get_axis(3))  # Use the vertical axis of the joystick to control z position


        # Rudimentary time keeping, will drift relative to wall clock.
        time_until_next_step = (m.opt.timestep - (time.time() - start_time))
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
