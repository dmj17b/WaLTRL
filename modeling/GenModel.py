import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # Add parent directory to path
from typing import Any
from absl import app
from pathlib import Path
import yaml
import numpy as np
import mujoco



class GenModel():
    def __init__(
        self,
        model_config_path: Any,
        motor_config_path: Any,
        include_waist_joint: bool = True,
    ) -> None:
        # Build model using Mujoco Spec:
        spec = mujoco.MjSpec()

        color = np.array([177/255, 166/255, 136/255, 1])

        # Parse Configs:
        model_config = yaml.safe_load(Path(model_config_path).read_text())


        # Torso Params:
        torso_length = model_config['torso_params']['length']
        torso_width = model_config['torso_params']['width']
        torso_height = model_config['torso_params']['height']
        torso_mass = model_config['torso_params']['mass']
        torso_start_pos = model_config['torso_params']['start_pos']

        # Head Params:
        head_length = model_config['head_params']['length']
        head_width = model_config['head_params']['width']
        head_height = model_config['head_params']['height']
        head_mass = model_config['head_params']['mass']
        head_offset = np.asarray(model_config['head_params']['offset'])

        # Thigh Params:
        thigh_length = model_config['thigh_params']['length']
        thigh_width = model_config['thigh_params']['width']
        thigh_mass = model_config['thigh_params']['mass']
        thigh_torso_offset = np.asarray(model_config['thigh_params']['torso_offset'])
        thigh_head_offset = np.asarray(model_config['thigh_params']['head_offset'])

        # Shin Params:
        shin_length = model_config['shin_params']['length']
        shin_width = model_config['shin_params']['width']
        shin_mass = model_config['shin_params']['mass']
        shin_offset = np.asarray(model_config['shin_params']['offset'])

        # Wheel Params:
        wheel_radius = model_config['wheel_params']['radius']
        wheel_width = model_config['wheel_params']['width']
        wheel_mass = model_config['wheel_params']['mass']
        wheel_offset = np.asarray(model_config['wheel_params']['offset'])
        wheel_friction = np.asarray(model_config['wheel_params']['friction'])
        wheel_solref = np.asarray(model_config['wheel_params']['solref'])

        motor_config = yaml.safe_load(Path(motor_config_path).read_text())
        if(motor_config['waist_params']['range'][0] == 0):
            print ('waist jnt off')

        waist_spring_stiffness = motor_config['waist_params']['spring_stiffness']
        waist_damping = motor_config['waist_params']['damping']
        waist_range = motor_config['waist_params']['range']

        hip_kp = motor_config['hip_params']['Kp']
        hip_kd = motor_config['hip_params']['Kd']
        hip_gear_ratio = motor_config['hip_params']['gear_ratio']
        hip_stall_torque = motor_config['hip_params']['stall_torque'] * hip_gear_ratio  # Convert rotor stall torque to joint stall torque
        hip_no_load_speed = motor_config['hip_params']['no_load_speed']
        hip_rotor_inertia = motor_config['hip_params']['rotor_inertia']
        hip_damping = motor_config['hip_params']['damping']

        hip_armature = hip_rotor_inertia*hip_gear_ratio**2

        knee_kp = motor_config['knee_params']['Kp']
        knee_kd = motor_config['knee_params']['Kd']
        knee_gear_ratio = motor_config['knee_params']['gear_ratio']
        knee_stall_torque = motor_config['knee_params']['stall_torque'] * knee_gear_ratio  # Convert rotor stall torque to joint stall torque
        knee_no_load_speed = motor_config['knee_params']['no_load_speed']
        knee_rotor_inertia = motor_config['knee_params']['rotor_inertia']
        knee_damping = motor_config['knee_params']['damping']

        knee_armature = knee_rotor_inertia*knee_gear_ratio**2


        wheel_kp = motor_config['wheel_params']['Kp']
        wheel_kd = motor_config['wheel_params']['Kd']
        wheel_gear_ratio = motor_config['wheel_params']['gear_ratio']
        wheel_stall_torque = motor_config['wheel_params']['stall_torque'] * wheel_gear_ratio  # Convert rotor stall torque to joint stall torque
        wheel_no_load_speed = motor_config['wheel_params']['no_load_speed']
        wheel_rotor_inertia = motor_config['wheel_params']['rotor_inertia']
        wheel_damping = motor_config['wheel_params']['damping']


        wheel_armature = wheel_rotor_inertia*wheel_gear_ratio**2

        # Turn off contacts for all bodies:
        spec.default.geom.contype = 0
        spec.default.geom.conaffinity = 0

        # Add Torso to World Body:
        torso_body = spec.worldbody.add_body(
            name='torso',
            pos=torso_start_pos,
            quat=[1, 0, 0, 0],
        )
        torso_body.add_joint(
            type=mujoco.mjtJoint.mjJNT_FREE,
            name='torso_joint',
        )
        torso_body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[
                torso_length / 2, torso_width / 2, torso_height / 2,
            ],
            mass=torso_mass,
            rgba = color,
            name = 'torso_geom',
        )
        # Add sensor site to torso:
        torso_body.add_site(
            name = 'torso_com',
            pos = [0.15, 0, 0],
            type = mujoco.mjtGeom.mjGEOM_SPHERE,
            rgba = [1, 0, 0, 1],
            size = [0.05, 0.05, 0.05],
            )

        # Torso Kinematic Chain:
        parents = ['torso', 'thigh', 'shin', 'shin']
        children = ['thigh', 'shin', 'front_wheel', 'rear_wheel']
        side = ['left', 'right']
        mirror = [np.array([1, 1, 1]), np.array([1, -1, 1])]
        torso_children_params = {
            'thigh': {
                'body_pos': np.array([0, torso_width / 2, 0]) + thigh_torso_offset,
                'body_quat': np.array([1, 0, 0, 0]),
                'geom_type': mujoco.mjtGeom.mjGEOM_CAPSULE,
                'geom_size': np.array([thigh_width / 2, thigh_length / 2, 0]),
                'geom_pos': np.array([0, thigh_width / 2, -thigh_length / 2]),
                'geom_quat': np.array([1, 0, 0, 0]),
                'mass': thigh_mass,
                'armature': hip_armature,
                'damping': hip_damping,
            },
            'shin': {
                'body_pos': np.array([0, shin_width / 2 + thigh_width, -thigh_length]) + shin_offset,
                'body_quat': np.array([1, 0, 0, 0]),
                'geom_type': mujoco.mjtGeom.mjGEOM_CAPSULE,
                'geom_size': np.array([shin_width / 2, shin_length / 2, 0]),
                'geom_pos': np.array([0, 0, 0]),
                'geom_quat': np.array([1, 0, 1, 0]),
                'mass': shin_mass,
                'armature': knee_armature,
                'damping': knee_damping,
            },
            'front_wheel': {
                'body_pos': np.array([shin_length / 2, wheel_width / 2 + shin_width/2, 0]) + wheel_offset,
                'body_quat': np.array([1, 0, 0, 0]),
                'geom_type': mujoco.mjtGeom.mjGEOM_ELLIPSOID,
                'geom_size': np.array([wheel_radius, wheel_radius, wheel_width/2]),
                'geom_pos': np.array([0, 0, 0]),
                'geom_quat': np.array([1, 1, 0, 0]),
                'mass': wheel_mass,
                'armature': wheel_armature,
                'damping': wheel_damping,
            },
            'rear_wheel': {
                'body_pos': np.array([-shin_length / 2, wheel_width / 2 + shin_width/2, 0]) + wheel_offset,
                'body_quat': np.array([1, 0, 0, 0]),
                'geom_type': mujoco.mjtGeom.mjGEOM_ELLIPSOID,
                'geom_size': np.array([wheel_radius, wheel_radius, wheel_width/2]),
                'geom_pos': np.array([0, 0, 0]),
                'geom_quat': np.array([1, 1, 0, 0]),
                'mass': wheel_mass,
                'armature': wheel_armature,
                'damping': wheel_damping,
            },
        }

        # Torso Children:
        for side, mirror in zip(side, mirror):
            for parent, child in zip(parents, children):
                if parent != 'torso':
                    parent_name = f'torso_{side}_{parent}'
                    body_name = f'torso_{side}_{child}'
                    joint_name = f'torso_{side}_{parent}_{child}_joint'
                else:
                    parent_name = parent
                    body_name = f'torso_{side}_{child}'
                    joint_name = f'{parent}_{side}_{child}_joint'
                geom_name = f'{body_name}_geom'
                parent_body = spec.worldbody.find_child(parent_name)
                body = parent_body.add_body(
                    name=body_name,
                    pos=mirror * torso_children_params[child]['body_pos'],
                    quat=torso_children_params[child]['body_quat'],
                )
                body.add_joint(
                    type=mujoco.mjtJoint.mjJNT_HINGE,
                    name=joint_name,
                    axis=[0, 1, 0],
                    armature = torso_children_params[child]['armature'],
                    damping = torso_children_params[child]['damping'],
                )

                if(child == 'front_wheel' or child == 'rear_wheel'):
                    body.add_geom(
                        name=geom_name,
                        type=torso_children_params[child]['geom_type'],
                        size=torso_children_params[child]['geom_size'],
                        pos=mirror * torso_children_params[child]['geom_pos'],
                        quat=torso_children_params[child]['geom_quat'],
                        mass=torso_children_params[child]['mass'],
                        friction = wheel_friction,
                        solref = wheel_solref,
                        rgba = color,
                    )
                else:
                    body.add_geom(
                        name=geom_name,
                        type=torso_children_params[child]['geom_type'],
                        size=torso_children_params[child]['geom_size'],
                        pos=mirror * torso_children_params[child]['geom_pos'],
                        quat=torso_children_params[child]['geom_quat'],
                        mass=torso_children_params[child]['mass'],
                        rgba = color,
                    )
                    


        # Add Head to Torso:
        head_position = np.asarray([
            (head_length / 2) + (torso_length / 2), 0, 0] + head_offset,
        )
        joint_position = -np.asarray([
            (head_length / 2), 0, 0
        ])
        head_body = torso_body.add_body(
            name='head',
            pos=head_position,
            quat=[1, 0, 0, 0],
        )
        if include_waist_joint:
            head_body.add_joint(
                type=mujoco.mjtJoint.mjJNT_HINGE,
                name='head_joint',
                pos=joint_position,
                axis=[1, 0, 0],
                stiffness = waist_spring_stiffness,
                damping = waist_damping,
                range = waist_range,
            )
        head_body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[head_length / 2, head_width / 2, head_height / 2],
            pos=[0, 0, 0],
            quat=[1, 0, 0, 0],
            mass=head_mass,
            rgba = color,
            name = 'head_geom',
        )
        # Add sensor site to head:
        head_body.add_site(
            name = 'head_com',
            pos = [0.0, 0, 0],
            type = mujoco.mjtGeom.mjGEOM_SPHERE,
            rgba = [1, 0, 0, 1],
            size = [0.05, 0.05, 0.05],
            )

        # Head Kinematic Chain:
        parents = ['head', 'thigh', 'shin', 'shin']
        children = ['thigh', 'shin', 'front_wheel', 'rear_wheel']
        side = ['left', 'right']
        mirror = [np.array([1, 1, 1]), np.array([1, -1, 1])]
        head_children_params = {
            'thigh': {
                'body_pos': np.array([0, head_width / 2, 0]) + thigh_head_offset,
                'body_quat': np.array([1, 0, 0, 0]),
                'geom_type': mujoco.mjtGeom.mjGEOM_CAPSULE,
                'geom_size': np.array([thigh_width / 2, thigh_length / 2, 0]),
                'geom_pos': np.array([0, thigh_width / 2, -thigh_length / 2]),
                'geom_quat': np.array([1, 0, 0, 0]),
                'mass': thigh_mass,
                'armature': hip_armature,
                'damping': hip_damping,
            },
            'shin': {
                'body_pos': np.array([0, shin_width / 2 + thigh_width, -thigh_length]) + shin_offset,
                'body_quat': np.array([1, 0, 0, 0]),
                'geom_type': mujoco.mjtGeom.mjGEOM_CAPSULE,
                'geom_size': np.array([shin_width / 2, shin_length / 2, 0]),
                'geom_pos': np.array([0, 0, 0]),
                'geom_quat': np.array([1, 0, 1, 0]),
                'mass': shin_mass,
                'armature': knee_armature,
                'damping': knee_damping,
            },
            'front_wheel': {
                'body_pos': np.array([shin_length / 2, wheel_width / 2 + shin_width/2, 0]) + wheel_offset,
                'body_quat': np.array([1, 0, 0, 0]),
                'geom_type': mujoco.mjtGeom.mjGEOM_ELLIPSOID,
                'geom_size': np.array([wheel_radius, wheel_radius, wheel_width/2]),
                'geom_pos': np.array([0, 0, 0]),
                'geom_quat': np.array([1, 1, 0, 0]),
                'mass': wheel_mass,
                'armature': wheel_armature,
                'damping': wheel_damping,
            },
            'rear_wheel': {
                'body_pos': np.array([-shin_length / 2, wheel_width / 2 + shin_width/2, 0]) + wheel_offset,
                'body_quat': np.array([1, 0, 0, 0]),
                'geom_type': mujoco.mjtGeom.mjGEOM_ELLIPSOID,
                'geom_size': np.array([wheel_radius, wheel_radius, wheel_width/2]),
                'geom_pos': np.array([0, 0, 0]),
                'geom_quat': np.array([1, 1, 0, 0]),
                'mass': wheel_mass,
                'armature': wheel_armature,
                'damping': wheel_damping,
            },
        }

        # head Children:
        for side, mirror in zip(side, mirror):
            for parent, child in zip(parents, children):
                if parent != 'head':
                    parent_name = f'head_{side}_{parent}'
                    body_name = f'head_{side}_{child}'
                    joint_name = f'head_{side}_{parent}_{child}_joint'
                else:
                    parent_name = parent
                    body_name = f'head_{side}_{child}'
                    joint_name = f'{parent}_{side}_{child}_joint'
                geom_name = f'{body_name}_geom'
                parent_body = spec.worldbody.find_child(parent_name)
                body = parent_body.add_body(
                    name=body_name,
                    pos=mirror * head_children_params[child]['body_pos'],
                    quat=head_children_params[child]['body_quat'],
                )
                body.add_joint(
                    type=mujoco.mjtJoint.mjJNT_HINGE,
                    name=joint_name,
                    axis=[0, 1, 0],
                    armature = head_children_params[child]['armature'],
                    damping = head_children_params[child]['damping'],
                )                

                if(child == 'front_wheel' or child == 'rear_wheel'):
                    body.add_geom(
                        name=geom_name,
                        type=head_children_params[child]['geom_type'],
                        size=head_children_params[child]['geom_size'],
                        pos=mirror * head_children_params[child]['geom_pos'],
                        quat=head_children_params[child]['geom_quat'],
                        mass=head_children_params[child]['mass'],
                        friction = wheel_friction,
                        solref = wheel_solref,
                        rgba = color,
                    )
                else:
                    body.add_geom(
                        name=geom_name,
                        type=head_children_params[child]['geom_type'],
                        size=head_children_params[child]['geom_size'],
                        pos=mirror * head_children_params[child]['geom_pos'],
                        quat=head_children_params[child]['geom_quat'],
                        mass=head_children_params[child]['mass'],
                        rgba = color,
                    )


# Adding Actuators:
        # Back left leg
        spec.add_actuator(
            name='bl_hip',
            target='torso_left_thigh_joint',
            trntype = mujoco.mjtTrn.mjTRN_JOINT,
            forcerange = [-hip_stall_torque, hip_stall_torque],
        )
        spec.add_actuator(
            name='bl_knee',
            target='torso_left_thigh_shin_joint',
            trntype = mujoco.mjtTrn.mjTRN_JOINT,
            forcerange = [-knee_stall_torque, knee_stall_torque],
        )
        spec.add_actuator(
            name='bl_wheel1',
            target='torso_left_shin_front_wheel_joint',
            trntype = mujoco.mjtTrn.mjTRN_JOINT,
            forcerange = [-wheel_stall_torque, wheel_stall_torque],
        )
        spec.add_actuator(
            name='bl_wheel2',
            target='torso_left_shin_rear_wheel_joint',
            trntype = mujoco.mjtTrn.mjTRN_JOINT,
            forcerange = [-wheel_stall_torque, wheel_stall_torque],
        )

        # Back right leg
        spec.add_actuator(
            name='br_hip',
            target='torso_right_thigh_joint',
            trntype = mujoco.mjtTrn.mjTRN_JOINT,
            forcerange = [-hip_stall_torque, hip_stall_torque],
        )
        spec.add_actuator(
            name='br_knee',
            target='torso_right_thigh_shin_joint',
            trntype = mujoco.mjtTrn.mjTRN_JOINT,
            forcerange = [-knee_stall_torque, knee_stall_torque],
        )
        spec.add_actuator(
            name='br_wheel1',
            target='torso_right_shin_front_wheel_joint',
            trntype = mujoco.mjtTrn.mjTRN_JOINT,
            forcerange = [-wheel_stall_torque, wheel_stall_torque],
        )
        spec.add_actuator(
            name='br_wheel2',
            target='torso_right_shin_rear_wheel_joint',
            trntype = mujoco.mjtTrn.mjTRN_JOINT,
            forcerange = [-wheel_stall_torque, wheel_stall_torque],
        )

        # Front left leg
        spec.add_actuator(
            name='fl_hip',
            target='head_left_thigh_joint',
            trntype = mujoco.mjtTrn.mjTRN_JOINT,
            forcerange = [-hip_stall_torque, hip_stall_torque],
        )
        spec.add_actuator(
            name='fl_knee',
            target='head_left_thigh_shin_joint',
            trntype = mujoco.mjtTrn.mjTRN_JOINT,
            forcerange = [-knee_stall_torque, knee_stall_torque],
        )
        spec.add_actuator(
            name='fl_wheel1',
            target='head_left_shin_front_wheel_joint',
            trntype = mujoco.mjtTrn.mjTRN_JOINT,
            forcerange = [-wheel_stall_torque, wheel_stall_torque],
        )
        spec.add_actuator(
            name='fl_wheel2',
            target='head_left_shin_rear_wheel_joint',
            trntype = mujoco.mjtTrn.mjTRN_JOINT,
            forcerange = [-wheel_stall_torque, wheel_stall_torque],
        )

        # Front right leg
        spec.add_actuator(
            name='fr_hip',
            target='head_right_thigh_joint',
            trntype = mujoco.mjtTrn.mjTRN_JOINT,
            forcerange = [-hip_stall_torque, hip_stall_torque],
        )
        spec.add_actuator(
            name='fr_knee',
            target='head_right_thigh_shin_joint',
            trntype = mujoco.mjtTrn.mjTRN_JOINT,
            forcerange = [-knee_stall_torque, knee_stall_torque],
        )
        spec.add_actuator(
            name='fr_wheel1',
            target='head_right_shin_front_wheel_joint',
            trntype = mujoco.mjtTrn.mjTRN_JOINT,
            forcerange = [-wheel_stall_torque, wheel_stall_torque],
        )
        spec.add_actuator(
            name='fr_wheel2',
            target='head_right_shin_rear_wheel_joint',
            trntype = mujoco.mjtTrn.mjTRN_JOINT,
            forcerange = [-wheel_stall_torque, wheel_stall_torque],
        )

        # Add linear velocity sensor:
        spec.add_sensor(
            name = 'torso_lin_vel',
            type = mujoco.mjtSensor.mjSENS_VELOCIMETER,
            objname = 'torso_com',
            objtype = mujoco.mjtObj.mjOBJ_SITE,
        )
        # Add accelerometer and gyroscope sensors:
        spec.add_sensor(
            name = 'torso_accel',
            type = mujoco.mjtSensor.mjSENS_ACCELEROMETER,
            objname = 'torso_com',
            objtype = mujoco.mjtObj.mjOBJ_SITE,
        )
        spec.add_sensor(
            name = 'torso_gyro',
            type = mujoco.mjtSensor.mjSENS_GYRO,
            objname = 'torso_com',
            objtype = mujoco.mjtObj.mjOBJ_SITE,
        )

        # Orientation sensor:
        spec.add_sensor(
            name = 'torso_upvector',
            type = mujoco.mjtSensor.mjSENS_FRAMEZAXIS,
            objname = 'torso_com',
            objtype = mujoco.mjtObj.mjOBJ_SITE,
        )


        self.spec = spec


    def add_contact_pairs(self,
                          added_obstacles: list = None):
        # Add contact pairs for wheel-ground interactions:
        wheel_names = [
            'torso_left_front_wheel_geom',
            'torso_left_rear_wheel_geom',
            'torso_right_front_wheel_geom',
            'torso_right_rear_wheel_geom',
            'head_left_front_wheel_geom',
            'head_left_rear_wheel_geom',
            'head_right_front_wheel_geom',
            'head_right_rear_wheel_geom',
        ]
        shin_names = [
            'torso_left_shin_geom',
            'torso_right_shin_geom',
            'head_left_shin_geom',
            'head_right_shin_geom',]
        
        # Head and Torso to Floor Contact Pairs:
        self.spec.add_pair(
            geomname1 = 'torso_geom',
            geomname2 = 'floor',
        )
        self.spec.add_pair(
            geomname1 = 'head_geom',
            geomname2 = 'floor',
        )

        # Wheel - Floor Contact Pairs:
        for wheel_name in wheel_names:
            self.spec.add_pair(
                geomname1 = wheel_name,
                geomname2 = 'floor',
            )
        # Shin - Floor Contact Pairs:
        for shin_name in shin_names:
            self.spec.add_pair(
                geomname1 = shin_name,
                geomname2 = 'floor',
            )
        # Wheel-Wheel Contact Pairs:
        for i in range(len(wheel_names)):
            if 'left' in wheel_names[i] and 'torso' in wheel_names[i]:
                for j in range(len(wheel_names)):
                    if 'left' in wheel_names[j] and 'head' in wheel_names[j]:
                        self.spec.add_pair(
                            geomname1 = wheel_names[i],
                            geomname2 = wheel_names[j],
                        )
            if 'right' in wheel_names[i] and 'torso' in wheel_names[i]:
                for j in range(len(wheel_names)):
                    if 'right' in wheel_names[j] and 'head' in wheel_names[j]:
                        self.spec.add_pair(
                            geomname1 = wheel_names[i],
                            geomname2 = wheel_names[j],
                        )

        # If additional obstacles are provided, add contact pairs for them:
        if added_obstacles is not None:
            for obstacle in added_obstacles:
                self.spec.add_pair(
                    geomname1 = 'torso_geom',
                    geomname2 = obstacle,
                )
                self.spec.add_pair(
                    geomname1 = 'head_geom',
                    geomname2 = obstacle,
                )
                for wheel_name in wheel_names:
                    self.spec.add_pair(
                        geomname1 = wheel_name,
                        geomname2 = obstacle,
                    )
                for shin_name in shin_names:
                    self.spec.add_pair(
                        geomname1 = shin_name,
                        geomname2 = obstacle,
                    )

    def add_scene(self):
        # Create skybox so background isn't just black
        self.spec.add_texture(type = mujoco.mjtTexture.mjTEXTURE_SKYBOX,
                              builtin = mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                                width = 300,
                                height = 300,
                                name="skybox",
                                rgb2 = [0.4, 0.7, 0.9],)
        # Ground plane texture/material
        self.spec.add_material(name="groundplane_material",
                        texrepeat=[2, 2],
                        reflectance=0., 
                        ).textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = 'ground_texture'
        # Create ground plane texture/material
        ground = self.spec.add_texture(type = mujoco.mjtTexture.mjTEXTURE_2D,
                              name="ground_texture",
                              builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER, 
                              width=200, 
                              height=200, 
                              rgb1=[0.5, 0.8, 0.9], 
                              rgb2=[0.5, 0.9, 0.8],
                              markrgb=[0.8, 0.8, 0.8])
        # Add an array of lights to the scene:
        for i in range(15):
            for j in range(15):
                self.spec.worldbody.add_light(
                    pos=[3*i, 3*j, 50],
                    dir=[0, 0, -1],
                    diffuse=[0.1, 0.1, 0.1],
                    specular=[0.1, 0.1, 0.1],
                    intensity=1.0,
                )
    def add_groundplane(self):
        self.spec.worldbody.add_geom(
            type=mujoco.mjtGeom.mjGEOM_PLANE,
            size=[0, 0, 0.05],
            material="groundplane_material",
            name = 'floor',
        )


    def add_box(self, pos:list, size:list, **kwargs):
        if 'name' in kwargs:
            name = kwargs['name']
        else:
            name = 'box'
        if 'rotation' in kwargs:
            rotation = kwargs['rotation']
        else:
            rotation = [0, 0, 0]
        self.spec.worldbody.add_body(pos=pos,
                                     name=name,
                                     euler = rotation).add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=size,
            )

            
    def add_payload(self, mass: float = 32.0, body_loc: list = [0,0,0.2], size = [0.1, 0.1, 0.1]):
        payload = self.spec.bodies[1].add_body(
            name='payload',
            pos=body_loc,
            quat=[1, 0, 0, 0],
        )
        payload.add_geom(
            type = mujoco.mjtGeom.mjGEOM_BOX,
            size = size,
            mass = mass
        )
    def add_log(self, d: float = 0.3, length: float = 5.0, pos: list = [0,-3,0.15]):
        self.spec.worldbody.add_geom(
            type = mujoco.mjtGeom.mjGEOM_CAPSULE,
            size = [d/2, length/2, d],
            pos = [pos[0], pos[1], d/2],
            quat = [1, 0, 1, 0],
            name = 'log'
        )
        
    def add_incline(self, angle_deg: float = 30, length: float = 5.0, pos: list = [0,3,2], width: float = 2):
        angle_rad = np.deg2rad(angle_deg)
        height = length/2*np.sin(abs(angle_rad))
        self.spec.worldbody.add_geom(
            type = mujoco.mjtGeom.mjGEOM_BOX,
            size = [length/2, width, 0.1],
            pos = [pos[0], pos[1], height-0.1],
            quat = [ 0, np.cos(angle_rad/2), 0, np.sin(angle_rad/2)],
        )

    def add_stairs(self, pos: list = [2, 0, 0], rise: float = 0.1, run: float = 0.1, width: float = 2.0, num_steps: int = 5):
            pos[2] += rise / 2
            for i in range(num_steps):
                step_name = f"stair_step_{i}"
                self.add_box(
                    name=step_name,
                    pos=[pos[0] + i * run, pos[1], pos[2] + i * rise],
                    size=[run / 2, width / 2, rise / 2],
                )
        
def main(argv=None):
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir,)))
    model_config_path = 'modeling/model_configs/2_7_Scale/model_config.yaml'
    motor_config_path = 'modeling/model_configs/2_7_Scale/motor_config.yaml'
    model_class = GenModel(model_config_path, motor_config_path)

    xml_path = os.path.join(
        os.path.dirname(__file__),
        "WaLTER_Model.xml",
    )

    with open(xml_path, "w") as f:
        f.writelines(model_class.model_xml)

    # Open mujoco viewer
    mujoco.viewer.launch(model_class.model_xml)


if __name__ == '__main__':
    app.run(main)