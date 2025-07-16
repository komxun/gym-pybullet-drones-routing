# -*- coding: utf-8 -*-
"""Script demonstrating the joint use of simulation and control.
The simulation is run by a `RoutingAviary` and `CtrlAviary` or `VisionAviary` environment.
The path is calculated by the IFDS algorithm in `IFDSRoute`
The control is given by the PID implementation in `DSLPIDControl`.

v2.4
- Each drone has its own IFDS path 
- Each drones can switch between global and local path
- Waypoint Skipping logic is implemented (using PID to control drones)
- Hovering activation (manual) is implemented

Next version
------------
- Accelerate and Decelerate control

Example
-------
In a terminal, run as:

    $ python fly.py

"""

import sys
sys.path.append('../../')   # Locate gym_pybullet_drones directory
import os
import time
import argparse
from datetime import datetime
import pdb
import math
import random
import numpy as np
import pybullet as p
import matplotlib.pyplot as plt

from gym_pybullet_drones.envs.BaseAviary import DroneModel, Physics
from gym_pybullet_drones.envs.RoutingAviary import RoutingAviary
from gym_pybullet_drones.envs.multi_agent_rl.AutoroutingMASAviary_discrete import AutoroutingMASAviary_discrete

from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.control.SimplePIDControl import SimplePIDControl
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync, str2bool

from gym_pybullet_drones.routing.BaseRouting import RouteCommandFlag, SpeedCommandFlag
from gym_pybullet_drones.routing.IFDSRoute import IFDSRoute



def adjust_target_vel(cur_vel, target_vel, max_acceleration):
    # Calculate the change in velocity required
    delta_vel = target_vel - cur_vel
    
    # Limit the change to the maximum acceleration
    delta_vel = np.clip(delta_vel, -max_acceleration, max_acceleration)
    
    # Adjust the target velocity
    new_target_vel = cur_vel + delta_vel
    return new_target_vel

if __name__ == "__main__":

    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='Helix flight script using CtrlAviary or VisionAviary and DSLPIDControl')
    parser.add_argument('--drone',              default="cf2x",     type=DroneModel,    help='Drone model (default: CF2X)', metavar='', choices=DroneModel)
    parser.add_argument('--num_drones',         default=5,          type=int,           help='Number of drones (default: 3)', metavar='')
    parser.add_argument('--physics',            default="pyb",      type=Physics,       help='Physics updates (default: PYB)', metavar='', choices=Physics)
    parser.add_argument('--vision',             default=False,      type=str2bool,      help='Whether to use VisionAviary (default: False)', metavar='')
    parser.add_argument('--gui',                default=True,       type=str2bool,      help='Whether to use PyBullet GUI (default: True)', metavar='')
    parser.add_argument('--record_video',       default=False,      type=str2bool,      help='Whether to record a video (default: False)', metavar='')
    parser.add_argument('--plot',               default=True,       type=str2bool,      help='Whether to plot the simulation results (default: True)', metavar='')
    parser.add_argument('--aggregate',          default=True,       type=str2bool,      help='Whether to aggregate physics steps (default: True)', metavar='')
    parser.add_argument('--obstacles',          default=True,       type=str2bool,      help='Whether to add obstacles to the environment (default: True)', metavar='')
    parser.add_argument('--simulation_freq_hz', default=240,        type=int,           help='Simulation frequency in Hz (default: 240)', metavar='')
    parser.add_argument('--control_freq_hz',    default=120,         type=int,           help='Control frequency in Hz (default: 48)', metavar='')
    parser.add_argument('--duration_sec',       default=20,         type=int,           help='Duration of the simulation in seconds (default: 5)', metavar='')
    ARGS = parser.parse_args()

    #### Initialize the simulation #############################
    H = .2
    H_STEP = .05
    R = .3
    # size: Nx3
    # INIT_XYZS = np.array([[((-1.4)**i)*(i*0.1)+0.5,-1*(i*0.05), 0.5+ 0.05*i ] for i in range(ARGS.num_drones)])
    # INIT_XYZS = np.array([[((-1)**i)*(i*0.1)+0.2,-1*(i*0.05), 0.05*i ] for i in range(ARGS.num_drones)])
    # INIT_RPYS = np.array([[0, 0,  0] for i in range(ARGS.num_drones)])
    AGGR_PHY_STEPS = int(ARGS.simulation_freq_hz/ARGS.control_freq_hz) if ARGS.aggregate else 1


    #### Create the environment with or without video capture ##
    
    env = AutoroutingMASAviary_discrete(drone_model=ARGS.drone,
                        num_drones=ARGS.num_drones,
                        physics=ARGS.physics,
                        freq=ARGS.simulation_freq_hz,
                        aggregate_phy_steps=AGGR_PHY_STEPS,
                        gui=ARGS.gui,
                        record=ARGS.record_video,
                        )

    #### Obtain the PyBullet Client ID from the environment ####
    PYB_CLIENT = env.getPyBulletClient()

    #### Initialize the logger #################################
    logger = Logger(logging_freq_hz=int(ARGS.simulation_freq_hz/AGGR_PHY_STEPS),
                    num_drones=ARGS.num_drones
                    )

    #### Initialize the controllers ############################
    if ARGS.drone in [DroneModel.CF2X, DroneModel.CF2P]:
        ctrl = [DSLPIDControl(drone_model=ARGS.drone) for i in range(ARGS.num_drones)]
    elif ARGS.drone in [DroneModel.HB]:
        ctrl = [SimplePIDControl(drone_model=ARGS.drone) for i in range(ARGS.num_drones)]
    
    ctrlCounter = 0
    routeCounter = 1
        
    #++++ Initialize Routing +++++++++++++++++++++++++++++++++++
    routing = [IFDSRoute(drone_model=ARGS.drone, drone_id=i) for i in range(ARGS.num_drones)]

    #### Run the simulation ####################################
    CTRL_EVERY_N_STEPS = int(np.floor(env.SIM_FREQ/ARGS.control_freq_hz))
    
    START = time.time()
    
    # Starts with the 2nd waypoint in the route
    wp_counters = 40*np.ones(ARGS.num_drones, dtype=int)
    k = -1*np.ones(ARGS.num_drones, dtype=int)
    TARGET_WP = [np.array([]) for _ in range(ARGS.num_drones)]
    flagHover = [0 for _ in range(ARGS.num_drones)]
    hoverPos = [np.array([]) for _ in range(ARGS.num_drones)]
    targetVel = [np.array([]) for _ in range(ARGS.num_drones)]
    curSpeed = [0 for _ in range(ARGS.num_drones)]
    for i in range(0, int(ARGS.duration_sec*env.SIM_FREQ), AGGR_PHY_STEPS):
        epEnd = False
        env.reset()
        print(f"...... Initial RPYS is {env.INIT_RPYS}")

        while not epEnd:
            #### Step the simulation ###################################
            action = {drone_idx: 0 for drone_idx in range(ARGS.num_drones)}
            obs, reward, done, info = env.step(action)
            # print(f">>>>>>>>>>> done = {done}")
            if done['__all__']==True:
                print(f"========== EPISODE ENDED ==============")
                epEnd = True

            env.render()
            #### Sync the simulation ###################################
            # if ARGS.gui:
            #     sync(i, START, env.TIMESTEP)
            

    #### Close the environment #################################
    env.close()

    #### Save the simulation results ###########################
    logger.save()
    logger.save_as_csv("pid") # Optional CSV save

    #### Plot the simulation results ###########################
    if ARGS.plot:
        logger.plot()


#%% Plot Global Routes

# Plot initialization
ax = plt.figure().add_subplot(projection='3d')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Plot the trajectory
for j in range(len(routing)):
    ax.plot(routing[j].GLOBAL_PATH[0, :], routing[j].GLOBAL_PATH[1, :], routing[j].GLOBAL_PATH[2, :], label='Global Path agent '+str(j+1))

ax.legend()
ax.set_aspect('equal', adjustable='box')
plt.ion
plt.rcParams.update({'font.size': 12})
# plt.axis([0, 200, -100, 100, 0, 100])
# ax.set_zlim(0, 100)
# ax.set_xlim(0, 200)
# ax.set_ylim(-100, 100)
plt.show()

#%%

# Plot initialization
ax = plt.figure().add_subplot(projection='3d')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Plot the trajectory
for j in range(len(routing)):
    ax.plot(path[0, :], path[1, :], path[2, :], label='Global Path agent '+str(j+1))

ax.legend()
ax.set_aspect('equal', adjustable='box')
plt.ion
plt.rcParams.update({'font.size': 12})
# plt.axis([0, 200, -100, 100, 0, 100])
# ax.set_zlim(0, 100)
# ax.set_xlim(0, 200)
# ax.set_ylim(-100, 100)
plt.show()
