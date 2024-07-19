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
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.envs.RoutingAviary import RoutingAviary
from gym_pybullet_drones.envs.VisionAviary import VisionAviary
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
    parser.add_argument('--drone',              default="cf2p",     type=DroneModel,    help='Drone model (default: CF2X)', metavar='', choices=DroneModel)
    parser.add_argument('--num_drones',         default=2,          type=int,           help='Number of drones (default: 3)', metavar='')
    parser.add_argument('--physics',            default="pyb",      type=Physics,       help='Physics updates (default: PYB)', metavar='', choices=Physics)
    parser.add_argument('--vision',             default=False,      type=str2bool,      help='Whether to use VisionAviary (default: False)', metavar='')
    parser.add_argument('--gui',                default=True,       type=str2bool,      help='Whether to use PyBullet GUI (default: True)', metavar='')
    parser.add_argument('--record_video',       default=False,      type=str2bool,      help='Whether to record a video (default: False)', metavar='')
    parser.add_argument('--plot',               default=True,       type=str2bool,      help='Whether to plot the simulation results (default: True)', metavar='')
    parser.add_argument('--user_debug_gui',     default=False,      type=str2bool,      help='Whether to add debug lines and parameters to the GUI (default: False)', metavar='')
    parser.add_argument('--aggregate',          default=True,       type=str2bool,      help='Whether to aggregate physics steps (default: True)', metavar='')
    parser.add_argument('--obstacles',          default=True,       type=str2bool,      help='Whether to add obstacles to the environment (default: True)', metavar='')
    parser.add_argument('--simulation_freq_hz', default=120,        type=int,           help='Simulation frequency in Hz (default: 240)', metavar='')
    parser.add_argument('--control_freq_hz',    default=48,         type=int,           help='Control frequency in Hz (default: 48)', metavar='')
    parser.add_argument('--duration_sec',       default=60,         type=int,           help='Duration of the simulation in seconds (default: 5)', metavar='')
    ARGS = parser.parse_args()

    #### Initialize the simulation #############################
    H = .2
    H_STEP = .05
    R = .3
    # size: Nx3
    INIT_XYZS = np.array([[((-1)**i)*(i*0.1)+0.5,-1*(i*0.05), 0.5+ 0.05*i ] for i in range(ARGS.num_drones)])
    # INIT_XYZS = np.array([[((-1)**i)*(i*0.1)+0.2,-1*(i*0.05), 0.05*i ] for i in range(ARGS.num_drones)])
    INIT_RPYS = np.array([[0, 0,  0] for i in range(ARGS.num_drones)])
    AGGR_PHY_STEPS = int(ARGS.simulation_freq_hz/ARGS.control_freq_hz) if ARGS.aggregate else 1


    #### Create the environment with or without video capture ##
    if ARGS.vision: 
        env = VisionAviary(drone_model=ARGS.drone,
                           num_drones=ARGS.num_drones,
                           initial_xyzs=INIT_XYZS,
                           initial_rpys=INIT_RPYS,
                           physics=ARGS.physics,
                           neighbourhood_radius=10,
                           freq=ARGS.simulation_freq_hz,
                           aggregate_phy_steps=AGGR_PHY_STEPS,
                           gui=ARGS.gui,
                           record=ARGS.record_video,
                           obstacles=ARGS.obstacles
                           )
    else: 
        env = RoutingAviary(drone_model=ARGS.drone,
                         num_drones=ARGS.num_drones,
                         initial_xyzs=INIT_XYZS,
                         initial_rpys=INIT_RPYS,
                         physics=ARGS.physics,
                         neighbourhood_radius=10,
                         freq=ARGS.simulation_freq_hz,
                         aggregate_phy_steps=AGGR_PHY_STEPS,
                         gui=ARGS.gui,
                         record=ARGS.record_video,
                         obstacles=ARGS.obstacles,
                         user_debug_gui=ARGS.user_debug_gui
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
    routing = [IFDSRoute(drone_model=ARGS.drone) for i in range(ARGS.num_drones)]

    #### Run the simulation ####################################
    CTRL_EVERY_N_STEPS = int(np.floor(env.SIM_FREQ/ARGS.control_freq_hz))
    action = {str(i): np.array([0,0,0,0]) for i in range(ARGS.num_drones)}
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

        #### Make it rain rubber ducks #############################
        # if i/env.SIM_FREQ>5 and i%10==0 and i/env.SIM_FREQ<10: p.loadURDF("duck_vhacd.urdf", [0+random.gauss(0, 0.3),-0.5+random.gauss(0, 0.3),3], p.getQuaternionFromEuler([random.randint(0,360),random.randint(0,360),random.randint(0,360)]), physicsClientId=PYB_CLIENT)

        #### Step the simulation ###################################
        obs, reward, done, info = env.step(action)
        
        #### Compute control at the desired frequency ##############
        if i%CTRL_EVERY_N_STEPS == 0:
            ctrlCounter+=1
            
            for j in range(ARGS.num_drones):
             
                #------- Compute route (waypoint) to follow ----------------
                foundPath, path = routing[j].computeRouteFromState(route_timestep=routing[j].route_counter, 
                                                      state = obs[str(j)]["state"], 
                                                      home_pos = np.array((0,0,0)), 
                                                      target_pos = np.array((((-1)**j)*(j*0.2), 10, 0.2)),
                                                      speed_limit = env.SPEED_LIMIT,
                                                      obstacle_data = env.OBSTACLE_DATA
                                                      )
                if foundPath>0:
                    routeCounter+=1
                    # env._plotRoute(path)

                if ctrlCounter == 1:
                    routing[j].setGlobalRoute(path)
                # elif ctrlCounter == 100:
                #     print("ALERT*****Switching to Local Path********")
                #     routing[j].SIM_MODE = 2
                # elif ctrlCounter == 350:
                #     print("ALERT======Switching to GLOBAL PATH=======")
                #     routing[j].ACTIVATE_GLOBAL_PATH = 1
                            
                
                NUM_WP = path.shape[1]
                
                # ---------- Manual logic to activate hovering mode ----------
                # if ctrlCounter >= 500 and ctrlCounter < 800:
                #     flagHover[j] = 1
                # else:
                #     flagHover[j] = 0
                # ------------------------------------------------------------
                # if ctrlCounter > 200:
                #     routing[j]._setCommand(RouteCommandFlag, "follow_global")
                # elif ctrlCounter > 10:
                #     routing[j]._setCommand(RouteCommandFlag, "follow_local")
                # if ctrlCounter==80 or ctrlCounter==400:
                #     routing[j]._setCommand(RouteCommandFlag, "change_route")
                    
                
                # ---------- Manual logic to accelerate/decelerate ----------
                # if ctrlCounter >= 100 and ctrlCounter < 300:
                #     routing[j]._setCommand(SpeedCommandFlag, "accelerate", -0.06) # [m/s^2]
                #     routing[0]._setCommand(SpeedCommandFlag, "hover")
                # else:
                #     routing[j]._setCommand(SpeedCommandFlag, "accelerate", 0.01)
                    
                # if ctrlCounter == 400 and j==1:
                #     routing[j]._setCommand(RouteCommandFlag, "follow_local")
                                  
                # if ctrlCounter == 400 and j==2:
                #     routing[j]._setCommand(RouteCommandFlag, "change_route")
                # ------------------------------------------------------------
                
                #### Compute control for the current way point #############
                action[str(j)], _, _ = ctrl[j].computeControlFromState(control_timestep=CTRL_EVERY_N_STEPS*env.TIMESTEP,
                                                                       state=obs[str(j)]["state"],
                                                                       target_pos = routing[j].TARGET_POS, 
                                                                       target_rpy=INIT_RPYS[j, :],
                                                                       target_vel = routing[j].TARGET_VEL
                                                                       )

        #### Log the simulation ####################################
        # for j in range(ARGS.num_drones):
        #     logger.log(drone=j,
        #                timestamp=i/env.SIM_FREQ,
        #                state= obs[str(j)]["state"],
        #                control=np.hstack([TARGET_POS[wp_counters[j], 0:2], INIT_XYZS[j, 2], INIT_RPYS[j, :], np.zeros(6)])
        #                # control=np.hstack([INIT_XYZS[j, :]+TARGET_POS[wp_counters[j], :], INIT_RPYS[j, :], np.zeros(6)])
        #                )

        #### Printout ##############################################
        if i%env.SIM_FREQ == 0:
            env.render()
            #### Print matrices with the images captured by each drone #
            if ARGS.vision:
                for j in range(ARGS.num_drones):
                    print(obs[str(j)]["rgb"].shape, np.average(obs[str(j)]["rgb"]),
                          obs[str(j)]["dep"].shape, np.average(obs[str(j)]["dep"]),
                          obs[str(j)]["seg"].shape, np.average(obs[str(j)]["seg"])
                          )

        #### Sync the simulation ###################################
        if ARGS.gui:
            sync(i, START, env.TIMESTEP)
            

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
