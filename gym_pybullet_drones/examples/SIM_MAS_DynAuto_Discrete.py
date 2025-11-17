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
import time
import argparse
import itertools
import numpy as np
import pybullet as p
import matplotlib.pyplot as plt
from ray.rllib.agents import qmix

from gym_pybullet_drones.envs.BaseAviary import DroneModel, Physics
from gym_pybullet_drones.envs.multi_agent_rl.AutoroutingMASAviary_discrete import AutoroutingMASAviary_discrete

from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.control.SimplePIDControl import SimplePIDControl
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync, str2bool
from gym_pybullet_drones.routing.IFDSRoute import IFDSRoute

def draw_circle_around_drone(center, radius=1.0, color=[0, 1, 0], segments=36, z_offset=0.05):
    """Draw a circle around a given center position (drone) in the XY plane."""
    circle_lines = []
    theta = np.linspace(0, 2 * np.pi, segments + 1)
    for i in range(segments):
        x1 = center[0] + radius * np.cos(theta[i])
        y1 = center[1] + radius * np.sin(theta[i])
        z1 = center[2] + z_offset

        x2 = center[0] + radius * np.cos(theta[i + 1])
        y2 = center[1] + radius * np.sin(theta[i + 1])
        z2 = center[2] + z_offset

        line_id = p.addUserDebugLine([x1, y1, z1], [x2, y2, z2], color, lineWidth=1)
        circle_lines.append(line_id)
    return circle_lines




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
    parser.add_argument('--drone',              default="hb",     type=DroneModel,    help='Drone model (default: CF2X)', metavar='', choices=DroneModel)
    parser.add_argument('--num_drones',         default=8,          type=int,           help='Number of drones (default: 3)', metavar='')
    parser.add_argument('--physics',            default="pyb",      type=Physics,       help='Physics updates (default: PYB)', metavar='', choices=Physics)
    parser.add_argument('--vision',             default=False,      type=str2bool,      help='Whether to use VisionAviary (default: False)', metavar='')
    parser.add_argument('--gui',                default=True,       type=str2bool,      help='Whether to use PyBullet GUI (default: True)', metavar='')
    parser.add_argument('--record_video',       default=False,      type=str2bool,      help='Whether to record a video (default: False)', metavar='')
    parser.add_argument('--plot',               default=True,       type=str2bool,      help='Whether to plot the simulation results (default: True)', metavar='')
    parser.add_argument('--aggregate',          default=True,       type=str2bool,      help='Whether to aggregate physics steps (default: True)', metavar='')
    parser.add_argument('--obstacles',          default=True,       type=str2bool,      help='Whether to add obstacles to the environment (default: True)', metavar='')
    parser.add_argument('--simulation_freq_hz', default=60,        type=int,           help='Simulation frequency in Hz (default: 240)', metavar='')
    parser.add_argument('--control_freq_hz',    default=60,         type=int,           help='Control frequency in Hz (default: 48)', metavar='')
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

    NUM_DRONES = ARGS.num_drones

    config = qmix.DEFAULT_CONFIG.copy() # For the default config, see github.com/ray-project/ray/blob/master/rllib/agents/trainer.py
    #### Create the environment with or without video capture ##
    
    env = AutoroutingMASAviary_discrete(drone_model=ARGS.drone,
                        num_drones=ARGS.num_drones,
                        physics=ARGS.physics,
                        freq=10,
                        aggregate_phy_steps=1,
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
   
    
    ctrlCounter = 0
    routeCounter = 1
        
    #++++ Initialize Routing +++++++++++++++++++++++++++++++++++
    routing = [IFDSRoute(drone_model=ARGS.drone, drone_id=i) for i in range(ARGS.num_drones)]

    #### Run the simulation ####################################
    
    START = time.time()
    
    # Starts with the 2nd waypoint in the route
    wp_counters = 40*np.ones(ARGS.num_drones, dtype=int)
    k = -1*np.ones(ARGS.num_drones, dtype=int)
    TARGET_WP = [np.array([]) for _ in range(ARGS.num_drones)]
    flagHover = [0 for _ in range(ARGS.num_drones)]
    hoverPos = [np.array([]) for _ in range(ARGS.num_drones)]
    targetVel = [np.array([]) for _ in range(ARGS.num_drones)]
    curSpeed = [0 for _ in range(ARGS.num_drones)]

    num_ep_test = 20
    
    # Define mapping from action index → label
    ACTION_LABELS = {
        0: "Accelerate",
        1: "Decelerate",
        2: "Constant Velocity"
    }
    for i in range(0, num_ep_test):
        epEnd = False
        env.reset()
        # print(f"...... Initial RPYS is {env.INIT_RPYS}")
        SAFE_DISTANCE = 5.0  # meters
        debug_items = []     # store current debug visuals
        min_dists = []       # store history if you want to plot later

        # Ground fixed camera
        p.resetDebugVisualizerCamera(cameraDistance=35, cameraYaw=0, cameraPitch=-60, cameraTargetPosition=[0,0,0])
        count = 0
        while not epEnd:
            #### Step the simulation ###################################
            count+=1
            if count >= 4*env.SIM_FREQ and count < 10*env.SIM_FREQ:
                act = 1
                # print(f"<<<< braking")
            else:
                act = 0
            action = {drone_idx: act for drone_idx in range(ARGS.num_drones)}
            obs, reward, done, info = env.step(action)
            # print(f"reward is {reward}")
            # === Compute and visualize inter-drone distances ===
            # Clear previous debug items
            p.removeAllUserDebugItems
            for item in debug_items:
                p.removeUserDebugItem(item)
            debug_items = []
            min_dists_all = []   # optional history logging 

            # Get all drone positions
            positions = [p.getBasePositionAndOrientation(env.DRONE_IDS[i])[0]
                        for i in range(ARGS.num_drones)]
            
            # === Compute pairwise distances ===
            dist_matrix = np.full((NUM_DRONES, NUM_DRONES), np.inf)
            for i, j in itertools.combinations(range(NUM_DRONES), 2):
                dist = np.linalg.norm(np.array(positions[i]) - np.array(positions[j]))  # works now ✅
                dist_matrix[i, j] = dist_matrix[j, i] = dist

                # Draw color-coded line
                color = [1, 0, 0] if dist < SAFE_DISTANCE else [0, 1, 0]
                # debug_items.append(p.addUserDebugLine(positions[i], positions[j], color, lineWidth=2))

            # === Compute and display per-drone min separation ===
            per_drone_min = np.min(dist_matrix, axis=1)
            min_dists_all.append(per_drone_min)

            for i in range(NUM_DRONES):
                min_d = per_drone_min[i]
                text_color = [1, 1, 0] if min_d < SAFE_DISTANCE else [0, 1, 0]
                debug_items.append(
                    p.addUserDebugText(
                        f"{min_d:.2f} m",
                        [0, 0, 0.2],  # small offset above the drone
                        textColorRGB=text_color,
                        textSize=1.2,
                        parentObjectUniqueId=env.DRONE_IDS[i]
                    )
                )
                # Display Action took by agents
                # act = action[i]
                # action_label = ACTION_LABELS.get(act, f"Action {act}")
                # debug_items.append(
                #     p.addUserDebugText(
                #         f"{action_label}",
                #         [0, 0, 2.5],  # stacked above the distance text
                #         textColorRGB=[0.2, 0.8, 1.0],  # light blue
                #         textSize=1.1,
                #         parentObjectUniqueId=env.DRONE_IDS[i]
                #     )
                # )


             # Draw a circle around each drone
                circle_color = [1, 1, 0] if min_d < SAFE_DISTANCE else [0, 1, 0]
                inner_color = [1, 0, 0] if min_d < SAFE_DISTANCE else [0, 1, 0]
                circle_ids = draw_circle_around_drone(
                    center=positions[i],
                    radius=SAFE_DISTANCE,
                    color=circle_color,
                    segments=36,
                    z_offset=0.05
                )
                circle_inner_ids = draw_circle_around_drone(
                    center=positions[i],
                    radius=SAFE_DISTANCE-2,
                    color=[0, 1, 0],
                    segments=36,
                    z_offset=0.05
                )
                debug_items.extend(circle_ids)
                debug_items.extend(circle_inner_ids)
            # === Compute global minimum separation ===
            global_min_dist = np.min(dist_matrix)

            # === Display global stats at fixed location ===
            # For example, top-left of the scene: x=-5, y=-5, z=5
            debug_items.append(
                p.addUserDebugText(
                    f"Global min separation: {global_min_dist:.2f} m",
                    [-35, 25, 5],
                    textColorRGB=[1, 1, 1],  # white text
                    textSize=1.5
                )
            )
            debug_items.append(
                p.addUserDebugText(
                    f"Timestep: {count}",
                    [-35, 20, 5],  # slightly below the first text
                    textColorRGB=[1, 1, 0],  # yellow
                    textSize=1.5
                )
            )

            # print(f">>>>>>>>>>> done = {done}")
            # print(f"sector info is {env.routing[0].SECTOR_INFO}")
            # print(f"step #{env.step_counter}")
            if done['__all__']==True:
                print(f"========== EPISODE ENDED ==============")
                epEnd = True

            env.render()
            # Drone #0 tracking cameara
            # p.resetDebugVisualizerCamera(cameraDistance=15, cameraYaw=env.routing[0].CUR_RPY[2]*180/np.pi - 90, cameraPitch=-60, cameraTargetPosition=env.routing[0].CUR_POS)
            
            #### Sync the simulation ###################################
            # if ARGS.gui:
            # sync(i, START, env.TIMESTEP)
            

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
