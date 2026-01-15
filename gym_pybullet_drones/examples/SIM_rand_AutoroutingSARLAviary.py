"""Script demonstrating the joint use of simulation and control.

The simulation is run by a `CtrlAviary` environment.
The control is given by the PID implementation in `DSLPIDControl`.

Example
-------
In a terminal, run as:

    $ python pid.py

Notes
-----g
The drones move, at different altitudes, along cicular trajectories 
in the X-Y plane, around point (0, -.3).

"""
import os
import time
import argparse
import random
import numpy as np
import pybullet as p
import matplotlib.pyplot as plt
import itertools
import random


from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.AutoroutingRLAviary import AutoroutingRLAviary
from gym_pybullet_drones.envs.AutoroutingSARLAviary import AutoroutingSARLAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync, str2bool

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

DEFAULT_DRONES = DroneModel("hb")
DEFAULT_NUM_DRONES = 5
DEFAULT_PHYSICS = Physics("pyb")
DEFAULT_GUI = True
DEFAULT_RECORD_VISION = False
DEFAULT_PLOT = False
DEFAULT_USER_DEBUG_GUI = False
DEFAULT_OBSTACLES = True
DEFAULT_SIMULATION_FREQ_HZ = 60
DEFAULT_CONTROL_FREQ_HZ = 60
DEFAULT_DURATION_SEC = 20
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_COLAB = False

def run(
        drone=DEFAULT_DRONES,
        num_drones=DEFAULT_NUM_DRONES,
        physics=DEFAULT_PHYSICS,
        gui=DEFAULT_GUI,
        record_video=DEFAULT_RECORD_VISION,
        plot=DEFAULT_PLOT,
        user_debug_gui=DEFAULT_USER_DEBUG_GUI,
        obstacles=DEFAULT_OBSTACLES,
        simulation_freq_hz=DEFAULT_SIMULATION_FREQ_HZ,
        control_freq_hz=DEFAULT_CONTROL_FREQ_HZ,
        duration_sec=DEFAULT_DURATION_SEC,
        output_folder=DEFAULT_OUTPUT_FOLDER,
        colab=DEFAULT_COLAB
        ):
    #### Initialize the simulation #############################
    H = .1
    H_STEP = .05
    R = .3

    #### Create the environment ################################
    env = AutoroutingSARLAviary(
                 drone_model = drone,
                 num_drones = DEFAULT_NUM_DRONES, 
                 physics= physics,
                 pyb_freq=simulation_freq_hz,
                 ctrl_freq=control_freq_hz,
                 gui=gui,
                 record=record_video,
                 )

    #### Obtain the PyBullet Client ID from the environment ####
    PYB_CLIENT = env.getPyBulletClient()
    NUM_DRONES = ARGS.num_drones

    #### Initialize the logger #################################
    # logger = Logger(logging_freq_hz=control_freq_hz,
    #                 num_drones=num_drones,
    #                 output_folder=output_folder,
    #                 colab=colab
    #                 )

    #### Run the simulation ####################################
    # action = np.zeros((num_drones,4))
    
    for _ in range(20):
        epEnd = False
        count = 0
        START = time.time()
        env.reset()
        SAFE_DISTANCE = 5.0  # meters
        debug_items = []     # store current debug visuals
        min_dists = []       # store history if you want to plot later
        # Ground fixed camera
        # p.resetDebugVisualizerCamera(cameraDistance=35, cameraYaw=0, cameraPitch=-60, cameraTargetPosition=[0,0,0])
        while not epEnd:
            count += 1
            # ======Random Action!!=========
            # action = random.randint(0, 2)
            if count >= 4*simulation_freq_hz and count < 10*simulation_freq_hz:
                action = 1
                # print(f"<<<< braking")
            else:
                action = 0

            #### Step the simulation ###################################
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                print(f"========== EPISODE ENDED ==============")
                epEnd = True
            # Clear previous debug items
            p.removeAllUserDebugItems
            for item in debug_items:
                p.removeUserDebugItem(item)
            debug_items = []
            min_dists_all = []   # optional history logging 
            # Drone's 1 following camera
            p.resetDebugVisualizerCamera(cameraDistance=35, cameraYaw=0, cameraPitch=-60, cameraTargetPosition=env.routing[0].CUR_POS)

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

            for i in range(1):
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
            
            #### Log the simulation ####################################
            # for j in range(num_drones):
            #     logger.log(drone=j,
            #                timestamp=i/env.CTRL_FREQ,
            #                state=obs[j],
            #                control=np.hstack([routing[j].TARGET_POS, INIT_XYZS[j, 2], INIT_RPYS[j, :], np.zeros(6)])
            #                # control=np.hstack([INIT_XYZS[j, :]+TARGET_POS[wp_counters[j], :], INIT_RPYS[j, :], np.zeros(6)])
            #                )

            #### Printout ##############################################
            env.render()

            #### Sync the simulation ###################################
            if gui:
                sync(i, START, env.CTRL_TIMESTEP)
            i+=1

    #### Close the environment #################################
    env.close()

    #### Save the simulation results ###########################
    # logger.save()
    # logger.save_as_csv("pid") # Optional CSV save

    #### Plot the simulation results ###########################
    if plot:
        logger.plot()

if __name__ == "__main__":
    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='Helix flight script using CtrlAviary and DSLPIDControl')
    parser.add_argument('--drone',              default=DEFAULT_DRONES,     type=DroneModel,    help='Drone model (default: CF2X)', metavar='', choices=DroneModel)
    parser.add_argument('--num_drones',         default=DEFAULT_NUM_DRONES,          type=int,           help='Number of drones (default: 3)', metavar='')
    parser.add_argument('--physics',            default=DEFAULT_PHYSICS,      type=Physics,       help='Physics updates (default: PYB)', metavar='', choices=Physics)
    parser.add_argument('--gui',                default=DEFAULT_GUI,       type=str2bool,      help='Whether to use PyBullet GUI (default: True)', metavar='')
    parser.add_argument('--record_video',       default=DEFAULT_RECORD_VISION,      type=str2bool,      help='Whether to record a video (default: False)', metavar='')
    parser.add_argument('--plot',               default=DEFAULT_PLOT,       type=str2bool,      help='Whether to plot the simulation results (default: True)', metavar='')
    parser.add_argument('--user_debug_gui',     default=DEFAULT_USER_DEBUG_GUI,      type=str2bool,      help='Whether to add debug lines and parameters to the GUI (default: False)', metavar='')
    parser.add_argument('--obstacles',          default=DEFAULT_OBSTACLES,       type=str2bool,      help='Whether to add obstacles to the environment (default: True)', metavar='')
    parser.add_argument('--simulation_freq_hz', default=DEFAULT_SIMULATION_FREQ_HZ,        type=int,           help='Simulation frequency in Hz (default: 240)', metavar='')
    parser.add_argument('--control_freq_hz',    default=DEFAULT_CONTROL_FREQ_HZ,         type=int,           help='Control frequency in Hz (default: 48)', metavar='')
    parser.add_argument('--duration_sec',       default=DEFAULT_DURATION_SEC,         type=int,           help='Duration of the simulation in seconds (default: 5)', metavar='')
    parser.add_argument('--output_folder',     default=DEFAULT_OUTPUT_FOLDER, type=str,           help='Folder where to save logs (default: "results")', metavar='')
    parser.add_argument('--colab',              default=DEFAULT_COLAB, type=bool,           help='Whether example is being run by a notebook (default: "False")', metavar='')
    ARGS = parser.parse_args()

    run(**vars(ARGS))
