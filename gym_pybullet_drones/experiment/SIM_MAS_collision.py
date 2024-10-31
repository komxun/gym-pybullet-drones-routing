"""Script demonstrating the joint use of simulation and control.

The simulation is run by a `CtrlAviary` environment.
The control is given by the PID implementation in `DSLPIDControl`.

Example
-------
In a terminal, run as:

    $ python pid.py

Notes
-----
The drones move, at different altitudes, along cicular trajectories 
in the X-Y plane, around point (0, -.3).

"""
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

from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.envs.RoutingAviary import RoutingAviary
from gym_pybullet_drones.envs.AutoroutingRLAviary import AutoroutingRLAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync, str2bool
from gym_pybullet_drones.routing.BaseRouting import RouteCommandFlag, SpeedCommandFlag, SpeedStatus
from gym_pybullet_drones.routing.IFDSRoute import IFDSRoute
from gym_pybullet_drones.routing.RouteMission import RouteMission

DEFAULT_DRONES = DroneModel("cf2x")
DEFAULT_PHYSICS = Physics("pyb")
DEFAULT_GUI = True
DEFAULT_RECORD_VISION = False
DEFAULT_PLOT = False
DEFAULT_USER_DEBUG_GUI = False
DEFAULT_OBSTACLES = True
DEFAULT_SIMULATION_FREQ_HZ = 60
DEFAULT_CONTROL_FREQ_HZ = 60
DEFAULT_DURATION_SEC = 90
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_COLAB = False
DEFAULT_NUM_DRONES = 20
DEFAULT_SCENARIO = 2

MISSION = RouteMission()
MISSION.generateMission(numDrones=DEFAULT_NUM_DRONES,scenario=DEFAULT_SCENARIO)


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
    INIT_XYZS = MISSION.INIT_XYZS
    INIT_RPYS = MISSION.INIT_RPYS
    DESTINS = MISSION.DESTINS
    #### Create the environment ################################
    
    env = RoutingAviary(drone_model=drone,
                         num_drones=num_drones,
                         initial_xyzs=INIT_XYZS,
                         initial_rpys=INIT_RPYS,
                         physics=physics,
                         neighbourhood_radius=10,
                         pyb_freq=simulation_freq_hz,
                         ctrl_freq=control_freq_hz,
                         gui=gui,
                         record=record_video,
                         obstacles=obstacles,
                         user_debug_gui=user_debug_gui
                         )
    

    #### Obtain the PyBullet Client ID from the environment ####
    PYB_CLIENT = env.getPyBulletClient()

    #### Initialize the logger #################################
    # logger = Logger(logging_freq_hz=control_freq_hz,
    #                 num_drones=num_drones,
    #                 output_folder=output_folder,
    #                 colab=colab
    #                 )

    #### Initialize the controllers ############################
    if drone in [DroneModel.CF2X, DroneModel.CF2P]:
        ctrl = [DSLPIDControl(drone_model=drone) for i in range(num_drones)]
        
    #++++ Initialize Routing +++++++++++++++++++++++++++++++++++
    routing = [IFDSRoute(drone_model=drone, drone_id=i) for i in range(num_drones)]
    routeCounter = 1
    for j in range(num_drones):
        routing[j].CUR_POS = INIT_XYZS[j,:]
        routing[j].HOME_POS = INIT_XYZS[j,:]
        routing[j].DESTINATION = [10,0,0]
        routing[j].DESTINATION = DESTINS[j,:]


    camSwitch = 1
    camSwitchFreq_step = 60

    #### Run the simulation ####################################
    action = np.zeros((num_drones,4))
    START = time.time()
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    routeCounter = np.ones(num_drones)
    for i in range(0, int(duration_sec*env.CTRL_FREQ)):
        camSwitch = -1*camSwitch if i%camSwitchFreq_step == 0 else camSwitch
        #### Make it rain rubber ducks #############################
        # if i/env.SIM_FREQ>5 and i%10==0 and i/env.SIM_FREQ<10: p.loadURDF("duck_vhacd.urdf", [0+random.gauss(0, 0.3),-0.5+random.gauss(0, 0.3),3], p.getQuaternionFromEuler([random.randint(0,360),random.randint(0,360),random.randint(0,360)]), physicsClientId=PYB_CLIENT)
        #### Step the simulation ###################################
        obs, reward, terminated, truncated, info = env.step(action)

        #### Compute control for the current way point #############
        for j in range(num_drones):
            
            if routing[j].REACH_DESTIN:
                routing[j].reset()
                tempDestin = routing[j].DESTINATION
                tempHome = routing[j].HOME_POS
                routing[j].DESTINATION = tempHome
                routing[j].HOME_POS = tempDestin
                routeCounter[j] = 1
                
            #------- Compute route (waypoint) to follow ----------------
            foundPath, path = routing[j].computeRouteFromState(route_timestep=routing[j].route_counter, 
                                                  state = obs[j], 
                                                  home_pos = routing[j].HOME_POS, 
                                                  target_pos = routing[j].DESTINATION,
                                                  speed_limit = env.SPEED_LIMIT,
                                                  obstacle_data = env.OBSTACLE_DATA,
                                                  drone_ids = env.DRONE_IDS
                                                  )
        
            if foundPath>0:
                routeCounter[j]+=1
                if routeCounter[j]>=2 :
                    print("****************Re-calculating destination")
                    routing[j].setGlobalRoute(path)
            
            if j==0:
                routing[j]._setCommand(SpeedCommandFlag, "accelerate",2)
                routing[j]._setCommand(RouteCommandFlag, "follow_local")
            else:
                routing[j]._setCommand(SpeedCommandFlag, "accelerate", random.random())
                # routing[j]._setCommand(SpeedCommandFlag, "accelerate",2)
                routing[j]._setCommand(RouteCommandFlag, "follow_global")
            
            action[j, :], _, _ = ctrl[j].computeControlFromState(control_timestep=env.CTRL_TIMESTEP,
                                                                    state=obs[j],
                                                                    target_pos=routing[j].TARGET_POS,
                                                                    target_rpy=INIT_RPYS[j, :],
                                                                    target_vel = routing[j].TARGET_VEL
                                                                    )
            
            # p.resetDebugVisualizerCamera(cameraDistance=5, cameraYaw=0, cameraPitch=-92, cameraTargetPosition=routing[0].CUR_POS)
            # p.resetDebugVisualizerCamera(cameraDistance=0.5, cameraYaw=obs[0][5], cameraPitch=-20, cameraTargetPosition=routing[0].CUR_POS)
            # if camSwitch>0:
            #     p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=45, cameraPitch=-60, cameraTargetPosition=routing[0].CUR_POS)
            # else:
            #     # p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=0, cameraPitch=-20, cameraTargetPosition=routing[0].CUR_POS)
            #     p.resetDebugVisualizerCamera(cameraDistance=2.5, cameraYaw=0, cameraPitch=-91, cameraTargetPosition=routing[0].CUR_POS)
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

    #### Close the environment #################################
    env.close()

    #### Save the simulation results ###########################
    # logger.save()
    # logger.save_as_csv("pid") # Optional CSV save

    # #### Plot the simulation results ###########################
    # if plot:
    #     logger.plot()

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
