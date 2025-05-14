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
import random


from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.envs.RoutingAviary import RoutingAviary
from gym_pybullet_drones.envs.AutoroutingRLAviary import AutoroutingRLAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync, str2bool
from gym_pybullet_drones.routing.BaseRouting import RouteCommandFlag, SpeedCommandFlag, SpeedStatus
from gym_pybullet_drones.routing.IFDSRoute import IFDSRoute

DEFAULT_DRONES = DroneModel("cf2x")
DEFAULT_NUM_DRONES = 1
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
    # INIT_XYZS = np.array([[R*np.cos((i/6)*2*np.pi+np.pi/2), R*np.sin((i/6)*2*np.pi+np.pi/2)-R, H+i*H_STEP] for i in range(num_drones)])
    INIT_XYZS = np.array([[((-1)**i)*(i*0.2)+0.5,-3*(i*0.05), 1.3+ 0.05*i ] for i in range(ARGS.num_drones)])
    INIT_RPYS = np.array([[0, 0,  i * (np.pi/2)/num_drones] for i in range(num_drones)])

    # Load the Model
    value_model_fn = lambda nS, nA: FCDuelingQ(nS, nA, hidden_dims=(512,128))

    # TODO: automate reading the correct dimension of nS and nA
    numObserv = 1013
    numAct = 5
    model = value_model_fn(numObserv, numAct)
    fileName = "Komsun_DRL/"
    # fileName += "Model-3actions-DuelingDDQN-10.17.2024_16.44.32.pth" # success
    # fileName += "Model-5actions-DuelingDDQN-10.17.2024_17.21.36.pth"
    # fileName += "Model-DuelingDDQN-10.18.2024_16.14.20.pth" # train for 1 hour -> still bad (wtf?)
    # fileName += "Model-DuelingDDQN-10.18.2024_16.45.31.pth"
    # fileName += "Model-PER-10.18.2024_17.35.31.pth" # partly work but weird
    # fileName += "Model-DuelingDDQN-10.21.2024_11.07.26.pth"
    # fileName += "Model-PER-10.21.2024_15.35.27.pth" # Good
    # fileName += "Model-PER-10.21.2024_16.48.43.pth" # 60 minutes 1124 episodes
    # fileName += "Model-PER-10.22.2024_16.28.16.pth" # ok result
    fileName += "Model-DuelingDDQN-10.22.2024_17.52.46.pth"  # Good!!!!
    # fileName += "Model-PER-10.23.2024_12.47.15.pth" # PER 40 min (success sometime)
    # fileName += "Model-DuelingDDQN-10.24.2024_15.34.32.pth"  # 90 minutes (stay hovering at the start (why?))
    # CAUTION: If change number of actions -> need to also modify the action space in testing environment (AutoroutingRLAviary)!!!!
    model.load_state_dict(torch.load(fileName,map_location=torch.device('cpu'), weights_only=True))
    model.eval()
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # model = model.to(device)
    model = model.to("cpu")
    print("Model loaded and ready for inference!")

    #### Create the environment ################################
    
    env = AutoroutingRLAviary(
                 drone_model = drone,
                 initial_xyzs=INIT_XYZS,
                 initial_rpys=INIT_RPYS,
                 physics= physics,
                 pyb_freq=simulation_freq_hz,
                 ctrl_freq=control_freq_hz,
                 gui=gui,
                 record=record_video,
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

    #### Run the simulation ####################################
    # action = np.zeros((num_drones,4))
    START = time.time()
    
    for i in range(0, int(duration_sec*env.CTRL_FREQ)):

        #### Make it rain rubber ducks #############################
        # if i/env.SIM_FREQ>5 and i%10==0 and i/env.SIM_FREQ<10: p.loadURDF("duck_vhacd.urdf", [0+random.gauss(0, 0.3),-0.5+random.gauss(0, 0.3),3], p.getQuaternionFromEuler([random.randint(0,360),random.randint(0,360),random.randint(0,360)]), physicsClientId=PYB_CLIENT)
        # ======Random Action!!=========
        action = random.randint(0, 3)
        # if i<80:
        #     action = 2
        # else:
        #     print(f"\nDecelerating!!!!!\n")
        #     action = 1
        # action = 0
        # action = 1

        #### Step the simulation ###################################
        obs, reward, terminated, truncated, info = env.step(action)
        # print(f"obs len = {len(obs[0])}")
        print(f"truncated = {truncated}")
        
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

    #### Plot the simulation results ###########################
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
