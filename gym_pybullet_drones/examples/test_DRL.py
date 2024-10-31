import os
import time
import random
import gymnasium as gym
import numpy as np
import torch
import pybullet as p
from gym_pybullet_drones.routing.RouteMission import RouteMission
print(f"CUDA is available : {torch.cuda.is_available()}")
# print(torch.cuda.get_device_name(0))
# from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.envs.AutoroutingRLAviary import AutoroutingRLAviary
from gym_pybullet_drones.envs.AutoroutingSARLAviary import AutoroutingSARLAviary
from gym_pybullet_drones.utils.utils import sync, str2bool
from gym_pybullet_drones.utils.enums import Physics, DroneModel
# from gym_pybullet_drones.drl_custom.networks.FCQ import FCQ
from gym_pybullet_drones.drl_custom.networks.FCDuelingQ import FCDuelingQ

DEFAULT_DRONES = DroneModel("cf2x")
DEFAULT_RECORD_VISION = False
DEFAULT_GUI = True
DEFAULT_RECORD_VIDEO = False
DEFAULT_PHYSICS = Physics("pyb")
DEFAULT_CONTROL_FREQ_HZ = 60
DEFAULT_SIMULATION_FREQ_HZ = 60
DEFAULT_AGENTS = 1
DEFAULT_SCENARIO = 1

MISSION = RouteMission()
MISSION.generateMission(numDrones=DEFAULT_AGENTS,scenario=DEFAULT_SCENARIO)
INIT_XYZS = MISSION.INIT_XYZS
INIT_RPYS = MISSION.INIT_RPYS
DESTINS = MISSION.DESTINS
# =================================================================

# TODO: automate file-reading to select proper model e.g. DQN->use FCQ, DuelingDQN->use FCDueling

# value_model_fn = lambda nS, nA: FCQ(nS, nA, hidden_dims=(512,128))
value_model_fn = lambda nS, nA: FCDuelingQ(nS, nA, hidden_dims=(512,128))

# TODO: automate reading the correct dimension of nS and nA
# numRays = 24
numRays = 24
numObserv = 13 + 5*numRays
numAct = 11
model = value_model_fn(numObserv, numAct)
fileName = "Komsun_DRL/"
# fileName += "Model-3actions-DuelingDDQN-10.17.2024_16.44.32.pth" # success
# fileName += "Model-DuelingDDQN-10.22.2024_17.52.46.pth"  # Good!!!!
# fileName += "Model-DuelingDDQN-10.25.2024_11.18.04.pth" # Very Good! (reduce num_rays to 10: nS=63, rayLen=1.25, reward_choice=8)
# fileName += "Model-DuelingDDQN-10.28.2024_17.44.21.pth"  # Very Good (num_rays = 24: nS=133, rayLen=1.5, reward_choice=9 (near hit not ok))
# fileName += "Model-DuelingDDQN-10.29.2024_17.50.05.pth"  # 11 actions (num_rays = 24: nS=133, rayLen=1.5, reward_choice=9 ())
fileName += "Model-DuelingDDQN-10.29.2024_18.33.13.pth"  # Smarter: very good (num_rays = 24: nS=133, rayLen=1.5, reward_choice=8)
# fileName += "Model-DuelingDDQN-10.30.2024_11.37.30.pth"  # Discourage local route: good (num_rays = 24: nS=133, rayLen=1.5, reward_choice=10)
# fileName += "Model-DuelingDDQN-10.30.2024_11.55.38.pth"  # My first Multi-agent (3 agents) (num_rays = 24: nS=133*3, rayLen=1.5, reward_choice=10)
# fileName += "Model-DuelingDDQN-10.30.2024_12.25.21.pth"   # Good 3 Agents result!!
# fileName += "Model-DuelingDDQN-10.31.2024_11.00.45.pth"  #3 agents (reward_choice = 10 (new)) (with 3xobservations)
# fileName += "Model-DuelingDDQN-10.30.2024_14.21.45.pth" # 5 Agents (with 5xobservations)
# fileName += "Model-DuelingDDQN-10.31.2024_11.11.20.pth" #10 Agents (reward_choice = 10 (new)) (with 10xobservations)
# fileName += "Model-DuelingDDQN-10.31.2024_17.17.14.pth"   #10 Agents 3 actions
# fileName += "Model-DuelingDDQN-10.31.2024_18.03.18.pth"   # N agents 3 actions
# CAUTION: If change number of actions -> need to also modify the action space in testing environment (AutoroutingRLAviary)!!!!
model.load_state_dict(torch.load(fileName,map_location=torch.device('cpu'), weights_only=True))
model.eval()
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# model = model.to(device)
model = model.to("cpu")
print("Model loaded and ready for inference!")
# input("Press Enter to continue . . .")


#========== Visualizing ========
# env = AutoroutingRLAviary(
#                 drone_model = DEFAULT_DRONES,
#                 initial_xyzs=INIT_XYZS,
#                 initial_rpys=INIT_RPYS,
#                 physics= DEFAULT_PHYSICS,
#                 pyb_freq=DEFAULT_SIMULATION_FREQ_HZ,
#                 ctrl_freq=DEFAULT_CONTROL_FREQ_HZ,
#                 gui=DEFAULT_GUI,
#                 record=DEFAULT_RECORD_VIDEO,
#                 num_drones=DEFAULT_AGENTS
#                 )
env = AutoroutingSARLAviary(
                drone_model = DEFAULT_DRONES,
                initial_xyzs=INIT_XYZS,
                initial_rpys=INIT_RPYS,
                physics= DEFAULT_PHYSICS,
                pyb_freq=DEFAULT_SIMULATION_FREQ_HZ,
                ctrl_freq=DEFAULT_CONTROL_FREQ_HZ,
                gui=DEFAULT_GUI,
                record=DEFAULT_RECORD_VIDEO,
                num_drones=DEFAULT_AGENTS
                )
for j in range(DEFAULT_AGENTS):
    env.routing[j].CUR_POS = INIT_XYZS[j,:]
    env.routing[j].HOME_POS = INIT_XYZS[j,:]
    env.routing[j].DESTINATION = DESTINS[j,:]

p.setRealTimeSimulation(1) 
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
camSwitch = 1
camSwitchFreq_step = 120
for _ in range(20):
    START = time.time()
    state, _ = env.reset()
    epEnd = False
    count = 0

    
    
    while not epEnd:
        # state = state[0:numObserv] # only for the first drone
        # Convert the state to tensor if needed
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        # Get the action using the trained model 
        with torch.no_grad():
            action = model(state_tensor).argmax(dim=1).item()
            print(f"action = {action}")

        #### Step the simulation ###################################
        # Take the action in the environment
        next_state, reward, done, truncated, info = env.step(action)
        state = next_state

        
        if camSwitch>0:
            p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=45, cameraPitch=-60, cameraTargetPosition=state[0:3])
        else:
            p.resetDebugVisualizerCamera(cameraDistance=0.5, cameraYaw=0, cameraPitch=-20, cameraTargetPosition=state[0:3])
        # p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=45, cameraPitch=-60, cameraTargetPosition=state[0:3])
        epEnd = done or truncated
        #### Printout ##############################################
        env.render()

        sync(count, START, env.CTRL_TIMESTEP)
        camSwitch = -1*camSwitch if count%camSwitchFreq_step == 0 else camSwitch
        count += 1

#### Close the environment #################################
env.close()
    
# agent.demo_last()

