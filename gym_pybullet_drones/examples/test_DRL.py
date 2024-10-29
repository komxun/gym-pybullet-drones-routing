import os
import time
import random
import gymnasium as gym
import numpy as np
import torch
import pybullet as p
print(f"CUDA is available : {torch.cuda.is_available()}")
# print(torch.cuda.get_device_name(0))
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.envs.AutoroutingRLAviary import AutoroutingRLAviary
from gym_pybullet_drones.utils.utils import sync, str2bool
from gym_pybullet_drones.utils.enums import Physics, DroneModel

# Custom DQN
from gym_pybullet_drones.drl_custom.networks.FCQ import FCQ
from gym_pybullet_drones.drl_custom.networks.FCDuelingQ import FCDuelingQ

import pybullet_data


DEFAULT_DRONES = DroneModel("cf2x")
DEFAULT_RECORD_VISION = False
DEFAULT_GUI = True
DEFAULT_RECORD_VIDEO = False
DEFAULT_AGENTS = 1
DEFAULT_PHYSICS = Physics("pyb")
DEFAULT_CONTROL_FREQ_HZ = 60
DEFAULT_SIMULATION_FREQ_HZ = 60

INIT_XYZS = np.array([[((-1)**i)*(i*0.2)+0.5,-3*(i*0.05), 0.5+ 0.05*i ] for i in range(DEFAULT_AGENTS)])
INIT_RPYS = np.array([[0, 0,  i * (np.pi/2)/DEFAULT_AGENTS] for i in range(DEFAULT_AGENTS)])

# TODO: automate file-reading to select proper model e.g. DQN->use FCQ, DuelingDQN->use FCDueling

# value_model_fn = lambda nS, nA: FCQ(nS, nA, hidden_dims=(512,128))
value_model_fn = lambda nS, nA: FCDuelingQ(nS, nA, hidden_dims=(512,128))

# TODO: automate reading the correct dimension of nS and nA
numRays = 24
numObserv = 133
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
# fileName += "Model-DuelingDDQN-10.22.2024_17.52.46.pth"  # Good!!!!
# fileName += "Model-PER-10.23.2024_12.47.15.pth" # PER 40 min (success sometime)
# fileName += "Model-DuelingDDQN-10.24.2024_15.34.32.pth"  # 90 minutes (stay hovering at the start (why?))
# fileName += "Model-DuelingDDQN-10.25.2024_11.18.04.pth" # Very Good! (reduce num_rays to 10: nS=63, rayLen=1.25, reward_choice=8)
# fileName += "Model-PER-10.25.2024_12.18.39.pth"  # Not so good
fileName += "Model-DuelingDDQN-10.28.2024_17.44.21.pth"  # Very Good (num_rays = 24: nS=133, rayLen=1.5, reward_choice=8 (near hit ok))
# fileName += "Model-DuelingDDQN-10.28.2024_18.41.40.pth"
# CAUTION: If change number of actions -> need to also modify the action space in testing environment (AutoroutingRLAviary)!!!!
model.load_state_dict(torch.load(fileName,map_location=torch.device('cpu'), weights_only=True))
model.eval()
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# model = model.to(device)
model = model.to("cpu")
print("Model loaded and ready for inference!")
# input("Press Enter to continue . . .")


#========== Visualizing ========
env = AutoroutingRLAviary(
                drone_model = DEFAULT_DRONES,
                initial_xyzs=INIT_XYZS,
                initial_rpys=INIT_RPYS,
                physics= DEFAULT_PHYSICS,
                pyb_freq=DEFAULT_SIMULATION_FREQ_HZ,
                ctrl_freq=DEFAULT_CONTROL_FREQ_HZ,
                gui=DEFAULT_GUI,
                record=DEFAULT_RECORD_VIDEO,
                )

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
        # Convert the state to tensor if needed
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        # Get the action using the trained model 
        with torch.no_grad():
            action = model(state_tensor).argmax(dim=1).item()
            print(f"action = {action}")
        # # ======Random Action!!=========
        # action = random.randint(2, 3)
        # action = 3

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

