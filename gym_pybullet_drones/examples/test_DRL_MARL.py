import os
import time
import random
import gymnasium as gym
import numpy as np
import torch
import pybullet as p
import matplotlib.pyplot as plt
from gym_pybullet_drones.routing.RouteMission import RouteMission
print(f"CUDA is available : {torch.cuda.is_available()}")
# print(torch.cuda.get_device_name(0))
# from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.envs.AutoroutingRLAviary import AutoroutingRLAviary
from gym_pybullet_drones.envs.AutoroutingMARLAviary import AutoroutingMARLAviary
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
DEFAULT_AGENTS = 5
DEFAULT_SCENARIO = 1

MISSION = RouteMission()

# =================================================================

# TODO: automate file-reading to select proper model e.g. DQN->use FCQ, DuelingDQN->use FCDueling

# value_model_fn = lambda nS, nA: FCQ(nS, nA, hidden_dims=(512,128))
value_model_fn = lambda nS, nA: FCDuelingQ(nS, nA, hidden_dims=(512,128))

# TODO: automate reading the correct dimension of nS and nA
numAgents = 5
numRays = 200
# numRays = 200
numObserv = numAgents*12 + 1 + 5*numRays
numAct = 11
model = value_model_fn(numObserv, numAct)
fileName = "Komsun_DRL/"

fileName += "Model-DuelingDDQN-05.12.2025_12.08.04.pth"  # Discourage local route: good (num_rays = 24: nS=133, rayLen=1.5, reward_choice=10)
# fileName += "Model-DuelingDDQN-10.30.2024_11.55.38.pth"  # My first Multi-agent (3 agents) (num_rays = 24: nS=133*3, rayLen=1.5, reward_choice=10)
# fileName += "Model-DuelingDDQN-10.30.2024_12.25.21.pth"   # Good 3 Agents result!!
# fileName += "Model-DuelingDDQN-10.31.2024_11.00.45.pth"  #3 agents (reward_choice = 10 (new)) (with 3xobservations)
# fileName += "Model-DuelingDDQN-10.30.2024_14.21.45.pth" # 5 Agents (with 5xobservations)
# fileName += "Model-DuelingDDQN-10.31.2024_11.11.20.pth" #10 Agents (reward_choice = 10 (new)) (with 10xobservations)
# fileName += "Model-DuelingDDQN-10.31.2024_17.17.14.pth"   #10 Agents 3 actions


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


MISSION.generateRandomMission(maxNumDrone=DEFAULT_AGENTS, minNumDrone=DEFAULT_AGENTS)
INIT_XYZS = MISSION.INIT_XYZS
INIT_RPYS = MISSION.INIT_RPYS
DESTINS = MISSION.DESTINS
env = AutoroutingMARLAviary(
                drone_model = DEFAULT_DRONES,
                initial_xyzs=INIT_XYZS,
                initial_rpys=INIT_RPYS,
                physics= DEFAULT_PHYSICS,
                pyb_freq=DEFAULT_SIMULATION_FREQ_HZ,
                ctrl_freq=DEFAULT_CONTROL_FREQ_HZ,
                gui=DEFAULT_GUI,
                record=DEFAULT_RECORD_VIDEO,
                num_drones=DEFAULT_AGENTS,
                )
for j in range(DEFAULT_AGENTS):
    env.routing[j].NUM_RAYS = numRays 
    env.routing[j].DESTINATION = MISSION.DESTINS[j,:]

p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
# p.configureDebugVisualizer(rgbBackground=[0, 0, 0])
# 
camSwitch = 1
camSwitchFreq_step = 120
SEEDS = (12, 34, 5)

# Lists to store trajectory lengths and steps for each episode
trajectory_lengths = []
step_counts = []

for ep in range(20):
    p.setRealTimeSimulation(1) 
 
    START = time.time()
    state, _ = env.reset()

    MISSION.generateRandomMission(maxNumDrone=DEFAULT_AGENTS, minNumDrone=DEFAULT_AGENTS)
    
    epEnd = False
    count = 0
    # Variable to store total trajectory length
    total_trajectory_length = 0.0  

    while not epEnd:
        # state = state[0:numObserv] # only for the first drone
        # Convert the state to tensor if needed
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        # Get the action using the trained model 
        with torch.no_grad():
            action = model(state_tensor).argmax(dim=1).item()
            # action = 4
            print(f"action = {action}")

        #### Step the simulation ###################################
        # Take the action in the environment
        next_state, reward, done, truncated, info = env.step(action)

        # Compute distance between consecutive states
        step_distance = np.linalg.norm(np.array(state[0:3]) - np.array(next_state[0:3]))
        total_trajectory_length += step_distance  # Update cumulative distance

        trajColor = [1/np.linalg.norm(state[6:9])**4, 0.2/np.linalg.norm(state[6:9])**4,  0.2/np.linalg.norm(state[6:9])**4]
        p.addUserDebugLine(state[0:3], next_state[0:3], trajColor, lineWidth=5, lifeTime=5)
        state = next_state



        
        if camSwitch>0:
            p.resetDebugVisualizerCamera(cameraDistance=1, cameraYaw=45, cameraPitch=-60, cameraTargetPosition=state[0:3])
        else:
            p.resetDebugVisualizerCamera(cameraDistance=0.5, cameraYaw=0, cameraPitch=-20, cameraTargetPosition=state[0:3])
        # p.resetDebugVisualizerCamera(cameraDistance=1, cameraYaw=45, cameraPitch=-60, cameraTargetPosition=state[0:3])
        # p.resetDebugVisualizerCamera(cameraDistance=0.5, cameraYaw=0, cameraPitch=-20, cameraTargetPosition=state[0:3])
        # p.resetDebugVisualizerCamera(cameraDistance=4, cameraYaw=0, cameraPitch=-89, cameraTargetPosition=np.array([0, 3.5, 1]))
        # p.resetDebugVisualizerCamera(cameraDistance=5, cameraYaw=0, cameraPitch=-89, cameraTargetPosition=np.array([0, 6, 1]))
        # p.resetDebugVisualizerCamera(cameraDistance=3, cameraYaw=0, cameraPitch=-89, cameraTargetPosition=state[0:3])
        epEnd = done or truncated
        #### Printout ##############################################
        env.render()

        sync(count, START, env.CTRL_TIMESTEP)
        camSwitch = -1*camSwitch if count%camSwitchFreq_step == 0 else camSwitch
        count += 1
    # Append results for the episode
    if not truncated:
        trajectory_lengths.append(total_trajectory_length)
        step_counts.append(count)

    print(f"Episode {ep + 1}: Total trajectory length = {total_trajectory_length}, Steps = {count}")



env.close()

#### Close the environment #################################

    
# Plotting
episodes = range(1, 21)
# Create the figure and subplots
fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Subplot 1: Total Trajectory Length
axs[0].plot(episodes, trajectory_lengths, marker='o', label='Trajectory Length', color='blue')
axs[0].set_title('Total Trajectory Length per Episode')
axs[0].set_ylabel('Trajectory Length (m)')
axs[0].grid(True)
axs[0].legend()

# Subplot 2: Total Steps
axs[1].plot(episodes, step_counts, marker='s', label='Step Counts', color='green')
axs[1].set_title('Total Steps per Episode')
axs[1].set_xlabel('Successful Episode')
axs[1].set_ylabel('Step Count')
axs[1].grid(True)
axs[1].legend()

# Adjust layout
plt.tight_layout()
plt.show()