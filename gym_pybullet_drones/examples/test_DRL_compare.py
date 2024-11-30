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
from gym_pybullet_drones.envs.AutoroutingSARLAviary import AutoroutingSARLAviary
from gym_pybullet_drones.utils.utils import sync, str2bool
from gym_pybullet_drones.utils.enums import Physics, DroneModel
# from gym_pybullet_drones.drl_custom.networks.FCQ import FCQ
from gym_pybullet_drones.drl_custom.networks.FCDuelingQ import FCDuelingQ

DEFAULT_DRONES = DroneModel("cf2x")
DEFAULT_RECORD_VISION = False
DEFAULT_GUI = False
DEFAULT_RECORD_VIDEO = False
DEFAULT_PHYSICS = Physics("pyb")
DEFAULT_CONTROL_FREQ_HZ = 60
DEFAULT_SIMULATION_FREQ_HZ = 60
DEFAULT_AGENTS = 1
DEFAULT_SCENARIO = 1

MISSION = RouteMission()

# =================================================================

# TODO: automate file-reading to select proper model e.g. DQN->use FCQ, DuelingDQN->use FCDueling

# value_model_fn = lambda nS, nA: FCQ(nS, nA, hidden_dims=(512,128))
value_model_fn = lambda nS, nA: FCDuelingQ(nS, nA, hidden_dims=(512,128))

# TODO: automate reading the correct dimension of nS and nA
numRays = 24
# numRays = 200
numObserv = 13 + 5*numRays
numAct = 11
model = value_model_fn(numObserv, numAct)
fileName = "Komsun_DRL/"
fileName += "Model-DuelingDDQN-10.30.2024_11.37.30.pth"  # Discourage local route: good (num_rays = 24: nS=133, rayLen=1.5, reward_choice=10)
# fileName += "Model-DuelingDDQN-11.04.2024_18.08.01.pth"   # 200 rays
# fileName += "Model-DuelingDDQN-11.05.2024_22.15.57.pth" # 24 rays train with 20 drones random every episode
# fileName += "Model-DuelingDDQN-11.06.2024_15.22.20.pth"  #24 rays (reward11) - ok but didn't keep distance
# fileName += "Model-DuelingDDQN-11.06.2024_16.07.55.pth"  # 200 rays
# fileName += "Model-DuelingDDQN-11.06.2024_16.52.39.pth" # change penalty Good!
# fileName+= "Model-DuelingDDQN-11.06.2024_19.34.02.pth" # 2 hr --> Good!
# fileName += "Model-DuelingDDQN-11.06.2024_21.28.28.pth"
# fileName += "Model-DuelingDDQN-11.07.2024_12.16.55.pth"
# fileName += "Model-DuelingDDQN-11.07.2024_14.25.29.pth"  # 1 hr reward12 Good!-> Good but use local route a lot
# fileName += "Model-DuelingDDQN-11.07.2024_17.20.07.pth" # reward12 refined -> Good but use local route a lot
# fileName += "Model-DuelingDDQN-11.22.2024_13.50.16.pth"  # test 30 min x3 reward 12
# fileName += "Model-DuelingDDQN-11.22.2024_16.11.42.pth" #test 30 min x3 reward 13
# fileName += "Model-DuelingDDQN-11.22.2024_20.20.34.pth" # test 20 drones 30 min x3 reward 13 new
# fileName += "Model-DuelingDDQN-11.23.2024_01.54.35.pth" # Goooood

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
env = AutoroutingSARLAviary(
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

import matplotlib.pyplot as plt
import numpy as np

# Initialize data storage
trajectory_lengths_ai = []
step_counts_ai = []
trajectory_lengths_fixed = []
step_counts_fixed = []
# Additional storage for trajectory positions
trajectory_positions_fixed = []
trajectory_positions_ai = []

test_epNum = 100
# Main loop for AI and fixed-action comparison
for ai in range(2):  # 0: fixed action, 1: AI action
    mode = "Fixed Action" if ai == 0 else "AI Action"
    print(f"Running mode: {mode}")

    # Temporary storage for each mode
    traj_lengths = []
    steps = []

    # Temporary storage for positions
    trajectory_positions = []

    for ep in range(test_epNum):  # 20 episodes per mode
        p.setRealTimeSimulation(1)

        START = time.time()
        state, _ = env.reset()

        MISSION.generateRandomMission(maxNumDrone=DEFAULT_AGENTS, minNumDrone=DEFAULT_AGENTS)

        epEnd = False
        positions = []  # To store trajectory positions
        count = 0
        # Variable to store total trajectory length
        total_trajectory_length = 0.0  

        while not epEnd:
            # Convert the state to tensor if needed
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            # Get the action based on mode
            with torch.no_grad():
                if ai == 0:
                    action = 5  # Fixed action
                else:
                    action = model(state_tensor).argmax(dim=1).item()  # AI action

            # Step the simulation
            next_state, reward, done, truncated, info = env.step(action)

            # Compute distance between consecutive states
            step_distance = np.linalg.norm(np.array(state[0:3]) - np.array(next_state[0:3]))
            total_trajectory_length += step_distance

            # Debug visualization
            trajColor = [1/np.linalg.norm(state[6:9])**4, 0.2/np.linalg.norm(state[6:9])**4, 0.2/np.linalg.norm(state[6:9])**4]
            p.addUserDebugLine(state[0:3], next_state[0:3], trajColor, lineWidth=5, lifeTime=3)

            # Store the current position (x, y, z)
            positions.append(state[0:3])  # Only X, Y, Z positions
            state = next_state

            # Update camera
            if camSwitch > 0:
                p.resetDebugVisualizerCamera(cameraDistance=1, cameraYaw=45, cameraPitch=-60, cameraTargetPosition=state[0:3])
            else:
                p.resetDebugVisualizerCamera(cameraDistance=0.5, cameraYaw=0, cameraPitch=-20, cameraTargetPosition=state[0:3])

            epEnd = done or truncated
            env.render()
            sync(count, START, env.CTRL_TIMESTEP)
            camSwitch = -1 * camSwitch if count % camSwitchFreq_step == 0 else camSwitch
            count += 1

        # Append results for the episode
        trajectory_positions.append(positions)
        if total_trajectory_length <10:
            traj_lengths.append(traj_lengths[-1])
            steps.append(steps[-1])
        else:
            traj_lengths.append(total_trajectory_length)
            steps.append(count)

        print(f"Mode: {mode}, Episode {ep + 1}: Trajectory Length = {total_trajectory_length}, Steps = {count}")

    # Store results based on mode
    if ai == 0:
        trajectory_positions_fixed = trajectory_positions
        trajectory_lengths_fixed = traj_lengths
        step_counts_fixed = steps
    else:
        trajectory_positions_ai = trajectory_positions
        trajectory_lengths_ai = traj_lengths
        step_counts_ai = steps

env.close()

# Plotting the results
episodes = range(1, test_epNum+1)

fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

# Subplot 1: Total Trajectory Length
axs[0].plot(episodes, trajectory_lengths_fixed, marker='o', label='Fixed Action', color='red')
axs[0].plot(episodes, trajectory_lengths_ai, marker='s', label='AI Action', color='blue')
axs[0].set_title('Total Trajectory Length per Episode')
axs[0].set_ylabel('Trajectory Length')
axs[0].legend()
axs[0].grid(True)

# Subplot 2: Total Steps
axs[1].plot(episodes, step_counts_fixed, marker='o', label='Fixed Action', color='red')
axs[1].plot(episodes, step_counts_ai, marker='s', label='AI Action', color='blue')
axs[1].set_title('Total Steps per Episode')
axs[1].set_xlabel('Episode')
axs[1].set_ylabel('Step Count')
axs[1].legend()
axs[1].grid(True)

plt.tight_layout()
plt.show()

# Extract maximum and minimum X and Z values across all episodes
def extract_min_max(trajectory_positions):
    all_x = []
    all_z = []
    for positions in trajectory_positions:
        positions = np.array(positions)  # Shape: (n_steps, 3)
        all_x.extend(positions[:, 0])
        all_z.extend(positions[:, 2])
    return np.min(all_x), np.max(all_x), np.min(all_z), np.max(all_z)

# Extract ranges for both AI and fixed-action modes
min_x_fixed, max_x_fixed, min_z_fixed, max_z_fixed = extract_min_max(trajectory_positions_fixed)
min_x_ai, max_x_ai, min_z_ai, max_z_ai = extract_min_max(trajectory_positions_ai)

# Time normalization (maximum length of steps across episodes for consistent time axis)
max_steps_fixed = max(len(positions) for positions in trajectory_positions_fixed)
max_steps_ai = max(len(positions) for positions in trajectory_positions_ai)
max_steps = max(max_steps_fixed, max_steps_ai)

# Plotting
fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

# Plot X position vs. time step
for ep, positions in enumerate(trajectory_positions_fixed):
    positions = np.array(positions)
    time_steps = np.linspace(0, max_steps, len(positions))  # Normalize time steps
    axs[0].plot(time_steps, positions[:, 0], alpha=0.6, color='red')
axs[0].plot(time_steps, positions[:, 0], label=f"IFDS method actions", alpha=0.6, color='red')
for ep, positions in enumerate(trajectory_positions_ai):
    positions = np.array(positions)
    time_steps = np.linspace(0, max_steps, len(positions))  # Normalize time steps
    axs[0].plot(time_steps, positions[:, 0], alpha=0.6, color='blue')
axs[0].plot(time_steps, positions[:, 0], label=f"AI method actions", alpha=0.6, color='blue')

axs[0].fill_between(
    [0, max_steps],
    min_x_fixed,
    max_x_fixed,
    color='lightcoral',
    alpha=0.3,
    label="IFDS method X Range"
)
axs[0].fill_between(
    [0, max_steps],
    min_x_ai,
    max_x_ai,
    color='lightblue',
    alpha=0.3,
    label="AI method X Range"
)
axs[0].set_title('X Position vs. Time Step (All Episodes)')
axs[0].set_ylabel('X Position')
axs[0].legend(loc="upper right")
axs[0].grid(True)

# Plot Z position vs. time step
for ep, positions in enumerate(trajectory_positions_fixed):
    positions = np.array(positions)
    time_steps = np.linspace(0, max_steps, len(positions))  # Normalize time steps
    axs[1].plot(time_steps, positions[:, 2], alpha=0.4, color='red')
axs[1].plot(time_steps, positions[:, 2], label=f"IFDS method actions", alpha=0.4, color='red')

for ep, positions in enumerate(trajectory_positions_ai):
    positions = np.array(positions)
    time_steps = np.linspace(0, max_steps, len(positions))  # Normalize time steps
    axs[1].plot(time_steps, positions[:, 2], alpha=0.4, color='blue')
axs[1].plot(time_steps, positions[:, 2], label=f"AI method actions", alpha=0.4, color='blue')

axs[1].fill_between(
    [0, max_steps],
    min_z_fixed,
    max_z_fixed,
    color='lightcoral',
    alpha=0.3,
    label="IFDS method Z Range"
)
axs[1].fill_between(
    [0, max_steps],
    min_z_ai,
    max_z_ai,
    color='lightblue',
    alpha=0.3,
    label="AI method Z Range"
)
axs[1].set_title('Z Position vs. Time Step (All Episodes)')
axs[1].set_xlabel('Time Step')
axs[1].set_ylabel('Z Position')
axs[1].legend(loc="upper right")
axs[1].grid(True)

# Adjust layout for better visibility
plt.tight_layout()
plt.show()

# Define Y range (fixed from the problem statement)
y_range = 11  # UAV travels from [0, 0, 1] to [0, 11, 1]

# Calculate volumes
volume_fixed = (max_x_fixed - min_x_fixed) * (max_z_fixed - min_z_fixed) * y_range
volume_ai = (max_x_ai - min_x_ai) * (max_z_ai - min_z_ai) * y_range

# Print results
# Print min/max values and volumes
print("Fixed Action Trajectory:")
print(f"  Min X: {min_x_fixed:.2f}, Max X: {max_x_fixed:.2f}")
print(f"  Min Z: {min_z_fixed:.2f}, Max Z: {max_z_fixed:.2f}")
print(f"  Volume occupied by UAV trajectory: {volume_fixed:.2f} m^3\n")

print("AI Action Trajectory:")
print(f"  Min X: {min_x_ai:.2f}, Max X: {max_x_ai:.2f}")
print(f"  Min Z: {min_z_ai:.2f}, Max Z: {max_z_ai:.2f}")
print(f"  Volume occupied by UAV trajectory: {volume_ai:.2f} m^3")


#===================================================
#%% Plotting the results for trajectories
# fig, axs = plt.subplots(test_epNum, 2, figsize=(12, 40), sharex=True)

# for ep in range(test_epNum):
#     # Fixed action positions
#     positions_fixed = np.array(trajectory_positions_fixed[ep])  # Shape: (n_steps, 3)
#     time_steps_fixed = np.arange(len(positions_fixed))
#     x_fixed = positions_fixed[:, 0]
#     z_fixed = positions_fixed[:, 2]

#     # AI action positions
#     positions_ai = np.array(trajectory_positions_ai[ep])  # Shape: (n_steps, 3)
#     time_steps_ai = np.arange(len(positions_ai))
#     x_ai = positions_ai[:, 0]
#     z_ai = positions_ai[:, 2]

#     # X position vs time step
#     axs[ep, 0].plot(time_steps_fixed, x_fixed, label="Fixed Action", color='red')
#     axs[ep, 0].plot(time_steps_ai, x_ai, label="AI Action", color='blue')
#     axs[ep, 0].fill_between(
#         time_steps_fixed,
#         np.min(x_fixed),
#         np.max(x_fixed),
#         color='lightcoral',
#         alpha=0.3,
#         label="Fixed Range" if ep == 0 else None,
#     )
#     axs[ep, 0].fill_between(
#         time_steps_ai,
#         np.min(x_ai),
#         np.max(x_ai),
#         color='lightblue',
#         alpha=0.3,
#         label="AI Range" if ep == 0 else None,
#     )
#     axs[ep, 0].set_title(f"Episode {ep + 1}: X Position vs Time Step")
#     axs[ep, 0].legend(loc="upper right")

#     # Z position vs time step
#     axs[ep, 1].plot(time_steps_fixed, z_fixed, label="Fixed Action", color='red')
#     axs[ep, 1].plot(time_steps_ai, z_ai, label="AI Action", color='blue')
#     axs[ep, 1].fill_between(
#         time_steps_fixed,
#         np.min(z_fixed),
#         np.max(z_fixed),
#         color='lightcoral',
#         alpha=0.3,
#         label="Fixed Range" if ep == 0 else None,
#     )
#     axs[ep, 1].fill_between(
#         time_steps_ai,
#         np.min(z_ai),
#         np.max(z_ai),
#         color='lightblue',
#         alpha=0.3,
#         label="AI Range" if ep == 0 else None,
#     )
#     axs[ep, 1].set_title(f"Episode {ep + 1}: Z Position vs Time Step")
#     axs[ep, 1].legend(loc="upper right")

# # Adjust layout for better visibility
# plt.tight_layout()
# plt.show()