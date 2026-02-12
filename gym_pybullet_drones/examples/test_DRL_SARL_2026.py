import time
import random
import gymnasium as gym
import itertools
import numpy as np
import torch
import pybullet as p
import matplotlib.pyplot as plt
print(f"CUDA is available : {torch.cuda.is_available()}")
# print(torch.cuda.get_device_name(0))
# from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.envs.AutoroutingSARLAviary import AutoroutingSARLAviary
from gym_pybullet_drones.utils.utils import sync, str2bool
from gym_pybullet_drones.utils.enums import Physics, DroneModel
# from gym_pybullet_drones.drl_custom.networks.FCQ import FCQ
from gym_pybullet_drones.drl_custom.networks.FCDuelingQ import FCDuelingQ


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

def draw_drone_id_labels(env, positions, debug_items, z_offset=0.35, text_size=1.4, color=[1, 1, 1]):
    """
    Draw env.routing[i].DRONE_ID above each drone using PyBullet debug text.
    The text is placed at a local offset and parented to the drone body.
    """
    for i in range(len(positions)):
        drone_id = getattr(env.routing[i], "DRONE_ID", i)  # fallback to i if missing
        debug_items.append(
            p.addUserDebugText(
                text=str(drone_id),
                textPosition=[0, 0, z_offset],          # local offset (because of parent)
                textColorRGB=color,
                textSize=text_size,
                parentObjectUniqueId=env.DRONE_IDS[i]    # attach to the drone
            )
        )


DEFAULT_DRONES = DroneModel("hb")
DEFAULT_NUM_DRONES = 10
DEFAULT_PHYSICS = Physics("pyb")
DEFAULT_GUI = True
DEFAULT_RECORD_VISION = False
DEFAULT_PLOT = False
DEFAULT_USER_DEBUG_GUI = False
DEFAULT_OBSTACLES = True
DEFAULT_SIMULATION_FREQ_HZ = 30
DEFAULT_CONTROL_FREQ_HZ = 30
DEFAULT_DURATION_SEC = 20
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_COLAB = False

# =================================================================
# TODO: automate file-reading to select proper model e.g. DQN->use FCQ, DuelingDQN->use FCDueling

# value_model_fn = lambda nS, nA: FCQ(nS, nA, hidden_dims=(512,128))
value_model_fn = lambda nS, nA: FCDuelingQ(nS, nA, hidden_dims=(512,128))

env = AutoroutingSARLAviary(
                drone_model = DEFAULT_DRONES,
                num_drones = DEFAULT_NUM_DRONES, 
                physics= DEFAULT_PHYSICS,
                pyb_freq=DEFAULT_SIMULATION_FREQ_HZ,
                ctrl_freq=DEFAULT_CONTROL_FREQ_HZ,
                gui=DEFAULT_GUI,
                record=DEFAULT_RECORD_VISION,
                )
numObserv = env.observation_space.shape[0]
numAct = env.action_space.n
model = value_model_fn(numObserv, numAct)

# fileName = "Komsun_DRL/Model-DuelingDDQN-01.21.2026_16.18.44.pth" # Trial
# fileName = "Komsun_DRL/Model-DuelingDDQN-01.22.2026_12.33.42.pth"
# fileName = "Komsun_DRL/Model-DuelingDDQN-01.23.2026_20.41.55.pth"
# fileName = "Komsun_DRL/Model-DuelingDDQN-01.23.2026_22.56.22.pth"
# fileName = "Komsun_DRL/Model-DuelingDDQN-01.26.2026_12.49.36.pth"
# fileName = "Komsun_DRL/Model-DuelingDDQN-01.26.2026_16.50.48.pth"
# fileName = "Komsun_DRL/Model-DuelingDDQN-01.27.2026_14.21.23.pth"   #OBS_CHOICE = "sensor"
# fileName = "Komsun_DRL/Model-DuelingDDQN-01.28.2026_10.52.08.pth"
# fileName = "Komsun_DRL/Model-DuelingDDQN-01.28.2026_13.24.20.pth"
# fileName = "Komsun_DRL/Model-DuelingDDQN-01.28.2026_14.29.40.pth"  # Good?
# fileName = "Komsun_DRL/Model-DuelingDDQN-01.28.2026_16.53.16.pth"
# fileName = "Komsun_DRL/Model-DuelingDDQN-02.03.2026_11.42.36.pth" # Almost good
# fileName = "Komsun_DRL/Model-DuelingDDQN-02.03.2026_12.33.30.pth" # higher penalty
# fileName = "Komsun_DRL/Model-DuelingDDQN-02.03.2026_14.54.41.pth"  # Added LOS in observation
# fileName = "Komsun_DRL/Model-DuelingDDQN-02.03.2026_16.41.16.pth"
# fileName = "Komsun_DRL/Model-DuelingDDQN-02.05.2026_19.33.43.pth"
# fileName = "Komsun_DRL/Model-DuelingDDQN-02.06.2026_18.42.23.pth"
fileName = "Komsun_DRL/Model-DuelingDDQN-02.06.2026_23.29.10.pth" # 500 ep
# fileName = "Komsun_DRL/Model-DuelingDDQN-01.27.2026_16.25.02.pth"  #OBS_CHOICE = "ray"

# CAUTION: If change number of actions -> need to also modify the action space in testing environment (AutoroutingRLAviary)!!!!
model.load_state_dict(torch.load(fileName,map_location=torch.device('cpu'), weights_only=True))
model.eval()
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# model = model.to(device)
model = model.to("cpu")
print("Model loaded and ready for inference!")
# input("Press Enter to continue . . .")

def select_action(state):
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to("cpu")

    with torch.no_grad():
        q_values = model(state)

    action = q_values.argmax(dim=-1).item()
    return action




#### Create the environment ################################

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
    p.setRealTimeSimulation(0) 
    epEnd = False
    count = 0
    START = time.time()
    state, _ = env.reset()
    SAFE_DISTANCE = env.routing[0].ROV  # meters
    FLIGHT_GEO = 2* env.routing[0].SFG/2
    debug_items = []     # store current debug visuals
    # Variable to store total trajectory length
    total_trajectory_length = 0.0  

    while not epEnd:
        print(f"cum_reward = {env.CUM_REWARD}")
        count += 1
        # state = state[0:numObserv] # only for the first drone
        state_tensor = torch.tensor(state, dtype=torch.float32)
        # state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

        # Get the action using the trained model
        
        action = select_action(state)

        print(f"Action is {action}")
        # with torch.no_grad():
        #     action = model(state_tensor).argmax(dim=1).item()
        #     print(f"Action is {action}")

        #### Step the simulation ###################################
        # Take the action in the environment
        next_state, reward, done, truncated, info = env.step(action)
        state = next_state
        epEnd = done or truncated

        if epEnd:
            print(f"--------EPISODE ENDED: TOTAL STEPS = {count}")

        # Compute distance between consecutive states
        step_distance = np.linalg.norm(np.array(state[0:3]) - np.array(next_state[0:3]))
        total_trajectory_length += step_distance  # Update cumulative distance

        trajColor = [1/np.linalg.norm(state[0][6:9])**4, 0.2/np.linalg.norm(state[0][6:9])**4,  0.2/np.linalg.norm(state[6:9])**4]
        # p.addUserDebugLine(state[0][0:3], next_state[0][0:3], trajColor, lineWidth=5, lifeTime=5)

        # p.resetDebugVisualizerCamera(cameraDistance=5, cameraYaw=0, cameraPitch=-89, cameraTargetPosition=np.array([0, 6, 1]))
        for item in debug_items:
            p.removeUserDebugItem(item)
        debug_items = []
        min_dists_all = []   # optional history logging 
        # Drone's 1 following camera
        p.resetDebugVisualizerCamera(cameraDistance=35, cameraYaw=0, cameraPitch=-60, cameraTargetPosition=env.routing[0].CUR_POS)
        
        
        positions = [p.getBasePositionAndOrientation(env.DRONE_IDS[i])[0]
                        for i in range(DEFAULT_NUM_DRONES)]
        # === Draw DRONE_ID above each drone ===
        draw_drone_id_labels(env, positions, debug_items,
                             z_offset=0.35, text_size=1.4, color=[1, 1, 1])
            
        # === Compute pairwise distances ===
        dist_matrix = np.full((DEFAULT_NUM_DRONES, DEFAULT_NUM_DRONES), np.inf)
        for i, j in itertools.combinations(range(DEFAULT_NUM_DRONES), 2):
            dist = np.linalg.norm(np.array(positions[i]) - np.array(positions[j]))  # works now ✅
            dist_matrix[i, j] = dist_matrix[j, i] = dist

            # Draw color-coded line
            color = [1, 0, 0] if dist < SAFE_DISTANCE else [0, 1, 0]
            # debug_items.append(p.addUserDebugLine(positions[i], positions[j], color, lineWidth=2))

        # === Compute and display per-drone min separation ===
        per_drone_min = np.min(dist_matrix, axis=1)
        min_dists_all.append(per_drone_min)
        min_d = per_drone_min[0]
        if env.routing[0].RAYS_INFO.any():
            text_color = [1, 1, 0] if min_d < SAFE_DISTANCE else [0, 1, 0]
            debug_items.append(
                p.addUserDebugText(
                    f"{min_d:.2f} m",
                    [0, 0, 0.2],  # small offset above the drone
                    textColorRGB=text_color,
                    textSize=1.2,
                    parentObjectUniqueId=env.DRONE_IDS[0]
                )
            )

        # Draw a circle around each drone
        circle_color = [1, 1, 0] if min_d < SAFE_DISTANCE else [0, 1, 0]
        inner_color = [1, 0, 0] if min_d < SAFE_DISTANCE else [0, 1, 0]
        circle_ids = draw_circle_around_drone(
            center=positions[0],
            radius=SAFE_DISTANCE,
            color=circle_color,
            segments=36,
            z_offset=0.05
        )
        circle_inner_ids = draw_circle_around_drone(
            center=positions[0],
            radius=FLIGHT_GEO,
            color=[0, 1, 0],
            segments=36,
            z_offset=0.05
        )
        debug_items.extend(circle_ids)
        debug_items.extend(circle_inner_ids)
        #### Printout ##############################################
        env.render()

        sync(count, START, env.CTRL_TIMESTEP)
        camSwitch = -1*camSwitch if count%camSwitchFreq_step == 0 else camSwitch
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