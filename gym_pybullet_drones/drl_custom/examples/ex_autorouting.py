import gymnasium as gym
import os
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS'] = '16'

import time
import torch.optim as optim
import numpy as np
from tqdm import tqdm

from gym_pybullet_drones.drl_custom.networks.FCQ import FCQ
from gym_pybullet_drones.drl_custom.exploration_strategies import EGreedyExpStrategy, GreedyStrategy
from gym_pybullet_drones.drl_custom.utils import get_make_env_fn
from gym_pybullet_drones.utils.utils import sync, str2bool

from gym_pybullet_drones.drl_custom.DQN import DQN
from gym_pybullet_drones.drl_custom.replay_buffers.ReplayBuffer import ReplayBuffer
from gym_pybullet_drones.utils.enums import ObservationType, ActionType, Physics

from gym_pybullet_drones.envs.RoutingAviary import RoutingAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics
LEAVE_PRINT_EVERY_N_SECS = 60
ERASE_LINE = '\x1b[2K'
EPS = 1e-6
BEEP = lambda: os.system("printf '\a'")
RESULTS_DIR = os.path.join('..', 'results')
SEEDS = (12, 34, 56, 78, 90)

num_training_episodes = 100

DEFAULT_DRONES = DroneModel("cf2x")
DEFAULT_GUI = True
DEFAULT_RECORD_VIDEO = False
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_COLAB = False

DEFAULT_OBS = ObservationType('kin') # 'kin' or 'rgb'
DEFAULT_ACT = ActionType('autorouting') # 'rpm' or 'pid' or 'vel' or 'one_d_rpm' or 'one_d_pid'
DEFAULT_AGENTS = 1
DEFAULT_MA = False
DEFAULT_PHYSICS = Physics("pyb")
DEFAULT_CONTROL_FREQ_HZ = 60
DEFAULT_SIMULATION_FREQ_HZ = 60

INIT_XYZS = np.array([[((-1)**i)*(i*0.2)+0.5,-3*(i*0.05), 0.5+ 0.05*i ] for i in range(DEFAULT_AGENTS)])
INIT_RPYS = np.array([[0, 0,  i * (np.pi/2)/DEFAULT_AGENTS] for i in range(DEFAULT_AGENTS)])

env = gym.make("autorouting-aviary-v0", drone_model = DEFAULT_DRONES,
                                        gui=True,
                                        obs=DEFAULT_OBS, 
                                        act=DEFAULT_ACT, 
                                        physics=DEFAULT_PHYSICS, 
                                        ctrl_freq = DEFAULT_CONTROL_FREQ_HZ, 
                                        pyb_freq = DEFAULT_SIMULATION_FREQ_HZ,
                                        initial_xyzs=INIT_XYZS,
                                        initial_rpys=INIT_RPYS,
                                        record = False,
                                             )





start = time.time()
for episode_num in range(num_training_episodes):
    observation, info = env.reset()

    episode_over = False
    while not episode_over:
        action = env.action_space.sample()  # agent policy that uses the observation and info
        observation, reward, terminated, truncated, info = env.step(action)

        episode_over = terminated or truncated
        env.render()
        sync(episode_num, start, env.CTRL_TIMESTEP)

env.close()