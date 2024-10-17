import gymnasium as gym
import os
import torch.optim as optim
import numpy as np
from tqdm import tqdm

from gym_pybullet_drones.drl_custom.networks.FCQ import FCQ
from gym_pybullet_drones.drl_custom.exploration_strategies import EGreedyExpStrategy, GreedyStrategy
from gym_pybullet_drones.drl_custom.utils import get_make_env_fn

from gym_pybullet_drones.drl_custom.DQN import DQN
from gym_pybullet_drones.drl_custom.replay_buffers.ReplayBuffer import ReplayBuffer


LEAVE_PRINT_EVERY_N_SECS = 60
ERASE_LINE = '\x1b[2K'
EPS = 1e-6
BEEP = lambda: os.system("printf '\a'")
RESULTS_DIR = os.path.join('..', 'results')
SEEDS = (12, 34, 56, 78, 90)

num_training_episodes = 100


env = gym.make("CartPole-v1", render_mode="human")


for episode_num in tqdm(range(num_training_episodes), desc='Episodes: ', leave=False):
    observation, info = env.reset()

    episode_over = False
    while not episode_over:
        action = env.action_space.sample()  # agent policy that uses the observation and info
        observation, reward, terminated, truncated, info = env.step(action)

        episode_over = terminated or truncated

env.close()