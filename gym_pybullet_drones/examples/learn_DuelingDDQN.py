import os
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
# os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS'] = '1'
import time
from datetime import datetime
import argparse
import gymnasium as gym
import numpy as np
import torch
print(f"CUDA is available : {torch.cuda.is_available()}")
# print(torch.cuda.get_device_name(0))
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.envs.HoverAviary import HoverAviary
from gym_pybullet_drones.envs.AutoroutingRLAviary import AutoroutingRLAviary
from gym_pybullet_drones.envs.MultiHoverAviary import MultiHoverAviary
from gym_pybullet_drones.utils.utils import sync, str2bool
from gym_pybullet_drones.utils.enums import ObservationType, ActionType, Physics

# Custom DQN
import torch.optim as optim
from gym_pybullet_drones.drl_custom.networks.FCQ import FCQ
from gym_pybullet_drones.drl_custom.networks.FCDuelingQ import FCDuelingQ
from gym_pybullet_drones.drl_custom.exploration_strategies import EGreedyExpStrategy, GreedyStrategy
from gym_pybullet_drones.drl_custom.utils import get_make_env_fn

from gym_pybullet_drones.drl_custom.value_based_DRL.DQN import DQN
from gym_pybullet_drones.drl_custom.value_based_DRL.DuelingDDQN import DuelingDDQN
from gym_pybullet_drones.drl_custom.replay_buffers.ReplayBuffer import ReplayBuffer

# DEFAULT_GUI = True
# DEFAULT_RECORD_VIDEO = False
# DEFAULT_OUTPUT_FOLDER = 'results'
# DEFAULT_COLAB = False

# DEFAULT_OBS = ObservationType('kin') # 'kin' or 'rgb'
# DEFAULT_ACT = ActionType('autorouting') # 'rpm' or 'pid' or 'vel' or 'one_d_rpm' or 'one_d_pid'

# DEFAULT_MA = False
# DEFAULT_PHYSICS = Physics("pyb")
# DEFAULT_CONTROL_FREQ_HZ = 60
# DEFAULT_SIMULATION_FREQ_HZ = 60

# INIT_XYZS = np.array([[((-1)**i)*(i*0.2)+0.5,-3*(i*0.05), 0.5+ 0.05*i ] for i in range(DEFAULT_AGENTS)])
# INIT_RPYS = np.array([[0, 0,  i * (np.pi/2)/DEFAULT_AGENTS] for i in range(DEFAULT_AGENTS)])


# New

LEAVE_PRINT_EVERY_N_SECS = 60
ERASE_LINE = '\x1b[2K'
EPS = 1e-6
BEEP = lambda: os.system("printf '\a'")
RESULTS_DIR = os.path.join('..', 'results')
# SEEDS = (12, 34, 56, 78, 90)
# SEEDS = (12, 34, 5)
# SEEDS = tuple(range(100))
SEEDS = range(1,)

DEFAULT_AGENTS = 20


dueling_ddqn_results = []
dueling_ddqn_agents, best_dueling_ddqn_agent_key, best_eval_score = {}, None, float('-inf')
for seed in SEEDS:
    # DEFAULT_AGENTS += 5
    # environment_settings = {
    #     'env_name': 'autorouting-aviary-v0',
    #     'gamma': 0.9, # 0.995
    #     'max_minutes': 5,
    #     'max_episodes': 10000,
    #     'goal_mean_100_reward': 3000  # 150 to be determined properly
    # }
    environment_settings = {
        'env_name': 'autorouting-sa-aviary-v0',
        'gamma': 0.9, # 0.995
        'max_minutes': 20,
        'max_episodes': 20000,
        'goal_mean_100_reward': 4500  # 150 to be determined properly
    }
    
    # value_model_fn = lambda nS, nA: FCQ(nS, nA, hidden_dims=(512,128))
    value_model_fn = lambda nS, nA: FCDuelingQ(nS, nA, hidden_dims=(512,128))
    value_optimizer_fn = lambda net, lr: optim.RMSprop(net.parameters(), lr=lr)
    value_optimizer_lr = 0.0005
    max_gradient_norm = float('inf')

    training_strategy_fn = lambda: EGreedyExpStrategy(init_epsilon=1.0,  
                                                      min_epsilon=0.3, 
                                                      decay_steps=100000)
    # training_strategy_fn = lambda: EGreedyExpStrategy(init_epsilon=1.0,  
    #                                                   min_epsilon=0.3, 
    #                                                   decay_steps=10000)
    evaluation_strategy_fn = lambda: GreedyStrategy()

    replay_buffer_fn = lambda: ReplayBuffer(max_size=50000, batch_size=64)
    n_warmup_batches = 5   
    update_target_every_steps = 1
    tau = 0.1

    env_name, gamma, max_minutes, \
    max_episodes, goal_mean_100_reward = environment_settings.values()
    agent = DuelingDDQN(replay_buffer_fn,
                        value_model_fn,
                        value_optimizer_fn,
                        value_optimizer_lr,
                        max_gradient_norm,
                        training_strategy_fn,
                        evaluation_strategy_fn,
                        n_warmup_batches,
                        update_target_every_steps,
                        tau)

    make_env_fn, make_env_kargs = get_make_env_fn(env_name=env_name, num_drones=DEFAULT_AGENTS)
    result, final_eval_score, training_time, wallclock_time = agent.train(
        make_env_fn, make_env_kargs, seed, gamma, max_minutes, max_episodes, goal_mean_100_reward)
    dueling_ddqn_results.append(result)
    dueling_ddqn_agents[seed] = agent
    if final_eval_score > best_eval_score:
        best_eval_score = final_eval_score
        best_dueling_ddqn_agent_key = seed
dueling_ddqn_results = np.array(dueling_ddqn_results)
_ = BEEP()

dueling_ddqn_agents[best_dueling_ddqn_agent_key].save_model()

#%%
# best_agent.demo_progression()
# input("Press Enter to continue...")
# dueling_ddqn_agents[best_dueling_ddqn_agent_key].demo_last()

