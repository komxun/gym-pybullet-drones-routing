import os
# os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
# os.environ['CUDA_VISIBLE_DEVICES']=''
# os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np
import torch
print(f"CUDA is available : {torch.cuda.is_available()}")
# print(torch.cuda.get_device_name(0))
from gym_pybullet_drones.utils.enums import ObservationType, ActionType, Physics

# Custom DQN
import torch.optim as optim
from gym_pybullet_drones.drl_custom.networks.FCDuelingQ import FCDuelingQ
from gym_pybullet_drones.drl_custom.exploration_strategies import EGreedyExpStrategy, GreedyStrategy
from gym_pybullet_drones.drl_custom.utils import get_make_env_fn

from gym_pybullet_drones.drl_custom.value_based_DRL.PER import PER
from gym_pybullet_drones.drl_custom.replay_buffers.PrioritizedReplayBuffer import PrioritizedReplayBuffer

DEFAULT_GUI = True
DEFAULT_RECORD_VIDEO = False
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_COLAB = False

DEFAULT_OBS = ObservationType('kin') # 'kin' or 'rgb'
DEFAULT_ACT = ActionType('autorouting') # 'rpm' or 'pid' or 'vel' or 'one_d_rpm' or 'one_d_pid'
DEFAULT_AGENTS = 20
DEFAULT_MA = False
DEFAULT_PHYSICS = Physics("pyb")
DEFAULT_CONTROL_FREQ_HZ = 60
DEFAULT_SIMULATION_FREQ_HZ = 60

INIT_XYZS = np.array([[((-1)**i)*(i*0.2)+0.5,-3*(i*0.05), 0.5+ 0.05*i ] for i in range(DEFAULT_AGENTS)])
INIT_RPYS = np.array([[0, 0,  i * (np.pi/2)/DEFAULT_AGENTS] for i in range(DEFAULT_AGENTS)])

# New

LEAVE_PRINT_EVERY_N_SECS = 60
ERASE_LINE = '\x1b[2K'
EPS = 1e-6
BEEP = lambda: os.system("printf '\a'")
RESULTS_DIR = os.path.join('..', 'results')
# SEEDS = (12, 34, 56, 78, 90)
SEEDS = (12, 34, 69)
# SEEDS = tuple(range(10))

per_results = []
best_agent, best_eval_score = None, float('-inf')
for seed in SEEDS:
    print(f"seed = {seed}")
    # environment_settings = {
    #     'env_name': 'autorouting-aviary-v0',
    #     'gamma': 0.995,  #before 0.95
    #     'max_minutes': 10,
    #     'max_episodes': 10000,
    #     'goal_mean_100_reward': 110  # to be determined properly
    # }
    environment_settings = {
        'env_name': 'autorouting-sa-aviary-v0',
        'gamma': 0.95,  #before 0.95
        'max_minutes': 10,
        'max_episodes': 20,
        'goal_mean_100_reward': 3000  # to be determined properly
    }
    value_model_fn = lambda nS, nA: FCDuelingQ(nS, nA, hidden_dims=(512,128))
    value_optimizer_fn = lambda net, lr: optim.RMSprop(net.parameters(), lr=lr)
    value_optimizer_lr = 0.0005
    max_gradient_norm = float('inf')

    training_strategy_fn = lambda: EGreedyExpStrategy(init_epsilon=1.0,  
                                                      min_epsilon=0.3, 
                                                      decay_steps=100000)
    evaluation_strategy_fn = lambda: GreedyStrategy()

    # replay_buffer_fn = lambda: ReplayBuffer(max_size=10000, batch_size=64)
    # replay_buffer_fn = lambda: PrioritizedReplayBuffer(
    #     max_samples=10000, batch_size=64, rank_based=True, 
    #     alpha=0.6, beta0=0.1, beta_rate=0.99995)
    replay_buffer_fn = lambda: PrioritizedReplayBuffer(
        max_samples=20000, batch_size=64, rank_based=False,
        alpha=0.6, beta0=0.1, beta_rate=0.99995)
    n_warmup_batches = 5
    update_target_every_steps = 1
    tau = 0.1

    env_name, gamma, max_minutes, \
    max_episodes, goal_mean_100_reward = environment_settings.values()
    agent = PER(replay_buffer_fn, 
                value_model_fn, 
                value_optimizer_fn, 
                value_optimizer_lr,
                max_gradient_norm,
                training_strategy_fn,
                evaluation_strategy_fn,
                n_warmup_batches,
                update_target_every_steps,
                tau)

    make_env_fn, make_env_kargs = get_make_env_fn(env_name=env_name, num_drones=DEFAULT_AGENTS, seed=seed)
    result, final_eval_score, training_time, wallclock_time = agent.train(
        make_env_fn, make_env_kargs, seed, gamma, max_minutes, max_episodes, goal_mean_100_reward)
    per_results.append(result)
    if final_eval_score > best_eval_score:
        best_eval_score = final_eval_score
        best_agent = agent
per_results = np.array(per_results)
_ = BEEP()

best_agent.save_model()