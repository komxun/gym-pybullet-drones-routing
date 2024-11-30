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
import matplotlib.pyplot as plt
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
RESULTS_DIR = os.path.join('..', 'KOMSUN_results')
# SEEDS = (12, 34, 56, 78, 90)
SEEDS = (12, 34, 5)
# SEEDS = tuple(range(100))
# SEEDS = range(1,)

DEFAULT_AGENTS = 1


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
        'max_minutes': 60,
        'max_episodes': 100,
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
input("Press Enter to continue...")
# ddqn_root_dir = os.path.join(RESULTS_DIR, 'ddqn')
# ddqn_x = np.load(os.path.join(ddqn_root_dir, 'x.npy'))

# ddqn_max_r = np.load(os.path.join(ddqn_root_dir, 'max_r.npy'))
# ddqn_min_r = np.load(os.path.join(ddqn_root_dir, 'min_r.npy'))
# ddqn_mean_r = np.load(os.path.join(ddqn_root_dir, 'mean_r.npy'))

# ddqn_max_s = np.load(os.path.join(ddqn_root_dir, 'max_s.npy'))
# ddqn_min_s = np.load(os.path.join(ddqn_root_dir, 'min_s.npy'))
# ddqn_mean_s = np.load(os.path.join(ddqn_root_dir, 'mean_s.npy'))

# ddqn_max_t = np.load(os.path.join(ddqn_root_dir, 'max_t.npy'))
# ddqn_min_t = np.load(os.path.join(ddqn_root_dir, 'min_t.npy'))
# ddqn_mean_t = np.load(os.path.join(ddqn_root_dir, 'mean_t.npy'))

# ddqn_max_sec = np.load(os.path.join(ddqn_root_dir, 'max_sec.npy'))
# ddqn_min_sec = np.load(os.path.join(ddqn_root_dir, 'min_sec.npy'))
# ddqn_mean_sec = np.load(os.path.join(ddqn_root_dir, 'mean_sec.npy'))

# ddqn_max_rt = np.load(os.path.join(ddqn_root_dir, 'max_rt.npy'))
# ddqn_min_rt = np.load(os.path.join(ddqn_root_dir, 'min_rt.npy'))
# ddqn_mean_rt = np.load(os.path.join(ddqn_root_dir, 'mean_rt.npy'))

dueling_ddqn_max_t, dueling_ddqn_max_r, dueling_ddqn_max_s, \
dueling_ddqn_max_sec, dueling_ddqn_max_rt = np.max(dueling_ddqn_results, axis=0).T
dueling_ddqn_min_t, dueling_ddqn_min_r, dueling_ddqn_min_s, \
dueling_ddqn_min_sec, dueling_ddqn_min_rt = np.min(dueling_ddqn_results, axis=0).T
dueling_ddqn_mean_t, dueling_ddqn_mean_r, dueling_ddqn_mean_s, \
dueling_ddqn_mean_sec, dueling_ddqn_mean_rt = np.mean(dueling_ddqn_results, axis=0).T
dueling_ddqn_x = np.arange(np.max(
    (len(dueling_ddqn_mean_s), len(dueling_ddqn_mean_s))))


#%% PLOT
fig, axs = plt.subplots(5, 1, figsize=(15,30), sharey=False, sharex=True)


# Dueling DDQN
axs[0].plot(dueling_ddqn_max_r, 'r', linewidth=1)
axs[0].plot(dueling_ddqn_min_r, 'r', linewidth=1)
axs[0].plot(dueling_ddqn_mean_r, 'r:', label='Dueling DDQN', linewidth=2)
axs[0].fill_between(
    dueling_ddqn_x, dueling_ddqn_min_r, dueling_ddqn_max_r, facecolor='r', alpha=0.3)

axs[1].plot(dueling_ddqn_max_s, 'r', linewidth=1)
axs[1].plot(dueling_ddqn_min_s, 'r', linewidth=1)
axs[1].plot(dueling_ddqn_mean_s, 'r:', label='Dueling DDQN', linewidth=2)
axs[1].fill_between(
    dueling_ddqn_x, dueling_ddqn_min_s, dueling_ddqn_max_s, facecolor='r', alpha=0.3)

axs[2].plot(dueling_ddqn_max_t, 'r', linewidth=1)
axs[2].plot(dueling_ddqn_min_t, 'r', linewidth=1)
axs[2].plot(dueling_ddqn_mean_t, 'r:', label='Dueling DDQN', linewidth=2)
axs[2].fill_between(
    dueling_ddqn_x, dueling_ddqn_min_t, dueling_ddqn_max_t, facecolor='r', alpha=0.3)

axs[3].plot(dueling_ddqn_max_sec, 'r', linewidth=1)
axs[3].plot(dueling_ddqn_min_sec, 'r', linewidth=1)
axs[3].plot(dueling_ddqn_mean_sec, 'r:', label='Dueling DDQN', linewidth=2)
axs[3].fill_between(
    dueling_ddqn_x, dueling_ddqn_min_sec, dueling_ddqn_max_sec, facecolor='r', alpha=0.3)

axs[4].plot(dueling_ddqn_max_rt, 'r', linewidth=1)
axs[4].plot(dueling_ddqn_min_rt, 'r', linewidth=1)
axs[4].plot(dueling_ddqn_mean_rt, 'r:', label='Dueling DDQN', linewidth=2)
axs[4].fill_between(
    dueling_ddqn_x, dueling_ddqn_min_rt, dueling_ddqn_max_rt, facecolor='r', alpha=0.3)

# ALL
axs[0].set_title('Moving Avg Reward (Training)')
axs[1].set_title('Moving Avg Reward (Evaluation)')
axs[2].set_title('Total Steps')
axs[3].set_title('Training Time')
axs[4].set_title('Wall-clock Time')
plt.xlabel('Episodes')
axs[0].legend(loc='upper left')
plt.show()

dueling_ddqn_root_dir = os.path.join(RESULTS_DIR, 'dueling_ddqn')
not os.path.exists(dueling_ddqn_root_dir) and os.makedirs(dueling_ddqn_root_dir)

np.save(os.path.join(dueling_ddqn_root_dir, 'x'), dueling_ddqn_x)

np.save(os.path.join(dueling_ddqn_root_dir, 'max_r'), dueling_ddqn_max_r)
np.save(os.path.join(dueling_ddqn_root_dir, 'min_r'), dueling_ddqn_min_r)
np.save(os.path.join(dueling_ddqn_root_dir, 'mean_r'), dueling_ddqn_mean_r)

np.save(os.path.join(dueling_ddqn_root_dir, 'max_s'), dueling_ddqn_max_s)
np.save(os.path.join(dueling_ddqn_root_dir, 'min_s'), dueling_ddqn_min_s )
np.save(os.path.join(dueling_ddqn_root_dir, 'mean_s'), dueling_ddqn_mean_s)

np.save(os.path.join(dueling_ddqn_root_dir, 'max_t'), dueling_ddqn_max_t)
np.save(os.path.join(dueling_ddqn_root_dir, 'min_t'), dueling_ddqn_min_t)
np.save(os.path.join(dueling_ddqn_root_dir, 'mean_t'), dueling_ddqn_mean_t)

np.save(os.path.join(dueling_ddqn_root_dir, 'max_sec'), dueling_ddqn_max_sec)
np.save(os.path.join(dueling_ddqn_root_dir, 'min_sec'), dueling_ddqn_min_sec)
np.save(os.path.join(dueling_ddqn_root_dir, 'mean_sec'), dueling_ddqn_mean_sec)

np.save(os.path.join(dueling_ddqn_root_dir, 'max_rt'), dueling_ddqn_max_rt)
np.save(os.path.join(dueling_ddqn_root_dir, 'min_rt'), dueling_ddqn_min_rt)
np.save(os.path.join(dueling_ddqn_root_dir, 'mean_rt'), dueling_ddqn_mean_rt)


#%%

# q_values = dueling_ddqn_agents[best_dueling_ddqn_agent_key].online_model(state).detach().cpu().numpy()[0]
# print(f"\nq values = {q_values}\n")
# q_s = q_values
# v_s = q_values.mean()
# a_s = q_values - q_values.mean()

# plt.bar(('a1 (accelerate)','a2 (decelerate)', 'a3 (stop and hove)', 'a4 (follow global path)', 'a5 (follow local path #1)', \
#          'a6 (follow local path #2)', 'a7 (follow local path #3)', 'a8 (follow local path #4)', 'a9 (follow local path #5)',\
#          'a10 (follow local path #6)', 'a11 (follow local path #7)', 'a12 (follow local path #8)'), q_s)
# plt.xlabel('Action')
# plt.ylabel('Estimate')
# plt.title("Action-value function, Q(" + str(np.round(state,2)) + ")")
# plt.show()

# %% Violin plot
env = make_env_fn(**make_env_kargs, seed=123, monitor_mode='evaluation')

states = []
for agent in dueling_ddqn_agents.values():
    for episode in range(100):
        state, done = env.reset(), False
        while not done:
            states.append(state)
            action = agent.evaluation_strategy.select_action(agent.online_model, state)
            state, _, done, _ = env.step(action)
env.close()
del env

x = np.array(states)[:,0]
xd = np.array(states)[:,1]
a = np.array(states)[:,2]
ad = np.array(states)[:,3]

parts = plt.violinplot((x, xd, a, ad), 
                       vert=False, showmeans=False, showmedians=False, showextrema=False)

colors = ['red','green','yellow','blue']
for i, pc in enumerate(parts['bodies']):
    pc.set_facecolor(colors[i])
    pc.set_edgecolor(colors[i])
    pc.set_alpha(0.5)

plt.yticks(range(1,5), ["cart position", "cart velocity", "pole angle", "pole velocity"])
plt.yticks(rotation=45)
plt.title('Range of state-variable values for ' + str(
    dueling_ddqn_agents[best_dueling_ddqn_agent_key].__class__.__name__))

plt.show()