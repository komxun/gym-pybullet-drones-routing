import os
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

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


dqn_results = []
best_agent, best_eval_score = None, float('-inf')
for seed in SEEDS:
    environment_settings = {
        'env_name': 'CartPole-v1',
        'gamma': 1.00,
        'max_minutes': 20,
        'max_episodes': 1000,
        'goal_mean_100_reward': 300
    }
    
    value_model_fn = lambda nS, nA: FCQ(nS, nA, hidden_dims=(512,128))
    value_optimizer_fn = lambda net, lr: optim.RMSprop(net.parameters(), lr=lr)
    value_optimizer_lr = 0.0005

    # training_strategy_fn = lambda: EGreedyStrategy(epsilon=0.5)
    # training_strategy_fn = lambda: EGreedyLinearStrategy(init_epsilon=1.0,
    #                                                      min_epsilon=0.3, 
    #                                                      max_steps=20000)
    # training_strategy_fn = lambda: SoftMaxStrategy(init_temp=1.0, 
    #                                                min_temp=0.1, 
    #                                                exploration_ratio=0.8, 
    #                                                max_steps=20000)
    training_strategy_fn = lambda: EGreedyExpStrategy(init_epsilon=1.0,  
                                                      min_epsilon=0.3, 
                                                      decay_steps=20000)
    evaluation_strategy_fn = lambda: GreedyStrategy()

    replay_buffer_fn = lambda: ReplayBuffer(max_size=50000, batch_size=64)
    n_warmup_batches = 5
    update_target_every_steps = 10

    env_name, gamma, max_minutes, \
    max_episodes, goal_mean_100_reward = environment_settings.values()
    agent = DQN(replay_buffer_fn,
                value_model_fn,
                value_optimizer_fn,
                value_optimizer_lr,
                training_strategy_fn,
                evaluation_strategy_fn,
                n_warmup_batches,
                update_target_every_steps)

    make_env_fn, make_env_kargs = get_make_env_fn(env_name=env_name)
    print(f"========training for seed #{seed}")
    result, final_eval_score, training_time, wallclock_time = agent.train(
        make_env_fn, make_env_kargs, seed, gamma, max_minutes, max_episodes, goal_mean_100_reward)
    dqn_results.append(result)
    if final_eval_score > best_eval_score:
        best_eval_score = final_eval_score
        best_agent = agent
dqn_results = np.array(dqn_results)
_ = BEEP()

#%%
# best_agent.demo_progression()

best_agent.demo_last()

#%% Plotting
nfq_root_dir = os.path.join(RESULTS_DIR, 'nfq')
nfq_x = np.load(os.path.join(nfq_root_dir, 'x.npy'))

nfq_max_r = np.load(os.path.join(nfq_root_dir, 'max_r.npy'))
nfq_min_r = np.load(os.path.join(nfq_root_dir, 'min_r.npy'))
nfq_mean_r = np.load(os.path.join(nfq_root_dir, 'mean_r.npy'))

nfq_max_s = np.load(os.path.join(nfq_root_dir, 'max_s.npy'))
nfq_min_s = np.load(os.path.join(nfq_root_dir, 'min_s.npy'))
nfq_mean_s = np.load(os.path.join(nfq_root_dir, 'mean_s.npy'))

nfq_max_t = np.load(os.path.join(nfq_root_dir, 'max_t.npy'))
nfq_min_t = np.load(os.path.join(nfq_root_dir, 'min_t.npy'))
nfq_mean_t = np.load(os.path.join(nfq_root_dir, 'mean_t.npy'))

nfq_max_sec = np.load(os.path.join(nfq_root_dir, 'max_sec.npy'))
nfq_min_sec = np.load(os.path.join(nfq_root_dir, 'min_sec.npy'))
nfq_mean_sec = np.load(os.path.join(nfq_root_dir, 'mean_sec.npy'))

nfq_max_rt = np.load(os.path.join(nfq_root_dir, 'max_rt.npy'))
nfq_min_rt = np.load(os.path.join(nfq_root_dir, 'min_rt.npy'))
nfq_mean_rt = np.load(os.path.join(nfq_root_dir, 'mean_rt.npy'))


dqn_max_t, dqn_max_r, dqn_max_s, \
    dqn_max_sec, dqn_max_rt = np.max(dqn_results, axis=0).T
dqn_min_t, dqn_min_r, dqn_min_s, \
    dqn_min_sec, dqn_min_rt = np.min(dqn_results, axis=0).T
dqn_mean_t, dqn_mean_r, dqn_mean_s, \
    dqn_mean_sec, dqn_mean_rt = np.mean(dqn_results, axis=0).T
dqn_x = np.arange(np.max((len(dqn_mean_s), len(nfq_mean_s))))


fig, axs = plt.subplots(5, 1, figsize=(15,30), sharey=False, sharex=True)

# NFQ
axs[0].plot(nfq_max_r, 'y', linewidth=1)
axs[0].plot(nfq_min_r, 'y', linewidth=1)
axs[0].plot(nfq_mean_r, 'y', label='NFQ', linewidth=2)
axs[0].fill_between(nfq_x, nfq_min_r, nfq_max_r, facecolor='y', alpha=0.3)

axs[1].plot(nfq_max_s, 'y', linewidth=1)
axs[1].plot(nfq_min_s, 'y', linewidth=1)
axs[1].plot(nfq_mean_s, 'y', label='NFQ', linewidth=2)
axs[1].fill_between(nfq_x, nfq_min_s, nfq_max_s, facecolor='y', alpha=0.3)

axs[2].plot(nfq_max_t, 'y', linewidth=1)
axs[2].plot(nfq_min_t, 'y', linewidth=1)
axs[2].plot(nfq_mean_t, 'y', label='NFQ', linewidth=2)
axs[2].fill_between(nfq_x, nfq_min_t, nfq_max_t, facecolor='y', alpha=0.3)

axs[3].plot(nfq_max_sec, 'y', linewidth=1)
axs[3].plot(nfq_min_sec, 'y', linewidth=1)
axs[3].plot(nfq_mean_sec, 'y', label='NFQ', linewidth=2)
axs[3].fill_between(nfq_x, nfq_min_sec, nfq_max_sec, facecolor='y', alpha=0.3)

axs[4].plot(nfq_max_rt, 'y', linewidth=1)
axs[4].plot(nfq_min_rt, 'y', linewidth=1)
axs[4].plot(nfq_mean_rt, 'y', label='NFQ', linewidth=2)
axs[4].fill_between(nfq_x, nfq_min_rt, nfq_max_rt, facecolor='y', alpha=0.3)

# DQN
axs[0].plot(dqn_max_r, 'b', linewidth=1)
axs[0].plot(dqn_min_r, 'b', linewidth=1)
axs[0].plot(dqn_mean_r, 'b--', label='DQN', linewidth=2)
axs[0].fill_between(dqn_x, dqn_min_r, dqn_max_r, facecolor='b', alpha=0.3)

axs[1].plot(dqn_max_s, 'b', linewidth=1)
axs[1].plot(dqn_min_s, 'b', linewidth=1)
axs[1].plot(dqn_mean_s, 'b--', label='DQN', linewidth=2)
axs[1].fill_between(dqn_x, dqn_min_s, dqn_max_s, facecolor='b', alpha=0.3)

axs[2].plot(dqn_max_t, 'b', linewidth=1)
axs[2].plot(dqn_min_t, 'b', linewidth=1)
axs[2].plot(dqn_mean_t, 'b--', label='DQN', linewidth=2)
axs[2].fill_between(dqn_x, dqn_min_t, dqn_max_t, facecolor='b', alpha=0.3)

axs[3].plot(dqn_max_sec, 'b', linewidth=1)
axs[3].plot(dqn_min_sec, 'b', linewidth=1)
axs[3].plot(dqn_mean_sec, 'b--', label='DQN', linewidth=2)
axs[3].fill_between(dqn_x, dqn_min_sec, dqn_max_sec, facecolor='b', alpha=0.3)

axs[4].plot(dqn_max_rt, 'b', linewidth=1)
axs[4].plot(dqn_min_rt, 'b', linewidth=1)
axs[4].plot(dqn_mean_rt, 'b--', label='DQN', linewidth=2)
axs[4].fill_between(dqn_x, dqn_min_rt, dqn_max_rt, facecolor='b', alpha=0.3)

# ALL
axs[0].set_title('Moving Avg Reward (Training)')
axs[1].set_title('Moving Avg Reward (Evaluation)')
axs[2].set_title('Total Steps')
axs[3].set_title('Training Time')
axs[4].set_title('Wall-clock Time')
plt.xlabel('Episodes')
axs[0].legend(loc='upper left')
plt.show()


dqn_root_dir = os.path.join(RESULTS_DIR, 'dqn')
not os.path.exists(dqn_root_dir) and os.makedirs(dqn_root_dir)

np.save(os.path.join(dqn_root_dir, 'x'), dqn_x)

np.save(os.path.join(dqn_root_dir, 'max_r'), dqn_max_r)
np.save(os.path.join(dqn_root_dir, 'min_r'), dqn_min_r)
np.save(os.path.join(dqn_root_dir, 'mean_r'), dqn_mean_r)

np.save(os.path.join(dqn_root_dir, 'max_s'), dqn_max_s)
np.save(os.path.join(dqn_root_dir, 'min_s'), dqn_min_s )
np.save(os.path.join(dqn_root_dir, 'mean_s'), dqn_mean_s)

np.save(os.path.join(dqn_root_dir, 'max_t'), dqn_max_t)
np.save(os.path.join(dqn_root_dir, 'min_t'), dqn_min_t)
np.save(os.path.join(dqn_root_dir, 'mean_t'), dqn_mean_t)

np.save(os.path.join(dqn_root_dir, 'max_sec'), dqn_max_sec)
np.save(os.path.join(dqn_root_dir, 'min_sec'), dqn_min_sec)
np.save(os.path.join(dqn_root_dir, 'mean_sec'), dqn_mean_sec)

np.save(os.path.join(dqn_root_dir, 'max_rt'), dqn_max_rt)
np.save(os.path.join(dqn_root_dir, 'min_rt'), dqn_min_rt)
np.save(os.path.join(dqn_root_dir, 'mean_rt'), dqn_mean_rt)