"""Learning script for multi-agent problems.

Agents are based on `ray[rllib]`'s implementation of QMIX.

Fixed version addressing the observation space issue.
"""
import os
import time
import argparse
from datetime import datetime
from sys import platform
import subprocess
import pdb
import math
import numpy as np
import pybullet as p
import pickle
import matplotlib.pyplot as plt
import gym
from gym import error, spaces, utils
from gym.utils import seeding
from gym.spaces import Box, Dict, Discrete, MultiDiscrete
import torch
import torch.nn as nn
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork
import ray
from ray import tune
from ray.tune.logger import DEFAULT_LOGGERS
from ray.tune import register_env
from ray.rllib.agents import ppo, dqn, qmix
from ray.rllib.agents.qmix import QMixTrainer
from ray.rllib.agents.ppo import PPOTrainer, PPOTFPolicy
from ray.rllib.agents.dqn import DQNTrainer
from ray.rllib.agents.dqn.dqn_torch_model import DQNTorchModel
from ray.rllib.examples.policy.random_policy import RandomPolicy
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.env.multi_agent_env import ENV_STATE

from gym_pybullet_drones.envs.multi_agent_rl.AutoroutingMASAviary_discrete import AutoroutingMASAviary_discrete
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ActionType, ObservationType
from gym_pybullet_drones.utils.Logger import Logger

import shared_constants

OWN_OBS_VEC_SIZE = None # Modified at runtime
ACTION_VEC_SIZE = None # Modified at runtime

if __name__ == "__main__":

    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='Multi-agent reinforcement learning experiments script')
    parser.add_argument('--num_drones',  default=2,                 type=int,                                                                 help='Number of drones (default: 2)', metavar='')
    parser.add_argument('--env',         default='autorouting-mas-aviary-v0',  type=str,             choices=['leaderfollower', 'flock', 'meetup'],      help='Task (default: leaderfollower)', metavar='')
    parser.add_argument('--obs',         default='kin',             type=ObservationType,                                                     help='Observation space (default: kin)', metavar='')
    parser.add_argument('--act',         default='autorouting',       type=ActionType,                                                          help='Action space (default: one_d_rpm)', metavar='')
    parser.add_argument('--algo',        default='cc',              type=str,             choices=['cc'],                                     help='MARL approach (default: cc)', metavar='')
    parser.add_argument('--workers',     default=9,                 type=int,                                                                 help='Number of RLlib workers (default: 0)', metavar='')        
    ARGS = parser.parse_args()

    #### Save directory ########################################
    filename = os.path.dirname(os.path.abspath(__file__))+'/results/save-'+ARGS.env+'-'+str(ARGS.num_drones)+'-'+ARGS.algo+'-'+ARGS.obs.value+'-'+ARGS.act.value+'-'+datetime.now().strftime("%m.%d.%Y_%H.%M.%S")
    if not os.path.exists(filename):
        os.makedirs(filename+'/')

    #### Print out current git commit hash #####################
    if platform == "linux" or platform == "darwin":
        git_commit = subprocess.check_output(["git", "describe", "--tags"]).strip()
        with open(filename+'/git_commit.txt', 'w+') as f:
            f.write(str(git_commit))

    #### Constants, and errors #################################
    if ARGS.obs==ObservationType.KIN:
        OWN_OBS_VEC_SIZE = 127 # 24 rays
    elif ARGS.obs==ObservationType.RGB:
        print("[ERROR] ObservationType.RGB for multi-agent systems not yet implemented")
        exit()
    else:
        print("[ERROR] unknown ObservationType")
        exit()
    if ARGS.act in [ActionType.ONE_D_RPM, ActionType.ONE_D_DYN, ActionType.ONE_D_PID, ActionType.AUTOROUTING]:
        ACTION_VEC_SIZE = 1
    else:
        print("[ERROR] unknown ActionType")
        exit()

    #### Initialize Ray Tune ###################################
    ray.shutdown()
    ray.init(ignore_reinit_error=True)

    #### Unused env to extract the act and obs spaces ##########
    temp_env = AutoroutingMASAviary_discrete(num_drones=ARGS.num_drones,
                                             freq = 120,
                                            aggregate_phy_steps=1,
                                            obs=ARGS.obs,
                                            act=ARGS.act
                                            )

    # For QMIX, we need to define observation and action spaces as tuples
    # QMIX expects Tuple observation spaces for grouped agents
    individual_obs_space = temp_env.observation_space[0]
    individual_act_space = temp_env.action_space[0]
    
    # Create grouped observation and action spaces for QMIX
    # Each group should have Tuple spaces
    # grouped_obs_space = spaces.Tuple([individual_obs_space] * ARGS.num_drones)
    # grouped_act_space = spaces.Tuple([individual_act_space] * ARGS.num_drones)

    # grouped_obs_space = spaces.Tuple([
    #     Dict({
    #         "obs": temp_env.observation_space[0],
    #     }),
    #     Dict({
    #         "obs": temp_env.observation_space[1],
    #     }),
    # ])
    grouped_obs_space = spaces.Tuple([
        temp_env.observation_space[0],
        temp_env.observation_space[1],
    ])
    grouped_act_space = spaces.Tuple([
        temp_env.action_space[0],
        temp_env.action_space[1],
    ])

    #### Register the environment ##############################
    temp_env_name = "autorouting-mas-aviary-v0"

    # Define agent groups for QMIX - all agents in one group
    grouping = {
        "group_1": list(range(ARGS.num_drones)),  # All agents in one group
    }

    # For QMIX, we MUST use agent groups with Tuple observation spaces
    register_env(temp_env_name, lambda _: AutoroutingMASAviary_discrete(num_drones=ARGS.num_drones,
                                                                        freq = 120,
                                                                        aggregate_phy_steps=1,
                                                                        obs=ARGS.obs,
                                                                        act=ARGS.act
                                                                        ).with_agent_groups(
                                                                            groups=grouping,
                                                                            obs_space=grouped_obs_space,
                                                                            act_space=grouped_act_space
                                                                        ))

    #### Set up the trainer's config ###########################
    config = qmix.DEFAULT_CONFIG.copy()
    config.update({
        "env": temp_env_name,
        "num_workers": 0 + ARGS.workers,
        "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
        "batch_mode": "complete_episodes",
        "framework": "torch",
    })
    
    #### Set up the multiagent params of the trainer's config ##
    config["multiagent"] = {
        "policies": {
            # QMIX uses grouped policies - one policy per group
            "group_1": (None, grouped_obs_space, grouped_act_space, {})
        },
        "policy_mapping_fn": lambda agent_id: "group_1",  # All agents map to group_1
    }

    

    #### Ray Tune stopping conditions ##########################
    stop = {
        "timesteps_total": int(1e6),
        # "timesteps_total": 120000, # 100000 ~= 10'
        # "episode_reward_mean": 100,
        # "training_iteration": 0,
    }
    
    results = tune.run(
        "QMIX",
        stop=stop,
        config=config,
        verbose=True,
        checkpoint_at_end=True,
        local_dir=filename,
    )

    #### Save agent ############################################
    checkpoints = results.get_trial_checkpoints_paths(trial=results.get_best_trial('episode_reward_mean',
                                                                                   mode='max'
                                                                                   ),
                                                      metric='episode_reward_mean'
                                                      )
    with open(filename+'/checkpoint.txt', 'w+') as f:
        f.write(checkpoints[0][0])

    #### Shut down Ray #########################################
    ray.shutdown()