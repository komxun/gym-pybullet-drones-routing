"""Resume training script for multi-agent problems.

This script resumes training from a saved checkpoint.

Example
-------
To run the script, type in a terminal:

    $ python resume_multiagent.py --checkpoint_path <path_to_checkpoint> --num_drones <num_drones> --env <env> --obs <ObservationType> --act <ActionType> --algo <alg> --workers <num_workers>

Notes
-----
Check Ray's status at:

    http://127.0.0.1:8265

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

############################################################
############################################################
def central_critic_observer(agent_obs, **kw):
    new_obs = {
        0: {
            "obs": agent_obs
        },
        1: {
            "obs": agent_obs,
        },
    }
    return new_obs

############################################################
if __name__ == "__main__":

    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='Resume multi-agent reinforcement learning training')
    parser.add_argument('--num_drones',      default=4,          type=int,                                                                 help='Number of drones (default: 4)', metavar='')
    parser.add_argument('--env',             default='autorouting-mas-aviary-v0',  type=str,             choices=['leaderfollower', 'flock', 'meetup'],      help='Task (default: leaderfollower)', metavar='')
    parser.add_argument('--obs',             default='kin',      type=ObservationType,                                                     help='Observation space (default: kin)', metavar='')
    parser.add_argument('--act',             default='autorouting', type=ActionType,                                                       help='Action space (default: autorouting)', metavar='')
    parser.add_argument('--algo',            default='cc',     type=str,             choices=['cc'],                                     help='MARL approach (default: QMIX)', metavar='')
    parser.add_argument('--workers',         default=1,          type=int,                                                                 help='Number of RLlib workers (default: 1)', metavar='')
    ARGS = parser.parse_args()

    #### Save directory ########################################
    trained_filename = 'save-autorouting-mas-aviary-v0-4-cc-kin-autorouting-07.18.2025_17.15.15'  #1 M Training (Almost good!)
    filename = os.path.dirname(os.path.abspath(__file__)) + '/results/' + trained_filename
    if not os.path.exists(filename):
        os.makedirs(filename+'/')
    else:
        print(f"Loading {trained_filename}")

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

    agent_list = list(range(ARGS.num_drones))
    obs_space = spaces.Tuple([temp_env.observation_space[i] for i in agent_list])
    act_space = spaces.Tuple([temp_env.action_space[i] for i in agent_list])
  
    #### Register the environment ##############################
    temp_env_name = "autorouting-mas-aviary-v0"

    grouping = {
        "group_1": agent_list,
    }

    register_env(temp_env_name, lambda _: AutoroutingMASAviary_discrete(num_drones=ARGS.num_drones,
                                                                        freq = 120,
                                                                        aggregate_phy_steps=1,
                                                                        obs=ARGS.obs,
                                                                        act=ARGS.act
                                                                        ).with_agent_groups(
                                                                            groups=grouping,
                                                                            obs_space=obs_space,
                                                                            act_space=act_space,
                                                                            )
                    )
    
    #### Set up the trainer's config ###########################
    config = qmix.DEFAULT_CONFIG.copy()
    config = {
        "env": temp_env_name,
        "num_workers": 0 + ARGS.workers,
        "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
        "batch_mode": "complete_episodes",
        "framework": "torch",
        'buffer_size': 250
    }
    
    config["multiagent"] = {
        "policies": {
            "shared_policy": (None, obs_space, act_space, {}),
        },
        "policy_mapping_fn": lambda agent_id: "shared_policy",
    }

    print(f"[INFO] Results will be saved to: {filename}")
    #### Ray Tune stopping conditions ##########################
    stop = {
        "timesteps_total": int(1e6),
    }

    #### Resume training from checkpoint #######################
    results = tune.run(
        "QMIX",
        stop=stop,
        config=config,
        verbose=True,
        checkpoint_at_end=True,
        local_dir=filename,
        resume=True,
    )

    print("Training resumed and completed!")

    #### Save new checkpoint information #######################
    checkpoints = results.get_trial_checkpoints_paths(trial=results.get_best_trial('episode_reward_mean',
                                                                                   mode='max'
                                                                                   ),
                                                      metric='episode_reward_mean'
                                                      )
    with open(filename+'/checkpoint.txt', 'w+') as f:
        f.write(checkpoints[0][0])

    print(f"[INFO] New checkpoint saved to: {checkpoints[0][0]}")

    #### Shut down Ray #########################################
    ray.shutdown()