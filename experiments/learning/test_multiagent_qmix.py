"""Test script for multiagent problems.

IF RUNNING AS A SCRIPT, MAKE SURE TO CD TO gym-pybullet-drones-routing\experiments\learning
This scripts runs the best model found by one of the executions of `multiagent.py`


Example
-------
To run the script, type in a terminal:

    $ python test_multiagent.py --exp ./results/save-<env>-<num_drones>-<algo>-<obs>-<act>-<time_date>

"""
import os
import time
import argparse
from datetime import datetime
import math
import numpy as np
import pybullet as p
import pickle
import matplotlib.pyplot as plt
import gym
from gym import error, spaces, utils
from gym.utils import seeding
from gym.spaces import Box, Dict, Discrete
import torch
import torch.nn as nn
import ray
from ray import tune
from ray.tune import register_env
from ray.rllib.agents import qmix
from ray.rllib.examples.policy.random_policy import RandomPolicy
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.sample_batch import SampleBatch

from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.multi_agent_rl.AutoroutingMASAviary_discrete import AutoroutingMASAviary_discrete
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ActionType, ObservationType
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync


import shared_constants

OWN_OBS_VEC_SIZE = None # Modified at runtime
ACTION_VEC_SIZE = None # Modified at runtime

############################################################
if __name__ == "__main__":
 
    # file_loc = "./results/save-autorouting-mas-aviary-v0-4-QMIX-kin-autorouting-07.19.2025_14.28.21"
    file_loc = "./results/save-autorouting-mas-aviary-v0-4-QMIX-kin-autorouting-07.23.2025_11.25.52"
    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='Multi-agent reinforcement learning experiments script')
    parser.add_argument('--exp',    default=file_loc, type=str,       help='The experiment folder written as ./results/save-<env>-<num_drones>-<algo>-<obs>-<act>-<time_date>', metavar='')
    ARGS = parser.parse_args()

    #### Parameters to recreate the environment ################
    NUM_DRONES = 4
    OBS = ObservationType.KIN
    
    # Parse ActionType instance from file name
    # action_name = ARGS.exp.split("-")[5]
    action_name = 'autorouting'
    ACT = [action for action in ActionType if action.value == action_name]
    if len(ACT) != 1:
        raise AssertionError("Result file could have gotten corrupted. Extracted action type does not match any of the existing ones.")
    ACT = ACT.pop()

    #### Constants, and errors #################################
    if OBS == ObservationType.KIN:
        OWN_OBS_VEC_SIZE = 127
    else:
        print("[ERROR] unknown ObservationType")
        exit()
    if ACT in [ActionType.ONE_D_RPM, ActionType.ONE_D_DYN, ActionType.ONE_D_PID, ActionType.AUTOROUTING]:
        ACTION_VEC_SIZE = 1
    else:
        print("[ERROR] unknown ActionType")
        exit()

    #### Initialize Ray Tune ###################################
    ray.shutdown()
    ray.init(ignore_reinit_error=True)

    #### Unused env to extract the act and obs spaces ##########
    temp_env = AutoroutingMASAviary_discrete(num_drones=NUM_DRONES,
                                             freq = 120,
                                            aggregate_phy_steps=1,
                                            obs=OBS,
                                            act=ACT
                                            )

    agent_list = list(range(NUM_DRONES))
    obs_space = spaces.Tuple([temp_env.observation_space[i] for i in agent_list])
    act_space = spaces.Tuple([temp_env.action_space[i] for i in agent_list])
  
    #### Register the environment ##############################
    temp_env_name = "autorouting-mas-aviary-v0"

    grouping = {
        "group_1": agent_list,
    }
    
    register_env(temp_env_name, lambda _: AutoroutingMASAviary_discrete(num_drones=NUM_DRONES,
                                                                        freq = 120,
                                                                        aggregate_phy_steps=1,
                                                                        obs=OBS,
                                                                        act=ACT
                                                                        ).with_agent_groups(
                                                                            groups=grouping,
                                                                            obs_space=obs_space,
                                                                            act_space=act_space,
                                                                            )
                    )

    #### Set up the trainer's config ###########################
    config = qmix.DEFAULT_CONFIG.copy() # For the default config, see github.com/ray-project/ray/blob/master/rllib/agents/trainer.py
    config = {
        "env": temp_env_name,
        "num_workers": 1, # How many environment workers that parellely collect samples from their own environment clone(s)
        "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")), # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0
        "batch_mode": "complete_episodes",
        "framework": "torch",
        'buffer_size': 250,
        "explore": False,
    }
    
    #### Restore agent #########################################
    agent = qmix.QMixTrainer(config=config)
    with open(ARGS.exp+'/checkpoint.txt', 'r+') as f:
        checkpoint = f.read()
    agent.restore(checkpoint)



    #### Extract and print policies ############################
    # shared_policy = agent.get_policy("default_policy")
    policy_name = list(agent.config['multiagent']['policies'].keys())[0]
    print(f">>>>> Using policy: {policy_name}")
    shared_policy = agent.get_policy(policy_name)
    # print("action model 0", policy0.model.action_model)
    # print("value model 0", policy0.model.value_model)

    print("Check the policy :")
    print(shared_policy.model)
    input("Press Enter to continue...")

    #### Create test environment ###############################
    p.disconnect() # Closing all prior environment (important!)
    test_env = AutoroutingMASAviary_discrete(num_drones=NUM_DRONES,
                            obs=OBS,
                            act=ACT,
                            freq = 120,
                            aggregate_phy_steps=1,
                            gui=True,
                            record=False
                            )
    
    PYB_CLIENT = test_env.getPyBulletClient()
    
    #### Show, record a video, and log the model's performance #
    # obs = test_env.reset()
    logger = Logger(logging_freq_hz=int(test_env.SIM_FREQ/test_env.AGGR_PHY_STEPS),
                    num_drones=NUM_DRONES
                    )
    if ACT in [ActionType.ONE_D_RPM, ActionType.ONE_D_DYN, ActionType.ONE_D_PID, ActionType.AUTOROUTING]:
        action = {i: np.array([0]) for i in range(NUM_DRONES)}
    else:
        print("[ERROR] unknown ActionType")
        exit()
    
    for i in range(50): # Up to 6''
        p.setRealTimeSimulation(1)    
        start = time.time()  
        #### Deploy the policies ###################################
        epEnd = False
        count = 0

        obs = test_env.reset()
        temp = {}
        state_out = shared_policy.get_initial_state()

        while not epEnd:

            all_agent_obs = np.concatenate([obs[i] for i in range(len(obs))])
            action_tuple, state_out, info = shared_policy.compute_single_action(all_agent_obs, state=state_out)

            actions = {i: action_tuple[i] for i in range(len(obs))}
            print(f"Actions: {actions}")
            obs, reward, done, info = test_env.step(actions)
            
            if done['__all__']==True:
                print(f"========== EPISODE ENDED ==============")
                epEnd = True
            

            test_env.render()
            sync(count, start, test_env.TIMESTEP)
            count += 1
            # if OBS==ObservationType.KIN: 
            #     for j in range(NUM_DRONES):
            #         logger.log(drone=j,
            #                 timestamp=i/test_env.SIM_FREQ,
            #                 state= np.hstack([obs[j][0:3], np.zeros(4), obs[j][3:15], np.resize(action[j], (4))]),
            #                 control=np.zeros(12)
            #                 )
            
            # sync(i, start,test_env.TIMESTEP)
            # sync(np.floor(i*test_env.AGGR_PHY_STEPS), start, test_env.TIMESTEP)

        # if done["__all__"]: obs = test_env.reset() # OPTIONAL EPISODE HALT
    test_env.close()
    # -- save and plot UAV's state
    # logger.save_as_csv("ma") # Optional CSV save
    # logger.plot()

    #### Shut down Ray #########################################
    ray.shutdown()
