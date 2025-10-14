"""Test script for multiagent problems.

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
import pdb
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
from ray.rllib.agents import ppo, dqn, qmix
from ray.rllib.agents.ppo import PPOTrainer, PPOTFPolicy
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

def Retrieve_env_info():
    #### Unused env to extract the act and obs spaces ##########
    temp_env = AutoroutingMASAviary_discrete(num_drones=NUM_DRONES,
                                    freq = 120,
                                    aggregate_phy_steps=1,
                                    obs=OBS,
                                    act=ACT
                                    )

    observer_space = temp_env.observation_space[0]
    action_space = temp_env.action_space[0]
    temp_env.close()
    return (action_space, observer_space)

############################################################
if __name__ == "__main__":
    
    # file_loc = "./results/save-autorouting-mas-aviary-v0-2-cc-kin-autorouting-06.05.2025_16.25.42"
    # file_loc = "./results/save-autorouting-mas-aviary-v0-2-cc-kin-autorouting-07.08.2025_14.56.30"
    # file_loc = "./results/save-autorouting-mas-aviary-v0-2-cc-kin-autorouting-07.09.2025_15.03.13"
    # file_loc = "./results/save-autorouting-mas-aviary-v0-2-cc-kin-autorouting-07.09.2025_15.29.59" # bad
    file_loc = "./results/save-autorouting-mas-aviary-v0-2-cc-kin-autorouting-07.10.2025_15.44.42"
    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='Multi-agent reinforcement learning experiments script')
    parser.add_argument('--exp',    default=file_loc, type=str,       help='The experiment folder written as ./results/save-<env>-<num_drones>-<algo>-<obs>-<act>-<time_date>', metavar='')
    ARGS = parser.parse_args()

    #### Parameters to recreate the environment ################
    # NUM_DRONES = int(ARGS.exp.split("-")[2])
    NUM_DRONES = 2
    # OBS = ObservationType.KIN if ARGS.exp.split("-")[4] == 'kin' else ObservationType.RGB
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
        # OWN_OBS_VEC_SIZE = 12
        # OWN_OBS_VEC_SIZE = 1013
        OWN_OBS_VEC_SIZE = 127
    elif OBS == ObservationType.RGB:
        print("[ERROR] ObservationType.RGB for multi-agent systems not yet implemented")
        exit()
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

    #### Register the custom centralized critic model ##########
    ModelCatalog.register_custom_model("cc_model", CustomTorchCentralizedCriticModel)

    #### Register the environment ##############################
    temp_env_name = "this-aviary-v0"
    
    register_env(temp_env_name, lambda _: AutoroutingMASAviary_discrete(num_drones=NUM_DRONES,
                                                                freq = 120,
                                                                aggregate_phy_steps=1,
                                                                obs=OBS,
                                                                act=ACT,
                                                                )
                     )

    
    action_space, observer_space = Retrieve_env_info()
    #### Set up the trainer's config ###########################
    config = dqn.DEFAULT_CONFIG.copy() # For the default config, see github.com/ray-project/ray/blob/master/rllib/agents/trainer.py
    config = {
        "env": temp_env_name,
        "num_workers": 2, # How many environment workers that parellely collect samples from their own environment clone(s)
        "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")), # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0
        "batch_mode": "complete_episodes",
        "framework": "torch",
        "explore": False,
    }
    config["multiagent"] = { 
        "policies": {
            # uses shared policies
            "shared_policy": (None, observer_space, action_space, {})
        },
        "policy_mapping_fn": lambda agent_id: "shared_policy",  # All agents map to shared_policy
        # "observation_fn": central_critic_observer, # See rllib/evaluation/observation_function.py for more info
    }

    #### Restore agent #########################################
    agent = dqn.DQNTrainer(config=config)
    with open(ARGS.exp+'/checkpoint.txt', 'r+') as f:
        checkpoint = f.read()
    agent.restore(checkpoint)

    #### Extract and print policies ############################
    shared_policy = agent.get_policy("shared_policy")
    # print("action model 0", policy0.model.action_model)
    # print("value model 0", policy0.model.value_model)

    #### Create test environment ###############################
    # p.disconnect() # Closing all prior environment (important!)
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
    elif ACT in [ActionType.RPM, ActionType.DYN, ActionType.VEL]:
        action = {i: np.array([0, 0, 0, 0]) for i in range(NUM_DRONES)}
    elif ACT==ActionType.PID:
         action = {i: np.array([0, 0, 0]) for i in range(NUM_DRONES)}
    else:
        print("[ERROR] unknown ActionType")
        exit()
    
    for i in range(50): # Up to 6''
        p.setRealTimeSimulation(1)    
        start = time.time()  
        #### Deploy the policies ###################################
        epEnd = False
        count = 0
        # test_env.reset()
        obs = test_env.reset()
        temp = {}
        while not epEnd:
            # temp[0] = policy0.compute_single_action(np.hstack([action[1], obs[1], obs[0]])) # Counterintuitive order, check params.json
            # temp[1] = policy1.compute_single_action(np.hstack([action[0], obs[0], obs[1]]))

            action_1 = shared_policy.compute_single_action(obs[0])
            action_2 = shared_policy.compute_single_action(obs[1])

            # actions = {0: temp[0][0], 1: temp[1][0]}
            actions = {0: action_1[0], 1: action_2[0]}
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
