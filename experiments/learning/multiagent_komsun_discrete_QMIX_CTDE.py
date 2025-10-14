"""Learning script for multi-agent problems.

Agents are based on `ray[rllib]`'s implementation of PPO and use a custom centralized critic.

Example
-------
To run the script, type in a terminal:

    $ python multiagent.py --num_drones <num_drones> --env <env> --obs <ObservationType> --act <ActionType> --algo <alg> --workers <num_workers>

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
from ray.rllib.agents.qmix import QMixTrainer, qmix
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.sample_batch import SampleBatch

from gym_pybullet_drones.envs.multi_agent_rl.AutoroutingMASAviary_discrete import AutoroutingMASAviary_discrete
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ActionType, ObservationType
from gym_pybullet_drones.utils.Logger import Logger

import shared_constants

OWN_OBS_VEC_SIZE = None # Modified at runtime
ACTION_VEC_SIZE = None # Modified at runtime

#### Useful links ##########################################
# Workflow: github.com/ray-project/ray/blob/master/doc/source/rllib-training.rst
# ENV_STATE example: github.com/ray-project/ray/blob/master/rllib/examples/env/two_step_game.py
# Competing policies example: github.com/ray-project/ray/blob/master/rllib/examples/rock_paper_scissors_multiagent.py

############################################################
###############################################################
class CustomTorchCentralizedCriticModel(TorchModelV2, nn.Module):
    """Multi-agent model that implements a centralized value function.

    It assumes the observation is a dict with 'own_obs' and 'opponent_obs', the
    former of which can be used for computing actions (i.e., decentralized
    execution), and the latter for optimization (i.e., centralized learning).

    This model has two parts:
    - An action model that looks at just 'own_obs' to compute actions
    - A value model that also looks at the 'opponent_obs' / 'opponent_action'
      to compute the value (it does this by using the 'obs_flat' tensor).
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        # Action model gets only own_obs (i.e. decentralized execution)
        self.action_model = FullyConnectedNetwork(
                                                  Box(low=-1, high=1, shape=(OWN_OBS_VEC_SIZE, )), 
                                                  action_space,
                                                  num_outputs,
                                                  model_config,
                                                  name + "_action"
                                                  )
        # Value model gets full obs (own + opponent), including opponent action
        self.value_model = FullyConnectedNetwork(
                                                 obs_space, 
                                                 action_space,
                                                 1, 
                                                 model_config, 
                                                 name + "_vf"
                                                 )
        self._model_in = None

    def forward(self, input_dict, state, seq_lens):
        self._model_in = [input_dict["obs_flat"], state, seq_lens]
        return self.action_model({"obs": input_dict["obs"]["own_obs"]}, state, seq_lens)

    def value_function(self):
        value_out, _ = self.value_model({"obs": self._model_in[0]}, self._model_in[1], self._model_in[2])
        return torch.reshape(value_out, [-1])
############################################################

############################################################
def central_critic_observer(agent_obs, **kw):

    print(f"Agent_obs is {agent_obs}")
    new_obs = {
        0: {
            "own_obs": agent_obs["group_1"][0],
            "opponent_obs": agent_obs["group_1"][1],
            "opponent_action": np.zeros(ACTION_VEC_SIZE), # Filled in by FillInActions
        },
        1: {
            "own_obs": agent_obs["group_1"][1],
            "opponent_obs": agent_obs["group_1"][0],
            "opponent_action": np.zeros(ACTION_VEC_SIZE), # Filled in by FillInActions
        },
    }
    return new_obs

############################################################
if __name__ == "__main__":

    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='Multi-agent reinforcement learning experiments script')
    parser.add_argument('--num_drones',  default=2,                 type=int,                                                                 help='Number of drones (default: 2)', metavar='')
    parser.add_argument('--env',         default='autorouting-mas-aviary-v0',  type=str,             choices=['leaderfollower', 'flock', 'meetup'],      help='Task (default: leaderfollower)', metavar='')
    parser.add_argument('--obs',         default='kin',             type=ObservationType,                                                     help='Observation space (default: kin)', metavar='')
    parser.add_argument('--act',         default='autorouting',       type=ActionType,                                                          help='Action space (default: one_d_rpm)', metavar='')
    parser.add_argument('--algo',        default='QMIX',              type=str,             choices=['cc'],                                     help='MARL approach (default: cc)', metavar='')
    parser.add_argument('--workers',     default= 0,                 type=int,                                                                 help='Number of RLlib workers (default: 0)', metavar='')        
    ARGS = parser.parse_args()

    #### Save directory ########################################
    # filename = os.path.dirname(os.path.abspath(__file__))+'/results/save-'+ARGS.env+'-'+str(ARGS.num_drones)+'-'+ARGS.algo+'-'+ARGS.obs.value+'-'+ARGS.act.value+'-'+datetime.now().strftime("%m.%d.%Y_%H.%M.%S")
    filename = os.path.dirname(os.path.abspath(__file__))+'/results/try-'+ARGS.env+'-'+str(ARGS.num_drones)+'-'+ARGS.algo+'-'+ARGS.obs.value+'-'+ARGS.act.value+'-'+datetime.now().strftime("%m.%d.%Y_%H.%M.%S")

    if not os.path.exists(filename):
        os.makedirs(filename+'/')

    #### Constants, and errors #################################
    if ARGS.obs==ObservationType.KIN:
        OWN_OBS_VEC_SIZE = 127 # 24 rays
        # OWN_OBS_VEC_SIZE = 133 # 24 rays
    else:
        print("[ERROR] unknown ObservationType")
        exit()
    if ARGS.act in [ActionType.ONE_D_RPM, ActionType.ONE_D_DYN, ActionType.ONE_D_PID, ActionType.AUTOROUTING]:
        ACTION_VEC_SIZE = 1
    else:
        print("[ERROR] unknown ActionType")
        exit()

    #### Uncomment to debug slurm scripts ######################
    # exit()

    #### Initialize Ray Tune ###################################
    ray.shutdown()
    ray.init(ignore_reinit_error=True)

    #### Register the custom centralized critic model ##########
    ModelCatalog.register_custom_model("cc_model", CustomTorchCentralizedCriticModel)

    #### Unused env to extract the act and obs spaces ##########
    temp_env = AutoroutingMASAviary_discrete(num_drones=ARGS.num_drones,
                                             freq = 120,
                                            aggregate_phy_steps=1,
                                            obs=ARGS.obs,
                                            act=ARGS.act
                                            )

    agent_list = list(range(ARGS.num_drones))

    temp_obs = {
            "own_obs": temp_env.observation_space[0],
            "opponent_obs": temp_env.observation_space[0],
            "opponent_action": np.zeros(ACTION_VEC_SIZE), # Filled in by FillInActions
        }


    all_obs_space = spaces.Tuple([temp_env.observation_space[i] for i in agent_list])
    all_act_space = spaces.Tuple([temp_env.action_space[i] for i in agent_list])
    obs_space = spaces.Tuple([temp_env.observation_space[0]])
    act_space = spaces.Tuple([temp_env.action_space[0]])

    observer_space = spaces.Tuple([
        spaces.Dict({
        "own_obs": temp_env.observation_space[0],
        "opponent_obs": temp_env.observation_space[1],
        "opponent_action":temp_env.action_space[1],
        }),
        spaces.Dict({
        "own_obs": temp_env.observation_space[1],
        "opponent_obs": temp_env.observation_space[0],
        "opponent_action":temp_env.action_space[0],    
        }),
    ])
  
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
                                                                            obs_space=observer_space,
                                                                            act_space=all_act_space,
                                                                            )
                    )
    
    #### Set up the trainer's config ###########################
    config = qmix.DEFAULT_CONFIG.copy() # For the default config, see github.com/ray-project/ray/blob/master/rllib/agents/trainer.py
    config = {
        "env": temp_env_name,
        "num_workers": 0 + ARGS.workers, # How many environment workers that parellely collect samples from their own environment clone(s)
        "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")), # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0
        "batch_mode": "complete_episodes",
        "framework": "torch",
        'buffer_size': 250,
        "gamma": 0.8,
    }

    #### Set up the model parameters of the trainer's config ###
    config["model"] = { 
        "custom_model": "cc_model",
    }
    
    config["multiagent"] = {
        "policies": {
            "polgroup_1": (None, observer_space, all_act_space, {}),
        },
        "policy_mapping_fn": lambda agent_id: f"pol{agent_id}",  # agent_id will be "group_1"
        "observation_fn": central_critic_observer, # See /rllib/evaluation/observation_function.py for more info
    }
    
    # print("=========== Check the following parameters==========")
    # policy_dict = {f"pol{i}": (None, obs_space, act_space, {"agent_id": i}) for i in range(ARGS.num_drones)}
    # policy_dict = {f"pol{i}": (None, obs_space[i], act_space[i], {}) for i in range(ARGS.num_drones)}
    # input("Press Enter to start training . . .")
    # config["multiagent"] = { 
    #     "policies": policy_dict,
    #     "policy_mapping_fn": lambda agent_id: f"pol{agent_id}", # # Function mapping agent ids to policy ids
    #     # "observation_fn": central_critic_observer, # See rllib/evaluation/observation_function.py for more info
    # }
    #### Ray Tune stopping conditions ##########################
    stop = {
        "timesteps_total": int(7e5),
    }
    results = tune.run(
        "QMIX",
        stop=stop,
        config=config,
        verbose=True,
        checkpoint_at_end=True,
        local_dir=filename,
    )

    # print("Best config: ", results.get_best_config(metric="mean_loss", mode="min"))

    # check_learning_achieved(results, 1.0)

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
