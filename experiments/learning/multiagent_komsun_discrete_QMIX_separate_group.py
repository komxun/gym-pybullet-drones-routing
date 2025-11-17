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
# import time
import argparse
from datetime import datetime
from sys import platform
import subprocess
from gym import spaces
import ray
from ray import tune
# from ray.tune.logger import DEFAULT_LOGGERS
from ray.tune import register_env
from ray.rllib.agents import qmix

from gym_pybullet_drones.envs.multi_agent_rl.AutoroutingMASAviary_discrete import AutoroutingMASAviary_discrete
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ActionType, ObservationType
# from gym_pybullet_drones.utils.Logger import Logger

OWN_OBS_VEC_SIZE = None # Modified at runtime
ACTION_VEC_SIZE = None # Modified at runtime

#### Useful links ##########################################
# Workflow: github.com/ray-project/ray/blob/master/doc/source/rllib-training.rst
# ENV_STATE example: github.com/ray-project/ray/blob/master/rllib/examples/env/two_step_game.py
# Competing policies example: github.com/ray-project/ray/blob/master/rllib/examples/rock_paper_scissors_multiagent.py

############################################################

############################################################
if __name__ == "__main__":

    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='Multi-agent reinforcement learning experiments script')
    parser.add_argument('--num_drones',  default=8,                 type=int,                                                                 help='Number of drones (default: 2)', metavar='')
    parser.add_argument('--env',         default='autorouting-mas-aviary-v0',  type=str,             choices=['leaderfollower', 'flock', 'meetup'],      help='Task (default: leaderfollower)', metavar='')
    parser.add_argument('--obs',         default='kin',             type=ObservationType,                                                     help='Observation space (default: kin)', metavar='')
    parser.add_argument('--act',         default='autorouting',       type=ActionType,                                                          help='Action space (default: one_d_rpm)', metavar='')
    parser.add_argument('--algo',        default='QMIX',              type=str,             choices=['cc'],                                     help='MARL approach (default: cc)', metavar='')
    parser.add_argument('--workers',     default= 1,                 type=int,                                                                 help='Number of RLlib workers (default: 0)', metavar='')        
    ARGS = parser.parse_args()

    #### Save directory ########################################
    filename = os.path.dirname(os.path.abspath(__file__))+'/results/separate-'+ARGS.env+'-'+str(ARGS.num_drones)+'-'+ARGS.algo+'-'+datetime.now().strftime("%m.%d.%Y_%H.%M.%S")
    if not os.path.exists(filename):
        os.makedirs(filename+'/')

    #### Print out current git commit hash #####################
    if platform == "linux" or platform == "darwin":
        git_commit = subprocess.check_output(["git", "describe", "--tags"]).strip()
        with open(filename+'/git_commit.txt', 'w+') as f:
            f.write(str(git_commit))

    FREQ = 10 # Hz

    #### Constants, and errors #################################
    # if ARGS.obs==ObservationType.KIN:
    #     OWN_OBS_VEC_SIZE = 127 # 24 rays
    #     # OWN_OBS_VEC_SIZE = 133 # 24 rays

    # elif ARGS.obs==ObservationType.RGB:
    #     print("[ERROR] ObservationType.RGB for multi-agent systems not yet implemented")
    #     exit()
    # else:
    #     print("[ERROR] unknown ObservationType")
    #     exit()
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

    #### Unused env to extract the act and obs spaces ##########
    temp_env = AutoroutingMASAviary_discrete(num_drones=ARGS.num_drones,
                                             freq = FREQ,
                                            aggregate_phy_steps=1,
                                            obs=ARGS.obs,
                                            act=ARGS.act
                                            )

    agent_list = list(range(ARGS.num_drones))
    # obs_space = spaces.Tuple([temp_env.observation_space[i] for i in agent_list])
    # act_space = spaces.Tuple([temp_env.action_space[i] for i in agent_list])
    obs_space = spaces.Tuple([temp_env.observation_space[0]])
    act_space = spaces.Tuple([temp_env.action_space[0]])
  
    #### Register the environment ##############################
    temp_env_name = "autorouting-mas-aviary-v0"

    # grouping = {
    #     "group_1": agent_list,
    # }
    grouping = {f"group_{i}": [i] for i in range(ARGS.num_drones)}
    register_env(temp_env_name, lambda _: AutoroutingMASAviary_discrete(num_drones=ARGS.num_drones,
                                                                        freq = FREQ,
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
    # Check more config option: https://github.com/ray-project/ray/blob/releases/1.11.0/rllib/agents/trainer.py
    config = qmix.DEFAULT_CONFIG.copy() # For the default config, see https://github.com/ray-project/ray/blob/releases/1.1.0/rllib/agents/qmix/qmix.py
    config = {
        "env": temp_env_name,
        "num_workers": ARGS.workers, # How many environment workers that parellely collect samples from their own environment clone(s)
        "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")), # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0
        "batch_mode": "complete_episodes",
        "framework": "torch",
        'buffer_size': 10000,
        
        "train_batch_size": 64,
        "no_done_at_end": True, # MUST SET TO TRUE
        #----- Algorithm related settings
        "mixer": "qmix",        # either "qmix" or "vdn" or None (default: qmix)
        "mixing_embed_dim": 32, # Size of the mixing network embedding (default: 32)
        "double_q": True,       # Whether to use Double_Q learning (default: True)
        "gamma": 0.7,
        # #------ Evaluation ------
        # "evaluation_interval": 10,  # evaluate with every x training iterations
        # "evaluation_duration": 100, # default unit is episodes
        # "evaluation_config":{
        #     "explore": False,
        # }
    }

    config["exploration_config"] = {
        "type": "EpsilonGreedy",  # The Exploration class to use.
        # Config for the Exploration class' constructor:
        "initial_epsilon": 1.0,
        "final_epsilon": 0.3,  # (default: 0.02)
        "epsilon_timesteps": 250000,  # Timesteps over which to anneal epsilon. (default: 10000)

        # For soft_q, use:
        # "exploration_config" = {
        #   "type": "SoftQ"
        #   "temperature": [float, e.g. 1.0]
        # }
    }
    
    config["multiagent"] = {
        "policies": {
            "shared_policy": (None, obs_space, act_space, {}),
        },
        "policy_mapping_fn": lambda agent_id: "shared_policy",  # agent_id will be "group_1"
        'replay_mode': 'independent',  # 'lockstep' or 'independent
        # When replay_mode=lockstep, RLlib will replay all the agent transitions at a particular timestep together in a batch. 
        # This allows the policy to implement differentiable shared computations between agents it controls at that timestep. 
        # When replay_mode=independent, transitions are replayed independently per policy.
    }
    
    #### Ray Tune stopping conditions ##########################
    stop = {
        "timesteps_total": int(1e6),
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
