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
import itertools
import argparse
import numpy as np
import pybullet as p
from gym import spaces
import ray
from ray.tune import register_env
from ray.rllib.agents import qmix

from gym_pybullet_drones.envs.multi_agent_rl.AutoroutingMASAviary_discrete import AutoroutingMASAviary_discrete
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ActionType, ObservationType
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync

OWN_OBS_VEC_SIZE = None # Modified at runtime
ACTION_VEC_SIZE = None # Modified at runtime

############################################################
if __name__ == "__main__":



    file_loc = "./results/separate-autorouting-mas-aviary-v0-8-QMIX-10.16.2025_10.28.50"  # Very Good (10 Hz)
    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='Multi-agent reinforcement learning experiments script')
    parser.add_argument('--exp',    default=file_loc, type=str,       help='The experiment folder written as ./results/save-<env>-<num_drones>-<algo>-<obs>-<act>-<time_date>', metavar='')
    ARGS = parser.parse_args()

    #### Parameters to recreate the environment ################
    NUM_DRONES = 8
    OBS = ObservationType.KIN
    
    # Parse ActionType instance from file name
    # action_name = ARGS.exp.split("-")[5]
    action_name = 'autorouting'
    ACT = [action for action in ActionType if action.value == action_name]
    if len(ACT) != 1:
        raise AssertionError("Result file could have gotten corrupted. Extracted action type does not match any of the existing ones.")
    ACT = ACT.pop()

    #### Constants #################################
    ACTION_VEC_SIZE = 1

    #### Initialize Ray Tune ###################################
    ray.shutdown()
    ray.init(ignore_reinit_error=True)

    #### Unused env to extract the act and obs spaces ##########
    temp_env = AutoroutingMASAviary_discrete(num_drones=NUM_DRONES,
                                             freq = 10,
                                            aggregate_phy_steps=1,
                                            obs=OBS,
                                            act=ACT
                                            )
    
    OWN_OBS_VEC_SIZE = temp_env.observation_space[0].shape[0]

    agent_list = list(range(NUM_DRONES))
    obs_space = spaces.Tuple([temp_env.observation_space[0]])
    act_space = spaces.Tuple([temp_env.action_space[0]])
    #### Register the environment ##############################
    temp_env_name = "autorouting-mas-aviary-v0"

    grouping = {f"group_{i}": [i] for i in range(NUM_DRONES)}

    
    register_env(temp_env_name, lambda _: AutoroutingMASAviary_discrete(num_drones=NUM_DRONES,
                                                                        freq = 10,
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
        'buffer_size': 10000,
        "explore": False,
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
    config["multiagent"] = {
        "policies": {
            "shared_policy": (None, obs_space, act_space, {}),
        },
        "policy_mapping_fn": lambda agent_id: "shared_policy",  # agent_id will be "group_1"
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
    # input("Press Enter to continue...")

    #### Create test environment ###############################
    p.disconnect() # Closing all prior environment (important!)
    test_env = AutoroutingMASAviary_discrete(num_drones=NUM_DRONES,
                            obs=OBS,
                            act=ACT,
                            freq = 10,
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
        initial_state = shared_policy.get_initial_state()
        agent_ids = list(range(NUM_DRONES))
        agent_states = {i: initial_state for i in agent_ids}  # init hidden state per agent
        actions = {}
        p.resetDebugVisualizerCamera(cameraDistance=24, cameraYaw=0, cameraPitch=-80, cameraTargetPosition=[0,0,0])
        
        while not epEnd:

            for agent_id in range(NUM_DRONES):
                obs_i = obs[agent_id]
                state_i = agent_states[agent_id]
                action, new_state, _ = shared_policy.compute_single_action(obs_i, state=state_i)
                actions[agent_id] = action[0]
                agent_states[agent_id] = new_state 

            
            print(f"Actions: {actions}")
            obs, reward, done, info = test_env.step(actions)
            
            if done['__all__']==True:
                print(f"========== EPISODE ENDED ==============")
                epEnd = True
            

            test_env.render()
            # sync(count, start, test_env.TIMESTEP)
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
