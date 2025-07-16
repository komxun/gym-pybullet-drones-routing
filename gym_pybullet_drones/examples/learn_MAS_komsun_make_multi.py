"""Script demonstrating the use of `gym_pybullet_drones`'s Gymnasium interface.

Classes HoverAviary and MultiHoverAviary are used as learning envs for the PPO algorithm.

Example
-------
In a terminal, run as:

    $ python learn.py --multiagent false
    $ python learn.py --multiagent true

Notes
-----
This is a minimal working example integrating `gym-pybullet-drones` with 
reinforcement learning library `ray[rllib]`.

"""
import os
import time
from datetime import datetime
import argparse
import gymnasium as gym
import numpy as np
import torch

# Updated Ray imports for compatibility
from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.env.env_context import EnvContext
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
import ray
from ray.rllib.env.multi_agent_env import make_multi_agent

from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.envs.HoverAviary import HoverAviary
from gym_pybullet_drones.envs.AutoroutingSARLAviary import AutoroutingSARLAviary
from gym_pybullet_drones.envs.AutoroutingMARLAviary import AutoroutingMARLAviary
from gym_pybullet_drones.utils.utils import sync, str2bool
from gym_pybullet_drones.utils.enums import ObservationType, ActionType
from gym_pybullet_drones.utils.enums import DroneModel, Physics

DEFAULT_GUI = True
DEFAULT_RECORD_VIDEO = False
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_COLAB = False

DEFAULT_OBS = ObservationType('kin') # 'kin' or 'rgb'
DEFAULT_ACT = ActionType('one_d_rpm') # 'rpm' or 'pid' or 'vel' or 'one_d_rpm' or 'one_d_pid'
DEFAULT_MA = True


def make_env(env_config):
    """Environment factory function for Ray RLlib"""
    return env_config.get("env_class", AutoroutingMARLAviary)(**env_config.get("env_kwargs", {}))


def run(multiagent=DEFAULT_MA, output_folder=DEFAULT_OUTPUT_FOLDER, gui=DEFAULT_GUI, plot=True, colab=DEFAULT_COLAB, record_video=DEFAULT_RECORD_VIDEO, local=True):

    filename = os.path.join(output_folder, 'save-'+datetime.now().strftime("%m.%d.%Y_%H.%M.%S"))
    if not os.path.exists(filename):
        os.makedirs(filename+'/')

    DEFAULT_DRONES = DroneModel("cf2x")
    DEFAULT_PHYSICS = Physics("pyb")
    DEFAULT_NUM_DRONES = 2
    INIT_XYZS = np.array([[((-1)**i)*(i*0.2)+0.5,-3*(i*0.05), 1.3+ 0.05*i ] for i in range(DEFAULT_NUM_DRONES)])
    INIT_RPYS = np.array([[0, 0,  i * (np.pi/2)/DEFAULT_NUM_DRONES] for i in range(DEFAULT_NUM_DRONES)])
    DEFAULT_SIMULATION_FREQ_HZ = 240
    DEFAULT_CONTROL_FREQ_HZ = 48

    # Initialize Ray
    if not ray.is_initialized():
        ray.init()

    # Environment configuration
    env_config = {
        "env_class": AutoroutingMARLAviary,
        "env_kwargs": {
            "drone_model": DEFAULT_DRONES,
            "num_drones": DEFAULT_NUM_DRONES, 
            "initial_xyzs": INIT_XYZS,
            "initial_rpys": INIT_RPYS,
            "physics": DEFAULT_PHYSICS,
            "pyb_freq": DEFAULT_SIMULATION_FREQ_HZ,
            "ctrl_freq": DEFAULT_CONTROL_FREQ_HZ,
            "gui": False,  # Set to False for training
            "record": False,
        }
    }

    # Create a test environment to get spaces
    test_env = make_env(env_config)
    
    #### Check the environment's spaces ########################
    print('[INFO] Action space:', test_env.action_space)
    print('[INFO] Observation space:', test_env.observation_space)

    #### Train the model #######################################
    try:
        algo = DQNConfig()\
        .environment(env=make_env, env_config=env_config)\
        .multi_agent(
            policies={
                "policy_1": (
                    None, test_env.observation_space, test_env.action_space, {"gamma": 0.80}
                ),
                "policy_2": (
                    None, test_env.observation_space, test_env.action_space, {"gamma": 0.95}
                ),
            },
            policy_mapping_fn=lambda agent_id, episode, worker, **kwargs: f"policy_{agent_id}",
        )\
        .framework("torch")\
        .build()

        for ep in range(10):
            result = algo.train()
            print(f"Episode {ep}: {result}")

        print("Training complete. Saving algorithms to create checkpoints . . .")
        checkpoint = algo.save() # Save algorithms to create checkpoints
        print(checkpoint)
        
    except Exception as e:
        print(f"Training error: {e}")
        print("This might be due to environment compatibility issues with Ray RLlib")
        print("Consider using stable-baselines3 instead for this environment")

    # Clean up
    test_env.close()

    ############################################################
    ############################################################
    ############################################################
    ############################################################
    ############################################################



if __name__ == '__main__':
    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='Single agent reinforcement learning example script')
    parser.add_argument('--multiagent',         default=DEFAULT_MA,            type=str2bool,      help='Whether to use example LeaderFollower instead of Hover (default: False)', metavar='')
    parser.add_argument('--gui',                default=DEFAULT_GUI,           type=str2bool,      help='Whether to use PyBullet GUI (default: True)', metavar='')
    parser.add_argument('--record_video',       default=DEFAULT_RECORD_VIDEO,  type=str2bool,      help='Whether to record a video (default: False)', metavar='')
    parser.add_argument('--output_folder',      default=DEFAULT_OUTPUT_FOLDER, type=str,           help='Folder where to save logs (default: "results")', metavar='')
    parser.add_argument('--colab',              default=DEFAULT_COLAB,         type=bool,          help='Whether example is being run by a notebook (default: "False")', metavar='')
    ARGS = parser.parse_args()

    run(**vars(ARGS))