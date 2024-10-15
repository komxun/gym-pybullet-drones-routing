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
reinforcement learning library `stable-baselines3`.

"""
import os
import time
from datetime import datetime
import argparse
import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.evaluation import evaluate_policy





from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.envs.HoverAviary import HoverAviary
from gym_pybullet_drones.envs.AutoroutingRLAviary import AutoroutingRLAviary
from gym_pybullet_drones.envs.MultiHoverAviary import MultiHoverAviary
from gym_pybullet_drones.utils.utils import sync, str2bool
from gym_pybullet_drones.utils.enums import ObservationType, ActionType, Physics

DEFAULT_GUI = True
DEFAULT_RECORD_VIDEO = False
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_COLAB = False

DEFAULT_OBS = ObservationType('kin') # 'kin' or 'rgb'
DEFAULT_ACT = ActionType('autorouting') # 'rpm' or 'pid' or 'vel' or 'one_d_rpm' or 'one_d_pid'
DEFAULT_AGENTS = 1
DEFAULT_MA = False
DEFAULT_PHYSICS = Physics("pyb")
DEFAULT_CONTROL_FREQ_HZ = 60
DEFAULT_SIMULATION_FREQ_HZ = 60

INIT_XYZS = np.array([[((-1)**i)*(i*0.2)+0.5,-3*(i*0.05), 0.5+ 0.05*i ] for i in range(DEFAULT_AGENTS)])
INIT_RPYS = np.array([[0, 0,  i * (np.pi/2)/DEFAULT_AGENTS] for i in range(DEFAULT_AGENTS)])

def run(multiagent=DEFAULT_MA, output_folder=DEFAULT_OUTPUT_FOLDER, gui=DEFAULT_GUI, plot=True, colab=DEFAULT_COLAB, record_video=DEFAULT_RECORD_VIDEO, local=True):

    # filename = os.path.join('results', 'save-09.26.2024_15.41.28')
    
    # os.makedirs(filename+'/')
    # if os.path.isfile(filename+'/final_model.zip'):
    #     path = filename+'/final_model.zip'
    # if os.path.isfile(filename+'/best_model.zip'):
    #     path = filename+'/best_model.zip'
    # else:
    #     print("[ERROR]: no model under the specified path", filename)
    # path = 'results/save-09.26.2024_15.41.28/best_model.zip'
    # path = 'results/save-09.26.2024_18.55.51/best_model.zip'
    # path = 'results/save-09.27.2024_16.19.17/best_model.zip'
    path = 'results/save-10.01.2024_14.20.00/best_model.zip'  # DQN
    model = DQN.load(path)  
    #### Show (and record a video of) the model's performance ##
    if not multiagent:
        print("========= Testing for Single agent (no GUI) ============")
        # test_env = AutoroutingRLAviary(gui=gui,
        #                        obs=DEFAULT_OBS,
        #                        act=DEFAULT_ACT,
        #                        physics=DEFAULT_PHYSICS, 
        #                        ctrl_freq = DEFAULT_CONTROL_FREQ_HZ,
        #                        pyb_freq = DEFAULT_SIMULATION_FREQ_HZ,
        #                        initial_xyzs=INIT_XYZS,
        #                         initial_rpys=INIT_RPYS,)
        test_env_nogui = AutoroutingRLAviary(gui =False,
                                             obs=DEFAULT_OBS, 
                                             act=DEFAULT_ACT, 
                                             physics=DEFAULT_PHYSICS, 
                                             ctrl_freq = DEFAULT_CONTROL_FREQ_HZ,
                                             pyb_freq = DEFAULT_SIMULATION_FREQ_HZ,
                                             initial_xyzs=INIT_XYZS,
                                                initial_rpys=INIT_RPYS,)
    else:
        test_env = MultiHoverAviary(gui=gui,
                                        num_drones=DEFAULT_AGENTS,
                                        obs=DEFAULT_OBS,
                                        act=DEFAULT_ACT,
                                        record=record_video)
        test_env_nogui = MultiHoverAviary(num_drones=DEFAULT_AGENTS, obs=DEFAULT_OBS, act=DEFAULT_ACT)
    

    mean_reward, std_reward = evaluate_policy(model,
                                              test_env_nogui,
                                              n_eval_episodes=10
                                              )
    print("\n\n\nMean reward ", mean_reward, " +- ", std_reward, "\n\n")
    test_env_nogui.close()  
    input("Press Enter to continue...")
    test_env = AutoroutingRLAviary(gui=gui,
                               obs=DEFAULT_OBS,
                               act=DEFAULT_ACT,
                               physics=DEFAULT_PHYSICS, 
                               ctrl_freq = DEFAULT_CONTROL_FREQ_HZ,
                               pyb_freq = DEFAULT_SIMULATION_FREQ_HZ,
                               initial_xyzs=INIT_XYZS,
                                initial_rpys=INIT_RPYS,)
    
    logger = Logger(logging_freq_hz=int(test_env.CTRL_FREQ),
                num_drones=DEFAULT_AGENTS if multiagent else 1,
                output_folder=output_folder,
                colab=colab
                )

    
    obs, info = test_env.reset(seed=42)
    
    start = time.time()
    for i in range(0, (test_env.EPISODE_LEN_SEC+60)*test_env.CTRL_FREQ):
        
        action, _states = model.predict(obs,
                                        deterministic=True
                                        )
        
        obs, reward, terminated, truncated, info = test_env.step(action)
        # obs2 = obs.squeeze()
        # act2 = action.squeeze()
        print("Obs:", obs, "\tAction", action, "\tReward:", reward, "\tTerminated:", terminated, "\tTruncated:", truncated)
        # if DEFAULT_OBS == ObservationType.KIN:
        #     if not multiagent:
        #         logger.log(drone=0,
        #             timestamp=i/test_env.CTRL_FREQ,
        #             state=np.hstack([obs2[0:3],
        #                                 np.zeros(4),
        #                                 obs2[3:15],
        #                                 act2
        #                                 ]),
        #             control=np.zeros(12)
        #             )
        #     else:
        #         for d in range(DEFAULT_AGENTS):
        #             logger.log(drone=d,
        #                 timestamp=i/test_env.CTRL_FREQ,
        #                 state=np.hstack([obs2[d][0:3],
        #                                     np.zeros(4),
        #                                     obs2[d][3:15],
        #                                     act2[d]
        #                                     ]),
        #                 control=np.zeros(12)
        #                 )
        test_env.render()
        print(f"terminated = {terminated}")
        sync(i, start, test_env.CTRL_TIMESTEP)
        if terminated:
            obs, _  = test_env.reset(seed=42)
    test_env.close()

    # if plot and DEFAULT_OBS == ObservationType.KIN:
    #     logger.plot()

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
