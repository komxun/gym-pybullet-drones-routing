# https://docs.ray.io/en/latest/rllib/package_ref/env/multi_agent_env.html

import os
import time
from datetime import datetime
from ray.rllib.env.multi_agent_env import make_multi_agent
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray import tune

from gym_pybullet_drones.envs.AutoroutingRLAviary import AutoroutingRLAviary
import time
import random
import pybullet as p
from gym_pybullet_drones.utils.utils import sync, str2bool


#### Save directory ########################################
filename = os.path.dirname(os.path.abspath(__file__))+'/results/save-'+"mas_autorouting"+'-'+datetime.now().strftime("%m.%d.%Y_%H.%M.%S")
if not os.path.exists(filename):
    os.makedirs(filename+'/')

# https://docs.ray.io/en/latest/rllib/package_ref/env/multi_agent_env.html#module-ray.rllib.env.multi_agent_env
ma_AutoroutingSARLAviary_cls = make_multi_agent('autorouting-sa-aviary-v0')


env = ma_AutoroutingSARLAviary_cls({"num_agents": 2})



config = {
    "env": ma_AutoroutingSARLAviary_cls,
    "env_config": {
        "num_agents": 2,
    },
    "num_workers": 1,
    "framework": "torch",
}

#### Ray Tune stopping conditions ##########################
stop = {
    "timesteps_total":int(2e6),  # number of totoal time steps (should exceed 200k) good: 24M (20*120000)
    # "episode_reward_mean": 100,
    # "training_iteration": 0,
}
#### Train #################################################
    
results = tune.run(
    "PPO",
    stop=stop,
    config=config,
    verbose=True,
    checkpoint_at_end=True,
    storage_path=filename,
)



