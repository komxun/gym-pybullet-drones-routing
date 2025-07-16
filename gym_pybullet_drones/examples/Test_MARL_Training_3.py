# https://docs.ray.io/en/latest/rllib/package_ref/env/multi_agent_env.html

import os
import time
from datetime import datetime
from ray.rllib.env.multi_agent_env import make_multi_agent
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.dqn import DQNConfig
from ray.tune.logger import pretty_print
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

config = (
    DQNConfig()
    .training(gamma=0.9, lr=0.01)
    .environment(ma_AutoroutingSARLAviary_cls)
    .env_runners(num_env_runners=0)
)

pretty_print(config.to_dict())
algo = config.build() 


#### Train #################################################
for i in range(10):
    result = algo.train()  
# print("Training completed. Printing final results . . .")
# print(pretty_print(result))  

print("Training complete. Saving algorithms to create checkpoints . . .")
checkpoint = algo.save() # Save algorithms to create checkpoints
print(checkpoint)


