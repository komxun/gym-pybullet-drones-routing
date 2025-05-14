# https://docs.ray.io/en/latest/rllib/package_ref/env/multi_agent_env.html

from ray.rllib.env.multi_agent_env import make_multi_agent
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
import gymnasium as gym
import numpy as np
import torch
import matplotlib.pyplot as plt
print(f"CUDA is available : {torch.cuda.is_available()}")
# print(torch.cuda.get_device_name(0))
from gym_pybullet_drones.envs.AutoroutingRLAviary import AutoroutingRLAviary

# https://docs.ray.io/en/latest/rllib/package_ref/env/multi_agent_env.html#module-ray.rllib.env.multi_agent_env
# Create the multi-agent environment class
ma_AutoroutingSARLAviary_cls = make_multi_agent('autorouting-sa-aviary-v0')

# Register it with RLlib
register_env("autorouting-multi-agent-v0", lambda config: ma_AutoroutingSARLAviary_cls(config))

config = PPOConfig().environment(
    env="autorouting-multi-agent-v0",  # This matches what you registered
    env_config={"num_agents": 2}       # Config gets passed into the env callable
)

algo = config.build()
print(algo.train())

