# https://docs.ray.io/en/latest/rllib/package_ref/env/multi_agent_env.html

from ray.rllib.env.multi_agent_env import make_multi_agent
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from gym_pybullet_drones.envs.AutoroutingRLAviary import AutoroutingRLAviary
import time
import random
import pybullet as p
from gym_pybullet_drones.utils.utils import sync, str2bool

# https://docs.ray.io/en/latest/rllib/package_ref/env/multi_agent_env.html#module-ray.rllib.env.multi_agent_env
ma_AutoroutingSARLAviary_cls = make_multi_agent('autorouting-sa-aviary-v0')

env = ma_AutoroutingSARLAviary_cls({"num_agents": 2})


for _ in range(20):
    
    epEnd = False
    i = 0
    START = time.time()
    env.reset()
    # for j in range(num_drones):
    #     env.routing[j].reset()
    while not epEnd:

        #### Step the simulation ###################################
        # obs, reward, terminated, truncated, info = env.step(action)
        obs, rewards, terminated, truncated, info = env.step(action_dict={
            0: 4, 
            1: 5,
        })

        # p.resetDebugVisualizerCamera(cameraDistance=2, cameraYaw=45, cameraPitch=-60, cameraTargetPosition=obs[0:3])
        if terminated or truncated:
            epEnd = True
        # print(f"obs len = {len(obs[0])}")
        print(f"truncated = {truncated}")

        #### Printout ##############################################
        # env.render()
        #### Sync the simulation ###################################
        # sync(i, START, env.CTRL_TIMESTEP)
        i+=1

#### Close the environment #################################
env.close()

config = (
    PPOConfig()
    .environment(env='autorouting-sa-aviary-v0')
    .multi_agent(
        policy_mapping_fn=lambda agent_id, episode, **kwargs: (
            "agent1" if agent_id.startswith("0")
            else "agent12"
        ),
        algorithm_config_overrides_per_module={
            0: PPOConfig.overrides(gamma=0.85),
            1: PPOConfig.overrides(gamma=0.85),
        },
    )
    .rl_module(
        rl_module_spec=MultiRLModuleSpec(rl_module_specs={
            0: RLModuleSpec(),
            1: RLModuleSpec(),
        }),
    )
)

algo = config.build()
print(algo.train())

