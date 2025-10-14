import os
import time
import torch
import ray
from ray.rllib.agents.qmix import QMixTrainer
from ray.tune.registry import register_env
from gym_pybullet_drones.envs.multi_agent_rl.AutoroutingMASAviary_discrete import AutoroutingMASAviary_discrete
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ActionType, ObservationType
from ray.rllib.env import MultiAgentEnv
from gym import spaces

# === Configs ===
CHECKPOINT_DIR = "./results/save-autorouting-mas-aviary-v0-4-QMIX-kin-autorouting-07.23.2025_11.25.52"  # Replace with your actual dir
CHECKPOINT_FILE = os.path.join(CHECKPOINT_DIR, "checkpoint.txt")
NUM_DRONES = 4
EPISODES = 3

OBS_TYPE = ObservationType.KIN
ACT_TYPE = ActionType.AUTOROUTING
ENV_NAME = "autorouting-mas-aviary-v0"

# === Grouped agent setup ===
agent_ids = list(range(NUM_DRONES))
grouping = {"group_1": agent_ids}

# === Register grouped environment ===
def env_creator(_):
    env = AutoroutingMASAviary_discrete(
        num_drones=NUM_DRONES,
        freq=120,
        aggregate_phy_steps=1,
        obs=OBS_TYPE,
        act=ACT_TYPE
    )
    obs_space = spaces.Tuple([env.observation_space[i] for i in agent_ids])
    act_space = spaces.Tuple([env.action_space[i] for i in agent_ids])
    return env.with_agent_groups(groups=grouping, obs_space=obs_space, act_space=act_space)

register_env(ENV_NAME, env_creator)

# === Init Ray ===
ray.init(ignore_reinit_error=True)

# === Load trainer from checkpoint ===
trainer = QMixTrainer(env=ENV_NAME, config={
    "num_workers": 0,
    "framework": "torch",
    "explore": False,  # No exploration during inference
    "env_config": {},  # Optional
})

# === Load checkpoint ===
with open(CHECKPOINT_FILE, "r") as f:
    checkpoint_path = f.read().strip()
trainer.restore(checkpoint_path)

# === Create actual env instance (not grouped) ===
env = AutoroutingMASAviary_discrete(
    num_drones=NUM_DRONES,
    freq=120,
    aggregate_phy_steps=1,
    obs=OBS_TYPE,
    act=ACT_TYPE,
    gui=True,
)

# === Run Decentralized Execution ===
for ep in range(EPISODES):
    obs = env.reset()
    done = {i: False for i in agent_ids}
    total_reward = 0

    while not all(done.values()):
        actions = {}
        # at each timestep:
        joint_obs = tuple(obs[i] for i in agent_ids)   # tuple with per-agent arrays
        actions_dict = trainer.compute_actions({ "group_1": joint_obs })
        action_for_group = actions_dict["group_1"]
        for i in agent_ids:
            agent_obs = obs[i]
            policy_id = "default_policy"
            action = trainer.compute_action(agent_obs, policy_id=policy_id)
            actions[i] = action

        obs, rewards, done, infos = env.step(actions)
        total_reward += sum(rewards.values())
        env.render()
        # time.sleep(1.0 / env.SIM_FREQ)  # Real-time sleep for visualization

    print(f"[Episode {ep+1}] Total reward: {total_reward:.2f}")

env.close()
ray.shutdown()
