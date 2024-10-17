import gymnasium as gym
from gymnasium.envs.registration import register

register(
    id='autorouting-aviary-v0',
    entry_point='gym_pybullet_drones.envs:AutoroutingRLAviary',
)


train_env = gym.make("autorouting-aviary-v0")