from ray.rllib.env.multi_agent_env import MultiAgentEnv
from gym_pybullet_drones.envs.AutoroutingSARLAviary import AutoroutingSARLAviary
import os


class MultiAgentAutorouting(AutoroutingSARLAviary, MultiAgentEnv):


    def reset(self):
        pass

    def _computeTerminated(self, agent_id):
        return super()._computeTerminated()

