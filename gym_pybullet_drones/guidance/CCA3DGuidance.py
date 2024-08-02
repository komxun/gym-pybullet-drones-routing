import math
import numpy as np
# import pybullet as p

from gym_pybullet_drones.guidance import BaseGuidance
from gym_pybullet_drones.envs.BaseAviary import DroneModel, BaseAviary

class CCA3DGuidance(BaseGuidance):
    """3D Carrot-Chasing path-following class"""
    
    