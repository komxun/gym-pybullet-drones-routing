import os
import math
import time
from enum import Enum
import numpy as np
import pybullet as p

from gym_pybullet_drones.envs.BaseAviary import DroneModel, BaseAviary


class BaseGuidance(object):
    """Base class for guidance law.
    
    Implements `__init__()`, `reset(), and interface `computeRouteFromState()`,
    the main method `computeRoute()` should be implemented by its subclasses.
    
    """
    
    ################################################################################

    def __init__(self,
                 drone_model: DroneModel,
                 g: float=9.8
                 ):
        """Common routing classes __init__ method.

        Parameters
        ----------
        drone_model : DroneModel
            The type of drone to generate route (detailed in an .urdf file in folder `assets`).
        g : float, optional
            The gravitational acceleration in m/s^2.

        """
        #### Set general use constants #############################
        self.DRONE_MODEL = drone_model
        """DroneModel: The type of drone to control."""
        self.GRAVITY = g*self._getURDFParameter('m')
        """float: The gravitational force (M*g) acting on each drone."""
        self.KF = self._getURDFParameter('kf')
        """float: The coefficient converting RPMs into thrust."""
        self.KM = self._getURDFParameter('km')
        """float: The coefficient converting RPMs into torque."""
        
        self.CUR_POS = np.array([0,0,0])
        self.CUR_VEL = np.array([0,0,0])
        self.DESTINATION = np.array([0,0,0])
        self.TARGET_POS   = np.array([])   # Check-> initialize with empty array should work
        self.TARGET_VEL  =  np.array([0,0,0])
        self.HOME_POS = np.array([0,0,0])
        
        self.SIM_MODE = 2
        self._resetAllCommands()
        self.route_counter = 0
        self.reset()
        
    ################################################################################

    def reset(self):
        """Reset the routing classes.

        A general use counter is set to zero.

        """
        # self.route_counter = 0

    ################################################################################

    ################################################################################
    def followPath(self, path2follow):
        pass
