import numpy as np
from gym import spaces
import pybullet as p
from gym_pybullet_drones.envs.BaseAviary import DroneModel, Physics, BaseAviary
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary

class RoutingAviary(CtrlAviary):
    """Single-drone environment class for routing application"""
    
    OBSTACLE_IDS = []
    
    def __init__(self, 
                 drone_model: DroneModel=DroneModel.CF2X,
                 num_drones: int=1,
                 neighbourhood_radius: float=np.inf,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics=Physics.PYB,
                 freq: int=240,
                 aggregate_phy_steps: int=1,
                 gui=False,
                 record=False,
                 obstacles=False,
                 user_debug_gui=True
                 ):
        """Initialization of an aviary environment for routing applications.
        
        Parameters
        ----------
        drone_model : DroneModel, optional
            The desired drone type (detailed in an .urdf file in folder `assets`).
        num_drones : int, optional
            The desired number of drones in the aviary.
        neighbourhood_radius : float, optional
            Radius used to compute the drones' adjacency matrix, in meters.
        initial_xyzs: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial XYZ position of the drones.
        initial_rpys: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial orientations of the drones (in radians).
        physics : Physics, optional
            The desired implementation of PyBullet physics/custom dynamics.
        freq : int, optional
            The frequency (Hz) at which the physics engine steps.
        aggregate_phy_steps : int, optional
            The number of physics steps within one call to `BaseAviary.step()`.
        gui : bool, optional
            Whether to use PyBullet's GUI.
        record : bool, optional
            Whether to save a video of the simulation in folder `files/videos/`.
        obstacles : bool, optional
            Whether to add obstacles to the simulation.
        user_debug_gui : bool, optional
            Whether to draw the drones' axes and the GUI RPMs sliders.
        
        """
        super().__init__(drone_model=drone_model,
                         num_drones=num_drones,
                         neighbourhood_radius=neighbourhood_radius,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         freq=freq,
                         aggregate_phy_steps=aggregate_phy_steps,
                         gui=gui,
                         record=record,
                         obstacles=obstacles,
                         user_debug_gui=user_debug_gui
                         )
        #### Set a limit on the maximum target speed ###############
        speedLimitingFactor = 0.5   # 0.03
        self.SPEED_LIMIT = speedLimitingFactor * self.MAX_SPEED_KMH * (1000/3600)
        
        # # New ---------Get obstacle information
        self._getObstaclesData()
    
    ################################################################################
        
    def _addObstacles(self):
        """Add obstacles to the environment.

        These obstacles are loaded from standard URDF files included in Bullet.

        """
        # RoutingAviary.OBSTACLE_IDS.append(
        #                     p.loadURDF("samurai.urdf",
        #                     physicsClientId=self.CLIENT
        #                     ))

        RoutingAviary.OBSTACLE_IDS.append(
            p.loadURDF("cube_no_rotation.urdf",
                   [-.5, -2.5, .5],
                   p.getQuaternionFromEuler([0, 0, 0]),
                   physicsClientId=self.CLIENT
                   ))
        RoutingAviary.OBSTACLE_IDS.append(
            p.loadURDF("sphere2.urdf",
                   [0, 2, .5],
                   p.getQuaternionFromEuler([0,0,0]),
                   useFixedBase = True, 
                   physicsClientId=self.CLIENT
                   ))
        RoutingAviary.OBSTACLE_IDS.append(
            p.loadURDF("sphere2.urdf", 
                   [0.8, 5, 0+1], 
                   p.getQuaternionFromEuler([0,0,0]), 
                   useFixedBase = True, 
                   physicsClientId=self.CLIENT))
        RoutingAviary.OBSTACLE_IDS.append(
            p.loadURDF("sphere2.urdf", 
                   [0.5, 3, 0.5], 
                   p.getQuaternionFromEuler([0,0,0]), 
                   useFixedBase = True, 
                   physicsClientId=self.CLIENT))
        RoutingAviary.OBSTACLE_IDS.append(
            p.loadURDF("sphere2.urdf", 
                   [0.8, 7, 0], 
                   p.getQuaternionFromEuler([0,0,0]), 
                   useFixedBase = True, 
                   physicsClientId=self.CLIENT))
        # self._getObstaclesData()
 
    def _getObstaclesData(self):
        idsList = RoutingAviary.OBSTACLE_IDS
        obs_pos = []
        obs_size = []
        for j in range(len(idsList)):
            pos, orn = p.getBasePositionAndOrientation(idsList[j])

            obs_pos.append(pos)
            vsd = p.getVisualShapeData(idsList[j])
            obs_size.append(vsd[0][3])
        
        self.obs_size = np.array(obs_size)  # matrix of size Nx3
        self.obs_pos = np.array(obs_pos)    # matrix of size Nx3
        
        