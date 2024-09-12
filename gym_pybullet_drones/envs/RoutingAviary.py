import os
import numpy as np
from gymnasium import spaces
from datetime import datetime

from gym_pybullet_drones.envs.BaseAviary import BaseAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ImageType

import pybullet as p
from PIL import Image

class RoutingAviary(BaseAviary):
    """Multi-drone environment class for control applications."""
    
    OBSTACLE_IDS = []

    ################################################################################

    def __init__(self,
                  drone_model: DroneModel=DroneModel.CF2X,
                  num_drones: int=1,
                  neighbourhood_radius: float=np.inf,
                  initial_xyzs=None,
                  initial_rpys=None,
                  physics: Physics=Physics.PYB,
                  pyb_freq: int = 240,
                  ctrl_freq: int = 240,
                  gui=False,
                  record=False,
                  obstacles=False,
                  user_debug_gui=True,
                  vision_attributes=False,
                  output_folder='results'
                  ):
        """Initialization of a generic aviary environment.

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
        pyb_freq : int, optional
            The frequency at which PyBullet steps (a multiple of ctrl_freq).
        ctrl_freq : int, optional
            The frequency at which the environment steps.
        gui : bool, optional
            Whether to use PyBullet's GUI.
        record : bool, optional
            Whether to save a video of the simulation.
        obstacles : bool, optional
            Whether to add obstacles to the simulation.
        user_debug_gui : bool, optional
            Whether to draw the drones' axes and the GUI RPMs sliders.
        vision_attributes : bool, optional
            Whether to allocate the attributes needed by vision-based aviary subclasses.

        """
        super().__init__(drone_model=drone_model,
                          num_drones=num_drones,
                          neighbourhood_radius=neighbourhood_radius,
                          initial_xyzs=initial_xyzs,
                          initial_rpys=initial_rpys,
                          physics=physics,
                          pyb_freq=pyb_freq,
                          ctrl_freq=ctrl_freq,
                          gui=gui,
                          record=record,
                          obstacles=obstacles,
                          user_debug_gui=user_debug_gui,
                          output_folder=output_folder
                          )
        
        #### Set a limit on the maximum target speed ###############
        speedLimitingFactor = 0.1   # 0.03
        self.SPEED_LIMIT = speedLimitingFactor * self.MAX_SPEED_KMH * (1000/3600)
        self.CONTACT_POINTS = [() for _ in range(self.NUM_DRONES)]
        self.CONTACT_FLAGS = np.zeros(self.NUM_DRONES)
        
        self.OBSTACLE_DATA = {}
        self._getObstaclesData()
        
    ################################################################################
    def step(self,
             action
             ):
        """Advances the environment by one simulation step.

        Parameters
        ----------
        action : ndarray | dict[..]
            The input action for one or more drones, translated into RPMs by
            the specific implementation of `_preprocessAction()` in each subclass.

        Returns
        -------
        ndarray | dict[..]
            The step's observation, check the specific implementation of `_computeObs()`
            in each subclass for its format.
        float | dict[..]
            The step's reward value(s), check the specific implementation of `_computeReward()`
            in each subclass for its format.
        bool | dict[..]
            Whether the current episode is over, check the specific implementation of `_computeTerminated()`
            in each subclass for its format.
        bool | dict[..]
            Whether the current episode is truncated, check the specific implementation of `_computeTruncated()`
            in each subclass for its format.
        bool | dict[..]
            Whether the current episode is trunacted, always false.
        dict[..]
            Additional information as a dictionary, check the specific implementation of `_computeInfo()`
            in each subclass for its format.

        """
        #### Save PNG video frames if RECORD=True and GUI=False ####
        if self.RECORD and not self.GUI and self.step_counter%self.CAPTURE_FREQ == 0:
            [w, h, rgb, dep, seg] = p.getCameraImage(width=self.VID_WIDTH,
                                                     height=self.VID_HEIGHT,
                                                     shadow=1,
                                                     viewMatrix=self.CAM_VIEW,
                                                     projectionMatrix=self.CAM_PRO,
                                                     renderer=p.ER_TINY_RENDERER,
                                                     flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
                                                     physicsClientId=self.CLIENT
                                                     )
            (Image.fromarray(np.reshape(rgb, (h, w, 4)), 'RGBA')).save(os.path.join(self.IMG_PATH, "frame_"+str(self.FRAME_NUM)+".png"))
            #### Save the depth or segmentation view instead #######
            # dep = ((dep-np.min(dep)) * 255 / (np.max(dep)-np.min(dep))).astype('uint8')
            # (Image.fromarray(np.reshape(dep, (h, w)))).save(self.IMG_PATH+"frame_"+str(self.FRAME_NUM)+".png")
            # seg = ((seg-np.min(seg)) * 255 / (np.max(seg)-np.min(seg))).astype('uint8')
            # (Image.fromarray(np.reshape(seg, (h, w)))).save(self.IMG_PATH+"frame_"+str(self.FRAME_NUM)+".png")
            self.FRAME_NUM += 1
            if self.VISION_ATTR:
                for i in range(self.NUM_DRONES):
                    self.rgb[i], self.dep[i], self.seg[i] = self._getDroneImages(i)
                    #### Printing observation to PNG frames example ############
                    self._exportImage(img_type=ImageType.RGB, # ImageType.BW, ImageType.DEP, ImageType.SEG
                                    img_input=self.rgb[i],
                                    path=self.ONBOARD_IMG_PATH+"/drone_"+str(i)+"/",
                                    frame_num=int(self.step_counter/self.IMG_CAPTURE_FREQ)
                                    )
        #### Read the GUI's input parameters #######################
        if self.GUI and self.USER_DEBUG:
            current_input_switch = p.readUserDebugParameter(self.INPUT_SWITCH, physicsClientId=self.CLIENT)
            if current_input_switch > self.last_input_switch:
                self.last_input_switch = current_input_switch
                self.USE_GUI_RPM = True if self.USE_GUI_RPM == False else False
        if self.USE_GUI_RPM:
            for i in range(4):
                self.gui_input[i] = p.readUserDebugParameter(int(self.SLIDERS[i]), physicsClientId=self.CLIENT)
            clipped_action = np.tile(self.gui_input, (self.NUM_DRONES, 1))
            if self.step_counter%(self.PYB_FREQ/2) == 0:
                self.GUI_INPUT_TEXT = [p.addUserDebugText("Using GUI RPM",
                                                          textPosition=[0, 0, 0],
                                                          textColorRGB=[1, 0, 0],
                                                          lifeTime=1,
                                                          textSize=2,
                                                          parentObjectUniqueId=self.DRONE_IDS[i],
                                                          parentLinkIndex=-1,
                                                          replaceItemUniqueId=int(self.GUI_INPUT_TEXT[i]),
                                                          physicsClientId=self.CLIENT
                                                          ) for i in range(self.NUM_DRONES)]
        #### Save, preprocess, and clip the action to the max. RPM #
        else:
            clipped_action = np.reshape(self._preprocessAction(action), (self.NUM_DRONES, 4))
        #### Repeat for as many as the aggregate physics steps #####
        for _ in range(self.PYB_STEPS_PER_CTRL):
            #### Update and store the drones kinematic info for certain
            #### Between aggregate steps for certain types of update ###
            if self.PYB_STEPS_PER_CTRL > 1 and self.PHYSICS in [Physics.DYN, Physics.PYB_GND, Physics.PYB_DRAG, Physics.PYB_DW, Physics.PYB_GND_DRAG_DW]:
                self._updateAndStoreKinematicInformation()
            #### Step the simulation using the desired physics update ##
            for i in range (self.NUM_DRONES):
                if self.PHYSICS == Physics.PYB:
                    self._physics(clipped_action[i, :], i)
                elif self.PHYSICS == Physics.DYN:
                    self._dynamics(clipped_action[i, :], i)
                elif self.PHYSICS == Physics.PYB_GND:
                    self._physics(clipped_action[i, :], i)
                    self._groundEffect(clipped_action[i, :], i)
                elif self.PHYSICS == Physics.PYB_DRAG:
                    self._physics(clipped_action[i, :], i)
                    self._drag(self.last_clipped_action[i, :], i)
                elif self.PHYSICS == Physics.PYB_DW:
                    self._physics(clipped_action[i, :], i)
                    self._downwash(i)
                elif self.PHYSICS == Physics.PYB_GND_DRAG_DW:
                    self._physics(clipped_action[i, :], i)
                    self._groundEffect(clipped_action[i, :], i)
                    self._drag(self.last_clipped_action[i, :], i)
                    self._downwash(i)
            #### PyBullet computes the new state, unless Physics.DYN ###
            if self.PHYSICS != Physics.DYN:
                p.stepSimulation(physicsClientId=self.CLIENT)
            #### Save the last applied action (e.g. to compute drag) ###
            self.last_clipped_action = clipped_action
        #### Update and store the drones kinematic information #####
        self._updateAndStoreKinematicInformation()
        self._detectCollision()
        #### Prepare the return values #############################
        obs = self._computeObs()
        reward = self._computeReward()
        terminated = self._computeTerminated()
        truncated = self._computeTruncated()
        info = self._computeInfo()
        #### Advance the step counter ##############################
        self.step_counter = self.step_counter + (1 * self.PYB_STEPS_PER_CTRL)
        return obs, reward, terminated, truncated, info
    
    # def step(self,
    #          action
    #          ):
    #     """Advances the environment by one simulation step.

    #     Parameters
    #     ----------
    #     action : ndarray | dict[..]
    #         The input action for one or more drones, translated into RPMs by
    #         the specific implementation of `_preprocessAction()` in each subclass.

    #     Returns
    #     -------
    #     ndarray | dict[..]
    #         The step's observation, check the specific implementation of `_computeObs()`
    #         in each subclass for its format.
    #     float | dict[..]
    #         The step's reward value(s), check the specific implementation of `_computeReward()`
    #         in each subclass for its format.
    #     bool | dict[..]
    #         Whether the current episode is over, check the specific implementation of `_computeTerminated()`
    #         in each subclass for its format.
    #     bool | dict[..]
    #         Whether the current episode is truncated, check the specific implementation of `_computeTruncated()`
    #         in each subclass for its format.
    #     bool | dict[..]
    #         Whether the current episode is trunacted, always false.
    #     dict[..]
    #         Additional information as a dictionary, check the specific implementation of `_computeInfo()`
    #         in each subclass for its format.

    #     """
        
    #     super().step(action=action)
    #     self._detectCollision()
            
    ################################################################################
    
    def _addObstacles(self):
        """Add obstacles to the environment.

        These obstacles are loaded from standard URDF files included in Bullet.

        """
        scene = 1
        
        if scene == 1:
        
            RoutingAviary.OBSTACLE_IDS.append(
                p.loadURDF("cube.urdf",
                        [0, 2, .5],
                        p.getQuaternionFromEuler([0,0,0]),
                        useFixedBase = True, 
                        physicsClientId=self.CLIENT
                        ))
            
            
            RoutingAviary.OBSTACLE_IDS.append(
                p.loadURDF("cube.urdf", 
                        [0.5, 3, 0.5], 
                        p.getQuaternionFromEuler([0,0,0]), 
                        useFixedBase = True, 
                        physicsClientId=self.CLIENT))
            
            RoutingAviary.OBSTACLE_IDS.append(
                p.loadURDF("cube.urdf", 
                        [-0.2, 5, 0+1], 
                        p.getQuaternionFromEuler([0,0,0]), 
                        useFixedBase = True, 
                        physicsClientId=self.CLIENT))
            
            RoutingAviary.OBSTACLE_IDS.append(
                p.loadURDF("cube.urdf", 
                        [0.8, 7, 0], 
                        p.getQuaternionFromEuler([0,0,0]), 
                        useFixedBase = True, 
                        physicsClientId=self.CLIENT))    
            
            RoutingAviary.OBSTACLE_IDS.append(
                p.loadURDF("cube.urdf", 
                        [1, 4, 0], 
                        p.getQuaternionFromEuler([0,0,0]), 
                        useFixedBase = True, 
                        physicsClientId=self.CLIENT)) 
            
            RoutingAviary.OBSTACLE_IDS.append(
                p.loadURDF("cube.urdf", 
                        [2.5, 4, 1.5], 
                        p.getQuaternionFromEuler([0,0,0]), 
                        useFixedBase = True, 
                        physicsClientId=self.CLIENT)) 
            RoutingAviary.OBSTACLE_IDS.append(
                p.loadURDF("cube.urdf", 
                        [3.5, 4, 1.5], 
                        p.getQuaternionFromEuler([0,0,0]), 
                        useFixedBase = True, 
                        physicsClientId=self.CLIENT)) 
            RoutingAviary.OBSTACLE_IDS.append(
                p.loadURDF("cube.urdf", 
                        [3.5, 6, 2.5], 
                        p.getQuaternionFromEuler([0,0,0]), 
                        useFixedBase = True, 
                        physicsClientId=self.CLIENT)) 
            RoutingAviary.OBSTACLE_IDS.append(
                p.loadURDF("cube.urdf", 
                        [4.5, 7, 1.5], 
                        p.getQuaternionFromEuler([0,0,0]), 
                        useFixedBase = True, 
                        physicsClientId=self.CLIENT)) 
            RoutingAviary.OBSTACLE_IDS.append(
                p.loadURDF("cube.urdf", 
                        [3, 6, 0.5], 
                        p.getQuaternionFromEuler([0,0,0]), 
                        useFixedBase = True, 
                        physicsClientId=self.CLIENT)) 
            
            
            RoutingAviary.OBSTACLE_IDS.append(
                p.loadURDF("cube.urdf", 
                        [1.5, 8.5, 2.8], 
                        p.getQuaternionFromEuler([0,0,0]), 
                        useFixedBase = True, 
                        physicsClientId=self.CLIENT)) 
            # RoutingAviary.OBSTACLE_IDS.append(
            #     p.loadURDF("cube.urdf", 
            #             [2.5, 8.5, 1.5], 
            #             p.getQuaternionFromEuler([0,0,0]), 
            #             useFixedBase = True, 
            #             physicsClientId=self.CLIENT)) 
            RoutingAviary.OBSTACLE_IDS.append(
                p.loadURDF("sphere2.urdf", 
                        [1.5, 9.5, 1], 
                        p.getQuaternionFromEuler([0,0,0]), 
                        useFixedBase = True, 
                        physicsClientId=self.CLIENT)) 
            
            
            # RoutingAviary.OBSTACLE_IDS.append(
            #     p.loadURDF("cube.urdf", 
            #             [0, 8, 2.5], 
            #             p.getQuaternionFromEuler([0,0,0]), 
            #             useFixedBase = True, 
            #             physicsClientId=self.CLIENT)) 
            # RoutingAviary.OBSTACLE_IDS.append(
            #     p.loadURDF("cube.urdf", 
            #             [0, 8, 1.5], 
            #             p.getQuaternionFromEuler([0,0,0]), 
            #             useFixedBase = True, 
            #             physicsClientId=self.CLIENT)) 
            # RoutingAviary.OBSTACLE_IDS.append(
            #     p.loadURDF("cube.urdf", 
            #             [0, 8, 0.5], 
            #             p.getQuaternionFromEuler([0,0,0]), 
            #             useFixedBase = True, 
            #             physicsClientId=self.CLIENT)) 
            
            RoutingAviary.OBSTACLE_IDS.append(
                p.loadURDF("cube.urdf", 
                        [-0.5, 8, 1], 
                        p.getQuaternionFromEuler([0,0,0]), 
                        useFixedBase = True, 
                        physicsClientId=self.CLIENT)) 
            
            
        elif scene == 2:
            
            RoutingAviary.OBSTACLE_IDS.append(
                p.loadURDF("sphere2.urdf",
                        [0, 2, .5],
                        p.getQuaternionFromEuler([0,0,0]),
                        useFixedBase = False, 
                        physicsClientId=self.CLIENT
                        ))
            
            RoutingAviary.OBSTACLE_IDS.append(
                p.loadURDF("sphere2.urdf", 
                        [0.5, 3, 0.5], 
                        p.getQuaternionFromEuler([0,0,0]), 
                        useFixedBase = True, 
                        physicsClientId=self.CLIENT))
            
            # RoutingAviary.OBSTACLE_IDS.append(
            #     p.loadURDF("sphere2.urdf", 
            #             [0.8, 5, 0+1], 
            #             p.getQuaternionFromEuler([0,0,0]), 
            #             useFixedBase = True, 
            #             physicsClientId=self.CLIENT))
            
            RoutingAviary.OBSTACLE_IDS.append(
                p.loadURDF("sphere2.urdf", 
                        [0, 5, 0.5], 
                        p.getQuaternionFromEuler([0,0,0]), 
                        useFixedBase = False, 
                        physicsClientId=self.CLIENT))
            
            RoutingAviary.OBSTACLE_IDS.append(
                p.loadURDF("cube.urdf", 
                        [0.8, 7, 0.5], 
                        p.getQuaternionFromEuler([0,0,0]), 
                        useFixedBase = True, 
                        physicsClientId=self.CLIENT))    
            
            RoutingAviary.OBSTACLE_IDS.append(
                p.loadURDF("sphere2.urdf", 
                        [-0.5, 4, 1.5], 
                        p.getQuaternionFromEuler([0,0,0]), 
                        useFixedBase = False, 
                        physicsClientId=self.CLIENT)) 
            
            RoutingAviary.OBSTACLE_IDS.append(
                p.loadURDF("cube.urdf", 
                        [4.5, 7, 1.5], 
                        p.getQuaternionFromEuler([0,0,0]), 
                        useFixedBase = True, 
                        physicsClientId=self.CLIENT)) 
            RoutingAviary.OBSTACLE_IDS.append(
                p.loadURDF("cube.urdf", 
                        [3, 6, 0.5], 
                        p.getQuaternionFromEuler([0,0,0]), 
                        useFixedBase = True, 
                        physicsClientId=self.CLIENT)) 
            
            
            RoutingAviary.OBSTACLE_IDS.append(
                p.loadURDF("cube.urdf", 
                        [2, 8.5, 2.5], 
                        p.getQuaternionFromEuler([0,0,0]), 
                        useFixedBase = True, 
                        physicsClientId=self.CLIENT)) 
            
            
            RoutingAviary.OBSTACLE_IDS.append(
                p.loadURDF("cube.urdf", 
                        [0.8, 7, 0], 
                        p.getQuaternionFromEuler([0,0,0]), 
                        useFixedBase = True, 
                        physicsClientId=self.CLIENT))    
            
            RoutingAviary.OBSTACLE_IDS.append(
                p.loadURDF("cube.urdf", 
                        [1, 4, 0], 
                        p.getQuaternionFromEuler([0,0,0]), 
                        useFixedBase = True, 
                        physicsClientId=self.CLIENT)) 
            
            RoutingAviary.OBSTACLE_IDS.append(
                p.loadURDF("cube.urdf", 
                        [2.5, 4, 1.5], 
                        p.getQuaternionFromEuler([0,0,0]), 
                        useFixedBase = True, 
                        physicsClientId=self.CLIENT)) 
            RoutingAviary.OBSTACLE_IDS.append(
                p.loadURDF("cube.urdf", 
                        [3.5, 4, 1.5], 
                        p.getQuaternionFromEuler([0,0,0]), 
                        useFixedBase = True, 
                        physicsClientId=self.CLIENT)) 
            
        # self._getObstaclesData()
    ################################################################################
    
    def _detectCollision(self):
       # print("Processing Collision Detection . . .")
       for i in range(self.NUM_DRONES):
           self.CONTACT_POINTS[i] = p.getContactPoints(self.DRONE_IDS[i])
 
           if len(self.CONTACT_POINTS[i]) > 0:
               self.CONTACT_FLAGS[i] = 1
               print("Agent" + str(i) + ": Collided !!!!!!!!!!!!!!!")
           else:
               self.CONTACT_FLAGS[i] = 0
               
    ################################################################################   
 
    def _getObstaclesData(self):
        idsList = RoutingAviary.OBSTACLE_IDS
        # idsList = list(self.DRONE_IDS) + idsList   # Include Drone's ids
        
        droneList = list(self.DRONE_IDS)
        observationRange = 1.5;  # [m] (to be matched with raycast range)
    
        # Store obstacles data
        for j in range(len(idsList)):
     
            pos, orn = p.getBasePositionAndOrientation(idsList[j])
            vsd = p.getVisualShapeData(idsList[j])
            
            self.OBSTACLE_DATA[str(idsList[j])] = {"position": pos,
                                                   "size": vsd[0][3]}
            
            for k in range(self.NUM_DRONES):
                csp = p.getClosestPoints(self.DRONE_IDS[k], idsList[j], observationRange)
                if len(csp)!=0:
                    self.OBSTACLE_DATA[str(idsList[j])]["closestPoint"] = csp[0][5]
            # for i in range(len(csp)):
            #     print(i, csp[i][5])
            
        # Store drones data
        for j in range(len(droneList)):
            pos_drone, orn_drone = p.getBasePositionAndOrientation(droneList[j])
            vsd_drone = p.getVisualShapeData(droneList[j])
            
            mod_vsd_drone = np.array(vsd_drone[0][3])/10
            drone_size = tuple(mod_vsd_drone)
            self.OBSTACLE_DATA[str(droneList[j])] = {"position": pos_drone,
                                                   "size": drone_size}
 
    ################################################################################
    def _applyForceToObstacle(self):
        # print("Applying forces")
        # Apply force to object -> dynamic obstacles
        t = self.step_counter/2
        altering_step = 40
        
        sign = 1 if (t // altering_step) % 2 == 0 else -1
        if len(self.OBSTACLE_IDS) != 0:
            for j in self.OBSTACLE_IDS:
                if j%2 == 0:
                    sign *= -1
                p.changeDynamics(j, -1, linearDamping=2)
                pos, orn = p.getBasePositionAndOrientation(j)
                p.applyExternalForce(j, -1, [6*sign,sign,9.81], pos, flags=p.WORLD_FRAME)   
                
    
    ################################################################################
    
    def _actionSpace(self):
        """Returns the action space of the environment.

        Returns
        -------
        spaces.Box
            An ndarray of shape (NUM_DRONES, 4) for the commanded RPMs.

        """
        #### Action vector ######## P0            P1            P2            P3
        act_lower_bound = np.array([[0.,           0.,           0.,           0.] for i in range(self.NUM_DRONES)])
        act_upper_bound = np.array([[self.MAX_RPM, self.MAX_RPM, self.MAX_RPM, self.MAX_RPM] for i in range(self.NUM_DRONES)])
        return spaces.Box(low=act_lower_bound, high=act_upper_bound, dtype=np.float32)
    
    ################################################################################

    def _observationSpace(self):
        """Returns the observation space of the environment.

        Returns
        -------
        spaces.Box
            The observation space, i.e., an ndarray of shape (NUM_DRONES, 20).

        """
        #### Observation vector ### X        Y        Z       Q1   Q2   Q3   Q4   R       P       Y       VX       VY       VZ       WX       WY       WZ       P0            P1            P2            P3
        obs_lower_bound = np.array([[-np.inf, -np.inf, 0.,     -1., -1., -1., -1., -np.pi, -np.pi, -np.pi, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, 0.,           0.,           0.,           0.] for i in range(self.NUM_DRONES)])
        obs_upper_bound = np.array([[np.inf,  np.inf,  np.inf, 1.,  1.,  1.,  1.,  np.pi,  np.pi,  np.pi,  np.inf,  np.inf,  np.inf,  np.inf,  np.inf,  np.inf,  self.MAX_RPM, self.MAX_RPM, self.MAX_RPM, self.MAX_RPM] for i in range(self.NUM_DRONES)])
        return spaces.Box(low=obs_lower_bound, high=obs_upper_bound, dtype=np.float32)

    ################################################################################

    def _computeObs(self):
        """Returns the current observation of the environment.

        For the value of the state, see the implementation of `_getDroneStateVector()`.

        Returns
        -------
        ndarray
            An ndarray of shape (NUM_DRONES, 20) with the state of each drone.

        """
        return np.array([self._getDroneStateVector(i) for i in range(self.NUM_DRONES)])

    ################################################################################

    def _preprocessAction(self,
                          action
                          ):
        """Pre-processes the action passed to `.step()` into motors' RPMs.

        Clips and converts a dictionary into a 2D array.

        Parameters
        ----------
        action : ndarray
            The (unbounded) input action for each drone, to be translated into feasible RPMs.

        Returns
        -------
        ndarray
            (NUM_DRONES, 4)-shaped array of ints containing to clipped RPMs
            commanded to the 4 motors of each drone.

        """
        return np.array([np.clip(action[i, :], 0, self.MAX_RPM) for i in range(self.NUM_DRONES)])

    ################################################################################

    def _computeReward(self):
        """Computes the current reward value(s).

        Unused as this subclass is not meant for reinforcement learning.

        Returns
        -------
        int
            Dummy value.

        """
        return -1

    ################################################################################
    
    def _computeTerminated(self):
        """Computes the current terminated value(s).

        Unused as this subclass is not meant for reinforcement learning.

        Returns
        -------
        bool
            Dummy value.

        """
        return False
    
    ################################################################################
    
    def _computeTruncated(self):
        """Computes the current truncated value(s).

        Unused as this subclass is not meant for reinforcement learning.

        Returns
        -------
        bool
            Dummy value.

        """
        return False

    ################################################################################
    
    def _computeInfo(self):
        """Computes the current info dict(s).

        Unused as this subclass is not meant for reinforcement learning.

        Returns
        -------
        dict[str, int]
            Dummy value.

        """
        return {"answer": 42} #### Calculated by the Deep Thought supercomputer in 7.5M years
