import os
import numpy as np
from gymnasium import spaces
from datetime import datetime
from gym_pybullet_drones.envs.BaseAviary import BaseAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ImageType
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.routing.BaseRouting import RouteCommandFlag, SpeedCommandFlag, SpeedStatus
from gym_pybullet_drones.routing.IFDSRoute import IFDSRoute

import pybullet as p
from PIL import Image

class MASAviary(BaseAviary):
    """Multi-drone environment class for control applications."""
    
    OBSTACLE_IDS = set([])

    ################################################################################

    def __init__(self,
                  drone_model: DroneModel=DroneModel.CF2X,
                  num_drones: int=1,
                  num_other_drones: int=5,
                  neighbourhood_radius: float=np.inf,
                  initial_xyzs=None,
                  initial_rpys=None,
                  physics: Physics=Physics.PYB,
                  pyb_freq: int = 240,
                  ctrl_freq: int = 240,
                  gui=False,
                  record=False,
                  obstacles=True,
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
        self.INIT_XYZS = initial_xyzs
        self.INIT_RPYS = initial_rpys
        self.NUM_OTHER_DRONES = num_other_drones
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

        self._initializeOtherDrones()
        #### Set a limit on the maximum target speed ###############
        speedLimitingFactor = 0.03  #0.1 # 0.03
        self.SPEED_LIMIT = speedLimitingFactor * self.MAX_SPEED_KMH * (1000/3600)
        self.CONTACT_POINTS = [() for _ in range(self.NUM_DRONES)]
        self.CONTACT_FLAGS = np.zeros(self.NUM_DRONES, dtype=int)
        self.OBSTACLE_DATA = {}
        self._getObstaclesData()
        
    def _initializeOtherDrones(self):
        H_STEP = 0.02
        R = 2
        R_D = 4
        num_drones = self.NUM_OTHER_DRONES
        INIT_XYZS = np.zeros((num_drones, 3))
        INIT_RPYS = np.zeros((num_drones, 3))
        DESTINS = np.zeros((num_drones, 3))
        self.routeCounter = np.ones(num_drones)
        self.OTHER_DRONES_ACTIONS = np.zeros((self.NUM_OTHER_DRONES,4))

        # Loop over the range of num_drones
        INIT_XYZS[0] = [0,0,0.8]
        INIT_RPYS[0] = [0,0,0]
        DESTINS[0] = [0,5,1]
        ORIGIN = [0,4,1]

        for i in range(1, num_drones):
            # Initialize INIT_XYZS
            INIT_XYZS[i] = [ORIGIN[0]+(R)*np.cos((i/num_drones)*2*np.pi+np.pi/2),
                            ORIGIN[1]+(R)*np.sin((i/num_drones)*2*np.pi+np.pi/2)-(R), 
                            ORIGIN[2]+H_STEP]
            # INIT_XYZS[i] = [ORIGIN[0]+(R)*np.cos((i/6)*2*np.pi+np.pi/2),
            #                 ORIGIN[1]+(R)*np.sin((i/6)*2*np.pi+np.pi/2)-(R), 
            #                 ORIGIN[2]+H_STEP*i]
            
            # Initialize INIT_RPYS
            INIT_RPYS[i] = [0, 0, i * (np.pi/2)/num_drones]
            
            # Initialize DESTINS
            DESTINS[i] = [ORIGIN[0]+(R_D)*np.cos((i/num_drones)*2*np.pi+np.pi/2+1*np.pi/2),
                            ORIGIN[1]+np.sin((i/num_drones)*2*np.pi+np.pi/2+3*np.pi/2)-(R_D), 
                            ORIGIN[2]]
            self.ctrl = [DSLPIDControl(drone_model=DroneModel.CF2X) for i in range(num_drones)]
            self.routing = [IFDSRoute(drone_model=DroneModel.CF2X, drone_id=i) for i in range(num_drones)]
            for j in range(len(self.routing)):
                self.routing[j].HOME_POS = INIT_XYZS[i]
                self.routing[j].DESTINATION = DESTINS[i]
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
            # print(f"action = {action}")
            clipped_action = np.reshape(self._preprocessAction(action), (self.NUM_DRONES, 4))
        #### Repeat for as many as the aggregate physics steps #####
        for _ in range(self.PYB_STEPS_PER_CTRL):
            p.performCollisionDetection(physicsClientId=self.CLIENT)
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
                # self._applyForceToObstacle()
                p.performCollisionDetection(physicsClientId=self.CLIENT)
                p.stepSimulation(physicsClientId=self.CLIENT)
                
            #### Save the last applied action (e.g. to compute drag) ###
            self.last_clipped_action = clipped_action
        #### Update and store the drones kinematic information #####
        self._flyOtherDrones()
        self._applyForceToObstacle()
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

    def reset(self,
              seed : int = None,
              options : dict = None):
        """Resets the environment.

        Parameters
        ----------
        seed : int, optional
            Random seed.
        options : dict[..], optional
            Additinonal options, unused

        Returns
        -------
        ndarray | dict[..]
            The initial observation, check the specific implementation of `_computeObs()`
            in each subclass for its format.
        dict[..]
            Additional information as a dictionary, check the specific implementation of `_computeInfo()`
            in each subclass for its format.

        """

        # TODO : initialize random number generator with seed
        p.resetSimulation(physicsClientId=self.CLIENT)
        #### Housekeeping ##########################################
        self._housekeeping()
        #### Update and store the drones kinematic information #####
        self._updateAndStoreKinematicInformation()
        p.performCollisionDetection(physicsClientId=self.CLIENT)
        self._detectCollision()
        #### Start video recording #################################
        self._startVideoRecording()
        #### Return the initial observation ########################
        initial_obs = self._computeObs()
        initial_info = self._computeInfo()
        return initial_obs, initial_info
    
    ################################################################################
    def save_object_data(func):
        def wrapper(*arg, **kwarg):
            id = func(*arg, **kwarg)
            MASAviary.OBSTACLE_IDS.add(id)
            return id
        return wrapper
    
    def _addObstacles(self):
        """Add obstacles to the environment.

        These obstacles are loaded from standard URDF files included in Bullet.

        """

        scene = 0
        if scene == 0:
            pass

        elif scene == 1:
            MASAviary.OBSTACLE_IDS.add(
                p.loadURDF("cube.urdf",
                        [0, 2, .5],
                        p.getQuaternionFromEuler([0,0,0]),
                        useFixedBase = True, 
                        physicsClientId=self.CLIENT
                        ))
            
            
            MASAviary.OBSTACLE_IDS.add(
                p.loadURDF("cube.urdf", 
                        [0.5, 3, 0.5], 
                        p.getQuaternionFromEuler([0,0,0]), 
                        useFixedBase = True, 
                        physicsClientId=self.CLIENT))
            
            MASAviary.OBSTACLE_IDS.add(
                p.loadURDF("cube.urdf", 
                        [-0.2, 5, 0+1], 
                        p.getQuaternionFromEuler([0,0,0]), 
                        useFixedBase = True, 
                        physicsClientId=self.CLIENT))
            
            MASAviary.OBSTACLE_IDS.add(
                p.loadURDF("cube.urdf", 
                        [0.8, 7, 0], 
                        p.getQuaternionFromEuler([0,0,0]), 
                        useFixedBase = True, 
                        physicsClientId=self.CLIENT))    
            
            MASAviary.OBSTACLE_IDS.add(
                p.loadURDF("cube.urdf", 
                        [1, 4, 0], 
                        p.getQuaternionFromEuler([0,0,0]), 
                        useFixedBase = True, 
                        physicsClientId=self.CLIENT)) 
            
            MASAviary.OBSTACLE_IDS.add(
                p.loadURDF("cube.urdf", 
                        [2.5, 4, 1.5], 
                        p.getQuaternionFromEuler([0,0,0]), 
                        useFixedBase = True, 
                        physicsClientId=self.CLIENT)) 
            MASAviary.OBSTACLE_IDS.add(
                p.loadURDF("cube.urdf", 
                        [3.5, 4, 1.5], 
                        p.getQuaternionFromEuler([0,0,0]), 
                        useFixedBase = True, 
                        physicsClientId=self.CLIENT)) 
            MASAviary.OBSTACLE_IDS.add(
                p.loadURDF("cube.urdf", 
                        [3.5, 6, 2.5], 
                        p.getQuaternionFromEuler([0,0,0]), 
                        useFixedBase = True, 
                        physicsClientId=self.CLIENT)) 
            MASAviary.OBSTACLE_IDS.add(
                p.loadURDF("cube.urdf", 
                        [4.5, 7, 1.5], 
                        p.getQuaternionFromEuler([0,0,0]), 
                        useFixedBase = True, 
                        physicsClientId=self.CLIENT)) 
            MASAviary.OBSTACLE_IDS.add(
                p.loadURDF("cube.urdf", 
                        [3, 6, 0.5], 
                        p.getQuaternionFromEuler([0,0,0]), 
                        useFixedBase = True, 
                        physicsClientId=self.CLIENT)) 
            
            
            MASAviary.OBSTACLE_IDS.add(
                p.loadURDF("cube.urdf", 
                        [1.5, 8.5, 2.8], 
                        p.getQuaternionFromEuler([0,0,0]), 
                        useFixedBase = True, 
                        physicsClientId=self.CLIENT)) 
            # MASAviary.OBSTACLE_IDS.append(
            #     p.loadURDF("cube.urdf", 
            #             [2.5, 8.5, 1.5], 
            #             p.getQuaternionFromEuler([0,0,0]), 
            #             useFixedBase = True, 
            #             physicsClientId=self.CLIENT)) 
            MASAviary.OBSTACLE_IDS.add(
                p.loadURDF("sphere2.urdf", 
                        [1.5, 9.5, 1], 
                        p.getQuaternionFromEuler([0,0,0]), 
                        useFixedBase = True, 
                        physicsClientId=self.CLIENT)) 
            
            MASAviary.OBSTACLE_IDS.add(
                p.loadURDF("cube.urdf", 
                        [-0.5, 8, 1], 
                        p.getQuaternionFromEuler([0,0,0]), 
                        useFixedBase = True, 
                        physicsClientId=self.CLIENT)) 
            
            
        elif scene == 2:
            
            # Moving obstacles
            id = p.loadURDF("sphere2.urdf",
                        [0, 2, .5],
                        p.getQuaternionFromEuler([0,0,0]),
                        useFixedBase = False, 
                        globalScaling = 1.2, 
                        physicsClientId=self.CLIENT
                        )
            MASAviary.OBSTACLE_IDS.add(id)
            p.changeDynamics(id, -1, mass=1, linearDamping=2)

            
            id = p.loadURDF("sphere2.urdf", 
                        [0, 5, 0.5], 
                        p.getQuaternionFromEuler([0,0,0]), 
                        globalScaling = 0.8,
                        useFixedBase = False, 
                        physicsClientId=self.CLIENT)
            MASAviary.OBSTACLE_IDS.add(id)
            p.changeDynamics(id, -1, mass=1, linearDamping=2)
            
            id = p.loadURDF("sphere2.urdf", 
                        [-0.5, 3.5, 0.5], 
                        p.getQuaternionFromEuler([0,0,0]), 
                        globalScaling = 1,
                        useFixedBase = False,
                        physicsClientId=self.CLIENT)
            MASAviary.OBSTACLE_IDS.add(id) 
            p.changeDynamics(id, -1, linearDamping=2, mass=1)
            
            # Static obstacle
            MASAviary.OBSTACLE_IDS.add(
                p.loadURDF("cube.urdf", 
                        [2, 5, 0.5], 
                        p.getQuaternionFromEuler([0,0,0]), 
                        useFixedBase = True, 
                        globalScaling = 1.2, 
                        physicsClientId=self.CLIENT,
                        )) 
            MASAviary.OBSTACLE_IDS.add(
                p.loadURDF("cube.urdf", 
                        [-2, 5, 0.5], 
                        p.getQuaternionFromEuler([0,0,0]), 
                        useFixedBase = True, 
                        globalScaling = 1.2, 
                        physicsClientId=self.CLIENT)) 
            
            MASAviary.OBSTACLE_IDS.add(
                p.loadURDF("cube.urdf", 
                        [2, 5, 0.5], 
                        p.getQuaternionFromEuler([0,0,0]), 
                        useFixedBase = True, 
                        globalScaling = 1.2, 
                        physicsClientId=self.CLIENT)) 
            
            MASAviary.OBSTACLE_IDS.add(
                p.loadURDF("cube.urdf", 
                        [-3, 2, 0.5], 
                        p.getQuaternionFromEuler([0,0,0]), 
                        useFixedBase = True, 
                        globalScaling = 1.2, 
                        physicsClientId=self.CLIENT)) 
            MASAviary.OBSTACLE_IDS.add(
                p.loadURDF("cube.urdf", 
                        [2, 2, 0.5], 
                        p.getQuaternionFromEuler([0,0,0]), 
                        useFixedBase = True, 
                        physicsClientId=self.CLIENT))
            MASAviary.OBSTACLE_IDS.add(
                p.loadURDF("cube.urdf", 
                        [-2.5, 3.5, 0.5], 
                        p.getQuaternionFromEuler([0,0,0]), 
                        useFixedBase = True, 
                        physicsClientId=self.CLIENT))   
            
            MASAviary.OBSTACLE_IDS.add(
                p.loadURDF("cube.urdf", 
                        [4.5, 7, 1.5], 
                        p.getQuaternionFromEuler([0,0,0]), 
                        useFixedBase = True, 
                        physicsClientId=self.CLIENT)) 
            MASAviary.OBSTACLE_IDS.add(
                p.loadURDF("cube.urdf", 
                        [3, 6, 0.5], 
                        p.getQuaternionFromEuler([0,0,0]), 
                        useFixedBase = True, 
                        physicsClientId=self.CLIENT)) 
            
            
            MASAviary.OBSTACLE_IDS.add(
                p.loadURDF("cube.urdf", 
                        [2, 8.5, 2.5], 
                        p.getQuaternionFromEuler([0,0,0]), 
                        useFixedBase = True, 
                        physicsClientId=self.CLIENT)) 
            
            
            MASAviary.OBSTACLE_IDS.add(
                p.loadURDF("cube.urdf", 
                        [0.8, 7, 0], 
                        p.getQuaternionFromEuler([0,0,0]), 
                        useFixedBase = True, 
                        physicsClientId=self.CLIENT))    
            
            MASAviary.OBSTACLE_IDS.add(
                p.loadURDF("cube.urdf", 
                        [2, 3.5, 0], 
                        p.getQuaternionFromEuler([0,0,0]), 
                        useFixedBase = True, 
                        physicsClientId=self.CLIENT)) 
            
            MASAviary.OBSTACLE_IDS.add(
                p.loadURDF("cube.urdf", 
                        [2.5, 4, 1.5], 
                        p.getQuaternionFromEuler([0,0,0]), 
                        useFixedBase = True, 
                        physicsClientId=self.CLIENT)) 
            MASAviary.OBSTACLE_IDS.add(
                p.loadURDF("cube.urdf", 
                        [3.5, 4, 1.5], 
                        p.getQuaternionFromEuler([0,0,0]), 
                        useFixedBase = True, 
                        physicsClientId=self.CLIENT)) 
        elif scene == 3:
            
            # Moving obstacles
            id = p.loadURDF("sphere2.urdf",
                        [0, 2, .5],
                        p.getQuaternionFromEuler([0,0,0]),
                        useFixedBase = False, 
                        globalScaling = 1.2, 
                        physicsClientId=self.CLIENT
                        )
            MASAviary.OBSTACLE_IDS.add(id)
            p.changeDynamics(id, -1, mass=1, linearDamping=2)

            
            id = p.loadURDF("sphere2.urdf", 
                        [0, 5, 0.5], 
                        p.getQuaternionFromEuler([0,0,0]), 
                        globalScaling = 0.8,
                        useFixedBase = False, 
                        physicsClientId=self.CLIENT)
            MASAviary.OBSTACLE_IDS.add(id)
            p.changeDynamics(id, -1, mass=1, linearDamping=2)
            
            id = p.loadURDF("sphere2.urdf", 
                        [-0.5, 3.5, 0.5], 
                        p.getQuaternionFromEuler([0,0,0]), 
                        globalScaling = 1,
                        useFixedBase = False,
                        physicsClientId=self.CLIENT)
            MASAviary.OBSTACLE_IDS.add(id) 
            p.changeDynamics(id, -1, linearDamping=2, mass=1)
            
            # Static obstacle
            MASAviary.OBSTACLE_IDS.add(
                p.loadURDF("cube.urdf", 
                        [2, 5, 0.5], 
                        p.getQuaternionFromEuler([0,0,0]), 
                        useFixedBase = True, 
                        globalScaling = 1.2, 
                        physicsClientId=self.CLIENT,
                        )) 
            MASAviary.OBSTACLE_IDS.add(
                p.loadURDF("cube.urdf", 
                        [-2, 5, 0.5], 
                        p.getQuaternionFromEuler([0,0,0]), 
                        useFixedBase = True, 
                        globalScaling = 1.2, 
                        physicsClientId=self.CLIENT)) 
            
            MASAviary.OBSTACLE_IDS.add(
                p.loadURDF("cube.urdf", 
                        [2, 5, 0.5], 
                        p.getQuaternionFromEuler([0,0,0]), 
                        useFixedBase = True, 
                        globalScaling = 1.2, 
                        physicsClientId=self.CLIENT)) 
            
            MASAviary.OBSTACLE_IDS.add(
                p.loadURDF("cube.urdf", 
                        [-3, 2, 0.5], 
                        p.getQuaternionFromEuler([0,0,0]), 
                        useFixedBase = True, 
                        globalScaling = 1.2, 
                        physicsClientId=self.CLIENT)) 
            MASAviary.OBSTACLE_IDS.add(
                p.loadURDF("cube.urdf", 
                        [2, 2, 0.5], 
                        p.getQuaternionFromEuler([0,0,0]), 
                        useFixedBase = True, 
                        physicsClientId=self.CLIENT))
            MASAviary.OBSTACLE_IDS.add(
                p.loadURDF("cube.urdf", 
                        [-2.5, 3.5, 0.5], 
                        p.getQuaternionFromEuler([0,0,0]), 
                        useFixedBase = True, 
                        physicsClientId=self.CLIENT))  
            
            # MASAviary.OBSTACLE_IDS.add(
            #     p.loadURDF("cube.urdf", 
            #             [0.5, 3, 0], 
            #             p.getQuaternionFromEuler([0,0,0]), 
            #             useFixedBase = True, 
            #             physicsClientId=self.CLIENT))
            
            MASAviary.OBSTACLE_IDS.add(
                p.loadURDF("cube.urdf", 
                        [0.8, 7, 0.5], 
                        p.getQuaternionFromEuler([0,0,0]), 
                        useFixedBase = True, 
                        physicsClientId=self.CLIENT))    
            
            MASAviary.OBSTACLE_IDS.add(
                p.loadURDF("cube.urdf", 
                        [4.5, 7, 1.5], 
                        p.getQuaternionFromEuler([0,0,0]), 
                        useFixedBase = True, 
                        physicsClientId=self.CLIENT)) 
            MASAviary.OBSTACLE_IDS.add(
                p.loadURDF("cube.urdf", 
                        [3, 6, 0.5], 
                        p.getQuaternionFromEuler([0,0,0]), 
                        useFixedBase = True, 
                        physicsClientId=self.CLIENT)) 
            
            
            MASAviary.OBSTACLE_IDS.add(
                p.loadURDF("cube.urdf", 
                        [2, 8.5, 2.5], 
                        p.getQuaternionFromEuler([0,0,0]), 
                        useFixedBase = True, 
                        physicsClientId=self.CLIENT)) 
            
            
            MASAviary.OBSTACLE_IDS.add(
                p.loadURDF("cube.urdf", 
                        [0.8, 7, 0], 
                        p.getQuaternionFromEuler([0,0,0]), 
                        useFixedBase = True, 
                        physicsClientId=self.CLIENT))    
            
            MASAviary.OBSTACLE_IDS.add(
                p.loadURDF("cube.urdf", 
                        [2, 3.5, 0], 
                        p.getQuaternionFromEuler([0,0,0]), 
                        useFixedBase = True, 
                        physicsClientId=self.CLIENT)) 
            
            MASAviary.OBSTACLE_IDS.add(
                p.loadURDF("cube.urdf", 
                        [2.5, 4, 1.5], 
                        p.getQuaternionFromEuler([0,0,0]), 
                        useFixedBase = True, 
                        physicsClientId=self.CLIENT)) 
            MASAviary.OBSTACLE_IDS.add(
                p.loadURDF("cube.urdf", 
                        [3.5, 4, 1.5], 
                        p.getQuaternionFromEuler([0,0,0]), 
                        useFixedBase = True, 
                        physicsClientId=self.CLIENT))
            
        # self._getObstaclesData()
    ################################################################################

    def _flyOtherDrones(self):
        action = self.OTHER_DRONES_ACTIONS
        obs, _, _, _, _ = self.step(action)
        for j in range(self.NUM_OTHER_DRONES):
            state = self._getDroneStateVector(j)
            if self.routing[j].REACH_DESTIN:
                self.routing[j].reset()
                self.routing[j].DESTINATION = self.routing[j].HOME_POS
                self.routing[j].HOME_POS = self.routing[j].DESTINATION
                self.routeCounter[j] = 1
                
            #------- Compute route (waypoint) to follow ----------------
            foundPath, path = self.routing[j].computeRouteFromState(route_timestep=self.routing[j].route_counter, 
                                                  state = state, 
                                                  home_pos = self.routing[j].CUR_POS, 
                                                  target_pos = self.routing[j].DESTINATION,
                                                  speed_limit = self.SPEED_LIMIT,
                                                  obstacle_data = self.OBSTACLE_DATA,
                                                  drone_ids = self.routing[j].DRONE_ID
                                                  )
        
            if foundPath>0:
                self.routeCounter[j]+=1
                if self.routeCounter[j]>=2 :
                    print("****************Re-calculating destination")
                    self.routing[j].setGlobalRoute(path)
            
            if j==0:
                self.routing[j]._setCommand(SpeedCommandFlag, "accelerate",2)
                self.routing[j]._setCommand(RouteCommandFlag, "follow_local")
            else:
                self.routing[j]._setCommand(SpeedCommandFlag, "accelerate", random.random())
                # routing[j]._setCommand(SpeedCommandFlag, "accelerate",2)
                self.routing[j]._setCommand(RouteCommandFlag, "follow_global")
            
            actions, _, _ = self.ctrl[j].computeControlFromState(control_timestep=self.CTRL_TIMESTEP,
                                                                    state=state,
                                                                    target_pos=self.routing[j].TARGET_POS,
                                                                    target_rpy=self.INIT_RPYS[j, :],
                                                                    target_vel = self.routing[j].TARGET_VEL
                                                                    )
            self.OTHER_DRONES_ACTIONS = actions
    
    
    def _detectCollision(self):
    #    print("Processing Collision Detection . . .")
       for i in range(self.NUM_DRONES):
           self.CONTACT_POINTS[i] = p.getContactPoints(self.DRONE_IDS[i])
 
           if len(self.CONTACT_POINTS[i]) > 0:
               self.CONTACT_FLAGS[i] = 1
            #    print("Agent" + str(i) + ": Collided !")
           else:
               self.CONTACT_FLAGS[i] = 0
               
    ################################################################################   
 
    def _getObstaclesData(self):
        print("Getting obstacle data . . .")
        idsList = MASAviary.OBSTACLE_IDS
        # idsList = list(self.DRONE_IDS) + idsList   # Include Drone's ids
        
        droneList = list(self.DRONE_IDS)
        observationRange = 1.5;  # [m] (to be matched with raycast range)
    
        # Store obstacles data
        for id in idsList:
     
            pos, orn = p.getBasePositionAndOrientation(id)
            vsd = p.getVisualShapeData(id)
            
            self.OBSTACLE_DATA[str(id)] = {"position": pos,
                                                   "size": vsd[0][3]}
            
            for k in range(self.NUM_DRONES):
                csp = p.getClosestPoints(self.DRONE_IDS[k], id, observationRange)
                if len(csp)!=0:
                    self.OBSTACLE_DATA[str(id)]["closestPoint"] = csp[0][5]
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
        
        # Apply force to object -> dynamic obstacles
        t = self.step_counter/self.PYB_FREQ
        # print(f"Applying forces. step_counter = {self.step_counter}")
        # print(f"step_counter={t}")
        altering_sec = 1.5
        
        sign = -1 if (t // altering_sec) % 2 == 0 else 1
        if len(self.OBSTACLE_IDS) != 0:
            for id in self.OBSTACLE_IDS:
                if id%2 == 0:
                    sign *= -1
                # p.changeDynamics(id, -1, linearDamping=2)
                obj_info = p.getDynamicsInfo(id, -1)
                mass = obj_info[0]
                # print(f"Object #{id}: mass = {mass}")
                pos, orn = p.getBasePositionAndOrientation(id)
                p.applyExternalForce(id, -1, [50*sign,0,-mass*9.81], pos, flags=p.WORLD_FRAME) 
        else:
            print("[ERROR] in MASAviary, No obstacles")  
                
    
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
