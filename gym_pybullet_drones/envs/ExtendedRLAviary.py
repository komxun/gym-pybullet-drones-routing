import os
import numpy as np
import pybullet as p
from gymnasium import spaces
from collections import deque

from gym_pybullet_drones.envs.BaseAviary import BaseAviary
from gym_pybullet_drones.envs.RoutingAviary import RoutingAviary

from gym_pybullet_drones.routing.BaseRouting import RouteCommandFlag, SpeedCommandFlag
from gym_pybullet_drones.routing.IFDSRoute import IFDSRoute
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType, ImageType
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl

class ExtendedRLAviary(RoutingAviary):
    """Base single and multi-agent environment class for reinforcement learning."""
    
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
                 obs: ObservationType=ObservationType.KIN,
                 act: ActionType=ActionType.RPM
                 ):
        """Initialization of a generic single and multi-agent RL environment.

        Attributes `vision_attributes` and `dynamics_attributes` are selected
        based on the choice of `obs` and `act`; `obstacles` is set to True 
        and overridden with landmarks for vision applications; 
        `user_debug_gui` is set to False for performance.

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
        obs : ObservationType, optional
            The type of observation space (kinematic information or vision)
        act : ActionType, optional
            The type of action space (1 or 3D; RPMS, thurst and torques, waypoint or velocity with PID control; etc.)

        """
        # =============================================================================
        homePos =  np.array([0,0,0.5]) 
        destin  =  np.array([0.2, 10, 1])
        self.HOME_POS = homePos
        self.DESTIN = destin
        # =============================================================================

        #### Create a buffer for the last .5 sec of actions ########
        # self.ACTION_BUFFER_SIZE = int(ctrl_freq//2)
        self.ACTION_BUFFER_SIZE = 0
        self.action_buffer = deque(maxlen=self.ACTION_BUFFER_SIZE)
        
        ####
        vision_attributes = True if obs == ObservationType.RGB else False
        self.OBS_TYPE = obs
        self.ACT_TYPE = act
        self.COMPUTE_DONE = False
        #### Create integrated controllers #########################
        if act in [ActionType.PID, ActionType.VEL, ActionType.ONE_D_PID, ActionType.AUTOROUTING]:
            os.environ['KMP_DUPLICATE_LIB_OK']='True'
            if drone_model in [DroneModel.CF2X, DroneModel.CF2P]:
                self.ctrl = [DSLPIDControl(drone_model=DroneModel.CF2X) for i in range(num_drones)]
                self.routing = [IFDSRoute(drone_model=DroneModel.CF2X, drone_id=i) for i in range(num_drones)]
                
                for j in range(len(self.routing)):
                    self.routing[j].HOME_POS = homePos
                    self.routing[j].DESTINATION = destin
            else:
                print("[ERROR] in BaseRLAviary.__init()__, no controller is available for the specified drone_model")

        #### Create a buffer for the last .5 sec of Sensors ########
        # self.SENSOR_BUFFER_SIZE = int(ctrl_freq//2)  # 5: five informations from raycast (obj_id, hit_fraction, (hit_xyz))
        # self.SENSOR_BUFFER_SIZE = 2   # 5: five informations from raycast (obj_id, hit_fraction, (hit_xyz))
        self.SENSOR_BUFFER_SIZE = 1   # 5: five informations from raycast (obj_id, hit_fraction, (hit_xyz))
        self.sensor_buffer = deque(maxlen=self.SENSOR_BUFFER_SIZE)
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
                         obstacles=True, # Add obstacles for RGB observations and/or FlyThruGate
                         user_debug_gui=False, # Remove of RPM sliders from all single agent learning aviaries
                         vision_attributes=vision_attributes,
                         )
        #### Set a limit on the maximum target speed ###############
        if act == ActionType.VEL:
            self.SPEED_LIMIT = 0.03 * self.MAX_SPEED_KMH * (1000/3600)

    ################################################################################

    def _actionSpace(self):
        """Returns the action space of the environment.

        Returns
        -------
        spaces.Box
            A Box of size NUM_DRONES x 4, 3, or 1, depending on the action type.

        """
        if self.ACT_TYPE == ActionType.AUTOROUTING:
            size = 1
            for i in range(self.ACTION_BUFFER_SIZE):
                self.action_buffer.append(np.zeros((self.NUM_DRONES,size)))
            return spaces.Discrete(5)  # 5 discrete actions, details in _preprocessAction()
        else:
            if self.ACT_TYPE in [ActionType.RPM, ActionType.VEL]:
                size = 4
            elif self.ACT_TYPE==ActionType.PID:
                size = 3
            elif self.ACT_TYPE in [ActionType.ONE_D_RPM, ActionType.ONE_D_PID]:
                size = 1
            else:
                print("[ERROR] in BaseRLAviary._actionSpace()")
                exit()
            act_lower_bound = np.array([-1*np.ones(size) for i in range(self.NUM_DRONES)])
            act_upper_bound = np.array([+1*np.ones(size) for i in range(self.NUM_DRONES)])
            #
            for i in range(self.ACTION_BUFFER_SIZE):
                self.action_buffer.append(np.zeros((self.NUM_DRONES,size)))
            #
            return spaces.Box(low=act_lower_bound, high=act_upper_bound, dtype=np.float32)

    ################################################################################

    def _preprocessAction(self,
                          action
                          ):
        """Pre-processes the action passed to `.step()` into motors' RPMs.

        Parameter `action` is processed differenly for each of the different
        action types: the input to n-th drone, `action[n]` can be of length
        1, 3, or 4, and represent RPMs, desired thrust and torques, or the next
        target position to reach using PID control, etc.

        Parameters
        ----------
        action : ndarray
            The input action for each drone, to be translated into RPMs.

        Returns
        -------
        ndarray
            (NUM_DRONES, 4)-shaped array of ints containing to clipped RPMs
            commanded to the 4 motors of each drone.

        """
        # p.removeAllUserDebugItems()
        self.action_buffer.append(np.array([[float(action)]])) # Need to revise this to have N-number of drones
                                                        # (similar to [[discrete_act_lo] for i in range(self.NUM_DRONES)])])
        rpm = np.zeros((self.NUM_DRONES,4))
        for k in range(1):  # k: num drone
            # Process action based on ACT_TYPE
            if self.ACT_TYPE == ActionType.AUTOROUTING:
                state = self._getDroneStateVector(k)
            
                # Initially set to accelerate
                # self.routing._setCommand(SpeedCommandFlag, "accelerate", 0.02)
                # action = 1
                
                #------- Compute route (waypoint) to follow ----------------
                foundPath, path = self.routing[k].computeRouteFromState(route_timestep=self.routing[k].route_counter, 
                                                                    state = state, 
                                                                    home_pos = self.HOME_POS, 
                                                                    target_pos = self.DESTIN,
                                                                    speed_limit = self.SPEED_LIMIT,
                                                                    obstacle_data = self.OBSTACLE_DATA,
                                                                    drone_ids = self.DRONE_IDS
                                                                    )
                
                if self.routing[k].route_counter == 1:
                    if foundPath>0:
                        print("Calculating Global Route . . .")
                        self.routing[k].setGlobalRoute(path)
                    else:
                        raise ValueError("[Error] Global route was not found. Mission aborted.")        

                # ==== PASSIVE BEHAVIOUR ======
                
                if action ==0:
                    self.routing[k]._setCommand(SpeedCommandFlag, "constant")
                    self.routing[k]._setCommand(RouteCommandFlag, "follow_global")
                elif action ==1:
                    self.routing[k]._setCommand(SpeedCommandFlag, "constant")
                    self.routing[k]._setCommand(RouteCommandFlag, "follow_local")
                elif action ==2:
                    self.routing[k]._setCommand(SpeedCommandFlag, "accelerate", 0.05)
                elif action ==3:
                    self.routing[k]._setCommand(SpeedCommandFlag, "accelerate", -4)
                elif action==4:
                    self.routing[k]._setCommand(SpeedCommandFlag, "hover")
                
                # self.routing[k]._setCommand(RouteCommandFlag, "follow_global")
                # if action ==0:
                #     self.routing[k]._setCommand(SpeedCommandFlag, "accelerate", 0)
                # elif action ==1:
                #     self.routing[k]._setCommand(SpeedCommandFlag, "accelerate", -4)
                # elif action==2:
                #     self.routing[k]._setCommand(SpeedCommandFlag, "hover")
                    
                #     if self.routing.STAT:
                #         print("Alert: no route found!--> following global route")
                #         self.routing._setCommand(RouteCommandFlag, "follow_global")
                else:
                    print("[ERROR] in ExtendedSingleAgentAviary._preprocessAction()")
                    raise ValueError(f"Invalid action: {action}")
                
                #### Compute control for the current way point #############
                rpm_k, _, _ = self.ctrl[k].computeControlFromState(control_timestep=self.CTRL_TIMESTEP,
                                                                state=state,
                                                                target_pos = self.routing[k].TARGET_POS, 
                                                                target_rpy = self.INIT_RPYS[k, :],
                                                                target_vel = self.routing[k].TARGET_VEL
                                                                )
                rpm[k,:] = rpm_k
            # ================================    
            else:
                print("[ERROR] in BaseRLAviary._preprocessAction()")
                exit()
        return rpm

    ################################################################################

    def _observationSpace(self):
        """Returns the observation space of the environment.

        Returns
        -------
        ndarray
            A Box() of shape (NUM_DRONES,H,W,4) or (NUM_DRONES,12) depending on the observation type.

        """
        if self.OBS_TYPE == ObservationType.RGB:
            return spaces.Box(low=0,
                              high=255,
                              shape=(self.NUM_DRONES, self.IMG_RES[1], self.IMG_RES[0], 4), dtype=np.uint8)
        elif self.OBS_TYPE == ObservationType.KIN:
            ############################################################
            #### OBS SPACE OF SIZE 12
            #### Observation vector ### X        Y        Z       Q1   Q2   Q3   Q4   R       P       Y       VX       VY       VZ       WX       WY       WZ
            lo = -np.inf
            hi = np.inf
            obs_lower_bound = np.array([[lo,lo,0, lo,lo,lo,lo,lo,lo,lo,lo,lo] for i in range(self.NUM_DRONES)])
            obs_upper_bound = np.array([[hi,hi,hi,hi,hi,hi,hi,hi,hi,hi,hi,hi] for i in range(self.NUM_DRONES)])
            #### Add action buffer to observation space ################
            act_lo = -1
            act_hi = +1
            # print(f"action buffer size = {self.ACTION_BUFFER_SIZE}")
            # print(f"sensor buffer size = {self.SENSOR_BUFFER_SIZE}")
            for i in range(self.ACTION_BUFFER_SIZE):
                if self.ACT_TYPE == ActionType.AUTOROUTING:
                    discrete_act_lo = 0
                    discrete_act_hi = 4
                    obs_lower_bound = np.hstack([obs_lower_bound, np.array([[discrete_act_lo] for i in range(self.NUM_DRONES)])])
                    obs_upper_bound = np.hstack([obs_upper_bound, np.array([[discrete_act_hi] for i in range(self.NUM_DRONES)])])
                    
                else:
                    if self.ACT_TYPE in [ActionType.RPM, ActionType.VEL]:
                        obs_lower_bound = np.hstack([obs_lower_bound, np.array([[act_lo,act_lo,act_lo,act_lo] for i in range(self.NUM_DRONES)])])
                        obs_upper_bound = np.hstack([obs_upper_bound, np.array([[act_hi,act_hi,act_hi,act_hi] for i in range(self.NUM_DRONES)])])
                    elif self.ACT_TYPE==ActionType.PID:
                        obs_lower_bound = np.hstack([obs_lower_bound, np.array([[act_lo,act_lo,act_lo] for i in range(self.NUM_DRONES)])])
                        obs_upper_bound = np.hstack([obs_upper_bound, np.array([[act_hi,act_hi,act_hi] for i in range(self.NUM_DRONES)])])
                    elif self.ACT_TYPE in [ActionType.ONE_D_RPM, ActionType.ONE_D_PID]:
                        obs_lower_bound = np.hstack([obs_lower_bound, np.array([[act_lo] for i in range(self.NUM_DRONES)])])
                        obs_upper_bound = np.hstack([obs_upper_bound, np.array([[act_hi] for i in range(self.NUM_DRONES)])])
            
            ray_lo = np.tile([-1 ,0, -np.inf, -np.inf, -np.inf], self.routing[0].NUM_RAYS)
            ray_hi = np.tile([np.inf ,1, np.inf, np.inf, np.inf], self.routing[0].NUM_RAYS)
            # ++++++ Add distance-to-destination to observation space ++++++
            obs_lower_bound = np.hstack([obs_lower_bound, np.array([[0] for i in range(self.NUM_DRONES)])])
            obs_upper_bound = np.hstack([obs_upper_bound, np.array([[np.inf] for i in range(self.NUM_DRONES)])])
           
            for _ in range(self.SENSOR_BUFFER_SIZE):
                self.sensor_buffer.append(np.zeros((self.NUM_DRONES, 5*self.routing[0].NUM_RAYS))) # 5: info from rayCast
                #++++++ Add sensor buffer to observation space +++++++++++++
                # For now, sensor is RayCasting -> need to generalize observation to more types of sensors
                # Rayinfo: [obj_id,  hit_fraction,  hitPos_x,  hitPos_y,  hitPos_z] per ray
                obs_lower_bound = np.hstack([obs_lower_bound, np.array([list(ray_lo) for _ in range(self.NUM_DRONES)])])
                obs_upper_bound = np.hstack([obs_upper_bound, np.array([list(ray_hi) for _ in range(self.NUM_DRONES)])])
            ############################################################
            # added 20241015
            obs_lower_bound =  obs_lower_bound.reshape(obs_lower_bound.shape[1],)
            obs_upper_bound =  obs_upper_bound.reshape(obs_upper_bound.shape[1],)
            return spaces.Box(low=obs_lower_bound, high=obs_upper_bound, dtype=np.float32)
        else:
            print("[ERROR] in BaseRLAviary._observationSpace()")
    
    ################################################################################

    def _computeObs(self):
        """Returns the current observation of the environment.

        Returns
        -------
        ndarray
            A Box() of shape (NUM_DRONES,H,W,4) or (NUM_DRONES,12) depending on the observation type.

        """
        if self.OBS_TYPE == ObservationType.RGB:
            if self.step_counter%self.IMG_CAPTURE_FREQ == 0:
                for i in range(self.NUM_DRONES):
                    self.rgb[i], self.dep[i], self.seg[i] = self._getDroneImages(i,
                                                                                 segmentation=False
                                                                                 )
                    #### Printing observation to PNG frames example ############
                    if self.RECORD:
                        self._exportImage(img_type=ImageType.RGB,
                                          img_input=self.rgb[i],
                                          path=self.ONBOARD_IMG_PATH+"drone_"+str(i),
                                          frame_num=int(self.step_counter/self.IMG_CAPTURE_FREQ)
                                          )
            return np.array([self.rgb[i] for i in range(self.NUM_DRONES)]).astype('float32')
        elif self.OBS_TYPE == ObservationType.KIN:
            ############################################################
            #### OBS SPACE OF SIZE 12
            obs_12 = np.zeros((self.NUM_DRONES,12))
            for i in range(self.NUM_DRONES):
                #obs = self._clipAndNormalizeState(self._getDroneStateVector(i))
                obs = self._getDroneStateVector(i)
                # (x, y, z, R, P, Y, vx, vy, vz, wx, wy,)
                obs_12[i, :] = np.hstack([obs[0:3], obs[7:10], obs[10:13], obs[13:16]]).reshape(12,)
                self.sensor_buffer.append(np.array([list(self.routing[i].RAYS_INFO)])) 
                
            ret = np.array([obs_12[i, :] for i in range(self.NUM_DRONES)]).astype('float32')
            #### Add action buffer to observation #######################
            for i in range(self.ACTION_BUFFER_SIZE):
                ret = np.hstack([ret, np.array([self.action_buffer[i][j, :] for j in range(self.NUM_DRONES)])])
         
            # #++++++ Add distance-to-destination +++++++++++++++++++++++++
            ret = np.hstack([ret, np.array([[self.routing[i].getDistanceToDestin()] for i in range(self.NUM_DRONES)])])
          
            # #++++++ Add sensor buffer to observation  +++++++++++++++++++
            for i in range(self.SENSOR_BUFFER_SIZE):
                ret = np.hstack([ret, np.array([self.sensor_buffer[i][j, :] for j in range(self.NUM_DRONES)])])
            ret = ret.reshape(ret.shape[1], ).astype('float32')
            return ret
            ############################################################
        else:
            print("[ERROR] in ExtendedRLAviary._computeObs()")
