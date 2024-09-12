import os
from datetime import datetime
from enum import Enum
import numpy as np
from gym import spaces
import pybullet as p
import pybullet_data

from gym_pybullet_drones.envs.BaseAviary import DroneModel, Physics, ImageType
from gym_pybullet_drones.utils.utils import nnlsRPM
# from gym_pybullet_drones.envs.BaseSingleAgentAviary import BaseSingleAgentAviary
from gym_pybullet_drones.envs.RoutingAviary import RoutingAviary

from gym_pybullet_drones.routing.BaseRouting import RouteCommandFlag, SpeedCommandFlag
from gym_pybullet_drones.routing.IFDSRoute import IFDSRoute
from gym_pybullet_drones.control.SimplePIDControl import SimplePIDControl
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ActionType, ObservationType

# class ActionType(ActionType):
#     """Action type enumeration class."""
#     RPM = "rpm"                 # RPMS
#     DYN = "dyn"                 # Desired thrust and torques
#     PID = "pid"                 # PID control
#     VEL = "vel"                 # Velocity input (using PID control)
#     TUN = "tun"                 # Tune the coefficients of a PID controller
#     ONE_D_RPM = "one_d_rpm"     # 1D (identical input to all motors) with RPMs
#     ONE_D_DYN = "one_d_dyn"     # 1D (identical input to all motors) with desired thrust and torques
#     ONE_D_PID = "one_d_pid"     # 1D (identical input to all motors) with PID control
#     # New Action
#     AUTOROUTING = "autorouting"  # Route selection with speed adjustment
    

################################################################################

# class ObservationType(Enum):
#     """Observation type enumeration class."""
#     KIN = "kin"     # Kinematic information (pose, linear and angular velocities)
#     RGB = "rgb"     # RGB camera capture in each drone's POV

################################################################################

class ExtendedSingleAgentAviary(RoutingAviary):
    """Extended single drone environment class for reinforcement learning."""

    ################################################################################
    
    def __init__(self,
                 drone_model: DroneModel=DroneModel.CF2X,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics=Physics.PYB,
                 freq: int=240,
                 aggregate_phy_steps: int=1,
                 gui=False,
                 record=False, 
                 obs: ObservationType=ObservationType.KIN,
                 act: ActionType=ActionType.RPM
                 ):
        """Initialization of a generic single agent RL environment.

        Attribute `num_drones` is automatically set to 1; `vision_attributes`
        and `dynamics_attributes` are selected based on the choice of `obs`
        and `act`; `obstacles` is set to True and overridden with landmarks for
        vision applications; `user_debug_gui` is set to False for performance.

        Parameters
        ----------
        drone_model : DroneModel, optional
            The desired drone type (detailed in an .urdf file in folder `assets`).
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
        obs : ObservationType, optional
            The type of observation space (kinematic information or vision)
        act : ActionType, optional
            The type of action space (1 or 3D; RPMS, thurst and torques, waypoint or velocity with PID control; etc.)

        """
        
        # =============================================================================
        homePos =  np.array([0,0,0.5]) 
        destin  =  np.array([0.2, 12, 1])
        self.HOME_POS = homePos
        self.DESTIN = destin
        # =============================================================================
        
        
        # vision_attributes = True if obs == ObservationType.RGB else False
        # dynamics_attributes = True if act in [ActionType.DYN, ActionType.ONE_D_DYN] else False
        self.OBS_TYPE = obs
        self.ACT_TYPE = act
        self.EPISODE_LEN_SEC = 30
        #### Create integrated Controllers and Routers #########################
        if act in [ActionType.AUTOROUTING]:
            os.environ['KMP_DUPLICATE_LIB_OK']='True'
            if drone_model in [DroneModel.CF2X, DroneModel.CF2P]:
                self.ctrl = DSLPIDControl(drone_model=DroneModel.CF2X)
                self.routing = IFDSRoute(drone_model=DroneModel.CF2X)
            elif drone_model == DroneModel.HB:
                self.ctrl = SimplePIDControl(drone_model=DroneModel.HB)
                self.routing = IFDSRoute(drone_model=DroneModel.HB)
            else:
                print("[ERROR] in ExtendedSingleAgentAviary.__init()__, no controller and router are available for the specified drone_model")
            
            self.routing.HOME_POS = homePos
            self.routing.DESTINATION = destin
            
        super().__init__(drone_model=drone_model,
                         num_drones=1,
                         # initial_xyzs=initial_xyzs,
                         initial_xyzs = homePos.reshape(1,3),
                         initial_rpys = initial_rpys,
                         physics=physics, 
                         freq=freq,
                         aggregate_phy_steps=aggregate_phy_steps,
                         gui=gui,
                         record=record, 
                         obstacles=True, # Add obstacles for RGB observations and/or FlyThruGate
                         user_debug_gui=False, # Remove of RPM sliders from all single agent learning aviaries
                         # vision_attributes=vision_attributes,
                         # dynamics_attributes=dynamics_attributes
                         )
        
        #### Set a limit on the maximum target speed ###############
        if act == ActionType.VEL:
            self.SPEED_LIMIT = 0.03 * self.MAX_SPEED_KMH * (1000/3600)
        #### Try _trajectoryTrackingRPMs exists IFF ActionType.TUN #
        if act == ActionType.TUN and not (hasattr(self.__class__, '_trajectoryTrackingRPMs') and callable(getattr(self.__class__, '_trajectoryTrackingRPMs'))):
                print("[ERROR] in BaseSingleAgentAviary.__init__(), ActionType.TUN requires an implementation of _trajectoryTrackingRPMs in the instantiated subclass")
                exit()
    

    def _actionSpace(self):
        """Returns the action space of the environment.

        Returns
        -------
        ndarray
            A Box() of size 1, 3, 4, or 6 depending on the action type.

        """
        if self.ACT_TYPE == ActionType.AUTOROUTING:
            return spaces.Discrete(2)  # 5 discrete actions, details in _preprocessAction()
            # return spaces.Box(low=0, high=4, shape=(1,), dtype=np.float32)
        else:
            return super()._actionSpace()

    ################################################################################

    def _preprocessAction(self,
                          action
                          ):
        """Pre-processes the action passed to BaseAviary's `.step()` into motors' RPMs.

        Parameter `action` is processed differenly for each of the different
        action types: `action` can be of length 1, 3, 4, or 6 and represent 
        RPMs, desired thrust and torques, the next target position to reach 
        using PID control, a desired velocity vector, new PID coefficients, etc.

        Parameters
        ----------
        action : ndarray
            The input action for each drone, to be translated into RPMs.

        Returns
        -------
        ndarray
            (4,)-shaped array of ints containing to clipped RPMs
            commanded to the 4 motors of each drone.

        """
        # self._applyForceToObstacle()
        if self.ACT_TYPE == ActionType.AUTOROUTING:
            
            
            # CHECK THIS CAREFULLY!!!
            # action = int(action[0])
            
            state = self._getDroneStateVector(0)
            
            # Initially set to accelerate
            # self.routing._setCommand(SpeedCommandFlag, "accelerate", 0.02)
            # action = 1
            
            #------- Compute route (waypoint) to follow ----------------
            foundPath, path = self.routing.computeRouteFromState(route_timestep=self.routing.route_counter, 
                                                                 state = state, 
                                                                 home_pos = self.HOME_POS, 
                                                                 target_pos = self.DESTIN,
                                                                 speed_limit = self.SPEED_LIMIT,
                                                                 obstacle_data = self.OBSTACLE_DATA,
                                                                 drone_ids = self.DRONE_IDS
                                                                 )
        
            if self.routing.route_counter == 1:
                if foundPath>0:
                    print("Calculating Global Route . . .")
                    self.routing.setGlobalRoute(path)
                else:
                    raise ValueError("[Error] Global route was not found. Mission aborted.")
            
            # if action == 0:  # Constant Vel
            #     self.routing._setCommand(SpeedCommandFlag, "accelerate", 0)
            # elif action == 1:  # Accelerate
            #     # self.routing._setCommand(SpeedCommandFlag, "accelerate", 0.02)
            #     self.routing._setCommand(SpeedCommandFlag, "accelerate", 0.1)
            # elif action == 2:  # Decelerate
            #     # self.routing._setCommand(SpeedCommandFlag, "accelerate", -0.06)
            #     self.routing._setCommand(SpeedCommandFlag, "accelerate", -2)
            # elif action == 3:  # Stop (Hover)
            #     self.routing._setCommand(RouteCommandFlag, "follow_global")
            # elif action == 4:  # Stop (Hover)
            #     self.routing._setCommand(RouteCommandFlag, "follow_local")
            # elif action == 5:  # Change Route
            #     self.routing._setCommand(SpeedCommandFlag, "hover")
                
            self.routing._setCommand(SpeedCommandFlag, "accelerate", 0)
            if action ==0:
                self.routing._setCommand(RouteCommandFlag, "follow_global")
                
            elif action ==1:
                self.routing._setCommand(RouteCommandFlag, "follow_local")
                
            #     if self.routing.STAT:
            #         print("Alert: no route found!--> following global route")
            #         self.routing._setCommand(RouteCommandFlag, "follow_global")
            else:
                print("[ERROR] in ExtendedSingleAgentAviary._preprocessAction()")
                raise ValueError(f"Invalid action: {action}")
                
            #### Compute control for the current way point #############
            rpm, _, _ = self.ctrl.computeControlFromState(control_timestep=self.AGGR_PHY_STEPS*self.TIMESTEP,
                                                                    state=state,
                                                                    target_pos = self.routing.TARGET_POS, 
                                                                    target_rpy = np.array([0,0,0]),
                                                                    target_vel = self.routing.TARGET_VEL
                                                                    )
            
            
            # rpm, _, _ = self.ctrl.computeControl(control_timestep=self.AGGR_PHY_STEPS*self.TIMESTEP, 
            #                                      cur_pos=state[0:3],
            #                                      cur_quat=state[3:7],
            #                                      cur_vel=state[10:13],
            #                                      cur_ang_vel=state[13:16],
            #                                      target_pos=self.routing.TARGET_POS,
            #                                      target_vel=self.routing.TARGET_VEL
            #                                      )
            # self.routing._updateCurPos(state[0:3])
            # self.routing._updateCurVel(state[10:13])
            return rpm.astype(np.float32)  # Ensure rpm is returned as float32        
        else:
            return super()._preprocessAction(action)

    ################################################################################
    
    ################################################################################

    def _observationSpace(self):
        """Returns the observation space of the environment.

        Returns
        -------
        ndarray
            A Box() of shape (H,W,4) or (12,) depending on the observation type.

        """
        if self.OBS_TYPE == ObservationType.RGB:
            return spaces.Box(low=0,
                              high=255,
                              shape=(self.IMG_RES[1], self.IMG_RES[0], 4),
                              dtype=np.uint8
                              )
        elif self.OBS_TYPE == ObservationType.KIN:
            ############################################################
            
            #### OBS OF SIZE 20 (WITH QUATERNION AND RPMS)
            # Observations: [X Y Z Q1 Q2 Q3 Q4 R P Y Vx Vy Vz Wx Wy Wz P0 P1 P1 P3 relDist2Destin obj1_range obj1_bearing obj1_dH ...]
            #### Observation vector ###    X   Y    Z   Q1   Q2   Q3   Q4    R    P    Y   VX   VY   VZ   WX    WY    WZ    P0    P1    P2    P3
            # obs_lower_bound = np.array([-1,  -1,  0,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,   -1,   -1,   -1,   -1,   -1,   -1])
            # obs_upper_bound = np.array([ 1,   1,  1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,    1,    1,    1,    1,    1,    1])          
            # return spaces.Box( low=obs_lower_bound, high=obs_upper_bound, dtype=np.float32 )
            
            ############################################################
            #### OBS SPACE OF SIZE 12
            # [X, Y, Z, R, P, Y, Vx, Vy, Vz, Wx, Wy, Wz]
            # return spaces.Box(low=np.array([-1,-1,0, -1,-1,-1, -1,-1,-1, -1,-1,-1]),
            #                   high=np.array([1,1,1, 1,1,1, 1,1,1, 1,1,1]),
            #                   dtype=np.float32
            #                   )
        
            ############################################################
            ##### NEW OBS SPACE OF SIZE 13
            # [X, Y, Z, R, P, Y, Vx, Vy, Vz, Wx, Wy, Wz, D2Destin]
            return spaces.Box(low=np.array([-1,-1,0, -1,-1,-1, -1,-1,-1, -1,-1,-1, 0]),
                              high=np.array([1,1,1, 1,1,1, 1,1,1, 1,1,1, 1]),
                              dtype=np.float32
                              )
        else:
            print("[ERROR] in ExtendedSingleAgentAviary._observationSpace()")
    
    ################################################################################

    def _computeObs(self):
        """Returns the current observation of the environment.

        Returns
        -------
        ndarray
            A Box() of shape (H,W,4) or (12,) depending on the observation type.

        """
        if self.OBS_TYPE == ObservationType.RGB:
            if self.step_counter%self.IMG_CAPTURE_FREQ == 0: 
                self.rgb[0], self.dep[0], self.seg[0] = self._getDroneImages(0,
                                                                             segmentation=False
                                                                             )
                #### Printing observation to PNG frames example ############
                if self.RECORD:
                    self._exportImage(img_type=ImageType.RGB,
                                      img_input=self.rgb[0],
                                      path=self.ONBOARD_IMG_PATH,
                                      frame_num=int(self.step_counter/self.IMG_CAPTURE_FREQ)
                                      )
            return self.rgb[0]
        elif self.OBS_TYPE == ObservationType.KIN: 
            obs = self._clipAndNormalizeState(self._getDroneStateVector(0))
            ############################################################
            #### OBS OF SIZE 20 (WITH QUATERNION AND RPMS)
            # return obs
            ############################################################
            #### OBS SPACE OF SIZE 12
            # return np.hstack([obs[0:3], obs[7:10], obs[10:13], obs[13:16]]).reshape(12,)
            ############################################################
            #### OBS SPACE OF SIZE 13 (With D2Destin)
            return obs.reshape(13,)
            ############################################################
        else:
            print("[ERROR] in BaseSingleAgentAviary._computeObs()")
            
   ################################################################################
            
    def _normalizeDetection(self):
        """Normalizes a drone's state to the [-1,1] range.

        Must be implemented in a subclass.

        Parameters
        ----------
        state : ndarray
            Array containing the non-normalized state of a single drone.

        """
        raise NotImplementedError
    
    ################################################################################

    def _clipAndNormalizeState(self,
                               state
                               ):
        """Normalizes a drone's state to the [-1,1] range.

        Must be implemented in a subclass.

        Parameters
        ----------
        state : ndarray
            Array containing the non-normalized state of a single drone.

        """
        raise NotImplementedError
