import os
import numpy as np
import pybullet as p
from gym import spaces
from collections import deque
from ray.rllib.env.multi_agent_env import MultiAgentEnv

from gym_pybullet_drones.envs.RoutingAviary import RoutingAviary
from gym_pybullet_drones.routing.BaseRouting import RouteCommandFlag, SpeedCommandFlag
from gym_pybullet_drones.routing.IFDSRoute import IFDSRoute
from gym_pybullet_drones.routing.RouteMission import RouteMission

from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ActionType, ObservationType
# from gym_pybullet_drones.utils.utils import nnlsRPM
# from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
# from gym_pybullet_drones.control.SimplePIDControl import SimplePIDControl
from gym_pybullet_drones.control.PIDVelocityControl import PIDVelocityControl
from gym_pybullet_drones.guidance.CCA3DGuidance import CCA3DGuidance


class ExtendedMultiagentAviary_discrete(RoutingAviary, MultiAgentEnv):
    """Base multi-agent environment class for reinforcement learning."""
    
    ################################################################################

    def __init__(self,
                 drone_model: DroneModel=DroneModel.HB,
                 num_drones: int=2,
                 neighbourhood_radius: float=np.inf,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics=Physics.PYB,
                 freq: int=240,
                 aggregate_phy_steps: int=1,
                 gui=False,
                 record=False, 
                 obs: ObservationType=ObservationType.KIN,
                 act: ActionType=ActionType.AUTOROUTING
                 ):
        """Initialization of a generic multi-agent RL environment.

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
        #===============================
        homePos =  np.array([0,0,0.5]) 
        destin  =  np.array([0.2, 10, 1])
        # self.HOME_POS = homePos
        # self.DESTIN = destin
        self.MISSION = RouteMission()
        self.OBS_CHOICE = 3  # 1: 12 kinematic states + d2d + raysinfo | 2: 6 kinamitic states + d2d + raysinfo | 3: 6 kinematic states + d2d + 8 sectorsinfo
        
        self.DONE = [False for _ in range(num_drones)]
        # self.MISSION.generateMission(numDrones=num_drones, scenario=1)
        self.MISSION.generateRandomMission(maxNumDrone=num_drones, minNumDrone=num_drones)
        #===============================

        # if num_drones < 2:
        #     print("[ERROR] in ExtendedMultiagentAviary_discrete.__init__(), num_drones should be >= 2")
        #     exit()
        if act == ActionType.TUN:
            print("[ERROR] in BaseMultiagentAviary.__init__(), ActionType.TUN can only used with BaseSingleAgentAviary")
            exit()
        vision_attributes = True if obs == ObservationType.RGB else False
        dynamics_attributes = True if act in [ActionType.DYN, ActionType.ONE_D_DYN] else False
        self.OBS_TYPE = obs
        self.ACT_TYPE = act
        self.EPISODE_LEN_SEC = 20
        #### Create integrated controllers #########################
        # print(f">>>>>>>>> Komsun : action is {act}")
        if act in [ActionType.PID, ActionType.VEL, ActionType.ONE_D_PID, ActionType.AUTOROUTING]:
            os.environ['KMP_DUPLICATE_LIB_OK']='True'
            if drone_model in [DroneModel.CF2X, DroneModel.CF2P, DroneModel.HB]:
                # self.ctrl = [DSLPIDControl(drone_model=DroneModel.CF2X) for i in range(num_drones)]
                self.ctrl = [PIDVelocityControl(drone_model=DroneModel.HB) for i in range(num_drones)]
                self.routing = [IFDSRoute(drone_model=DroneModel.CF2X, drone_id=i) for i in range(num_drones)]
                self.guidance = CCA3DGuidance(drone_model=DroneModel.CF2X)

                self.INIT_XYZS = self.MISSION.INIT_XYZS
                self.INIT_RPYS = self.MISSION.INIT_RPYS
                # self.INIT_XYZS = np.zeros((len(self.routing), 3))
                for j in range(num_drones):
                    self.INIT_XYZS[j,:] = self.MISSION.INIT_XYZS[j,:]
                    self.routing[j].HOME_POS = self.MISSION.INIT_XYZS[j,:]
                    self.routing[j].DESTINATION = self.MISSION.DESTINS[j,:]
                    self.routing[j].CUR_POS = self.MISSION.INIT_XYZS[j, :]
                    self.routing[j].CUR_RPY = self.MISSION.INIT_RPYS[j,:]
                    
                    # self.routing[j].HOME_POS = homePos
                    # self.routing[j].DESTINATION = np.array([destin[0], destin[1]+0.1*j, destin[2]])
            elif drone_model == DroneModel.HB:
                # self.ctrl = [SimplePIDControl(drone_model=DroneModel.HB) for i in range(num_drones)]
                self.ctrl = [PIDVelocityControl(drone_model=DroneModel.HB) for i in range(num_drones)]
            else:
                print("[ERROR] in BaseMultiagentAviary.__init()__, no controller is available for the specified drone_model")
        
        self.SENSOR_BUFFER_SIZE = 1   # 5: five informations from raycast (obj_id, hit_fraction, (hit_xyz))
        self.sensor_buffer = deque(maxlen=self.SENSOR_BUFFER_SIZE)
        super().__init__(drone_model=drone_model,
                         num_drones=num_drones,
                         neighbourhood_radius=neighbourhood_radius,
                         initial_xyzs=self.MISSION.INIT_XYZS,
                         initial_rpys=self.INIT_RPYS,
                         physics=physics,
                         freq=freq,
                         aggregate_phy_steps=aggregate_phy_steps,
                         gui=gui,
                         record=record, 
                         obstacles=True, # Add obstacles for RGB observations and/or FlyThruGate
                         user_debug_gui=False, # Remove of RPM sliders from all single agent learning aviaries
                         )
        #### Set a limit on the maximum target speed ###############
        if act == ActionType.VEL:
            self.SPEED_LIMIT = 0.03 * self.MAX_SPEED_KMH * (1000/3600)

    # ################################################################################

    # def _addObstacles(self):
    #     """Add obstacles to the environment.

    #     Only if the observation is of type RGB, 4 landmarks are added.
    #     Overrides BaseAviary's method.

    #     """
    #     if self.OBS_TYPE == ObservationType.RGB:
    #         p.loadURDF("block.urdf",
    #                    [1, 0, .1],
    #                    p.getQuaternionFromEuler([0, 0, 0]),
    #                    physicsClientId=self.CLIENT
    #                    )
    #         p.loadURDF("cube_small.urdf",
    #                    [0, 1, .1],
    #                    p.getQuaternionFromEuler([0, 0, 0]),
    #                    physicsClientId=self.CLIENT
    #                    )
    #         p.loadURDF("duck_vhacd.urdf",
    #                    [-1, 0, .1],
    #                    p.getQuaternionFromEuler([0, 0, 0]),
    #                    physicsClientId=self.CLIENT
    #                    )
    #         p.loadURDF("teddy_vhacd.urdf",
    #                    [0, -1, .1],
    #                    p.getQuaternionFromEuler([0, 0, 0]),
    #                    physicsClientId=self.CLIENT
    #                    )
    #     else:
    #         pass

    # ################################################################################

    def _actionSpace(self):
        """Returns the action space of the environment.

        Returns
        -------
        dict[int, ndarray]
            A Dict() of Box() of size 1, 3, or 3, depending on the action type,
            indexed by drone Id in integer format.

        """
        
        action_size = 1
        for _ in range(self.ACTION_BUFFER_SIZE):
            self.action_buffer.append(np.zeros((1,action_size)))
        num_actions_per_agent = 3
        actions_array = [num_actions_per_agent for _ in range(self.NUM_DRONES)]
        # return spaces.MultiDiscrete(actions_array)
        return spaces.Dict({i: spaces.Discrete(3) for i in range(self.NUM_DRONES)}) 
        # return spaces.Dict({i: spaces.Box(low=0*np.ones(action_size),
        #                                 high=2*np.ones(action_size),
        #                                 dtype=np.int16
        #                                 ) for i in range(self.NUM_DRONES)})
        # 11 discrete actions, details in _preprocessAction()
    

    ################################################################################

    def _preprocessAction(self,
                          action
                          ):
        """Pre-processes the action passed to `.step()` into motors' RPMs.

        Parameter `action` is processed differenly for each of the different
        action types: the input to n-th drone, `action[n]` can be of length
        1, 3, or 4, and represent RPMs, desired thrust and torques, or the next
        target position to reach using PID control.

        Parameter `action` is processed differenly for each of the different
        action types: `action` can be of length 1, 3, or 4 and represent 
        RPMs, desired thrust and torques, the next target position to reach 
        using PID control, a desired velocity vector, etc.

        Parameters
        ----------
        action : dict[str, ndarray]
            The input action for each drone, to be translated into RPMs.

        Returns
        -------
        ndarray
            (NUM_DRONES, 4)-shaped array of ints containing to clipped RPMs
            commanded to the 4 motors of each drone.

        """
        rpm = np.zeros((self.NUM_DRONES,4))
        # print(f"++++ Action is {action}")
        for k, val in action.items():
            val = int(val)
            # Process action based on ACT_TYPE
            if self.ACT_TYPE == ActionType.AUTOROUTING:
                state = self._getDroneStateVector(k)
            
                # Initially set to accelerate
                # self.routing._setCommand(SpeedCommandFlag, "accelerate", 0.02)
                # action = 1
                
                #------- Compute route (waypoint) to follow ----------------
                foundPath, path = self.routing[k].computeRouteFromState(route_timestep=self.routing[k].route_counter, 
                                                                    state = state, 
                                                                    home_pos = self.routing[k].HOME_POS,   # self.HOME_POS
                                                                    target_pos = self.routing[k].DESTINATION,   # self.DESTIN
                                                                    speed_limit = self.SPEED_LIMIT,
                                                                    obstacle_data = self.OBSTACLE_DATA,
                                                                    drone_ids = k
                                                                    )
                
                 # ==== PASSIVE BEHAVIOUR ======
                if self.routing[k].route_counter == 1:
                    if foundPath>0:
                        # print("Calculating Global Route . . .")
                        self.routing[k].setGlobalRoute(path)
                    else:
                        fromPos = self.routing[k].HOME_POS
                        toPos = self.routing[k].DESTINATION
                        n_wp = 100
                        gpath = self.routing[k]._generateWaypoints(fromPos, toPos, n_wp)
                        self.routing[k].setGlobalRoute(np.array(gpath).reshape((3,n_wp)))
                        # raise ValueError("[Error] Global route was not found. Mission aborted.")    
                # if self.routing[k].route_counter == 1:
                #     fromPos = self.routing[k].HOME_POS
                #     toPos = self.routing[k].DESTINATION
                #     n_wp = 100
                #     # print("Calculating Global path again")
                #     gpath = self.routing[k]._generateWaypoints(fromPos, toPos, n_wp)
                #     self.routing[k].setGlobalRoute(np.array(gpath).reshape((3,n_wp)))
                #     # self.routing[k]._updateTargetPosAndVel(gpath, self.routing[k].route_counter, self.SPEED_LIMIT)
                #     # raise ValueError("[Error] Global route was not found. Mission aborted.")    

                # ==== PASSIVE BEHAVIOUR ======
                # ======= 3 Actions ==================================================
                # self.routing[k]._setCommand(RouteCommandFlag, "follow_global")
                # if action ==0:
                #     self.routing[k]._setCommand(SpeedCommandFlag, "accelerate", 0.0)
                # elif action ==1:
                #     self.routing[k]._setCommand(SpeedCommandFlag, "accelerate", -4)
                # elif action ==2:
                #     self.routing[k]._setCommand(SpeedCommandFlag, "hover")

                # ======= 11 Actions ==================================================
                # print(f"xxxxxxxxxxxxx  Action is {action}")
                
                if val ==0:
                    # print(f"Agent {k}: action 0 >>>>> Accelerating . . .")
                    self.routing[k]._setCommand(SpeedCommandFlag, "accelerate", 1)  # 0.05
                elif val ==1:
                    # print(f"Agent {k}: action 1 <<<<< Decelerating . . .")
                    self.routing[k]._setCommand(SpeedCommandFlag, "accelerate", -1)   # -4
                elif val ==2:
                    # print(f"Agent {k}: action 2 ===== Hovering . . .")
                    # self.routing[k]._setCommand(SpeedCommandFlag, "hover")
                    self.routing[k]._setCommand(SpeedCommandFlag, "constant")
                    self.routing[k]._setCommand(RouteCommandFlag, "follow_global", 1)
                elif val ==3:
                    # print("This is action 3 >>>> following global route . . .")
                    self.routing[k]._setCommand(SpeedCommandFlag, "constant")
                    self.routing[k]._setCommand(RouteCommandFlag, "follow_global", 1)
                    # self.routing[k]._setCommand(SpeedCommandFlag, "accelerate", 0)  
                elif val==4:
                    self.routing[k]._setCommand(SpeedCommandFlag, "constant")
                    self.routing[k]._setCommand(RouteCommandFlag, "follow_local", 1)
                elif val==5:
                    self.routing[k]._setCommand(SpeedCommandFlag, "constant")
                    self.routing[k]._setCommand(RouteCommandFlag, "follow_local", 2)
                elif val==6:
                    self.routing[k]._setCommand(SpeedCommandFlag, "constant")
                    self.routing[k]._setCommand(RouteCommandFlag, "follow_local", 3)
                elif val==7:
                    self.routing[k]._setCommand(SpeedCommandFlag, "constant")
                    self.routing[k]._setCommand(RouteCommandFlag, "follow_local", 4)
                elif action.get(k)==8:
                    self.routing[k]._setCommand(SpeedCommandFlag, "constant")
                    self.routing[k]._setCommand(RouteCommandFlag, "follow_local", 5)
                elif val==9:
                    self.routing[k]._setCommand(SpeedCommandFlag, "constant")
                    self.routing[k]._setCommand(RouteCommandFlag, "follow_local", 6)
                elif val==10:
                    self.routing[k]._setCommand(SpeedCommandFlag, "constant")
                    self.routing[k]._setCommand(RouteCommandFlag, "follow_local", 7)
                elif val==11:
                    self.routing[k]._setCommand(SpeedCommandFlag, "constant")
                    self.routing[k]._setCommand(RouteCommandFlag, "follow_local", 8)

                #     if self.routing.STAT:
                #         print("Alert: no route found!--> following global route")
                #         self.routing._setCommand(RouteCommandFlag, "follow_global")
                else:
                    self.routing[k]._setCommand(SpeedCommandFlag, "hover")
                    # print(f"[ERROR] in ExtendedMultiagentAviary._preprocessAction() >>>>> action is {action}, action.get is {action.get(k)}")
                    raise ValueError(f"Invalid action: {action}: hovering")
                
                state2follow = self.guidance.followPath(path = path, 
                                                    state = state, 
                                                    target_vel = self.routing[k].TARGET_VEL,
                                                    speed_limit = self.SPEED_LIMIT)
                
                #### Compute control for the current way point #############
                # rpm_k, _, _ = self.ctrl[k].computeControlFromState(control_timestep=self.CTRL_TIMESTEP,
                #                                                 state=state,
                #                                                 target_pos = self.routing[k].TARGET_POS, 
                #                                                 target_rpy = self.MISSION.INIT_RPYS[k,:],
                #                                                 target_vel = self.routing[k].TARGET_VEL
                #                                                 )
                # rpm_k, _, _ = self.ctrl[k].computeControl(control_timestep=self.CTRL_TIMESTEP, 
                #                                  cur_pos=state[0:3],
                #                                  cur_quat=state[3:7],
                #                                  cur_vel=state[10:13],
                #                                  cur_ang_vel=state[13:16],
                #                                  target_pos=self.routing[k].TARGET_POS,
                #                                  target_rpy=state2follow[3:6],
                #                                  target_vel=self.routing[k].TARGET_VEL
                #                                  )

                # ------------ velocity control ------------
                rpm_k, _, _ = self.ctrl[k].computeControl(control_timestep=self.CTRL_TIMESTEP, 
                                                 cur_pos=state[0:3],
                                                 cur_quat=state[3:7],
                                                 cur_vel=state[10:13],
                                                 cur_ang_vel=state[13:16],
                                                 target_rpy=state2follow[3:6],
                                                 target_vel=self.routing[k].TARGET_VEL
                                                 )
                rpm[k,:] = rpm_k
            # ================================   
            else:
                print("[ERROR] in BaseMultiagentAviary._preprocessAction()")
                exit()
        return rpm

    ################################################################################

    def _observationSpace(self):
        """Returns the observation space of the environment.

        Returns
        -------
        dict[int, ndarray]
            A Dict with NUM_DRONES entries indexed by Id in integer format,
            each a Box() os shape (H,W,4) or (12,) depending on the observation type.

        """
        if self.OBS_TYPE == ObservationType.RGB:
            return spaces.Dict({i: spaces.Box(low=0,
                                              high=255,
                                              shape=(self.IMG_RES[1], self.IMG_RES[0], 4), dtype=np.uint8
                                              ) for i in range(self.NUM_DRONES)})
        elif self.OBS_TYPE == ObservationType.KIN:
            ############################################################
            #### OBS OF SIZE 20 (WITH QUATERNION AND RPMS)
            #### Observation vector ### X        Y        Z       Q1   Q2   Q3   Q4   R       P       Y       VX       VY       VZ       WX       WY       WZ       P0            P1            P2            P3
            # obs_lower_bound = np.array([-1,      -1,      0,      -1,  -1,  -1,  -1,  -1,     -1,     -1,     -1,      -1,      -1,      -1,      -1,      -1,      -1,           -1,           -1,           -1])
            # obs_upper_bound = np.array([1,       1,       1,      1,   1,   1,   1,   1,      1,      1,      1,       1,       1,       1,       1,       1,       1,            1,            1,            1])          
            # return spaces.Box( low=obs_lower_bound, high=obs_upper_bound, dtype=np.float32 )
            ############################################################
        
            #### OBS SPACE OF SIZE 12
            #### Observation vector 12 ### X        Y        Z       R       P       Y       VX       VY       VZ       WX       WY       WZ
            #### Observation vector 12 ### X        Y        Z       VX       VY       VZ      
            obs_choice = self.OBS_CHOICE  # 1: 12+1+5*n_ray |  2: 6+1+5*n_ray
            if obs_choice == 1:
                lo = -1
                hi = 1
                obs_lower_bound = np.array([[lo,lo,0, lo,lo,lo,lo,lo,lo,lo,lo,lo] for i in range(1)])
                obs_upper_bound = np.array([[hi,hi,hi,hi,hi,hi,hi,hi,hi,hi,hi,hi] for i in range(1)])
                
                obs_lower_bound =  obs_lower_bound.reshape(1, obs_lower_bound.shape[1]*obs_lower_bound.shape[0])
                obs_upper_bound =  obs_upper_bound.reshape(1, obs_upper_bound.shape[1]*obs_upper_bound.shape[0])
            
                # Rayinfo: [obj_id,  hit_fraction,  hitPos_x,  hitPos_y,  hitPos_z] per ray
                ray_lo = np.tile([-1, 0, -1, -1, 0], self.routing[0].NUM_RAYS)
                ray_hi = np.tile([1,  1,  1,  1, 1], self.routing[0].NUM_RAYS)
                # ++++++ Add distance-to-destination to observation space ++++++
                obs_lower_bound = np.hstack([obs_lower_bound, np.array([[0] for i in range(1)])])
                obs_upper_bound = np.hstack([obs_upper_bound, np.array([[1] for i in range(1)])])
            
                # +++++ Add ray info to observation space ++++++
                obs_lower_bound = np.hstack([obs_lower_bound, np.array([list(ray_lo) for _ in range(1)])])
                obs_upper_bound = np.hstack([obs_upper_bound, np.array([list(ray_hi) for _ in range(1)])])
                # added 20241015
                obs_lower_bound =  obs_lower_bound.reshape(obs_lower_bound.shape[1]*obs_lower_bound.shape[0],)
                obs_upper_bound =  obs_upper_bound.reshape(obs_upper_bound.shape[1]*obs_upper_bound.shape[0],)

            elif obs_choice == 2:
                lo = -1
                hi = 1
                #### Observation vector 6 ### X        Y        Z       VX       VY       VZ   
                obs_lower_bound = np.array([[lo,lo,0,lo,lo,lo] for i in range(1)])
                obs_upper_bound = np.array([[hi,hi,hi,hi,hi,hi] for i in range(1)])
                
                obs_lower_bound =  obs_lower_bound.reshape(1, obs_lower_bound.shape[1]*obs_lower_bound.shape[0])
                obs_upper_bound =  obs_upper_bound.reshape(1, obs_upper_bound.shape[1]*obs_upper_bound.shape[0])
            
                # Rayinfo: [hitPos_x,  hitPos_y,  hitPos_z] per ray
                ray_lo = np.tile([-1, -1, 0], self.routing[0].NUM_RAYS)
                ray_hi = np.tile([1,  1, 1], self.routing[0].NUM_RAYS)
                # ++++++ Add distance-to-destination to observation space ++++++
                obs_lower_bound = np.hstack([obs_lower_bound, np.array([[0] for i in range(1)])])
                obs_upper_bound = np.hstack([obs_upper_bound, np.array([[1] for i in range(1)])])
            
                # +++++ Add ray info to observation space ++++++
                obs_lower_bound = np.hstack([obs_lower_bound, np.array([list(ray_lo) for _ in range(1)])])
                obs_upper_bound = np.hstack([obs_upper_bound, np.array([list(ray_hi) for _ in range(1)])])
                # added 20241015
                obs_lower_bound =  obs_lower_bound.reshape(obs_lower_bound.shape[1]*obs_lower_bound.shape[0],)
                obs_upper_bound =  obs_upper_bound.reshape(obs_upper_bound.shape[1]*obs_upper_bound.shape[0],)

            elif obs_choice == 3: # 8 sectors summary stats
                lo = -1
                hi = 1
                #### Observation vector 6 ### X        Y        Z       VX       VY       VZ   
                obs_lower_bound = np.array([[lo,lo,0,lo,lo,lo] for i in range(1)])
                obs_upper_bound = np.array([[hi,hi,hi,hi,hi,hi] for i in range(1)])
                
                obs_lower_bound =  obs_lower_bound.reshape(1, obs_lower_bound.shape[1]*obs_lower_bound.shape[0])
                obs_upper_bound =  obs_upper_bound.reshape(1, obs_upper_bound.shape[1]*obs_upper_bound.shape[0])
                # ++++++ Add distance-to-destination to observation space ++++++
                obs_lower_bound = np.hstack([obs_lower_bound, np.array([[0] for _ in range(1)])])
                obs_upper_bound = np.hstack([obs_upper_bound, np.array([[1] for _ in range(1)])])
                # Sector info
                # Sector Info: [r_min,  r_mean, dhit] per sector 
                num_sector = 8
                sector_lo = np.tile([0, 0, 0], num_sector)
                sector_hi = np.tile([1, 1, 1], num_sector)
                obs_lower_bound = np.hstack([obs_lower_bound, np.array([list(sector_lo) for _ in range(1)])])
                obs_upper_bound = np.hstack([obs_upper_bound, np.array([list(sector_hi) for _ in range(1)])])

                obs_lower_bound =  obs_lower_bound.reshape(obs_lower_bound.shape[1]*obs_lower_bound.shape[0],)
                obs_upper_bound =  obs_upper_bound.reshape(obs_upper_bound.shape[1]*obs_upper_bound.shape[0],)



            return spaces.Dict({i: spaces.Box(low=obs_lower_bound,
                                              high=obs_upper_bound,
                                              dtype=np.float32
                                              ) for i in range(self.NUM_DRONES)})
        else:
            print("[ERROR] in BaseMultiagentAviary._observationSpace()")
    
    ################################################################################

    def _computeObs(self):
        """Returns the current observation of the environment.

        Returns
        -------
        dict[int, ndarray]
            A Dict with NUM_DRONES entries indexed by Id in integer format,
            each a Box() os shape (H,W,4) or (12,) depending on the observation type.

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
            return {i: self.rgb[i] for i in range(self.NUM_DRONES)}
        elif self.OBS_TYPE == ObservationType.KIN: 
            ############################################################
            #### OBS OF SIZE 20 (WITH QUATERNION AND RPMS)
            # return {   i   : self._clipAndNormalizeState(self._getDroneStateVector(i)) for i in range(self.NUM_DRONES) }
            ############################################################
            
            # obs_1013 = np.zeros((self.NUM_DRONES,1013))
            size_obs = self.observation_space[0].shape[0]
            obs_array = np.zeros((self.NUM_DRONES,size_obs))
            rayinfo_1000 = np.zeros((self.NUM_DRONES, self.routing[0].NUM_RAYS*3))
            sectorinfo_array = np.zeros((self.NUM_DRONES, 8*3))  # 8 SECTORS 3 FEATURES  


            if self.OBS_CHOICE == 1 or self.OBS_CHOICE == 2:

                for i in range(self.NUM_DRONES):
                    obs = self._clipAndNormalizeState(self._getDroneStateVector(i))
                    self.routing[i]._batchRayCast(self.routing[i].DRONE_ID)

                    d2destin = self.routing[i].getDistanceToDestin()
                    d2destin_normalized = self._clipAndNormalizeD2Destin(d2destin, i)

                    ray_matrix = self.routing[i].RAYS_INFO  # size of (numrays, 3 or 5)
                    ray_matrix_normalized = np.apply_along_axis(self._clipAndNormalizeRay, 1, ray_matrix)

                    # rayinfo_1000[i,:] = np.array([list(self.routing[i].RAYS_INFO.reshape(5*self.routing[i].NUM_RAYS,))])  # extract 5 info from ray
                    rayinfo_1000[i,:] = np.array([list(ray_matrix_normalized.reshape(3*self.routing[i].NUM_RAYS,))])   # extract only 3 info from ray

                    if self.OBS_CHOICE == 1:
                        obs_array[i, :] = np.hstack([obs[0:3], obs[7:10], obs[10:13], obs[13:16], # omit 3:7 since they are 
                                                    d2destin_normalized,
                                                    rayinfo_1000[i,:]
                                                    ]).reshape(size_obs,)
                    elif self.OBS_CHOICE == 2:
                        # If we choose observation_space = 6 (i.e. obs_choice == 2)
                        obs_array[i, :] = np.hstack([obs[0:3], obs[10:13], 
                                                    d2destin_normalized,
                                                    rayinfo_1000[i,:]
                                                    ]).reshape(size_obs,)
            elif self.OBS_CHOICE == 3:
                for i in range(self.NUM_DRONES):
                    obs = self._clipAndNormalizeState(self._getDroneStateVector(i))
                    self.routing[i]._batchRayCast(self.routing[i].DRONE_ID)

                    d2destin = self.routing[i].getDistanceToDestin()
                    d2destin_normalized = self._clipAndNormalizeD2Destin(d2destin, i)

                    sectorinfo_array[i,:] = self.routing[i].SECTOR_INFO  # size of (num_sector x num_feature, ) -- default is (24, ) (already normalized)
                    obs_array[i, :] = np.hstack([obs[0:3], obs[10:13], 
                                                 d2destin_normalized,
                                                 sectorinfo_array[i,:]
                                                ]).reshape(size_obs,)

            return {i: obs_array[i, :] for i in range(self.NUM_DRONES)}
            ############################################################
        else:
            print("[ERROR] in BaseMultiagentAviary._computeObs()")

    ################################################################################

    def _clipAndNormalizeState(self,
                               state
                               ):
        """Normalizes a ray's information to the [-1,1] range.

        Must be implemented in a subclass.

        Parameters
        ----------
        state : ndarray
            Array containing the non-normalized information of a single ray.

        """
        raise NotImplementedError
    
    def _clipAndNormalizeD2Destin(self, d2destin, drone_id):
        """Normalize a dinstance to destination to the [0, 1] range"""
        raise NotImplementedError
    
    def _clipAndNormalizeRay(self,
                             rayinfo
                            ):
        """Normalizes a drone's state to the [-1,1] range.

        Must be implemented in a subclass.

        Parameters
        ----------
        state : ndarray
            Array containing the non-normalized state of a single drone.

        """
        raise NotImplementedError
    def _clipAndNormalizeSector(self,
                             sectorinfo
                            ):
        """Normalizes a drone's state to the [-1,1] range.

        Must be implemented in a subclass.

        Parameters
        ----------
        state : ndarray
            Array containing the non-normalized state of a single drone.

        """
        raise NotImplementedError
    
    def _calculateNextStep(self, current_position, destination, step_size=1):
        """
        Calculates intermediate waypoint
        towards drone's destination
        from drone's current position

        Enables drones to reach distant waypoints without
        losing control/crashing, and hover on arrival at destintion

        Parameters
        ----------
        current_position : ndarray
            drone's current position from state vector
        destination : ndarray
            drone's target position 
        step_size: int
            distance next waypoint is from current position, default 1

        Returns
        ----------
        next_pos: int 
            intermediate waypoint for drone

        """
        direction = (
            destination - current_position
        )  # Calculate the direction vector
        distance = np.linalg.norm(
            direction
        )  # Calculate the distance to the destination

        if distance <= step_size:
            # If the remaining distance is less than or equal to the step size,
            # return the destination
            return destination

        normalized_direction = (
            direction / distance
        )  # Normalize the direction vector
        next_step = (
            current_position + normalized_direction * step_size
        )  # Calculate the next step
        return next_step
    
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
        self.MISSION.generateRandomMission(maxNumDrone=self.NUM_DRONES, minNumDrone=self.NUM_DRONES)
        # self.MISSION.generateMission(numDrones=self.NUM_DRONES, scenario=1)
        # TODO : initialize random number generator with seed
        p.resetSimulation(physicsClientId=self.CLIENT)
        # print(f"I'm resetting the ExtendedMultiagentAviary !")

        #### Housekeeping ##########################################
        self._housekeeping()
        self.step_counter = 0
        #### Update and store the drones kinematic information #####
        self._updateAndStoreKinematicInformation()
        self.DONE = [False for _ in range(self.NUM_DRONES)]
        for j in range(self.NUM_DRONES):
            self.routing[j].reset()
            self.routing[j].CUR_POS = self.MISSION.INIT_XYZS[j,:]
            self.routing[j].CUR_RPY = self.MISSION.INIT_RPYS[j,:]
            self.routing[j].HOME_POS = self.MISSION.INIT_XYZS[j,:]
            # self.routing[j].GLOBAL_PATH = np.array([])
            self.routing[j].DESTINATION = self.MISSION.DESTINS[j,:]
            self.CONTACT_FLAGS[j] = 0
            self.INIT_XYZS[j,:] = self.MISSION.INIT_XYZS[j,:]
            self.INIT_RPYS[j,:] = self.MISSION.INIT_RPYS[j,:]
            
        self.OBSTACLE_DATA = {}
        self._getObstaclesData()
        p.performCollisionDetection(physicsClientId=self.CLIENT)
        self._detectCollision()
        #### Start video recording #################################
        self._startVideoRecording()
        #### Return the initial observation ########################
        # initial_obs = self._computeObs()
        # initial_info = self._computeInfo()
        # print(f"step_counter is {self.step_counter}")
        return self._computeObs()
