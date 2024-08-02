import math
import numpy as np
# import pybullet as p

from gym_pybullet_drones.routing.BaseRouting import BaseRouting, SpeedCommandFlag, SpeedStatus
from gym_pybullet_drones.envs.BaseAviary import DroneModel, BaseAviary
# from gym_pybullet_drones.envs.RoutingAviary import RoutingAviary
# from gym_pybullet_drones.utils.utils import nnlsRPM

class IFDSRoute(BaseRouting):
    """IFDS path-planning class."""

    ################################################################################

    def __init__(self,
                 drone_model: DroneModel,
                 g: float=9.8
                 ):
        """Common routing classes __init__ method.

        Parameters
        ----------
        drone_model : DroneModel
            The type of drone to control (detailed in an .urdf file in folder `assets`).
        g : float, optional
            The gravitational acceleration in m/s^2.

        """
        super().__init__(drone_model=drone_model, g=g)

        # self.RHO0_IFDS = 6.5
        self.RHO0_IFDS = 8.5
        self.SIGMA0_IFDS = 0.1  # 1
        self.SF_IFDS = 0
        self.TARGET_THRESH = 0.1
        # self.SIM_MODE = 2
        self.DT = 0.5  # 0.1
        self.TSIM = 60
        self.RTSIM = 200
        
        self.reset()

    ################################################################################

    def reset(self):
        """Resets the routing classes.

        The previous step's and integral errors for both position and attitude are set to zero.

        """
        super().reset()
        #### Initialized PID control variables #####################
        self.last_pos_e = np.zeros(3)
        self.integral_pos_e = np.zeros(3)
        self.last_rpy_e = np.zeros(3)
        self.integral_rpy_e = np.zeros(3)
    
    ################################################################################
    
    def computeRoute(self,
                     route_timestep,
                     cur_pos,
                     cur_quat,
                     cur_vel,
                     cur_ang_vel,
                     home_pos,
                     target_pos,
                     speed_limit,
                     obstacle_data=None
                     ):
        """Computes the IFDS path for a single drone.

        This methods sequentially calls `_IFDS()`.
        Parameters `cur_ang_vel` is unused.

        Parameters
        ----------
        route_timestep : float
            The time step at which the route is computed.
        cur_pos : ndarray
            (3,1)-shaped array of floats containing the current position.
        cur_vel : ndarray
            (3,1)-shaped array of floats containing the current velocity.
        target_pos : ndarray
            (3,1)-shaped array of floats containing the desired position.

        Returns
        -------

        """
        self._updateCurPos(cur_pos)
        self._updateCurVel(cur_vel)
        self.route_counter += 1
        
        # Pre-allocation waypoints and paths
        start_pos = cur_pos
        wp = start_pos.reshape(3,1)
 
        # vel = np.linalg.norm(cur_vel)
        vel = 0.3
 
        # Generate a route from the IFDS path-planning algorithm
        foundPath, path = self._IFDS(wp, route_timestep, cur_pos, vel, start_pos, target_pos, obstacle_data)
        self._guidanceFromRoute(path, route_timestep, speed_limit)
        
        self._plotRoute(path)
        self._batchRayCast()
        return foundPath, path
    ################################################################################
    
    def _guidanceFromRoute(self, path, route_timestep, speed_limit):
        """
        Process the route and give the waypoint to be followed by the UAV
        Args:
             (TYPE): DESCRIPTION.

        Returns:
            None.

        """
        # Process Command
        self._processCommand()
        self._updateTargetPosAndVel(path, route_timestep, speed_limit)
     
            
    ################################################################################
    
    def _updateTargetPosAndVel(self, path, route_timestep, speed_limit):
        path_vect_unit = self._waypointSkipping(path, route_timestep, speed_limit)
        self._updateTargetVel(route_timestep, speed_limit, path_vect_unit)
        
        self._resetRouteCommand()
        self._resetSpeedCommand()
        
    def _updateTargetVel(self, route_timestep, speed_limit, path_vect_unit):
        # -------------------- Target Velocity Vector ----------------------
        curSpeed = np.linalg.norm(self.CUR_VEL)
        
        if np.linalg.norm(self.CUR_POS - self.DESTINATION) <= 1:
            # print("Reaching destination -> Stopping . . .")
            self.TARGET_VEL = np.zeros(3)
            # acceleration = 0
        else:
            # if np.linalg.norm(self.CUR_POS - self.DESTINATION) < 4: # [m]
            #     print("Approaching destination -> Decelerating . . .")
            #     self._setCommand(SpeedCommandFlag, "accelerate", -0.02)  # [m/s^2]
            
            if self.COMMANDS[1]._name == SpeedCommandFlag.ACCEL.value:
                acceleration = self.COMMANDS[1]._value
                # self.TARGET_VEL = (curSpeed + acceleration*route_timestep*100) * path_vect_unit
                self.TARGET_VEL = (curSpeed + acceleration*self.route_counter*0.01) * path_vect_unit
            
        self._processTargetVel(speed_limit)
            
    def _processTargetVel(self, speed_limit):
        # Process target_vel
        # if np.linalg.norm(self.TARGET_VEL) <= 0.03:
        #     print("stopping")
        #     # self.TARGET_VEL = np.zeros(3)  # No need since already update in _processSpeedCommand()
        #     self._setCommand(SpeedCommandFlag, "hover")
            
        if np.linalg.norm(self.TARGET_VEL) > speed_limit:
            # targetVel[j] = np.clip(targetVel[j], 0, env.SPEED_LIMIT)
            # self.TARGET_VEL = np.zeros(3)
            self.TARGET_VEL = np.clip(self.TARGET_VEL, 0, speed_limit)
            print("Max speed reached: " + str(np.linalg.norm(self.TARGET_VEL)))
            # print(": max speed limit reached -> Decelerating")
            # self._setCommand(SpeedCommandFlag, "accelerate", -0.06)
            
    def _waypointSkipping(self, path, route_timestep, speed_limit):
        
        # -------------------- Target Position -----------------------------
        # -----------------Waypoint Skipping Logic--------------------------
        path_vect_unit = np.zeros(3)
        k = 1   # Initial waypoint number to follow
        while True:
            
            if k >= path.shape[1]-1:
                # print("waypoint #" + str(k) + " exceeded path length")
                break
            Wi = path[:,k]
            Wf = path[:,k+1]
            
            path_vect = Wf - Wi
            path_vect_unit = path_vect / np.linalg.norm(path_vect)
            a = path_vect[0]
            b = path_vect[1]
            c = path_vect[2]
            
            # wp_closeness_threshold = speed_limit/100  # [m]
            wp_closeness_threshold = speed_limit/100  # [m]
 
            # k[j] += 1
            
            # Check if the waypoing is ahead of current position
            if a*(self.CUR_POS[0] - Wf[0]) + b*(self.CUR_POS[1] - Wf[1]) + c*(self.CUR_POS[2]- Wf[2]) < 0:
                self.TARGET_POS = Wf   # Wi or Wf??
                # print("Agent " + str(j+1) + ": targeting WP #" + str(k[j]))
                if np.linalg.norm(self.CUR_POS.reshape(3,1) - Wf.reshape(3,1)) <= wp_closeness_threshold: 
                    k += 1
                else:
                    break
            else:
                k += 1
        
        return path_vect_unit

    ################################################################################

    def _IFDS(self,
              wp,
              route_timestep,
              cur_pos,
              v,
              home_pos,
              target_pos,
              obstacle_data=None
              ):
        """
        Generate 3D path using Interfered Fluid Dynamical System (IFDS) algorithm

        Args:
            param (dict): Contain all essential parameters for UAV and simulation.
            wp (Array 3x1): Initial location to generate a path from.

        Returns:
            Path (Array 3x_): A 3D path.

        """
        
        posList = []
        sizeList = []
        obstacles_pos = np.array([])
        obstacles_size = np.array([])
        if bool(obstacle_data):  # Boolean of empty dict return False
            for j in self.DETECTED_OBS_IDS:
                posList.append(obstacle_data[str(j)]["position"])
                sizeList.append(obstacle_data[str(j)]["size"])
            obstacles_pos = np.array(posList).reshape(len(self.DETECTED_OBS_IDS), 3)
            obstacles_size = np.array(sizeList).reshape(len(self.DETECTED_OBS_IDS), 3)
        
        def _CalcUBar(Obj):
            """
            Calculate modified velocity UBar for the IFDS algorithm

            Args:
                param (dict): Contain all essential parameters for UAV and simulation.
                loc (Array 3x1): Current location of the UAV (x, y, z).
                Obj (list): A list of dictionaries containing information of the obstacles.

            Returns:
                UBar (Array 3x1): The modified velocity vector.
            """
            # Load parameters
            rho0   = self.RHO0_IFDS
            sigma0 = self.SIGMA0_IFDS
            # (X, Y, Z)=  cur_pos
            loc = wp[:, -1]
            (X, Y, Z) = loc
            (xd, yd, zd) = target_pos
            
            dist = np.linalg.norm(loc.reshape(3,1) - target_pos.reshape(3,1))

            u = -np.array([[v*(X - xd)/dist],
                           [v*(Y - yd)/dist],
                           [v*(Z - zd)/dist]])
            # Pre-allocation
            Mm = np.zeros((3,3))
            sum_w = 0;
            
            if len(Obj) != 0:
                # print("DETECTED " + str(len(Obj)) + " OBSTACLES!")
                for j in range(len(Obj)):
                    # Reading Gamma for each obstacle
                    Gamma = Obj[j]['Gamma']
                    # Unit normal and tangential vector
                    n = Obj[j]['n']
                    t = Obj[j]['t']
                    dist_obs = np.linalg.norm(loc.reshape(3,1) - Obj[j]['origin'].reshape(3,1))
                    ntu = np.dot(np.transpose(n), u)
                    if ntu < 0 or self.SF_IFDS == 1:
                        rho   = rho0   * math.exp(1 - 1/(dist_obs * dist))
                        sigma = sigma0 * math.exp(1 - 1/(dist_obs * dist))
                        n_t = np.transpose(n)
                        M = np.identity(3) - np.dot(n,n_t)/(abs(Gamma)**(1/rho)*np.dot(n_t,n)) + \
                            np.dot(t,n_t)/(abs(Gamma)**(1/sigma)*np.linalg.norm(t)*np.linalg.norm(n))
                        
                    elif ntu >= 0 and self.SF_IFDS == 0:
                        M = np.identity(3)
                    else:
                        raise ValueError("[Error] in _CalcUBar in IFDSRoute")
                        # print("error in _CalcUBar")
                        # UBar = u
                        
                    # Calculate Weight
                    w = 1
                    if len(Obj) > 1:
                        w = [w*(Obj[i]['Gamma'] - 1)/((Obj[j]['Gamma'] - 1) + (Obj[i]['Gamma']-1)) for i in range(len(Obj)) if i!=j][0]
                    # Saving into each obstacles
                    Obj[j]["w"] = w
                    Obj[j]["M"] = M
                    sum_w = sum_w + w
                for j in range(len(Obj)):
                    w_tilde = Obj[j]["w"]/sum_w
                    Mm = Mm + w_tilde*Obj[j]["M"] 
                
                UBar = np.dot(Mm, u)
                return UBar
            else:
                UBar = u
                return UBar   
        ############################################################    

        def _Loop(wp, t):
            flagBreak = 0
            flagReturn = 0
            foundPath = 0
            if t > 1000:
                flagBreak = 1 # break
            loc = wp[:, -1]
            
            # Create scenario with obstacles
            Obstacle = self._CreateScene(loc, obstacles_pos, obstacles_size)
            
            # Check if all obstacles is static
            envIsStatic = all([Obstacle[i]["type"] == "Static" for i in range(len(Obstacle))])
            
            # Manual Condition to use global path
            useGlobalRoute_manual = envIsStatic and route_timestep>1 and self.SIM_MODE==2

                
            
            if np.linalg.norm(loc - target_pos) < self.TARGET_THRESH:
                # print("Path found at step #" + str(t))
                wp = wp[:, :-1]
                # Path[rt] = wp
                foundPath = 1
                flagBreak = 1 # break
            else:
                UBar = _CalcUBar(Obstacle)
                wp = np.append(wp, wp[:, -1].reshape(3, 1)+ UBar * self.DT, axis=1)
                
            return (flagReturn, flagBreak, foundPath, wp)
  
        # Initialization
        Path = np.array([])
        foundPath = 0
        if self.SIM_MODE == 1:
            # Mode 1: Simulate by limiting steps
            for t in range(self.TSIM):
                flagReturn, flagBreak, foundPath, wp = _Loop(wp, t)
                if flagBreak:
                    break
        else:
            # Mode 2: Simulate by reaching distance
            t = 0
            while True:
                flagReturn, flagBreak, foundPath, wp = _Loop(wp, t)
                if flagBreak:
                    break
                t += 1
            
        wp = wp[:, 0:t]
        Path = np.delete(wp, np.s_[t+1:len(wp)], axis=1)
        if foundPath == 2:
            Path = self.GLOBAL_PATH
            # print("Using Global Route")
        return (foundPath, Path)      
        
    ################################################################################

    def _CreateScene(self, cur_pos, obstacles_pos, obstacles_size):
        """
        Create scenarios with obstacles

        Args:
            cur_pos (Array 3x1): Current UAV's position.
            obstacles_pos (Array Nx3): Obstacles' positions.
            obstacles_size (Array Nx3): Obstacles' sizes.

        Returns:
            Obj (list): A list of dictionaries containing obstacles' informations.
        """
        
        # Initilize an empty list
        Obj = []
        
        (X, Y, Z) = cur_pos
        def Shape(isDynamic, shape, x0, y0, z0, D, h=0):
            def CalcGamma():
                Gamma = ((X - x0)/a)**(2*p) + ((Y - y0)/b)**(2*q) + ((Z - z0)/c)**(2*r)
                return Gamma
            def CalcDg():
                dGdx = (2*p*((X - x0)/a)**(2*p - 1))/a
                dGdy = (2*q*((Y - y0)/b)**(2*q - 1))/b
                dGdz = (2*r*((Z - z0)/c)**(2*r - 1))/c
                return (dGdx, dGdy, dGdz)
            
            if shape == "sphere":
                (a, b, c) = (D/2, D/2, D/2)
                (p, q, r) = (1, 1, 1)
            elif shape == "cylinder":
                (a, b, c) = (D/2, D/2, h)
                (p, q, r) = (1, 1, 4)
            elif shape == "cone":
                (a, b, c) = (D/2, D/2, h)
                (p, q, r) = (1, 1, 0.5)
            
            Gamma = CalcGamma()
            (dGdx, dGdy, dGdz) = CalcDg()
            n = np.array([[dGdx], [dGdy], [dGdz]])
            t = np.array([[dGdy],[-dGdx], [0]])
            origin = np.array([x0, y0, z0])
            # Add a new object to the list
            dType = "Dynamic" if isDynamic else "Static"
            Obj.append({'Gamma': Gamma, 'n': n, 't': t, 'origin': origin, 'type': dType})
        
        numObj = obstacles_pos.shape[0]

        for j in range(numObj):
            Shape(0, "sphere", obstacles_pos[j][0], obstacles_pos[j][1], obstacles_pos[j][2], 1*obstacles_size[j][0])
    
        return Obj
 
