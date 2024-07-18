import os
import math
from enum import Enum
import numpy as np
import pybullet as p
import xml.etree.ElementTree as etxml

from gym_pybullet_drones.envs.BaseAviary import DroneModel, BaseAviary


class RouteStatus(Enum):
    GLOBAL = "global route"
    LOCAL  = "local route"                 
class SpeedStatus(Enum):
    CONSTANT   = "constant speed"
    ACCELERATE = "accelerating" 
    DECELERATE = "decelerating"
    HOVERING   = "hovering"
class RouteCommandFlag(Enum):
    CHANGE = "change_route"
    FOLLOW_GLOBAL = "follow_global"
    FOLLOW_LOCAL  = "follow_local"
    NONE = "none"
class SpeedCommandFlag(Enum):
    ACCEL         = "accelerate"
    CONST         = "constant"
    HOVER         = "hover"
    NONE          = "none"

class CommandTypeError(KeyError): pass
class CommandValueError(ValueError): pass
    
class Commander:
    classList = [SpeedStatus, SpeedCommandFlag, RouteCommandFlag]
    
    def __init__(self, commandType, command_str, value=None):
        if commandType not in Commander.classList:
            raise CommandTypeError("Invalid command type")

        command = self._parse_command(command_str, commandType)
        
        self._value = value
        self._name = command.value
        self._type = commandType

    def _parse_command(self, command_str, commandType):
        for member in commandType:
            if member.value == command_str:
                return member
        raise CommandValueError(f"Invalid command '{command_str}' for the command type {commandType}")

    
class BaseRouting(object):
    """Base class for routing.

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
        self.GLOBAL_PATH = np.array([])
        """ndarray (3,N) : The global static route of UAV from starting to destination"""
        
        self.SPEED_STAT = "Dummy"
        self.CUR_POS = np.array([0,0,0])
        self.CUR_VEL = np.array([0,0,0])
        self.DESTINATION = np.array([0,0,0])
        self.TARGET_POS   = np.array([])   # Check-> initialize with empty array should work
        self.TARGET_VEL  =  np.array([0,0,0])
        self.HOME = np.array([0,0,0])
        self.STAT = [RouteStatus.GLOBAL, SpeedStatus.CONSTANT]
        # self.COMMAND = [RouteCommandFlag.NONE, SpeedCommandFlag.NONE]
        # self.COMMAND_ROUTE = Commander(RouteCommandFlag, "none")
        # self.COMMAND_SPEED = Commander(SpeedCommandFlag, "constant", 0)
        self.COMMANDS = [Commander(RouteCommandFlag, "none"), Commander(SpeedCommandFlag, "none")]
        self.SIM_MODE = 2
        self._resetAllCommands()
        self.route_counter = 0
        self.DETECTED_OBS_IDS = []
        self.DETECTED_OBS_DATA = {}
        self.reset()

    ################################################################################
        
    def _setCommand(self, commandType, command_str, value=None):
        def _parse_command(command_str, commandType):
            for member in commandType:
                if member.value == command_str:
                    return member
            raise CommandValueError(f"Invalid command '{command_str}' for the command type {commandType}")
            
        classList = [RouteCommandFlag, SpeedCommandFlag]
        
        if commandType not in classList:
            raise CommandTypeError("Invalid command type")
        elif commandType == RouteCommandFlag:
            idx = 0
        elif commandType == SpeedCommandFlag:
            idx = 1
        
        # Update the COMMANDS attribute    
        self.COMMANDS[idx] = Commander(commandType, command_str, value)
        self._processCommand()
        
    ################################################################################
    
    def _processCommand(self):
        """Process the command and reset"""
        
        # First command position: route command
        if self.COMMANDS[0]._name != 'none':
            self._processRouteCommand()
            
        if self.COMMANDS[1]._name != 'none':
            self._processSpeedCommand()
                
    
    def _processRouteCommand(self):
        if self.COMMANDS[0]._name == RouteCommandFlag.CHANGE.value:
            self.switchRoute()
        elif self.COMMANDS[0]._name == RouteCommandFlag.FOLLOW_GLOBAL.value:
            self.STAT[0] = RouteStatus.GLOBAL
            self.SIM_MODE = 2
        elif self.COMMANDS[0]._name == RouteCommandFlag.FOLLOW_LOCAL.value:
            self.STAT[0] = RouteStatus.LOCAL
            self.SIM_MODE = 1
        else:
            print("[Error] in _processRouteCommand()")
            
        # Reset the route command
        # self._resetRouteCommand()
            
    def _processSpeedCommand(self):
        if self.COMMANDS[1]._name == SpeedCommandFlag.ACCEL.value:
            # accelerate
            if self.COMMANDS[1]._value > 0:
                # print("Accelerating . . .")
                self.STAT[1] = SpeedStatus.ACCELERATE
            elif self.COMMANDS[1]._value < 0:
                # print("Decelerating . . .")
                self.STAT[1] = SpeedStatus.DECELERATE
            elif self.COMMANDS[1]._value == 0:
                # print("Constant Speed . . .")
                self.STAT[1] = SpeedStatus.CONSTANT
                self.TARGET_VEL = np.zeros(3)
            
        elif self.COMMANDS[1]._name == SpeedCommandFlag.CONST.value:
            # print("Constant Speed . . .")
            self.TARGET_VEL = np.zeros(3)
            
        elif self.COMMANDS[1]._name == SpeedCommandFlag.HOVER.value:
            # hover
            if self.STAT[1] != SpeedStatus.HOVERING:
                # print("*Activate Hovering Mode!")
                self.TARGET_POS = self.CUR_POS
                self.TARGET_VEL = np.zeros(3)
                self.STAT[1] = SpeedStatus.HOVERING
        else:
            print("[Error] in _processSpeedCommand()")
        
        # Reset the speed command
        # self._resetSpeedCommand() 
                
    def switchRoute(self):
        """Switch current route from global to local, or from local to global"""
        if self.STAT[0].value == RouteStatus.GLOBAL.value:
            print("Switching to Local route")
            # self.STAT[0] = RouteStatus.LOCAL
            # self.SIM_MODE = 1
            
            self.COMMANDS[0]._name = RouteCommandFlag.FOLLOW_LOCAL.value
            self._processRouteCommand()
        
        elif self.STAT[0].value == RouteStatus.LOCAL.value:
            print("Switching to Global route")
            # self.STAT[0] = RouteStatus.GLOBAL
            # self.SIM_MODE =2
            
            self.COMMANDS[0]._name = RouteCommandFlag.FOLLOW_GLOBAL.value
            self._processRouteCommand()
             
        else:
            print("[Error] in switchRoute()")
        
    def _resetAllCommands(self):
        self._resetRouteCommand()
        self._resetSpeedCommand()
        
    def _resetRouteCommand(self):
        self.COMMANDS[0] = Commander(RouteCommandFlag, "none")
    
    def _resetSpeedCommand(self):
        self.COMMANDS[1] = Commander(RouteCommandFlag, "none")

    ################################################################################

    def reset(self):
        """Reset the routing classes.

        A general use counter is set to zero.

        """
        # self.route_counter = 0

    ################################################################################

    def setGlobalRoute(self, route):
        """Store global route
        Parmaters
        ---------
        route : ndarray
            (3,N)-shaped array of floats containing the global route
        """
        self.GLOBAL_PATH = route
        # print("Setting a global route")
    
    ################################################################################
    
    def _updateCurPos(self, pos):
        self.CUR_POS = pos
        
    def _updateCurVel(self, vel):
        self.CUR_VEL = vel
        
    def _batchRayCast(self):
        """
        Update self.DETECTED_OBS_IDS based on batch ray casting. DETECTED_OBS_IDS is a list of 
        detected obstacle's id

        Returns:
            None.

        """
        # rayFrom = self.CUR_POS
        # p.removeAllUserDebugItems()
        detected_obs_ids = []
        rayTo = []
        rayIds = []
        # numRays = 1024
        numRays = 500
        # numRays = 250
        rayLen = 1.5
        rayHitColor = [1, 0, 0]
        rayMissColor = [0, 1, 0]

        replaceLines = True

        # sunflower on a sphere: https://stackoverflow.com/questions/9600801/evenly-distributing-n-points-on-a-sphere/44164075#44164075
        indices = np.arange(0, numRays, dtype=float) + 0.5


        phi = np.arccos(1 - 2*indices/numRays)
        theta = np.pi * (1 + 5**0.5) * indices

        x, y, z = rayLen* np.cos(theta) * np.sin(phi), rayLen* np.sin(theta) * np.sin(phi), rayLen*np.cos(phi);
        rayFrom = [self.CUR_POS for _ in range(numRays)]
        rayTo = [[self.CUR_POS[0]+x[i], self.CUR_POS[1]+y[i], self.CUR_POS[2]+z[i]] for i in range(numRays)]
        # rayIds = [p.addUserDebugLine(rayFrom[i], rayTo[i], rayMissColor) for i in range(numRays)]
        results = p.rayTestBatch(rayFrom, rayTo)
        
        for i in range(numRays):
            hitObjectUid = results[i][0]
            
            if (hitObjectUid < 0):
                hitPosition = [0, 0, 0]
                # p.addUserDebugLine(rayFrom[i], rayTo[i], rayMissColor, replaceItemUniqueId=rayIds[i], lifeTime=0.1)
            else:
                if hitObjectUid!=0:
                    detected_obs_ids.append(hitObjectUid) if hitObjectUid not in detected_obs_ids and hitObjectUid != 0 else detected_obs_ids
                    hitPosition = results[i][3]
                    # p.addUserDebugLine(rayFrom[i], hitPosition, rayHitColor, replaceItemUniqueId=rayIds[i], lifeTime=0.1)
                    p.addUserDebugLine(rayFrom[i], hitPosition, rayHitColor, lifeTime=0.1)
    
        self.DETECTED_OBS_IDS = detected_obs_ids

    ################################################################################
    
    def _processDetection(self, obstacle_data):
        """
        Screen obstacle_data based on the detection from self.DETECTED_OBS_IDS. 

        Args:
            obstacle_data (dict): dictionary of dictionary of obstacles data where
                key is the obstacle's id and values include 'position' and 'size'.

        Returns:
            None.

        """
        if len(self.DETECTED_OBS_IDS) != 0:
            tempObs = []
            for j in self.DETECTED_OBS_IDS:
                self.DETECTED_OBS_DATA[str(j)] = {"position": obstacle_data[str(j)]["position"],
                                                      "size": obstacle_data[str(j)]["size"]}
                tempObs.append(obstacle_data[str(j)]["position"])
        else:
            self.DETECTED_OBS_DATA = {}
            
        
        
        

    ################################################################################

    def computeRouteFromState(self,
                            route_timestep,
                            state,
                            home_pos,
                            target_pos,
                            speed_limit,
                            obstacle_data=None
                            ):
        """Interface method using `computeRoute`.

        It can be used to compute a route directly from the value of key "state"
        in the `obs` returned by a call to BaseAviary.step().

        Parameters
        ----------
        route_timestep : float
            The time step at which the route is computed.
        state : ndarray
            (20,)-shaped array of floats containing the current state of the drone.
        target_pos : ndarray
            (3,1)-shaped array of floats containing the desired position.
        obstacles_pos : ndarray
            (N,3)-shaped array of floats containing obstacles' positions. The first one is environment
        obstacles_size : ndarray
            (N,3)-shaped array of floats containing obstacles' sizes. The first one is environment
        """
        self.HOME= home_pos
        self.DESTINATION = target_pos
        
        self._processDetection(obstacle_data)
        
        
        return self.computeRoute(route_timestep=route_timestep,
                                   cur_pos=state[0:3],
                                   cur_quat=state[3:7],
                                   cur_vel=state[10:13],
                                   cur_ang_vel=state[13:16],
                                   home_pos = np.array((0,0,0)),
                                   target_pos=target_pos,
                                   speed_limit = speed_limit,
                                   obstacle_data = self.DETECTED_OBS_DATA
                                   )

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
        """Abstract method to compute the route for a single drone.

        It must be implemented by each subclass of `BaseRoute`.

        Parameters
        ----------
        route_timestep : float
            The time step at which the route is computed.
        cur_pos : ndarray
            (3,1)-shaped array of floats containing the current position.
        cur_quat : ndarray
            (4,1)-shaped array of floats containing the current orientation as a quaternion.
        cur_vel : ndarray
            (3,1)-shaped array of floats containing the current velocity.
        cur_ang_vel : ndarray
            (3,1)-shaped array of floats containing the current angular velocity.
        target_pos : ndarray
            (3,1)-shaped array of floats containing the desired position.

        """
        raise NotImplementedError

    ################################################################################

    def _plotRoute(self, path):
        pathColor = [0, 0, 1]
        for i in range(0, path.shape[1]-1, 1):
            p.addUserDebugLine(path[:,i], path[:,i+1], pathColor, lineWidth=5, lifeTime=0.1)

    def setIFDSCoefficients(self, rho0_ifds=None, sigma0_ifds=None, sf_ifds=None):
        """Sets the coefficients of the IFDS path planning algorithm.

        This method throws an error message and exist is the coefficients
        were not initialized (e.g. when the routing algorithm is not the IFDS).

        Parameters
        ----------
        rho0_ifds : float, optional
            Normal repulsive coefficients (minimum is 0.1).
        sigma0_ifds : float, optional
            Tangential repulsive coefficients (minimum is 0.1).
        sf : boolean, optional
            Shape-following feature activation.
        """
        ATTR_LIST = ['RHO0_IFDS', 'SIGMA0_IFDS', 'SF_IFDS']
        if not all(hasattr(self, attr) for attr in ATTR_LIST):
            print("[ERROR] in BaseRouting.setIFDSCoefficients(), not all IFDS coefficients exist as attributes in the instantiated routing class.")
            exit()
        else:
            self.RHO0_IFDS = self.RHO0_IFDS if rho0_ifds is None else rho0_ifds
            self.SIGMA0_IFDS = self.SIGMA0_IFDS if sigma0_ifds is None else sigma0_ifds
            self.SF_IFDS = self.SF0_IFDS if sf_ifds is None else sf_ifds

    ################################################################################
    
    def _getURDFParameter(self,
                          parameter_name: str
                          ):
        """Reads a parameter from a drone's URDF file.

        This method is nothing more than a custom XML parser for the .urdf
        files in folder `assets/`.

        Parameters
        ----------
        parameter_name : str
            The name of the parameter to read.

        Returns
        -------
        float
            The value of the parameter.

        """
        #### Get the XML tree of the drone model to control ########
        URDF = self.DRONE_MODEL.value + ".urdf"
        URDF_TREE = etxml.parse(os.path.dirname(os.path.abspath(__file__))+"/../assets/"+URDF).getroot()
        #### Find and return the desired parameter #################
        if parameter_name == 'm':
            return float(URDF_TREE[1][0][1].attrib['value'])
        elif parameter_name in ['ixx', 'iyy', 'izz']:
            return float(URDF_TREE[1][0][2].attrib[parameter_name])
        elif parameter_name in ['arm', 'thrust2weight', 'kf', 'km', 'max_speed_kmh', 'gnd_eff_coeff' 'prop_radius', \
                                'drag_coeff_xy', 'drag_coeff_z', 'dw_coeff_1', 'dw_coeff_2', 'dw_coeff_3']:
            return float(URDF_TREE[0].attrib[parameter_name])
        elif parameter_name in ['length', 'radius']:
            return float(URDF_TREE[1][2][1][0].attrib[parameter_name])
        elif parameter_name == 'collision_z_offset':
            COLLISION_SHAPE_OFFSETS = [float(s) for s in URDF_TREE[1][2][0].attrib['xyz'].split(' ')]
            return COLLISION_SHAPE_OFFSETS[2]
