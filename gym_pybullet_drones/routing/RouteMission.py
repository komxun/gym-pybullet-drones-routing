import numpy as np
import random
class RouteMission:
    
    def __init__(self, numDrones=1, scenario=1, timeLimit=None):
        self.NUM_DRONES = numDrones
        self.SCENE = scenario
        self.TIME_LIMIT = timeLimit
        self.INIT_XYZS = np.zeros(3)
        self.INIT_RPYS = np.zeros(3)
        self.DESTINS   = np.zeros(3)
        self.WAYPOINTS = []
        

    def generateMission(self, numDrones=1, scenario=1, timeLimit=None):
        self.NUM_DRONES = numDrones
        self.SCENE = scenario
        self.TIME_LIMIT = timeLimit
        if numDrones == 1:
            self._singleAgentMission(scenario)
        elif numDrones > 1:
            self._multiAgentMission(scenario, numDrones)
        else:
            raise ValueError("[Error] in RouteMission: Invalid number of drones")
        
    def generateRandomMission(self, maxNumDrone, minNumDrone=1, seed=None):
        # Function input validation
        if not isinstance(minNumDrone, int) or not isinstance(maxNumDrone, int):
            raise TypeError("[Error] in generateRandomMission: Both minNumDrone and maxNumDrone must be integers.")
        if minNumDrone < 0 or maxNumDrone < 0:
            raise ValueError("[Error] in generateRandomMission:Both minNumDrone and maxNumDrone must be non-negative.")
        if minNumDrone > maxNumDrone:
            raise ValueError("[Error] in generateRandomMission: minNumDrone cannot be greater than maxNumDrone.")
        if seed is not None and not isinstance(seed, int):
            raise TypeError("[Error] in generateRandomMission:Seed must be an integer or None.")
        
        # Generate a random number of drones between minNumDrone and maxNumDrone
        num_drones = random.randint(minNumDrone, maxNumDrone)
        self.NUM_DRONES = num_drones

        # Set the random seed for reproducibility if provided
        if seed:
            print(f"\nRANDOM MISSION IS GENERATE WITH SEED = {seed}\n")
            random.seed(seed)
        else:
            print(f"\n[RouteMission]: Generating random scenario with {self.NUM_DRONES} drones.\n")

        # Define constants
        ORIGIN = [0, 9, 1]     # Reference point for positioning
        BASE_R = 4             # Base radius for initial position
        BASE_R_D = 4          # Base radius for destination position
        H_STEP = 0           # Base vertical step increment for each drone
        # RADIUS_VARIATION = 1 # Max random variation for radius
        RADIUS_VARIATION = 0.2
        # ANGLE_VARIATION = 0.2  # Max random variation for angle (in radians)
        ANGLE_VARIATION = 90 * np.pi/180
        # Z_VARIATION = 0.2  #0.3     # Max random variation for initial z-axis
        Z_VARIATION = 0.1

        # Initialize arrays
        INIT_XYZS = np.zeros((num_drones, 3))
        INIT_RPYS = np.zeros((num_drones, 3))
        DESTINS = np.zeros((num_drones, 3))
        WAYPOINTS = []

        # Set the first drone's position, orientation, and destination explicitly
        INIT_XYZS[0] = [0, 0, ORIGIN[2]]
        INIT_RPYS[0] = [0, 0, 0]
        DESTINS[0] = [0, 11, 1]
        WAYPOINTS.append(np.vstack((INIT_XYZS[0], DESTINS[0])))

        # Position remaining drones in circular paths with added randomness
        for i in range(1, num_drones):
            # Random radius and angle variations for initial positions
            if i==0:
                random_radius = BASE_R+2
            else:
                random_radius = BASE_R + random.uniform(-RADIUS_VARIATION, RADIUS_VARIATION)
            random_angle_init = (i / num_drones) * 2 * np.pi + np.pi / 2 + random.uniform(-ANGLE_VARIATION, ANGLE_VARIATION)
            
            # Initial position with random radius and angle
            INIT_XYZS[i] = [
                ORIGIN[0] + random_radius * np.cos(random_angle_init),
                ORIGIN[1] + random_radius * np.sin(random_angle_init) - BASE_R,
                ORIGIN[2] + i * H_STEP + random.uniform(-Z_VARIATION, Z_VARIATION)
            ]

            # Orientation (roll, pitch fixed, random yaw)
            INIT_RPYS[i] = [0, 0, random.uniform(0, 2 * np.pi)]

            # Random radius and angle variations for destination positions
            random_radius_dest = BASE_R_D + random.uniform(-RADIUS_VARIATION, RADIUS_VARIATION)
            random_angle_dest = random_angle_init + np.pi / 2 + random.uniform(-ANGLE_VARIATION, ANGLE_VARIATION)
            
            # Destination position with random radius and angle
            DESTINS[i] = [
                ORIGIN[0] + random_radius_dest * np.cos(random_angle_dest),
                ORIGIN[1] + random_radius_dest * np.sin(random_angle_dest) - BASE_R_D,
                ORIGIN[2] + i * H_STEP + random.uniform(-Z_VARIATION, Z_VARIATION)
                # ORIGIN[2]  # Keep z-level consistent for destinations
            ]

            WAYPOINTS.append(np.vstack((INIT_XYZS[i], DESTINS[i])))

        self.NUM_DRONES = num_drones
        self.INIT_XYZS = INIT_XYZS
        self.INIT_RPYS = INIT_RPYS
        self.DESTINS   = DESTINS
        self.WAYPOINTS = WAYPOINTS

    def _singleAgentMission(self, scenario):
        if not isinstance(scenario, int):
            raise TypeError("[Error] in _singleAgentMission: scenario number must be positive integer.")
        if scenario <0:
            raise ValueError("[Error] in _singleAgentMission: scenario number must be non-negative.")
        
        if scenario == 1:
            """Simple straight line"""
            INIT_XYZS = np.array([0, 0, 0.5]).reshape(1,3)
            INIT_RPYS = np.array([0, 0, 0]).reshape(1,3)
            DESTINS   = np.array([0, 12, 0.5]).reshape(1,3)

            WAYPOINTS = np.vstack((INIT_XYZS, DESTINS))
        else:
            raise ValueError("[Error] in RouteMission: Undefined scenario")
        self.INIT_XYZS = INIT_XYZS
        self.INIT_RPYS = INIT_RPYS
        self.DESTINS   = DESTINS
        self.WAYPOINTS = WAYPOINTS

    def _multiAgentMission(self, scenario, num_drones):
        if num_drones == 1:
            print("[WARNING] Creating Multi-Agent mission for one agent.")
        if scenario == 1:
            """Parallel straight lines (scanning)"""
            INIT_XYZS = np.array([[((-1)**i)*(i*0.2)+0.5,-3*(i*0.05), 0.5+ 0.05*i ] for i in range(num_drones)])
            INIT_RPYS = np.array([[0, 0,  i * (np.pi/2)/num_drones] for i in range(num_drones)])
            DESTINS = np.array([[((-1)**i)*(i*0.2), 12, 0.5] for i in range(num_drones)])
            WAYPOINTS = np.vstack((INIT_XYZS, DESTINS))
        
        elif scenario == 2:
            """Random straight lines in an oval"""
            H_STEP = 0.1
            R = 1  # 2
            R_D = 1  #2
            ORIGIN = [0,4,1]
            # Initialize empty arrays
            INIT_XYZS = np.zeros((num_drones, 3))
            INIT_RPYS = np.zeros((num_drones, 3))
            DESTINS = np.zeros((num_drones, 3))
            WAYPOINTS = []

            INIT_XYZS[0] = [0,-0.7,ORIGIN[2]]
            INIT_RPYS[0] = [0,0,0]
            DESTINS[0] = [0,5,1]
            WAYPOINTS.append(np.vstack((INIT_XYZS[0], DESTINS[0])))

            for i in range(1, num_drones):
                INIT_XYZS[i] = [ORIGIN[0]+(R)*np.cos((i/num_drones)*2*np.pi+np.pi/2),
                                ORIGIN[1]+(R)*np.sin((i/num_drones)*2*np.pi+np.pi/2)-(R), 
                                ORIGIN[2]+H_STEP]
                # INIT_XYZS[i] = [ORIGIN[0]+(R)*np.cos((i/6)*2*np.pi+np.pi/2),
                #                 ORIGIN[1]+(R)*np.sin((i/6)*2*np.pi+np.pi/2)-(R), 
                #                 ORIGIN[2]+H_STEP*i]
                
                INIT_RPYS[i] = [0, 0, i * (np.pi/2)/num_drones]
                DESTINS[i] = [ORIGIN[0]+(R_D)*np.cos((i/num_drones)*2*np.pi+np.pi/2+1*np.pi/2),
                                ORIGIN[1]+np.sin((i/num_drones)*2*np.pi+np.pi/2+3*np.pi/2)-(R_D), 
                                ORIGIN[2]]
                WAYPOINTS.append(np.vstack((INIT_XYZS[i], DESTINS[i])))
            
        else:
            raise ValueError("[Error] in RouteMission: Undefined scenario")
                
        self.INIT_XYZS = INIT_XYZS
        self.INIT_RPYS = INIT_RPYS
        self.DESTINS   = DESTINS
        self.WAYPOINTS = WAYPOINTS


