import numpy as np
class RouteMission:
    
    def __init__(self, numDrones=1, scenario=1, timeLimit=None):
        self.NUM_DRONES = numDrones
        self.SCENE = scenario
        self.TIME_LIMIT = timeLimit
        self.INIT_XYZS = np.zeros(3)
        self.INIT_RPYS = np.zeros(3)
        self.DESTINS   = np.zeros(3)
        self.WAYPOINTS = []
        

    def generateMission(self, numDrones=1, scenario=1, timeLimit=None, seed=None):
        self.NUM_DRONES = numDrones
        self.SCENE = scenario
        self.TIME_LIMIT = timeLimit
        if numDrones == 1:
            self._singleAgentMission(scenario)
        elif numDrones > 1:
            self._multiAgentMission(scenario, numDrones)
        else:
            raise ValueError("[Error] in RouteMission: Invalid number of drones")

    def _singleAgentMission(self, scenario):
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
            R = 2  # 2
            R_D = 4
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


