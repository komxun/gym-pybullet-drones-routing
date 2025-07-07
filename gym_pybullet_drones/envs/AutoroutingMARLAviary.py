import numpy as np

from gym_pybullet_drones.envs.ExtendedMARLAviary import ExtendedMARLAviary
from gym_pybullet_drones.envs.ExtendedMARLAviary2 import ExtendedMARLAviary2
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType
from gym_pybullet_drones.routing.BaseRouting import RouteCommandFlag, RouteStatus
from ray.rllib.env.multi_agent_env import MultiAgentEnv

class AutoroutingMARLAviary(ExtendedMARLAviary, MultiAgentEnv):
    """Multi agent RL problem: Multiple-routing."""

    ################################################################################
    
    def __init__(self,
                 drone_model: DroneModel=DroneModel.CF2X,
                 num_drones: int=1,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics=Physics.PYB,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 240,
                 gui=False,
                 record=False,
                 obs: ObservationType=ObservationType.KIN,
                 act: ActionType=ActionType.AUTOROUTING
                 ):
        """Initialization of a single agent RL environment.

        Using the generic single agent RL superclass.

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
            The type of action space (1 or 3D; RPMS, thurst and torques, or waypoint with PID control)

        """
        # self.TARGET_POS = np.array([0.2, 8, 1])
        self.EPISODE_LEN_SEC = 20
        super().__init__(drone_model=drone_model,
                         num_drones=num_drones,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         pyb_freq=pyb_freq,
                         ctrl_freq=ctrl_freq,
                         gui=gui,
                         record=record,
                         obs=obs,
                         act=act
                         )
        self.CURRENT_POS = self.HOME_POS

    ################################################################################
    
    def _computeReward(self):
        """Computes the current reward value.

        Returns
        -------
        float
            The reward.

        """
        
        # state = self._getDroneStateVector(0)
        
        
        norm_ep_time = (self.step_counter/self.PYB_FREQ ) / self.EPISODE_LEN_SEC
        elapsed_time_sec = self.step_counter/self.PYB_FREQ

        ret = 0
        # for i in range(self.NUM_DRONES):
        #     ret += self._calcRewardPerDrone(i)
        for i in range(1):
            ret += self._calcRewardPerDrone(i)
        return ret


    def _calcRewardPerDrone(self, drone_idx):
        """Computes reward per drone
        """
        state = self._getDroneStateVector(drone_idx)
        curPos = np.array(state[0:3])
         # ---------Reward design-------------
        reachThreshold_m = 0.5  #0.2
        reward_choice = 8  # 4:best  8: best  10: 2nd best 11: Good

        prevd2destin = np.linalg.norm(self.routing[drone_idx].DESTINATION - self.routing[drone_idx].CUR_POS)
        d2destin = np.linalg.norm(self.routing[drone_idx].DESTINATION - state[0:3])
        h2destin = np.linalg.norm(self.routing[drone_idx].DESTINATION - self.routing[drone_idx].HOME_POS)
        self.routing[drone_idx].CUR_POS = curPos

        if reward_choice == 8:
            """Positive reward design: reach destination asap"""
            step_cost = (prevd2destin - d2destin) * (1/d2destin)
            collide_reward = -10 + step_cost
            destin_reward = 100*(1/d2destin)

            reward = step_cost
            if np.linalg.norm(self.routing[0].DESTINATION-state[0:3]) < reachThreshold_m:
                reward = destin_reward
                print(f"\n [drone {drone_idx}]: ====== Reached Destination!!! ====== reward = {reward}\n")

            elif int(self.CONTACT_FLAGS[0]) == 1:
                reward = collide_reward
                # print(f"\n***Collided*** ret = {ret}\n")
        return reward

    ################################################################################
    
    def _computeTerminated(self):
        """Computes the current done value.

        Returns
        -------
        bool
            Whether the current episode is done.

        """
        
        # cond2 : reached destination area
        # cond2 = np.linalg.norm(self.routing.DESTINATION.reshape(3,1) - state[0:3].reshape(3,1)) <= 0.5
        
        num_drones = self.NUM_DRONES
        num_drones = 1
        reachThreshold_m = 0.5  #0.0001
        drones_reached = np.zeros(num_drones)
        for i in range(num_drones):
            state = self._getDroneStateVector(i)
            if np.linalg.norm(self.routing[i].DESTINATION-state[0:3]) <= reachThreshold_m:
                drones_reached[i] = 1
            elif int(self.CONTACT_FLAGS[i]) == 1:
                return True
            else:
                self.COMPUTE_DONE = False
                return False
        
        if np.all(drones_reached==1):
            print(f"****** ALL DRONES HAS REACHED THEIR DESTINATIONS!!!!!! *******")
            return True
        
    ################################################################################
    
    def _computeTruncated(self):
        """Computes the current truncated value.

        Returns
        -------
        bool
            Whether the current episode timed out.

        """
        mapBorderXYZ = [10,15,5]
        # TODO replace "0" with ith drones
        state = self._getDroneStateVector(0)

        # Truncate when the drone collides
        # if int(self.CONTACT_FLAGS[0]) == 1:
        #     # print(f"Ayooo it collides!!!")
        #     return True
        
        # if (abs(state[0]) > mapBorderXYZ[0] or abs(state[1]) > mapBorderXYZ[1] or state[2] > mapBorderXYZ[2] # Truncate when the drone is too far away
        #      or abs(state[7]) > .4 or abs(state[8]) > .4 # Truncate when the drone is too tilted
        # ):
        #     return True
        if self.step_counter/self.PYB_FREQ > self.EPISODE_LEN_SEC:
            print("\n-------- Truncated due to time out ----------\n")
            return True
        else:
            return False

    ################################################################################
    
    def _computeInfo(self):
        """Computes the current info dict(s).

        Unused.

        Returns
        -------
        dict[str, int]
            Dummy value.

        """
        return {"answer": 42} #### Calculated by the Deep Thought supercomputer in 7.5M years
