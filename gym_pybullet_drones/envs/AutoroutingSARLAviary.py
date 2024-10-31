import numpy as np

from gym_pybullet_drones.envs.ExtendedSARLAviary import ExtendedSARLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType
from gym_pybullet_drones.routing.BaseRouting import RouteCommandFlag, RouteStatus

class AutoroutingSARLAviary(ExtendedSARLAviary):
    """Single agent RL problem: hover at position."""

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
        
        state = self._getDroneStateVector(0)
        curPos = np.array(state[0:3])
        
        norm_ep_time = (self.step_counter/self.PYB_FREQ ) / self.EPISODE_LEN_SEC
        elapsed_time_sec = self.step_counter/self.PYB_FREQ

        # ---------Reward design-------------
        reachThreshold_m = 0.5  #0.2
        reward_choice = 10  # 4:best  8: best
        # prevd2destin = np.linalg.norm(self.TARGET_POS - self.CURRENT_POS)
        # d2destin = np.linalg.norm(self.TARGET_POS - state[0:3])
        # h2destin = np.linalg.norm(self.TARGET_POS - self.HOME_POS)
        # self.CURRENT_POS = curPos
        # CHECK THIS AGAIN!!!!!!
        prevd2destin = np.linalg.norm(self.routing[0].DESTINATION - self.routing[0].CUR_POS)
        d2destin = np.linalg.norm(self.routing[0].DESTINATION - state[0:3])
        h2destin = np.linalg.norm(self.routing[0].DESTINATION - self.routing[0].HOME_POS)
        self.routing[0].CUR_POS = curPos

        if reward_choice == 8:
            """Positive reward design: reach destination asap"""
            step_cost = (prevd2destin - d2destin) * (1/d2destin)
            collide_reward = -10 + step_cost
            destin_reward = 100*(1/d2destin)

            ret = step_cost
            if np.linalg.norm(self.routing[0].DESTINATION-state[0:3]) < reachThreshold_m:
                ret = destin_reward
                print(f"\n====== Reached Destination!!! ====== reward = {ret}\n")

            elif int(self.CONTACT_FLAGS[0]) == 1:
                ret = collide_reward
                # print(f"\n***Collided*** ret = {ret}\n")
        elif reward_choice == 9:
            """Encourage keeping distance > 10% of ray (0.15 m) away from obstacles"""
            step_cost = (prevd2destin - d2destin) * (1/d2destin)
            collide_reward = -10 + step_cost
            destin_reward = 100*(1/d2destin)

            ret = step_cost
            if np.linalg.norm(self.routing[0].DESTINATION-state[0:3]) <= reachThreshold_m:
                ret = destin_reward
                print(f"\n====== Reached Destination!!! ====== reward = {ret}\n")

            elif int(self.CONTACT_FLAGS[0]) == 1:
                ret = collide_reward
            elif any(self.routing[0].RAYS_INFO[:,1]<0.1):
                ret = step_cost/2
        elif reward_choice == 10:
            """Same as reward8, but discourage the use of local path"""
            step_reward = 1000*(prevd2destin - d2destin) * (1/d2destin) # If step_reward too high -> agent tends to hover near the destination
            collide_reward = -10 + step_reward
            # destin_reward = 100*(1/d2destin)
            destin_reward = 100*h2destin

            if np.linalg.norm(self.routing[0].DESTINATION-state[0:3]) < reachThreshold_m:
                ret = destin_reward
                print(f"\n====== Reached Destination!!! ====== reward = {ret}\n")

            elif int(self.CONTACT_FLAGS[0]) == 1:
                ret = collide_reward
                # print(f"\n***Collided*** ret = {ret}\n")
            else:
                # if self.routing[0].STAT[0] == RouteStatus.LOCAL:
                if self.routing[0].COMMANDS[0]._name == RouteCommandFlag.FOLLOW_LOCAL.value:
                    
                    # ret = step_reward/2
                    # ret = step_reward/4
                    ret = 0
                    # print(f"ALERT: using local route, ret = {ret}\n")
                else:
                    
                    ret = step_reward
                    # print(f"\nUse Global Route ret = {ret}\n")


        return ret

    ################################################################################
    
    def _computeTerminated(self):
        """Computes the current done value.

        Returns
        -------
        bool
            Whether the current episode is done.

        """
        state = self._getDroneStateVector(0)
        # cond2 : reached destination area
        # cond2 = np.linalg.norm(self.routing.DESTINATION.reshape(3,1) - state[0:3].reshape(3,1)) <= 0.5
        
        reachThreshold_m = 0.5  #0.0001

        if np.linalg.norm(self.routing[0].DESTINATION-state[0:3]) <= reachThreshold_m:
            return True
        elif int(self.CONTACT_FLAGS[0]) == 1:
            return True
        else:
            self.COMPUTE_DONE = False
            return False
        
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
