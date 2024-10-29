import numpy as np

from gym_pybullet_drones.envs.ExtendedRLAviary import ExtendedRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType

class AutoroutingRLAviary(ExtendedRLAviary):
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
        self.TARGET_POS = np.array([0.2, 8, 1])
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
        reward_choice = 9  # 4:best  8: best
        prevd2destin = np.linalg.norm(self.TARGET_POS - self.CURRENT_POS)
        d2destin = np.linalg.norm(self.TARGET_POS - state[0:3])
        h2destin = np.linalg.norm(self.TARGET_POS - self.HOME_POS)
        self.CURRENT_POS = curPos
        
        if reward_choice == 1:
            # Initialize reward
            ret = -1 
            
            # Check if the UAV reached the goal
            if np.linalg.norm(self.TARGET_POS-state[0:3]) < reachThreshold_m:
                ret = 20
            elif int(self.CONTACT_FLAGS[0]) == 1:
                # print("- penalty from collision")
                ret = -10
                # self.COMPUTE_DONE = True
        elif reward_choice ==2:
            ret = -10 * elapsed_time_sec * np.linalg.norm(np.array([0.2, 12, 1]).reshape(1,3) - curPos.reshape(1,3))**2
            
            if int(self.CONTACT_FLAGS[0]) == 1:
                ret *= 2
                print(f"Penalty due to collision: reward = {ret}")
            # print("Reward = " + str(reward))
            # ret = max(0, 2 - np.linalg.norm(self.TARGET_POS-state[0:3])**4)
        elif reward_choice ==3:
            step_cost = d2destin
            desire_reach_time_s = 5
            # destin_reward = desire_reach_time_s * self.PYB_FREQ * step_cost
            destin_reward = desire_reach_time_s * self.PYB_FREQ * h2destin
            ret = 0
            ret -= step_cost

            if np.linalg.norm(self.TARGET_POS-state[0:3]) < reachThreshold_m:
                print("====== Reached Destination!!! ======")
                ret += destin_reward

            if int(self.CONTACT_FLAGS[0]) == 1:
                # print("Collided!")
                ret = -(destin_reward - self.step_counter*step_cost)
        elif reward_choice ==4:
            # step_cost = d2destin**1
            step_cost = 10
            desire_reach_time_s = 10
            desire_num_step = desire_reach_time_s * self.PYB_FREQ

            a_1 = reachThreshold_m
            a_n = h2destin
            n = desire_reach_time_s * self.PYB_FREQ

            d2destin_vect = np.linspace(reachThreshold_m, h2destin, desire_num_step)
            stepCost_vect = d2destin_vect**1

            destin_reward = desire_num_step * step_cost
            # destin_reward = sum(stepCost_vect)
            # print(f"destin reward = {destin_reward}")
            ret = 0
            ret -= step_cost

            if np.linalg.norm(self.TARGET_POS-state[0:3]) < reachThreshold_m:
                
                # ret += destin_reward
                ret = destin_reward
                print(f"\n====== Reached Destination!!! ====== ret = {ret}\n")

            elif int(self.CONTACT_FLAGS[0]) == 1:
                # ret -= d2destin
                ret = -d2destin
                # print(f"Collided: ret = {ret}")
        elif reward_choice ==5:
            d2destin = np.linalg.norm(self.TARGET_POS - state[0:3])
            h2destin = np.linalg.norm(self.TARGET_POS - self.HOME_POS)
            # step_cost = d2destin**1
            
            step_cost = self.step_counter
            desire_reach_time_s = 10
            desire_num_step = desire_reach_time_s * self.PYB_FREQ
            ret = desire_num_step - self.step_counter

            d2destin_vect = np.linspace(reachThreshold_m, h2destin, desire_num_step)
            stepCost_vect = d2destin_vect**1

            destin_reward = desire_num_step
            # destin_reward = sum(stepCost_vect)
            # print(f"destin reward = {destin_reward}")
            
            ret -= step_cost

            if np.linalg.norm(self.TARGET_POS-state[0:3]) < reachThreshold_m:
                print("====== Reached Destination!!! ======")
                ret += destin_reward

            if int(self.CONTACT_FLAGS[0]) == 1:
                print("Collided!")
                ret -= d2destin/h2destin * 10*desire_num_step
            # print(f"reward = {ret}")
        elif reward_choice == 6:
            step_cost = d2destin/h2destin
            desire_reach_time_s = 20
            desire_num_step = desire_reach_time_s * self.PYB_FREQ
            ret = 0
            d2destin_vect = np.linspace(reachThreshold_m, h2destin, desire_num_step)
            stepCost_vect = d2destin_vect/h2destin

            # destin_reward = desire_num_step
            collide_reward = 2*step_cost
            destin_reward = sum(stepCost_vect)

            
            ret -= step_cost

            if np.linalg.norm(self.TARGET_POS-state[0:3]) < reachThreshold_m:
                ret = destin_reward
                print(f"\n====== Reached Destination!!! ====== reward = {ret}\n")

            elif int(self.CONTACT_FLAGS[0]) == 1:
                ret = -collide_reward
                # print(f"\n***Collided*** ret = {ret}\n")
        elif reward_choice == 7:
            # Positive reward
            step_cost = (1/d2destin)
            collide_reward = -10*step_cost
            destin_reward = 10*step_cost

            ret = step_cost
            if np.linalg.norm(self.TARGET_POS-state[0:3]) < reachThreshold_m:
                ret = destin_reward
                print(f"\n====== Reached Destination!!! ====== reward = {ret}\n")

            elif int(self.CONTACT_FLAGS[0]) == 1:
                ret = collide_reward
                # print(f"\n***Collided*** ret = {ret}\n")
        elif reward_choice == 8:
            # Positive reward
            step_cost = (prevd2destin - d2destin) * (1/d2destin)
            collide_reward = -10 + step_cost
            destin_reward = 100*(1/d2destin)

            ret = step_cost
            if np.linalg.norm(self.TARGET_POS-state[0:3]) < reachThreshold_m:
                ret = destin_reward
                print(f"\n====== Reached Destination!!! ====== reward = {ret}\n")

            elif int(self.CONTACT_FLAGS[0]) == 1:
                ret = collide_reward
                # print(f"\n***Collided*** ret = {ret}\n")
        elif reward_choice == 9:
            step_cost = (prevd2destin - d2destin) * (1/d2destin)
            collide_reward = -10 + step_cost
            destin_reward = 100*(1/d2destin)

            ret = step_cost
            if np.linalg.norm(self.TARGET_POS-state[0:3]) < reachThreshold_m:
                ret = destin_reward
                print(f"\n====== Reached Destination!!! ====== reward = {ret}\n")

            elif int(self.CONTACT_FLAGS[0]) == 1:
                ret = collide_reward
            elif any(self.routing[0].RAYS_INFO[:,1]<0.1):
                ret = step_cost/2

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

        if np.linalg.norm(self.TARGET_POS-state[0:3]) < reachThreshold_m:
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
