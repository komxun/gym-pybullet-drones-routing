import numpy as np

from gym_pybullet_drones.envs.ExtendedRLAviary import ExtendedRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType

class AutoroutingRLAviary(ExtendedRLAviary):
    """Single agent RL problem: hover at position."""

    ################################################################################
    
    def __init__(self,
                 drone_model: DroneModel=DroneModel.CF2X,
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
        self.TARGET_POS = np.array([0.2, 12, 1])
        self.EPISODE_LEN_SEC = 20
        super().__init__(drone_model=drone_model,
                         num_drones=1,
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

        # reward = -10 * norm_ep_time* np.linalg.norm(np.array([0.2, 12, 1]).reshape(1,3) - curPos.reshape(1,3))**2
        # ret = -10 * norm_ep_time
        ret = -10 * elapsed_time_sec * np.linalg.norm(np.array([0.2, 12, 1]).reshape(1,3) - curPos.reshape(1,3))**2
        # reward = -10 * norm_ep_time
        if int(self.CONTACT_FLAGS[0]) == 1:
            # reward -= 10000
            # ret -= -1e3
            ret *= 2
            print(f"Penalty due to collision: reward = {ret}")
        # print("Reward = " + str(reward))
        # ret = max(0, 2 - np.linalg.norm(self.TARGET_POS-state[0:3])**4)

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
        
        reachThreshold_m = 0.2  #0.0001

        if np.linalg.norm(self.TARGET_POS-state[0:3]) < reachThreshold_m:
            return True
        elif int(self.CONTACT_FLAGS[0]) == 1:
            return True
        else:
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
        #     print(f"Ayooo it collides!!!")
        #     return True
        
        if (abs(state[0]) > mapBorderXYZ[0] or abs(state[1]) > mapBorderXYZ[1] or state[2] > mapBorderXYZ[2] # Truncate when the drone is too far away
             or abs(state[7]) > .4 or abs(state[8]) > .4 # Truncate when the drone is too tilted
        ):
            return True
        if self.step_counter/self.PYB_FREQ > self.EPISODE_LEN_SEC:
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
