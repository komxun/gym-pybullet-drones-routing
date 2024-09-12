import os
import numpy as np
from gym import spaces
import pybullet as p

from gym_pybullet_drones.envs.BaseAviary import DroneModel, Physics, BaseAviary
from gym_pybullet_drones.envs.single_agent_rl.ExtendedSingleAgentAviary import ActionType, ObservationType, ExtendedSingleAgentAviary


class AutoroutingAviary(ExtendedSingleAgentAviary):
    """Single agent RL problem: Autorouting for Collision Avoidance."""
    
    ################################################################################
    
    # def __init__(self,
    #              drone_model: DroneModel=DroneModel.CF2X,
    #              initial_xyzs=np.array([0,0,0.5]).reshape(1,3),
    #              initial_rpys=None,
    #              physics: Physics=Physics.PYB,
    #              freq: int=240,
    #              aggregate_phy_steps: int=1,
    #              gui=False,
    #              record=False, 
    #              obs: ObservationType=ObservationType.KIN,
    #              act: ActionType=ActionType.RPM
    #              ):
    #     """Initialization of a single agent RL environment.

    #     Using the generic single agent RL superclass.

    #     Parameters
    #     ----------
    #     drone_model : DroneModel, optional
    #         The desired drone type (detailed in an .urdf file in folder `assets`).
    #     initial_xyzs: ndarray | None, optional
    #         (NUM_DRONES, 3)-shaped array containing the initial XYZ position of the drones.
    #     initial_rpys: ndarray | None, optional
    #         (NUM_DRONES, 3)-shaped array containing the initial orientations of the drones (in radians).
    #     physics : Physics, optional
    #         The desired implementation of PyBullet physics/custom dynamics.
    #     freq : int, optional
    #         The frequency (Hz) at which the physics engine steps.
    #     aggregate_phy_steps : int, optional
    #         The number of physics steps within one call to `BaseAviary.step()`.
    #     gui : bool, optional
    #         Whether to use PyBullet's GUI.
    #     record : bool, optional
    #         Whether to save a video of the simulation in folder `files/videos/`.
    #     obs : ObservationType, optional
    #         The type of observation space (kinematic information or vision)
    #     act : ActionType, optional
    #         The type of action space (1 or 3D; RPMS, thurst and torques, or waypoint with PID control)

    #     """
    #     super().__init__(drone_model=drone_model,
    #                      initial_xyzs=initial_xyzs,
    #                      initial_rpys=initial_rpys,
    #                      physics=physics,
    #                      freq=freq,
    #                      aggregate_phy_steps=aggregate_phy_steps,
    #                      gui=gui,
    #                      record=record,
    #                      obs=obs,
    #                      act=act
    #                      )
    
    def _computeReward(self):
        """Computes the current reward value.

        Returns
        -------
        float
            The reward.

        """
        choice = 2
        
        state = self._getDroneStateVector(0)
        curPos = np.array(state[0:3])
        norm_ep_time = (self.step_counter/self.SIM_FREQ) / self.EPISODE_LEN_SEC
        
        if choice == 1:
            # Simple reward based on closeness to destination over time
            reward = -10 * norm_ep_time* np.linalg.norm(np.array([0.2, 12, 1])-state[0:3])**2
        elif choice == 2:
            # Added penalty to collision
            # reward = -10 * norm_ep_time* np.linalg.norm(np.array([0.2, 12, 1]).reshape(1,3) - curPos.reshape(1,3))**2
            reward = -10 * norm_ep_time
            # reward = -10 * norm_ep_time
            if self.CONTACT_FLAGS[0] == 1:
                # print("Penalty due to collision")
                # reward -= 10000
                # reward *= 2
                reward = -1e4
                # print(f"Penalty due to collision: reward = {reward}")
            # print("Reward = " + str(reward))
        else:
            raise NotImplementedError
            
        return reward

    ################################################################################
    
    def _computeDone(self):
        """Computes the current done value.

        Returns
        -------
        bool
            Whether the current episode is done.

        """
        state = self._getDroneStateVector(0)
        # cond1 : time exceeded limit
        cond1 = self.step_counter/self.SIM_FREQ > self.EPISODE_LEN_SEC
        # cond2 : reached destination area
        cond2 = np.linalg.norm(self.routing.DESTINATION.reshape(3,1) - state[0:3].reshape(3,1)) <= 0.5
        # cond3 : collided
        cond3 = self.CONTACT_FLAGS[0] == 1
        if cond1 or cond3 or cond2:
            # if cond1:
            #     print("########## Exit episode due to timeout ############")
            # if cond2:
            #     print("o=o=o Exit episode due to destination reached o=o=o")
            # if cond3:
            #     print('!!!!!!!!! Exit episode due to collision !!!!!!!!!!!')
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

    ################################################################################
    # def _normalizeDetection(self):
    #     posList = []
    #     sizeList = []
    #     obstacles_pos = np.array([])
    #     obstacles_size = np.array([])
    #     if bool(obstacle_data):  # Boolean of empty dict return False
    #         for j in self.DETECTED_OBS_IDS:
    #             posList.append(obstacle_data[str(j)]["position"])
    #             sizeList.append(obstacle_data[str(j)]["size"])
    #         obstacles_pos = np.array(posList).reshape(len(self.DETECTED_OBS_IDS), 3)
    #         obstacles_size = np.array(sizeList).reshape(len(self.DETECTED_OBS_IDS), 3)
    
    
    
    def _clipAndNormalizeState(self,
                               state
                               ):
        """Normalizes a drone's state to the [-1,1] range.

        Parameters
        ----------
        state : ndarray
            (12,)-shaped array of floats containing the non-normalized state of a single drone.

        Returns
        -------
        ndarray
            (13,)-shaped array of floats containing the normalized state of a single drone + normalized distance-to-destination.

        """
        MAX_LIN_VEL_XY = 3 
        MAX_LIN_VEL_Z = 1

        MAX_XY = MAX_LIN_VEL_XY*self.EPISODE_LEN_SEC
        MAX_Z = MAX_LIN_VEL_Z*self.EPISODE_LEN_SEC

        MAX_PITCH_ROLL = np.pi # Full range
        MAX_RANGE = 2*np.linalg.norm(self.routing.HOME_POS.reshape(3,1) - self.routing.DESTINATION.reshape(3,1))
        
        # [X, Y, Z, R, P, Y, Vx, Vy, Vz, Wx, Wy, Wz, d2destin]
        clipped_pos_xy = np.clip(state[0:2], -MAX_XY, MAX_XY)
        clipped_pos_z = np.clip(state[2], 0, MAX_Z)
        clipped_rp = np.clip(state[3:5], -MAX_PITCH_ROLL, MAX_PITCH_ROLL)
        clipped_vel_xy = np.clip(state[6:8], -MAX_LIN_VEL_XY, MAX_LIN_VEL_XY)
        clipped_vel_z = np.clip(state[8], -MAX_LIN_VEL_Z, MAX_LIN_VEL_Z)
        
        d2destin = np.linalg.norm(state[0:3].reshape(3,1) - self.routing.DESTINATION.reshape(3,1))
        clipped_d2destin = np.clip(d2destin, 0, MAX_RANGE)
        
        if self.GUI:
            self._clipAndNormalizeStateWarning(state,
                                               clipped_pos_xy,
                                               clipped_pos_z,
                                               clipped_rp,
                                               clipped_vel_xy,
                                               clipped_vel_z
                                               )

        normalized_pos_xy = clipped_pos_xy / MAX_XY
        normalized_pos_z = clipped_pos_z / MAX_Z
        normalized_rp = clipped_rp / MAX_PITCH_ROLL
        normalized_y = state[5] / np.pi     # No reason to clip
        normalized_vel_xy = clipped_vel_xy / MAX_LIN_VEL_XY
        normalized_vel_z = clipped_vel_z / MAX_LIN_VEL_XY
        normalized_ang_vel  = state[9:12]/np.linalg.norm(state[9:12]) if np.linalg.norm(state[9:12]) != 0 else state[9:12]
        
        normalized_d2destin = clipped_d2destin/ MAX_RANGE
        

        norm_and_clipped = np.hstack([normalized_pos_xy,
                                      normalized_pos_z,
                                      normalized_rp,
                                      normalized_y,
                                      normalized_vel_xy,
                                      normalized_vel_z,
                                      normalized_ang_vel,
                                      normalized_d2destin
                                      ]).reshape(13,)

        return norm_and_clipped
    
    ################################################################################
    
    def _clipAndNormalizeStateWarning(self,
                                      state,
                                      clipped_pos_xy,
                                      clipped_pos_z,
                                      clipped_rp,
                                      clipped_vel_xy,
                                      clipped_vel_z,
                                      ):
        """Debugging printouts associated to `_clipAndNormalizeState`.

        Print a warning if values in a state vector is out of the clipping range.
        
        """
        if not(clipped_pos_xy == np.array(state[0:2])).all():
            print("[WARNING] it", self.step_counter, "in AutoroutingAviary._clipAndNormalizeState(), clipped xy position [{:.2f} {:.2f}]".format(state[0], state[1]))
        if not(clipped_pos_z == np.array(state[2])).all():
            print("[WARNING] it", self.step_counter, "in AutoroutingAviary._clipAndNormalizeState(), clipped z position [{:.2f}]".format(state[2]))
        if not(clipped_rp == np.array(state[3:5])).all():
            print("[WARNING] it", self.step_counter, "in AutoroutingAviary._clipAndNormalizeState(), clipped roll/pitch [{:.2f} {:.2f}]".format(state[7], state[8]))
        if not(clipped_vel_xy == np.array(state[6:8])).all():
            print("[WARNING] it", self.step_counter, "in AutoroutingAviary._clipAndNormalizeState(), clipped xy velocity [{:.2f} {:.2f}]".format(state[10], state[11]))
        if not(clipped_vel_z == np.array(state[8])).all():
            print("[WARNING] it", self.step_counter, "in AutoroutingAviary._clipAndNormalizeState(), clipped z velocity [{:.2f}]".format(state[12]))
