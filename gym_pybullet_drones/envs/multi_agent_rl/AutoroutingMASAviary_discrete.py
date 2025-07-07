import numpy as np

from gym_pybullet_drones.utils.enums import DroneModel, Physics
from collections import deque
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ActionType, ObservationType
from gym_pybullet_drones.envs.multi_agent_rl.ExtendedMultiagentAviary_discrete import ExtendedMultiagentAviary_discrete
from gym_pybullet_drones.routing.BaseRouting import RouteCommandFlag, RouteStatus

class AutoroutingMASAviary_discrete(ExtendedMultiagentAviary_discrete):
    """Multi-agent RL problem: leader-follower."""

    ################################################################################

    def __init__(self,
                 drone_model: DroneModel=DroneModel.CF2X,
                 num_drones: int=2,
                 neighbourhood_radius: float=np.inf,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics=Physics.PYB,
                 freq: int=240,
                 aggregate_phy_steps: int=1,  # used to be 1
                 gui=False,
                 record=False, 
                 obs: ObservationType=ObservationType.KIN,
                 act: ActionType=ActionType.AUTOROUTING):
        """Initialization of a multi-agent RL environment.

        Using the generic multi-agent RL superclass.

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
            The type of action space (1 or 3D; RPMS, thurst and torques, or waypoint with PID control)

        """
        self.ACTION_BUFFER_SIZE = int(freq//2)
        self.action_buffer = deque(maxlen=self.ACTION_BUFFER_SIZE)
        self.ALLOWED_WAITING_S = 5
        self.static_action_threshold = freq * self.ALLOWED_WAITING_S
        

        super().__init__(drone_model=drone_model,
                         num_drones=num_drones,
                         neighbourhood_radius=neighbourhood_radius,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         freq=freq,
                         aggregate_phy_steps=aggregate_phy_steps,
                         gui=gui,
                         record=record, 
                         obs=obs,
                         act=act
                         )

    ################################################################################
    
    # def _computeReward(self):
    #     """Computes the current reward value(s).

    #     Returns
    #     -------
    #     dict[int, float]
    #         The reward value for each drone.

    #     """
    #     rewards = {}
    #     states = np.array([self._getDroneStateVector(i) for i in range(self.NUM_DRONES)])
    #     rewards[0] = -1 * np.linalg.norm(np.array([0, 0, 0.5]) - states[0, 0:3])**2
    #     # rewards[1] = -1 * np.linalg.norm(np.array([states[1, 0], states[1, 1], 0.5]) - states[1, 0:3])**2 # DEBUG WITH INDEPENDENT REWARD 
    #     for i in range(1, self.NUM_DRONES):
    #         rewards[i] = -(1/self.NUM_DRONES) * np.linalg.norm(np.array([states[i, 0], states[i, 1], states[0, 2]]) - states[i, 0:3])**2
    #     return rewards

    def _computeReward(self):
        """Computes the current reward value.

        Returns
        -------
        dict[int, float]
            The reward.

        """
        
        
        
        
        # norm_ep_time = (self.step_counter/self.PYB_FREQ ) / self.EPISODE_LEN_SEC
        elapsed_time_sec = self.step_counter/self.SIM_FREQ

        # ---------Reward design-------------
        reachThreshold_m = 0.5  #0.2
        reward_choice = 8  # 8: best  10: 2nd best 11: Good
        # prevd2destin = np.linalg.norm(self.TARGET_POS - self.CURRENT_POS)
        # d2destin = np.linalg.norm(self.TARGET_POS - state[0:3])
        # h2destin = np.linalg.norm(self.TARGET_POS - self.HOME_POS)
        # self.CURRENT_POS = curPos
        # CHECK THIS AGAIN!!!!!!
        for i in range(0, self.NUM_DRONES):
            rewards = {}
            state = self._getDroneStateVector(i)
            curPos = np.array(state[0:3])
            prevd2destin = np.linalg.norm(self.routing[i].DESTINATION - self.routing[i].CUR_POS)
            d2destin = np.linalg.norm(self.routing[i].DESTINATION - state[0:3])
            h2destin = np.linalg.norm(self.routing[i].DESTINATION - self.routing[i].HOME_POS)
            self.routing[i].CUR_POS = curPos

            if reward_choice == 8:
                """Positive reward design: reach destination asap"""
                step_cost = 1*(prevd2destin - d2destin) * (1/d2destin)
                collide_reward = -10 + step_cost
                destin_reward = 100*(1/d2destin)
                # destin_reward = 10*(1/d2destin)

                ret = step_cost
                if np.linalg.norm(self.routing[i].DESTINATION-state[0:3]) < reachThreshold_m:
                    ret = destin_reward
                    print(f"\n Agent{i}: ====== Reached Destination!!! ====== reward = {ret}\n")
                elif int(self.CONTACT_FLAGS[i]) == 1:
                    ret = collide_reward
                    # print(f"\n***Collided*** ret = {ret}\n")
            elif reward_choice == 10:
                step_reward = 1*(prevd2destin - d2destin) * (1/d2destin) # If step_reward too high -> agent tends to hover near the destination
                collide_reward = -10 + step_reward
                destin_reward = 100*h2destin

                if np.linalg.norm(self.routing[i].DESTINATION-state[0:3]) < reachThreshold_m:
                    ret = destin_reward
                    print(f"\n Agent{i}: ====== Reached Destination at {elapsed_time_sec} s!!! ====== reward = {ret}\n")

                elif int(self.CONTACT_FLAGS[i]) == 1:
                    ret = collide_reward
                    print(f"***Agent {i} collided at {elapsed_time_sec} s, ret = {ret}")
                else:
                    if self.routing[0].STAT[0] == RouteStatus.LOCAL:
                    # if self.routing[0].COMMANDS[0]._name == RouteCommandFlag.FOLLOW_LOCAL.value:
                        
                        # ret = step_reward/2
                        ret = step_reward/4
                        # ret = 0
                        # ret = -0.1
                        # print(f"ALERT: using local route, ret = {ret}\n")
                    else:
                        ret = step_reward
            elif reward_choice == 11:
                """Same as reward10, but penalize getting too close to other agent"""
                step_reward = 1000*(prevd2destin - d2destin) * (1/d2destin) # If step_reward too high -> agent tends to hover near the destination
                collide_reward = -10 + step_reward
                too_close_reward = -2  + step_reward#-2 
                destin_reward = 100*h2destin

                ret = step_reward
                # if self.routing[0].STAT[0] == RouteStatus.LOCAL:
                # # if self.routing[0].COMMANDS[0]._name == RouteCommandFlag.FOLLOW_LOCAL.value:
                #     ret = step_reward/4
                if np.linalg.norm(self.routing[i].DESTINATION-state[0:3]) < reachThreshold_m:
                    ret = destin_reward
                    print(f"\n Agent{i}: ====== Reached Destination at {elapsed_time_sec} s!!! ====== reward = {ret}\n")

                elif int(self.CONTACT_FLAGS[i]) == 1:
                    ret = collide_reward
                    print(f"***Agent {i} collided at {elapsed_time_sec} s, ret = {ret}")
                
                elif any(self.routing[i].RAYS_INFO[:,1]<0.1):
                    ret = too_close_reward
            elif reward_choice == 20:
                max_distance = h2destin
                if d2destin < reachThreshold_m:
                    ret = 5.0
                elif int(self.CONTACT_FLAGS[i]) == 1:
                    ret = -1.0
                    print(f"***Agent {i} collided at {elapsed_time_sec} s, ret = {ret}")
                else:
                    # Proportional distance penalty
                    distance_penalty = -min(d2destin / max_distance, 1.0)
                    # Reward progress toward the goal
                    distance_delta = prevd2destin - d2destin
                    distance_progress_reward = max(distance_delta * 0.2, 0.0) # Reward only positive progress

                    # Check if the agent is static
                    # if np.allclose(prevd2destin, d2destin, atol=0.05):
                    #     # # Tracks consecutive static actions (used in reward function)
                    #     # self.routing[i].static_action_counter += 1  # (Already coded in _computeDone)
                    #     if self.routing[i].static_action_counter >= self.static_action_threshold:
                    #         # print(f"!+! Agent {i}: stayed static for 20 actions! Terminating episode.")
                    #         ret = distance_progress_reward + distance_penalty - 0.5
                    #     else:
                    #         ret = distance_progress_reward + distance_penalty - 0.1  # Small penalty for being static
                    #         # print(f"Agent {i}: Small penalty applied")
                    # else:
                    #     self.routing[i].static_action_counter = 0
                    #     ret = distance_progress_reward + distance_penalty
                    #     print(f"Agent {i}: progress with reward = {ret}")
                    self.routing[i].static_action_counter = 0
                    ret = distance_progress_reward + distance_penalty
                    # print(f"Agent {i}: progress with reward = {ret}")
                        
            rewards[i] = ret
        # print(f"--> Reward = {rewards}")
        return rewards
    ################################################################################
    
    def _computeDone(self):
        """Computes the current done value(s).

        Returns
        -------
        dict[int | "__all__", bool]
            Dictionary with the done value of each drone and 
            one additional boolean value for key "__all__".
        """
        states = np.array([self._getDroneStateVector(i) for i in range(self.NUM_DRONES)])
        reachThreshold_m = 0.5
        bool_vals = [False for _ in range(self.NUM_DRONES)]

        elapsed_time_sec = self.step_counter/self.SIM_FREQ
        
        # Check for any collisions
        collision_occurred = any(int(self.CONTACT_FLAGS[j]) == 1 for j in range(self.NUM_DRONES))
        
        for j in range(self.NUM_DRONES):
            state = self._getDroneStateVector(j)
            curPos = np.array(state[0:3])
            prevd2destin = np.linalg.norm(self.routing[j].DESTINATION - self.routing[j].CUR_POS)
            d2destin = np.linalg.norm(self.routing[j].DESTINATION - state[0:3])

            if np.linalg.norm(self.routing[j].DESTINATION - states[j, 0:3]) <= reachThreshold_m:
                bool_vals[j] = True
            elif int(self.CONTACT_FLAGS[j]) == 1:
                bool_vals[j] = True
            # Check if the agent is static
            elif np.allclose(prevd2destin, d2destin, atol=0.05):
                # Tracks consecutive static actions (used in reward function)
                self.routing[j].static_action_counter += 1
                if self.routing[j].static_action_counter >= self.static_action_threshold:
                    print(f"!! Agent {j}: stayed static for {self.ALLOWED_WAITING_S} s! Terminating episode.")
                    bool_vals[j] = True
            else:
                bool_vals[j] = False

        # If episode time is up, mark all done
        if self.step_counter / self.SIM_FREQ > self.EPISODE_LEN_SEC:
            done = {i: True for i in range(self.NUM_DRONES)}
            done["__all__"] = True
            print("<< TIME-OUT >>")
            return done

        # If any drone collides, end the entire episode
        elif collision_occurred:
            done = {i: True for i in range(self.NUM_DRONES)}
            done["__all__"] = True
            print(f"<< COLLISION at {elapsed_time_sec} s >>")
            return done
        

        # Otherwise, continue based on individual goals
        done = {i: bool_vals[i] for i in range(self.NUM_DRONES)}
        # done["__all__"] = all(bool_vals)   # Done if all finish
        # done["__all__"] = True if True in done.values() else False   # Done if any finish

        done["__all__"] = all(done.values())  # RLlib needs to know when ALL agents are done!
        # message += " END OF EPISODE <<"
        # print(message)
        # print(f"done = {done}")
        return done


    ################################################################################
    
    def _computeInfo(self):
        """Computes the current info dict(s).

        Unused.

        Returns
        -------
        dict[int, dict[]]
            Dictionary of empty dictionaries.

        """
        return {i: {} for i in range(self.NUM_DRONES)}

    ################################################################################

    def _clipAndNormalizeState(self,
                               state
                               ):
        """Normalizes a drone's state to the [-1,1] range.

        Parameters
        ----------
        state : ndarray
            (20,)-shaped array of floats containing the non-normalized state of a single drone.

        Returns
        -------
        ndarray
            (20,)-shaped array of floats containing the normalized state of a single drone.

        """
        MAX_LIN_VEL_XY = 3 
        MAX_LIN_VEL_Z = 1

        MAX_XY = MAX_LIN_VEL_XY*self.EPISODE_LEN_SEC
        MAX_Z = MAX_LIN_VEL_Z*self.EPISODE_LEN_SEC

        MAX_PITCH_ROLL = np.pi # Full range

        clipped_pos_xy = np.clip(state[0:2], -MAX_XY, MAX_XY)
        clipped_pos_z = np.clip(state[2], 0, MAX_Z)
        clipped_rp = np.clip(state[7:9], -MAX_PITCH_ROLL, MAX_PITCH_ROLL)
        clipped_vel_xy = np.clip(state[10:12], -MAX_LIN_VEL_XY, MAX_LIN_VEL_XY)
        clipped_vel_z = np.clip(state[12], -MAX_LIN_VEL_Z, MAX_LIN_VEL_Z)

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
        normalized_y = state[9] / np.pi # No reason to clip
        normalized_vel_xy = clipped_vel_xy / MAX_LIN_VEL_XY
        normalized_vel_z = clipped_vel_z / MAX_LIN_VEL_XY
        normalized_ang_vel = state[13:16]/np.linalg.norm(state[13:16]) if np.linalg.norm(state[13:16]) != 0 else state[13:16]

        norm_and_clipped = np.hstack([normalized_pos_xy,
                                      normalized_pos_z,
                                      state[3:7],
                                      normalized_rp,
                                      normalized_y,
                                      normalized_vel_xy,
                                      normalized_vel_z,
                                      normalized_ang_vel,
                                      state[16:20]
                                      ]).reshape(20,)

        return norm_and_clipped
    
    def _clipAndNormalizeRay(self,
                            rayinfo
                            ):
        """Normalizes a ray's informaiton to the [-1,1] range.

        Parameters
        ----------
        rayinfo : ndarray
            (5,)-shaped array of floats containing the non-normalized information of a SINGLE ray.

        Returns
        -------
        ndarray
            (5,)-shaped array of floats containing the normalized information of a SINGLE ray

        """
        # Ray info -- [hit_ids, hit_fraction, hit_pos_x, hit_pos_y, hit_pos_z] per ray
        MAX_HIT_IDS = self.NUM_DRONES
        MIN_HIT_IDS = -1

        MAX_LIN_VEL_XY = 3 
        MAX_LIN_VEL_Z = 1

        MAX_XY = MAX_LIN_VEL_XY*self.EPISODE_LEN_SEC
        MAX_Z = MAX_LIN_VEL_Z*self.EPISODE_LEN_SEC

        clipped_hit_ids = np.clip(rayinfo[0], MIN_HIT_IDS, MAX_HIT_IDS)
        clipped_hit_fraction = rayinfo[1]  # no need (already in [0,1] range)
        clipped_hit_pos_xy = np.clip(rayinfo[2:4], -MAX_XY, MAX_XY)
        clipped_hit_pos_z = np.clip(rayinfo[4], 0, MAX_Z)


        normalized_hit_ids = clipped_hit_ids / MAX_HIT_IDS     # [-1, 1]
        normalized_hit_fraction = clipped_hit_fraction # [0, 1]
        normalized_hit_pos_xy = clipped_hit_pos_xy / MAX_XY  #[-1, 1]
        normalized_hit_pos_z = clipped_hit_pos_z / MAX_Z   #[0, 1]

        norm_and_clipped = np.hstack([normalized_hit_ids,
                                      normalized_hit_fraction,
                                      normalized_hit_pos_xy,
                                      normalized_hit_pos_z,
                                      ]).reshape(5,)



        return norm_and_clipped
    
    def _clipAndNormalizeD2Destin(self, d2destin, drone_id):
        h2destin = np.linalg.norm(self.routing[drone_id].DESTINATION - self.routing[drone_id].HOME_POS)
        MIN_D2DESTIN = 0
        MAX_D2DESTIN = h2destin
        clipped_d2destin = np.clip(d2destin, MIN_D2DESTIN, MAX_D2DESTIN)
        normalized_d2destin = clipped_d2destin / MAX_D2DESTIN  #range [0, 1]
        return normalized_d2destin
    
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
            print("[WARNING] it", self.step_counter, "in LeaderFollowerAviary._clipAndNormalizeState(), clipped xy position [{:.2f} {:.2f}]".format(state[0], state[1]))
        if not(clipped_pos_z == np.array(state[2])).all():
            print("[WARNING] it", self.step_counter, "in LeaderFollowerAviary._clipAndNormalizeState(), clipped z position [{:.2f}]".format(state[2]))
        if not(clipped_rp == np.array(state[7:9])).all():
            print("[WARNING] it", self.step_counter, "in LeaderFollowerAviary._clipAndNormalizeState(), clipped roll/pitch [{:.2f} {:.2f}]".format(state[7], state[8]))
        if not(clipped_vel_xy == np.array(state[10:12])).all():
            print("[WARNING] it", self.step_counter, "in LeaderFollowerAviary._clipAndNormalizeState(), clipped xy velocity [{:.2f} {:.2f}]".format(state[10], state[11]))
        if not(clipped_vel_z == np.array(state[12])).all():
            print("[WARNING] it", self.step_counter, "in LeaderFollowerAviary._clipAndNormalizeState(), clipped z velocity [{:.2f}]".format(state[12]))
