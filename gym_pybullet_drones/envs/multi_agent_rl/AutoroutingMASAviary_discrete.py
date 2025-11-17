import numpy as np

from gym_pybullet_drones.utils.enums import DroneModel, Physics
from collections import deque
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ActionType, ObservationType
from gym_pybullet_drones.envs.multi_agent_rl.ExtendedMultiagentAviary_discrete import ExtendedMultiagentAviary_discrete
# from gym_pybullet_drones.routing.BaseRouting import RouteCommandFlag, RouteStatus

# class SimpleStatusPrinter:
#     def __init__(self, print_interval=200):  # Periodic updates
#         self.print_interval = print_interval
#         self.last_step = -1

#     def print_status(self, agent_statuses, step=None):
#         # agent_statuses is now a dict: {agent_id: status}
#         has_events = any(status != "Nominal" for status in agent_statuses.values())

#         should_print = (
#             has_events or 
#             (step is not None and step % self.print_interval == 0)
#         )

#         if should_print:
#             print(f"\n{'=' * 50}")
#             if step is not None:
#                 print(f"Step {step}:")

#             if has_events:
#                 # Show only agents with non-nominal statuses
#                 for agent_id, status in agent_statuses.items():
#                     if status != "Nominal":
#                         emoji = {
#                             "NMAC": "⚠️ ",
#                             "Collided": "💥",
#                             "Reached destination": "🎯",
#                             "Done": "✅"
#                         }.get(status, "❓")
#                         print(f"{emoji} Agent {agent_id}: {status}")
#             else:
#                 # Periodic update - all are nominal
#                 print(f"✅ All {len(agent_statuses)} agents: Nominal")

#             print("=" * 50)

#         self.last_step = step

            
class AutoroutingMASAviary_discrete(ExtendedMultiagentAviary_discrete):
    """Multi-agent RL problem: leader-follower."""

    ################################################################################

    def __init__(self,
                 drone_model: DroneModel=DroneModel.HB,
                 num_drones: int=2,
                 neighbourhood_radius: float=np.inf,
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
        self.ALLOWED_WAITING_S = 60
        self.static_action_threshold = freq * self.ALLOWED_WAITING_S
        
        

        super().__init__(drone_model=drone_model,
                         num_drones=num_drones,
                         neighbourhood_radius=neighbourhood_radius,
                         physics=physics,
                         freq=freq,
                         aggregate_phy_steps=aggregate_phy_steps,
                         gui=gui,
                         record=record, 
                         obs=obs,
                         act=act
                         )

    ################################################################################
    def _computeReward(self):
        """Computes a blended (individual + team) reward for each drone.
        Returns
        -------
        dict[int, float]
        """

        # ======== Reward Parameters ========
        reachThreshold_m = 3
        reward_collision = -2
        reward_reached = 2
        reward_timeout = -1
        team_ratio = 0.2   # <-- 0 = fully individual, 1 = fully team-based
        epsilon = 1e-3

        min_detect_fraction = self.routing[0].ROV / self.routing[0].RAY_LEN_M
        indiv_rewards = {i: 0.0 for i in range(self.NUM_DRONES)}

        # ======== Compute individual rewards ========
        for agent_id in range(self.NUM_DRONES):
            
            if self.DONE[agent_id]:
                indiv_rewards[agent_id] = 0.0
                continue

            state = self._getDroneStateVector(agent_id)
            d2destin = np.linalg.norm(self.routing[agent_id].DESTINATION - state[0:3])
            h2destin = np.linalg.norm(self.routing[agent_id].DESTINATION - self.routing[agent_id].HOME_POS)
            rmin_values = self.routing[agent_id].SECTOR_INFO[0::3]

            reward = 0.0

            # --- Collision ---
            if int(self.CONTACT_FLAGS[agent_id]) == 1:
                reward += reward_collision
                self.DONE[agent_id] = True

            # --- Timeout ---
            elif self.step_counter / self.SIM_FREQ > self.EPISODE_LEN_SEC:
                reward += reward_timeout
                self.DONE[agent_id] = True

            # --- Near obstacle ---
            elif np.any((rmin_values < min_detect_fraction) & (rmin_values > 0)):
                valid_rays = rmin_values[np.isfinite(rmin_values) & (rmin_values > 0)]
                if valid_rays.size > 0:
                    min_detected = np.min(valid_rays)
                    # print(f"min_detected is {min_detected}")
                    reward += -1 / (min_detected + epsilon)

            # --- Reached destination ---
            elif d2destin < reachThreshold_m:
                reward += reward_reached
                self.DONE[agent_id] = True

            # --- Progress toward goal ---
            else:
                # reward += 1 / (d2destin + epsilon)
                reward += ((h2destin - d2destin)/h2destin)**2

            indiv_rewards[agent_id] = reward

        # ======== Compute team reward (average) ========
        active_agents = [i for i in range(self.NUM_DRONES) if not self.DONE[i]]
        if active_agents:
            team_reward = np.mean([indiv_rewards[i] for i in active_agents])
        else:
            team_reward = 0.0

        # ======== Blend individual + team rewards ========
        final_rewards = {}
        for agent_id in range(self.NUM_DRONES):
            final_rewards[agent_id] = (
                (1 - team_ratio) * indiv_rewards[agent_id] + team_ratio * team_reward
            )

        return final_rewards


    # def _computeReward(self):
    #     """Computes the current reward value.
    #     Returns
    #     -------
    #     dict[int, float]"""
        
    #     # Initialize the printer if it doesn't exist
    #     # if not hasattr(self, 'status_printer'):
    #     #     self.status_printer = SimpleStatusPrinter()
        
    #     # elapsed_time_sec = self.step_counter/self.SIM_FREQ
    #     reachThreshold_m = 3
    #     #======= Reward Design =========
    #     reward_collision = -2
    #     reward_reached = 2
    #     common_reward = 0
    #     rewards = {}

        
        
    #     min_detect_fraction = self.routing[0].ROV / self.routing[0].RAY_LEN_M  # ROV = in the range between [0, 1]
    #                                                                            # 0 = next to the drone, 1 = max range (RAY_LEN_M), 
        
    #     # Track status for each agent
    #     # agent_statuses = {}

    #     for agent_id in [i for i in range(self.NUM_DRONES) if not self.DONE[i]]: # Only agent not done
    #         # if self.DONE[agent_id]:  # Skip agents that are already done
    #         #     rewards[agent_id] = 0.0
    #         #     agent_statuses.append("Done")
    #         #     continue
    #         state = self._getDroneStateVector(agent_id)
    #         d2destin = np.linalg.norm(self.routing[agent_id].DESTINATION - state[0:3])
    #         h2destin = np.linalg.norm(self.routing[agent_id].DESTINATION - self.routing[agent_id].HOME_POS)
    #         # Extract shortest detection  (rmin values) (every 3rd value starting at index 0)
    #         rmin_values = self.routing[agent_id].SECTOR_INFO[0::3]

    #         if int(self.CONTACT_FLAGS[agent_id]) == 1:
    #             common_reward += reward_collision
    #             # agent_statuses[agent_id] = "Collided"
    #             self.DONE[agent_id] = True

    #         elif self.step_counter / self.SIM_FREQ > self.EPISODE_LEN_SEC:
    #             common_reward += reward_collision
    #             # agent_statuses[agent_id] = "Time-out"
    #             self.DONE[agent_id] = True

    #         # elif any(self.routing[agent_id].RAYS_INFO[:,1]<min_detect_fraction):
    #         #     common_reward += reward_nmac
    #         #     # agent_statuses[agent_id] = "NMAC"

    #         elif np.any((rmin_values < min_detect_fraction) & (rmin_values > 0)):
    #             # common_reward += -1 # smaller = high penalty
    #             # common_reward += -1/np.min(rmin_values)
    #             common_reward += (np.min(rmin_values) - min_detect_fraction)/min_detect_fraction
    #             # agent_statuses[agent_id] = "NMAC"

    #         elif d2destin < reachThreshold_m:
    #             common_reward += reward_reached
    #             # agent_statuses[agent_id] = "Reached destination"
    #             self.DONE[agent_id] = True

    #         else:
    #             # common_reward += 1/d2destin # range [0, 1] # 0.1 too small, agent won't move
    #             # common_reward += d2destin/h2destin
    #             common_reward += ((h2destin - d2destin)/h2destin)**2
    #             # agent_statuses[agent_id] = "Nominal"

    #         # print(f"Agent{agent_id}: d2destin/h2destin = {d2destin/h2destin}")

    #     # Print status update (replaces previous print)
    #     # self.status_printer.print_status(agent_statuses, self.step_counter)
                
    #     rewards = {i: common_reward/self.NUM_DRONES for i in range(self.NUM_DRONES) if not self.DONE[i]}
    #     # print(f"rewards is {rewards}")
    #     return rewards

    def _computeDone(self):
        """Computes the current done value(s).

        Returns
        -------
        dict[int | "__all__", bool]
            Dictionary with the done value of each drone and 
            one additional boolean value for key "__all__".
        """
        # done = {i: False for i in range(self.NUM_DRONES)}
        done = {i: False for i in range(self.NUM_DRONES) if not self.DONE[i]}
        # states = np.array([self._getDroneStateVector(i) for i in range(self.NUM_DRONES)])
        reachThreshold_m = 1
        bool_vals = [False for _ in range(self.NUM_DRONES)]

        # elapsed_time_sec = self.step_counter/self.SIM_FREQ
        
        # Check for any collisions
        collision_occurred = any(int(self.CONTACT_FLAGS[j]) == 1 for j in range(self.NUM_DRONES))
        
        for j in range(self.NUM_DRONES):
            state = self._getDroneStateVector(j)
            # curPos = np.array(state[0:3])
            # prevd2destin = np.linalg.norm(self.routing[j].DESTINATION - self.routing[j].CUR_POS)
            d2destin = np.linalg.norm(self.routing[j].DESTINATION - state[0:3])

            if d2destin <= reachThreshold_m:
                bool_vals[j] = True
            elif int(self.CONTACT_FLAGS[j]) == 1:
                bool_vals[j] = True
            else:
                bool_vals[j] = False

        # If episode time is up, mark all done
        if self.step_counter / self.SIM_FREQ > self.EPISODE_LEN_SEC:
            # done = {i: True for i in range(self.NUM_DRONES)}
            # done = {0: True}
            done["__all__"] = True
            # print("<< TIME-OUT >>")
            return done

        # If any drone collides, end the entire episode
        elif collision_occurred:
            # done = {i: True for i in range(self.NUM_DRONES)}
            # done = {0: True}
            done["__all__"] = True
            # print(f"<< COLLISION >>")
            return done
        

        # Otherwise, continue based on individual goals
        # done = {i: bool_vals[i] for i in range(self.NUM_DRONES)}
        done = {i: bool_vals[i] for i in range(self.NUM_DRONES) if not self.DONE[i]}
        # done["__all__"] = all(bool_vals)   # Done if all finish
        done["__all__"] = all(self.DONE)
        # done["__all__"] = True if True in done.values() else False   # Done if any finish

        # message += " END OF EPISODE <<"
        # print(message)
        # print(f"done = {done}")
        # print(f"self.DONE = {self.DONE}")
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

        # if self.GUI:
        #     self._clipAndNormalizeStateWarning(state,
        #                                        clipped_pos_xy,
        #                                        clipped_pos_z,
        #                                        clipped_rp,
        #                                        clipped_vel_xy,
        #                                        clipped_vel_z
        #                                        )

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
        num_info_extract = len(rayinfo)   # 3 or 5 [hit_ids, hit_fraction, hit_pos_x, hit_pos_y, hit_pos_z] per ray
        MAX_HIT_IDS = self.NUM_DRONES
        MIN_HIT_IDS = -1

        MAX_LIN_VEL_XY = 3 
        MAX_LIN_VEL_Z = 1

        MAX_XY = MAX_LIN_VEL_XY*self.EPISODE_LEN_SEC
        MAX_Z = MAX_LIN_VEL_Z*self.EPISODE_LEN_SEC

        if num_info_extract == 5:
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
        elif num_info_extract == 3:
            clipped_hit_pos_xy = np.clip(rayinfo[0:2], -MAX_XY, MAX_XY)
            clipped_hit_pos_z = np.clip(rayinfo[2], 0, MAX_Z)

            normalized_hit_pos_xy = clipped_hit_pos_xy / MAX_XY  #[-1, 1]
            normalized_hit_pos_z = clipped_hit_pos_z / MAX_Z   #[0, 1]

            norm_and_clipped = np.hstack([
                                      normalized_hit_pos_xy,
                                      normalized_hit_pos_z,
                                      ]).reshape(3,)

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
