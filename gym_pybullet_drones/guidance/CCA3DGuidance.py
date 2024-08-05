import math
import numpy as np
# import pybullet as p

from gym_pybullet_drones.guidance import BaseGuidance
from gym_pybullet_drones.envs.BaseAviary import DroneModel, BaseAviary


class CCA3DGuidance(BaseGuidance):
    """3D Carrot-Chasing path-following class"""
    

    ###########################################################################
    
    def followPath(self, path, state, speed_limit):
        
        # Extract state information
        cur_pos = state[0:3],
        cur_quat = state[3:7],
        cur_angle = state[7:10],  # RPY
        cur_vel = state[10:13],
        cur_ang_vel = state[13:16],
        
        path = path.reshape(3,)
        count = 1
        dtcum = 0
        
        wp_closeness_threshold = speed_limit/100
        
        for j in range(len(path)-1):
            
    
            if dtcum >= 0.01:
                break
            
            Wi = path[:, j].reshape((3,1))
            Wf = path[:, j+1].reshape((3,1))
            

            path_vect = Wf - Wi
            (a, b, c) = path_vect.reshape(3,)
            # Check if the waypoint is ahead of current position
            if a*(cur_pos[0] - Wf[0]) + b*(cur_pos[1] - Wf[1]) + c*(cur_pos[2]- Wf[2]) < 0:
                # Check if the agent is too close to the targetted waypoint
                if np.linalg.norm(cur_pos.reshape(3,1) - Wf.reshape(3,1)) <= wp_closeness_threshold: 
                    continue
                
                target_pos = Wf   # Wi or Wf??
                self._CCA3D(state)
                
                
            else:
                continue
            
            
            
            
    ###########################################################################
    def CCA3D(self, Wi, Wf, state, uav_config):
        # Extract state information
        cur_pos=state[0:3],
        cur_quat=state[3:7],
        cur_angle = state[7:10],  # RPY
        cur_vel=state[10:13],
        cur_ang_vel=state[13:16],
        
        v = np.linalg.norm(cur_vel.reshape(3,1))
        
        dt = 0.01
        timeSpent = 0
        
        # UAV config
        Rmin = 10
        umax = v**2/ Rmin
        
        # CCA3D Tuning
        kappa = 50
        delta = 20
        
        (x, y, z) = cur_pos
        
        
        
        
        
        
        
        
        
        
        return updated_state
        
        
        
        
        
    ###########################################################################   
                
                
                
                
                
                