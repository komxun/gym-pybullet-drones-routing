import math
import numpy as np
# import pybullet as p

from gym_pybullet_drones.guidance.BaseGuidance import BaseGuidance
from gym_pybullet_drones.envs.BaseAviary import DroneModel, BaseAviary


class CCA3DGuidance(BaseGuidance):
    """3D Carrot-Chasing path-following class"""
    

    ###########################################################################
    
    def followPath(self, path, state, target_vel, speed_limit):
        
        # Extract state information
        cur_pos = state[0:3]
        cur_quat = state[3:7]
        cur_angle = state[7:10]  # RPY
        cur_vel = state[10:13]
        cur_ang_vel = state[13:16]
        
        # path = path.reshape(3,)
        # count = 1
        dtcum = 0
        dt = 0.01
        
        wp_closeness_threshold = speed_limit/100
        # wp_closeness_threshold = 1
        # print(wp_closeness_threshold)
        
        pathLen = path.shape[1]
        
        for j in range(pathLen-1):
            
            # if dtcum >= 1:
            #     break
            
            Wi = path[:, j]
            Wf = path[:, j+1]
            
            path_vect = Wf - Wi
            (a, b, c) = path_vect.reshape(3,)
            # Check if the waypoint is ahead of current position
            if a*(cur_pos[0] - Wf[0]) + b*(cur_pos[1] - Wf[1]) + c*(cur_pos[2]- Wf[2]) < 0:
                # Check if the agent is too close to the targetted waypoint
                if np.linalg.norm(cur_pos.reshape(3,1) - Wf.reshape(3,1)) <= wp_closeness_threshold: 
                    continue
                
                print("selected wp#", j)
                state2follow = self._CCA3D(Wi, Wf, state, target_vel, dt)
                cur_pos = state2follow[0:3]
                dtcum += dt
                break
            else:
                continue
        
        return state2follow
            
    ###########################################################################
    def _CCA3D(self, Wi, Wf, state, target_vel, dt):
        
        # Extract state information
        cur_pos=state[0:3],
        cur_quat=state[3:7],
        cur_angle = state[7:10],  # RPY
        cur_vel=state[10:13],
        cur_ang_vel=state[13:16],
        
        # v = np.linalg.norm(cur_vel)
        # v = np.linalg.norm(target_vel)
        v =5
        
        cur_pos = cur_pos[0]
        cur_angle = cur_angle[0]
        
        dt = 0.01
        
        # UAV config
        Rmin = 1
        umax = v**2/ Rmin
        
        # CCA3D Tuning
        kappa = 100
        delta = 0.5
    
        (x, y, z) = cur_pos
        (phi, gamma, psi) = cur_angle
         
        # Step 1: Ru = Distance between the initial waypoint to the final waypoint
        Ru_vect = Wi - cur_pos
        Ru = np.linalg.norm(Ru_vect)
        Rw_vect = Wf - Wi
        Rw = np.linalg.norm(Rw_vect)
        # Step 2: theta = Orientation of vector from initial waypoint to the final waypoint
        theta1 = math.atan2(Wf[1] - Wi[1], Wf[0] - Wi[0])
        theta2 = math.atan2(Wf[2] - Wi[2], math.sqrt((Wf[0] - Wi[0])**2 + (Wf[1] - Wi[1])**2))
        # Step 3: theta_u = Orientation of vector from initial waypoint to current UAV position
        theta_u1 = math.atan2(y - Wi[1], x - Wi[0])
        theta_u2 = math.atan2(z - Wi[2], math.sqrt( (x - Wi[0])**2 + (y - Wi[1])**2))

        # Step 4: R = Distance between initial waypoint and q
        if Ru != 0 and Rw != 0:
            alpha = np.real(round(np.dot(Ru_vect, Rw_vect)/ (Ru*Rw), 4))
        else:
            alpha = 0

        R = math.sqrt( Ru**2 - (Ru*math.sin(alpha))**2 )

        # Step 5: Carrot position, s = (xt, yt, zt)
        xt = Wi[0] + (R + delta) * math.cos(theta2) * math.cos(theta1)
        yt = Wi[1] + (R + delta) * math.cos(theta2) * math.sin(theta1)
        zt = Wi[2] + (R + delta) * math.sin(theta2)
        # Step 6: Desired heading angle and pitch angle, psi_d gamma_d
        psi_d = math.atan2(yt - y, xt - x)
        gamma_d = math.atan2(zt - z, math.sqrt( (xt - x)**2 + (yt - y)**2))
        
        # Wrapping up angles
        # psi_d = math.remainder(psi_d, 2*math.pi)
        # gamma_d = math.remainder(gamma_d, 2*math.pi)
        
        # Step 7: Guidance Yaw Command, u1
        del_psi = psi_d - psi
        del_gam = gamma_d - gamma
        
        u1 = (kappa*del_psi) * v
        u2 = (kappa*del_gam) * v
        
        # Limiting guidance acceleration command
        # if u1 > umax:
        #     u1 = umax
        # elif u1 < -umax:
        #     u1 = -umax
        
        # if u2 > umax:
        #     u2 = umax
        # elif u2 < -umax:
        #     u2 = -umax
        
        # UAV Dynamics
        dx = v * math.cos(gamma) * math.cos(psi)
        dy = v * math.cos(gamma) * math.sin(psi)
        dz = v * math.sin(gamma)
        dpsi = u1 / (v * math.cos(gamma))
        dgam = u2 / v
        
        # Update UAV targetted state
        x_tg = x + dx * dt
        y_tg = y + dy * dt
        z_tg = z + dz * dt
        psi_tg = psi + dpsi * dt
        gamma_tg = gamma + dgam * dt
        
        # no roll command
        state2follow = np.array([x_tg, y_tg, z_tg, 0, gamma_tg, psi_tg])
        
        return state2follow
        
        
        
        
        
    ###########################################################################   
                
                
                
                
                
                