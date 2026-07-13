from safe_control.robots.kinematic_bicycle2D import KinematicBicycle2D
import numpy as np
import casadi as ca

"""
It is based on the kinematic bicycle 2D model and overrides
only the continous and discrete-time CBF funcitions for Dynamic Parabolic CBF (DPCBF) counterparts:
"""

class KinematicBicycle2D_DPCBF(KinematicBicycle2D):
    def __init__(self, dt, robot_spec, k_lambda=0.1, k_mu=0.5):
        super().__init__(dt, robot_spec)
        self.robot_spec.setdefault('k_lambda', k_lambda)
        self.robot_spec.setdefault('k_mu', k_mu)
        self.robot_spec.setdefault('safety_scale', 1.05)
        self.robot_spec.setdefault('eps_d', 0.1)    # m
        self.robot_spec.setdefault('eps_v', 0.05)   # m/s

        self.k_lambda = float(self.robot_spec['k_lambda'])
        self.k_mu = float(self.robot_spec['k_mu'])
        self.safety_scale = float(self.robot_spec['safety_scale'])
        self.eps_d = float(self.robot_spec['eps_d'])
        self.eps_v = float(self.robot_spec['eps_v'])

    def agent_barrier(self, X, obs, robot_radius, s=None):
        """
        '''Continuous Time DPCBF'''
        Compute a Dynamic Parabolic Control Barrier Function for the Kinematic Bicycle2D.
        The barrier's relative degree is "1"
            h_dot = ∂h/∂x ⋅ f(x) + ∂h/∂x ⋅ g(x) ⋅ u + ∂h/∂p_obs ⋅ v_obs
        Define h from the collision cone idea:
            p_rel = [obs_x - x, obs_y - y]
            v_rel = [obs_x_dot-v_cos(theta), obs_y_dot-v_sin(theta)]
            dist = ||p_rel||
            R = robot_radius + obs_r
        """
        s = self.safety_scale if s is None else s

        theta = X[2, 0]
        v = X[3, 0]

        # Check if obstacles have velocity components (static or moving)
        if obs.shape[0] > 3:
            obs_vel_x = obs[3]
            obs_vel_y = obs[4]

        else:
            obs_vel_x = 0.0
            obs_vel_y = 0.0

        # Combine safety radius (= r_obs + r_rob) with a safe margin (s)
        ego_dim = (obs[2] + robot_radius) * s

        # Compute relative position and velocity
        p_rel = np.array([[obs[0] - X[0, 0]], 
                        [obs[1] - X[1, 0]]])
        v_rel = np.array([[obs_vel_x - v * np.cos(theta)], 
                        [obs_vel_y - v * np.sin(theta)]])
        # Compute norms
        p_rel_mag = np.linalg.norm(p_rel)
        v_rel_mag = np.linalg.norm(v_rel)

        p_rel_x = p_rel[0, 0]
        p_rel_y = p_rel[1, 0]

        # Rotation angle and transformation
        rot_angle = np.arctan2(p_rel_y, p_rel_x)
        R = np.array([[np.cos(rot_angle), np.sin(rot_angle)],
                    [-np.sin(rot_angle),  np.cos(rot_angle)]])

        # Transform v_rel into the new coordinate frame
        v_rel_new = R @ v_rel
        v_rel_new_x = v_rel_new[0, 0]
        v_rel_new_y = v_rel_new[1, 0]

        # Smooth clearance and relative speed
        sqrt_d = np.sqrt(p_rel_mag**2 - ego_dim**2 + self.eps_d**2)
        d_safe = sqrt_d - self.eps_d
        v_eps = np.sqrt(v_rel_mag**2 + self.eps_v**2)

        # DPCBF functions
        adaptive_scale = np.sqrt(s**2 - 1) / ego_dim  # using adaptive parameter
        func_lamda = self.k_lambda * d_safe / v_eps * adaptive_scale
        func_mu = self.k_mu * d_safe * adaptive_scale

        # Barrier function h(x)
        h = v_rel_new_x + func_lamda * (v_rel_new_y**2) + func_mu

        # Compute dh_dx for DPCBF
        dh_dx = np.zeros((1, 4))
        dh_dx[0, 0] = p_rel_y * v_rel_new_y / p_rel_mag**2 + adaptive_scale * (- self.k_lambda * p_rel_x * v_rel_new_y**2 / v_eps / sqrt_d - 2 * self.k_lambda * d_safe / v_eps * v_rel_new_y * p_rel_y / p_rel_mag**2 * v_rel_new_x - self.k_mu * p_rel_x / sqrt_d)
        dh_dx[0, 1] = - p_rel_x * v_rel_new_y / p_rel_mag**2 + adaptive_scale * (- self.k_lambda * p_rel_y * v_rel_new_y**2 / v_eps / sqrt_d + 2 * self.k_lambda * d_safe / v_eps * v_rel_new_y * p_rel_x / p_rel_mag**2 * v_rel_new_x - self.k_mu * p_rel_y / sqrt_d)
        dh_dx[0, 2] = - v * np.sin(rot_angle-theta) + adaptive_scale * (- self.k_lambda * d_safe * v * (obs_vel_x * np.sin(theta) - obs_vel_y * np.cos(theta)) * v_rel_new_y**2 / v_eps**3 - 2 * self.k_lambda * d_safe * v_rel_new_y * v * np.cos(rot_angle-theta) / v_eps)
        dh_dx[0, 3] = - np.cos(rot_angle-theta) + adaptive_scale * (- self.k_lambda * d_safe / v_eps**3 * (v - obs_vel_x * np.cos(theta) - obs_vel_y * np.sin(theta)) * v_rel_new_y**2 + 2 * self.k_lambda * d_safe * v_rel_new_y * np.sin(rot_angle-theta) / v_eps)

        return h, dh_dx

    def agent_barrier_dt(self, x_k, u_k, obs, robot_radius, s=None):
        '''Discrete Time DPCBF'''
        s = self.safety_scale if s is None else s

        # Dynamics equations for the next states
        x_k1 = self.step(x_k, u_k, casadi=True)

        def h(x, obs, robot_radius, s):
            theta = x[2, 0]
            v = x[3, 0]

            # Check if obstacles have velocity components (static or moving)
            if obs.shape[0] > 3:
                obs_vel_x = obs[3]
                obs_vel_y = obs[4]
            else:
                obs_vel_x = 0.0
                obs_vel_y = 0.0

            # Combine radius R
            ego_dim = (obs[2] + robot_radius) * s

            # Compute relative position and velocity
            p_rel = ca.vertcat(obs[0] - x[0, 0], obs[1] - x[1, 0])
            v_rel = ca.vertcat(obs_vel_x - v * ca.cos(theta), obs_vel_y - v * ca.sin(theta))

            # Compute the rotation angle
            rot_angle = ca.atan2(p_rel[1], p_rel[0])

            # Rotation matrix for transforming to the new coordinate frame:
            R = ca.vertcat( 
                ca.horzcat(ca.cos(rot_angle), ca.sin(rot_angle)),
                ca.horzcat(-ca.sin(rot_angle), ca.cos(rot_angle))
            )

            # Transform v_rel into the new coordinate frame
            v_rel_new = ca.mtimes(R, v_rel)

            p_rel_mag = ca.norm_2(p_rel)
            v_rel_mag = ca.norm_2(v_rel)

            d_safe = ca.sqrt(p_rel_mag**2 - ego_dim**2 + self.eps_d**2) - self.eps_d
            v_eps = ca.sqrt(v_rel_mag**2 + self.eps_v**2)

            adaptive_scale = np.sqrt(s**2 - 1) / ego_dim
            lamda = self.k_lambda * d_safe / v_eps * adaptive_scale
            mu = self.k_mu * d_safe * adaptive_scale

            # Compute h
            h = v_rel_new[0] + lamda * v_rel_new[1]**2 + mu

            return h

        if obs.shape[0] > 3:
            obs_k1 = ca.vertcat(obs[0] + obs[3] * self.dt,
                                obs[1] + obs[4] * self.dt,
                                obs[2], obs[3], obs[4])
        else:
            obs_k1 = obs

        h_k1 = h(x_k1, obs_k1, robot_radius, s)
        h_k = h(x_k, obs, robot_radius, s)

        d_h = h_k1 - h_k

        return h_k, d_h
