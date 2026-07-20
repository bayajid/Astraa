import numpy as np

class we_interpolating_pos:
    """_summary_
    """    
    def __init__(self) -> None:
        self.coef_x = None
        self.coef_y = None
        self.coef_z = None
        self.t0 = None
    def get_lin_interpolant(self, t1, dr1, t2, dr2):
        t_0 = 0
        t_1 = t2 - t1
        A = np.array([[1, t_0],
                      [1, t_1]])
        A_inv = np.linalg.inv(A)
        coeff_x = A_inv @ np.array([dr1[0], 
                                    dr2[0]]).transpose()
        coeff_y = A_inv @ np.array([dr1[1], 
                                    dr2[1]]).transpose()
        coeff_z = A_inv @ np.array([dr1[2], 
                                    dr2[2]]).transpose()
        self.coef_x = coeff_x
        self.coef_y = coeff_y
        self.coef_z = coeff_z
        self.t0 = t1
    def get_const_interpolant(self, t1, dr1):
        self.t0 = t1
        self.coef_x = [dr1[0]]
        self.coef_y = [dr1[1]]
        self.coef_z = [dr1[2]]
    def get_quad_interpolant(self, t1,t2, dr1, dr2, v2):
        self.t0 = t1
        self.t1 = t2
        
        dt1 = 0
        dt2 = t2-t1
        A = np.array([[1, dt1, dt1**2], [1, dt2, dt2**2], [0, 1, 2*dt2]])
        A_inv = np.linalg.inv(A)
        coeff_x = A_inv @ np.array([dr1[0], 
                                    dr2[0],
                                    v2[0]]).transpose()
        coeff_y = A_inv @ np.array([dr1[1], 
                                    dr2[1],
                                    v2[1]]).transpose()
        coeff_z = A_inv @ np.array([dr1[2], 
                                    dr2[2],
                                    v2[2]]).transpose()
        self.coef_x = coeff_x
        self.coef_y = coeff_y
        self.coef_z = coeff_z
    def interpolate(self, t_req):
        """interpolate after setting interpolants

        Args:
            t_req (float): time array [s]

        Returns:
            dr: interpolated position/correction
        """        
        if type(t_req) != float:
            t_req = t_req.reshape((t_req.shape[0],1))
        dt = t_req - self.t0
        dx = 0
        dy = 0
        dz = 0
        for ii, c_x in enumerate(self.coef_x):
            dx += dt**ii * c_x
        for ii, c_y in enumerate(self.coef_y):
            dy += dt**ii * c_y
        for ii, c_z in enumerate(self.coef_z):
            dz += dt**ii * c_z
        
        dr = np.hstack([dx, dy, dz])
        return dr

class we_interpolating:
    """_summary_
    """    
    def __init__(self) -> None:
        self.coeff = None # coefficients for all vals        
        self.t0 = None
    # def get_lin_interpolant(self, t1, dr1, t2, dr2):
    #     t_0 = 0
    #     t_1 = t2 - t1
    #     A = np.array([[1, t_0],
    #                   [1, t_1]])
    #     A_inv = np.linalg.inv(A)
    #     coeff_x = A_inv @ np.array([dr1[0], 
    #                                 dr2[0]]).transpose()
    #     coeff_y = A_inv @ np.array([dr1[1], 
    #                                 dr2[1]]).transpose()
    #     coeff_z = A_inv @ np.array([dr1[2], 
    #                                 dr2[2]]).transpose()
    #     self.coef_x = coeff_x
    #     self.coef_y = coeff_y
    #     self.coef_z = coeff_z
    #     self.t0 = t1
        
    def fix_quat_sign_scalar(self, q_prev, q):
        # q_prev, q, dq are 4-element lists/tuples
        dot = q_prev[0]*q[0] + q_prev[1]*q[1] + q_prev[2]*q[2] + q_prev[3]*q[3]
        if dot < 0.0:
            q = [-q[0], -q[1], -q[2], -q[3], -q[4], -q[5], -q[6], -q[7]]
        return q
    
    def get_lin_interpolant(self, t_both, r_both):
        t1 = t_both[0]
        t2 = t_both[1]

        self.t0 = t1
        self.t1 = t2
        
        dt1 = 0
        dt2 = t2-t1
        A = np.array([[1, dt1], 
                    [1, dt2]])
        A_inv = np.linalg.inv(A)

        interp_coeff = A_inv @ np.vstack((r_both))
        self.coeff = interp_coeff
    def get_const_interpolant(self, t1, dr1):
        self.t0 = t1
        self.coef_x = [dr1[0]]
        self.coef_y = [dr1[1]]
        self.coef_z = [dr1[2]]
    def get_quad_interpolant(self, t_both, r_both, v_both):
        t1 = t_both[0]
        t2 = t_both[1]

        self.t0 = t1
        self.t1 = t2
        
        dt1 = 0
        dt2 = t2-t1
        A = np.array([[1, dt1, dt1**2], 
                      [1, dt2, dt2**2], 
                      [0, 1, 2*dt2]])
        A_inv = np.linalg.inv(A)

        interp_coeff = A_inv @ np.vstack((r_both, v_both[1,:]))
        self.coeff = interp_coeff
    def interpolate(self, t_req):
        """interpolate after setting interpolants

        Args:
            t_req (float): time array [s]

        Returns:
            dr: interpolated position/correction
        """        
        if type(t_req) != float:
            t_req = t_req.reshape((t_req.shape[0],1))
        dt = t_req - self.t0
        r_interp = np.zeros((1,3)).flatten()
        for row_nr, co in enumerate(self.coeff[:,0]):
            r_interp = r_interp + self.coeff[row_nr,:] *dt**row_nr
        return r_interp
    def interpolate_flexible(self, t_req):
        """interpolate after setting interpolants

        Args:
            t_req (float): time array [s]

        Returns:
            interp: interpolated outputs, same shape as input
        """        
        # if type(t_req) != float:
        #     t_req = t_req.reshape((t_req.shape[0],1))
        dt = t_req - self.t0
        r_interp = np.zeros((1,self.coeff.shape[1])).flatten()
        for row_nr, co in enumerate(self.coeff[:,0]):
            r_interp = r_interp + self.coeff[row_nr,:] *dt**row_nr
        return r_interp
if __name__ == '__main__':
    interp_class = we_interpolating()
    t1 = 1388145558.816
    t2 = 1388145569.816
    s1 = np.array([ 7.33535035e+06,  2.92469705e+03,  1.67555740e+05, -1.68525719e+02,
        1.28607220e+02,  7.36789757e+03])
    s2 = np.array([ 7.33329457e+06,  4.21059985e+03,  2.41224987e+05, -2.42627669e+02,
        1.28571174e+02,  7.36582746e+03])
    interp_class.get_quad_interpolant([t1, t2], 
                                      r_both = np.vstack((s1[:3], s2[:3])), 
                                      v_both = np.vstack((s1[3:], s2[3:])))
    
    
    r_interp = interp_class.interpolate(t2 + 1)