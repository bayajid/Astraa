import sys
import numpy as np
from scipy.spatial.transform import Rotation as R

class SatelliteAOCS:
    def __init__(self):
        # 1. Satellite Properties (Step 2 & 4)
        self.mass = 150.0
        self.J = np.diag([15.0, 15.0, 20.0])  # kg*m^2
        self.J_inv = np.linalg.inv(self.J)
        
        # Actuator Limits
        self.max_torque = 0.05  # Nm (Typical for small-sat reaction wheels)
        self.max_momentum = 2.0  # Nms
        
        # 2. State Vector (Step 1)
        # quaternion in scipy style ordering used elsewhere: [x,y,z,w]
        self.q = np.array([0.0, 0.0, 0.0, 1.0])
        self.omega = np.array([0.0, 0.0, 0.0])
        self.bias = np.array([0.001, -0.001, 0.002]) # Gyro bias
        
        # 3. EKF Covariance
        self.P = np.eye(6) * 0.1
        self.Q_proc = np.eye(6) * 1e-6 # Process noise
        self.R_meas = np.eye(3) * (0.01 * np.pi/180)**2 # Star tracker noise
        
    def get_skew_symmetric(self, v):
        return np.array([[0, -v[2], v[1]],
                         [v[2], 0, -v[0]],
                         [-v[1], v[0], 0]])

    def dynamics(self, torque_ext, dt):
        """Rigid body dynamics (Step 2)"""
        # Euler's rotational equation: J*w_dot + w x (J*w) = Torque
        dw = self.J_inv @ (torque_ext - np.cross(self.omega, self.J @ self.omega))
        self.omega += dw * dt
        
        # Quaternion propagation (using quaternion kinematics)
        # convert omega (rad/s) -> quaternion derivative with ordering [x,y,z,w]
        omega_quat = np.array([self.omega[0], self.omega[1], self.omega[2], 0.0])
        dq = 0.5 * self.quat_multiply(self.q, omega_quat)
        self.q += dq * dt
        self.q /= np.linalg.norm(self.q)

    def quat_multiply(self, q, r):
        # q and r as [x,y,z,w]
        w1, x1, y1, z1 = q[3], q[0], q[1], q[2]
        w2, x2, y2, z2 = r[3], r[0], r[1], r[2]
        return np.array([
            x1*w2 + y1*z2 - z1*y2 + w1*x2,
            -x1*z2 + y1*w2 + z1*x2 + w1*y2,
            x1*y2 - y1*x2 + z1*w2 + w1*z2,
            -x1*x2 - y1*y2 - z1*z2 + w1*w2
        ])

    def control_law(self, q_dest, dt):
        """Step 4: Quaternion Feedback + Rate Damping"""
        # Error quaternion: q_err = q_curr^-1 * q_dest
        q_inv = np.array([-self.q[0], -self.q[1], -self.q[2], self.q[3]])
        q_err = self.quat_multiply(q_inv, q_dest)
        
        # Gains tuned for 150kg inertia
        Kp = 0.8 
        Kd = 1.2
        
        # PD Torque: Proportional to vector part of error quat
        torque = 2 * Kp * q_err[:3] * np.sign(q_err[3]) - Kd * self.omega
        return np.clip(torque, -self.max_torque, self.max_torque)

    def mekf_step(self, gyro_meas, star_tracker_meas, dt):
        """Step 3: Multiplicative EKF (simplified)"""
        unbiased_omega = gyro_meas - self.bias
        
        # Update Covariance (simplified)
        F = np.eye(6)
        self.P = F @ self.P @ F.T + self.Q_proc
        
        if star_tracker_meas is not None:
            H = np.hstack([np.eye(3), np.zeros((3,3))])
            K = self.P @ H.T @ np.linalg.inv(H @ self.P @ H.T + self.R_meas)
            self.P = (np.eye(6) - K @ H) @ self.P


class AdvancedSatellite:
    def __init__(self):
        # Sat Specs (150kg)
        self.J = np.diag([15.0, 15.0, 20.0])
        self.J_inv = np.linalg.inv(self.J)
        
        # Requirements
        self.target_accuracy = 0.05 # Degrees
        self.max_torque = 0.05      # Nm
        
        # State: [q0, q1, q2, q3, bx, by, bz]
        self.q = np.array([0., 0., 0., 1.])
        self.omega = np.zeros(3)
        self.bias = np.array([0.0001, -0.0001, 0.00005]) # Real physical bias
        
        # EKF State
        self.est_q = np.array([0., 0., 0., 1.])
        self.est_bias = np.zeros(3)
        self.P = np.eye(6) * 0.01 # Covariance
        
    def get_mesurement(self, eclipse):
        # Gyro: True rate + bias + noise
        gyro_noise = np.random.normal(0, 0.0002, 3)
        meas_omega = self.omega + self.bias + gyro_noise
        
        # Star Tracker: True orientation + noise (if not in eclipse)
        if not eclipse:
            st_noise = R.from_rotvec(np.random.normal(0, np.radians(0.005), 3)).as_quat()
            meas_q = (R.from_quat(self.q) * R.from_quat(st_noise)).as_quat()
            return meas_omega, meas_q
        return meas_omega, None

    def mekf_predict(self, gyro_meas, dt):
        # 1. Unbias the rate
        unbiased_w = gyro_meas - self.est_bias
        
        # 2. Propagate Quaternion (Multiplicative)
        dq = R.from_rotvec(unbiased_w * dt).as_quat()
        self.est_q = (R.from_quat(self.est_q) * R.from_quat(dq)).as_quat()
        
        # 3. Covariance Prop (Simplified F matrix)
        F = np.eye(6)
        F[0:3, 3:6] = -np.eye(3) * dt
        self.P = F @ self.P @ F.T + (np.eye(6) * 1e-7)

    def mekf_update(self, q_meas):
        # Compute Error Quaternion between measurement and estimate
        q_err_quat = (R.from_quat(q_meas) * R.from_quat(self.est_q).inv()).as_rotvec()
        
        # Kalman Gain
        H = np.hstack([np.eye(3), np.zeros((3,3))])
        R_mat = np.eye(3) * np.radians(0.005)**2
        K = self.P @ H.T @ np.linalg.inv(H @ self.P @ H.T + R_mat)
        
        # Update state vector
        dx = K @ q_err_quat
        dq_corr = R.from_rotvec(dx[0:3]).as_quat()
        self.est_q = (R.from_quat(dq_corr) * R.from_quat(self.est_q)).as_quat()
        self.est_bias += dx[3:6]
        
        # Update Covariance
        self.P = (np.eye(6) - K @ H) @ self.P

    def control(self, q_target):
        # Quaternion Error Logic
        qe = (R.from_quat(q_target) * R.from_quat(self.est_q).inv()).as_rotvec()
        
        # Agile Gains (Tuned for 150kg)
        Kp, Kd = 0.6, 1.8
        torque = Kp * qe - Kd * (self.omega - self.est_bias)
        return np.clip(torque, -self.max_torque, self.max_torque)

def run_basic_demo():
    sat = SatelliteAOCS()
    # target as quaternion [x,y,z,w]
    q_target = R.from_euler('xyz', [10, 0, 0], degrees=True).as_quat()
    dt = 0.1
    for step in range(200):  # 20 seconds at 10 Hz
        torque = sat.control_law(q_target, dt)
        sat.dynamics(torque, dt)
        if step % 10 == 0:
            dist = np.linalg.norm(sat.q - q_target)
            print(f"Time: {step*dt:4.1f}s | Dist to Target (quat L2): {dist:.6f}")

def run_advanced_montecarlo(iterations=20):
    def single_trial(sim_time=40.0, dt=0.1):
        sim = AdvancedSatellite()
        q_target = R.from_euler('xyz', [10, 0, 0], degrees=True).as_quat()
        dt = 0.1
        
        for t in np.arange(0, 40, dt):
            eclipse = 25.0 < t < 35.0 # Eclipse triggers drift
            
            # 1. Sensors
            w_meas, q_meas = sim.get_mesurement(eclipse)
            
            # 2. Estimate
            sim.mekf_predict(w_meas, dt)
            if q_meas is not None: sim.mekf_update(q_meas)
            
            # 3. Control & Physics
            trq = sim.control(q_target)
            
            # Euler Dynamics
            dw = sim.J_inv @ (trq - np.cross(sim.omega, sim.J @ sim.omega))
            sim.omega += dw * dt
            sim.q = (R.from_quat(sim.q) * R.from_rotvec(sim.omega * dt)).as_quat()
            
        final_err = np.linalg.norm((R.from_quat(sim.q) * R.from_quat(q_target).inv()).as_rotvec())
        return np.degrees(final_err)
            

    results = []
    for i in range(iterations):
        results.append(single_trial())
    results = np.array(results)

    # results is a 1D array of final-pointing-error scalars (degrees)
    final_errors = results
    print(f"Mean Final Error: {np.mean(final_errors):.4f}°")
    print(f"Success Rate (<0.05°): {np.mean(final_errors < 0.05) * 100:.1f}%")
    return final_errors

def main():
    if len(sys.argv) <= 1 or sys.argv[1] == 'help':
        print("Usage: python aocs_sim.py [basic|advanced] [iterations]")
        return
    mode = sys.argv[1]
    if mode == 'basic':
        run_basic_demo()
    elif mode == 'advanced':
        iters = int(sys.argv[2]) if len(sys.argv) > 2 else 20
        run_advanced_montecarlo(iters)
    else:
        print("Unknown mode. Use 'basic' or 'advanced'.")

if __name__ == "__main__":
    main()