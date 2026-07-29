"""
Multiplicative Extended Kalman Filter (MEKF) for Spacecraft Attitude Estimation
===============================================================================

This implementation follows the standard MEKF formulation from:
- Markley & Crassidis, "Fundamentals of Spacecraft Attitude Determination and Control" (2014)
- Lefferts, Markley & Shuster, "Kalman Filtering for Spacecraft Attitude Estimation" (1982)

Key features:
- Multiplicative quaternion error representation (avoids singularities)
- 6-state error vector: [δα (3), δβ (3)] for attitude error and gyro bias
- Proper quaternion normalization and reset after updates
- Vector measurement updates (sun sensor, magnetometer, star tracker)
- Optional gyro bias estimation
"""

import numpy as np
from scipy.linalg import expm
from typing import Optional, Tuple


# =============================================================================
# Utility Functions
# =============================================================================

class QuaternionMath:
    """Quaternion utilities using [w, x, y, z] convention (scalar-first)."""
    
    @staticmethod
    def normalize(q: np.ndarray) -> np.ndarray:
        """Normalize quaternion to unit length."""
        norm = np.linalg.norm(q)
        if norm < 1e-12:
            return np.array([1.0, 0.0, 0.0, 0.0])
        return q / norm
    
    @staticmethod
    def multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """
        Quaternion multiplication: q1 ⊗ q2
        Convention: [w, x, y, z]
        """
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])
    
    @staticmethod
    def conjugate(q: np.ndarray) -> np.ndarray:
        """Quaternion conjugate (inverse for unit quaternions)."""
        return np.array([q[0], -q[1], -q[2], -q[3]])
    
    @staticmethod
    def to_rotation_matrix(q: np.ndarray) -> np.ndarray:
        """Convert quaternion to 3×3 rotation matrix (DCM)."""
        q = QuaternionMath.normalize(q)
        w, x, y, z = q
        return np.array([
            [1-2*(y**2+z**2),   2*(x*y-w*z),     2*(x*z+w*y)],
            [2*(x*y+w*z),       1-2*(x**2+z**2), 2*(y*z-w*x)],
            [2*(x*z-w*y),       2*(y*z+w*x),     1-2*(x**2+y**2)]
        ])
    
    @staticmethod
    def from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
        """Create quaternion from axis-angle representation."""
        axis = axis / (np.linalg.norm(axis) + 1e-12)
        half_angle = 0.5 * angle
        return np.concatenate([
            [np.cos(half_angle)],
            axis * np.sin(half_angle)
        ])
    
    @staticmethod
    def to_axis_angle(q: np.ndarray) -> Tuple[np.ndarray, float]:
        """Extract axis and angle from quaternion."""
        q = QuaternionMath.normalize(q)
        if q[0] < 0:  # Ensure short path
            q = -q
        
        angle = 2.0 * np.arccos(np.clip(q[0], -1.0, 1.0))
        
        if angle < 1e-6:
            axis = np.array([1.0, 0.0, 0.0])
        else:
            axis = q[1:4] / np.sin(angle / 2.0)
            axis = axis / (np.linalg.norm(axis) + 1e-12)
        
        return axis, angle
    
    @staticmethod
    def error_angle(q_true: np.ndarray, q_est: np.ndarray) -> float:
        """
        Compute angular error between two quaternions (radians).
        Returns the rotation angle of q_error = q_true ⊗ q_est^(-1)
        """
        q_est_inv = QuaternionMath.conjugate(q_est)
        q_error = QuaternionMath.multiply(q_true, q_est_inv)
        _, angle = QuaternionMath.to_axis_angle(q_error)
        return angle


def skew_symmetric(v: np.ndarray) -> np.ndarray:
    """
    Create 3×3 skew-symmetric matrix from 3-vector.
    [v×] such that [v×]u = v × u
    """
    return np.array([
        [0,     -v[2],  v[1]],
        [v[2],   0,    -v[0]],
        [-v[1],  v[0],  0   ]
    ])


# =============================================================================
# Attitude Measurement Classes
# =============================================================================

class AttitudeMeasurement:
    """Container for vector observations (sun, mag, star tracker)."""
    
    def __init__(self, 
                 body_vector: np.ndarray,      # Measured in body frame
                 reference_vector: np.ndarray,  # Known in inertial frame
                 covariance: np.ndarray):       # 3×3 measurement noise
        self.body = body_vector / np.linalg.norm(body_vector)
        self.reference = reference_vector / np.linalg.norm(reference_vector)
        self.R = covariance


class QuaternionMeasurement:
    """
    Container for direct quaternion measurements (e.g., star tracker).
    These are converted to error-vector form for MEKF updates.
    """
    
    def __init__(self, q_measured: np.ndarray, covariance: np.ndarray):
        self.q = QuaternionMath.normalize(q_measured)
        self.R = covariance  # 3×3 covariance on attitude error


# =============================================================================
# MEKF Core Implementation
# =============================================================================

class MultiplicativeEKF_HighOrder:
    """
    Multiplicative Extended Kalman Filter for attitude estimation.
    
    State representation:
    - Reference quaternion: q̂ (4 parameters, unit norm)
    - Error state: δx = [δα, δβ] (6 parameters)
        - δα: 3D attitude error (rotation vector / Gibbs vector)
        - δβ: 3D gyro bias error
    
    The true state relates to estimate via:
        q_true = δq(δα) ⊗ q̂
        β_true = β̂ + δβ
    """
    
    def __init__(self,
                 q_init: np.ndarray,
                 P_init: Optional[np.ndarray] = None,
                 estimate_bias: bool = True,
                 gyro_noise_std: float = 1e-4,      # rad/s (ARW)
                 gyro_bias_std: float = 1e-6,       # rad/s² (RRW)
                 process_noise_scale: float = 1.0,
                 measurement_noise_scale: float = 1.0):
        """
        Initialize MEKF.
        
        Args:
            q_init: Initial quaternion estimate [w, x, y, z]
            P_init: Initial 6×6 covariance (default: identity scaled)
            estimate_bias: Whether to estimate gyro bias
            gyro_noise_std: Angular random walk (rad/s)
            gyro_bias_std: Rate random walk / bias instability (rad/s²)
            process_noise_scale: Multiplier for process noise
            measurement_noise_scale: Multiplier for measurement noise
        """
        # Reference quaternion (unit norm, scalar-first)
        self.q = QuaternionMath.normalize(q_init)
        
        # Gyro bias estimate
        self.bias = np.zeros(3)
        self.estimate_bias = estimate_bias
        
        # Error-state covariance (6×6)
        if P_init is None:
            P_init = np.eye(6) * 1e-6
            P_init[:3, :3] *= (1e-3)**2  # Attitude uncertainty (rad²)
            P_init[3:, 3:] *= (1e-5)**2  # Bias uncertainty (rad/s)²
        self.P = P_init
        
        # Process noise parameters
        self.sigma_v = gyro_noise_std * process_noise_scale       # ARW
        self.sigma_u = gyro_bias_std * process_noise_scale        # RRW
        self.measurement_noise_scale = measurement_noise_scale
        
        # State transition and noise matrices (computed in predict)
        self.F = np.zeros((6, 6))
        self.G = np.block([
            [-np.eye(3), np.zeros((3, 3))],
            [np.zeros((3, 3)), np.eye(3)]
        ])
        
        # Statistics
        self.innovation_history = []
    
    def predict(self, omega_gyro: np.ndarray, dt: float):
        """
        Propagate state and covariance forward using gyro measurement.
        
        Args:
            omega_gyro: Measured angular velocity (rad/s) [ωx, ωy, ωz]
            dt: Time step (seconds)
        
        Mathematical model:
            q̇ = 0.5 * Ω(ω) * q    where ω = ω_gyro - β̂
            δα̇ = -[ω̂×]δα - δβ - ηv
            δβ̇ = ηu
        """
        # Bias-corrected angular velocity
        omega_corrected = omega_gyro - self.bias
        
        # =====================================================================
        # 1. Propagate reference quaternion
        # =====================================================================
        # Build Ω matrix: q̇ = 0.5 Ω(ω) q
        Omega = np.array([
            [0,                 -omega_corrected[0], -omega_corrected[1], -omega_corrected[2]],
            [omega_corrected[0], 0,                   omega_corrected[2], -omega_corrected[1]],
            [omega_corrected[1],-omega_corrected[2],  0,                   omega_corrected[0]],
            [omega_corrected[2], omega_corrected[1], -omega_corrected[0],  0                 ]
        ])
        
        # Quaternion propagation (exact for constant ω over dt)
        # q(t+dt) = exp(0.5 * Ω * dt) * q(t)
        Phi_q = expm(0.5 * Omega * dt)
        self.q = Phi_q @ self.q
        self.q = QuaternionMath.normalize(self.q)
        
        # =====================================================================
        # 2. Propagate gyro bias (constant model)
        # =====================================================================
        # β̇ = 0  →  β(t+dt) = β(t)
        # (bias stays constant; will be updated by measurements)
        
        # =====================================================================
        # 3. Build error-state dynamics and propagate covariance
        # =====================================================================
        # Linearized error dynamics:
        # F = [ -[ω̂×]  -I ]
        #     [   0     0 ]
        omega_skew = skew_symmetric(omega_corrected)
        self.F[:3, :3] = -omega_skew
        self.F[:3, 3:] = -np.eye(3)
        self.F[3:, :] = 0.0
        
        # Discrete state transition (first-order approximation)
        Phi = np.eye(6) + self.F * dt
        
        # Process noise covariance (continuous-time)
        Q_c = np.diag([self.sigma_v**2] * 3 + [self.sigma_u**2] * 3)
        
        # Discretize process noise (Van Loan method approximation)
        # For simple model: Q_d ≈ G * Q_c * G^T * dt
        Q_d = self.G @ Q_c @ self.G.T * dt
        
        # Covariance propagation
        self.P = Phi @ self.P @ Phi.T + Q_d
        
        # Ensure symmetry (numerical stability)
        self.P = 0.5 * (self.P + self.P.T)
    
    def update_vector_measurement(self, measurement: AttitudeMeasurement):
        """
        Update using a vector observation (sun sensor, magnetometer, etc.).
        
        Measurement model:
            b_measured = A(q_true) * r_inertial + ν
            
        Linearized:
            δb ≈ [b_predicted ×] δα + ν
            
        where A(q) is the rotation matrix and [×] is skew-symmetric.
        
        Args:
            measurement: AttitudeMeasurement object
        """
        # Predicted measurement
        A = QuaternionMath.to_rotation_matrix(self.q)
        b_predicted = A @ measurement.reference
        
        # Innovation (measurement residual)
        innovation = measurement.body - b_predicted
        
        # Measurement sensitivity matrix
        # H = [∂h/∂δα, ∂h/∂δβ] = [[b_pred×], 0]
        H = np.zeros((3, 6))
        H[:, :3] = skew_symmetric(b_predicted)  # Note: some formulations use negative
        
        # Measurement noise (scaled)
        R = measurement.R * self.measurement_noise_scale
        
        # Innovation covariance
        S = H @ self.P @ H.T + R
        
        # Kalman gain
        try:
            K = self.P @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            print("Warning: Singular innovation covariance, skipping update")
            return
        
        # Error-state estimate
        delta_x = K @ innovation
        delta_alpha = delta_x[:3]
        delta_beta = delta_x[3:6]
        
        # =====================================================================
        # Multiplicative quaternion update (THE KEY STEP)
        # =====================================================================
        # Build error quaternion from δα
        angle = np.linalg.norm(delta_alpha)
        if angle > 1e-10:
            axis = delta_alpha / angle
            delta_q = QuaternionMath.from_axis_angle(axis, angle)
        else:
            # Small-angle approximation: δq ≈ [1, δα/2]
            delta_q = np.concatenate([[1.0], 0.5 * delta_alpha])
            delta_q = QuaternionMath.normalize(delta_q)
        
        # Apply multiplicative correction: q⁺ = δq ⊗ q⁻
        self.q = QuaternionMath.multiply(delta_q, self.q)
        self.q = QuaternionMath.normalize(self.q)
        
        # =====================================================================
        # Additive bias update
        # =====================================================================
        if self.estimate_bias:
            self.bias += delta_beta
        
        # =====================================================================
        # Covariance update (Joseph form for numerical stability)
        # =====================================================================
        I_KH = np.eye(6) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T
        
        # Ensure symmetry
        self.P = 0.5 * (self.P + self.P.T)
        
        # Store innovation for diagnostics
        self.innovation_history.append({
            'innovation': innovation,
            'innovation_cov': S,
            'mahalanobis': innovation.T @ np.linalg.inv(S) @ innovation
        })
    
    def update_quaternion_measurement(self, measurement: QuaternionMeasurement):
        """
        Update using a direct quaternion measurement (e.g., star tracker).
        
        The quaternion measurement is converted to a 3D attitude error vector:
            δq = q_meas ⊗ q̂^(-1)
            δα = 2 * [δq_x, δq_y, δq_z] / δq_w  (Gibbs vector)
        
        Args:
            measurement: QuaternionMeasurement object
        """
        # Compute error quaternion
        q_est_inv = QuaternionMath.conjugate(self.q)
        delta_q = QuaternionMath.multiply(measurement.q, q_est_inv)
        
        # Ensure short path (scalar component positive)
        if delta_q[0] < 0:
            delta_q = -delta_q
        
        # Convert to 3-parameter error representation (Gibbs vector)
        # For small errors: δα ≈ 2 * [δq_x, δq_y, δq_z] / δq_w
        # For numerical stability when δq_w ≈ 1:
        if delta_q[0] > 0.9:  # Small rotation
            delta_alpha = 2.0 * delta_q[1:4] / (delta_q[0] + 1e-12)
        else:  # Large rotation - use full formula
            axis, angle = QuaternionMath.to_axis_angle(delta_q)
            delta_alpha = axis * angle
        
        # Measurement model: we directly observe the attitude error
        # z = δα + ν  where ν ~ N(0, R)
        H = np.zeros((3, 6))
        H[:3, :3] = np.eye(3)  # Direct observation of δα
        
        R = measurement.R * self.measurement_noise_scale
        
        # Innovation covariance
        S = H @ self.P @ H.T + R
        
        # Kalman gain
        try:
            K = self.P @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            print("Warning: Singular innovation covariance, skipping update")
            return
        
        # Error-state update
        delta_x = K @ delta_alpha
        delta_alpha_corrected = delta_x[:3]
        delta_beta = delta_x[3:6]
        
        # Multiplicative quaternion correction
        angle = np.linalg.norm(delta_alpha_corrected)
        if angle > 1e-10:
            axis = delta_alpha_corrected / angle
            delta_q_corrected = QuaternionMath.from_axis_angle(axis, angle)
        else:
            delta_q_corrected = np.concatenate([[1.0], 0.5 * delta_alpha_corrected])
            delta_q_corrected = QuaternionMath.normalize(delta_q_corrected)
        
        self.q = QuaternionMath.multiply(delta_q_corrected, self.q)
        self.q = QuaternionMath.normalize(self.q)
        
        # Bias update
        if self.estimate_bias:
            self.bias += delta_beta
        
        # Covariance update (Joseph form)
        I_KH = np.eye(6) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T
        self.P = 0.5 * (self.P + self.P.T)
        
        # Diagnostics
        self.innovation_history.append({
            'innovation': delta_alpha,
            'innovation_cov': S,
            'mahalanobis': delta_alpha.T @ np.linalg.inv(S) @ delta_alpha
        })
    
    def get_state(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get current state estimate.
        
        Returns:
            q: Quaternion estimate [w, x, y, z]
            bias: Gyro bias estimate [rad/s]
            P: 6×6 covariance matrix
        """
        return self.q.copy(), self.bias.copy(), self.P.copy()
    
    def get_attitude_uncertainty(self) -> float:
        """
        Get 1-sigma attitude uncertainty (radians).
        Returns the trace of the attitude covariance block.
        """
        return np.sqrt(np.trace(self.P[:3, :3]))


# =============================================================================
# Comparison and Metrics
# =============================================================================

class MEKFComparator:
    """Utilities for comparing MEKF performance against ground truth."""
    
    @staticmethod
    def quaternion_error(q_true: np.ndarray, q_est: np.ndarray) -> float:
        """Angular error in radians."""
        return QuaternionMath.error_angle(q_true, q_est)
    
    @staticmethod
    def compute_metrics(errors: np.ndarray) -> dict:
        """Compute statistical metrics on error array (radians)."""
        return {
            'mean_rad': float(np.mean(errors)),
            'rms_rad': float(np.sqrt(np.mean(errors**2))),
            'std_rad': float(np.std(errors)),
            'max_rad': float(np.max(errors)),
            'mean_arcsec': float(np.mean(errors) * 206265),
            'rms_arcsec': float(np.sqrt(np.mean(errors**2)) * 206265),
            'max_arcsec': float(np.max(errors) * 206265),
        }
    
    @staticmethod
    def print_metrics(errors: np.ndarray, label: str = "MEKF"):
        """Print formatted error statistics."""
        m = MEKFComparator.compute_metrics(errors)
        print(f"\n{label} Performance:")
        print(f"  Mean error : {m['mean_rad']*1e6:8.2f} µrad  ({m['mean_arcsec']:8.2f} arcsec)")
        print(f"  RMS error  : {m['rms_rad']*1e6:8.2f} µrad  ({m['rms_arcsec']:8.2f} arcsec)")
        print(f"  Std dev    : {m['std_rad']*1e6:8.2f} µrad")
        print(f"  Max error  : {m['max_rad']*1e6:8.2f} µrad  ({m['max_arcsec']:8.2f} arcsec)")


# =============================================================================
# Utility: Convert q_dot to angular velocity
# =============================================================================

def angular_velocity_from_quaternion_derivative(q: np.ndarray, q_dot: np.ndarray) -> np.ndarray:
    """
    Extract angular velocity ω from quaternion and its derivative.
    
    Kinematic equation: q̇ = 0.5 * Ω(ω) * q
    Solving for ω: ω = 2 * [q̇ ⊗ q*]_vector
    
    Args:
        q: Quaternion [w, x, y, z]
        q_dot: Quaternion derivative [ẇ, ẋ, ẏ, ż]
    
    Returns:
        omega: Angular velocity [ωx, ωy, ωz] in rad/s
    """
    q_conj = QuaternionMath.conjugate(q)
    omega_q = QuaternionMath.multiply(q_dot, q_conj)
    return 2.0 * omega_q[1:4]  # Extract vector part and scale


if __name__ == "__main__":
    """Simple test of MEKF with synthetic data."""
    
    print("="*70)
    print("MEKF Unit Test")
    print("="*70)
    
    # Simulation parameters
    dt = 0.01  # 100 Hz
    T = 10.0   # 10 seconds
    N = int(T / dt)
    
    # True trajectory: constant rotation about z-axis
    omega_true = np.array([0.0, 0.0, 0.1])  # 0.1 rad/s ≈ 5.7 deg/s
    q_true = np.zeros((N, 4))
    q_true[0] = np.array([1.0, 0.0, 0.0, 0.0])
    
    for i in range(1, N):
        angle = np.linalg.norm(omega_true) * dt
        axis = omega_true / (np.linalg.norm(omega_true) + 1e-12)
        dq = QuaternionMath.from_axis_angle(axis, angle)
        q_true[i] = QuaternionMath.multiply(dq, q_true[i-1])
    
    # Initialize MEKF with small error
    q_init = QuaternionMath.multiply(
        QuaternionMath.from_axis_angle(np.array([1, 0, 0]), 0.01),  # 0.01 rad initial error
        q_true[0]
    )
    
    mekf = MultiplicativeEKF_HighOrder(
        q_init=q_init,
        estimate_bias=False,
        gyro_noise_std=1e-4,
        measurement_noise_scale=1e-3
    )
    
    # Simulate
    q_est = np.zeros((N, 4))
    errors = np.zeros(N)
    
    # Reference vector (sun direction in inertial frame)
    sun_inertial = np.array([1.0, 0.0, 0.0])
    
    for i in range(N):
        # Gyro measurement (with noise)
        omega_meas = omega_true + np.random.randn(3) * 1e-4
        
        # Predict
        mekf.predict(omega_meas, dt)
        
        # Update every 10 steps (10 Hz measurements)
        if i % 10 == 0:
            # Simulate sun sensor measurement
            A_true = QuaternionMath.to_rotation_matrix(q_true[i])
            sun_body_true = A_true @ sun_inertial
            sun_body_meas = sun_body_true + np.random.randn(3) * 1e-3
            sun_body_meas = sun_body_meas / np.linalg.norm(sun_body_meas)
            
            measurement = AttitudeMeasurement(
                body_vector=sun_body_meas,
                reference_vector=sun_inertial,
                covariance=np.eye(3) * (1e-3)**2
            )
            
            mekf.update_vector_measurement(measurement)
        
        q_est[i] = mekf.q
        errors[i] = QuaternionMath.error_angle(q_true[i], q_est[i])
    
    # Report
    MEKFComparator.print_metrics(errors, "MEKF Test")
    
    print(f"\nFinal attitude uncertainty: {mekf.get_attitude_uncertainty()*1e6:.2f} µrad")
    print(f"Final bias estimate: {mekf.bias} rad/s")
    print("\n" + "="*70)