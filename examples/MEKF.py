import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt
import os
import pandas as pd


# =============================================================================
# Data Containers
# =============================================================================

@dataclass
class AttitudeMeasurement:
    """Container for timestamped attitude data (Scalar-First Quaternions)"""
    timestamp: float
    q: np.ndarray               # 4-element quaternion [w, x, y, z]
    q_dot: np.ndarray           # 4-element quaternion derivative [w_dot, x_dot, y_dot, z_dot]
    q_covariance: Optional[np.ndarray] = None


@dataclass
class BatchResult:
    """Container for batch processing results"""
    timestamps: np.ndarray
    q_predicted: np.ndarray     # Shape: (N, 4)
    q_corrected: np.ndarray     # Shape: (N, 4)
    q_dot_corrected: np.ndarray # Shape: (N, 4)
    covariances: np.ndarray     # Shape: (N, 7, 7) or (N, 4, 4)
    biases: np.ndarray          # Shape: (N, 3) or empty if not estimating
    innovations: np.ndarray     # Shape: (N, 3) - rotation vector residuals


# =============================================================================
# Quaternion Math Utilities
# =============================================================================

class QuaternionMath:
    """Utility class for Scalar-First Quaternion operations [w, x, y, z]"""

    @staticmethod
    def normalize(q: np.ndarray) -> np.ndarray:
        q = np.array(q, dtype=float)
        norm = np.linalg.norm(q)
        if norm < 1e-10:
            raise ValueError("Cannot normalize zero quaternion")
        return q / norm

    @staticmethod
    def conjugate(q: np.ndarray) -> np.ndarray:
        return np.array([q[0], -q[1], -q[2], -q[3]])

    @staticmethod
    def multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])

    @staticmethod
    def inverse(q: np.ndarray) -> np.ndarray:
        return QuaternionMath.conjugate(q)

    @staticmethod
    def from_rotation_vector(theta: np.ndarray) -> np.ndarray:
        theta = np.array(theta)
        theta_norm = np.linalg.norm(theta)
        if theta_norm < 1e-10:
            return np.array([1.0, 0.0, 0.0, 0.0])
        half_theta = 0.5 * theta_norm
        axis = theta / theta_norm
        return np.array([
            np.cos(half_theta),
            axis[0] * np.sin(half_theta),
            axis[1] * np.sin(half_theta),
            axis[2] * np.sin(half_theta)
        ])

    @staticmethod
    def to_rotation_vector(q: np.ndarray) -> np.ndarray:
        q = QuaternionMath.normalize(q)
        w = np.clip(q[0], -1.0, 1.0)
        angle = 2.0 * np.arccos(w)
        v = q[1:]
        v_norm = np.linalg.norm(v)
        if v_norm < 1e-10:
            return np.zeros(3)
        return (v / v_norm) * angle

    @staticmethod
    def extract_omega_from_qdot(q: np.ndarray, q_dot: np.ndarray) -> np.ndarray:
        q_inv = QuaternionMath.inverse(q)
        temp = QuaternionMath.multiply(q_inv, q_dot)
        return 2.0 * temp[1:]


# =============================================================================
# Base MEKF
# =============================================================================

class MultiplicativeEKF:
    """
    Multiplicative EKF for attitude estimation (Scalar-First Quaternions).

    Note: This class is designed to be extended. In this codebase,
    MultiplicativeEKF_HighOrder is always used. The predict() method here
    provides a first-order Euler fallback; the HighOrder subclass overrides
    it with a higher-accuracy integrator.
    """

    def __init__(
        self,
        q_init: np.ndarray,
        estimate_bias: bool = True,
        process_noise_scale: float = 0.01,
        measurement_noise_scale: float = 0.1,
        bias_process_noise: float = 0.001
    ):
        self.q = QuaternionMath.normalize(np.array(q_init))
        self.estimate_bias = estimate_bias
        self.state_dim = 7 if estimate_bias else 4
        self.bias = np.zeros(3) if estimate_bias else None
        self.P = np.eye(self.state_dim) * 0.1
        self.Q = np.eye(self.state_dim) * process_noise_scale
        self.R = np.eye(3) * measurement_noise_scale
        self.bias_process_noise = bias_process_noise
        self.last_timestamp: Optional[float] = None
        self.history: List[Dict] = []

    def _get_jacobian_f(self, omega: np.ndarray, dt: float) -> np.ndarray:
        """State transition Jacobian for covariance prediction"""
        J = np.eye(self.state_dim)
        wx, wy, wz = omega
        omega_skew = np.array([
            [0,   -wz,  wy],
            [wz,   0,  -wx],
            [-wy,  wx,   0]
        ])
        J[:3, :3] = np.eye(3) - omega_skew * dt
        if self.estimate_bias:
            J[4:, 4:] = np.eye(3)
            J[:3, 4:] = -np.eye(3) * dt
        return J

    def _get_jacobian_h(self) -> np.ndarray:
        """Measurement Jacobian — maps state error to rotation-vector observation"""
        H = np.zeros((3, self.state_dim))
        H[:, :3] = np.eye(3)
        return H

    def predict(self, q_dot: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """First-order Euler prediction step using q_dot (overridden by HighOrder subclass)"""
        q_dot = np.array(q_dot)
        omega_raw = QuaternionMath.extract_omega_from_qdot(self.q, q_dot)
        omega = omega_raw - self.bias if self.estimate_bias else omega_raw

        delta_q = QuaternionMath.from_rotation_vector(0.5 * omega * dt)
        self.q = QuaternionMath.normalize(QuaternionMath.multiply(self.q, delta_q))

        J = self._get_jacobian_f(omega, dt)
        self.P = J @ self.P @ J.T + self.Q
        self.Q[:4, :4] *= dt
        if self.estimate_bias:
            self.Q[4:, 4:] *= dt * self.bias_process_noise

        return self.q.copy(), self.P.copy()

    def update(self, q_meas: np.ndarray, q_meas_cov: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Multiplicative update step.
        Returns (q_corrected, P_corrected, innovation_rotation_vector).
        """
        q_meas = QuaternionMath.normalize(np.array(q_meas))

        # Ensure shortest-path sign consistency
        if np.dot(self.q, q_meas) < 0:
            q_meas = -q_meas

        # Innovation as rotation vector
        q_error = QuaternionMath.multiply(q_meas, QuaternionMath.inverse(self.q))
        delta_theta = QuaternionMath.to_rotation_vector(q_error)

        H = self._get_jacobian_h()

        # Resolve measurement covariance
        if q_meas_cov is not None:
            if q_meas_cov.shape == (4, 4):
                R = q_meas_cov[1:, 1:]
            elif q_meas_cov.shape == (3, 3):
                R = q_meas_cov
            else:
                raise ValueError(f"Invalid measurement covariance shape: {q_meas_cov.shape}")
        else:
            R = self.R

        # Kalman gain
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        delta_x = K @ delta_theta

        # Apply multiplicative quaternion correction
        delta_theta_corr = delta_x[:3]
        delta_q = QuaternionMath.from_rotation_vector(0.5 * delta_theta_corr)
        self.q = QuaternionMath.normalize(QuaternionMath.multiply(delta_q, self.q))

        # Optional bias correction
        if self.estimate_bias:
            self.bias = self.bias + delta_x[4:]

        # Joseph form covariance update (numerically stable)
        I_KH = np.eye(self.state_dim) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T

        return self.q.copy(), self.P.copy(), delta_theta

    def process_single(self, measurement: AttitudeMeasurement) -> Dict:
        """Run predict + update for one measurement; append to history."""
        if self.last_timestamp is not None:
            dt = measurement.timestamp - self.last_timestamp
            if dt <= 0:
                raise ValueError("Timestamps must be strictly increasing")
        else:
            dt = 0.01

        self.last_timestamp = measurement.timestamp

        q_pred, P_pred = self.predict(measurement.q_dot, dt)
        q_corr, P_corr, innovation = self.update(measurement.q, measurement.q_covariance)

        record = {
            'timestamp':   measurement.timestamp,
            'q_predicted': q_pred,
            'q_corrected': q_corr,
            'covariance':  P_corr,
            'bias':        self.bias.copy() if self.estimate_bias else None,
            'innovation':  innovation,
            'dt':          dt
        }
        self.history.append(record)
        return record

    def process_batch(self, measurements: List[AttitudeMeasurement]) -> BatchResult:
        """Process a list of measurements chronologically and return a BatchResult."""
        if not measurements:
            raise ValueError("Empty measurement list")

        measurements = sorted(measurements, key=lambda m: m.timestamp)
        self.history.clear()
        self.last_timestamp = None

        N = len(measurements)
        timestamps      = np.zeros(N)
        q_predicted     = np.zeros((N, 4))
        q_corrected     = np.zeros((N, 4))
        q_dot_corrected = np.zeros((N, 4))
        covariances     = np.zeros((N, self.state_dim, self.state_dim))
        biases          = np.zeros((N, 3)) if self.estimate_bias else np.zeros((N, 0))
        innovations     = np.zeros((N, 3))

        for i, meas in enumerate(measurements):
            record = self.process_single(meas)

            timestamps[i]  = record['timestamp']
            q_predicted[i] = record['q_predicted']
            q_corrected[i] = record['q_corrected']
            covariances[i] = record['covariance']
            innovations[i] = record['innovation']

            if self.estimate_bias:
                biases[i] = record['bias']

            # Reconstruct corrected q_dot from bias-corrected angular velocity
            omega_raw = QuaternionMath.extract_omega_from_qdot(meas.q, meas.q_dot)
            omega_corrected = omega_raw - record['bias'] if self.estimate_bias else omega_raw
            q_dot_corrected[i] = 0.5 * QuaternionMath.multiply(
                q_corrected[i],
                np.concatenate([[0], omega_corrected])
            )

        return BatchResult(
            timestamps=timestamps,
            q_predicted=q_predicted,
            q_corrected=q_corrected,
            q_dot_corrected=q_dot_corrected,
            covariances=covariances,
            biases=biases,
            innovations=innovations
        )

    def reset(self, q_init: np.ndarray):
        """Reset filter to a new initial quaternion."""
        self.q = QuaternionMath.normalize(np.array(q_init))
        self.P = np.eye(self.state_dim) * 0.1
        if self.estimate_bias:
            self.bias = np.zeros(3)
        self.last_timestamp = None
        self.history.clear()


# =============================================================================
# High-Order Integrator
# =============================================================================

def integrate_high_order_aocs(
    q: np.ndarray,
    omega: np.ndarray,
    alpha: np.ndarray,
    jerk: np.ndarray,
    dt: float,
    norm_threshold: float = 1e-6
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    High-order quaternion integration using angular velocity, acceleration, and jerk.
    Uses a Taylor-expanded rotation vector (Algorithm 5) for accuracy far beyond
    first-order Euler integration.

    Returns: (q_next, q_dot, omega_next, alpha_next)
    """
    # 1. Propagate kinematics
    alpha_next = alpha + jerk * dt
    omega_next = omega + alpha * dt + 0.5 * jerk * (dt**2)

    # 2. Rotation increment vector
    dphi = omega * dt + 0.5 * alpha * (dt**2) + (1/6) * jerk * (dt**3)
    dx, dy, dz = dphi

    # 3. Higher-order quaternion increment via Taylor expansion of sin/cos
    d2 = dx**2 + dy**2 + dz**2
    d4 = d2**2
    s = 0.5 - (d2 / 48.0) + (d4 / 3840.0)
    c =     - (d2 / 8.0)  + (d4 / 384.0)
    sx, sy, sz = s * dx, s * dy, s * dz

    dq = np.array([
        c*q[0] - sx*q[1] - sy*q[2] - sz*q[3],
        c*q[1] + sx*q[0] + sz*q[3] - sy*q[2],
        c*q[2] + sy*q[0] + sx*q[3] - sz*q[1],
        c*q[3] + sz*q[0] + sy*q[1] - sx*q[2]
    ])

    # 4. Update and normalize
    q_next = q + dq
    norm_sq = np.dot(q_next, q_next)
    if abs(1.0 - norm_sq) > norm_threshold:
        q_next = q_next / np.sqrt(norm_sq)
    else:
        q_next = q_next * (1.5 - 0.5 * norm_sq)  # Fast Padé normalization

    # 5. Instantaneous q_dot at current step
    q_dot = 0.5 * np.array([
        -q[1]*omega[0] - q[2]*omega[1] - q[3]*omega[2],
         q[0]*omega[0] + q[2]*omega[2] - q[3]*omega[1],
         q[0]*omega[1] - q[1]*omega[2] + q[3]*omega[0],
         q[0]*omega[2] + q[1]*omega[1] - q[2]*omega[0]
    ])

    return q_next, q_dot, omega_next, alpha_next


class MultiplicativeEKF_HighOrder(MultiplicativeEKF):
    """
    MEKF with high-order quaternion integration.
    Overrides predict() to use integrate_high_order_aocs() instead of first-order Euler.
    Angular acceleration (alpha) and jerk are computed by finite-differencing omega.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_omega: Optional[np.ndarray] = None
        self.last_alpha: Optional[np.ndarray] = None

    def _compute_kinematic_derivatives(
        self, q: np.ndarray, q_dot: np.ndarray, dt: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Finite-difference omega to obtain alpha and jerk."""
        omega = QuaternionMath.extract_omega_from_qdot(q, q_dot)

        alpha = (omega - self.last_omega) / dt if self.last_omega is not None else np.zeros(3)
        jerk  = (alpha - self.last_alpha) / dt if self.last_alpha is not None else np.zeros(3)

        self.last_omega = omega.copy()
        self.last_alpha = alpha.copy()

        return omega, alpha, jerk

    def predict(self, q_dot: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """High-order prediction step."""
        q_dot = np.array(q_dot)
        omega, alpha, jerk = self._compute_kinematic_derivatives(self.q, q_dot, dt)

        q_next, _, _, _ = integrate_high_order_aocs(self.q, omega, alpha, jerk, dt)
        self.q = q_next

        J = self._get_jacobian_f(omega, dt)
        self.P = J @ self.P @ J.T + self.Q
        self.Q[:4, :4] *= dt
        if self.estimate_bias:
            self.Q[4:, 4:] *= dt * self.bias_process_noise

        return self.q.copy(), self.P.copy()

    def reset(self, q_init: np.ndarray):
        """Reset filter state including cached kinematic derivatives."""
        super().reset(q_init)
        self.last_omega = None
        self.last_alpha = None


# =============================================================================
# Accuracy Evaluation
# =============================================================================

@dataclass
class AccuracyMetrics:
    """Container for accuracy evaluation metrics"""
    mean_q_error:    float   # Mean quaternion error (radians)
    std_q_error:     float   # Std dev of quaternion error
    max_q_error:     float   # Maximum quaternion error
    mean_bias_error: float  # Mean bias estimation error
    std_bias_error:  float  # Std dev of bias error
    mean_innovation: float  # Mean innovation magnitude
    rmse_q:          float   # RMSE of quaternion error
    rmse_omega:      float   # RMSE of angular velocity error


class MEKFComparator:
    """
    Compare MEKF batch results against ground truth.
    Generates accuracy metrics and visualizations.
    """

    def __init__(self, timestamps: np.ndarray):
        self.timestamps = timestamps
        self.N = len(timestamps)

    @staticmethod
    def quaternion_error(q1: np.ndarray, q2: np.ndarray) -> float:
        """Angular error between two quaternions (radians), taking the shortest path."""
        q1 = QuaternionMath.normalize(q1)
        q2 = QuaternionMath.normalize(q2)
        if np.dot(q1, q2) < 0:
            q2 = -q2
        dot = np.clip(np.abs(np.dot(q1, q2)), 0.0, 1.0)
        return 2.0 * np.arccos(dot)

    def calculate_metrics(
        self,
        q_true:          np.ndarray,
        q_corrected:     np.ndarray,
        q_predicted:     np.ndarray,
        omega_true:      np.ndarray,
        omega_estimated: np.ndarray,
        bias_true:       Optional[np.ndarray] = None,
        bias_estimated:  Optional[np.ndarray] = None,
        innovations:     Optional[np.ndarray] = None
    ) -> AccuracyMetrics:
        """Compute comprehensive accuracy metrics."""
        q_errors = np.array([
            self.quaternion_error(q_true[i], q_corrected[i]) for i in range(self.N)
        ])
        omega_errors = np.linalg.norm(omega_true - omega_estimated, axis=1)

        if bias_true is not None and bias_estimated is not None:
            bias_errors = np.linalg.norm(bias_true - bias_estimated, axis=1)
            mean_bias_error = float(np.mean(bias_errors))
            std_bias_error  = float(np.std(bias_errors))
        else:
            mean_bias_error = std_bias_error = 0.0

        mean_innovation = float(np.mean(np.linalg.norm(innovations, axis=1))) if innovations is not None else 0.0

        return AccuracyMetrics(
            mean_q_error=float(np.mean(q_errors)),
            std_q_error=float(np.std(q_errors)),
            max_q_error=float(np.max(q_errors)),
            mean_bias_error=mean_bias_error,
            std_bias_error=std_bias_error,
            mean_innovation=mean_innovation,
            rmse_q=float(np.sqrt(np.mean(q_errors**2))),
            rmse_omega=float(np.sqrt(np.mean(omega_errors**2)))
        )

    def plot_comparison(
        self,
        q_true:          np.ndarray,
        q_corrected:     np.ndarray,
        q_predicted:     np.ndarray,
        omega_true:      np.ndarray,
        omega_estimated: np.ndarray,
        bias_true:       Optional[np.ndarray] = None,
        bias_estimated:  Optional[np.ndarray] = None,
        innovations:     Optional[np.ndarray] = None,
        save_path:       Optional[str] = None,
        show:            bool = True
    ):
        """Generate a 3×3 panel comparison plot."""
        plt.figure(figsize=(16, 12))

        # --- Plot 1: Quaternion Components ---
        ax1 = plt.subplot(3, 3, 1)
        for i, label in enumerate(['w', 'x', 'y', 'z']):
            ax1.plot(self.timestamps, q_true[:, i],      'k-',  lw=1.5, alpha=0.5, label=f'True {label}')
            ax1.plot(self.timestamps, q_corrected[:, i], 'b-',  lw=1.5, label=f'Corrected {label}')
            ax1.plot(self.timestamps, q_predicted[:, i], 'r--', lw=1,   alpha=0.7,
                     label='Predicted' if i == 0 else "")
        ax1.set(xlabel='Time (s)', ylabel='Quaternion Component', title='Quaternion Components')
        ax1.legend(loc='upper right', fontsize=8)
        ax1.grid(True, alpha=0.3)

        # --- Plot 2: Quaternion Error ---
        ax2 = plt.subplot(3, 3, 2)
        q_errors = np.array([self.quaternion_error(q_true[i], q_corrected[i]) for i in range(self.N)])
        ax2.plot(self.timestamps, q_errors * 1e6, 'b-', lw=1.5)
        ax2.axhline(np.mean(q_errors) * 1e6, color='r', ls='--',
                    label=f'Mean: {np.mean(q_errors)*1e6:.2f} µrad')
        ax2.set(xlabel='Time (s)', ylabel='Error (µrad)', title='Attitude Estimation Error')
        ax2.legend(); ax2.grid(True, alpha=0.3)

        # --- Plot 3: Angular Velocity Comparison ---
        ax3 = plt.subplot(3, 3, 3)
        for i, label in enumerate(['ω_x', 'ω_y', 'ω_z']):
            ax3.plot(self.timestamps, omega_true[:, i],      'k-', lw=1, alpha=0.5,
                     label='True' if i == 0 else "")
            ax3.plot(self.timestamps, omega_estimated[:, i], 'b-', lw=1,
                     label='Estimated' if i == 0 else "")
        ax3.set(xlabel='Time (s)', ylabel='Angular Velocity (rad/s)', title='Angular Velocity Comparison')
        ax3.legend(loc='upper right', fontsize=8); ax3.grid(True, alpha=0.3)

        # --- Plot 4: Angular Velocity Error ---
        ax4 = plt.subplot(3, 3, 4)
        omega_errors = np.linalg.norm(omega_true - omega_estimated, axis=1)
        ax4.plot(self.timestamps, omega_errors, 'b-', lw=1.5)
        ax4.axhline(np.mean(omega_errors), color='r', ls='--',
                    label=f'Mean: {np.mean(omega_errors):.4f} rad/s')
        ax4.set(xlabel='Time (s)', ylabel='Error (rad/s)', title='Angular Velocity Error')
        ax4.legend(); ax4.grid(True, alpha=0.3)

        # --- Plots 5 & 6: Bias (or placeholder) ---
        if bias_true is not None and bias_estimated is not None:
            ax5 = plt.subplot(3, 3, 5)
            for i, label in enumerate(['b_x', 'b_y', 'b_z']):
                ax5.plot(self.timestamps, bias_true[:, i],      'k-', lw=1, alpha=0.5,
                         label='True' if i == 0 else "")
                ax5.plot(self.timestamps, bias_estimated[:, i], 'b-', lw=1,
                         label='Estimated' if i == 0 else "")
            ax5.set(xlabel='Time (s)', ylabel='Bias (rad/s)', title='Gyroscope Bias Estimation')
            ax5.legend(loc='upper right', fontsize=8); ax5.grid(True, alpha=0.3)

            ax6 = plt.subplot(3, 3, 6)
            bias_errors = np.linalg.norm(bias_true - bias_estimated, axis=1)
            ax6.plot(self.timestamps, bias_errors, 'b-', lw=1.5)
            ax6.axhline(np.mean(bias_errors), color='r', ls='--',
                        label=f'Mean: {np.mean(bias_errors):.6f} rad/s')
            ax6.set(xlabel='Time (s)', ylabel='Bias Error (µrad)', title='Bias Estimation Error')
            ax6.legend(); ax6.grid(True, alpha=0.3)
        else:
            for pos in (5, 6):
                ax = plt.subplot(3, 3, pos)
                ax.text(0.5, 0.5, 'Bias estimation\nnot available', ha='center', va='center', fontsize=12)
                ax.axis('off')

        # --- Plot 7: Innovation Magnitude ---
        if innovations is not None:
            ax7 = plt.subplot(3, 3, 7)
            innovation_mags = np.linalg.norm(innovations, axis=1)
            ax7.plot(self.timestamps, innovation_mags, 'b-', lw=1.5)
            ax7.axhline(np.mean(innovation_mags), color='r', ls='--',
                        label=f'Mean: {np.mean(innovation_mags):.4f}')
            ax7.set(xlabel='Time (s)', ylabel='Innovation Magnitude', title='Filter Innovation (Residual)')
            ax7.legend(); ax7.grid(True, alpha=0.3)
        else:
            ax7 = plt.subplot(3, 3, 7)
            ax7.text(0.5, 0.5, 'Innovation data\nnot available', ha='center', va='center', fontsize=12)
            ax7.axis('off')

        # --- Plot 8: Error Distribution ---
        ax8 = plt.subplot(3, 3, 8)
        q_errors_deg = np.degrees(q_errors)
        ax8.hist(q_errors_deg, bins=50, color='blue', alpha=0.7, edgecolor='black')
        ax8.axvline(np.mean(q_errors_deg),          color='red',   ls='--', lw=2,
                    label=f'Mean: {np.mean(q_errors_deg):.3f}°')
        ax8.axvline(np.percentile(q_errors_deg, 95), color='green', ls=':', lw=2,
                    label=f'95th pct: {np.percentile(q_errors_deg, 95):.3f}°')
        ax8.set(xlabel='Error (µrad)', ylabel='Frequency', title='Error Distribution')
        ax8.legend(); ax8.grid(True, alpha=0.3)

        # --- Plot 9: Cumulative Average Error ---
        ax9 = plt.subplot(3, 3, 9)
        cumulative_error = np.cumsum(q_errors_deg) / np.arange(1, self.N + 1)
        ax9.plot(self.timestamps, cumulative_error, 'b-', lw=1.5)
        ax9.set(xlabel='Time (s)', ylabel='Cumulative Mean Error (degrees)', title='Cumulative Average Error')
        ax9.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        if show:
            plt.show()
        plt.close()

    def print_summary_report(self, metrics: AccuracyMetrics):
        """Print a formatted accuracy summary."""
        print("=" * 60)
        print("MEKF ACCURACY EVALUATION SUMMARY")
        print("=" * 60)
        print(f"\n📊 ATTITUDE ESTIMATION (Quaternion)")
        print(f"   Mean Error: {metrics.mean_q_error * 1e6:.2f} µrad")
        print(f"   Std Dev:    {metrics.std_q_error  * 1e6:.2f} µrad")
        print(f"   Max Error:  {metrics.max_q_error  * 1e6:.2f} µrad")
        print(f"   RMSE:       {metrics.rmse_q       * 1e6:.2f} µrad")
        print(f"\n🔄 ANGULAR VELOCITY")
        print(f"   RMSE:       {metrics.rmse_omega   * 1e6:.2f} µrad")
        if metrics.mean_bias_error > 0:
            print(f"\n🔧 GYROSCOPE BIAS ESTIMATION")
            print(f"   Mean Error: {metrics.mean_bias_error * 1e6:.2f} µrad")
            print(f"   Std Dev:    {metrics.std_bias_error  * 1e6:.2f} µrad")
        if metrics.mean_innovation > 0:
            print(f"\n📈 FILTER INNOVATION")
            print(f"   Mean Magnitude: {metrics.mean_innovation * 1e6:.2f} µrad")
        print("\n" + "=" * 60)
        if metrics.mean_q_error < 0.0001:
            print("✅ EXCELLENT: Sub-100 µrad accuracy achieved")
        elif metrics.mean_q_error < 0.001:
            print("✅ GOOD: Sub-milliradian accuracy achieved")
        elif metrics.mean_q_error < 0.01:
            print("⚠️  ACCEPTABLE: Moderate accuracy")
        else:
            print("❌ NEEDS IMPROVEMENT: Consider tuning noise parameters")
        print("=" * 60)


def evaluate_mekf_accuracy(
    result: BatchResult,
    q_true: np.ndarray,
    omega_true: np.ndarray,
    bias_true: Optional[np.ndarray] = None,
    save_plot: Optional[str] = "mekf_accuracy_comparison.png",
    show_plot: bool = True
) -> AccuracyMetrics:
    """
    Complete evaluation pipeline: compute metrics, print report, generate plots.

    Args:
        result:     BatchResult from process_batch()
        q_true:     Ground truth quaternions (N, 4)
        omega_true: Ground truth angular velocities (N, 3)
        bias_true:  Ground truth gyro bias (N, 3) — optional
        save_plot:  File path for plot output (None to skip)
        show_plot:  Whether to display the plot interactively
    """
    N = len(result.timestamps)

    omega_estimated = np.array([
        QuaternionMath.extract_omega_from_qdot(result.q_corrected[i], result.q_dot_corrected[i])
        for i in range(N)
    ])

    comparator = MEKFComparator(result.timestamps)

    metrics = comparator.calculate_metrics(
        q_true=q_true,
        q_corrected=result.q_corrected,
        q_predicted=result.q_predicted,
        omega_true=omega_true,
        omega_estimated=omega_estimated,
        bias_true=bias_true,
        bias_estimated=result.biases if result.biases.size > 0 else None,
        innovations=result.innovations
    )

    comparator.print_summary_report(metrics)
    comparator.plot_comparison(
        q_true=q_true,
        q_corrected=result.q_corrected,
        q_predicted=result.q_predicted,
        omega_true=omega_true,
        omega_estimated=omega_estimated,
        bias_true=bias_true,
        bias_estimated=result.biases if result.biases.size > 0 else None,
        innovations=result.innovations,
        save_path=save_plot,
        show=show_plot
    )

    return metrics


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    datadir   = os.path.join(os.path.dirname(__file__), 'output_data/tables/rocketlab_march_quatpred')
    true_file = os.path.join(datadir, 'true_quat_rocketlab_march.csv')

    print("=" * 80)
    print("LOADING DATA")
    print("=" * 80)

    df        = pd.read_csv(true_file)
    q_raw     = df.iloc[:, 1:5].values.copy()   # [q_w, q_x, q_y, q_z]
    q_dot_raw = df.iloc[:, 5:9].values.copy()   # [q_w_dot, q_x_dot, q_y_dot, q_z_dot]
    timestamps = df.iloc[:, 0].values.copy()

    N = q_raw.shape[0]
    print(f"✓ Loaded {N} samples")

    # Add calibrated noise to simulate real sensor measurements
    np.random.seed(42)
    q_meas_noise = 0.0001
    q_dot_noise  = 0.00001

    measurements = []
    for i in range(N):
        q_meas = QuaternionMath.normalize(q_raw[i] + np.random.randn(4) * q_meas_noise)
        q_dot_meas = q_dot_raw[i] + np.random.randn(4) * q_dot_noise
        measurements.append(AttitudeMeasurement(
            timestamp=timestamps[i],
            q=q_meas,
            q_dot=q_dot_meas,
            q_covariance=np.eye(4) * q_meas_noise**2
        ))

    print("\n" + "=" * 80)
    print("RUNNING MEKF WITH HIGH-ORDER INTEGRATION")
    print("=" * 80)

    mekf = MultiplicativeEKF_HighOrder(
        q_init=q_raw[0],
        estimate_bias=False,
        process_noise_scale=1e-10,
        measurement_noise_scale=1e-8,
        bias_process_noise=0.0
    )

    result = mekf.process_batch(measurements)
    print(f"✓ Batch processing complete: {len(result.timestamps)} samples")

    print("\n" + "=" * 80)
    print("EVALUATING ACCURACY")
    print("=" * 80)

    omega_true = np.array([
        QuaternionMath.extract_omega_from_qdot(q_raw[i], q_dot_raw[i]) for i in range(N)
    ])

    metrics = evaluate_mekf_accuracy(
        result=result,
        q_true=q_raw,
        omega_true=omega_true,
        bias_true=None,
        save_plot="mekf_high_order_accuracy.png",
        show_plot=False
    )

    print(f"\n{'='*80}")
    print("ACCURACY METRICS (ALL IN µrads)")
    print(f"{'='*80}")
    print(f"Mean Error:      {metrics.mean_q_error  * 1e6:.2f} µrad")
    print(f"Std Dev:         {metrics.std_q_error   * 1e6:.2f} µrad")
    print(f"Max Error:       {metrics.max_q_error   * 1e6:.2f} µrad")
    print(f"RMSE:            {metrics.rmse_q        * 1e6:.2f} µrad")
    print(f"RMSE Omega:      {metrics.rmse_omega    * 1e6:.2f} µrad")
    print(f"Innovation Mean: {metrics.mean_innovation * 1e6:.2f} µrad")

    if metrics.mean_q_error < 0.0001:
        print("\n✅ TARGET ACHIEVED: < 100 µrad accuracy!")
    else:
        print(f"\n⚠️  Above 100 µrad target (got {metrics.mean_q_error * 1e6:.2f} µrad)")
    print("=" * 80)