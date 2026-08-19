# ==================================================
# File: Untitled-1
# Author: Bayajid Khan
# Created: 2026-08-17
# Description: 
# ==================================================
import numpy as np

def fix_sign_ambiguity(q):
    q = q.copy()
    for i in range(1, len(q)):
        if np.dot(q[i], q[i-1]) < 0:
            q[i:] = -q[i:]      # flip everything from here onward
            break               # usually only one big reset in a dataset
    return q

def normalize(q):
    """Normalize quaternion(s) safely."""
    q = np.asarray(q)
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    return np.where(norm > 1e-12, q / norm, q)

def quat_inverse(q):
    """
    Inverse of unit quaternion: q⁻¹ = conjugate for unit quats
    """
    q = np.asarray(q)
    q_inv = q.copy()
    q_inv[..., 1:] *= -1
    return q_inv

def slerp(q1, q2, t):
    """Robust SLERP with short-path correction."""
    q1 = normalize(q1)
    q2 = normalize(q2)
    dot = np.clip(np.sum(q1 * q2), -1.0, 1.0)
    if dot < 0.0:
        q2 = -q2
        dot = -dot
    if dot > 0.9995:
        return normalize(q1 + t * (q2 - q1))
    theta = np.arccos(dot)
    return normalize((np.sin((1 - t) * theta) * q1 + np.sin(t * theta) * q2) / np.sin(theta))

def squad(q0, q1, s0, s1, t):
    return slerp(
        slerp(q0, q1, t),
        slerp(s0, s1, t),
        2 * t * (1 - t)
    )

def quat_multiply(q1, q2):
    """
    Hamilton product q1 ⊗ q2
    q1, q2: (..., 4) arrays, order [w, x, y, z]
    """
    q1 = np.asarray(q1)
    q2 = np.asarray(q2)
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]

    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2

    return np.stack([w, x, y, z], axis=-1)

def quat_angle_error(q_est: np.ndarray, q_ref: np.ndarray)-> float:
    """
    Smallest angular error between two quaternions (in µrads)
    Handles double cover automatically (takes shortest arc)
    
    Parameters
    ----------
    q_est, q_ref : array-like (N, 4) or (4,)
        Estimated and reference quaternions [w, x, y, z]
        
    Returns
    -------
    angle_urad : ndarray (N,)
        Angular error in µrad
    """
    q_est = np.asarray(q_est)
    q_ref = np.asarray(q_ref)

    # Normalize just in case
    q_est = q_est / np.linalg.norm(q_est, axis=-1, keepdims=True)
    q_ref = q_ref / np.linalg.norm(q_ref, axis=-1, keepdims=True)

    # Relative quaternion: q_err = q_ref⁻¹ ⊗ q_est
    q_err = quat_multiply(quat_inverse(q_ref), q_est)

    # Take absolute value of scalar part to handle double cover
    cos_theta = np.abs(q_err[..., 0])
    cos_theta = np.clip(cos_theta, -1.0, 1.0)  # numerical safety

    # θ = 2 * arccos(|w|)
    angle_rad = 2.0 * np.arccos(cos_theta)

    return angle_rad * 1e6  # microradians

def quat_conjugate(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])

def omega_to_quat_delta(omega, dt):
    """
    Quaternion increment from angular velocity.

    dq = exp(0.5*omega*dt)

    """

    angle = np.linalg.norm(omega)*dt

    if angle < 1e-12:
        return np.array([1.0,0,0,0])

    axis = omega/np.linalg.norm(omega)

    return np.array([
        np.cos(angle/2),
        *(np.sin(angle/2)*axis)
    ])

def quat_exp(v):
    theta = np.linalg.norm(v)
    if theta < 1e-10:
        return np.array([1.0, 0, 0, 0])
    axis = v / theta
    return np.array([np.cos(theta), *(np.sin(theta) * axis)])

def quat_log(q):
    q = normalize(q)
    v = q[1:]
    vnorm = np.linalg.norm(v)
    if vnorm < 1e-10:
        return np.zeros(3)
    return np.arctan2(vnorm, q[0]) * v / vnorm

def angular_velocity_from_qdot(q, qdot):
    
    """
    Compute body-frame angular velocity from quaternion derivative.
    
    Args:
        q     : quaternion [w, x, y, z]
        qdot  : quaternion derivative dq/dt [w, x, y, z]
    
    Returns:
        omega : angular velocity [ωx, ωy, ωz] in body frame
        ω = 2 * q̄ ⊗ q̇ (vector part)
    """
    q_conj = quat_conjugate(q)
    omega_quat = 2 * quat_multiply(q_conj, qdot)  # treat qdot as pure vector quat
    return omega_quat[1:]

def quat_from_euler_deg(roll_deg, pitch_deg, yaw_deg):
    """Convert Euler angles (deg, 3-2-1 sequence) to unit quaternion [w,x,y,z]"""
    roll  = np.deg2rad(roll_deg)
    pitch = np.deg2rad(pitch_deg)
    yaw   = np.deg2rad(yaw_deg)
    
    cr, sr = np.cos(roll/2),  np.sin(roll/2)
    cp, sp = np.cos(pitch/2), np.sin(pitch/2)
    cy, sy = np.cos(yaw/2),   np.sin(yaw/2)
    
    w = cr*cp*cy + sr*sp*sy
    x = sr*cp*cy - cr*sp*sy
    y = cr*sp*cy + sr*cp*sy
    z = cr*cp*sy - sr*sp*cy
    return np.array([w, x, y, z])

def quaternion_derivative(q, omega):
    """
    Compute dq/dt = 0.5 * q ⊗ [0, ω]
    Args:
      q:     quaternion [w, x, y, z], shape (..., 4)
      omega: angular velocity [ωx, ωy, ωz], shape (..., 3)
    """
    q = np.asarray(q, dtype=float)
    omega = np.asarray(omega, dtype=float)
    
    # Pad omega to [0, ωx, ωy, ωz]
    omega_quat = np.concatenate([np.zeros_like(omega[...,:1]), omega], axis=-1)
    
    dq = 0.5 * quat_multiply(q, omega_quat)
    return dq

def euler_from_quat(quat, seq='zyx', quat_type='wxyz', degrees=True):
    """
    Convert quaternion(s) to Euler angles for any intrinsic rotation sequence.

    Parameters
    ----------
    quat : array_like, shape (...,4)
        Quaternion(s).
    seq : str
        Intrinsic rotation sequence, e.g., 'xyz', 'zyx', etc.
    quat_type : str
        Format of input quaternion: 'xyzw' (x,y,z,w) or 'wxyz' (w,x,y,z)
    degrees : bool
        Return angles in degrees if True, radians otherwise.

    Returns
    -------
    euler : ndarray, shape (...,3)
        Euler angles corresponding to the given sequence.
        -  'xyz' → returns [roll, pitch, yaw]   (x, y, z order)
        -  'zyx' → returns [yaw, pitch, roll]   (z, y, x order)
    """
    
    q = np.asarray(quat, dtype=float)
    original_shape = q.shape
    
    if q.ndim == 1:
        q = q.reshape(1, 4)
    
    if q.shape[-1] != 4:
        raise ValueError(f"Last dimension must be 4, got shape {original_shape}")
    
    # Convert to w, x, y, z internally
    if quat_type == 'xyzw':
        q = q[..., [3, 0, 1, 2]]  # [x,y,z,w] -> [w,x,y,z]
    elif quat_type != 'wxyz':
        raise ValueError("quat_type must be 'xyzw' or 'wxyz'")
    
    # Normalize
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    
    # Compute Euler angles based on sequence
    euler = np.empty(q.shape[:-1] + (3,))
    
    if seq.lower() == 'xyz':
        # Roll (x), Pitch (y), Yaw (z)
        sinr_cosp = 2*(w*x + y*z)
        cosr_cosp = 1 - 2*(x*x + y*y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        sinp = 2*(w*y - z*x)
        sinp = np.clip(sinp, -1.0, 1.0)
        pitch = np.arcsin(sinp)

        siny_cosp = 2*(w*z + x*y)
        cosy_cosp = 1 - 2*(y*y + z*z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        
        euler[..., 0] = roll
        euler[..., 1] = pitch
        euler[..., 2] = yaw
        
    elif seq.lower() == 'zyx':
        # Yaw (z), Pitch (y), Roll (x)
        siny_cosp = 2*(w*z + x*y)
        cosy_cosp = 1 - 2*(y*y + z*z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        sinp = 2*(w*y - z*x)
        sinp = np.clip(sinp, -1.0, 1.0)
        pitch = np.arcsin(sinp)

        sinr_cosp = 2*(w*x + y*z)
        cosr_cosp = 1 - 2*(x*x + y*y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        euler[..., 0] = yaw
        euler[..., 1] = pitch
        euler[..., 2] = roll
        
    else:
        raise ValueError("Currently only 'xyz' or 'zyx' sequences are supported")
    
    if degrees:
        euler = np.rad2deg(euler)
    
    if original_shape == (4,):
        return euler[0]
    return euler

def euler_rates_321(rpy_deg, omega_body_deg, deg=True):
    """
    Compute Euler angle rates for 3-2-1 (ZYX) sequence.
    
    Input:  rpy_deg = [roll, pitch, yaw] in degrees
            omega_body_deg = [ωx, ωy, ωz] in deg/s (body frame)
    Output: [roll_dot, pitch_dot, yaw_dot] in deg/s
    """
    rpy = np.asarray(rpy_deg, dtype=float)
    omega = np.asarray(omega_body_deg, dtype=float)
    
    if deg:
        rpy_rad = np.deg2rad(rpy)
        omega_rad = np.deg2rad(omega)
    else:
        rpy_rad = rpy
        omega_rad = omega
    
    # Handle both 1D and 2D inputs
    if rpy_rad.ndim == 1:
        r, p, y = rpy_rad
        wx, wy, wz = omega_rad
    else:
        r, p, y = rpy_rad[..., 0], rpy_rad[..., 1], rpy_rad[..., 2]
        wx, wy, wz = omega_rad[..., 0], omega_rad[..., 1], omega_rad[..., 2]
    
    sr, cr = np.sin(r), np.cos(r)
    sp, cp = np.sin(p), np.cos(p)
    tp = np.tan(p)
    
    roll_dot = wx + wy*sr*tp + wz*cr*tp
    pitch_dot = wy*cr - wz*sr
    yaw_dot = (wy*sr + wz*cr) / cp
    
    rates = np.stack([roll_dot, pitch_dot, yaw_dot], axis=-1)
    
    return np.rad2deg(rates) if deg else rates

def rk4_step(q, omega, alpha, jerk, dt):
    """
    RK4 integration step for attitude + angular rate + angular acceleration.
    
    Dynamics:
      dq/dt = 0.5 * q ⊗ [0, ω]
      domega/dt = alpha
      dalpha/dt = jerk
    """

    q       = np.asarray(q, dtype=float)
    omega   = np.asarray(omega, dtype=float)
    alpha   = np.asarray(alpha, dtype=float)
    jerk    = np.asarray(jerk, dtype=float)

    # Handle non-batch case
    is_batch = q.ndim == 2
    if not is_batch:
        q = q[np.newaxis, :]
        omega = omega[np.newaxis, :]
        alpha = alpha[np.newaxis, :]
        jerk = jerk[np.newaxis, :]

    # ---------------------
    # k1
    # ---------------------
    dq1     = quaternion_derivative(q, omega)
    domega1 = alpha
    dalpha1 = jerk

    # ---------------------
    # k2
    # ---------------------
    q2     = (q + 0.5 * dt * dq1)
    omega2 = omega + 0.5 * dt * domega1
    alpha2 = alpha + 0.5 * dt * dalpha1

    dq2     = quaternion_derivative(q2, omega2)
    domega2 = alpha2
    dalpha2 = jerk

    # ---------------------
    # k3
    # ---------------------
    q3     = (q + 0.5 * dt * dq2)
    omega3 = omega + 0.5 * dt * domega2
    alpha3 = alpha + 0.5 * dt * dalpha2

    dq3     = quaternion_derivative(q3, omega3)
    domega3 = alpha3
    dalpha3 = jerk

    # ---------------------
    # k4
    # ---------------------
    q4     = (q + dt * dq3)
    omega4 = omega + dt * domega3
    alpha4 = alpha + dt * dalpha3

    dq4     = quaternion_derivative(q4, omega4)
    domega4 = alpha4
    dalpha4 = jerk

    # ---------------------
    # Combine increments
    # ---------------------
    q_new = q + (dt/6) * (dq1 + 2*dq2 + 2*dq3 + dq4)
    q_new = normalize(q_new)

    omega_new = omega + (dt/6) * (domega1 + 2*domega2 + 2*domega3 + domega4)
    alpha_new = alpha + (dt/6) * (dalpha1 + 2*dalpha2 + 2*dalpha3 + dalpha4)

    # Quaternion derivative at the new state
    q_dot_new = quaternion_derivative(q_new, omega_new)

    # Return without batch dimension
    if not is_batch:
        q_new      = q_new[0]
        omega_new  = omega_new[0]
        alpha_new  = alpha_new[0]
        q_dot_new  = q_dot_new[0]

    return q_new, q_dot_new, omega_new, alpha_new

def quat_angle(q):
    """Return rotation angle of quaternion q (single rotation)."""
    q = normalize(q)  # normalize just in case
    w = np.clip(q[0], -1.0, 1.0)  # clip to avoid numerical errors
    angle = 2 * np.arccos(w)  # radians
    return np.rad2deg(angle)

def angular_velocity(q1, q2, dt):
    """
    Compute angular velocity (rad/s) from two quaternions.
    
    Args:
        q1, q2: Quaternions in [w, x, y, z] format
        dt: Time delta in seconds
    
    Returns:
        Angular velocity vector in rad/s
    """
    if dt <= 0:
        raise ValueError("dt must be positive")
    
    # Ensure shortest path (handle quaternion sign ambiguity)
    if np.dot(q1, q2) < 0:
        q2 = -q2
    
    # Convert to scipy Rotation (expects [x, y, z, w])
    r1 = R.from_quat([q1[1], q2[2], q1[3], q1[0]])
    r2 = R.from_quat([q2[1], q2[2], q2[3], q2[0]])
    
    # Relative rotation from q1 to q2
    r = r2 * r1.inv()
    
    # Extract rotation vector and divide by dt
    rotvec = r.as_rotvec()
    
    return rotvec / dt

def integrate_high_order_aocs(q, omega, alpha, jerk, dt,  norm_threshold=1e-6):
    """
    Inputs:
        q_init:    (4,) Initial quaternion
        omega_arr: (N, 3) Angular velocity [rad/s]
        alpha_arr: (N, 3) Angular acceleration [rad/s^2]
        jerk_arr:  (N, 3) Angular jerk [rad/s^3]
        dt:        Scalar time step
    Returns:
        q_hist, q_dot_hist
    """
    # 1. Kinematic Update (Predicting the next velocity/acceleration)
    alpha_next = alpha + jerk * dt
    omega_next = omega + alpha * dt + 0.5 * jerk * (dt**2)
    
    # 2. Rotation Vector (dphi) for the current interval
    dphi = omega * dt + 0.5 * alpha * (dt**2) + (1/6) * jerk * (dt**3)
    dx, dy, dz = dphi

    # 3. Algorithm 5: Quaternion Integration
    d2 = dx**2 + dy**2 + dz**2
    d4 = d2**2
    s = 0.5 - (d2 / 48.0) + (d4 / 3840.0)
    c = - (d2 / 8.0) + (d4 / 384.0)
    
    sx, sy, sz = s * dx, s * dy, s * dz

    # dq calculation
    dq = np.array([
        c*q[0] - sx*q[1] - sy*q[2] - sz*q[3],
        c*q[1] + sx*q[0] + sz*q[3] - sy*q[2],
        c*q[2] + sy*q[0] + sx*q[3] - sz*q[1],
        c*q[3] + sz*q[0] + sy*q[1] - sx*q[2]
    ])

    # 4. State Update
    q_next = q + dq
    q_next /= np.linalg.norm(q_next) # Normalize to keep it a unit quaternion
    
    # 5. q_dot (Instantaneous derivative)
    q_dot = 0.5 * np.array([
        -q[1]*omega[0] - q[2]*omega[1] - q[3]*omega[2],
         q[0]*omega[0] + q[2]*omega[2] - q[3]*omega[1],
         q[0]*omega[1] - q[1]*omega[2] + q[3]*omega[0],
         q[0]*omega[2] + q[1]*omega[1] - q[2]*omega[0]
    ])



    norm_sq = np.dot(q_next, q_next)

    if abs(1.0 - norm_sq) > norm_threshold:
        # If the drift is significant, do a full normalization
        q_next = q_next / np.sqrt(norm_sq)
    else:
        # Linear normalization (First-order Taylor approximation of 1/sqrt(x))
        # This is MUCH faster than sqrt and keeps the quaternion "close enough"
        q_next = q_next * (1.5 - 0.5 * norm_sq)

    return q_next, q_dot, omega_next, alpha_next 

# ==========================================================
# All methods now restricted to the latest 2 (or max 3) points
# ==========================================================

def make_quadratic_interp_2pts_smarter(t, q, qdot, normalize_output=True):
    """
    Unchanged in spirit – already used only the last two points.
    Kept for completeness.
    """
    t = np.asarray(t, dtype=float)
    q = np.asarray(q, dtype=float)
    qdot = np.asarray(qdot, dtype=float)

    if len(q) < 2:
        raise ValueError("Need at least two quaternion samples.")

    # Force use of the latest two points only
    t    = t[-2:]
    q    = q[-2:]
    qdot = qdot[-2:]

    t0, t1 = t[0], t[1]
    dt = t1 - t0

    A = np.array([
        [1.0, 0.0,  0.0],
        [1.0, dt,   dt**2],
        [0.0, 1.0,  2*dt]
    ], dtype=float)

    coeff = np.zeros((3, 4))
    for j in range(4):
        rhs = np.array([q[0, j], q[1, j], qdot[1, j]])
        try:
            coeff[:, j] = np.linalg.solve(A, rhs)
        except np.linalg.LinAlgError:
            coeff[0, j] = q[0, j]
            coeff[1, j] = (q[1, j] - q[0, j]) / dt
            coeff[2, j] = 0.0

    def predictor(t_vec):
        t_vec = np.atleast_1d(np.asarray(t_vec, dtype=float))
        tau = t_vec - t0
        q_out = np.zeros((len(tau), 4))
        for j in range(4):
            a, b, c = coeff[:, j]
            q_out[:, j] = a + b*tau + c*tau**2
        if normalize_output:
            norms = np.linalg.norm(q_out, axis=1, keepdims=True)
            valid = norms[:, 0] > 1e-12
            q_out[valid] /= norms[valid]
            q_out[~valid] = q[1]
        return q_out

    return predictor

def make_slerp_interpolator(times, quats, qdots=None, sign_swap=False):
    """
    2-point constant-angular-velocity quaternion extrapolator.

    Uses:
        q0 -> q1
        omega = log(q0^-1 * q1) / dt

    Prediction:
        q(t) = q1 * exp(omega * (t - t1))
    """
    times = np.asarray(times, dtype=float)[-2:]
    quats = normalize(np.asarray(quats, dtype=float)[-2:])

    if len(times) < 2:
        raise ValueError("SLERP needs at least 2 samples.")

    q0 = quats[0].copy()
    q1 = quats[1].copy()

    # Quaternion sign consistency
    if np.dot(q0, q1) < 0.0:
        q1 = -q1

    dt = times[1] - times[0]

    if dt <= 0:
        raise ValueError("Times must be strictly increasing.")

    # Relative rotation q0 -> q1
    dq = quat_multiply(
        quat_conjugate(q0),
        q1
    )

    if dq[0] < 0.0:
        dq = -dq

    # Rotation vector over the interval
    phi = quat_log(dq)

    # Constant angular velocity
    omega = phi / dt

    def interp(tq):
        tq = np.atleast_1d(np.asarray(tq, dtype=float))
        out = np.empty((len(tq), 4))

        for k, t in enumerate(tq):

            delta_t = t - times[1]

            phi_pred = omega * delta_t

            out[k] = quat_multiply(
                q1,
                quat_exp(phi_pred)
            )

            out[k] = normalize(out[k])

        return out

    return interp


def make_hermite_interpolator(times, quats, qdots, sign_swap=False):
    """
    2-point second-order quaternion extrapolator.

    Uses quaternion derivatives at both endpoints to estimate
    angular acceleration.

    omega0 = omega(t0)
    omega1 = omega(t1)

    alpha = (omega1 - omega0) / dt

    Prediction:
        phi = omega1 * dt + 0.5 * alpha * dt^2
        q = q1 * exp(phi)
    """
    times = np.asarray(times, dtype=float)[-2:]
    quats = normalize(np.asarray(quats, dtype=float)[-2:])
    qdots = np.asarray(qdots, dtype=float)[-2:]

    if len(times) < 2:
        raise ValueError("Hermite needs at least 2 samples.")

    q0 = quats[0].copy()
    q1 = quats[1].copy()

    if np.dot(q0, q1) < 0.0:
        q1 = -q1

    dt = times[1] - times[0]

    if dt <= 0:
        raise ValueError("Times must be strictly increasing.")

    # Angular velocity from qdot
    omega0 = angular_velocity_from_qdot(q0, qdots[0])
    omega1 = angular_velocity_from_qdot(q1, qdots[1])

    # Angular acceleration
    alpha = (omega1 - omega0) / dt

    def interp(tq):
        tq = np.atleast_1d(np.asarray(tq, dtype=float))
        out = np.empty((len(tq), 4))

        for k, t in enumerate(tq):

            delta_t = t - times[1]

            phi = (
                omega1 * delta_t
                + 0.5 * alpha * delta_t**2
            )

            out[k] = quat_multiply(
                q1,
                quat_exp(phi)
            )

            out[k] = normalize(out[k])

        return out

    return interp


def make_cubic_spline_interpolator(times, quats, qdots=None, sign_swap=False):
    """
    3-point cubic extrapolator on SO(3).

    Uses the last 3 quaternion samples to estimate:
        angular velocity
        angular acceleration
        angular jerk

    Prediction:
        phi = omega*dt
            + 0.5*alpha*dt^2
            + 1/6*jerk*dt^3
    """
    times = np.asarray(times, dtype=float)
    quats = normalize(np.asarray(quats, dtype=float))

    n = len(times)

    if n < 3:
        # Fall back to the 2-point second-order predictor
        return make_hermite_interpolator(
            times[-2:],
            quats[-2:],
            np.asarray(qdots)[-2:] if qdots is not None else None,
            sign_swap
        )

    times = times[-3:]
    quats = quats[-3:]

    if qdots is not None:
        qdots = np.asarray(qdots, dtype=float)[-3:]

    # Sign consistency
    for i in range(1, 3):
        if np.dot(quats[i-1], quats[i]) < 0.0:
            quats[i] *= -1.0

    # --------------------------------------------------
    # Relative rotation vectors
    # --------------------------------------------------

    dq01 = quat_multiply(
        quat_conjugate(quats[0]),
        quats[1]
    )

    dq12 = quat_multiply(
        quat_conjugate(quats[1]),
        quats[2]
    )

    if dq01[0] < 0.0:
        dq01 = -dq01

    if dq12[0] < 0.0:
        dq12 = -dq12

    phi01 = quat_log(dq01)
    phi12 = quat_log(dq12)

    dt01 = times[1] - times[0]
    dt12 = times[2] - times[1]

    if dt01 <= 0 or dt12 <= 0:
        raise ValueError("Times must be strictly increasing.")

    # Angular velocities
    omega01 = phi01 / dt01
    omega12 = phi12 / dt12

    # --------------------------------------------------
    # Estimate endpoint angular velocity
    # --------------------------------------------------

    if qdots is not None:

        omega2 = angular_velocity_from_qdot(
            quats[2],
            qdots[2]
        )

        omega1 = angular_velocity_from_qdot(
            quats[1],
            qdots[1]
        )

        # Use measured endpoint rate but estimate
        # acceleration from the trajectory.
        alpha12 = (omega2 - omega1) / dt12

    else:

        omega2 = omega12

        # Approximate acceleration from two intervals
        alpha12 = (
            omega12 - omega01
        ) / (0.5 * (dt01 + dt12))

    # --------------------------------------------------
    # Estimate jerk
    # --------------------------------------------------

    if qdots is not None:

        omega0 = angular_velocity_from_qdot(
            quats[0],
            qdots[0]
        )

        alpha01 = (omega1 - omega0) / dt01

    else:

        alpha01 = alpha12

    jerk = (
        alpha12 - alpha01
    ) / (0.5 * (dt01 + dt12))

    def interp(tq):

        tq = np.atleast_1d(
            np.asarray(tq, dtype=float)
        )

        out = np.empty((len(tq), 4))

        q_last = quats[-1]

        for k, t in enumerate(tq):

            delta_t = t - times[-1]

            phi = (
                omega2 * delta_t
                + 0.5 * alpha12 * delta_t**2
                + (1.0 / 6.0) * jerk * delta_t**3
            )

            out[k] = quat_multiply(
                q_last,
                quat_exp(phi)
            )

            out[k] = normalize(out[k])

        return out

    return interp


def make_squad_interpolator(times, quats, qdots=None, sign_swap=False):
    """
    3-point SQUAD-style predictor.

    For extrapolation, uses the SQUAD tangent information to
    estimate angular velocity and acceleration at the final
    quaternion, then propagates beyond the final sample.

    Requires 3 samples.
    """
    times = np.asarray(times, dtype=float)[-3:]
    quats = normalize(
        np.asarray(quats, dtype=float)[-3:]
    )

    if len(times) < 3:
        return make_slerp_interpolator(
            times,
            quats,
            qdots,
            sign_swap
        )

    # --------------------------------------------------
    # Sign consistency
    # --------------------------------------------------

    for i in range(1, 3):
        if np.dot(quats[i-1], quats[i]) < 0.0:
            quats[i] *= -1.0

    q0, q1, q2 = quats

    dt01 = times[1] - times[0]
    dt12 = times[2] - times[1]

    if dt01 <= 0 or dt12 <= 0:
        raise ValueError("Times must be strictly increasing.")

    # --------------------------------------------------
    # Relative rotations
    # --------------------------------------------------

    dq01 = quat_multiply(
        quat_conjugate(q0),
        q1
    )

    dq12 = quat_multiply(
        quat_conjugate(q1),
        q2
    )

    if dq01[0] < 0:
        dq01 = -dq01

    if dq12[0] < 0:
        dq12 = -dq12

    phi01 = quat_log(dq01)
    phi12 = quat_log(dq12)

    omega01 = phi01 / dt01
    omega12 = phi12 / dt12

    # --------------------------------------------------
    # SQUAD tangent at q1
    # --------------------------------------------------

    log_10 = quat_log(
        quat_multiply(
            quat_conjugate(q1),
            q0
        )
    )

    log_12 = quat_log(
        quat_multiply(
            quat_conjugate(q1),
            q2
        )
    )

    tangent = -0.25 * (
        log_10 + log_12
    )

    # --------------------------------------------------
    # Final angular velocity
    #
    # Blend the measured/finite-difference velocity
    # with the SQUAD tangent.
    # --------------------------------------------------

    omega_squad = (
        2.0 * tangent / dt12
    )

    omega_final = 0.5 * (
        omega12 + omega_squad
    )

    # --------------------------------------------------
    # Estimate angular acceleration
    # --------------------------------------------------

    alpha = (
        omega12 - omega01
    ) / (0.5 * (dt01 + dt12))

    def interp(tq):

        tq = np.atleast_1d(
            np.asarray(tq, dtype=float)
        )

        out = np.empty((len(tq), 4))

        for k, t in enumerate(tq):

            delta_t = t - times[-1]

            phi = (
                omega_final * delta_t
                + 0.5 * alpha * delta_t**2
            )

            out[k] = quat_multiply(
                q2,
                quat_exp(phi)
            )

            out[k] = normalize(out[k])

        return out

    return out

def make_hermite_interpolator(times, quats, qdots, sign_swap: bool = False):
    """
    Restricted to the latest 2 points → classic cubic Hermite on SO(3).
    """
    times = np.asarray(times, dtype=float)[-2:]
    quats = normalize(np.asarray(quats, dtype=float)[-2:])
    qdots = np.asarray(qdots, dtype=float)[-2:]

    omegas = np.array([angular_velocity_from_qdot(q, qd) for q, qd in zip(quats, qdots)])

    def interp(tq):
        tq = np.atleast_1d(tq)
        out = []
        for t in tq:
            if t <= times[0]:
                delta_t = t - times[0]
                q = quat_multiply(quats[0], quat_exp(omegas[0] * delta_t))
            elif t >= times[-1]:
                delta_t = t - times[-1]
                q = quat_multiply(quats[-1], quat_exp(omegas[-1] * delta_t))
            else:
                t0, t1 = times[0], times[1]
                tau = (t - t0) / (t1 - t0)
                tau2, tau3 = tau**2, tau**3

                q0 = quats[0]
                q1 = quats[1].copy()
                if np.dot(q0, q1) < 0.0:
                    q1 = -q1

                h00 =  2*tau3 - 3*tau2 + 1
                h10 =    tau3 - 2*tau2 + tau
                h01 = -2*tau3 + 3*tau2
                h11 =    tau3 -   tau2

                log_rel = quat_log(quat_multiply(quat_conjugate(q0), q1))
                v0 = omegas[0] * (t1 - t0)
                v1 = omegas[1] * (t1 - t0)

                log_interp = h00*0.0 + h10*v0 + h01*log_rel + h11*v1
                q = normalize(quat_multiply(q0, quat_exp(log_interp)))
            out.append(q)
        return np.array(out)
    return interp

def make_cubic_spline_interpolator(times, quats, qdots=None, sign_swap: bool = False):
    """
    Restricted to the latest 2 (or 3 if available) points.
    With only 2 points this becomes a cubic Hermite in log-space
    (the only meaningful cubic that can be built from 2 samples).
    """
    n_use = min(3, len(times))          # use last 2 or 3
    times = np.asarray(times, dtype=float)[-n_use:]
    quats = normalize(np.asarray(quats, dtype=float)[-n_use:])
    if n_use == 2:
        return make_hermite_interpolator(
            times,
            quats,
            qdots,
            sign_swap
        )

    if qdots is not None:
        qdots = np.asarray(qdots, dtype=float)[-n_use:]
        omegas = np.array([angular_velocity_from_qdot(q, qd) for q, qd in zip(quats, qdots)])
    else:
        omegas = np.zeros((n_use, 3))
        for i in range(1, n_use):
            dq = quat_multiply(quat_conjugate(quats[i-1]), quats[i])
            if dq[0] < 0:
                dq = -dq
            omegas[i] = 2.0 * quat_log(dq) / max(times[i]-times[i-1], 1e-12)
        omegas[0] = omegas[1]

    # cumulative log relative to the oldest of the kept points
    log_pos = np.zeros((n_use, 3))
    cum = quats[0].copy()
    for i in range(1, n_use):
        q_rel = quat_multiply(quat_conjugate(cum), quats[i])
        if q_rel[0] < 0:
            q_rel = -q_rel
        log_pos[i] = log_pos[i-1] + quat_log(q_rel)
        cum = quat_multiply(cum, quat_exp(quat_log(q_rel)))

    log_vel = omegas.copy()

    def hermite(t0, t1, p0, p1, v0, v1, t):
        dt = t1 - t0
        if dt < 1e-12:
            return p0
        s = (t - t0) / dt
        s2, s3 = s*s, s*s*s
        h00 =  2*s3 - 3*s2 + 1
        h10 =    s3 - 2*s2 + s
        h01 = -2*s3 + 3*s2
        h11 =    s3 -   s2
        return h00*p0 + h10*dt*v0 + h01*p1 + h11*dt*v1

    def interp(tq):
        tq = np.atleast_1d(np.asarray(tq, dtype=float))
        out = np.zeros((len(tq), 4))

        for k, t in enumerate(tq):
            if t <= times[0]:
                log_d = log_vel[0] * (t - times[0])
                out[k] = quat_multiply(quats[0], quat_exp(log_d))
            elif t >= times[-1]:
                log_d = log_vel[-1] * (t - times[-1])
                out[k] = quat_multiply(quats[-1], quat_exp(log_d))
            else:
                # find segment among the kept points
                idx = np.searchsorted(times, t) - 1
                idx = int(np.clip(idx, 0, n_use-2))
                t0, t1 = times[idx], times[idx+1]
                p0, p1 = log_pos[idx], log_pos[idx+1]
                v0, v1 = log_vel[idx], log_vel[idx+1]

                log_t = np.array([hermite(t0, t1, p0[j], p1[j], v0[j], v1[j], t)
                                  for j in range(3)])
                out[k] = quat_multiply(quats[0], quat_exp(log_t))
            out[k] = normalize(out[k])
        return out

    return interp

def make_squad_interpolator(times, quats, qdots=None, sign_swap=False):
    """
    Restricted to the latest 3 points (SQUAD needs at least 3).
    Falls back to SLERP if only 2 points are available.
    """
    n_use = min(3, len(times))
    times = np.asarray(times, dtype=float)[-n_use:]
    quats = normalize(np.asarray(quats, dtype=float)[-n_use:])

    # sign consistency on the kept points
    for i in range(1, n_use):
        if np.dot(quats[i-1], quats[i]) < 0:
            quats[i] *= -1.0

    if n_use < 3:
        # fall back to SLERP
        return make_slerp_interpolator(times, quats, qdots, sign_swap)

    # classic SQUAD control points on the 3 samples
    controls = np.zeros_like(quats)
    controls[0] = quats[0]
    controls[-1] = quats[-1]

    for i in range(1, n_use-1):
        q_im1, q_i, q_ip1 = quats[i-1], quats[i], quats[i+1]
        term1 = quat_log(quat_multiply(quat_conjugate(q_i), q_im1))
        term2 = quat_log(quat_multiply(quat_conjugate(q_i), q_ip1))
        tangent = -0.25 * (term1 + term2)
        controls[i] = normalize(quat_multiply(q_i, quat_exp(tangent)))

    def interp(tq):
        tq = np.atleast_1d(tq)
        out = []
        for t in tq:
            if t <= times[0]:
                out.append(quats[0])
            elif t >= times[-1]:
                out.append(quats[-1])
            else:
                i = np.searchsorted(times, t) - 1
                i = np.clip(i, 0, n_use-2)
                tau = (t - times[i]) / (times[i+1] - times[i])
                q = squad(quats[i], quats[i+1], controls[i], controls[i+1], tau)
                out.append(normalize(q))
        return np.asarray(out)
    return interp

def load_truth_csv(filename):
    """Load an ASTRAA true-quaternion CSV in scalar-first ``[w, x, y, z]`` order."""
    data = np.genfromtxt(filename, delimiter=",", names=True, dtype=float)
    if data.size == 0:
        raise ValueError(f"{filename} contains no attitude samples.")
    data = np.atleast_1d(data)

    required = ("time", "q_w", "q_x", "q_y", "q_z")
    missing = [name for name in required if name not in data.dtype.names]
    if missing:
        raise ValueError(
            f"{filename} is missing required column(s): {', '.join(missing)}. "
            "Expected ASTRAA columns: time,q_w,q_x,q_y,q_z[,q_w_dot,q_x_dot,q_y_dot,q_z_dot]."
        )

    times = np.asarray(data["time"], dtype=float)
    quats = normalize(np.column_stack([data[name] for name in required[1:]]))
    if len(times) < 3 or np.any(~np.isfinite(times)) or np.any(np.diff(times) <= 0):
        raise ValueError("CSV needs at least three rows with finite, strictly increasing time values.")

    rate_columns = ("q_w_dot", "q_x_dot", "q_y_dot", "q_z_dot")
    if all(name in data.dtype.names for name in rate_columns):
        qdots = np.column_stack([data[name] for name in rate_columns])
    else:
        # Keep derivative-dependent methods usable with quaternion-only CSV files.
        qdots = np.gradient(quats, times, axis=0, edge_order=2)
        print("Quaternion-rate columns not found; estimated q_dot from the truth samples.")

    return times, quats, qdots

def benchmark_sliding_window(times, quats, qdots, method_names, window_size=2,update_rate=4.0,):
    """
    Fair comparison: every method only ever sees a sliding window
    of `window_size` points (2 or 3) and must predict the next sample(s).
    """
    results = {name: [] for name in method_names}
    t_pred_list = []
    #q_true_list = []
    # ---------------------------------------------------------
    # Truth sampling information
    # ---------------------------------------------------------
    truth_dt = np.median(np.diff(times))
    truth_rate = 1.0 / truth_dt

    # Number of truth samples between onboard updates
    stride = int(round(truth_rate / update_rate))
    if stride < 1:
        raise ValueError(
            f"Update rate {update_rate} Hz is higher than truth rate "
            f"{truth_rate:.3f} Hz."
        )

    
    print("\n========== BENCHMARK DEBUG ==========")
    print(f"Update rate       : {update_rate} Hz")
    print(f"Update interval   : {1/update_rate:.9f} s")
    print(f"Truth dt          : {truth_dt:.9f} s")
    print(f"Truth rate        : {truth_rate:.3f} Hz")
    print(f"Truth stride      : {stride}")
    print(f"Window size       : {window_size}")

    # ---------------------------------------------------------
    # Prediction samples
    # ---------------------------------------------------------
    prediction_indices = np.arange(window_size* stride,len(times),stride)

    print(f"Number predictions: {len(prediction_indices)}")
    print("First prediction indices:",prediction_indices[:5])
    
    # ---------------------------------------------------------
    # Benchmark
    # ---------------------------------------------------------
    for future_idx in prediction_indices:

        # Prediction time is EXACTLY a truth sample
        t_future = times[future_idx]
        # -----------------------------------------------------
        # Previous ONBOARD keyframes
        # -----------------------------------------------------
        window_indices = (future_idx- stride * np.arange(window_size, 0, -1))
        if window_indices[0] < 0:
            continue

        # Previous window
        # start_idx = future_idx - window_size
        # end_idx = future_idx

        t_win = times[window_indices]
        q_win = quats[window_indices]
        qdot_win = qdots[window_indices]

        q_true = quats[future_idx]
        # -----------------------------------------------------
        # Debug
        # -----------------------------------------------------
        if len(t_pred_list) < 5:

            print("\n--- Prediction debug ---")
            print(f"future_idx         : {future_idx}")
            print(f"t_future           : {t_future:.9f}")
            print(f"window_indices     : {window_indices}")
            print(f"Window times       : {t_win}")
            print(f"Last window time   : {t_win[-1]:.9f}")
            print(f"Prediction horizon : "f"{t_future - t_win[-1]:.9f} s")

        

        # -----------------------------------------------------
        # Each method
        # -----------------------------------------------------
        for name in method_names:
            predictor = METHODS[name](t_win,q_win,qdot_win)
            q_pred = predictor(t_future)[0]
            err = quat_angle_error(q_pred,q_true)

            if name == "quadratic" and len(t_pred_list) < 5:
                print(f"Quadratic error: "    f"{err:.9f} µrad")
            results[name].append(err)
        t_pred_list.append(t_future)
    # ---------------------------------------------------------
    # Convert to arrays
    # ---------------------------------------------------------
    for name in results:
        results[name] = np.asarray(results[name])

    return np.asarray(t_pred_list), results
#%%
if __name__ == "__main__":
    import numpy as np
    from scipy.spatial.transform import Rotation as R    
    import argparse
    from pathlib import Path
    # import matplotlib
    #matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    """
    python quaternion_slerp_squad.py --methods aocs-exponential --plot plot.png --no-show /home/bkhan/Documents/Git/astropynaric/examples/output_data/tables/rocketlab_march_quatpred/true_quat_rocketlab_march.csv

    """

    def fix_sign_ambiguity(q):
        q = q.copy()

    # ====================== CSV BENCHMARK RUNNER ==============

    METHODS = {
        "slerp"             : make_slerp_interpolator,
        "hermite"           : make_hermite_interpolator,
        "cubic-spline"      : make_cubic_spline_interpolator,
        "squad"             : make_squad_interpolator,        
        "quadratic"         : make_quadratic_interp_2pts_smarter,
    }

    # ================= CONFIG =================
    TRUTH_CSV=Path("/home/bkhan/Documents/Git/astropynaric/examples/output_data/tables/" \
    "rocketlab_march_quatpred/true_quat_rocketlab_march.csv")
    METHODS_TO_RUN= list(METHODS)  # or ["quad"]
    PLOT=Path("plot.png")
    NO_SHOW=False
    UPDATE_RATE = 4  #Hz 
    #Choose window size (2 or 3)
    WINDOW = 2          # or 3    
    # ===========================================

    selected_methods=METHODS_TO_RUN
    times,quats,qdots=load_truth_csv(TRUTH_CSV)
    KEYFRAME_STRIDE= int(1/(UPDATE_RATE*(times[1]-times[0]))) #200

   

    selected = ["quadratic", "slerp","hermite", "cubic-spline", "squad"]

    t_pred, errors_by_method = benchmark_sliding_window(
        times, quats, qdots, selected, window_size=WINDOW, update_rate=UPDATE_RATE,)

    # Then print / plot exactly as before
    print("Method                 RMS (µrad)     Max (µrad)    Mean (µrad)")
    for name, errors in errors_by_method.items():
        print(f"{name:20s} {np.sqrt(np.mean(errors**2)):13.6g} "
            f"{np.max(errors):14.6g} {np.mean(errors):14.6g}")
        
    # ============================================================
    # PLOTTING for sliding-window results
    # ============================================================
    if PLOT or not NO_SHOW:

        fig, ax = plt.subplots(figsize=(12, 7))

        for name, errors in errors_by_method.items():
            ax.plot(t_pred, errors, label=name, linewidth=1.4)

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Attitude error (µrad)")
        ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.3)

        ax.set_title(
            f"Sliding-window quaternion prediction error\n"
            f"Window size = {WINDOW}  |  Update rate : {UPDATE_RATE}, Truth rate ≈ {1/(times[1]-times[0]):.1f} Hz",
            fontsize=12, pad=12
        )

        ax.legend(loc="upper right")

        # ----- Summary table -----
        table_data = []
        for name, errors in errors_by_method.items():
            table_data.append([
                name,
                f"{np.sqrt(np.mean(errors**2)):.5g}",
                f"{np.max(errors):.5g}",
                f"{np.mean(errors):.5g}"
            ])

        table = ax.table(
            cellText=table_data,
            colLabels=["Method", "RMS (µrad)", "Max (µrad)", "Mean (µrad)"],
            cellLoc="center",
            colLoc="center",
            bbox=[0.0, -0.42, 1.0, 0.28]
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.1)

        fig.subplots_adjust(left=0.09, right=0.98, top=0.88, bottom=0.32)

        if PLOT:
            plt.savefig(PLOT, dpi=200, bbox_inches="tight")
            print(f"Saved plot → {PLOT}")

        if not NO_SHOW:
            plt.show()

        plt.close()
    
# %%
