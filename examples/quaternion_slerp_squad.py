import numpy as np
from scipy.spatial.transform import Rotation as R

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

def quat_angle_error(q_est, q_ref):
    """
    Smallest angular error between two quaternions (in microradians)
    Handles double cover automatically (takes shortest arc)
    
    Parameters
    ----------
    q_est, q_ref : array-like (N, 4) or (4,)
        Estimated and reference quaternions [w, x, y, z]
        
    Returns
    -------
    angle_urad : ndarray (N,)
        Angular error in microradians
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

# ====================== INTERPOLATORS ======================
 
def make_slerp_interpolator(times, quats, qdots=None, sign_swap: bool = False):
    times = np.array(times)
    quats = normalize(np.array(quats))
    omegas = None
    if qdots is not None:
        qdots = np.array(qdots)
        omegas = np.array([angular_velocity_from_qdot(q, qd) for q, qd in zip(quats, qdots)])

    # Global sign consistency (optional)
    if sign_swap:
        for i in range(1, len(quats)):
            if np.dot(quats[i-1], quats[i]) < 0:
                quats[i:] = -quats[i:]
                if omegas is not None:
                    omegas[i:] = -omegas[i:]

    def interp(tq):
        tq = np.atleast_1d(tq)
        result = []
        for t in tq:
            if t <= times[0]:
                dt = t - times[0]
                q = quat_multiply(quats[0], quat_exp(omegas[0] * dt)) if omegas is not None else quats[0]
            elif t >= times[-1]:
                dt = t - times[-1]
                q = quat_multiply(quats[-1], quat_exp(omegas[-1] * dt)) if omegas is not None else quats[-1]
            else:
                i = np.searchsorted(times, t) - 1
                i = np.clip(i, 0, len(times)-2)
                tau = (t - times[i]) / (times[i+1] - times[i])

                q0 = quats[i]
                q1 = quats[i+1]

                # ←←← CRITICAL: local short-path correction (prevents 720° spin)
                if np.dot(q0, q1) < 0:
                    q1 = -q1

                q = slerp(q0, q1, tau)

                # Optional: if you have valid omegas, you can do constant-velocity extrapolation
                # within segment for C¹ continuity — but pure SLERP is already excellent
                # and safer when qdot is noisy

            result.append(normalize(q))
        return np.array(result)
    return interp

def make_hermite_interpolator(times, quats, qdots, sign_swap:bool = False):
    times = np.array(times)
    quats = normalize(np.array(quats))
    qdots = np.array(qdots)
    omegas = np.array([angular_velocity_from_qdot(q, qd) for q, qd in zip(quats, qdots)])

    # Sign consistency
    # if sign_swap:
    #     for i in range(1, len(quats)):
    #         if np.dot(quats[i-1], quats[i]) < 0:
    #             quats[i:] = -quats[i:]
    #             qdots[i:] = -qdots[i:]
    #             omegas[i:] = -omegas[i:]

    def interp(tq):
        tq = np.atleast_1d(tq)
        out = []
        for t in tq:
            if t <= times[0]:
                delta_t = t - times[0]
                delta_q = quat_exp(omegas[0] * delta_t)
                q = quat_multiply(quats[0], delta_q)
            elif t >= times[-1]:
                delta_t = t - times[-1]
                delta_q = quat_exp(omegas[-1] * delta_t)
                q = quat_multiply(quats[-1], delta_q)
            else:
                i = np.searchsorted(times, t) - 1
                i = np.clip(i, 0, len(times)-2)
                t0, t1 = times[i], times[i+1]
                tau = (t - t0) / (t1 - t0)
                tau2, tau3 = tau**2, tau**3

                q0 = quats[i]
                q1 = quats[i+1].copy()                # ← important: copy!
                # if np.dot(q0, q1) < 0:                 # ← LOCAL short-path fix
                #     q1 = -q1

                # ── Force short arc EVERY segment ────────────────
                dot = np.dot(q0, q1)
                if dot < 0.0:
                    q1 = -q1
                    # dot = -dot   # no longer needed              


                # Cubic Hermite in log-space
                h00 = 2*tau3 - 3*tau2 + 1
                h10 = tau3 - 2*tau2 + tau
                h01 = -2*tau3 + 3*tau2
                h11 = tau3 - tau2

                q0_inv = quat_conjugate(q0)                      # q0 is already the right one
                log_rel = quat_log(quat_multiply(q0_inv, q1))    # ← use q1 (locally corrected)
                v0 = omegas[i] * (t1 - t0)
                v1 = omegas[i+1] * (t1 - t0)

                log_interp = h00*0 + h10*v0 + h01*log_rel + h11*v1
                q_interp = quat_multiply(quats[i], quat_exp(log_interp))
                q = normalize(q_interp)
            out.append(q)
        return np.array(out)
    return interp

def make_cubic_spline_interpolator(times, quats, qdots=None, sign_swap:bool = False):
    """
    Cubic spline interpolation of quaternions in log-space (position-only, no velocity tangents used).
    Uses scipy CubicSpline on the 3D log vectors.
    qdots parameter kept for compatibility but ignored.
    """
    times = np.array(times)
    quats = normalize(np.array(quats))
    
    # We do NOT use qdots / omegas for this version → pure position-based spline
    
    # Optional global sign fix (usually better to keep off and rely on local correction)
    # if sign_swap:
    #     for i in range(1, len(quats)):
    #         if np.dot(quats[i-1], quats[i]) < 0:
    #             quats[i:] = -quats[i:]

    # Precompute log differences relative to FIRST keyframe (cumulative log)
    # This is one common way; alternative is per-segment but spline needs global parameter
    log_positions = np.zeros((len(times), 3))
    cumulative_q = quats[0].copy()
    log_positions[0] = np.zeros(3)
    
    for i in range(1, len(times)):
        q_prev = cumulative_q
        q_curr = quats[i].copy()
        if np.dot(q_prev, q_curr) < 0:
            q_curr = -q_curr
        delta_log = quat_log(quat_multiply(quat_conjugate(q_prev), q_curr))
        log_positions[i] = log_positions[i-1] + delta_log
        cumulative_q = quat_multiply(q_prev, quat_exp(delta_log))  # update cumulative

    # Create one cubic spline per axis of the log vector
    from scipy.interpolate import CubicSpline
    splines = [CubicSpline(times, log_positions[:, j], bc_type='natural') for j in range(3)]

    def interp(tq):
        tq = np.atleast_1d(tq)
        out = []
        for t in tq:
            if t <= times[0]:
                # Extrapolate using first segment velocity
                delta_t = t - times[0]
                # Approximate velocity from first spline derivative
                v_approx = np.array([spl(0, nu=1) for spl in splines])
                log_delta = v_approx * delta_t
                q = quat_multiply(quats[0], quat_exp(log_delta))
            elif t >= times[-1]:
                delta_t = t - times[-1]
                v_approx = np.array([spl(times[-1], nu=1) for spl in splines])
                log_delta = v_approx * delta_t
                q = quat_multiply(quats[-1], quat_exp(log_delta))
            else:
                # Evaluate spline at t
                log_interp = np.array([spl(t) for spl in splines])
                # The spline is cumulative → we exponentiate relative to q[0]
                q_interp = quat_multiply(quats[0], quat_exp(log_interp))
                q = normalize(q_interp)
            out.append(q)
        return np.array(out)
    return interp

def make_squad_interpolator(times, quats, qdots=None, sign_swap=False):

    times = np.asarray(times)
    quats = normalize(np.asarray(quats))

    # --------------------------------------------------
    # Sign consistency
    # --------------------------------------------------
    for i in range(1, len(quats)):
        if np.dot(quats[i-1], quats[i]) < 0:
            quats[i] *= -1.0

    n = len(quats)

    # --------------------------------------------------
    # Compute Squad control quaternions
    # --------------------------------------------------
    controls = np.zeros_like(quats)

    controls[0] = quats[0]
    controls[-1] = quats[-1]

    for i in range(1, n - 1):

        q_im1 = quats[i - 1]
        q_i   = quats[i]
        q_ip1 = quats[i + 1]

        term1 = quat_log(
            quat_multiply(
                quat_conjugate(q_i),
                q_im1
            )
        )

        term2 = quat_log(
            quat_multiply(
                quat_conjugate(q_i),
                q_ip1
            )
        )

        tangent = -0.25 * (term1 + term2)

        controls[i] = normalize(
            quat_multiply(
                q_i,
                quat_exp(tangent)
            )
        )

    # --------------------------------------------------
    # Interpolator
    # --------------------------------------------------
    def interp(tq):

        tq = np.atleast_1d(tq)

        out = []

        for t in tq:

            if t <= times[0]:
                out.append(quats[0])
                continue

            if t >= times[-1]:
                out.append(quats[-1])
                continue

            i = np.searchsorted(times, t) - 1
            i = np.clip(i, 0, n - 2)

            tau = (
                (t - times[i])
                /
                (times[i+1] - times[i])
            )

            q0 = quats[i]
            q1 = quats[i+1]

            s0 = controls[i]
            s1 = controls[i+1]

            q = squad(q0, q1, s0, s1, tau)

            out.append(normalize(q))

        return np.asarray(out)

    return interp

def make_aocs_exponential_predictor(t, q, qdot, normalize_output=True, max_step=0.01):
    """
    Returns an attitude predictor callable.
    Fast spacecraft attitude predictor using quaternion exponential propagation.

    Input
    -----
    t:
        Time array (N,)
    q:
        Quaternion samples (N,4), scalar first [w,x,y,z]
    qdot:
        Quaternion derivatives (N,4)

    Output
    ------
    predictor(t_eval):
        Quaternion prediction array (M,4)

    Model
    -----
    Uses:

        q_dot = 0.5 * q ⊗ [0, omega]

    and propagates using:

        q(t+dt) = q(t) ⊗ exp(0.5*omega*dt)

    Angular acceleration is estimated from consecutive omega samples.
    """

    # -----------------------------
    # Prepare inputs
    # -----------------------------
    t = np.asarray(t, dtype=float)
    q = np.asarray(q, dtype=float)
    qdot = np.asarray(qdot, dtype=float)

    if len(q) < 2:
        raise ValueError("Need at least two quaternion samples.")

    if len(t) != len(q) or len(q) != len(qdot):
        raise ValueError("t, q, and qdot must contain the same number of samples.")

    if np.any(np.diff(t) <= 0):
        raise ValueError("t must be strictly increasing.")

    # Normalize all input quaternions
    q = np.asarray([normalize(qi) for qi in q])


    # -----------------------------
    # Compute angular velocity once
    # -----------------------------
    omega = np.asarray([angular_velocity_from_qdot(qi, qdi)for qi, qdi in zip(q, qdot)])

    # Estimate angular acceleration
    alpha = np.zeros_like(omega)
    dt_key = np.diff(t)
    alpha[1:] = (omega[1:] - omega[:-1]) / dt_key[:, None]


    # -----------------------------
    # RK4 propagation step
    # -----------------------------
    if max_step <= 0:
        raise ValueError("max_step must be positive.")

    def quaternion_rhs(q_state, omega_state):
        """q_dot = 0.5 * q ⊗ [0, omega] for a body-frame omega."""
        return 0.5 * quat_multiply(q_state, np.array([0.0, *omega_state]))

    def propagate_quaternion(q0, omega0, alpha0, dt):
        """Integrate a constant-angular-acceleration model with RK4."""
        n_steps = max(1, int(np.ceil(abs(dt) / max_step)))
        h = dt / n_steps
        q_state = q0.copy()
        elapsed = 0.0

        for _ in range(n_steps):
            omega_1 = omega0 + alpha0 * elapsed
            omega_2 = omega0 + alpha0 * (elapsed + 0.5 * h)
            omega_4 = omega0 + alpha0 * (elapsed + h)

            k1 = quaternion_rhs(q_state, omega_1)
            k2 = quaternion_rhs(normalize(q_state + 0.5 * h * k1), omega_2)
            k3 = quaternion_rhs(normalize(q_state + 0.5 * h * k2), omega_2)
            k4 = quaternion_rhs(normalize(q_state + h * k3), omega_4)

            q_state = normalize(
                q_state + (h / 6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)
            )
            elapsed += h

        return q_state



    # -----------------------------
    # Predictor function
    # -----------------------------
    def predictor(t_eval):

        t_eval = np.atleast_1d(t_eval)
        q_out = np.empty((len(t_eval),4))

        # Find keyframe index once
        indices = np.searchsorted(t,t_eval,side="right" ) - 1

        for k, te in enumerate(t_eval):
            # Before first sample
            if te <= t[0]:
                q_out[k] = q[0]
                continue

            i = np.clip( indices[k],0,len(t)-1)
            dt = te - t[i]

            q_pred = propagate_quaternion(q[i],omega[i],alpha[i],dt)
            if normalize_output:
                q_pred = normalize(q_pred)

            q_out[k] = q_pred
        return q_out


    return predictor

def make_aocs_taylor_predict(t, q, qdot, normalize_output=True):
    """
    Returns an AOCS Taylor quaternion predictor callable.

    Usage
    -----
    predictor = aocs_taylor_predict(t, q, qdot)
    q_pred = predictor(t_eval)


    Inputs
    ------
    t:
        Quaternion timestamps (N,)

    q:
        Quaternion samples (N,4)
        [w,x,y,z]

    qdot:
        Quaternion derivatives (N,4)

    normalize_output:
        Normalize predicted quaternions


    Model
    -----
    Second-order Taylor expansion:

        q(t+dt) =
            q1
          + qdot1*dt
          + 0.5*qddot*dt^2


    where:

        qddot =
            (qdot1-qdot0)/(t1-t0)

    """

    # -----------------------------
    # Prepare inputs
    # -----------------------------

    t = np.asarray(t, dtype=float)
    q = np.asarray(q, dtype=float)
    qdot = np.asarray(qdot, dtype=float)


    if len(q) < 2:
        raise ValueError(
            "Need at least two quaternion samples."
        )

    if len(t) != len(q) or len(q) != len(qdot):
        raise ValueError(
            "t, q, and qdot must have same length."
        )

    if np.any(np.diff(t) <= 0):
        raise ValueError(
            "t must be strictly increasing."
        )


    # Normalize input quaternions
    q = np.asarray([
        qi / np.linalg.norm(qi)
        for qi in q
    ])


    # -----------------------------
    # Pre-compute quaternion acceleration
    # -----------------------------

    qddot = np.zeros_like(qdot)

    qddot[1:] = (
        qdot[1:] - qdot[:-1]
    ) / np.diff(t)[:, None]


    # -----------------------------
    # Predictor function
    # -----------------------------

    def predictor(t_eval):

        t_eval = np.atleast_1d(t_eval)

        q_out = np.zeros(
            (len(t_eval),4)
        )


        # Find corresponding keyframe
        indices = np.searchsorted(
            t,
            t_eval,
            side="right"
        ) - 1


        for k, te in enumerate(t_eval):

            # Before first sample
            if te <= t[0]:
                q_out[k] = q[0]
                continue


            i = np.clip(
                indices[k],
                1,
                len(t)-1
            )


            # propagate from latest keyframe
            dt = te - t[i]


            q_pred = (
                q[i]
                + qdot[i]*dt
                + 0.5*qddot[i]*dt*dt
            )


            if normalize_output:
                norm = np.linalg.norm(q_pred)

                if norm > 1e-12:
                    q_pred /= norm
                else:
                    q_pred = q[i]


            q_out[k] = q_pred


        return q_out


    return predictor
# ====================== MAIN COMPARISON ==============

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

def benchmark_methods(times, quats, qdots, method_names, keyframe_stride):
    """Compare methods by interpolating sparse CSV keyframes at every CSV timestamp."""
    if keyframe_stride < 1:
        raise ValueError("keyframe_stride must be at least 1.")

    key_indices = np.arange(0, len(times), keyframe_stride)
    if key_indices[-1] != len(times) - 1:
        key_indices = np.append(key_indices, len(times) - 1)
    if len(key_indices) < 2:
        raise ValueError("Choose a smaller keyframe stride so that at least two keyframes remain.")
    
    key_times, key_quats, key_qdots = times[key_indices], quats[key_indices], qdots[key_indices]
    results = {}
    for name in method_names:
        interpolator = METHODS[name](key_times, key_quats, key_qdots)
        predicted = interpolator(times)
        results[name] = quat_angle_error(predicted, quats)
    return key_indices, results

def _parse_methods(values):
    selected = list(METHODS) if values == ["all"] else values
    unknown = sorted(set(selected).difference(METHODS))
    if unknown:
        raise ValueError(f"Unknown method(s): {', '.join(unknown)}. Choose from: {', '.join(METHODS)}")
    return list(dict.fromkeys(selected))

if __name__ == "__main__":
    import numpy as np
    from scipy.spatial.transform import Rotation as R
    import argparse
    from pathlib import Path
    import matplotlib
    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    """
    python quaternion_slerp_squad.py --methods aocs-exponential --plot plot.png --no-show /home/bkhan/Documents/Git/astropynaric/examples/output_data/tables/rocketlab_march_quatpred/true_quat_rocketlab_march.csv

    """

    def fix_sign_ambiguity(q):
        q = q.copy()

    # ====================== CSV BENCHMARK RUNNER ==============

    METHODS = {
        "slerp": make_slerp_interpolator,
        "hermite": make_hermite_interpolator,
        "cubic-spline": make_cubic_spline_interpolator,
        "squad": make_squad_interpolator,
        "aocs-taylor": make_aocs_taylor_predict,
        "aocs-exponential": make_aocs_exponential_predictor,
    }

    parser = argparse.ArgumentParser(
        description="Compare quaternion interpolation methods against an ASTRAA true-quaternion CSV."
    )
    parser.add_argument("truth_csv", type=Path, help="ASTRAA CSV containing time,q_w,q_x,q_y,q_z and optional q_*_dot columns.")
    parser.add_argument(
        "--methods", nargs="+", default=["all"],
        help=f"Methods to run (default: all): {', '.join(METHODS)}",
    )
    parser.add_argument(
        "--keyframe-stride", type=int, default=20,
        help="Use every Nth truth row as an interpolation keyframe (default: 20).",
    )
    parser.add_argument("--plot", type=Path, help="Save the error plot to this file.")
    parser.add_argument("--no-show", action="store_true", help="Do not open the plot window.")
    args = parser.parse_args()

    selected_methods = _parse_methods(args.methods)
    times, quats, qdots = load_truth_csv(args.truth_csv)
    key_indices, errors_by_method = benchmark_methods(
        times, quats, qdots, selected_methods, args.keyframe_stride)
    print(f"Truth samples: {len(times)} | keyframes: {len(key_indices)} | stride: {args.keyframe_stride}")
    print("Method                 RMS (urad)     Max (urad)    Mean (urad)")
    for name, errors in errors_by_method.items():
        print(f"{name:20s} {np.sqrt(np.mean(errors**2)):13.6g} {np.max(errors):14.6g} {np.mean(errors):14.6g}")

    if args.plot or not args.no_show:
        import matplotlib.pyplot as plt

        for name, errors in errors_by_method.items():
            plt.plot(times, errors, label=name, linewidth=1.5)
        plt.scatter(times[key_indices], np.zeros(len(key_indices)), marker="|", color="black", label="keyframes")
        plt.xlabel("Time (s)")
        plt.ylabel("Attitude error (microrad)")
        plt.title("Quaternion interpolation error against ASTRAA truth CSV")
        plt.legend()
        plt.tight_layout()
        if args.plot:
            plt.savefig(args.plot, dpi=200)
            print(f"Saved plot: {args.plot}")
            plt.show()
        if not args.no_show:
            plt.show()



    # quats = [
    #     np.array([1.0, 0.0, 0.0, 0.0]),
    #     np.array([0.707, 0.707, 0.0, 0.0]),
    #     np.array([0.0, 1.0, 0.0, 0.0])
    # ]
    # quat_dots = [
    #     np.array([0.0, 0.1, 0.0, 0.0]),
    #     np.array([0.0, 0.2, 0.1, 0.0]),
    #     np.array([0.0, 0.3, 0.0, 0.0])
    # ]

    # def truth_quaternion(t):
    #     angle = np.pi/2 * t/1.0   # rad

    #     return np.array([
    #         np.cos(angle/2),
            
            
    #         np.sin(angle/2),
    #         0.0,
    #         0.0           
            
    #     ])
    
    

  