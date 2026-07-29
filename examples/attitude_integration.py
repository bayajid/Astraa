import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import quaternion_slerp_squad as quat_slerp

# ===============================================================
# Inputs (Your exact values)
# ===============================================================
rpy0_deg = np.array([5.0, 10.0, 45.0])   # roll, pitch, yaw (3-2-1)
omega0   = np.array([0.03983717, -0.03983717,  0.03983717])
alpha    = np.array([0.04832422, -0.04832422,  0.04832422])

T  = 100.0      # simulation duration (s)
dt = 0.01       # RK4 step
N  = int(T/dt)

# ===============================================================
# Quaternion utilities (scalar-first)
# ===============================================================
def quat_from_euler321_SF(rpy_deg):
    """ Euler 3-2-1 (zyx intrinsic) → scalar-first quaternion.
        Input: rpy_deg  shape (3,) or (N,3) in degrees
    """
    # q_xyzw = R.from_euler("zyx", rpy_deg, degrees=True).as_quat()
    # return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])
    rpy_deg = np.asarray(rpy_deg, dtype=float)
    if rpy_deg.shape == (3,):
        single = True
        rpy_deg = rpy_deg[None, :]
    else:
        single = False

    q_xyzw = R.from_euler("zyx", rpy_deg, degrees=True).as_quat()
    q_wxyz = q_xyzw[..., [3, 0, 1, 2]]  # scalar last → scalar first

    return q_wxyz[0] if single else q_wxyz

def quat_to_euler321_SF(q):
    """
        Scalar-first quaternion [w, x, y, z] → Euler 3-2-1 (ZYX) in degrees.
        Input: q  shape (4,) or (N,4)
    """
    q = np.asarray(q, dtype=float)
    if q.shape == (4,):
        single = True
        q = q[None, :]
    else:
        single = False

    # q is [w,x,y,z] → reorder to SciPy [x,y,z,w]
    q_xyzw = q[..., [1, 2, 3, 0]]

    euler = R.from_quat(q_xyzw).as_euler("zyx", degrees=True)

    return euler[0] if single else euler

def quat_derivative(q, omega):
    """Quaternion derivative q̇ = 0.5 q ⊗ [0,ω]  (scalar-first)."""
    w, x, y, z = q
    ox, oy, oz = omega
    return 0.5 * np.array([
        -x*ox - y*oy - z*oz,
         w*ox + y*oz - z*oy,
         w*oy - x*oz + z*ox,
         w*oz + x*oy - y*ox
    ])

def rk4_step(q, t, dt):
    """One RK4 step for quaternion integration."""
    w1 = omega0 + alpha*t
    k1 = quat_derivative(q, w1)

    w2 = omega0 + alpha*(t + 0.5*dt)
    k2 = quat_derivative(q + 0.5*dt*k1, w2)

    k3 = quat_derivative(q + 0.5*dt*k2, w2)

    w4 = omega0 + alpha*(t + dt)
    k4 = quat_derivative(q + dt*k3, w4)

    q_new = q + dt*(k1 + 2*k2 + 2*k3 + k4)/6
    return q_new / np.linalg.norm(q_new)

# ===============================================================
# Closed-form analytic solution (exact)
# ===============================================================
def analytic_quaternion(t):
    """
    Analytic solution for:
       omega(t) = omega0 + alpha*t
    Angle rotated: θ(t) = ω0*t + ½ α t²
    Rotation axis direction changes (because alpha ≠ parallel to omega0),
    so we integrate the rotation vector exactly.
    """
    theta = omega0*t + 0.5*alpha*t*t   # integrated angular position (vector)

    angle = np.linalg.norm(theta)
    if angle < 1e-12:
        return q0.copy()
    axis = theta / angle

    q_xyzw = R.from_rotvec(axis*angle).as_quat()
    q_rot = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])

    # Total quaternion = q0 ⊗ q_rot
    w0,x0,y0,z0 = q0
    wr,xr,yr,zr = q_rot

    q = np.array([
        w0*wr - x0*xr - y0*yr - z0*zr,
        w0*xr + x0*wr + y0*zr - z0*yr,
        w0*yr - x0*zr + y0*wr + z0*xr,
        w0*zr + x0*yr - y0*xr + z0*wr
    ])
    return q / np.linalg.norm(q)

# ===============================================================
# Run RK4 Simulation
# ===============================================================
#q0 = quat_from_euler321_SF(rpy0_deg)
q0 = quat_slerp.quat_from_euler_deg(rpy0_deg[0], rpy0_deg[1], rpy0_deg[2])

q_hist = np.zeros((N+1,4))
rpy_hist = np.zeros((N+1,3))
omega_hist = np.zeros((N+1,3))

q = q0.copy()
t = 0.0

q_hist[0] = q
rpy_hist[0] = quat_to_euler321_SF(q)
omega_hist[0] = omega0

for i in range(1, N+1):
    q = rk4_step(q, t, dt)
    t += dt
    q_hist[i] = q
    rpy_hist[i] = quat_to_euler321_SF(q)
    omega_hist[i] = omega0 + alpha*t

# ===============================================================
# Analytic final quaternion
# ===============================================================
q_exact = analytic_quaternion(T)
rpy_exact = quat_to_euler321_SF(q_exact)

rpy_quat_slerp = [quat_slerp.euler_from_quat(q) for q in q_hist]

print("\nFinal RPY (RK4):", rpy_hist[-1])
print("Final RPY (analytic):", rpy_exact)
print("RK4 error (deg):", rpy_hist[-1] - rpy_exact)

# ===============================================================
# PLOTS
# ===============================================================

time = np.linspace(0,T,N+1)

# ----- RPY -----
plt.figure(figsize=(10,6))
plt.plot(time, rpy_hist[:,0], label="Roll  RK4")
plt.plot(time, rpy_hist[:,1], label="Pitch RK4")
plt.plot(time, rpy_hist[:,2], label="Yaw   RK4")
plt.axhline(rpy_exact[0], color='r', linestyle='--', label="Roll exact")
plt.axhline(rpy_exact[1], color='g', linestyle='--', label="Pitch exact")
plt.axhline(rpy_exact[2], color='b', linestyle='--', label="Yaw exact")
plt.xlabel("Time (s)")
plt.ylabel("RPY (deg)")
plt.title("Euler Angles 3-2-1 (ZYX) — RK4 vs Analytic")
plt.legend()
plt.grid(True)

# ----- Angular velocity -----
plt.figure(figsize=(10,6))
plt.plot(time, omega_hist[:,0], label="ωx")
plt.plot(time, omega_hist[:,1], label="ωy")
plt.plot(time, omega_hist[:,2], label="ωz")
plt.xlabel("Time (s)")
plt.ylabel("Angular velocity (rad/s)")
plt.title("Angular Velocity History")
plt.legend()
plt.grid(True)

# ----- Quaternion elements -----
plt.figure(figsize=(10,6))
plt.plot(time, q_hist[:,0], label="w")
plt.plot(time, q_hist[:,1], label="x")
plt.plot(time, q_hist[:,2], label="y")
plt.plot(time, q_hist[:,3], label="z")
plt.xlabel("Time (s)")
plt.ylabel("Quaternion components")
plt.title("Quaternion (Scalar-First) History")
plt.legend()
plt.grid(True)

# ----- RK4 vs analytic error -----
err_rpy = rpy_hist - rpy_exact
plt.figure(figsize=(10,6))
plt.plot(time, err_rpy[:,0], label="Roll error")
plt.plot(time, err_rpy[:,1], label="Pitch error")
plt.plot(time, err_rpy[:,2], label="Yaw error")
plt.xlabel("Time (s)")
plt.ylabel("Error (deg)")
plt.title("RK4 − Analytic Error in Euler 3-2-1")
plt.legend()
plt.grid(True)

plt.show()
