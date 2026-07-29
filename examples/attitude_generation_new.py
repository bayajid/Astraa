#%%
import numpy as np
from numpy.linalg import norm
import quaternion_slerp_squad as quat_slerp
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

# ==============================================================
# USER INPUT
# ==============================================================

# Initial attitude
roll0_deg, pitch0_deg, yaw0_deg = 5.0, 10.0, 45.0

# Initial angular velocity (deg/s)
omega_mag = 0.069
omega0_deg_s = np.array([1, -1, 1]) * omega_mag / np.sqrt(3)
omega0_deg_s /= np.sqrt(3)

# Initial angular acceleration (deg/s²)
alpha_mag = 0.0837
alpha_deg_s2 = np.array([1, -1, 1]) * alpha_mag / np.sqrt(3)
alpha_deg_s2 /= np.sqrt(3)

# Angular jerk (deg/s³)
om_ddot_mag = 0.5756
omega_ddot_vec_deg_s3 = np.array([1,-1,1]) * om_ddot_mag / np.sqrt(3)

# Convert to radians
omega0 = np.deg2rad(omega0_deg_s)
alpha0 = np.deg2rad(alpha_deg_s2)
jerk0  = np.deg2rad(omega_ddot_vec_deg_s3)

# Time
dt = 0.01
t_final = 100.0
times = np.arange(0, t_final+dt, dt)
N = len(times)

# ==============================================================
# STATE VECTORS
# ==============================================================

q_history     = np.zeros((N, 4))
q_dot_history = np.zeros((N, 4))
omega_history = np.zeros((N, 3))
alpha_history = np.zeros((N, 3))
jerk_history  = np.zeros((N, 3))

# Initial conditions
q = quat_slerp.quat_from_euler_deg(roll0_deg, pitch0_deg, yaw0_deg)
q = quat_slerp.normalize(q)

omega = omega0.copy()
alpha = alpha0.copy()
jerk = jerk0.copy()

q_history[0] = q
q_dot_history[0] = quat_slerp.quaternion_derivative(q, omega)
omega_history[0] = omega
alpha_history[0] = alpha


# State machine flags
acc_go_mode = True
hold_active = False
t_hold_end = None
dt_hold1 = 1.5
dt_hold2 = 0.5
t_hold_switch = 50.0 

# ==============================================================
# MAIN LOOP (RK4 + jerk logic)
# ==============================================================
settings = "ROCKET_LAB"
# settings = "OTHER"

if settings == "ROCKET_LAB":
    for ii in range(1,N):
        t = times[ii]
        if ii == 0:
            jerk_history[0] = jerk
            continue
        # ----------------------------------------------------------
        # 1) JERK / HOLD STATE MACHINE  (correct RocketLab logic)
        # ----------------------------------------------------------
        acc_mag = norm(alpha)   ### FIXED ###
        alpha_limit = np.deg2rad(alpha_mag)        

        if acc_mag >= alpha_limit:  # ← FIXED: >= not 
            if t > t_hold_switch:
                hold_duration = dt_hold2
            else:
                hold_duration = dt_hold1
                
            if acc_go_mode:
                hold_active = True
                
            if acc_go_mode and hold_active:
                t_hold_end = t + hold_duration
                acc_go_mode = False
            
            if t <= t_hold_end:
                jerk = np.zeros(3)
            else:
                jerk = -np.sign(alpha[0]) * jerk0  # ← FIXED: use jerk0, not unit vector
                acc_go_mode = True
                hold_active = False
        #else:
        #    jerk = np.zeros(3) 

        # ----------------------------------------------------------
        # 2) RK4 PROPAGATION
        # ----------------------------------------------------------
        q, q_dot, omega, alpha = quat_slerp.rk4_step(q, omega, alpha, jerk, dt)

        # ----------------------------------------------------------
        # 3) SAVE STATES
        # ----------------------------------------------------------
        q_history[ii]       = q
        q_dot_history[ii]   = q_dot
        omega_history[ii]   = omega
        alpha_history[ii]   = alpha
        jerk_history[ii]    = jerk   

else:
    # Reset initial conditions
    alpha_initial = alpha0.copy()
    alpha_swapped  = -alpha_initial # rad/s²
    jerk_current   = np.zeros(3)  # jerk = alpha_dot

    swap_time = 40.0
    jerk_flag = 0#True

    # RK4 loop
    for ii, t in enumerate(times[1:], start=1):
        # Swap angular acceleration at t >= 40 s
        
        if t >= swap_time and jerk_flag:
            alpha_current = - alpha0
        else:
            alpha_current =  alpha0

        # RK4 step (jerk = 0)
        q, q_dot, omega, _ =  quat_slerp.rk4_step(q, omega, alpha_current, np.zeros(3), dt)

        # Store
        q_history[ii]       = q
        q_dot_history[ii]   = q_dot
        omega_history[ii]   = omega
        alpha_history[ii]   = alpha_current
        jerk_history[ii]    = jerk_current  

ea_all      = np.vstack([quat_slerp.euler_from_quat(q) for q in q_history]) #quat_slerp.euler_from_quat(q_history)
ea_all_scipy = R.from_quat(q_history[:, [1, 2, 3, 0]]).as_euler('xyz', degrees=True) # Same thing using Scipy
ea_dot_all  = quat_slerp.euler_rates_321(ea_all, omega_history)
       
# ==============================================================
# RESULTS
# ==============================================================

final_rpy_deg = (ea_all[-1])
final_omega_deg_s = np.rad2deg(omega_history[-1])
final_alpha_deg_s2 = np.rad2deg(alpha_history[-1])
omega_magnitude = np.rad2deg(norm(omega_history, axis=1))
max_omega_deg_s = np.max(omega_magnitude)
angle = 2 * np.arccos(q_history[-1,0])

def quat_angle(q):
    """Return rotation angle [deg] of quaternion q (single rotation)."""
    q = quat_slerp.normalize(q)  # normalize just in case
    w = np.clip(q[0], -1.0, 1.0)  # clip to avoid numerical errors
    angle = 2 * np.arccos(w)  # radians
    return np.rad2deg(angle)

def quat_relative(q1, q2):
    """Return quaternion representing rotation from q1 to q2."""
    # Conjugate of q1
    q1_conj = q1.copy()
    q1_conj[1:] *= -1
    # q_rel = q2 * q1_conj
    w1,x1,y1,z1 = q2
    w2,x2,y2,z2 = q1_conj
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return np.array([w,x,y,z])

# 1) Net rotation from initial to final
net_angle = quat_angle(q_history[-1])
print(f"Net rotation from start to end: {net_angle:.3f} deg")

# 2) Cumulative rotation along the path
cumulative_angle = 0.0
for i in range(1, len(q_history)):
    q_rel = quat_relative(q_history[i-1], q_history[i])
    cumulative_angle += quat_angle(q_rel)

print(f"Cumulative rotation along path: {cumulative_angle:.3f} deg")

print("="*60)
print("FINAL RESULTS (t = 100.00 s)")
print("="*60)
print(f"Final Roll, Pitch, Yaw       : {final_rpy_deg[0]:8.3f}, {final_rpy_deg[1]:8.3f}, {final_rpy_deg[2]:8.3f}  [deg]")
print(f"Final angular velocity       : {final_omega_deg_s}  [deg/s]")
print(f"MAX angular velocity (mag)   : {max_omega_deg_s:8.3f}  [deg/s]")
print(f"Final angular acceleration   : {final_alpha_deg_s2}  [deg/s²]")
print(f"Quaternion norm error (max)  : {np.max(np.abs(norm(q_history, axis=1)-1)):.2e}")


print("Total rotation magnitude:", np.rad2deg(angle))

print("="*60)
#%%
# ==============================================================
# PLOTS
# ==============================================================

# ==============================================================
# PLOTTING SECTION (Complete, Cleaned, Recovered)
# ==============================================================
N = q_history.shape[0]

# Preallocate rotation increment array
rotation_increment_deg = np.zeros(N)
cumulative_rotation_deg = np.zeros(N)

for i in range(1, N):
    # Relative rotation quaternion between steps
    q_prev = q_history[i-1]
    q_curr = q_history[i]
    
    # Compute relative quaternion: q_rel = q_prev^* ⊗ q_curr
    w, x, y, z = q_prev
    q_conj = np.array([w, -x, -y, -z])
    
    # Quaternion multiplication: q_rel = q_conj ⊗ q_curr
    w1, x1, y1, z1 = q_conj
    w2, x2, y2, z2 = q_curr
    w_rel = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x_rel = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y_rel = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z_rel = w1*z2 + x1*y2 - y1*x2 + z1*w2
    q_rel = np.array([w_rel, x_rel, y_rel, z_rel])
    
    # Ensure quaternion is normalized
    q_rel = quat_slerp.normalize(q_rel)
    
    # Angle of rotation for this step
    delta_theta = 2 * np.arccos(np.clip(q_rel[0], -1.0, 1.0))  # radians
    rotation_increment_deg[i] = np.rad2deg(delta_theta)
    
    # Cumulative rotation
    cumulative_rotation_deg[i] = cumulative_rotation_deg[i-1] + rotation_increment_deg[i]

print("Total rotation magnitude (deg):", cumulative_rotation_deg[-1])

# Optional: plot

plt.figure(figsize=(10,4))
plt.plot(times, cumulative_rotation_deg)
plt.title("Cumulative Rotation Magnitude Over Time")
plt.xlabel("Time [s]")
plt.ylabel("Cumulative rotation [deg]")
plt.grid(True)



# ---- Convert quaternion history to Euler angles ----
euler_deg_history = ea_all# np.array([quat_slerp.euler_from_quat(q) for q in q_history])

# ---- Angular velocity magnitude ----
omega_mag_deg_s = np.rad2deg(np.linalg.norm(omega_history, axis=1))

# ---- Angular acceleration magnitude ----
alpha_mag_deg_s2 = np.rad2deg(np.linalg.norm(alpha_history, axis=1))

# ---- Jerk magnitude ----
jerk_mag_deg_s3 = np.rad2deg(np.linalg.norm(jerk_history, axis=1))


# ==============================================================
# 1. Euler Angles
# ==============================================================
# fig, axes = plt.subplots(3, 2, figsize=(10, 8), sharex=True)
# # Row 1
# axes[0, 0].plot(times, euler_deg_history[:,0], label='Roll')
# axes[0, 0].set_ylabel("R[deg]")
# axes[0, 1].plot(times, np.rad2deg(omega_history[:,0]), label='ωx')
# axes[0, 1].set_ylabel("R dot[deg/s]")


# # Row 2
# axes[1, 0].plot(times, euler_deg_history[:,1], label='Pitch')
# axes[1, 0].set_ylabel("P[deg]")
# axes[1, 1].plot(times, np.rad2deg(omega_history[:,1]), label='ωy')
# axes[1, 1].set_ylabel("P dot[deg/s]")


# # Row 3
# axes[2, 0].plot(times, euler_deg_history[:,2], label='Yaw')
# axes[2, 0].set_ylabel("Y[deg]")
# axes[2, 1].plot(times, np.rad2deg(omega_history[:,2]), label='ωy')
# axes[2, 1].set_ylabel("Y dot[deg/s]")

# # Labels
# for i in range(3):
#     for j in range(2):
#         axes[i, j].grid(True)  
# plt.tight_layout()
# plt.show()

plt.figure(figsize=(12,6))
plt.plot(times, euler_deg_history[:,0], label='Roll')
plt.plot(times, euler_deg_history[:,1], label='Pitch')
plt.plot(times, euler_deg_history[:,2], label='Yaw')
plt.title("Euler Angles [deg]")
plt.xlabel("Time [s]")
plt.ylabel("[deg]")
plt.grid(True)
plt.legend()
plt.tight_layout()



# ==============================================================
# 2. Angular Rates ω
# ==============================================================

# plt.figure(figsize=(12,6))
# plt.plot(times, np.rad2deg(omega_history[:,0]), label='ωx')
# plt.plot(times, np.rad2deg(omega_history[:,1]), label='ωy')
# plt.plot(times, np.rad2deg(omega_history[:,2]), label='ωz')
# plt.plot(times, omega_mag_deg_s, 'k--', label='|ω|')
# plt.title("Angular Velocity [deg/s]")
# plt.xlabel("Time [s]")
# plt.ylabel("[deg/s]")
# plt.grid(True)
# plt.legend()
# plt.tight_layout()



# ==============================================================
# 3. Angular Acceleration α
# ==============================================================

plt.figure(figsize=(12,6))
plt.plot(times, np.rad2deg(alpha_history[:,0]), label='αx')
plt.plot(times, np.rad2deg(alpha_history[:,1]), label='αy')
plt.plot(times, np.rad2deg(alpha_history[:,2]), label='αz')
plt.plot(times, alpha_mag_deg_s2, 'k--', label='|α|')
plt.title("Angular Acceleration [deg/s²]")
plt.xlabel("Time [s]")
plt.ylabel("[deg/s²]")
plt.grid(True)
plt.legend()
plt.tight_layout()



# ==============================================================
# 4. Angular Jerk (ω̈)
# ==============================================================

plt.figure(figsize=(12,6))
plt.plot(times, np.rad2deg(jerk_history[:,0]), label='jerk x')
plt.plot(times, np.rad2deg(jerk_history[:,1]), label='jerk y')
plt.plot(times, np.rad2deg(jerk_history[:,2]), label='jerk z')
plt.plot(times, jerk_mag_deg_s3, 'k--', label='|jerk|')
plt.title("Angular Jerk [deg/s³]")
plt.xlabel("Time [s]")
plt.ylabel("[deg/s³]")
plt.grid(True)
plt.legend()
plt.tight_layout()



# ==============================================================
# 5. Quaternion Components
# ==============================================================

plt.figure(figsize=(12,6))
plt.plot(times, q_history[:,0], label='w')
plt.plot(times, q_history[:,1], label='x')
plt.plot(times, q_history[:,2], label='y')
plt.plot(times, q_history[:,3], label='z')
plt.title("Quaternion Components")
plt.xlabel("Time [s]")
plt.ylabel("Value")
plt.grid(True)
plt.legend()
plt.tight_layout()


# ==============================================================
# 6. Angular Velocity Magnitude
# ==============================================================

plt.figure(figsize=(12,5))
plt.plot(times, omega_mag_deg_s)
plt.title("Angular Velocity Magnitude |ω| [deg/s]")
plt.xlabel("Time [s]")
plt.ylabel("[deg/s]")
plt.grid(True)
plt.tight_layout()
plt.show()


# %%
