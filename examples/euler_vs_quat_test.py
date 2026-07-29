# import numpy as np
# import quaternion_slerp_squad as quat_slerp

# # Initial conditions
# roll0, pitch0, yaw0 = 5.0, 10.0, 45.0  # deg
# omega0 = np.deg2rad([0.04, -0.04, 0.04])  # rad/s
# alpha0 = np.deg2rad([0.048, -0.048, 0.048])  # rad/s²

# dt = 0.01
# t_final = 100.0
# N = int(t_final / dt)

# # ==============================================================
# # METHOD 1: EULER ANGLE INTEGRATION
# # ==============================================================
# ea = np.array([roll0, pitch0, yaw0])
# omega_ea = omega0.copy()
# alpha_ea = alpha0.copy()

# for i in range(N):
#     # Compute Euler angle rates
#     # ea_rate = quat_slerp.euler_rates_321(ea, np.rad2deg(omega_ea), deg=True)
#     ea_rate = quat_slerp.euler_rates_321(ea, np.rad2deg(omega_ea), deg=True)
    
#     # Euler integration
#     ea = ea + dt * ea_rate
#     omega_ea = omega_ea + dt * alpha_ea

# print("METHOD 1: Euler Angle Integration")
# print(f"Final RPY: {ea}")
# print(f"Final omega: {np.rad2deg(omega_ea)}")

# # ==============================================================
# # METHOD 2: QUATERNION INTEGRATION (EULER METHOD, NOT RK4)
# # ==============================================================
# q = quat_slerp.quat_from_euler_deg(roll0, pitch0, yaw0)
# omega_q = omega0.copy()
# alpha_q = alpha0.copy()

# for i in range(N):
#     # Quaternion derivative
#     q_dot = quat_slerp.quaternion_derivative(q, omega_q)
    
#     # Euler integration (NOT RK4!)
#     q = q + dt * q_dot
#     q = quat_slerp.normalize(q)
#     omega_q = omega_q + dt * alpha_q

# # Convert quaternion to Euler angles
# ea_from_quat = quat_slerp.euler_from_quat(q, seq='xyz', quat_type='wxyz', degrees=True)

# print("\nMETHOD 2: Quaternion Integration (Euler)")
# print(f"Final RPY: {ea_from_quat}")
# print(f"Final omega: {np.rad2deg(omega_q)}")

# print("\nDIFFERENCE:")
# print(f"RPY diff: {ea - ea_from_quat}")

# print("\nDIFFERENCE:")
# diff = ea - ea_from_quat

# # Wrap differences to [-180, 180]
# diff = (diff + 180) % 360 - 180

# print(f"RPY diff (wrapped): {diff}")
# print(f"Omega diff: {np.rad2deg(omega_ea - omega_q)}")
# print(f"Omega diff: {np.rad2deg(omega_ea - omega_q)}")

import numpy as np
import quaternion_slerp_squad as quat_slerp

# ==============================================================
# INITIAL CONDITIONS (matching example file)
# ==============================================================
roll0, pitch0, yaw0 = 5.0, 10.0, 45.0
omega_mag = 0.069
alpha_mag = 0.0837
om_ddot_mag = 0.5756

omega0_deg = np.array([1, -1, 1]) * omega_mag / np.sqrt(3)
alpha_deg = np.array([1, -1, 1]) * alpha_mag / np.sqrt(3)
jerk_deg = np.array([1, -1, 1]) * om_ddot_mag / np.sqrt(3)

omega0 = np.deg2rad(omega0_deg)
alpha0 = np.deg2rad(alpha_deg)
jerk0 = np.deg2rad(jerk_deg)

dt = 0.01
t_final = 100.0
times = np.arange(0, t_final + dt, dt)
N = len(times)

# State machine parameters
dt_hold1 = 1.5
dt_hold2 = 0.5
t_hold_switch = 50.0
alpha_limit_rad = np.deg2rad(alpha_mag)

# ==============================================================
# METHOD 1: EULER ANGLE INTEGRATION
# ==============================================================
print("Running METHOD 1: Euler Angle Integration...")

ea = np.array([roll0, pitch0, yaw0])
omega_ea = omega0.copy()
alpha_ea = alpha0.copy()

acc_go_mode = True
hold_active = False
t_hold_end = 0.0
jerk_applied_count = 0
for i in range(1, N):
    t = times[i]
    
    # Jerk logic
    acc_mag = np.linalg.norm(alpha_ea)
    jerk_ea = np.zeros(3)
    
    if acc_mag >= alpha_limit_rad:
        if t > t_hold_switch:
            hold_duration = dt_hold2
        else:
            hold_duration = dt_hold1
        
        if acc_go_mode:
            hold_active = True
        
        if acc_go_mode and hold_active:
            t_hold_end = t + hold_duration
            acc_go_mode = False
            if i % 1000 == 0 or i < 200:
                print(f"t={t:.2f}: Entering HOLD, t_hold_end={t_hold_end:.2f}")
        
        
        if t <= t_hold_end:
            jerk_ea = np.zeros(3)
        else:
            jerk_ea = -np.sign(alpha_ea[0]) * jerk0
            jerk_applied_count += 1
            if jerk_applied_count < 50:  # print first 50 times
                print(f"t={t:.2f}: Applying jerk, alpha_mag={np.rad2deg(acc_mag):.6f}, alpha[0]={np.rad2deg(alpha_ea[0]):.6f}")

            acc_go_mode = True
            hold_active = False
         
    
    # Update omega and alpha
    omega_ea = omega_ea + dt * alpha_ea
    alpha_ea = alpha_ea + dt * jerk_ea
    
    # Euler angle rates
    ea_rate = quat_slerp.euler_rates_321(ea, np.rad2deg(omega_ea), deg=True)
    
    # Integrate Euler angles
    ea = ea + dt * ea_rate
print(f"\nJerk applied {jerk_applied_count} times out of {N} steps")
ea_method1 = ea.copy()
omega_method1 = omega_ea.copy()
alpha_method1 = alpha_ea.copy()

print(f"Final RPY: [{ea[0]:.3f}, {ea[1]:.3f}, {ea[2]:.3f}] deg")
print(f"Final omega: [{np.rad2deg(omega_ea[0]):.3f}, {np.rad2deg(omega_ea[1]):.3f}, {np.rad2deg(omega_ea[2]):.3f}] deg/s")
print(f"Final alpha: [{np.rad2deg(alpha_ea[0]):.4f}, {np.rad2deg(alpha_ea[1]):.4f}, {np.rad2deg(alpha_ea[2]):.4f}] deg/s²")

# ==============================================================
# METHOD 2: QUATERNION INTEGRATION (EULER METHOD)
# ==============================================================
print("\nRunning METHOD 2: Quaternion Integration...")

q = quat_slerp.quat_from_euler_deg(roll0, pitch0, yaw0)
q = quat_slerp.normalize(q)
omega_q = omega0.copy()
alpha_q = alpha0.copy()

acc_go_mode = True
hold_active = False
t_hold_end = 0.0

for i in range(1, N):
    t = times[i]
    
    # Jerk logic (identical)
    acc_mag = np.linalg.norm(alpha_q)
    jerk_q = np.zeros(3)
    
    if acc_mag >= alpha_limit_rad:
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
            jerk_q = np.zeros(3)
        else:
            jerk_q = -np.sign(alpha_q[0]) * jerk0
            acc_go_mode = True
            hold_active = False
    
    # Update omega and alpha
    omega_q = omega_q + dt * alpha_q
    alpha_q = alpha_q + dt * jerk_q
    
    # Quaternion derivative
    q_dot = quat_slerp.quaternion_derivative(q, omega_q)
    
    # Integrate quaternion (Euler method)
    q = q + dt * q_dot
    q = quat_slerp.normalize(q)

# Convert to Euler
ea_q = quat_slerp.euler_from_quat(q, seq='xyz', quat_type='wxyz', degrees=True)

print(f"Final RPY: [{ea_q[0]:.3f}, {ea_q[1]:.3f}, {ea_q[2]:.3f}] deg")
print(f"Final omega: [{np.rad2deg(omega_q[0]):.3f}, {np.rad2deg(omega_q[1]):.3f}, {np.rad2deg(omega_q[2]):.3f}] deg/s")
print(f"Final alpha: [{np.rad2deg(alpha_q[0]):.4f}, {np.rad2deg(alpha_q[1]):.4f}, {np.rad2deg(alpha_q[2]):.4f}] deg/s²")

# ==============================================================
# COMPARISON
# ==============================================================
print("\n" + "="*60)
print("COMPARISON")
print("="*60)

diff_rpy = ea_method1 - ea_q
diff_rpy_wrapped = (diff_rpy + 180) % 360 - 180

diff_omega = omega_method1 - omega_q
diff_alpha = alpha_method1 - alpha_q

print(f"RPY difference (wrapped): [{diff_rpy_wrapped[0]:.6f}, {diff_rpy_wrapped[1]:.6f}, {diff_rpy_wrapped[2]:.6f}] deg")
print(f"Omega difference: [{np.rad2deg(diff_omega[0]):.6f}, {np.rad2deg(diff_omega[1]):.6f}, {np.rad2deg(diff_omega[2]):.6f}] deg/s")
print(f"Alpha difference: [{np.rad2deg(diff_alpha[0]):.6f}, {np.rad2deg(diff_alpha[1]):.6f}, {np.rad2deg(diff_alpha[2]):.6f}] deg/s²")

max_rpy_error = np.max(np.abs(diff_rpy_wrapped))
print(f"\nMax RPY error: {max_rpy_error:.6f} deg")

if max_rpy_error < 0.1:
    print("✓ PASS: Methods agree within 0.1 degrees")
else:
    print("✗ FAIL: Methods differ by more than 0.1 degrees")