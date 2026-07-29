
#%%
# Simulate a trajectory where the target crosses the Sun-avoidance zone
# and the pointing vector adjusts accordingly to avoid the forbidden cone.


import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Helper functions
# -----------------------------
def normalize(v):
    return v / np.linalg.norm(v)

def angle_between(v1, v2):
    return np.arccos(np.clip(np.dot(normalize(v1), normalize(v2)), -1.0, 1.0))

def axis_angle_to_quaternion(axis, theta):
    axis = normalize(axis)
    w = np.cos(theta / 2)
    xyz = axis * np.sin(theta / 2)
    return np.array([w, *xyz])

def quaternion_rotate_vector(q, v):
    w, x, y, z = q
    q_vec = np.array([x, y, z])
    uv = np.cross(q_vec, v)
    uuv = np.cross(q_vec, uv)
    return v + 2 * (w * uv + uuv)

def avoid_sun_by_quaternion(r_target, r_sun, min_angle_deg=5.1):
    r_target = normalize(r_target)
    r_sun = normalize(r_sun)
    theta = angle_between(r_target, r_sun)

    if np.degrees(theta) >= min_angle_deg:
        return r_target, "track"

    # Construct orthogonal direction to r_sun in the plane of r_target and r_sun
    v_orth = r_target - np.dot(r_target, r_sun) * r_sun
    if np.linalg.norm(v_orth) < 1e-6:
        v_orth = np.cross(r_sun, np.array([1.0, 0.0, 0.0]))
        if np.linalg.norm(v_orth) < 1e-6:
            v_orth = np.cross(r_sun, np.array([0.0, 1.0, 0.0]))
    v_orth = normalize(v_orth)

    min_angle_rad = np.radians(min_angle_deg)
    q = axis_angle_to_quaternion(v_orth, min_angle_rad - theta)
    rotated_vec = quaternion_rotate_vector(q, r_target)
    return normalize(rotated_vec), "avoid"

# -----------------------------
# # Simulate
# # -----------------------------
# n_points = 200
# t = np.linspace(0, 1, n_points)
# r_sun = normalize(np.array([0.0, 0.0, 1.0]))

# r_target_traj = []
# avoidance_traj = []
# mode_traj = []

# # Create a trajectory that passes through the forbidden cone
# for i in range(n_points):
#     angle = np.radians(10 * np.sin(2 * np.pi * t[i]))  # Oscillates between -10° and +10°
#     r_target = normalize(np.array([np.sin(angle), 0, np.cos(angle)]))

#     pointing_vec, mode = avoid_sun_by_quaternion(r_target, r_sun, min_angle_deg=7)

#     r_target_traj.append(r_target)
#     avoidance_traj.append(pointing_vec)
#     mode_traj.append(mode)

# r_target_traj = np.array(r_target_traj)
# avoidance_traj = np.array(avoidance_traj)

# # -----------------------------
# # 3D Plotting
# # -----------------------------
# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')
# ax.set_title("3D View: Target Trajectory and Sun Avoidance")

# # Plot the original target trajectory
# ax.plot(r_target_traj[:, 0], r_target_traj[:, 1], r_target_traj[:, 2], label='Original Target LOS', color='green')

# # Plot the avoidance trajectory
# ax.scatter(avoidance_traj[:, 0], avoidance_traj[:, 1], avoidance_traj[:, 2], label='Sun Avoided LOS', color='blue')

# # Plot Sun vector
# ax.quiver(0, 0, 0, r_sun[0], r_sun[1], r_sun[2], color='orange', linewidth=2, label='Sun Vector')

# # Draw forbidden cone surface (approximate with circle)
# cone_angle = np.radians(5.0)
# circle_points = 100
# theta_circle = np.linspace(0, 2 * np.pi, circle_points)
# circle_radius = np.tan(cone_angle)
# circle_x = circle_radius * np.cos(theta_circle)
# circle_y = circle_radius * np.sin(theta_circle)
# circle_z = np.ones_like(circle_x)

# # Align circle with r_sun direction
# sun_dir = normalize(r_sun)
# R = np.eye(3) if np.allclose(sun_dir, [0, 0, 1]) else np.linalg.qr(np.vstack((sun_dir, np.random.rand(3), np.random.rand(3))).T)[0]
# circle_xyz = np.vstack((circle_x, circle_y, circle_z))
# rotated_circle = R @ circle_xyz
# ax.plot(rotated_circle[0], rotated_circle[1], rotated_circle[2], color='orange', linestyle='--', label='Forbidden Zone Boundary')

# ax.set_xlabel("X")
# ax.set_ylabel("Y")
# ax.set_zlabel("Z")
# ax.set_xlim([-1, 1])
# ax.set_ylim([-1, 1])
# ax.set_zlim([-0.2, 1.2])
# ax.legend()
# ax.grid(True)
# plt.tight_layout()
# plt.show()



####################################################
#%%
# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D


# def normalize(v):
#     return v / np.linalg.norm(v)


# def angle_between(v1, v2):
#     return np.degrees(np.arccos(np.clip(np.dot(normalize(v1), normalize(v2)), -1.0, 1.0)))


# def construct_tangent_arc(r_sun, r_entry, r_exit, n_points_arc=50):
#     """
#     Generate a circular arc that stays on the cone boundary between entry and exit.
#     """
#     # Vector orthogonal to Sun and entry (to define rotation plane)
#     n = normalize(np.cross(r_entry, r_sun))
#     theta_entry = angle_between(r_sun, r_entry)
#     theta_exit = angle_between(r_sun, r_exit)

#     # Define angle range along the arc
#     angle_range = np.linspace(0, angle_between(r_entry, r_exit), n_points_arc)
#     axis = normalize(np.cross(r_sun, r_entry))
#     if np.linalg.norm(axis) < 1e-6:
#         axis = normalize(np.cross(r_sun, np.array([1.0, 0.0, 0.0])))
#     axis = normalize(axis)

#     arc = []
#     for theta in angle_range:
#         rot_axis = axis
#         angle_rad = np.radians(theta)
#         v = (r_entry * np.cos(angle_rad) +
#              np.cross(rot_axis, r_entry) * np.sin(angle_rad) +
#              rot_axis * np.dot(rot_axis, r_entry) * (1 - np.cos(angle_rad)))
#         arc.append(normalize(v))
#     return arc


# # Simulation setup
# r_sun = normalize(np.array([0.0, 0.0, 1.0]))  # Sun direction
# min_angle_deg = 5.0  # Forbidden cone half-angle
# n_points = 100
# t = np.linspace(0, 1, n_points)

# # Target moves from left to right in arc close to the Sun
# r_target_traj = []
# avoidance_traj = []
# mode_traj = []

# # Construct an arc that gradually approaches and crosses near the Sun vector
# for i in range(n_points):
#     angle = np.radians(20 * (t[i] - 0.5))  # from -10° to +10° w.r.t Sun
#     r_target = normalize(np.array([np.sin(angle), 0, np.cos(angle)]))
#     r_target_traj.append(r_target)

# # Detect entry and exit from forbidden zone
# angles = [angle_between(r, r_sun) for r in r_target_traj]
# inside_zone = [theta < min_angle_deg for theta in angles]

# entry_idx = next((i for i, inside in enumerate(inside_zone) if inside), None)
# exit_idx = next((i for i in range(entry_idx, len(inside_zone)) if not inside_zone[i]), None) if entry_idx is not None else None

# # Build full trajectory
# for i in range(n_points):
#     if entry_idx is not None and entry_idx <= i < exit_idx:
#         mode = "avoid"
#     else:
#         mode = "track"
#     mode_traj.append(mode)

# # Construct full avoidance trajectory
# for i in range(n_points):
#     if mode_traj[i] == "track":
#         avoidance_traj.append(r_target_traj[i])
#     elif i == entry_idx:
#         r_entry = r_target_traj[entry_idx]
#         r_exit = r_target_traj[exit_idx]
#         arc_points = construct_tangent_arc(r_sun, r_entry, r_exit)
#         avoidance_traj.extend(arc_points)
#     elif i > entry_idx and i < exit_idx:
#         continue  # Already filled by arc
#     elif i >= exit_idx:
#         avoidance_traj.append(r_target_traj[i])

# # Convert to arrays for plotting
# r_target_traj = np.array(r_target_traj)
# avoidance_traj = np.array(avoidance_traj)

# # Plotting in 3D
# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')

# # Plot Sun direction
# ax.quiver(0, 0, 0, *r_sun, color='orange', linewidth=2, label='Sun Direction')

# # Plot forbidden cone as sphere cap
# u = np.linspace(0, 2 * np.pi, 100)
# v = np.linspace(0, np.radians(min_angle_deg), 50)
# x = np.outer(np.sin(v), np.cos(u))
# y = np.outer(np.sin(v), np.sin(u))
# z = np.outer(np.cos(v), np.ones_like(u))
# cone_x = x * r_sun[2] + r_sun[0]
# cone_y = y * r_sun[2] + r_sun[1]
# cone_z = z * r_sun[2] + r_sun[2]
# ax.plot_surface(cone_x, cone_y, cone_z, color='orange', alpha=0.3)

# # Plot trajectories
# ax.plot(r_target_traj[:, 0], r_target_traj[:, 1], r_target_traj[:, 2], 'g', label='Original Target LOS')
# ax.plot(avoidance_traj[:, 0], avoidance_traj[:, 1], avoidance_traj[:, 2], 'b', label='Sun-Avoided LOS')

# ax.set_title("3D LOS Trajectory with Sun Avoidance")
# ax.legend()
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# ax.set_box_aspect([1, 1, 1])
# plt.tight_layout()
# plt.show()




# # # %%
# import numpy as np
# import pandas as pd
# from datetime import datetime, timedelta
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# import os, sys

# # Constants
# R_EARTH = 6378.1366  # km, Earth Radius
# DEG2RAD = np.pi / 180
# RAD2DEG = 180 / np.pi
# ANG_THRESHOLD = 5.0  # degrees, forbidden zone half-angle
# INITIAL_ROT_ANGLE = 5.1  # degrees, initial rotation angle
# ROT_INCREMENT = 0.1  # degrees, increment for iterative adjustment
# AU = 149597870.7  # km, 1 AU
# SUN_VECTOR_SCALE = 20000  # km, for vectors in 3D plot
# SAMPLE_INTERVAL = 50  # Sample every 50th point for faster plots
# TOLERANCE = 0.01  # Tolerance for anti-parallel check
# MAX_ITERATIONS = 10  # Maximum iterations for adjustment

# def julian_date(dt):
#     """Compute Julian Date from datetime object"""
#     year, month, day = dt.year, dt.month, dt.day + (dt.hour + dt.minute / 60 + dt.second / 3600) / 24
#     if month <= 2:
#         year -= 1
#         month += 12
#     A = year // 100
#     B = 2 - A + A // 4
#     JD = np.floor(365.25 * (year + 4716)) + np.floor(30.6001 * (month + 1)) + day + B - 1524.5
#     return JD

# def sun_position_eci(jd):
#     """Approximate Sun position in ECI frame (km)"""
#     T = (jd - 2451545.0) / 36525.0
#     L0 = 280.46646 + 36000.76983 * T + 0.0003032 * T**2
#     M = 357.52911 + 35999.05029 * T - 0.0001537 * T**2
#     L0 = L0 % 360
#     M = M % 360
#     e = 0.016708634 - 0.000042037 * T
#     E = M + e * np.sin(M * DEG2RAD) * (1 + e * np.cos(M * DEG2RAD))
#     x_helio = np.cos(E * DEG2RAD) - e
#     y_helio = np.sqrt(1 - e**2) * np.sin(E * DEG2RAD)
#     r = np.sqrt(x_helio**2 + y_helio**2) * AU
#     lon = (L0 + np.arctan2(y_helio, x_helio) / DEG2RAD) % 360
#     eps = 23.44 * DEG2RAD
#     x_eci = r * np.cos(lon * DEG2RAD)
#     y_eci = r * (np.sin(lon * DEG2RAD) * np.cos(eps))
#     z_eci = r * (np.sin(lon * DEG2RAD) * np.sin(eps))
#     return np.array([x_eci, y_eci, z_eci])

# def check_earth_occultation(r_a, r_b):
#     """Check if Earth occults the LOS from r_a to r_b"""
#     d = r_b - r_a
#     t = -np.dot(r_a, d) / np.dot(d, d)
#     if t < 0:
#         r_closest = r_a
#     elif t > 1:
#         r_closest = r_b
#     else:
#         r_closest = r_a + t * d
#     return np.linalg.norm(r_closest) > R_EARTH

# def normalize(v):
#     """Normalize a vector"""
#     norm = np.linalg.norm(v)
#     return v / norm if norm > 1e-6 else np.zeros_like(v)

# def angle_between(v1, v2):
#     """Compute angle between two unit vectors in radians"""
#     dot = np.dot(normalize(v1), normalize(v2))
#     return np.arccos(np.clip(dot, -1.0, 1.0))

# def rodrigues_rotation(v, k, theta):
#     """Rotate vector v around axis k by angle theta (in radians) using Rodrigues' formula"""
#     k = normalize(k)
#     cos_theta = np.cos(theta)
#     sin_theta = np.sin(theta)
#     return (v * cos_theta +
#             np.cross(k, v) * sin_theta +
#             k * np.dot(k, v) * (1 - cos_theta))

# def avoidance_direction(los_unit, sun_unit):
#     """Compute rotated avoidance direction with iterative adjustment using v_orth"""
#     initial_angle = np.degrees(angle_between(los_unit, sun_unit))
#     if initial_angle > ANG_THRESHOLD or abs(np.dot(los_unit, sun_unit) + 1.0) < TOLERANCE:  # Safe or anti-parallel
#         return los_unit, "track" if initial_angle > ANG_THRESHOLD else "safe"

#     # Construct orthogonal direction to sun_unit in the plane of los_unit and sun_unit
#     v_orth = los_unit - np.dot(los_unit, sun_unit) * sun_unit
#     if np.linalg.norm(v_orth) < 1e-6:
#         v_orth = np.cross(sun_unit, np.array([1.0, 0.0, 0.0]))
#         if np.linalg.norm(v_orth) < 1e-6:
#             v_orth = np.cross(sun_unit, np.array([0.0, 1.0, 0.0]))
#     v_orth = normalize(v_orth)

#     # Rotation axis as normal to LOS-Sun plane
#     rot_axis = np.cross(los_unit, sun_unit)
#     if np.linalg.norm(rot_axis) < 1e-6:
#         rot_axis = np.cross(sun_unit, np.array([0.0, 0.0, 1.0]))
#     rot_axis = normalize(rot_axis)

#     # Determine rotation direction using v_orth
#     direction_sign = np.sign(np.dot(v_orth, rot_axis))
#     if direction_sign == 0:
#         direction_sign = 1.0  # Default to positive if ambiguous

#     # Iterative adjustment
#     rot_angle = np.radians(INITIAL_ROT_ANGLE) * direction_sign
#     new_los = los_unit
#     for _ in range(MAX_ITERATIONS):
#         new_los = rodrigues_rotation(los_unit, rot_axis, rot_angle)
#         new_angle = np.degrees(angle_between(new_los, sun_unit))
#         if new_angle > initial_angle and new_angle > ANG_THRESHOLD:
#             return normalize(new_los), "avoid"
#         rot_angle += np.radians(ROT_INCREMENT) * direction_sign

#     # Fallback to reverse direction if needed
#     rot_angle = -np.radians(INITIAL_ROT_ANGLE) * direction_sign
#     for _ in range(MAX_ITERATIONS):
#         new_los = rodrigues_rotation(los_unit, rot_axis, rot_angle)
#         new_angle = np.degrees(angle_between(new_los, sun_unit))
#         if new_angle > initial_angle and new_angle > ANG_THRESHOLD:
#             return normalize(new_los), "avoid"
#         rot_angle -= np.radians(ROT_INCREMENT) * direction_sign

#     return normalize(new_los), "avoid"  # Return last attempt if max iterations reached


# # Read data
# sys.path.insert(1, os.getcwd()[:os.getcwd().index('astropynaric')+13])

# import tudat_tools.data_processing.data_processing_utilities as dputil
# import basic_tools.time_conversion as t_conv

# path_cwd = os.getcwd()
# csv_output_path = 'examples/output_data/pointing_error'
# data_raw, simulation_parameters = dputil.load_constellation_data(full_path = csv_output_path)

# file_path = os.path.join(csv_output_path, 'state_history.dat')
# try:
#     statehistory = pd.read_csv(file_path, sep='\t', header=None)
# except FileNotFoundError:
#     print(f"Error: File '{file_path}' not found.")
#     exit()


# host_chosen = simulation_parameters['sat_names'][1]
# target_chosen = simulation_parameters['sat_names'][3]

# print(f"Host: {host_chosen}, \nTarget:{target_chosen}")
# # Extract HOST and CROSS TARGET positions (km)

# LEO_HOST_r = statehistory.iloc[:,simulation_parameters['r_index'][host_chosen]].to_numpy()   
# LEO_CROSS_TARGET_r  = statehistory.iloc[:,simulation_parameters['r_index'][target_chosen]].to_numpy()                  #[data_raw[:,simulation_parameters['r_index'][target_chosen]]
# time = statehistory.iloc[:, 0].values - statehistory.iloc[0, 0]




# # Verify data
# print("\nData verification:")
# print(f"LEO_HOST_r shape: {LEO_HOST_r.shape}")
# print(f"LEO_CROSS_TARGET_r shape: {LEO_CROSS_TARGET_r.shape}")
# print(f"First HOST position: {LEO_HOST_r[0]}")
# print(f"First TARGET position: {LEO_CROSS_TARGET_r[0]}")
# print(f"Time range: {time[0]:.2f} to {time[-1]:.2f} s")

# # Time reference: 2022-11-14 09:16:59 UTC
# start_dt = datetime(2022, 11, 14, 9, 16, 59)
# jd_start = julian_date(start_dt)
# jd_times = jd_start + time / 86400.0

# # Compute Sun unit vector
# sun_positions = np.array([sun_position_eci(jd) for jd in jd_times])
# sun_unit = normalize(sun_positions[0])  # Use first Sun position for simplicity (constant direction approx.)

# # Compute pointing directions and angles
# payload_traj = []
# mode_traj = []
# angles_to_sun = []
# occulted = []
# for i in range(len(time)):
#     r_target = LEO_CROSS_TARGET_r[i]
#     r_host = LEO_HOST_r[i]
#     los_unit = normalize(r_target - r_host)  # Calculate LOS unit vector
#     sun_vec = sun_positions[i] - r_host
#     sun_unit_i = normalize(sun_vec)  # Local Sun unit vector from host
#     angle = np.degrees(angle_between(los_unit, sun_unit_i))
#     angles_to_sun.append(angle)
#     is_occulted = not check_earth_occultation(r_host, r_target)
#     occulted.append(is_occulted)
#     if is_occulted:
#         payload_traj.append(np.zeros(3))
#         mode_traj.append("occulted")
#         continue
#     pointing_vec, mode = avoidance_direction(los_unit, sun_unit_i)
#     payload_traj.append(pointing_vec)
#     mode_traj.append(mode)

# payload_traj = np.array(payload_traj)
# angles_to_sun = np.array(angles_to_sun)
# occulted = np.array(occulted)

# # Print diagnostics
# print("\nDiagnostics:")
# print(f"Min LOS-Sun angle: {np.nanmin(angles_to_sun):.3f} deg")
# print(f"Max LOS-Sun angle: {np.nanmax(angles_to_sun):.3f} deg")
# print(f"Mean LOS-Sun angle: {np.nanmean(angles_to_sun):.3f} deg")
# print(f"Number of avoidance instances: {sum(m == 'avoid' for m in mode_traj)}")
# print(f"Number of occulted instances: {sum(m == 'occulted' for m in mode_traj)}")
# print(f"Number of track instances: {sum(m == 'track' for m in mode_traj)}")
# print(f"Number of safe instances: {sum(m == 'safe' for m in mode_traj)}")
# print("Sample LOS-Sun angles (first 5):")
# for i in range(min(5, len(time))):
#     print(f"t={time[i]:.2f} s, Angle={angles_to_sun[i]:.3f} deg, Mode={mode_traj[i]}, Occulted={occulted[i]}")

# # Compute dynamic plot range
# max_extent = max(np.max(np.abs(LEO_HOST_r)), np.max(np.abs(LEO_CROSS_TARGET_r)))
# PLOT_RANGE = max(max_extent * 1.5, SUN_VECTOR_SCALE * 1.1)
# print(f"Dynamic PLOT_RANGE: {PLOT_RANGE:.2f} km")

# # 3D Plot
# fig = plt.figure(figsize=(12, 10))
# ax = fig.add_subplot(111, projection='3d')

# # Plot Earth
# u = np.linspace(0, 2 * np.pi, 20)
# v = np.linspace(0, np.pi, 20)
# x = R_EARTH * np.outer(np.cos(u), np.sin(v))
# y = R_EARTH * np.outer(np.sin(u), np.sin(v))
# z = R_EARTH * np.outer(np.ones(np.size(u)), np.cos(v))
# ax.plot_wireframe(x, y, z, color='lightgray', alpha=0.3, label='Earth')

# # Plot trajectories (sampled)
# indices = range(0, len(time), SAMPLE_INTERVAL)
# ax.plot3D(LEO_HOST_r[indices, 0], LEO_HOST_r[indices, 1], LEO_HOST_r[indices, 2], 'b', label='LEO HOST')
# ax.plot3D(LEO_CROSS_TARGET_r[indices, 0], LEO_CROSS_TARGET_r[indices, 1], LEO_CROSS_TARGET_r[indices, 2], 'r', label='LEO CROSS TARGET')

# # Plot first few points
# ax.scatter3D(LEO_HOST_r[0:5, 0], LEO_HOST_r[0:5, 1], LEO_HOST_r[0:5, 2], c='blue', s=50, label='HOST Start')
# ax.scatter3D(LEO_CROSS_TARGET_r[0:5, 0], LEO_CROSS_TARGET_r[0:5, 1], LEO_CROSS_TARGET_r[0:5, 2], c='red', s=50, label='TARGET Start')

# # Plot payload pointing vectors (sampled)
# for i in indices:
#     if mode_traj[i] == "occulted":
#         continue
#     start = LEO_HOST_r[i]
#     direction = payload_traj[i] * 1000
#     end = start + direction
#     color = 'g' if mode_traj[i] == "track" else 'm' if mode_traj[i] == "avoid" else 'c'
#     label = 'Payload (Track)' if mode_traj[i] == "track" and i == indices[0] else ('Payload (Avoid)' if mode_traj[i] == "avoid" and i == indices[0] else ('Payload (Safe)' if mode_traj[i] == "safe" and i == indices[0] else ''))
#     ax.plot3D([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], color, label=label)
#     ax.scatter3D([start[0]], [start[1]], [start[2]], c='b', s=50)
#     if mode_traj[i] == "track":
#         ax.scatter3D([LEO_CROSS_TARGET_r[i, 0]], [LEO_CROSS_TARGET_r[i, 1]], [LEO_CROSS_TARGET_r[i, 2]], c='r', s=50)

# # Plot Sun direction vector
# sun_end = normalize(sun_positions[0]) * SUN_VECTOR_SCALE
# ax.plot3D([0, sun_end[0]], [0, sun_end[1]], [0, sun_end[2]], 'y-', linewidth=2, label='Sun Direction')
# ax.scatter3D([sun_end[0]], [sun_end[1]], [sun_end[2]], c='orange', s=100, marker='*', label='Sun (scaled)')

# ax.set_xlim3d(-PLOT_RANGE, PLOT_RANGE)
# ax.set_ylim3d(-PLOT_RANGE, PLOT_RANGE)
# ax.set_zlim3d(-PLOT_RANGE, PLOT_RANGE)
# ax.set_xlabel('ECI x [km]')
# ax.set_ylabel('ECI y [km]')
# ax.set_zlabel('ECI z [km]')
# ax.set_title('Satellite Trajectories with Payload Pointing and Sun Direction')
# ax.legend()
# ax.grid(True)

# # 2D Plot
# fig2, ax2 = plt.subplots(figsize=(8, 8))
# sun_norm = normalize(sun_positions[0])
# x_axis = normalize(np.array([1.0, 0.0, 0.0]))
# if np.abs(np.dot(x_axis, sun_norm)) > 0.999:
#     x_axis = np.array([0.0, 1.0, 0.0])
# y_axis = normalize(np.cross(sun_norm, x_axis))
# x_axis = normalize(np.cross(y_axis, sun_norm))

# def project_to_plane(v, x_basis, y_basis):
#     return np.array([np.dot(v, x_basis), np.dot(v, y_basis)])

# target_2d = np.array([project_to_plane(normalize(LEO_CROSS_TARGET_r[i] - LEO_HOST_r[i]), x_axis, y_axis) for i in range(0, len(time), SAMPLE_INTERVAL)])
# payload_2d = np.array([project_to_plane(payload_traj[i], x_axis, y_axis) if mode_traj[i] != "occulted" else [np.nan, np.nan] for i in range(0, len(time), SAMPLE_INTERVAL)])

# circle_radius = np.sin(np.radians(ANG_THRESHOLD))
# circle = plt.Circle((0, 0), circle_radius, color='orange', alpha=0.3, label='Sun Avoidance Zone (±5°)')
# ax2.add_patch(circle)
# ax2.plot(target_2d[:, 0], target_2d[:, 1], 'b--', label='Target LOS (Projected)')
# ax2.plot(payload_2d[:, 0], payload_2d[:, 1], 'g-', label='Payload Pointing (Safe)')
# for i in range(0, len(time), SAMPLE_INTERVAL):
#     if mode_traj[i] == "avoid":
#         ax2.plot(payload_2d[i // SAMPLE_INTERVAL, 0], payload_2d[i // SAMPLE_INTERVAL, 1], 'mo', markersize=4)
# ax2.plot(0, 0, 'o', color='orange', label='Sun')
# ax2.set_aspect('equal')
# ax2.set_xlabel('X (Sun-orthogonal frame)')
# ax2.set_ylabel('Y (Sun-orthogonal frame)')
# ax2.set_title('Payload Pointing in 2D with Sun Avoidance Zone')
# ax2.legend()
# ax2.grid(True)
# plt.show()
#%%
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os, sys
from astropy.time import Time

# Constants
R_EARTH = 6378.1366  # km, Earth Radius
DEG2RAD = np.pi / 180
RAD2DEG = 180 / np.pi
ANG_THRESHOLD = 5.0  # degrees, forbidden zone half-angle
INITIAL_ROT_ANGLE = 5.1  # degrees, initial rotation angle
ROT_INCREMENT = 0.1  # degrees, increment for iterative adjustment
AU = 149597870.7  # km, 1 AU
SUN_VECTOR_SCALE = 20000  # km, for vectors in 3D plot
SAMPLE_INTERVAL = 50  # Sample every 50th point for faster plots
TOLERANCE = 0.01  # Tolerance for anti-parallel check
MAX_ITERATIONS = 10  # Maximum iterations for adjustment


# -----------------------------
# Helper functions
# -----------------------------
def normalize(v):
    return v / np.linalg.norm(v)

def angle_between(v1, v2):
    return np.arccos(np.clip(np.dot(normalize(v1), normalize(v2)), -1.0, 1.0))

def axis_angle_to_quaternion(axis, theta):
    axis = normalize(axis)
    w = np.cos(theta / 2)
    xyz = axis * np.sin(theta / 2)
    return np.array([w, *xyz])

def quaternion_rotate_vector(q, v):
    w, x, y, z = q
    q_vec = np.array([x, y, z])
    uv = np.cross(q_vec, v)
    uuv = np.cross(q_vec, uv)
    return v + 2 * (w * uv + uuv)

def avoid_sun_by_quaternion(r_target, r_sun, min_angle_deg=5.1):
    r_target = normalize(r_target)
    r_sun = normalize(r_sun)
    theta = angle_between(r_target, r_sun)

    if np.degrees(theta) >= min_angle_deg:
        
        return r_target, "track"

    # Construct orthogonal direction to r_sun in the plane of r_target and r_sun
    v_orth = r_target - np.dot(r_target, r_sun) * r_sun
    if np.linalg.norm(v_orth) < 1e-6:
        v_orth = np.cross(r_sun, np.array([1.0, 0.0, 0.0]))
        if np.linalg.norm(v_orth) < 1e-6:
            v_orth = np.cross(r_sun, np.array([0.0, 1.0, 0.0]))
    v_orth = normalize(v_orth)

    min_angle_rad = np.radians(min_angle_deg)
    q = axis_angle_to_quaternion(v_orth, min_angle_rad - theta)
    rotated_vec = quaternion_rotate_vector(q, r_target)
    # print('SA activated')
    return normalize(rotated_vec), "avoid"

def julian_date(dt):
    """Compute Julian Date from datetime object"""
    year, month, day = dt.year, dt.month, dt.day + (dt.hour + dt.minute / 60 + dt.second / 3600) / 24
    if month <= 2:
        year -= 1
        month += 12
    A = year // 100
    B = 2 - A + A // 4
    JD = np.floor(365.25 * (year + 4716)) + np.floor(30.6001 * (month + 1)) + day + B - 1524.5
    return JD

def sun_position_eci(jd):
    """Approximate Sun position in ECI frame (km)"""
    T = (jd - 2451545.0) / 36525.0
    L0 = 280.46646 + 36000.76983 * T + 0.0003032 * T**2
    M = 357.52911 + 35999.05029 * T - 0.0001537 * T**2
    L0 = L0 % 360
    M = M % 360
    e = 0.016708634 - 0.000042037 * T
    E = M + e * np.sin(M * DEG2RAD) * (1 + e * np.cos(M * DEG2RAD))
    x_helio = np.cos(E * DEG2RAD) - e
    y_helio = np.sqrt(1 - e**2) * np.sin(E * DEG2RAD)
    r = np.sqrt(x_helio**2 + y_helio**2) * AU
    lon = (L0 + np.arctan2(y_helio, x_helio) / DEG2RAD) % 360
    eps = 23.44 * DEG2RAD
    x_eci = r * np.cos(lon * DEG2RAD)
    y_eci = r * (np.sin(lon * DEG2RAD) * np.cos(eps))
    z_eci = r * (np.sin(lon * DEG2RAD) * np.sin(eps))
    return np.array([x_eci, y_eci, z_eci])

def check_earth_occultation(r_a, r_b):
    """Check if Earth occults the LOS from r_a to r_b"""
    d = r_b - r_a
    t = -np.dot(r_a, d) / np.dot(d, d)
    if t < 0:
        r_closest = r_a
    elif t > 1:
        r_closest = r_b
    else:
        r_closest = r_a + t * d
    return np.linalg.norm(r_closest) > R_EARTH

# -----------------------------
# Simulate
# -----------------------------

# Read data
sys.path.insert(1, os.getcwd()[:os.getcwd().index('astropynaric')+13])

import tudat_tools.data_processing.data_processing_utilities as dputil
import basic_tools.time_conversion as t_conv

path_cwd = os.getcwd()
csv_output_path = 'examples/output_data/pointing_error'
data_raw, simulation_parameters = dputil.load_constellation_data(full_path = csv_output_path)

file_path = os.path.join(csv_output_path, 'state_history.dat')
try:
    statehistory = pd.read_csv(file_path, sep='\t', header=None)
except FileNotFoundError:
    print(f"Error: File '{file_path}' not found.")
    exit()

host_chosen = simulation_parameters['sat_names'][1]
target_chosen = simulation_parameters['sat_names'][3]

print(f"Host: {host_chosen}, \nTarget:{target_chosen}")
# Extract HOST and CROSS TARGET positions (km)

LEO_HOST_r = statehistory.iloc[:,simulation_parameters['r_index'][host_chosen]].to_numpy()   
LEO_CROSS_TARGET_r  = statehistory.iloc[:,simulation_parameters['r_index'][target_chosen]].to_numpy()                  #[data_raw[:,simulation_parameters['r_index'][target_chosen]]
time = statehistory.iloc[:, 0].values - statehistory.iloc[0, 0]

# Time reference: 2022-11-14 09:16:59 UTC
start_dt = datetime(2025, 7, 28, 17, 00, 00)
jd_start = julian_date(start_dt)
jd_times = jd_start + time / 86400.0

jd_times =Time(statehistory.iloc[:,0].values, format='gps', scale='utc').jd

# Compute Sun unit vector
# sun_positions = np.array([sun_position_eci(jd) for jd in jd_times])
# sun_unit = normalize(sun_positions[0])  # Use first Sun position for simplicity (constant direction approx.)
df  = pd.read_csv(os.path.join('examples/output_data/tables/SUN_MOON','sun_ephemeris_eci.csv'), sep = ',')
sun_positions = df.iloc[:,1:4].values
# Compute pointing directions and angles
payload_traj = []
mode_traj = []
angles_to_sun = []
occulted = []
for i in range(len(time)-1):
    r_target = LEO_CROSS_TARGET_r[i]
    r_host = LEO_HOST_r[i]
    los_unit = normalize(r_target - r_host)  # Calculate LOS unit vector
    sun_vec = sun_positions[i] - r_host
    sun_unit_i = normalize(sun_vec)  # Local Sun unit vector from host
    angle = np.degrees(angle_between(los_unit, sun_unit_i))
    angles_to_sun.append(angle)
    is_occulted = not check_earth_occultation(r_host, r_target)
    occulted.append(is_occulted)
    if is_occulted:
        payload_traj.append(np.zeros(3))
        mode_traj.append("occulted")
        continue
    # pointing_vec, mode = avoidance_direction(los_unit, sun_unit_i)
    pointing_vec, mode = avoid_sun_by_quaternion(r_target, sun_unit_i, min_angle_deg=7)
    payload_traj.append(pointing_vec)
    mode_traj.append(mode)

payload_traj = np.array(payload_traj)
angles_to_sun = np.array(angles_to_sun)
occulted = np.array(occulted)
r_target_traj = np.array(LEO_CROSS_TARGET_r)

# Print diagnostics
print("\nDiagnostics:")
print(f"Min LOS-Sun angle: {np.nanmin(angles_to_sun):.3f} deg")
print(f"Max LOS-Sun angle: {np.nanmax(angles_to_sun):.3f} deg")
print(f"Mean LOS-Sun angle: {np.nanmean(angles_to_sun):.3f} deg")
print(f"Number of avoidance instances: {sum(m == 'avoid' for m in mode_traj)}")
print(f"Number of occulted instances: {sum(m == 'occulted' for m in mode_traj)}")
print(f"Number of track instances: {sum(m == 'track' for m in mode_traj)}")
print(f"Number of S.A. instances: {sum(m == 'safe' for m in mode_traj)}")
print("Sample LOS-Sun angles (first 5):")
for i in range(min(5, len(time))):
    print(f"t={time[i]:.2f} s, Angle={angles_to_sun[i]:.3f} deg, Mode={mode_traj[i]}, Occulted={occulted[i]}")
# -----------------------------
# 3D Plotting
# -----------------------------
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_title("3D View: Target Trajectory and Sun Avoidance")

# Plot the original target trajectory
ax.plot(r_target_traj[:, 0], r_target_traj[:, 1], r_target_traj[:, 2], label='Original Target LOS', color='green')

# Plot the avoidance trajectory
ax.scatter(payload_traj[:, 0], payload_traj[:, 1], payload_traj[:, 2], label='Sun Avoided LOS', color='blue')

#ax.plot(occulted[:, 0], occulted[:, 1], occulted[:, 2], label='Occulted', color='cyan')

# Plot Sun vector
ax.quiver(0, 0, 0, sun_unit_i[0], sun_unit_i[1], sun_unit_i[2], color='orange', linewidth=2, label='Sun Vector')

# Draw forbidden cone surface (approximate with circle)
cone_angle = np.radians(5.0)
circle_points = 100
theta_circle = np.linspace(0, 2 * np.pi, circle_points)
circle_radius = np.tan(cone_angle)
circle_x = circle_radius * np.cos(theta_circle)
circle_y = circle_radius * np.sin(theta_circle)
circle_z = np.ones_like(circle_x)

# Align circle with r_sun direction
sun_dir = normalize(sun_unit_i)
R = np.eye(3) if np.allclose(sun_dir, [0, 0, 1]) else np.linalg.qr(np.vstack((sun_dir, np.random.rand(3), np.random.rand(3))).T)[0]
circle_xyz = np.vstack((circle_x, circle_y, circle_z))
rotated_circle = R @ circle_xyz
ax.plot(rotated_circle[0], rotated_circle[1], rotated_circle[2], color='orange', linestyle='--', label='Forbidden Zone Boundary')

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([-0.2, 1.2])
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.show()

# %%
# import numpy as np
# import matplotlib.pyplot as plt

# # Sweep angle between target and sun
# theta = np.linspace(0, np.pi, 500)  # 0 to 180 degrees

# # Sun vector (fixed along x-axis)
# s = np.array([1, 0, 0])

# # Target vector rotates in x-y plane
# t = np.vstack([np.cos(theta), np.sin(theta), np.zeros_like(theta)])

# # Vector definitions
# v = t.T - s  # subtraction vector
# v_perp = t.T - (t.T @ s)[:, None] * s  # orthogonalized vector

# # Normalize for angle calculation
# def angle_between(a, b):
#     dot = np.sum(a * b, axis=-1)
#     norm = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
#     return np.degrees(np.arccos(np.clip(dot / norm, -1.0, 1.0)))

# # Angles w.r.t sun vector
# angle_v_sun = angle_between(v, s)
# angle_vperp_sun = angle_between(v_perp, s)

# # Magnitudes
# mag_v = np.linalg.norm(v, axis=-1)
# mag_vperp = np.linalg.norm(v_perp, axis=-1)

# # Plot results
# fig, axs = plt.subplots(2, 1, figsize=(7, 8))

# axs[0].plot(np.degrees(theta), angle_v_sun, label="∠(v, Sun)")
# axs[0].plot(np.degrees(theta), angle_vperp_sun, label="∠(v⊥, Sun)", linestyle="--")
# axs[0].set_ylabel("Angle (deg)")
# axs[0].set_xlabel("Target-Sun separation angle (deg)")
# axs[0].set_title("Angular relationship")
# axs[0].legend()
# axs[0].grid(True)

# axs[1].plot(np.degrees(theta), mag_v, label="|v|")
# axs[1].plot(np.degrees(theta), mag_vperp, label="|v⊥|", linestyle="--")
# axs[1].set_ylabel("Magnitude")
# axs[1].set_xlabel("Target-Sun separation angle (deg)")
# axs[1].set_title("Vector magnitudes")
# axs[1].legend()
# axs[1].grid(True)

# plt.tight_layout()
# plt.show()

# %%
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_geometry(theta_deg):
    # Sun vector along x-axis
    s = np.array([1, 0, 0])
    # Target vector rotated in x-y plane
    theta = np.radians(theta_deg)
    t = np.array([np.cos(theta), np.sin(theta), 0])
    # Subtraction vector
    v = t - s
    # Orthogonalized vector
    v_perp = t - (np.dot(t, s)) * s

    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(111, projection='3d')
    
    # Draw Sun, Target, v, v_perp
    ax.quiver(0,0,0, *s, color='gold', label="Sun", linewidth=2)
    ax.quiver(0,0,0, *t, color='blue', label="Target", linewidth=2)
    ax.quiver(0,0,0, *v, color='red', label="v = t - s", linewidth=2)
    ax.quiver(0,0,0, *v_perp, color='green', label="v⊥", linewidth=2)
    
    # Set limits
    lim = 1.5
    ax.set_xlim([-lim, lim]); ax.set_ylim([-lim, lim]); ax.set_zlim([-lim, lim])
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_title(f"Geometry with target-sun angle = {theta_deg}°")
    ax.legend()
    plt.show()

# Show three cases: 30°, 90°, 150°
for angle in [30, 90, 150]:
    plot_geometry(angle)

# %%
import numpy as np
import matplotlib.pyplot as plt
# We'll compute a safe boresight when the target is too close to the Sun.
# Strategy (short):
# - If angle(t, s) >= min_sep: use t (no change)
# - Else:
#    - Build w = (t - (t.s) s) / |...|  (unit vector in plane, orthogonal to s, pointing toward t)
#    - new_boresight = cos(min_sep) * s + sin(min_sep) * w
# This new_boresight is unit length, lies in the s-t plane, and makes exactly min_sep with s.
# If t is (anti)collinear with s, choose an arbitrary perpendicular w (e.g., cross with z-axis fallback).

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v

def safe_boresight(t, s, min_sep_rad):
    # assume t and s are unit vectors
    # angle between t and s
    cos_theta = np.clip(np.dot(t, s), -1.0, 1.0)
    theta = np.arccos(cos_theta)
    if theta >= min_sep_rad:
        return t, theta  # already safe
    # compute orthogonal direction w in plane toward t
    v_perp = t - cos_theta * s
    v_perp_norm = np.linalg.norm(v_perp)
    if v_perp_norm < 1e-12:
        # degenerate: t is (anti)parallel to s. pick arbitrary perpendicular w.
        # choose a vector not collinear with s, e.g. (0,0,1) unless s ~ z then use (0,1,0)
        cand = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(cand, s)) > 0.99:
            cand = np.array([0.0, 1.0, 0.0])
        w = unit(np.cross(s, cand))  # guaranteed perpendicular to s
    else:
        w = v_perp / v_perp_norm
    # construct boresight at exactly min_sep from s in the s->t direction
    new_b = np.cos(min_sep_rad) * s + np.sin(min_sep_rad) * w
    new_b = unit(new_b)
    return new_b, theta

# Example cases
s = np.array([1.0, 0.0, 0.0])  # Sun along +X
cases = [
    {"theta_deg": 2.0},
    {"theta_deg": 4.0},
    {"theta_deg": 10.0},
    {"theta_deg": 0.1},   # very close
    {"theta_deg": 180.0}, # opposite
]

min_sep_deg = 5.0
min_sep = np.radians(min_sep_deg)

results = []
for c in cases:
    th = np.radians(c["theta_deg"])
    t = np.array([np.cos(th), np.sin(th), 0.0])
    new_b, theta = safe_boresight(t, s, min_sep)
    results.append({"theta_deg": c["theta_deg"], "t": t, "new_b": new_b, "orig_angle_deg": np.degrees(theta),
                    "new_angle_deg": np.degrees(np.arccos(np.clip(np.dot(new_b, s), -1, 1)))})

# Print results
for r in results:
    print(f"Case θ_target-sun = {r['theta_deg']:.3f}°: orig angle = {r['orig_angle_deg']:.3f}°, new angle = {r['new_angle_deg']:.3f}°")

# Make 3D plot for the first three cases
fig = plt.figure(figsize=(12,4))
for i, r in enumerate(results[:3]):
    ax = fig.add_subplot(1, 3, i+1, projection='3d')
    t = r['t']; new_b = r['new_b']
    ax.quiver(0,0,0, *s, length=1.0, linewidth=2, label='Sun')
    ax.quiver(0,0,0, *t, length=1.0, linewidth=2, label='Target')
    ax.quiver(0,0,0, *new_b, length=1.0, linewidth=2, color='green', label='Safe boresight')
    ax.set_xlim([-1.2,1.2]); ax.set_ylim([-1.2,1.2]); ax.set_zlim([-1.2,1.2])
    ax.set_title(f"θ_target-sun = {r['theta_deg']}° → new = {r['new_angle_deg']:.2f}°")
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.view_init(elev=20, azim=30)
    ax.legend()
plt.tight_layout()
# plt.show()

def enforce_sun_avoidance(target, sun, min_angle_deg=5):
    target = target / np.linalg.norm(target)
    sun = sun / np.linalg.norm(sun)

    dot = np.dot(target, sun)
    dot = np.clip(dot, -1.0, 1.0)
    angle = np.degrees(np.arccos(dot))

    if angle >= min_angle_deg:
        return target, angle, angle

    # Find an orthogonal basis to rotate target away from sun
    orth = target - dot * sun
    if np.linalg.norm(orth) < 1e-6:
        # Target is nearly parallel to Sun: pick arbitrary perpendicular
        if abs(sun[0]) < 0.9:
            ref = np.array([1,0,0])
        else:
            ref = np.array([0,1,0])
        orth = np.cross(sun, ref)
    orth /= np.linalg.norm(orth)

    min_angle_rad = np.radians(min_angle_deg)
    new_target = np.cos(min_angle_rad)*sun + np.sin(min_angle_rad)*orth

    new_dot = np.dot(new_target, sun)
    new_dot = np.clip(new_dot, -1.0, 1.0)
    new_angle = np.degrees(np.arccos(new_dot))

    return new_target, angle, new_angle

# Test different cases
cases = [
    (np.array([1,0,0]), np.array([np.cos(np.radians(2)), np.sin(np.radians(2)), 0])),
    (np.array([1,0,0]), np.array([np.cos(np.radians(4)), np.sin(np.radians(4)), 0])),
    (np.array([1,0,0]), np.array([np.cos(np.radians(10)), np.sin(np.radians(10)), 0])),
    (np.array([1,0,0]), np.array([np.cos(np.radians(0.1)), np.sin(np.radians(0.1)), 0])),
    (np.array([1,0,0]), np.array([-1,0,0]))
]

for target, sun in cases:
    new_target, orig_angle, new_angle = enforce_sun_avoidance(target, sun, min_angle_deg=5)
    print(f"Case θ_target-sun = {orig_angle:.3f}°: orig = {orig_angle:.3f}°, new = {new_angle:.3f}°")

plt.show()
# %%
import numpy as np
import matplotlib.pyplot as plt

# Simulated example data (100 timesteps at 1Hz)
timesteps = 100
np.random.seed(0)

# Example Sat1 position in ECI (LEO ~7000 km from center)
r_sat1 = np.linspace([7077e3, 0, 0], [7000e3, 500e3, 1000e3], timesteps)
v_sat1 = np.gradient(r_sat1, axis=0)  # Approximate velocity

# Example Sun position in ECI (simplified path across sky, 1 AU distance)
r_sun = np.linspace([1.5e11, 0, 0], [1.5e11, 1e9, 0], timesteps)

# Line-of-sight (LOS) vector from Sat1 to Sun
los = r_sun - r_sat1
los_unit = los / np.linalg.norm(los, axis=1)[:, np.newaxis]

# Create Sat2 starting slightly offset from LOS
offset_direction = np.cross(los_unit[0], [0, 0, 1])
offset_direction /= np.linalg.norm(offset_direction)

offset_distance = 10_000  # 10 km off the LOS
r_sat2 = r_sat1 + offset_distance * offset_direction

# Give Sat2 a small velocity component toward LOS so it crosses it
v_relative_to_los = -offset_direction * 100  # 100 m/s toward LOS
v_sat2 = v_sat1 + v_relative_to_los  # Add to base orbital motion

# ✅ Now r_sat2 and v_sat2 are generated
# You can visualize or export them as needed

# Optional: Visualize LOS crossing
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot(*r_sat1.T, label='Sat1', color='blue')
ax.plot(*r_sat2.T, label='Sat2', color='red')
ax.plot(*r_sun.T, label='Sun', color='orange', alpha=0.5)
ax.set_title("LEO Sat2 crossing LOS between Sat1 and Sun")
ax.legend()
plt.show()

# %%
