# import numpy as np
# import pandas as pd
# from datetime import datetime, timedelta
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# import os, sys
# from astropy.time import Time
# from mpl_toolkits.mplot3d.art3d import Poly3DCollection

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


# # -----------------------------
# # Helper functions
# # -----------------------------
# def draw_earth(ax, Re=6378.1366, alpha=0.15, color='#6699cc'):
#     u = np.linspace(0, 2*np.pi, 60)
#     v = np.linspace(0, np.pi, 30)
#     x = Re * np.outer(np.cos(u), np.sin(v))
#     y = Re * np.outer(np.sin(u), np.sin(v))
#     z = Re * np.outer(np.ones_like(u), np.cos(v))
#     ax.plot_surface(x, y, z, linewidth=0, alpha=alpha, color=color, shade=True)

# def normalize(v):
#     return v / np.linalg.norm(v)

# def angle_between(v1, v2):
#     return np.arccos(np.clip(np.dot(normalize(v1), normalize(v2)), -1.0, 1.0))

# def axis_angle_to_quaternion(axis, theta):
#     axis = normalize(axis)
#     w = np.cos(theta / 2)
#     xyz = axis * np.sin(theta / 2)
#     return np.array([w, *xyz])

# def quaternion_rotate_vector(q, v):
#     w, x, y, z = q
#     q_vec = np.array([x, y, z])
#     uv = np.cross(q_vec, v)
#     uuv = np.cross(q_vec, uv)
#     return v + 2 * (w * uv + uuv)

# def avoid_sun_by_quaternion(r_target, r_sun, min_angle_deg=0):
#     """
#     r_target        : LOS of Target from Host
#     r_sun           : LOS of the Sun from Host
#     min_nagle_deg   : Sun avoidance cone half angle

#     """
#     r_target = normalize(r_target)
#     r_sun = normalize(r_sun)
#     theta = angle_between(r_target, r_sun)

#     if np.degrees(theta) >= min_angle_deg:
        
#         return r_target, "track"

#     # Construct orthogonal direction to r_sun in the plane of r_target and r_sun
#     v_orth = r_target - np.dot(r_target, r_sun) * r_sun
#     if np.linalg.norm(v_orth) < 1e-6:
#         v_orth = np.cross(r_sun, np.array([1.0, 0.0, 0.0]))
#         if np.linalg.norm(v_orth) < 1e-6:
#             v_orth = np.cross(r_sun, np.array([0.0, 1.0, 0.0]))
#     v_orth = normalize(v_orth)

#     min_angle_rad = np.radians(min_angle_deg)
#     # q = axis_angle_to_quaternion(v_orth, min_angle_rad - theta)
#     # rotated_vec = quaternion_rotate_vector(q, r_target)

#     # other method
#     rotated_vec = r_target+v_orth*0.1
#     # print('SA activated')
#     if min_angle_rad - theta == 0:
#         print('zero')
#     return normalize(rotated_vec), "avoid"

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

# def eci2rsw(r,v):
#     R = r/np.linalg.norm(r)
#     W  = np.cross(r,v) /np.linalg.norm(np.cross(r,v))
#     S = np.cross(W,R)

#     Q_eci2rtn = np.column_stack((R, S, W))

#     return Q_eci2rtn.T @ r
# # -----------------------------
# # Simulate
# # -----------------------------

# # Read data
# sys.path.insert(1, os.getcwd()[:os.getcwd().index('astropynaric')+13])

# import tudat_tools.data_processing.data_processing_utilities as dputil
# import basic_tools.time_conversion as t_conv

# path_cwd = os.getcwd()
# csv_output_path = 'examples/output_data/High Precision'
# data_raw, simulation_parameters = dputil.load_constellation_data(full_path = csv_output_path)

# file_path = os.path.join(csv_output_path, 'state_history.dat')
# try:
#     statehistory = pd.read_csv(file_path, sep='\t', header=None)
# except FileNotFoundError:
#     print(f"Error: File '{file_path}' not found.")
#     exit()

# host_chosen = simulation_parameters['sat_names'][3]
# target_chosen = simulation_parameters['sat_names'][5]

# print(f"Host: {host_chosen}, \nTarget:{target_chosen}")
# # Extract HOST and CROSS TARGET positions (km)
# sun_avoidance_angle = 7

# LEO_HOST_r = statehistory.iloc[:,simulation_parameters['r_index'][host_chosen]].to_numpy()   
# LEO_HOST_v = statehistory.iloc[:,[x+1 for x in simulation_parameters['r_index'][host_chosen]]].to_numpy()   
# LEO_CROSS_TARGET_r  = statehistory.iloc[:,simulation_parameters['r_index'][target_chosen]].to_numpy()                  #[data_raw[:,simulation_parameters['r_index'][target_chosen]]
# LEO_CROSS_TARGET_v  = statehistory.iloc[:,[x+1 for x in simulation_parameters['r_index'][target_chosen]]].to_numpy()                  
# time = statehistory.iloc[:, 0].values - statehistory.iloc[0, 0]

# # Time reference: 2022-11-14 09:16:59 UTC
# start_dt = datetime(2025, 7, 28, 17, 00, 00)
# jd_start = julian_date(start_dt)
# jd_times = jd_start + time / 86400.0

# jd_times =Time(statehistory.iloc[:,0].values, format='gps', scale='utc').jd

# # Compute Sun unit vector
# # sun_positions = np.array([sun_position_eci(jd) for jd in jd_times])
# # sun_unit = normalize(sun_positions[0])  # Use first Sun position for simplicity (constant direction approx.)
# df  = pd.read_csv(os.path.join('examples/output_data/tables/SUN_MOON','sun_ephemeris_eci.csv'), sep = ',')
# sun_positions = df.iloc[:,1:4].values

# ##-------------Create Target----------------

# # Downsample Sun position data to match LEO_HOST shape.
# indices = np.linspace(0, len(sun_positions) - 1, LEO_HOST_r.shape[0]).astype(int)
# sun_positions_ds = sun_positions[indices]
# # # Line-of-sight (LOS) vector from Sat1 to Sun

# # los = sun_positions_ds - LEO_HOST_r
# # los_target_u = los / np.linalg.norm(los, axis=1)[:, np.newaxis]

# # # Create Sat2 starting slightly offset from LOS
# # offset_direction = np.cross(los_target_u[0], [0, 0, 1])
# # offset_direction /= np.linalg.norm(offset_direction)

# # offset_distance = 10_000  # 10 km off the LOS
# # r_sat2 = LEO_HOST_r + offset_distance * offset_direction

# # # Give Sat2 a small velocity component toward LOS so it crosses it
# # v_relative_to_los = -offset_direction * 100  # 100 m/s toward LOS
# # v_sat2 = LEO_HOST_v + v_relative_to_los  # Add to base orbital motion



# ###---------------------------------------------------------------------


# # Compute pointing directions and angles
# payload_traj = []
# mode_traj = []
# angles_to_sun = []
# occulted = []
# los_traj = []
# for i in range(len(time)-1):
#     r_target_ECI = LEO_CROSS_TARGET_r[i]
#     # r_target_ECI = r_sat2[i]
#     r_host_ECI = LEO_HOST_r[i]
#     los_target_u = normalize(r_target_ECI - r_host_ECI)  # Calculate LOS unit vector
#     # los_unit_rsw = eci2rsw
#     los_sun = sun_positions[i] - r_host_ECI
#     los_sun_u = normalize(los_sun)          # Local Sun unit vector from host
#     angle = np.degrees(angle_between(los_target_u, los_sun_u))
#     angles_to_sun.append(angle)
#     is_occulted = not check_earth_occultation(r_host_ECI, r_target_ECI)
#     occulted.append(is_occulted)
#     if is_occulted:
#         payload_traj.append(np.zeros(3))
#         mode_traj.append("occulted")
#         continue
#     pointing_vec, mode = avoid_sun_by_quaternion(los_target_u, los_sun_u, min_angle_deg=sun_avoidance_angle)
#     # pointing_vec, mode = avoid_sun_by_quaternion(r_target, los_sun_u, min_angle_deg=7)
#     payload_traj.append(pointing_vec)
#     mode_traj.append(mode)
#     los_traj.append(los_target_u)


# payload_traj = np.array(payload_traj)
# angles_to_sun = np.array(angles_to_sun)
# occulted = np.array(occulted)
# r_target_traj = np.array(LEO_CROSS_TARGET_r)
# los_traj = np.array(los_traj)

# # Print diagnostics
# print("\nDiagnostics:")
# print(f"Min LOS-Sun angle: {np.nanmin(angles_to_sun):.3f} deg")
# print(f"Max LOS-Sun angle: {np.nanmax(angles_to_sun):.3f} deg")
# print(f"Mean LOS-Sun angle: {np.nanmean(angles_to_sun):.3f} deg")
# print(f"Number of avoidance instances: {sum(m == 'avoid' for m in mode_traj)}")
# print(f"Number of occulted instances: {sum(m == 'occulted' for m in mode_traj)}")
# print(f"Number of track instances: {sum(m == 'track' for m in mode_traj)}")
# print(f"Number of S.A. instances: {sum(m == 'safe' for m in mode_traj)}")
# print("Sample LOS-Sun angles (first 5):")
# for i in range(len(time)-1):
#     if mode_traj[i] == 'avoid':
#         print(f"t={time[i]:.2f} s, Angle={angles_to_sun[i]:.3f} deg, Mode={mode_traj[i]}, Occulted={occulted[i]}")
# # -----------------------------
# # 3D Plotting
# # -----------------------------
# fig1 = plt.figure(figsize=(10,9))
# ax1 = fig1.add_subplot(111, projection='3d')

# # Earth
# # draw_earth(ax1, Re=R_EARTH, alpha=0.20, color='#88aadd')

# # Orbits
# # ax.plot(LEO_HOST_r[:,0], LEO_HOST_r[:,1], LEO_HOST_r[:,2], label=f'Orbit A: {host_chosen}', lw=2)
# # ax.plot(r_sat2[:,0], r_sat2[:,1], r_sat2[:,2], label=f'Orbit B: {target_chosen}', lw=2)
# ax1.scatter(payload_traj[:, 0], payload_traj[:, 1], payload_traj[:, 2], label='Sun Avoided LOS', alpha = 0.2, color='blue')
# #ax.plot(sun_positions_ds[:,0], sun_positions_ds[:,1], sun_positions_ds[:,2], label=f'Orbit Sun', lw=2)
# ax1.legend()  



# # Draw forbidden cone surface (approximate with circle)
# cone_angle = np.radians(sun_avoidance_angle)
# circle_points = 100
# theta_circle = np.linspace(0, 2 * np.pi, circle_points)
# circle_radius = np.tan(cone_angle)
# circle_x = circle_radius * np.cos(theta_circle)
# circle_y = circle_radius * np.sin(theta_circle)
# circle_z = np.ones_like(circle_x)

# # Align circle with r_sun direction
# sun_dir = los_sun_u
# R = np.eye(3) if np.allclose(sun_dir, [0, 0, 1]) else np.linalg.qr(np.vstack((sun_dir, np.random.rand(3), np.random.rand(3))).T)[0]
# circle_xyz = np.vstack((circle_x, circle_y, circle_z))
# rotated_circle = R @ circle_xyz

# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')
# ax.set_title(f"3D View: Target Trajectory and Sun Avoidance H:{host_chosen} T:{target_chosen}")

# # Plot the original target trajectory
# # ax.plot(r_target_traj[:, 0], r_target_traj[:, 1], r_target_traj[:, 2], label='Original Target LOS', color='green')

# # Plot the avoidance trajectory
# ax.scatter(payload_traj[:, 0], payload_traj[:, 1], payload_traj[:, 2], label='Sun Avoided LOS', alpha = 0.2, color='blue')
# ax.scatter(los_traj[:, 0], los_traj[:, 1], los_traj[:, 2], label='LOS', alpha = 0.2, color='red')
# # ax.plot(occulted[:, 0], occulted[:, 1], occulted[:, 2], label='Occulted', color='cyan')

# # Plot Sun vector
# ax.quiver(0, 0, 0, los_sun_u[0], los_sun_u[1], los_sun_u[2], color='orange', linewidth=2, label='Sun Vector')

# #ax.plot(payload_traj[:, 0],  payload_traj[:, 2], 'r+', alpha = 0.1, zdir='y', zs=1)#1.5)
# ax.plot(payload_traj[:, 1], payload_traj[:, 2], 'g+',  alpha = 0.1, zdir='x', zs=-3)#-0.5
# #ax.plot(payload_traj[:, 0], payload_traj[:, 1], 'k+',  alpha = 0.1, zdir='z', zs=-1)#-1.5

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

# plt.figure()
# plt.plot(payload_traj[:, 1], payload_traj[:, 2], 'g+',  alpha = 0.2)#-0.5
# plt.plot(los_sun_u[1], los_sun_u[2], 'b+')
# plt.plot(rotated_circle[0], rotated_circle[1],'r+',  alpha = 0.2)

# plt.plot()
# plt.title(f"H:{host_chosen} T:{target_chosen}, Avoidance angle:{sun_avoidance_angle}")
# plt.grid()
# plt.show()
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os, sys
from astropy.time import Time
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

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
def draw_earth(ax, Re=6378.1366, alpha=0.15, color='#6699cc'):
    u = np.linspace(0, 2*np.pi, 60)
    v = np.linspace(0, np.pi, 30)
    x = Re * np.outer(np.cos(u), np.sin(v))
    y = Re * np.outer(np.sin(u), np.sin(v))
    z = Re * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x, y, z, linewidth=0, alpha=alpha, color=color, shade=True)

def normalize(v):
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v

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

def rotation_matrix_from_vectors(vec1, vec2):
    """
    Find the rotation matrix that aligns vec1 to vec2
    """
    a = normalize(vec1)
    b = normalize(vec2)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    
    if s < 1e-10:  # Vectors are parallel
        return np.eye(3)
    
    kmat = np.array([[0, -v[2], v[1]], 
                     [v[2], 0, -v[0]], 
                     [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix

def create_cone_surface(apex, direction, half_angle_deg, height=1.0, n_points=50):
    """
    Create a cone surface for visualization
    apex: cone apex position (3D point)
    direction: cone axis direction (unit vector)
    half_angle_deg: half-angle of the cone in degrees
    height: height of the cone along the axis
    n_points: number of points around the circle
    """
    direction = normalize(direction)
    half_angle_rad = np.radians(half_angle_deg)
    
    # Create base circle at distance 'height' from apex
    theta = np.linspace(0, 2 * np.pi, n_points)
    radius = height * np.tan(half_angle_rad)
    
    # Circle in local coordinates (perpendicular to z-axis)
    circle_x = radius * np.cos(theta)
    circle_y = radius * np.sin(theta)
    circle_z = np.ones_like(theta) * height
    
    # Rotation matrix to align z-axis with direction
    z_axis = np.array([0, 0, 1])
    R = rotation_matrix_from_vectors(z_axis, direction)
    
    # Transform circle points
    circle_local = np.vstack([circle_x, circle_y, circle_z])
    circle_world = R @ circle_local + apex.reshape(3, 1)
    
    return circle_world, R

def avoid_sun_by_quaternion(r_target, r_sun, min_angle_deg=0):
    """
    r_target        : LOS of Target from Host
    r_sun           : LOS of the Sun from Host
    min_angle_deg   : Sun avoidance cone half angle
    
    Projects the target LOS onto the cone boundary at exactly min_angle_deg from sun vector
    """
    r_target = normalize(r_target)
    r_sun = normalize(r_sun)
    theta = angle_between(r_target, r_sun)

    if np.degrees(theta) >= min_angle_deg:
        return r_target, "track"

    # Target is inside forbidden zone - must project to cone boundary
    min_angle_rad = np.radians(min_angle_deg)
    
    # Find the component of r_target perpendicular to r_sun
    v_perp = r_target - np.dot(r_target, r_sun) * r_sun
    
    # Handle edge case where r_target is parallel/anti-parallel to r_sun
    if np.linalg.norm(v_perp) < 1e-6:
        # Choose arbitrary perpendicular direction
        v_perp = np.cross(r_sun, np.array([1.0, 0.0, 0.0]))
        if np.linalg.norm(v_perp) < 1e-6:
            v_perp = np.cross(r_sun, np.array([0.0, 1.0, 0.0]))
    
    v_perp = normalize(v_perp)
    
    # Construct vector exactly at min_angle_deg from r_sun
    # New vector = cos(min_angle) * r_sun + sin(min_angle) * v_perp
    avoided_vec = np.cos(min_angle_rad) * r_sun + np.sin(min_angle_rad) * v_perp
    avoided_vec = normalize(avoided_vec)
    
    # Verify the angle (for debugging)
    verify_angle = np.degrees(angle_between(avoided_vec, r_sun))
    if abs(verify_angle - min_angle_deg) > 0.01:
        print(f"Warning: Avoided angle {verify_angle:.3f}° != target {min_angle_deg}°")
    
    return avoided_vec, "avoid"



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

def eci2rsw(r, v):
    R = r / np.linalg.norm(r)
    W = np.cross(r, v) / np.linalg.norm(np.cross(r, v))
    S = np.cross(W, R)
    Q_eci2rtn = np.column_stack((R, S, W))
    return Q_eci2rtn.T @ r

# -----------------------------
# Simulate
# -----------------------------

# Read data
sys.path.insert(1, os.getcwd()[:os.getcwd().index('astropynaric')+13])

import tudat_tools.data_processing.data_processing_utilities as dputil
import basic_tools.time_conversion as t_conv

path_cwd = os.getcwd()
csv_output_path = 'examples/output_data/High Precision'
data_raw, simulation_parameters = dputil.load_constellation_data(full_path=csv_output_path)

file_path = os.path.join(csv_output_path, 'state_history.dat')
try:
    statehistory = pd.read_csv(file_path, sep='\t', header=None)
except FileNotFoundError:
    print(f"Error: File '{file_path}' not found.")
    exit()

host_chosen = simulation_parameters['sat_names'][3]
target_chosen = simulation_parameters['sat_names'][5]

print(f"Host: {host_chosen}, \nTarget:{target_chosen}")

# Extract HOST and CROSS TARGET positions (km)
sun_avoidance_angle = ANG_THRESHOLD

LEO_HOST_r = statehistory.iloc[:, simulation_parameters['r_index'][host_chosen]].to_numpy()   
LEO_HOST_v = statehistory.iloc[:, [x+1 for x in simulation_parameters['r_index'][host_chosen]]].to_numpy()   
LEO_CROSS_TARGET_r = statehistory.iloc[:, simulation_parameters['r_index'][target_chosen]].to_numpy()
LEO_CROSS_TARGET_v = statehistory.iloc[:, [x+1 for x in simulation_parameters['r_index'][target_chosen]]].to_numpy()                  
time = statehistory.iloc[:, 0].values - statehistory.iloc[0, 0]


# Read Sun positions
df = pd.read_csv(os.path.join('examples/output_data/tables/SUN_MOON', 'sun_ephemeris_eci.csv'), sep=',')
sun_positions = df.iloc[:, 1:4].values

# Downsample Sun position data to match LEO_HOST shape
indices = np.linspace(0, len(sun_positions) - 1, LEO_HOST_r.shape[0]).astype(int)
sun_positions_ds = sun_positions[indices]

# Compute pointing directions and angles
payload_traj = []
mode_traj = []
angles_to_sun = []
occulted = []
los_traj = []

for i in range(len(time)-1):
    r_target_ECI = LEO_CROSS_TARGET_r[i]
    r_host_ECI = LEO_HOST_r[i]
    los_target_u = normalize(r_target_ECI - r_host_ECI)
    los_sun_u = normalize(sun_positions[i] - r_host_ECI)
    angle = np.degrees(angle_between(los_target_u, los_sun_u))
    angles_to_sun.append(angle)
    is_occulted = not check_earth_occultation(r_host_ECI, r_target_ECI)
    occulted.append(is_occulted)
    
    if is_occulted:
        payload_traj.append(np.zeros(3))
        mode_traj.append("occulted")
        continue
    if angle< sun_avoidance_angle:
        pointing_vec, mode = avoid_sun_by_quaternion(los_target_u, los_sun_u, min_angle_deg=sun_avoidance_angle)
    else: 
        pointing_vec, mode = los_target_u, 'track'

    payload_traj.append(pointing_vec)
    mode_traj.append(mode)
    los_traj.append(los_target_u)

payload_traj = np.array(payload_traj)
angles_to_sun = np.array(angles_to_sun)
occulted = np.array(occulted)
r_target_traj = np.array(LEO_CROSS_TARGET_r)
los_traj = np.array(los_traj)

# Print diagnostics
print("\nDiagnostics:")
print(f"Min LOS-Sun angle: {np.nanmin(angles_to_sun):.3f} deg")
print(f"Max LOS-Sun angle: {np.nanmax(angles_to_sun):.3f} deg")
print(f"Mean LOS-Sun angle: {np.nanmean(angles_to_sun):.3f} deg")
print(f"Number of avoidance instances: {sum(m == 'avoid' for m in mode_traj)}")
print(f"Number of occulted instances: {sum(m == 'occulted' for m in mode_traj)}")
print(f"Number of track instances: {sum(m == 'track' for m in mode_traj)}")

# Verify avoided vectors are at correct angle
print("\n=== VERIFICATION ===")
avoid_indices = [i for i, m in enumerate(mode_traj) if m == 'avoid']
if len(avoid_indices) > 0:
    sun_unit_avg = normalize(np.mean(sun_positions_ds[avoid_indices] - LEO_HOST_r[avoid_indices], axis=0))
    avoided_angles = []
    for idx in avoid_indices:
        sun_vec = normalize(sun_positions[idx] - LEO_HOST_r[idx])
        avoided_angle = np.degrees(angle_between(payload_traj[idx], sun_vec))
        avoided_angles.append(avoided_angle)
    
    avoided_angles = np.array(avoided_angles)
    print(f"Avoided pointing angles from Sun:")
    print(f"  Min: {np.min(avoided_angles):.3f}° (should be ~{sun_avoidance_angle}°)")
    print(f"  Max: {np.max(avoided_angles):.3f}°")
    print(f"  Mean: {np.mean(avoided_angles):.3f}°")
    print(f"  Std: {np.std(avoided_angles):.3f}°")
    
    violations = avoided_angles < sun_avoidance_angle - 0.1
    if np.any(violations):
        print(f"  WARNING: {np.sum(violations)} points violate sun avoidance angle!")
    else:
        print(f"  ✓ All avoided points respect the {sun_avoidance_angle}° exclusion zone")

# Sample avoidance instances
print("\nSample avoidance instances:")
for i in range(len(time)-1):
    if mode_traj[i] == 'avoid':
        print(f"t={time[i]:.2f} s, Angle={angles_to_sun[i]:.3f} deg, Mode={mode_traj[i]}, Occulted={occulted[i]}")
        if sum(m == 'avoid' for m in mode_traj[:i+1]) >= 5:
            break

# -----------------------------
# 3D Plotting with CORRECTED forbidden zone
# -----------------------------

# Use average Sun direction for visualization (or pick a specific timestep)
sun_unit_avg = normalize(np.mean(sun_positions_ds - LEO_HOST_r, axis=0))

# Create forbidden zone cone for 3D plot
apex = np.array([0, 0, 0])  # Cone apex at origin (host satellite frame)
cone_circle, R_cone = create_cone_surface(apex, sun_unit_avg, sun_avoidance_angle, height=1.0, n_points=100)

# Create figure
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')
ax.set_title(f"3D View: Sun Avoidance\nHost: {host_chosen}, Target: {target_chosen}", fontsize=12)

# Plot trajectories
valid_payload = payload_traj[np.linalg.norm(payload_traj, axis=1) > 0]
valid_los = los_traj if len(los_traj) > 0 else np.array([])

if len(valid_los) > 0:
    ax.scatter(valid_los[:, 0], valid_los[:, 1], valid_los[:, 2], 
               label='Original LOS', alpha=0.3, color='red', s=10)

ax.scatter(valid_payload[:, 0], valid_payload[:, 1], valid_payload[:, 2], 
           label='Sun-Avoided LOS', alpha=0.3, color='blue', s=10)

# Plot Sun vector
ax.quiver(0, 0, 0, sun_unit_avg[0], sun_unit_avg[1], sun_unit_avg[2], 
          color='orange', linewidth=3, arrow_length_ratio=0.15, label='Sun Direction')

# Plot forbidden zone boundary (cone circle)
ax.plot(cone_circle[0], cone_circle[1], cone_circle[2], 
        color='orange', linestyle='--', linewidth=2, label=f'Forbidden Zone ({sun_avoidance_angle}°)')

# Optional: Draw cone surface with semi-transparent patches
n_cone_lines = 12
for i in range(n_cone_lines):
    theta = 2 * np.pi * i / n_cone_lines
    point_on_circle = cone_circle[:, int(len(cone_circle[0]) * i / n_cone_lines)]
    ax.plot([apex[0], point_on_circle[0]], 
            [apex[1], point_on_circle[1]], 
            [apex[2], point_on_circle[2]], 
            color='orange', alpha=0.2, linewidth=0.5)

# Projection plots on walls
ax.plot(valid_payload[:, 1], valid_payload[:, 2], 'g+', alpha=0.1, zdir='x', zs=-1)

# Formatting
ax.set_xlabel("X (ECI)", fontsize=10)
ax.set_ylabel("Y (ECI)", fontsize=10)
ax.set_zlabel("Z (ECI)", fontsize=10)
ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([-0.2, 1.2])
ax.legend(loc='upper right', fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()

# 2D projection plot - CORRECTED
fig2, ax2 = plt.subplots(figsize=(10, 10))

# Calculate angular distance from sun for each avoided LOS point
avoided_angles_yz = []
for i in range(len(valid_payload)):
    angle_from_sun = np.degrees(angle_between(valid_payload[i], sun_unit_avg))
    avoided_angles_yz.append(angle_from_sun)

# Plot the avoided LOS points
scatter = ax2.scatter(valid_payload[:, 1], valid_payload[:, 2], 
                      c=avoided_angles_yz, cmap='viridis', 
                      alpha=0.6, s=20, label='Sun-Avoided LOS')
cbar = plt.colorbar(scatter, ax=ax2)
cbar.set_label('Angle from Sun (deg)', fontsize=10)

# Plot sun direction
ax2.plot(sun_unit_avg[1], sun_unit_avg[2], 'o', color='orange', 
         markersize=15, label='Sun Direction', zorder=10)

# Draw the forbidden zone cone boundary in YZ projection
# This is more complex - we need to find all unit vectors at exactly sun_avoidance_angle from sun
n_circle = 200
phi = np.linspace(0, 2*np.pi, n_circle)

# Create vectors at exactly sun_avoidance_angle from sun_unit_avg
cone_boundary_3d = []
min_angle_rad = np.radians(sun_avoidance_angle)

# Find two orthogonal vectors perpendicular to sun_unit_avg
v1 = np.cross(sun_unit_avg, np.array([1, 0, 0]))
if np.linalg.norm(v1) < 1e-6:
    v1 = np.cross(sun_unit_avg, np.array([0, 1, 0]))
v1 = normalize(v1)

v2 = np.cross(sun_unit_avg, v1)
v2 = normalize(v2)

# Generate points on the cone at the forbidden angle
for p in phi:
    # Perpendicular component that rotates around sun vector
    perp = np.cos(p) * v1 + np.sin(p) * v2
    perp = normalize(perp)
    
    # Combine with sun direction at the specified angle
    boundary_vec = np.cos(min_angle_rad) * sun_unit_avg + np.sin(min_angle_rad) * perp
    boundary_vec = normalize(boundary_vec)
    cone_boundary_3d.append(boundary_vec)

cone_boundary_3d = np.array(cone_boundary_3d)

# Project to YZ plane
ax2.plot(cone_boundary_3d[:, 1], cone_boundary_3d[:, 2], 'r-', 
         linewidth=2.5, label=f'Forbidden Zone Boundary ({sun_avoidance_angle}°)')

# Add some radial lines to show cone structure
n_lines = 16
for i in range(n_lines):
    idx = int(len(cone_boundary_3d) * i / n_lines)
    ax2.plot([sun_unit_avg[1], cone_boundary_3d[idx, 1]], 
             [sun_unit_avg[2], cone_boundary_3d[idx, 2]], 
             'r-', alpha=0.15, linewidth=0.5)

ax2.set_xlabel('Y (ECI)', fontsize=12)
ax2.set_ylabel('Z (ECI)', fontsize=12)
ax2.set_title(f"2D Projection (YZ plane)\nHost: {host_chosen}, Target: {target_chosen}\nAll points should be OUTSIDE red circle", 
              fontsize=12)
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=10, loc='upper right')
ax2.axis('equal')

# Auto-scale with some padding
y_vals = np.concatenate([valid_payload[:, 1], cone_boundary_3d[:, 1]])
z_vals = np.concatenate([valid_payload[:, 2], cone_boundary_3d[:, 2]])
y_range = y_vals.max() - y_vals.min()
z_range = z_vals.max() - z_vals.min()
padding = 0.1 * max(y_range, z_range)

ax2.set_xlim([y_vals.min() - padding, y_vals.max() + padding])
ax2.set_ylim([z_vals.min() - padding, z_vals.max() + padding])

plt.tight_layout()

plt.show()