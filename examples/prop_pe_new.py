#%%
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os, sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# Constants
R_EARTH = 6378.1366  # km, Earth Radius
DEG2RAD = np.pi / 180
RAD2DEG = 180 / np.pi
ANG_THRESHOLD = 10.0  # degrees, angular threshold for Sun in background
AU = 149597870.7  # km, 1 AU
SUN_VECTOR_SCALE = 20000  # km, length for Sun vectors in plot
MAX_VECTORS = 50  # Limit number of vectors plotted per direction

def julian_date(dt):
    """Compute Julian Date from datetime object"""
    # Reference: Astronomical Algorithms by Jean Meeus
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
    # Simplified model: mean ecliptic longitude, neglecting nutation and aberration
    T = (jd - 2451545.0) / 36525.0  # Julian centuries since J2000.0
    L0 = 280.46646 + 36000.76983 * T + 0.0003032 * T**2  # Mean longitude (degrees)
    M = 357.52911 + 35999.05029 * T - 0.0001537 * T**2  # Mean anomaly (degrees)
    L0 = L0 % 360
    M = M % 360
    e = 0.016708634 - 0.000042037 * T  # Eccentricity
    E = M + e * np.sin(M * DEG2RAD) * (1 + e * np.cos(M * DEG2RAD))  # Eccentric anomaly (degrees, approximate)
    x_helio = np.cos(E * DEG2RAD) - e
    y_helio = np.sqrt(1 - e**2) * np.sin(E * DEG2RAD)
    r = np.sqrt(x_helio**2 + y_helio**2) * AU  # Distance to Sun (km)
    lon = (L0 + np.arctan2(y_helio, x_helio) / DEG2RAD) % 360
    # Ecliptic to ECI: obliquity of ecliptic ~23.44 degrees
    eps = 23.44 * DEG2RAD
    x_eci = r * np.cos(lon * DEG2RAD)
    y_eci = r * (np.sin(lon * DEG2RAD) * np.cos(eps))
    z_eci = r * (np.sin(lon * DEG2RAD) * np.sin(eps))
    return np.array([x_eci, y_eci, z_eci])

def check_earth_occultation(r_a, r_b):
    """Check if Earth occults the LOS from r_a to r_b"""
    # Line segment from r_a to r_b: r(t) = r_a + t * (r_b - r_a), t in [0, 1]
    d = r_b - r_a
    # Closest point to Earth's center (origin) occurs when dot(r_a + t*d, d) = 0
    t = -np.dot(r_a, d) / np.dot(d, d)
    # If t < 0, closest point is r_a; if t > 1, closest point is r_b
    if t < 0:
        r_closest = r_a
    elif t > 1:
        r_closest = r_b
    else:
        r_closest = r_a + t * d
    # Check if closest point is within Earth's radius
    return np.linalg.norm(r_closest) > R_EARTH

# Read data
#file_path = r"C:\Users\BKhan\OneDrive - Mynaric AG\Documents\Python Scripts\state_history.dat"
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


host_chosen = simulation_parameters['sat_names'][0]
target_chosen = simulation_parameters['sat_names'][1]

print(f"Host: {host_chosen}, \nTarget:{target_chosen}")
# Extract HOST and CROSS TARGET positions (km)

LEO_HOST_r = statehistory.iloc[:,simulation_parameters['r_index'][host_chosen]].to_numpy()   
LEO_CROSS_TARGET_r  = statehistory.iloc[:,simulation_parameters['r_index'][target_chosen]].to_numpy()                  #[data_raw[:,simulation_parameters['r_index'][target_chosen]]
time = statehistory.iloc[:, 0].values - statehistory.iloc[0, 0]

# Time reference: assume 2022-11-14 09:16:59 UTC
start_dt = datetime(2022, 11, 14, 9, 16, 59)
jd_start = julian_date(start_dt)
jd_times = jd_start + time / 86400.0  # Convert seconds to days

# Initialize results
sun_in_background_host = []  # (index, time, angle)
sun_in_background_target = []

# Compute Sun positions and check angles
for i in range(len(time)):
    sun_pos = sun_position_eci(jd_times[i])
    
    # HOST viewing CROSS TARGET
    los_h2t = LEO_CROSS_TARGET_r[i] - LEO_HOST_r[i]
    vec_h2sun = sun_pos - LEO_HOST_r[i]
    angle_h2t_sun = np.arccos(np.dot(los_h2t, vec_h2sun) / (np.linalg.norm(los_h2t) * np.linalg.norm(vec_h2sun))) * RAD2DEG
    if angle_h2t_sun <= ANG_THRESHOLD and check_earth_occultation(LEO_HOST_r[i], LEO_CROSS_TARGET_r[i]):
        sun_in_background_target.append((i, time[i], angle_h2t_sun, los_h2t, vec_h2sun))
    
    # CROSS TARGET viewing HOST
    los_t2h = LEO_HOST_r[i] - LEO_CROSS_TARGET_r[i]
    vec_t2sun = sun_pos - LEO_CROSS_TARGET_r[i]
    angle_t2h_sun = np.arccos(np.dot(los_t2h, vec_t2sun) / (np.linalg.norm(los_t2h) * np.linalg.norm(vec_t2sun))) * RAD2DEG
    if angle_t2h_sun <= ANG_THRESHOLD and check_earth_occultation(LEO_CROSS_TARGET_r[i], LEO_HOST_r[i]):
        sun_in_background_host.append((i, time[i], angle_t2h_sun, los_t2h, vec_t2sun))

# Print results
# print("\nSun in background of CROSS TARGET (from HOST perspective):")
# if sun_in_background_target:
#     for idx, t, angle, _, _ in sun_in_background_target:
#         print(f"Index: {idx}, Time: {t:.3f} s, Angle: {angle:.3f} deg")
# else:
#     print("No instances found.")

# print("\nSun in background of HOST (from CROSS TARGET perspective):")
# if sun_in_background_host:
#     for idx, t, angle, _, _ in sun_in_background_host:
#         print(f"Index: {idx}, Time: {t:.3f} s, Angle: {angle:.3f} deg")
# else:
#     print("No instances found.")

#%%
# 3D Plot
fig = plt.figure(figsize=(10, 8), facecolor='white')
ax = fig.add_subplot(111, projection='3d')

# Plot Earth as wireframe sphere
u = np.linspace(0, 2 * np.pi, 20)
v = np.linspace(0, np.pi, 20)
x = R_EARTH * np.outer(np.cos(u), np.sin(v))
y = R_EARTH * np.outer(np.sin(u), np.sin(v))
z = R_EARTH * np.outer(np.ones(np.size(u)), np.cos(v))
ax.plot_wireframe(x, y, z, color='lightgray', alpha=0.3, label='Earth')

# Plot trajectories
ax.plot3D(LEO_HOST_r[:, 0], LEO_HOST_r[:, 1], LEO_HOST_r[:, 2], 'b', label='LEO HOST')
ax.plot3D(LEO_CROSS_TARGET_r[:, 0], LEO_CROSS_TARGET_r[:, 1], LEO_CROSS_TARGET_r[:, 2], 'r', label='LEO CROSS TARGET')

# Plot LOS and Sun vectors (limit to MAX_VECTORS)
print("\nPlotting Sun vectors for up to", MAX_VECTORS, "instances per direction")
for i, (idx, t, angle, los, sun_vec) in enumerate(sun_in_background_target[:MAX_VECTORS]):
    # HOST to CROSS TARGET LOS
    start = LEO_HOST_r[idx]
    end = LEO_HOST_r[idx] + los
    ax.plot3D([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], 'g-', label='LOS (HOST to TARGET)' if i == 0 else '')
    # HOST to Sun
    sun_end = LEO_HOST_r[idx] + sun_vec / np.linalg.norm(sun_vec) * SUN_VECTOR_SCALE
    print(f"HOST to Sun (t={t:.3f} s): Start={start}, End={sun_end}")
    ax.plot3D([start[0], sun_end[0]], [start[1], sun_end[1]], [start[2], sun_end[2]], 'y-', linewidth=2, label='HOST to Sun' if i == 0 else '')
    # Satellite positions
    ax.scatter3D([start[0]], [start[1]], [start[2]], c='b', s=50, marker='o')
    ax.scatter3D([end[0]], [end[1]], [end[2]], c='r', s=50, marker='o')

for i, (idx, t, angle, los, sun_vec) in enumerate(sun_in_background_host[:MAX_VECTORS]):
    # CROSS TARGET to HOST LOS
    start = LEO_CROSS_TARGET_r[idx]
    end = LEO_CROSS_TARGET_r[idx] + los
    ax.plot3D([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], 'c-', label='LOS (TARGET to HOST)' if i == 0 else '')
    # CROSS TARGET to Sun
    sun_end = LEO_CROSS_TARGET_r[idx] + sun_vec / np.linalg.norm(sun_vec) * SUN_VECTOR_SCALE
    print(f"TARGET to Sun (t={t:.3f} s): Start={start}, End={sun_end}")
    ax.plot3D([start[0], sun_end[0]], [start[1], sun_end[1]], [start[2], sun_end[2]], 'm-', linewidth=2, label='TARGET to Sun' if i == 0 else '')
    # Satellite positions
    ax.scatter3D([start[0]], [start[1]], [start[2]], c='r', s=50, marker='o')
    ax.scatter3D([end[0]], [end[1]], [end[2]], c='b', s=50, marker='o')

ax.set_xlabel('ECI x [km]')
ax.set_ylabel('ECI y [km]')
ax.set_zlabel('ECI z [km]')
ax.set_title('Satellite Trajectories with LOS and Sun Vectors')
ax.legend()
ax.grid(True)
# plt.show()

plt.figure()
angles_h2t = [np.arccos(np.dot(LEO_CROSS_TARGET_r[i] - LEO_HOST_r[i], sun_position_eci(jd_times[i]) - LEO_HOST_r[i]) / (np.linalg.norm(LEO_CROSS_TARGET_r[i] - LEO_HOST_r[i]) * np.linalg.norm(sun_position_eci(jd_times[i]) - LEO_HOST_r[i]))) * RAD2DEG for i in range(len(time))]
plt.plot(time, angles_h2t)
plt.xlabel('Time [s]')
plt.ylabel('Angle to Sun [deg]')
plt.grid(True)

plt.show()
# %%
