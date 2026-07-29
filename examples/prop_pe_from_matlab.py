#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os, sys

# Add parent directory to paths
sys.path.insert(1, os.getcwd()[:os.getcwd().index('astropynaric')+13])
import tudat_tools.data_processing.data_processing_utilities as dputil
import basic_tools.time_conversion as t_conv

# Clear environment
plt.close('all')

# Constants
mu = 398600.44  # km^3/s^2
J2 = 1082.6267e-6
R = 6378.1366  # km, Earth Radius

path_cwd = os.getcwd()
csv_path = r'examples/output_data/pointing_error'
csv_output_path = os.path.join(path_cwd, csv_path)

data_raw, simulation_parameters = dputil.load_constellation_data(full_path = csv_output_path)
file_path = os.path.join(csv_output_path, 'state_history.dat')

# Load the data into a DataFrame
sat_data = pd.read_csv(file_path, sep='\t', header = None, comment='#')

host_chosen = simulation_parameters['sat_names'][0]
target_chosen = simulation_parameters['sat_names'][2]

print(f"\nHOST:{host_chosen}\t TARGET: {target_chosen}")

# --- Select which body to propagate: 'HOST' or 'TARGET' ---
propagate_selection = 'TARGET'

# Time arrays
t_j2000 = sat_data.iloc[:,0].to_numpy()
t_gps   = t_conv.j2000_to_gps(t_j2000)

# True states - NOTE: Converting to km (MATLAB divides by 1e3)
LEO_HOST_r = sat_data.iloc[:,simulation_parameters['r_index'][host_chosen]].to_numpy() * 1e-3  # km
LEO_HOST_v = sat_data.iloc[:,[ii+3 for ii in simulation_parameters['r_index'][host_chosen]]].to_numpy() * 1e-3  # km/s

r_target = sat_data.iloc[:,simulation_parameters['r_index'][target_chosen]].to_numpy() * 1e-3  # km
v_target  = sat_data.iloc[:,[ii+3 for ii in simulation_parameters['r_index'][target_chosen]]].to_numpy() * 1e-3  # km/s

# Map selection to chosen and other bodies
if propagate_selection.upper() == 'HOST':
    chosen_name = host_chosen
    r_true = LEO_HOST_r
    v_true = LEO_HOST_v
    r_other = r_target
else:
    chosen_name = target_chosen
    r_true = r_target
    v_true = v_target
    r_other = LEO_HOST_r

# LOS and range (true)
LOS_eci = r_true - r_other
ranges = np.sqrt(LOS_eci[:,0]**2+LOS_eci[:,1]**2+LOS_eci[:,2]**2)

# Simulation parameters
start_point = 2587  # MATLAB uses 2587 for ~250km range

# Propagation parameters
update_no = 2
update_step = 5  # MATLAB uses [10, 20, 30]
prop_duration = 10  # MATLAB uses 10 sec
prop_timeout = start_point + 100
tolerance_km = 10

print(f"Shifting OTHER initial zero position to: {start_point} sec")
print("So for simulation this will be OTHER initial position.")

# MATLAB doesn't filter by range, it uses direct start_point
# Remove the range filtering
r_true = r_true
v_true = v_true
r_other = r_other

print(f"\n{update_no} Updates will come in every {update_step} sec.")
print(f"SDA limiting propagation timeout (considering OTHER position shift by {start_point}): {prop_timeout} sec")
prop_end = prop_duration

# Initialize arrays
step = 1
prop_trajectory = []
LOS_true = []
LOS_prop = []
t1 = []
pos_init = start_point
i = 1
t_init = start_point  # MATLAB: target_delay = start_point (when input is 0)
r_prop_1 = np.zeros((1, 3))
v_prop_1 = np.zeros((1, 3))

# MATLAB: new_data = t_init + [10, 20, 30]
new_data = t_init + np.array([10, 20, 30])

print(new_data)

#%%
def normalize(v):
    return v/np.linalg.norm(v)

def orbit_prop(t, vec_sv):
    r0 = vec_sv[:3]
    v0 = vec_sv[3:6]
    r_norm = max(np.linalg.norm(r0),1e-9)
    a1 = -(mu / r_norm**3) * r0
    
    # J2 Perturbations
    const = -3 * J2 * mu * R**2 / (2 * r_norm**5)
    x, y, z = r0
    ai = const *x* (5 * r0[2]**2 / r_norm**2 - 1)
    aj = const *y* (5 * r0[2]**2 / r_norm**2 - 1)
    ak = const *z* (5 * r0[2]**2 / r_norm**2 - 3)
    acc = np.array([ai, aj, ak])
    a = a1 + acc
    
    return np.hstack([v0, a])

def odeRK(forbit, tspan, x0):
    N = len(tspan)
    n = len(x0)
    x0 = x0.reshape(-1, 1)
    x = np.zeros((N, n))
    x[0, :] = x0.flatten()
    w = x0.flatten()
    for i in range(N-1):
        h = tspan[i+1] - tspan[i]
        t = tspan[i]
        K1 = h * forbit(t, w)
        K2 = h * forbit(t + h/2, w + K1/2)
        K3 = h * forbit(t + h/2, w + K2/2)
        K4 = h * forbit(t + h, w + K3)
        w = w + (K1 + 2*K2 + 2*K3 + K4) / 6
        x[i+1, :] = w
    return tspan, x

# Propagation loop - EXACTLY like MATLAB
while prop_end <= prop_timeout:
    # MATLAB logic for deciding prop_end
    if i <= len(new_data) and new_data[i-1] <= start_point + prop_duration:
        prop_end = new_data[i-1]
        update = True
        print(f'\nNew data will arrive at: {prop_end}, so propagating upto: {prop_end-1} [sec]')
    else:
        prop_duration = start_point + prop_duration - 1
        
        update = False
        print(f'\nNo New data arrived within {prop_duration} sec, so Propagating upto: {prop_end} [sec]')

    print(f't_init: {t_init}')
    
    # MATLAB: tspan = (time(t_init:step:prop_end,1))
    # In Python, we need indices from t_init to prop_end (inclusive in MATLAB)
    tspan = t_gps[t_init:prop_end+1:step]

    # MATLAB condition: if (update) || (i<2)
    if update or i < 2:
        v_sv_op = np.hstack([r_true[t_init], v_true[t_init]])
    else:
        v_sv_op = np.hstack([r_prop_1[-1, :], v_prop_1[-1, :]])

    # Integration
    t, path1 = odeRK(orbit_prop, tspan, v_sv_op)
    print(f'{i} Propagating from {tspan[0]} to {tspan[-1]}')

    r_prop_1 = path1[:, :3]
    v_prop_1 = path1[:, 3:6]

    # MATLAB: LOS_true = vertcat(LOS_true, abs(LEO_HOST_r(start_point:start_point+length(t)-1,:)- r_target(t_init:t_init+length(t)-1,:)));
    # Note: MATLAB uses abs() for element-wise absolute value, creates LOS components
    other_slice = r_other[start_point:start_point+len(t), :]
    true_slice = r_true[t_init:t_init+len(t), :]
    
    los_true_new = np.abs(other_slice - true_slice)
    los_prop_new = np.abs(other_slice - r_prop_1)

    # Vertcat (vstack)
    LOS_true = np.vstack([LOS_true, los_true_new]) if len(LOS_true) > 0 else los_true_new
    LOS_prop = np.vstack([LOS_prop, los_prop_new]) if len(LOS_prop) > 0 else los_prop_new
    t1 = np.hstack([t1, t]) if len(t1) > 0 else t

    # Store for plotting
    aa = other_slice
    bb = true_slice
    cc = r_prop_1

    # Update loop variables - MATLAB order
    i += 1
    t_init = prop_end
    prop_end = t_init + prop_duration
    if prop_end > prop_timeout:
        break
    start_point = prop_end

# Compute LOS pointing errors (µrad) - MATLAB formula
theta_double = np.zeros(len(LOS_prop))
for ii in range(len(LOS_prop)):
    num = np.dot(LOS_prop[ii, :], LOS_true[ii, :])
    den = np.linalg.norm(LOS_prop[ii, :]) * np.linalg.norm(LOS_true[ii, :])
    if den == 0:
        theta_double[ii] = np.nan
    else:
        val = np.clip(num / den, -1.0, 1.0)
        theta_double[ii] = 1e6 * np.arccos(val)

#%%
# Plotting
plt.figure()
plt.plot(t1 - pos_init, theta_double, 'b-o', label='Double Precision')
plt.ylabel('LOS Pointing Error [µrad]')
plt.xlabel('Time [sec]')
plt.title(f'{chosen_name}, and propagation duration: {prop_duration} [sec]')
plt.grid(True)
plt.legend()

# Plot 3D trajectories
fig3d = plt.figure(facecolor='white')
ax3d = fig3d.add_subplot(111, projection='3d')
ax3d.plot(r_true[:, 0], r_true[:, 1], r_true[:, 2], label=f'{chosen_name} True')
ax3d.plot(r_other[start_point:, 0], r_other[start_point:, 1], r_other[start_point:, 2], label='Other')
ax3d.plot(aa[:, 0], aa[:, 1], aa[:, 2], '-o', markersize=10, markerfacecolor='yellow', label='Other (window)')
ax3d.plot(bb[:, 0], bb[:, 1], bb[:, 2], '-o', markersize=10, markerfacecolor='#D9FFFF', label='Target True (window)')
ax3d.set_xlabel('ECI x [km]')
ax3d.set_ylabel('ECI y [km]')
ax3d.set_zlabel('ECI z [km]')
ax3d.set_title('Satellite Orbit in ECI Coordinates')
ax3d.grid(True)
ax3d.legend()

plt.show()
# %%