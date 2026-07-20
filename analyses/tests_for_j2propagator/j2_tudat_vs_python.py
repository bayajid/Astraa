
## Templates for loading satellite data
# generating attitude
# and whatnot. 

## IMPORTS
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os 
import datetime
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import importlib
import os, sys
sys.path.insert(1, os.getcwd()[:os.getcwd().index('astropynaric')+13])

# path jazz
path_cwd = os.getcwd()
csv_output_path = 'examples/output_data/pointing_error'#'orbital_simulations/single_sat/leo_hp_prop_1s_1d'
# fname_simparam = 'simulation_parameters.json'
# fname_states = 'state_history.dat'
# Constants
mu = 398600.44  # km^3/s^2
J2 = 1082.6267e-6
R = 6378.1366  # km, Earth Radius
## MVP imports
import basic_tools.vector_operations as vec_calc
import astronomy_tools.constants as const
import astronomy_tools.astro_targets as where_sun
import plotting_tools.basic_plotting as bplt
import basic_tools.time_conversion as t_conv
import basic_tools.in_out as out
import prediction_methods.j2propagator as j2prop
## LIST COMPREHENSION ft. if else
# [x+1 if x >= 45 else x+5 for x in l]

## Loading satellite orbital data
#importlib.reload(j2prop)
import tudat_tools.data_processing.data_processing_utilities as dputil
data_raw, simulation_parameters = dputil.load_constellation_data(full_path = csv_output_path)

host_chosen = simulation_parameters['sat_names'][4]
# host_chosen = 'leo_target_polar'
# host_chosen = 'leo_host_incl'
# host_chosen = 'meo_target_incl'

def orbit_prop(t, vec_sv):
    r0 = vec_sv[:3]
    v0 = vec_sv[3:6]
    r_norm = np.linalg.norm(r0)
    a1 = -(mu / r_norm**3) * r0
    
    # J2 Perturbations
    const = -3 * J2 * mu * R**2 / (2 * r_norm**5)
    ai = const * (5 * r0[2]**2 / r_norm**2 - 1)
    aj = const * (5 * r0[2]**2 / r_norm**2 - 1)
    ak = const * (5 * r0[2]**2 / r_norm**2 - 3)
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


t_j2000 = data_raw[:,0]
t_gps = t_j2000 + t_conv.dt_j2000tt2gps()
r_host = data_raw[:,simulation_parameters['r_index'][host_chosen]]
v_host = data_raw[:,[ii+3 for ii in simulation_parameters['r_index'][host_chosen]]]
import tudat_tools.tudat_converter as tudatconv

# Setup j2 propagator
ii_0 = 0
smpl = 1
link_distance = 100e3
dt = t_j2000[1] - t_j2000[0]  # Original time step
downsample_factor = int(smpl / dt)  # Calculate factor for every 10 seconds (e.g., 10 if dt=1)

t_0 = t_j2000[ii_0]
r_0 = r_host[ii_0, :]  # Use the first element for consistency
v_0 = v_host[ii_0, :]
X = np.hstack((r_0, v_0))
r_host_predicted = j2prop.propagate_orbit(X, t_0, t_j2000[-1], downsample_factor)  # downsample_factor as step

# Corrected line: Slice t_j2000 with the step to create tspan
tspan = t_j2000[::downsample_factor]  # Starts from 0, ends at the last element, steps by downsample_factor

v_sv_op = np.hstack([r_0, v_0])
t, path1 = odeRK(orbit_prop, tspan, v_sv_op)  # Now uses the corrected tspan

# Downsample r_host and t_j2000 to every 10 seconds for consistency
indices_to_keep = np.arange(0, len(t_j2000), downsample_factor)  # e.g., [0, 10, 20, ...]
r_host_downsampled = r_host[indices_to_keep, :]  # Downsample r_host
t_j2000_downsampled = t_j2000[indices_to_keep]  # Downsample t_j2000  (Note: You might not need this if using tspan)

# Now, calculate dr using the downsampled data
dr = r_host_downsampled - r_host_predicted[:len(r_host_downsampled), [1, 2, 3]]  # Calculate dr with downsampled data

data_to_store = np.hstack((r_host_predicted, np.linalg.norm(dr, axis=1).reshape(len(dr), 1)))
data_to_store[:, 0] = t_gps[indices_to_keep]  # Update t_gps to match downsampled times
columns = ['t_gps', 'x', 'y', 'z', 'vx', 'vy', 'vz', 'dr_exp']
out.make_n_save('j2_prop_testdata', data=data_to_store, data_cols=columns, subfolder='j2prop_testdata')

#%% plot 
f, ax = plt.subplots()
ax.plot(t_j2000_downsampled - t_j2000_downsampled[0], np.linalg.norm(dr, axis=1))  # Use downsampled time array
ax.set_ylabel('dr [m]')
ax.set_xlabel('t [s]')

theta = np.arctan2(np.linalg.norm(dr, axis=1), link_distance)
dr_point = []
LOS_true = np.empty((0, 3))
LOS_prop = np.empty((0, 3))

f1, ax = plt.subplots()
ax.plot(t_j2000_downsampled - t_j2000_downsampled[0], theta * 1e6, 'C0')  # Use downsampled time array
ax.set_ylabel(r'$\theta$ [µrad]')
ax.set_xlabel('t [s]')
ax.grid()

for i in range(len(t_j2000_downsampled)):  # Loop over the downsampled length
    direction = r_host_downsampled[i] / np.linalg.norm(r_host_downsampled[i])  # Use downsampled r_host
    point_in_space = r_host_downsampled[i] - (link_distance * direction)  # Subtract vector of length link_distance
    LOS_true = np.vstack([LOS_true, (point_in_space - r_host_downsampled[i])])
    LOS_prop = np.vstack([LOS_prop, (point_in_space - r_host_predicted[i, [1, 2, 3]])])

# Normalize the vectors
LOS_true_norm = LOS_true / np.linalg.norm(LOS_true, axis=1, keepdims=True)
LOS_prop_norm = LOS_prop / np.linalg.norm(LOS_prop, axis=1, keepdims=True)

# Calculate theta using the dot product of normalized vectors
dot_product = np.einsum('ij,ij->i', LOS_prop_norm, LOS_true_norm)  # Efficiently compute dot products
dot_product = np.clip(dot_product, -1.0, 1.0)  # Clip to avoid invalid values
theta_second = 1e6 * np.arccos(dot_product)  # Renamed for clarity

ax.plot(t_j2000_downsampled - t_j2000_downsampled[0], theta_second, 'C1')  # Use downsampled time array
ax.set_ylabel(r'$\theta$ [µrad]')  # Only set labels once if needed
ax.set_xlabel('t [s]')
ax.grid()

plt.show()