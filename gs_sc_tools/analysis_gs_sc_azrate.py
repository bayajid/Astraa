#%% Imports
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import sys
sys.path.insert(1, os.getcwd()[:os.getcwd().index('astropynaric')+13])
from gs_sc_tools.gt_calc import make_lat_long_plot, make_ground_track_plot, calc_gt, dict_2_array, mod360_deg
from gs_sc_tools.gt_simulation_tools import simulate_ground_track, calc_fov_points, find_gs_in_fov, calc_required_fov, calc_sc_nadir_coord, calc_area_access_el, calculate_gs_visibility, calc_vis_area_point
import gs_sc_tools.GS_pass_calc_tools as gs_pass_tools
import gs_sc_tools.GS_coordinates as gs_coord
import time
## Constants
R_E = 6378.136e3 # Equatorial radius of Earth [m]
GM_E = 3.98600441e14 # Earth's gravitational parameter [m^3/s^2]
n_digits = 2 # number of decimal digits for results
#%% GS locations and parameters

h_a = 500 # km
i_a = 89 # degrees
# h_a = 600# km
# GS elevation threshold
eps_min = 10 # [deg], minimum elevation to avoid atmospheric distortion
# Instrument FOV
h_gs = 0 # [km] altitude of GS 
## GS coordinates/names from 
gs_names = gs_coord.gs_names
gs_coordinates = gs_coord.gs_coord
mean_annual_vis = gs_coord.mean_annual_vis
# Time settings
# T_analysis = 24 # Propagation time [h]
# T_analysis = 1.52*2 # Propagation time [h]
# T_analysis = 1.52 # Propagation time [h]
# T_analysis = 24*2 # Propagation time [h]
# T_analysis = 24*30 # Propagation time [h]
# T_analysis = 24*10 # Propagation time [h]
T_analysis = 24*180 # Propagation time [h]
dt = 5 # time-step

gs_index_used = 1 # 1 -> Oberpfaffenhofen
# Plotting conditionals
compute_fov_bounds = True
make_interm_plots = 1
plot_cond_2d = 1
plot_cond_3d = 0
save_cond_raw_df = 1
save_cond_vis_df = 1
plot_elevation = 0
# Output conditionals
check_for_gs_visibility = False
make_overview_df = True
# LCT limits (unused for general viewing angles)
eta_lct = [0, 90] # elevation FOV limits for LCT on ISS [deg] - [up, low]
phi_lct = [180, -180] # azimuth FOv limits for LCT on ISS [deg] - [left, right]

#%% Paths and constants
# Save paths
save_folder = 'simulation output'
save_name_incl = 'contact_times'

#%% Computing simulation parameters
sin_rho = R_E / (R_E + h_a * 1e3) # earth angle calculation
sin_rho_gs = (R_E +h_gs*1e3) / (R_E + h_a*1e3) # gs earth angle, in case altitude is higher
rho = np.arcsin(sin_rho) # rad, observable Earth angle
rho_gs = np.arcsin(sin_rho_gs)
### Simulate ground-track of ISS
a = R_E+h_a*1e3 # semi-major axis [m]
n = np.rad2deg(np.sqrt(GM_E/a**3)) # mean motion [deg/s]
T_orbit = int(360/n) # orbital period [s]
T_analysis = T_analysis * 3600 # [s]

## Setup IC to give high/low elevation passes
#initial longitude of orbital pole Initial Condition setting
az_0 = 231.5 

# create disctionary of initial conditions for ground track computation
initial_conditions = {}
initial_conditions['a'] = a # semi-major axis
initial_conditions['h'] = h_a # altitude 
initial_conditions['az_0'] = az_0 # initial longitude of orbital pole
initial_conditions['phi_s'] = 180 # initial azimuth from ascending node
initial_conditions['n'] = n # mean orbital motion
initial_conditions['lat_0'] = - i_a # initial latitude of sub-satellite point 
initial_conditions['i'] = i_a # inclination
initial_conditions['eta'] = eta_lct
#%% Simulate ground track
print(f'''
Analysis performed for current settings:
Analysis time: {T_analysis/3600:.1f} hours. 
Time-step: {dt:.1f} seconds
altitude: {h_a:.1f} km
inclination: {i_a:.2f} deg
Orbital revolutions per day: {24*3600/T_orbit:.1f} 
GS altitude: {h_gs} [km]
      ''')
run_time_start = time.time()    
sim_gt = simulate_ground_track(initial_conditions, T_analysis, dt) # Time-vec;[Lat, long [deg]]; heading angles
run_time_end = time.time()
print(f'Ground track simulated for {T_analysis/3600:.1f} hours. Executed in {run_time_end-run_time_start:.1f} s')
t_vec, lat_gt, long_gt= sim_gt[0], sim_gt[1][:,0], sim_gt[1][:,1]
heading_angle_gt = sim_gt[-1]
## Plot groudn track
#%%
if make_interm_plots:
    if plot_cond_2d:
        chosen_gt  = sim_gt[1]
        plot_glossary = {}
        plot_glossary['xlabel'] = 'Longitude [deg]'
        plot_glossary['ylabel'] = 'Latitude [deg]'
        plot_glossary['title'] = f'Ground track for LCT on ISS for {T_analysis/3600:.1f} hours'        
        fig, ax = make_ground_track_plot(chosen_gt[:,1], chosen_gt[:,0], gs_coordinates, plot_glossary)
    t_vec_saved = t_vec/86000
    gt_data = pd.DataFrame(data = np.hstack((t_vec_saved.reshape((len(t_vec),1)), chosen_gt[:,[1]], chosen_gt[:,[0]], heading_angle_gt)), columns = ['t_jd', 'long', 'lat','heading'])
    gt_data.to_csv('verification/gt_odcm.csv', index = False)
    t_vec_plot = t_vec / 86400
    f, ax = plt.subplots()
    heading_angle_gt_wrapped = [angle - 360 if angle > 180 else angle for angle in heading_angle_gt]
    ax.plot(t_vec_plot, heading_angle_gt_wrapped, label = 'Heading')
    ax.plot(t_vec_plot, long_gt, label = 'Long')
    ax.plot(t_vec_plot, lat_gt, label = 'Lat')
    ax.set_ylabel('Angles [deg]')
    ax.grid()
    ax.legend()
#%% Summarize GS data in a dictionary
# Output dictionaries for the new method
output_dict_full = {} # Data for each time step and GS
output_dict_overview = {} # data for each GS visibility
output_dict_overview['GS'] = {}
# Input ground station coordinates/labels for corresponding index
for ii, gs_coord in enumerate(gs_coordinates):
    output_dict_overview['GS'][ii] = {}
    output_dict_overview['GS'][ii]['label'] = gs_names[ii]
    output_dict_overview['GS'][ii]['long/lat'] = gs_coord
    output_dict_overview['GS'][ii]['mean_annual_availability'] = mean_annual_vis[ii]
run_time_start = time.time()   
t_nr_days = t_vec.flatten()/86400+1
gt_df = pd.DataFrame.from_dict(
    {
        't_days' : t_vec.flatten()/86400,
        't' : t_nr_days.astype(int),
        'long' : long_gt.flatten(),
        'lat' : lat_gt.flatten(),
        'heading' : heading_angle_gt.flatten(),
    }
)
# importlib.reload(gs_pass_tools)
output_dict, gs_azimuth = gs_pass_tools.process_gt_to_passes(gt_df, tit_app = f'h{h_a}km_T{T_analysis/86400:.0f}d_')

#%% DEBUGGING STUFF
if 0:
    ii_end = -1
    gs_az = gs_azimuth['gs_azimuth']
    gs_az_rate = gs_azimuth['gs_azimuth_rate']
    gs_el = gs_azimuth['gs_elevation']
    t = gs_azimuth['t'].iloc[:ii_end].values
    t = t - t[0]

    gs_az = gs_az.iloc[:ii_end]
    gs_el = gs_el.iloc[:ii_end]
    gs_az_rate = gs_az_rate.iloc[:ii_end]

    f, axs = plt.subplots(nrows = 3)

    axs[0].scatter(t, gs_az, label = 'az, gs')
    axs[0].set_ylabel('Az')
    axs[1].scatter(t, gs_az_rate)
    axs[1].set_ylabel('Az rate')
    axs[2].scatter(t, gs_el, label = 'el')
    axs[2].set_ylabel('Elevation')
    axs[2].set_ylim([0,100])