#%% Imports
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import plotting_functions as pl
from gt_calc import make_lat_long_plot, make_ground_track_plot, calc_gt, dict_2_array, mod360_deg
from old_simulate_ground_track import simulate_ground_track, calc_fov_points, find_gs_in_fov, calc_required_fov, calc_sc_nadir_coord, calc_area_access_el, calculate_gs_visibility, calc_vis_area_point
import time
## Constants
R_E = 6378.136e3 # Equatorial radius of Earth [m]
GM_E = 3.98600441e14 # Earth's gravitational parameter [m^3/s^2]
n_digits = 2 # number of decimal digits for results
#%% GS locations and parameters
# ISS orbit (only handles circular orbits)
h_a = 418 # km
# h_a = 600# km
i_a = 51.64 # degrees
# GS elevation threshold
eps_min = 10 # [deg], minimum elevation to avoid atmospheric distortion
# Instrument FOV
# eta_lct = [45, 69.5] # elevation FOV limits for LCT on ISS [deg] - [up, low]
# phi_lct = [45, -45] # azimuth FOv limits for LCT on ISS [deg] - [left, right]
### ROTATE TERMINAL TO ALLIGN AZIMUTH PLANE WITH Along-track/Radial plane
eta_lct = [0, 84] # elevation FOV limits for LCT on ISS [deg] - [up, low]
phi_lct = [15, -15] # azimuth FOv limits for LCT on ISS [deg] - [left, right]
# phi_lct = np.unwrap(phi_lct, 360)
# Ground station coordinates
h_gs = 0 # [km] altitude of GS 
gs_l3haris= [263.98, 33.02] # long, lat, GS coordinates. Long : {0:360}
gs_ober = [11.28, 48.08]  # oberpfaffenhofen OGS
gs_tene = [343.49, 28.3] # Tenerife 
gs_neme = [22.62, 37.7] # Nemeas, near Athens
gs_cret = [24.90, 35.21] # Crete Skinakas Observatory
gs_alme = [357.45, 37.23] # Almeria Caral Alto
gs_cal = [360-116.06, 34.23] # San Gabriel Mountains California Optical Communications Telescope Laboratory - https://ieeexplore-ieee-org.tudelft.idm.oclc.org/abstract/document/8357216
gs_names = ['L3 Harris, USA',
            'Oberpfaffenhofen, Germany',
            'Tenerife, Spain',
            'Nemeas, Greece',
            'Crete, Greece',
            'Almeria, Spain',
            'Cal, USA']
# Source for GS locations: 
gs_coordinates = [gs_l3haris,
                  gs_ober,
                  gs_tene,
                  gs_neme,
                  gs_cret,
                  gs_alme,
                  gs_cal
                ] # List of GS candidates. Each entry: [long, lat] deg
# visibility source: https://elib.dlr.de/55548/1/OLEO-DL_to_OGS_and_HAPs-IST07.pdf
mean_annual_vis = [
    0.7,
    0.45,
    .71,
    .74,
    .74,
    .64,
    .7
    ]
# Time settings
# T_analysis = 24 # Propagation time [h]
# T_analysis = 1.52*2 # Propagation time [h]
T_analysis = 1.52 # Propagation time [h]
# T_analysis = 24*2 # Propagation time [h]
# T_analysis = 24*30 # Propagation time [h]
# T_analysis = 24*180 # Propagation time [h]
dt = 10 # time-step

gs_index_used = 1 # 1 -> Oberpfaffenhofen
# Plotting conditionals
compute_fov_bounds = True
plot_cond_2d = 1
plot_cond_3d = 0
save_cond_raw_df = 1
save_cond_vis_df = 1
plot_elevation = 0
# Output conditionals
check_for_gs_visibility = False
make_overview_df = True


# select type of pass
pass_type_selected = 'medhigh_elev_68deg' # for 68 degree peak GS elevation (realistic good pass condition)
# pass_type_selected = 'high_elev'
# pass_type_selected = 'low_elev_west'
# pass_type_selected = 'low_elev_east'
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
az_0_high = 231.5 
if pass_type_selected == 'high_elev':
    az_0 = az_0_high 
elif pass_type_selected == 'low_elev_west':
    az_0 = az_0_high - 72 
elif pass_type_selected == 'low_elev_east':
    az_0 = az_0_high + 16.5
elif pass_type_selected == 'medhigh_elev_68deg':
    az_0 = 171.5

## UNCOMMENT FOR MAX ELEVATION CHOICES. Contact time written corresponds to 
# case with CPA Azimuth aligned with Radial-Along track plane
# az_0 = 230 # For 85 deg elev pass -> 180
# az_0 = 229 # For 80 deg elev pass -> 160
# az_0 = 228 # For 78 deg elev pass -> 140
# az_0 = 227 # For 75 deg elev pass -> 130
# az_0 = 226 # For 71 deg elev pass -> 110
# az_0 = 224 # For 65 deg elev pass -> 90
# az_0 = 222 # For 60 deg elev pass -> 60
# az_0 = 220 # For 56 deg elev pass -> 40
# az_0 = 219 # 54.7 peak el -> 30sec
# az_0 = 218 # 53.1 -> 20sec
# az_0 = 217.5 # 52.4 -> 10 sec
# az_0 = 217 # 51.7
# az_0 = 215 # 49
# az_0 = 210 # 45

# az_0 =171.5  ## 66 -> 100
az_0 =200  ## 44.5 -> doesnt hit

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
Orbital revolutions per day: {24*3600/T_analysis:.1f} 
GS altitude: {h_gs} [km]
      ''')
run_time_start = time.time()    
simulated_ground_track = simulate_ground_track(initial_conditions, T_analysis, dt) # Time-vec;[Lat, long [deg]]; heading angles
run_time_end = time.time()
print(f'Ground track simulated for {T_analysis/3600:.1f} hours. Executed in {run_time_end-run_time_start:.1f} s')
t_vec, lat_gt, long_gt= simulated_ground_track[0], simulated_ground_track[1][:,0], simulated_ground_track[1][:,1]
heading_angle_gt = simulated_ground_track[-1]
## Plot groudn track
if 1:
    if plot_cond_2d:
        chosen_gt  = simulated_ground_track[1]
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
#%% 
run_time_start = time.time()   
gs_dict = output_dict_overview['GS']
gs_dict, output_dict_full,  output_dict_raw = calculate_gs_visibility(gs_dict,
                                                                      simulated_ground_track,
                                                                      rho,
                                                                      sin_rho,
                                                                      eps_min,
                                                                      eta_lct,
                                                                      phi_lct,
                                                                      R_E,
                                                                      gs_used = gs_index_used,
                                                                      check_gs_visibility = True,
                                                                      n_digits = n_digits                                                                     
                                                                      )
print(f'GS Visibility calculated for {T_analysis/3600:.1f} hours. Executed in {time.time() -run_time_start:.1f} s')
#%% Process outputs OBERPFAFFENHOFEN ONLY
output_single_raw_df = pd.DataFrame(output_dict_raw)
output_single_raw_df = output_single_raw_df.transpose()
output_single_raw_df.insert(0, 't', output_single_raw_df.pop('t'))
output_single_raw_df['slant_range'] = output_single_raw_df['slant_range']/1e3 # Convert slant range to [km]
output_single_raw_visible = output_single_raw_df[output_single_raw_df['is_visible'] == True]
if save_cond_raw_df:
    gs_coord = gs_dict[gs_index_used]['long/lat']
    try:
        output_name_raw = f'raw_outputs_all_{gs_coord}_{int(T_analysis/3600/24)}d_{h_a}km_IC{az_0}.xlsx' 
        output_single_raw_df.to_excel(f'{save_folder}\\{output_name_raw}')
        print(f'Saved {save_folder}\\{output_name_raw}')
    except:
        print('Raw file too large to save')
    output_name_vis = f'raw_outputs_visible_{gs_coord}_{int(T_analysis/3600/24)}d_{h_a}kmIC{az_0}.xlsx'
    output_single_raw_visible.to_excel(f'{save_folder}\\{output_name_vis}')
    el_max = np.max(output_single_raw_visible['gs_elevation'])
    print(f'PEAK GS ELEVATION: {el_max:.1f} deg')
    t_obs_vec = output_single_raw_visible[output_single_raw_visible['is_observable']==True]['t'].values
    if len(t_obs_vec) > 1:
        contact_time = t_obs_vec[-1] - t_obs_vec[0]
    elif len(t_obs_vec) == 1:
        contact_time = 5
    else:
        contact_time = 0
    print(f'Contact time: {contact_time} s (only applicable if sim time<orbital period of ISS)')
    print(f'Saved {save_folder}\\{output_name_vis}')
#%% Generate FOV bounds for each grid point- Verification purposes 
if compute_fov_bounds:

    t_vec_obs = []
    n_points = 20
    for t in output_dict_full.keys():
        n_gs = output_dict_full[t]['n_observable_gs']
        n_vis = output_dict_full[t]['n_visible_gs']
        n_fov = output_dict_full[t]['n_fov_gs']
        # if n_gs > 0 or n_vis>0 or n_fov>0:
        if n_vis > 0:
            # print(f't={t} observ={n_gs}, vis = {n_vis}, fov = {n_fov}')
            t_vec_obs.append(t)
    t_vec_obs_used = [t_vec_obs[ii] for ii in np.linspace(0, len(t_vec_obs)-1, n_points).astype(int)]
    print('Computing FOV bounds.')

    # print('-set to False if running for multiple ground stations.')
    # print('-set to True if running for a single GS to speed up computations. (but change gs coordinate input)')
    output_dict = calc_required_fov(simulated_ground_track, 
                                    gs_l3haris, 
                                    eta_lct,
                                    phi_lct,
                                    sin_rho_gs,
                                    rho_gs,
                                    check_for_gs_visibility, 
                                    t_vec_obs_used)


#%% Plots
chosen_gt  = simulated_ground_track[gs_index_used]
if plot_cond_3d:
    t_plotted = [t_vec_obs_used[0], t_vec_obs_used[4],t_vec_obs_used[-1]]
    for jj, t in enumerate(t_plotted):
        ii = int(t/10)
        
        n_obs_gs = output_dict_full[t]['n_observable_gs']
        n_fov_gs = output_dict_full[t]['n_fov_gs']
        n_access_gs = output_dict_full[t]['n_visible_gs']
        chosen_gt = simulated_ground_track[1][ii-10:ii,:]
        fig_3d, ax_3d = pl.orbit_projection([gs_coordinates[1]],
                                            [chosen_gt[:,1],chosen_gt[:,0]] ,
                                            output_dict[t]['fov_bounds'],
                                            [output_dict[t]['vis_area']],
                                            title_f = f' obs_gs = {n_obs_gs}, n_FOV = {n_fov_gs}, n_access = {n_access_gs} ',
                                            )
        plt.show()