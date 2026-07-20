#%% August 21- quantify impact of error sources
# eg host/moon pos, host att, 
import scipy as sp
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
csv_output_path = r'orbital_simulations\tudat_raw_states'
fname_simparam = 'simulation_parameters.json'
fname_states = 'state_history.dat'

## MVP imports
import basic_tools.vector_operations as vec_calc
import basic_tools.time_conversion as t_conv
import plotting_tools.basic_plotting as bplt

import astronomy_tools.constants as const
import astronomy_tools.astro_targets as where_sun

import attitude_tools.attitude_resolution as att_res
import attitude_tools.conversions as conv
import attitude_tools.rotations as rot
import attitude_tools.attitude_simulation as att_sim

import basic_tools.in_out as savedat

import pointing_calculations.ae_calculation as ae_calc
import pointing_calculations.simulate_moon_scan as moon_scan
# Moon calibration condition file
calibration_time_path = r'outputs\tables\moon_conops_conditions\final_overview'
position_path = r'outputs\tables\moon_conops_conditions\filtered'


# pass_overviews = pd.read_csv(calibration_time_path)

t_used = '62d'
chosen_index = 2
files_all = os.listdir(calibration_time_path)
files_all = [f for f in files_all if t_used in f]
# sys.exit()
files_filtered = [ii for ii in files_all if t_used in ii]
for ii, file in enumerate(files_filtered):
    print(f'{ii} -> {file}')
fname_passdata = files_filtered[chosen_index]
sat_host = fname_passdata[:fname_passdata.index('_62')]

fname_positions = [ii for ii in os.listdir(position_path) if sat_host in ii][0]

path_data = fr'{calibration_time_path}/{fname_passdata}'
path_positions = fr'{position_path}/{fname_positions}'
print(f'\nInd = {chosen_index} -> {fname_passdata}')
if 'sat_leo_eq' in sat_host:
    sat_name = 'LEO 1000 km altitude, Equatorial.'
elif 'sat_leo_incl' in sat_host:
    sat_name = 'LEO 1000 km altitude, Inclined 53 deg.'
elif 'sat_leo_polar' in sat_host:
    sat_name = 'LEO 1000 km altitude, Near-Polar 89 deg.'
elif 'sat_meo' in sat_host:
    sat_name = 'MEO 13880 km altitude, Equatorial.'

dat_pass = pd.read_csv(path_data)
dat_pos = pd.read_csv(path_positions)
print(f'---Loaded \"{fname_passdata}\" AND \"{fname_positions}\"---\n')
#%% Unpack all required inputs based on chosen pass index
pass_chosen = 1
add_noise = 1
pass_stats = dat_pass[dat_pass['pass_nr'] == pass_chosen]
pass_position_data = dat_pos[dat_pos['pass_nr'] == pass_chosen]
print(f'''
Chosen pass nr : {pass_chosen}
stats : {pass_stats}

Number of rows available : {pass_position_data.shape[0]}
----Noise USAGE---- : {bool(add_noise)}
      ''')

r_host = pass_position_data.iloc[:,[1,2,3]].values
v_host = pass_position_data.iloc[:,[-4, -3, -2]].values
t_gps = pass_position_data.iloc[:,[0]].values
r_moon = pass_position_data.iloc[:,[4,5,6]].values
illumination = pass_position_data.iloc[:,[7]].values
dt_data = t_gps[1] - t_gps[0]
# Get attitude
# constrain_azel = 0
constrain_attitude = 1
run_mo_resolution = 1
nrows = r_host.shape[0]

mounting_offset_rpy = [5, 4, 2.5] # MOUNTING OFFSET random 3-axis rotation
rot_bf2lct = conv.convert_ea2dcm(mounting_offset_rpy)
quat_mounting_offset = conv.convert_ea2quat(mounting_offset_rpy)

# Placeholders for commanded attitude
ea_eci2bf_command_all = np.zeros((nrows, 6)) # EA; EA_rate [deg, deg/s]
quat_eci2bf_command_all = np.zeros((nrows, 4)) # scalar-first
make_3d_plot = 1 
scan_gap = 1
ii_ae = 0 
interp_dat = 1
ii_scans = [1]
ii_scans.append(ii_scans[0] + scan_gap)

# TODO select attitude profile
importlib.reload(att_sim)
importlib.reload(conv)
## COMPUTE ATTITUDE

# attitude_profiles = ['earth_point', 'earth_roll', 'sun_roll', 'sun_point']
attitude_profiles = ['sun_roll']
# attitude_profiles = ['sun_point']
# attitude_profiles = ['earth_point']
# attitude_profiles = ['earth_roll']
print(f'Using defined attitude profiles : {attitude_profiles}')
attitude_eci2bf = []

for ii, att_prof in enumerate(attitude_profiles):
    quat_eci2bf, quat_dot, rot_eci2bf = att_sim.calc_quat_eci2bf(r_host, v_host, att_prof, t_gps = t_gps.flatten())
    
    attitude_eci2bf.append(quat_eci2bf)
for ii, dcm_ii in enumerate(rot_eci2bf):
    ea_eci2bf_command_all[ii,:3] = conv.convert_dcm2ea(dcm_ii)
    
ea_eci2bf_command_all[:,3] =  np.gradient(ea_eci2bf_command_all[:,0], t_gps.flatten())
ea_eci2bf_command_all[:,4] =  np.gradient(ea_eci2bf_command_all[:,1], t_gps.flatten())
ea_eci2bf_command_all[:,5] =  np.gradient(ea_eci2bf_command_all[:,2], t_gps.flatten())
dt_required = 1


t_gps_fine = t_gps
r_host_fine = r_host
v_host_fine = v_host
r_moon_fine = r_moon
illumination_fine = illumination
ea_eci2bf_fine = ea_eci2bf_command_all


check_att_impact = 0
check_centroid_impact = 0
compute_nonlinearity_impact = 1
# lump random errors into centroid detection error term
pe_enc = 0.4
pe_th = 1
pe_rxpath = 0.1
pe_other = 0.1
pe_centroid = 2
pe_random = [pe_enc, pe_th, pe_rxpath, pe_other, pe_centroid]
pe_random_rss = np.linalg.norm(pe_random) # mrad

if 1: # scans and plots
    ii_scans = [0, 20]
    importlib.reload(moon_scan)
    ## SImulate seen and commanded Moon Azimuth/Elevation and peform mounting offset
    # resolution
    # Use fully interface code

    err_fact = 0.5
    err_per_axis_0 = 0.05
    factors_tried = np.logspace(10, -2, 100)
    # [10, 5, 3, 2, 1, 0.9, 0.8, 0.7, 0.5, 0.2, 0.1, 0.05]
    # try all error factor
    pe_mo = []
    err_att = []

    for ii, fac in enumerate(factors_tried):
        ae_moon_commanded_all, ae_moon_true_all, quat_resolved, po_ii, angle_between_los = moon_scan.simulate_moon_calib(
            ii_scans = ii_scans,
            r_host = r_host, 
            r_moon = r_moon, 
            ea_eci2bf_command_all = ea_eci2bf_command_all,
            quat_mounting_offset_t = quat_mounting_offset,
            quat_mounting_offset_c = None,
            add_noise = 1,
            add_centroid_error = 1,
            add_r_host_error = 1,
            add_r_moon_error = 1,
            add_att_host_error = 1,
            att_noise_factor= fac,
            print_cond = 1,
            print_full = 1
        )
        pe_mo.append(np.max(po_ii)/1e3) # mrad
        err_att.append(err_per_axis_0/fac) # mrad
    pe_mo = np.array(pe_mo)
    pe_mo = pe_mo.reshape((pe_mo.shape[0], 1))
    err_att = np.array(err_att)
    err_att = err_att.reshape((err_att.shape[0], 1))
    err_df = pd.DataFrame(data = np.hstack((err_att, pe_mo)), columns = ['err_att_mrad', 'pe_mrad'])
    if 0:
        err_df.to_csv(fr'outputs\tables\moon_mo_res_errors\Att_{att_prof}.csv', index = 0)
        f, ax = plt.subplots()
        ax.plot(err_att, pe_mo, '-o',label = att_prof)
        ax.set_xlabel('Attitude Error Per Axis [mrad]')
        ax.set_ylabel('Mounting Offset Resolution Error [mrad]')

    pe_mo = np.array(pe_mo)
    pe_mo = pe_mo.reshape((pe_mo.shape[0], 1))
    err_att = np.array(err_att)
    err_att = err_att.reshape((err_att.shape[0], 1))
    err_df = pd.DataFrame(data = np.hstack((err_att, pe_mo)), columns = ['err_att_mrad', 'pe_mrad'])
    if 0:
        err_df.to_csv(fr'outputs\tables\moon_mo_res_errors\Att_{att_prof}.csv', index = 0)
        f, ax = plt.subplots()
        ax.plot(err_att, pe_mo, '-o',label = att_prof)
        ax.set_xlabel('Attitude Error Per Axis [mrad]')
        ax.set_ylabel('Mounting Offset Resolution Error [mrad]')