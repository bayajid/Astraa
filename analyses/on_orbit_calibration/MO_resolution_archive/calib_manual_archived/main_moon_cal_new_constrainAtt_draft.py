## Script to plot the attitude profiles in 3D
# August 24, 2023
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
chosen_index = 0
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
constrain_azel = 0
constrain_attitude = 1

nrows = r_host.shape[0]

A_limits = [-195, 195]
E_limits = [15, 85]

mounting_offset_rpy = [5, 4, 2.5] # MOUNTING OFFSET random 3-axis rotation
rot_bf2lct = conv.convert_ea2dcm(mounting_offset_rpy)
quat_mounting_offset = conv.convert_ea2quat(mounting_offset_rpy)

# Placeholders for commanded attitude
ea_eci2bf_command_all = np.zeros((nrows, 6)) # EA; EA_rate [deg, deg/s]
quat_eci2bf_command_all = np.zeros((nrows, 4)) # scalar-first
make_3d_plot = 1 
scan_gap = 1
ii_ae = 0 
interp_dat = 0
ii_scans = [1]
ii_scans.append(ii_scans[0] + scan_gap)

# TODO select attitude profile
importlib.reload(att_sim)
importlib.reload(conv)
## COMPUTE ATTITUDE

# attitude_profiles = ['earth_point', 'earth_roll', 'sun_roll', 'sun_point']
# attitude_profiles = ['sun_roll']
# attitude_profiles = ['sun_point']
# attitude_profiles = ['earth_point']
attitude_profiles = ['earth_roll']
print(f'Using defined attitude profiles : {attitude_profiles}')
attitude_eci2bf = []


dt_required = 5
if interp_dat:
    print(f'Interpolating data to {dt_required} s resolution')
    t_gps_fine = np.round(np.arange(t_gps[0], t_gps[-1]+dt_required, dt_required),1)
    inerpolant = sp.interpolate.CubicSpline(t_gps.flatten(), r_host)
    r_host_fine = inerpolant(t_gps_fine)
    
    inerpolant = sp.interpolate.CubicSpline(t_gps.flatten(), v_host)
    v_host_fine = inerpolant(t_gps_fine)

    inerpolant = sp.interpolate.CubicSpline(t_gps.flatten(), r_moon)
    r_moon_fine = inerpolant(t_gps_fine)

    inerpolant = sp.interpolate.CubicSpline(t_gps.flatten(), illumination)
    illumination_fine = inerpolant(t_gps_fine)
else:
    t_gps_fine = t_gps
    r_host_fine = r_host
    v_host_fine = v_host
    t_gps_fine = t_gps
    r_moon_fine = r_moon
    illumination_fine = illumination

for ii, att_prof in enumerate(attitude_profiles):
    quat_eci2bf, rot_eci2bf = att_sim.calc_quat_eci2bf(r_host_fine, v_host_fine, att_prof, t_gps = t_gps_fine.flatten())
    
    attitude_eci2bf.append(quat_eci2bf)
for ii, dcm_ii in enumerate(rot_eci2bf):
    ea_eci2bf_command_all[ii,:3] = conv.convert_dcm2ea(dcm_ii)


importlib.reload(moon_scan)
if 1:
    ii_scans = [0, 20]
    # Use fully interface code
    ae_moon_commanded_all, ae_moon_true_all, quat_resolved, pe_mo = moon_scan.simulate_moon_calib(
        ii_scans = ii_scans,
        r_host = r_host, 
        r_moon = r_moon, 
        ea_eci2bf_command_all = ea_eci2bf_command_all,
        quat_mounting_offset = quat_mounting_offset,
        add_noise = 1,
        add_centroid_error = 1,
        add_r_host_error = 1,
        add_r_moon_error = 1,
        add_att_host_error = 1,
        att_noise_factor= 0.1,
        print_cond = 1,
        print_full = 1
    )
else:
    ## ADD NOISE
    use_errors = 1
    if use_errors:
        add_centroid_error = 1
        add_r_host_error = 1
        add_r_moon_error = 1
        add_att_host_error = 1
    else:
        add_centroid_error = 0
        add_r_host_error = 0
        add_r_moon_error = 0
        add_att_host_error = 0

    np.random.seed(1)
    err_r_host = np.zeros((nrows, 3))
    err_r_moon = np.zeros((nrows, 3))
    err_att_host = np.zeros((nrows, 3))
    err_eigenaxis_centroid = np.zeros((nrows, 4)) # Axis-angle representation for centroid detection error
    if add_noise:
        for ii in range(nrows):
            err_r_host[ii,:] = np.array([6, 7, 8]) * np.random.randn()
            err_r_moon[ii,:] = np.array([-6, 7, 2*8]) * np.random.randn()
            err_att_host[ii,:] = np.rad2deg([0.08e-3, 0.07e-3, 0.1e-3]) * np.random.randn()
        # Add Moon centroid error, maybe assume 2% non-illumination 
        moon_illum_frac = 0.98
        # 0.09 mrad
        err_centroid_mag = np.deg2rad((1-moon_illum_frac) * 0.5 * 1/2)
        err_eigenaxis_centroid[:,-1] = err_centroid_mag # [rad]
    ae_moon_commanded_all = np.zeros((nrows, 2))
    ae_moon_true_all = np.zeros((nrows, 2))

    for ii, ii_scan in enumerate(ii_scans):
        
        r_host_ii = r_host[ii_scan,:]
        v_host_ii = [0,0,0] # placeholder, not needed here

        r_moon_ii = r_moon[ii_scan,:]
        # _c - COMMANDED values, contains knowledge errors, mounting offset assumed 0
        # _t - TRUE values, does not contain knowledge errors, true mounting ofset used
        r_host_ii_c = r_host_ii    

        if not add_r_host_error:
            err_r_host[ii,:] = [0,0,0]
        if not add_r_moon_error:
            err_r_moon[ii,:] = [0,0,0]
        if not add_att_host_error:
            err_att_host[ii,:] = [0,0,0]
        if not add_centroid_error:
            centroid_err_used = None
        else:
            centroid_err_used = err_centroid_mag

        r_host_ii_c = r_host_ii + err_r_host[ii,:]
        r_host_ii_t = r_host_ii
        
        r_moon_ii_c = r_moon_ii + err_r_moon[ii,:]
        r_moon_ii_t = r_moon_ii
        
        ea_eci2bf_c = ea_eci2bf_command_all[ii,:3] + err_att_host[ii,:]
        ea_eci2bf_t = ea_eci2bf_command_all[ii,:3]
        q_eci2bf_c = conv.convert_ea2quat(ea_eci2bf_c)
        q_eci2bf_t = conv.convert_ea2quat(ea_eci2bf_command_all[ii,:3])

        q_mounting_offset_c = np.array([1,0,0,0]) # Unity quaternion initially used
        q_mounting_offset_t = quat_mounting_offset

        ae_moon_expected = ae_calc.calc_ae_full(r_host_ii_c, r_moon_ii_c, q_eci2bf_c, q_mounting_offset_c)[0]
        ae_moon_observed = ae_calc.calc_ae_full(r_host_ii_t, r_moon_ii_t, q_eci2bf_t, q_mounting_offset_t, centroid_error=centroid_err_used)[0]
        
        ae_moon_commanded_all[ii,:] = ae_moon_expected[:2]
        ae_moon_true_all[ii,:] = ae_moon_observed[:2]
    importlib.reload(att_res)
    importlib.reload(vec_calc)
    quat_resolved = att_res.get_mo_quat_fromscan(ae_moon_commanded_all, ae_moon_true_all)
    dcm_resolved = conv.convert_quat2dcm(quat_resolved)
    ea_resolved = conv.convert_dcm2ea(dcm_resolved)
    pe_mo = vec_calc.get_pe_for_rot(quat_resolved, quat_mounting_offset)
    print(f'''-------------------------\nErrors used
        Centroid detection : {bool(add_centroid_error)}
        host pos error : {bool(add_r_host_error)}
        moon pos error : {bool(add_r_moon_error)}
        host attitude error: {bool(add_att_host_error)}''')
    print(f'PE MO : \n{pe_mo[0][0]:.1f}, {pe_mo[1][0]:.1f}, {pe_mo[2][0]:.1f} urad')
    print(f'''
    Mounting offset TRUE : {mounting_offset_rpy[0]:.3f}, {mounting_offset_rpy[1]:.3f}, {mounting_offset_rpy[2]:.3f} deg
    Mounting offset RESOLVED : {ea_resolved[0]:.3f}, {ea_resolved[1]:.3f}, {ea_resolved[2]:.3f} deg
        ''')
        
        