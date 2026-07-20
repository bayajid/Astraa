# this script is used to test the minimum required
# code to compute the mounting offset quaternion from
# 2 sets of az/el angles and 
#%%
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
csv_output_path = r'orbital_simulations/tudat_raw_states'
fname_simparam = 'simulation_parameters.json'
fname_states = 'state_history.dat'

## MVP imports
import basic_tools.vector_operations as vec_calc
import basic_tools.time_conversion as t_conv
import plotting_tools.basic_plotting as bplt
import plotting_tools.plotting_utilities as plt_util

import astronomy_tools.constants as const
import astronomy_tools.astro_targets as where_sun

import attitude_tools.attitude_resolution as att_res
import attitude_tools.conversions as conv
import attitude_tools.rotations as rot
import attitude_tools.attitude_simulation as att_sim

import basic_tools.in_out as savedat

import pointing_calculations.ae_calculation as ae_calc
import pointing_calculations.simulate_moon_scan as moon_scan
import tudat_tools.data_processing.data_processing_utilities as dputil

data_raw, simulation_parameters = dputil.load_constellation_data(full_path = csv_output_path)
host_chosen = 'leo_host_polar'
t_j2000 = data_raw[:,0]
r_host_full = data_raw[:,simulation_parameters['r_index'][host_chosen]]
v_host_full = data_raw[:,[ii+3 for ii in simulation_parameters['r_index'][host_chosen]]]

case = '2000km'
# case = '200km'

ii_host = [0, 10]

# debug settings - switch off errors TODO implement
set_dr_to_zero = 1
set_datt_to_zero = 0
set_randerr_to_zero = 1 


r_host = r_host_full[ii_host,:]
v_host = v_host_full[ii_host,:]
t_gps = t_j2000[ii_host] + t_conv.dt_j2000tt2gps()
dt_data = t_gps[1] - t_gps[0]

if case == '2000km':
    ii_gap = 267
elif case == '200km':
    ii_gap = 27
r_target = r_host_full[[ii_host[0] + ii_gap, ii_host[1] + ii_gap],:]

nrows = r_host.shape[0]
mounting_offset_rpy = [90, 4, 2.5] # MOUNTING OFFSET random 3-axis wrotation
error_mounting_offset = 3e-3 # mrad, Mounting offset error after Phase A

rot_bf2lct = conv.convert_ea2dcm(mounting_offset_rpy)
quat_mounting_offset = conv.convert_ea2quat(mounting_offset_rpy)
quat_mounting_offset_known = conv.convert_dcm2quat(conv.convert_eigenaxis2dcm([-1,2,3],error_mounting_offset) @ conv.convert_quat2dcm(quat_mounting_offset))
            
# TODO select attitude profile
importlib.reload(att_sim)
importlib.reload(conv)
## COMPUTE ATTITUDE
attitude_profile = 'earth_roll'

err_r_host_default = np.array([6, 7, 8]) # RANDOMIZE DIRECTION? ACTUALLY TODO what magnitude??
err_r_moon_default = np.array([80, 60, 10]) # RANDOMIZE DIRECTION? 100 m target position error expected
# https://space.leonardo.com/documents/16277711/19573187/Copia_di_A_STR_Autonomous_Star_Trackers_LQ_mm07786_.pdf?t=1538987562062 - 20 arcsec/axis
err_att_host_default = np.rad2deg([3.2e-4, 3.2e-4, 3.2e-4]) # deg # RANDOMIZE DIRECTION?

# mrad, RSS components
RSS_components = {
    'gimbal_control' : 0.4,
    'thermal' : 0.8,
    'pos_pred' : 0.2, 
    'att_pred' : 0.2, 
}
RSS_random_errors = np.linalg.norm(list(RSS_components.values()))*1e-3 # rad

importlib.reload(moon_scan)
importlib.reload(ae_calc)

angles_desired = [70]
# angles_desired = [90]
# angles_desired = [40, 75, 90, 160, 0, 500, 370]
# 
errors_chosen = {
        'err_r_host': err_r_host_default,
        'err_r_moon': err_r_moon_default,
        'err_att_host': err_att_host_default,
        'RSS_random_errors': RSS_random_errors,
    }

err_scale_t2 = 1
errors_chosen['err_r_host'] = np.vstack((errors_chosen['err_r_host'],errors_chosen['err_r_host']*err_scale_t2))
errors_chosen['err_r_moon'] = np.vstack((errors_chosen['err_r_moon'],errors_chosen['err_r_moon']*err_scale_t2))
errors_chosen['err_att_host'] = np.vstack((errors_chosen['err_att_host'],errors_chosen['err_att_host']*err_scale_t2))


pe_mo = []
err_att = []
non_colin_angle = []
# SET SEED
np.random.seed(1)
for att_angle in angles_desired:
    roll_required = att_angle/dt_data
    # generate 2-point attitude
    ea_eci2bf = np.zeros((nrows, 6))
    try:
        quat_eci2bf, quatdot, rot_eci2bf = att_sim.calc_quat_eci2bf(r_host, v_host, attitude_profile,
                                                            t_gps = t_gps.flatten(), 
                                                            roll_velocity=roll_required,
                                                            rotation_axis=1)
    except:
        # quat conversion singularity. Offset a lil bit
        quat_eci2bf, quatdot, rot_eci2bf = att_sim.calc_quat_eci2bf(r_host, v_host, attitude_profile,
                                                            t_gps = t_gps.flatten(), 
                                                            roll_velocity=roll_required*1.001,
                                                            rotation_axis=1)
    for mm, dcm_ii in enumerate(rot_eci2bf):
        ea_eci2bf[mm,:3] = conv.convert_dcm2ea(dcm_ii)
        if 0:
            np.rad2deg(vec_calc.get_pe_for_rot(rot_eci2bf[0], rot_eci2bf[1], rot = 'dcm')/1e6) # deg
    # run-scan
    ae_moon_commanded_all, ae_moon_true_all, quat_resolved, po_ii, obs_angle = moon_scan.simulate_moon_calib(
        ii_scans = [0,1],
        r_host = r_host, 
        r_moon = r_target, 
        ea_eci2bf_command_all = ea_eci2bf,
        quat_mounting_offset_t = quat_mounting_offset,
        quat_mounting_offset_c = quat_mounting_offset_known,
        manual_error_dict=errors_chosen,
        add_noise = 1,
        add_centroid_error = 1,
        add_r_host_error = 1,
        add_r_moon_error = 1,
        add_att_host_error = 1,
        att_noise_factor= 0,
        centroid_err_factor=0,
        centroid_dirction_randomizer= 1,
        check_non_colin=1,
        randomize_dr_dtheta = 1,
        randomize_magnitude = 1,
        seed_used = 1,
        print_cond = 1,
        print_full = 1,
        use_clean_path = 1,
)
    # store
    pe_mo.append(np.max(po_ii)/1e3) # mrad
    non_colin_angle.append(np.rad2deg(obs_angle)) # deg
