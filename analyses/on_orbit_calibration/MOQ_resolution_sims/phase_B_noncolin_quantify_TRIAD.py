#%% September 29- quantify impact of error sources on Phase B of OCC
# 2 situations considered - 200km and 2000 km links
# similar architecture as moon scan -> 2 pairs of expected and found Az El pairs
# main differences - 
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

importlib.reload(moon_scan)

data_raw, simulation_parameters = dputil.load_constellation_data(full_path = csv_output_path)
host_chosen = 'leo_host_polar'
t_j2000 = data_raw[:,0]
r_host_full = data_raw[:,simulation_parameters['r_index'][host_chosen]]
v_host_full = data_raw[:,[ii+3 for ii in simulation_parameters['r_index'][host_chosen]]]

# DEBUG/VERIFICATION PURPOSES
# title_append = 'no_mean_errors'
# title_append = 'double_mean_errors'
# title_append = 'only_cmd_errors'
# title_append = 'only_att_errors'
title_append = 'att_errors20mrad'
# title_append = 'only_att_errorsx2'
# title_append = 'full_magnitude'
# title_append = ''

case = '2000km'
# case = '200km'

ii_host = [0, 10]

# select update rates
att_update_rate = 2 # [Hz]
pos_update_rate = 0.1 # [Hz]

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
mounting_offset_rpy = [5, 4, 2.5] # MOUNTING OFFSET random 3-axis wrotation
error_mounting_offset = 3e-3 # mrad, Mounting offset error after Phase A

rot_bf2lct = conv.convert_ea2dcm(mounting_offset_rpy)
quat_mounting_offset = conv.convert_ea2quat(mounting_offset_rpy)
quat_mounting_offset_known = conv.convert_dcm2quat(conv.convert_eigenaxis2dcm([-1,2,3],error_mounting_offset) @ conv.convert_quat2dcm(quat_mounting_offset))
            
# TODO select attitude profile
importlib.reload(att_sim)
importlib.reload(conv)
## COMPUTE ATTITUDE
attitude_profile = 'earth_roll'

# 
err_r_host_default = np.array([6, 7, 8]) # RANDOMIZE DIRECTION? ACTUALLY TODO what magnitude??
err_r_target_default = np.array([-6, -7, 8]) # RANDOMIZE DIRECTION? 100 m target position error expected
# https://space.leonardo.com/documents/16277711/19573187/Copia_di_A_STR_Autonomous_Star_Trackers_LQ_mm07786_.pdf?t=1538987562062 - 20 arcsec/axis
# Attitude error ~ 0.5 mrad
err_att_host_default = np.array([3.2e-4, 3.2e-4, 3.2e-4]) # rad # RANDOMIZE DIRECTION? (0.5 mrad)

# mrad, RSS components
RSS_components = {
    'gimbal_control' : 0.4,
    'thermal' : 0.8,
    'other' : 0.1,
}
mean_components = {
    'pointing_model_residuals' : 0.5
}
RSS_random_errors = np.linalg.norm(list(RSS_components.values()))*1e-3 # rad
sum_mean_errors = np.sum(list(mean_components.values())) * 1e-3
total_random_error_input = RSS_random_errors + sum_mean_errors
total_random_error_input = total_random_error_input # [rad]
importlib.reload(moon_scan)
importlib.reload(ae_calc)
n_samples = 60000 
n_samples = 1000 
angles_desired = np.linspace(0, 180, n_samples)
# angles_desired = [90]
# angles_desired = [40, 75, 90, 160, 0, 500, 370]
# 

if title_append == 'no_mean_errors':
    sum_mean_errors = sum_mean_errors * 0
elif title_append == 'double_mean_errors':
    sum_mean_errors = sum_mean_errors * 2
elif title_append == 'only_cmd_errors':
    sum_mean_errors = sum_mean_errors*0
    RSS_random_errors = RSS_random_errors*0
elif title_append == 'only_att_errors':
    sum_mean_errors = sum_mean_errors * 0
    RSS_random_errors = RSS_random_errors * 0
    err_r_host_default = err_r_host_default * 0
    err_r_target_default = err_r_target_default * 0
elif title_append == 'only_att_errorsx2':
    sum_mean_errors = sum_mean_errors * 0
    RSS_random_errors = RSS_random_errors * 0
    err_r_host_default = err_r_host_default * 0
    err_r_target_default = err_r_target_default * 0
    err_att_host_default = err_att_host_default * 2
elif title_append == 'att_errors20mrad':
    err_att_host_default = err_att_host_default * 10
    
errors_chosen = {
        'err_r_host': err_r_host_default,
        'err_r_target': err_r_target_default,
        'err_att_host': err_att_host_default,
        'sum_mean_errors': sum_mean_errors,
        'rss_random_errors': RSS_random_errors,
    }

if att_update_rate == 2:
    err_scale_att = 2
elif att_update_rate == 4:
    err_scale_att = 1.2
    
if pos_update_rate == 0.1:
    err_scale_pos = 2
elif pos_update_rate == 1/30:
    err_scale_pos = 5
    
# errors_chosen['err_r_host'] = np.vstack((errors_chosen['err_r_host'],errors_chosen['err_r_host']*err_scale_pos))
# errors_chosen['err_r_target'] = np.vstack((errors_chosen['err_r_target'],errors_chosen['err_r_target']*err_scale_pos))
# errors_chosen['err_att_host'] = np.vstack((errors_chosen['err_att_host'],errors_chosen['err_att_host']*err_scale_att))

if title_append == 'full_magnitude':
    randomize_magnitude = 0
else:
    randomize_magnitude = 1

pe_mo = []
err_att = []
non_colin_angle = []
# SET SEED

for ii_loop, att_angle in enumerate(angles_desired):
    seed_used = ii_loop
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
    # run-scan
    ae_moon_commanded_all, ae_moon_true_all, quat_resolved, po_ii, obs_angle, fct_used = moon_scan.simulate_ooc(
        ii_scans = [0,1],
        r_host = r_host, 
        r_target = r_target, 
        ea_eci2bf_command_all = ea_eci2bf,
        quat_mounting_offset_t = quat_mounting_offset,
        quat_mounting_offset_c = quat_mounting_offset_known,
        manual_error_dict=errors_chosen,
        check_non_colin=1,
        randomize_error_direction = 1,
        randomize_magnitude = randomize_magnitude,
        seed_used = seed_used,
)
    # store
    pe_mo.append(np.max(po_ii)/1e3) # mrad
    non_colin_angle.append(np.rad2deg(obs_angle)) # deg
data_df = pd.DataFrame.from_dict({
    'pe_mo' : pe_mo,
    'ang_sep' : non_colin_angle
})

#%% PROCESS df
angle_bins = []
pe_3sig = []
d_angle = 1
pe_2sig = []
for angle_0 in range(0, 180, d_angle): # get 3-sigma PE
    angle_range = [angle_0, angle_0 + d_angle]
    df_sliced_low = data_df[data_df['ang_sep'] > angle_range[0]]
    df_sliced_high = df_sliced_low[df_sliced_low['ang_sep'] < angle_range[1]]
    angle_bins.append(np.average(angle_range))
    pe_3sig.append(df_sliced_high['pe_mo'].quantile(0.997))
    pe_2sig.append(df_sliced_high['pe_mo'].quantile(0.95))
#%% PLOT
dr_h = np.linalg.norm(err_r_host_default) # m
dr_t = np.linalg.norm(err_r_target_default) # m
d_theta = np.linalg.norm(np.deg2rad(err_att_host_default)*1e3) # mrad
d_rand = RSS_random_errors*1e3 # mrad

if 1:
    if len(angles_desired) == 1:
        print(pe_mo)
    importlib.reload(plt_util)
    plot_all = 1
    plot_sep = 0
    markers = ['.', 'o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X']
    if plot_all:
        append_title = f'\n 3sig PEB used: Pos H = {dr_h:.1f} m; T = {dr_t:.1f} m; Att H : {d_theta:.2f} mrad; Other: {d_rand:.2f} mrad; \nLink distance : {case}'
    elif plot_sep:
        # plot and save 1 by 1 
        append_title = ''
    f, ax = plt.subplots()
    ax.scatter(non_colin_angle, pe_mo, color = 'g', alpha = 0.1)
    ax.plot(angle_bins, pe_3sig, label = '3-sigma', c = 'r')
    ax.plot(angle_bins, pe_2sig, label = '2-sigma', c = 'm')
    ax.legend(bbox_to_anchor=(1, 0.901))
    ax.set_xlabel('Angle between scanned LOS [deg]', fontweight = 'bold')
    ax.set_ylabel('Mounting Offset Resolution Error [mrad]', fontweight = 'bold')
    ax.grid('on')
    ax.set_xlim([10, 170])
        # ax.set_xscale('log')
        # ax.set_yscale('log')
    ax.set_ylim([-0.1,5])
    title = f'Phase B MOR Error. Phase A error : {error_mounting_offset*1e3:.1f} mrad. {append_title}'
    f.suptitle(f'{title}{title_append}')
    bplt.savefig(f, f'{case}_ooc_phaseb_mor_triad_{title_append}', subfolder ='phaseb_ooc')
    # Reformat to get angle from 0:90 for non-colinearity
    if 0:
        angle_recalc = [np.deg2rad(180*1e3)-ii if ii > np.deg2rad(90*1e3) else ii for ii in non_colin_angle]



        f, ax = plt.subplots()
        ax.plot(angle_recalc, pe_mo)
        # ax.plot([180/57.3*1e3, 180/57.3*1e3], [ax.get_ylim()[0], ax.get_ylim()[1]], 'r--')
        ax.set_xlabel('Non-colinearity angle [mrad]', fontweight = 'bold')
        ax.set_ylabel('Mounting Offset Resolution Error [mrad]', fontweight = 'bold')
        ax.grid('on')

    plt.show()