#%% April 30, simulating to check how ground tests are impacted by large attitude
# knowledge errors
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
fname_moon = 'analyses/on_orbit_calibration/MOQ_resolution_sims/gs2moon_data.csv'

## MVP imports
import basic_tools.vector_operations as vec_calc
import basic_tools.time_conversion as t_conv
import basic_tools.in_out as io
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
importlib.reload(bplt)
importlib.reload(moon_scan)
importlib.reload(att_res)
debug_mode = 1
trolley_angles_used= [0, 0.1, 0.5, 1, 1.5, 2, 2.5, 3]
randomize_magnitude = 0

data_df = pd.read_csv(fname_moon)  
data_sliced = data_df.values[106000:141000,:]
if debug_mode:
    # data_sliced = data_sliced[:10,:]
    trolley_angles_used = [0.5]
q_bf_full = data_sliced[:,16:24]
q_bf_full[:,:4] = q_bf_full[:,:4]
q_bf_full[:,4:] = q_bf_full[:,4:] 

t_gps = data_sliced[:,0]
s_host = data_sliced[:,4:10]
s_target = data_sliced[:,10:16]
r_host_full = s_host[:,:3]
v_host_full = s_host[:,3:]
r_target_full = s_target[:,:3]

# DEBUG/VERIFICATION PURPOSES
# title_append = 'no_mean_errors'
# title_append = 'double_mean_errors'
# title_append = 'only_cmd_errors'
# title_append = 'only_att_errors'
# title_append = 'only_att_errorsx2'
# title_append = 'full_magnitude'
# title_append = 'no_errors'
title_append = 'phase_A'
# title_append = 'ground_trolley_sim_no_rand_mag'
case = '2000km'
# case = '200km'

spacing = 'uniform'
n_samples = 30000
# n_samples = 1000

# use_quest = 0
use_triad = 1
function_option = 0
if not use_triad:
    nr_inputs = 6
    # nr_inputs = 2
else:
    nr_inputs = 2
    function_option = 0
# TODO loop over more host positions, intead of attitude angles

# select update rates
att_update_rate = 2 # [Hz]
pos_update_rate = 0.1 # [Hz]

if case == '2000km':
    ii_gap = 267
elif case == '200km':
    ii_gap = 27

nrows = r_host_full.shape[0]
mounting_offset_rpy = [5, 4, 2.5] # MOUNTING OFFSET random 3-axis wrotation
error_mounting_offset = 1.5e-3 # mrad, Mounting offset error

rot_bf2lct = conv.convert_ea2dcm(mounting_offset_rpy)
quat_mounting_offset = conv.convert_ea2quat(mounting_offset_rpy)
quat_mounting_offset_known = conv.convert_dcm2quat(conv.convert_eigenaxis2dcm([-1,2,3],error_mounting_offset) @ conv.convert_quat2dcm(quat_mounting_offset))
quat_mounting_offset_known = quat_mounting_offset_known / np.linalg.norm(quat_mounting_offset_known)

# TODO select attitude profile
importlib.reload(att_sim)
importlib.reload(conv)
importlib.reload(att_res)
storage_folder = 'ooc_ground_test'
## COMPUTE ATTITUDE

# 
err_r_host_default = np.array([6, 7, 8]) # RANDOMIZE DIRECTION? ACTUALLY TODO what magnitude??
# err_r_target_default = np.array([-60, -70, 80]) # RANDOMIZE DIRECTION? 100 m target position error expected

err_r_target_default = np.array([-60, -70, 80]) # RANDOMIZE DIRECTION? 10 m Ground station target position error expected

# https://space.leonardo.com/documents/16277711/19573187/Copia_di_A_STR_Autonomous_Star_Trackers_LQ_mm07786_.pdf?t=1538987562062 - 20 arcsec/axis
# Attitude error ~ 0.5 mrad
err_att_host_default = np.array([3.2e-4, 3.2e-4, 3.2e-4]) # rad # RANDOMIZE DIRECTION?

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
# moon_ilum_fraction = 0.9

moon_ilum_fraction = 0.9
centroid_err = 0.25*17*(1-moon_ilum_fraction)
for ii_loop, trolley_angle in enumerate(trolley_angles_used):
    seed_used = ii_loop

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
    elif title_append == 'no_errors':
        sum_mean_errors = sum_mean_errors * 0
        RSS_random_errors = RSS_random_errors * 0
        err_r_host_default = err_r_host_default * 0
        err_r_target_default = err_r_target_default * 0
        err_att_host_default = err_att_host_default * 0
    elif title_append == 'phase_A':
        err_r_host_default = err_r_host_default * 0
        err_r_target_default = err_r_target_default * 0
    elif title_append == 'ground_trolley_sim':
        err_r_host_default = err_r_host_default * 0
        err_r_target_default = err_r_target_default * 0
        err_att_host_default = err_att_host_default * 0

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

    ii_scans = list(np.arange(0,nr_inputs,1))

    error_text_box = f'''PEB Components Used:
    r_h : {np.linalg.norm(errors_chosen['err_r_host']):.1f} m
    r_t : {np.linalg.norm(errors_chosen['err_r_target']):.1f} m
    att_h : {np.linalg.norm(errors_chosen['err_att_host'])*1e3:.1f} mrad
    rss : {np.linalg.norm(errors_chosen['rss_random_errors'])*1e3:.1f} mrad
    mean : {np.linalg.norm(errors_chosen['sum_mean_errors'])*1e3:.1f} mrad'''

    pe_mo = []
    err_att = []
    non_colin_angle = []
    t_gap = []
    # loop over angles:
    for ii in range(nrows-1):
        t_gap_ii = t_gps[ii+1] - t_gps[0]
        r_host = r_host_full[[0,ii+1],:]
        r_target = r_target_full[[0,ii+1],:]
        quat_eci2bf = q_bf_full[[0, ii+1], :4]
        ea_eci2bf = np.array([conv.convert_quat2ea(q_ii) for q_ii in quat_eci2bf])
        # title_append = f'{title_append}_moon{moon_ilum_fraction}'
        mean_components = {
            'pointing_model_residuals' : 0.5,
            'centroid_detection' : centroid_err
        }
        RSS_random_errors = np.linalg.norm(list(RSS_components.values()))*1e-3 # rad
        sum_mean_errors = np.sum(list(mean_components.values())) * 1e-3
        total_random_error_input = RSS_random_errors + sum_mean_errors
        total_random_error_input = total_random_error_input # [rad]
        # run-scan # TODO integrate QUEST into simulating moon scan code
        ae_moon_commanded_all, ae_moon_true_all, quat_resolved, po_ii, obs_angle, fct_used = moon_scan.simulate_ooc(
            ii_scans = ii_scans,
            function_option = function_option,
            r_host = r_host, 
            r_target = r_target, 
            ea_eci2bf_command_all = ea_eci2bf,
            quat_mounting_offset_t = quat_mounting_offset,
            quat_mounting_offset_c = quat_mounting_offset_known,
            manual_error_dict=errors_chosen,
            check_non_colin=1,
            randomize_error_direction = 1,
            centroid_dirction_randomizer = 1,
            randomize_magnitude = randomize_magnitude,
            seed_used = ii*(ii_loop+1),
            ii_loop = ii,
            add_trolley_tilt= True,
            trolley_angle = trolley_angle
        )
            # store
        pe_mo.append(np.max(po_ii)/1e3) # mrad
        non_colin_angle.append(np.rad2deg(obs_angle)) # deg
        t_gap.append(t_gap_ii)
    data_df = pd.DataFrame.from_dict({
        'pe_mo' : pe_mo,
        'ang_sep' : non_colin_angle,
        't_gap' : t_gap
    })

    angle_bins = []
    pe_3sig = []
    d_angle = 1
    pe_2sig = []
    t_bins = []
    for angle_0 in range(0, 180, d_angle): # get 3-sigma PE
        angle_range = [angle_0, angle_0 + d_angle]
        df_sliced_low = data_df[data_df['ang_sep'] > angle_range[0]]
        df_sliced_high = df_sliced_low[df_sliced_low['ang_sep'] < angle_range[1]]
        angle_bins.append(np.average(angle_range))
        t_bins.append(df_sliced_high['t_gap'].quantile(1))
        pe_3sig.append(df_sliced_high['pe_mo'].quantile(0.997))
        pe_2sig.append(df_sliced_high['pe_mo'].quantile(0.95))

    importlib.reload(io)
    df_dict = {
        'ang_sep' : angle_bins,
        'pe_3sig': pe_3sig,
        'pe_2sig': pe_2sig,
        't_gap_s' : t_bins
    }
    # output_df = pd.DataFrame.from_dict(df_dict)
    df_title = f'ooc_{fct_used}_{title_append}_sample{nr_inputs}_datapt{n_samples}_trolley{trolley_angle}'
    output_df = io.save_dict_2_csv(df_dict, df_title, subfolder = storage_folder)
    # n_samples, datapt_used, datapt_spacing, weights?
    dr_h = np.linalg.norm(err_r_host_default) # m
    dr_t = np.linalg.norm(err_r_target_default) # m
    d_theta = np.linalg.norm(np.deg2rad(err_att_host_default)*1e3) # mrad
    d_rand = RSS_random_errors*1e3 # mrad

    if 1:

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
        ax.plot(ax.get_xlim(), [3,3], label = '3 mrad', c = 'y')
        ax.legend(bbox_to_anchor=(1, 0.901))
        ax.set_xlabel('Angle between scanned LOS [deg]', fontweight = 'bold')
        ax.set_ylabel('Mounting Offset Resolution Error [mrad]', fontweight = 'bold')
        ax.grid('on')
        ax.set_xlim([5, 170])
            # ax.set_xscale('log')
            # ax.set_yscale('log')
        ax.set_ylim([-0.1,20])
        title = f'''Phase A GROUND. Initial MOQ Error : {error_mounting_offset*1e3:.1f} mrad. Tilt : {trolley_angle} deg
        Centroid Det : {centroid_err:.1f} mrad; Moon Illum: {moon_ilum_fraction*100:.0f}%
        Algorithm used: '''
        f.suptitle(f'{title}{fct_used.upper()}')
        # Reformat to get angle from 0:90 for non-colinearity
        ax.text(140.85, 0.2, 
                error_text_box,
                fontsize = 12,            
                bbox = {'boxstyle': 'square',
                        'facecolor':'peachpuff',                    
                        }
                )
        figname = f'{case}_ooc_ground_mor_triad_{moon_ilum_fraction}_tilt{trolley_angle}'
        bplt.savefig(f, figname, subfolder =storage_folder)
        


plt.show()