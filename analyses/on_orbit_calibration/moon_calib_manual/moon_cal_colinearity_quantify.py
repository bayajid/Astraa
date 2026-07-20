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

r_host = np.array([[-2465085.48694964,   276433.98928685,  6918785.1310705 ], 
                   [-2875855.21012004,   299325.08750779,  6757340.5391917 ]])

v_host = np.array([[98.02791577,  1.        ,  1.        ],
       [98.02975018,  1.        ,  1.        ]])
t_gps = np.array([[1.35952921e+09],
       [1.35952927e+09]])
r_target = np.array([[-1.94362186e+08,  3.11696478e+08,  1.72957337e+08], 
                   [-1.94413308e+08,  3.11670348e+08,  1.72946859e+08]])

nr_inputs = 6
r_host = np.linspace(r_host[0], r_host[1], nr_inputs)
v_host = np.linspace(v_host[0], v_host[1], nr_inputs)
t_gps = np.linspace(t_gps[0], t_gps[1], nr_inputs)
r_target = np.linspace(r_target[0], r_target[1], nr_inputs)


# r_host[1,:] = r_host[0,:]                   
# r_moon[1,:] = r_moon[0,:]
dt_data = float(t_gps[1] - t_gps[0])
# Get attitude

nrows = r_host.shape[0]
mounting_offset_rpy = [5, 4, 2.5] # MOUNTING OFFSET random 3-axis rotation
rot_bf2lct = conv.convert_ea2dcm(mounting_offset_rpy)
quat_mounting_offset = conv.convert_ea2quat(mounting_offset_rpy)

# Placeholders for commanded attitude
ii_scans = np.arange(0, nr_inputs, 1)
print_azel = 1
            
# TODO select attitude profile
importlib.reload(att_sim)
importlib.reload(conv)
## COMPUTE ATTITUDE
# attitude_profile = 'moon_roll_perp'
attitude_profile = 'moon_roll_perp_tilt'


err_r_host_default = np.array([6, 7, 8])
err_r_moon_default = -np.array([6, 7, 8])
# https://space.leonardo.com/documents/16277711/19573187/Copia_di_A_STR_Autonomous_Star_Trackers_LQ_mm07786_.pdf?t=1538987562062 - 20 arcsec/axis
err_att_host_default = np.rad2deg([3.2e-4, 3.2e-4, 3.2e-4]) # deg
rss_random_errors = 2.3e-3 # rad
error_dict = {}

ind_pos = [1,2,3]
ind_att = [4,5,6,7,8]
ind_both = [9, 10, 11]
ind_center = [12]
ind_all = [13, 14, 15]
ind_all_random = [16,17,18]
for ii in ind_pos:
    error_dict[ii] = {
        'err_r_host': err_r_host_default*10**(ii),
        'err_r_target': err_r_moon_default*10**(ii),
        'err_att_host': 0,
        'rss_random_errors': 0,
    }

for ii in ind_att:
    error_dict[ii] = {
        'err_r_host': 0,
        'err_r_target': 0,
        'err_att_host': err_att_host_default * 10**(ii-4-2),
        'rss_random_errors': 0,
    }
    # selective error set-up
    if ii-4-2 == 1:

        error_dict[ii]['err_att_host'] = err_att_host_default * 10**(ii-4-2)/3


for ii in ind_both:
    error_dict[ii] = {
        'err_r_host': err_r_host_default * 10**(ii-9),
        'err_r_target': err_r_moon_default * 10**(ii-9),
        'err_att_host': err_att_host_default/3 * 10**(ii-9),
        'rss_random_errors': 0,
        'label': f'dr = {np.linalg.norm(err_r_host_default*10**(ii-2))} m; d$$\theta$$ = 0 deg'
    }
for ii in ind_center:
    error_dict[ii] = {
            'err_r_host': err_r_host_default * 0,
            'err_r_target': err_r_moon_default * 0,
            'err_att_host': err_att_host_default * 0,
            'rss_random_errors': rss_random_errors,
        }

for jj, ii in enumerate(ind_all):
    error_dict[ii] = {
            'err_r_host': err_r_host_default,
            'err_r_target': err_r_moon_default,
            'err_att_host': err_att_host_default,
            'rss_random_errors': rss_random_errors*(1/3*(jj+1)),
        }
for jj, ii in enumerate(ind_all_random):
    error_dict[ii] = {
            'err_r_host': err_r_host_default,
            'err_r_target': err_r_moon_default,
            'err_att_host': err_att_host_default,
            'rss_random_errors': rss_random_errors*(1/3*(jj+1)),
        }
# Add labels
for ii, key in enumerate(error_dict.keys()):
    errors_ii = error_dict[key]
    dr = np.linalg.norm(errors_ii['err_r_host'])
    dtheta = np.deg2rad(np.max(errors_ii['err_att_host']))*1e3
    dcentroid = np.max(errors_ii['rss_random_errors'])    
    label_string = fr'$dr = {dr:.0f} m, d\theta = {dtheta:.1f} mrad; dc = {dcentroid:.1f} mrad$'
    error_dict[key]['label'] = label_string

# add 2nd row
for ii, key in enumerate(error_dict.keys()):
    error_dict[key]['err_r_host'] = np.vstack((error_dict[key]['err_r_host'],error_dict[key]['err_r_host']*0))
    error_dict[key]['err_r_target'] = np.vstack((error_dict[key]['err_r_target'],error_dict[key]['err_r_target']*0))
    error_dict[key]['err_att_host'] = np.vstack((error_dict[key]['err_att_host'],error_dict[key]['err_att_host']*0))
    # error_dict[key][]

importlib.reload(moon_scan)
importlib.reload(ae_calc)

# angles_desired = np.linspace(0, 180, 1000)
angles_desired = [40, 50, 60, 75, 90]
# Might need to change attitude mode


for ii, key in enumerate(error_dict.keys()): 
    if key in ind_all_random:
        random_error = 1
    else:
        random_error = False
    errors_chosen = error_dict[key]
    pe_mo = []
    err_att = []
    non_colin_angle = []
    for att_angle in angles_desired:
        roll_required = att_angle/dt_data/nr_inputs
        # generate N-point attitude TODO
        
        ea_eci2bf = np.zeros((nrows, 6))
        quat_eci2bf, quatdot, rot_eci2bf = att_sim.calc_quat_eci2bf(r_host, v_host, attitude_profile,
                                                            t_gps = t_gps.flatten(), 
                                                            roll_velocity=roll_required)
        for mm, dcm_ii in enumerate(rot_eci2bf):
            ea_eci2bf[mm,:3] = conv.convert_dcm2ea(dcm_ii)
        
        
        ae_moon_commanded_all, ae_moon_true_all, quat_resolved, po_ii, obs_angle = moon_scan.simulate_moon_calib(
            ii_scans = ii_scans,
            r_host = r_host, 
            r_moon = r_target, 
            ea_eci2bf_command_all = ea_eci2bf,
            quat_mounting_offset_t = quat_mounting_offset,
            manual_error_dict=errors_chosen,
            add_noise = 1,
            add_centroid_error = 1,
            add_r_host_error = 1,
            add_r_moon_error = 1,
            add_att_host_error = 1,
            att_noise_factor= 0,
            centroid_err_factor=0,
            centroid_dirction_randomizer= random_error,
            check_non_colin=1,
            print_cond = 0,
                print_full = 0
        )
        # store
        pe_mo.append(np.max(po_ii)/1e3) # mrad
        err_att.append(0.1) # mrad
        non_colin_angle.append(obs_angle*1e3) # mrad
        if print_azel:
            print(f'Angle : {np.rad2deg(obs_angle)} AE: {np.rad2deg(ae_moon_commanded_all)}')
    error_dict[key]['pe'] = pe_mo
    error_dict[key]['non_colin_angle'] = non_colin_angle
    # # TODO get rotation
    # pe_mo = np.array(pe_mo)
    # pe_mo = pe_mo.reshape((pe_mo.shape[0], 1))
    # err_att = np.array(err_att)
    # err_att = err_att.reshape((err_att.shape[0], 1))
    # err_df = pd.DataFrame(data = np.hstack((err_att, pe_mo)), columns = ['err_att_mrad', 'pe_mrad'])
    # err_df.to_csv(fr'outputs\tables\moon_mo_res_errors\Att_{att_prof}.csv', index = 0)
#%% PLOTS PLOTS PLOTS <3
# adding labels
for ii, key in enumerate(error_dict.keys()):
    errors_ii = error_dict[key]
    dr = np.linalg.norm(errors_ii['err_r_host'])
    dtheta = np.deg2rad(np.max(errors_ii['err_att_host']))*1e3
    dcentroid = np.max(errors_ii['rss_random_errors'])    
    label_string = fr'$dr = {dr:.1e} m, d\theta = {dtheta:.1f} mrad; dc = {dcentroid*1e3:.1f} mrad$'
    error_dict[key]['label'] = label_string
# ind_pos = [1,2,3]
# ind_att = [4,5,6,7,8]
# ind_both = [9, 10, 11]
# ind_center = [12]
# ind_all = [13]
importlib.reload(plt_util)
plot_all = 1
plot_sep = 0
markers = ['.', 'o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X']
if plot_all:
    # plot included indices
    # ii_plotted = [2,6,7, 10, 13, 14, 15]
    # ii_plotted = [2,6,7, 10, 13, 14, 15, 16, 17, 18] # all
    # random vs non-random
    # ii_plotted = [13,16]
    mm = 2
    ii_plotted = [ind_all[mm], ind_all_random[mm]]

    append_title = '\nall error components considered'
elif plot_sep:
    # plot and save 1 by 1 
    append_title = ''
f, ax = plt.subplots()
colormap = plt_util.give_color_range(len(ii_plotted))
color_list = plt_util.give_color_list(ii_plotted, 4)
for ind, ii in enumerate(ii_plotted):
    # c_ii = colormap[ind]
    err_ii = error_dict[ii]
    c_ii = color_list[ind]
    m_ii = markers[ind]
    label = err_ii['label']

    if ii in ind_all_random:
        alpha = 0.3
        label = f'{label}, rand dc'
    else:
        alpha = 1
    if ii in ind_all:
        c_ii = 'm'
    
    non_colin_angle = err_ii['non_colin_angle']
    pe_mo = err_ii['pe']
    # label = str(ii) + ' ' + err_ii['label']
    ax.plot(non_colin_angle, pe_mo, label = label, color = c_ii, marker = m_ii, markevery = 60, alpha = alpha)
ax.plot([180/57.3*1e3, 180/57.3*1e3], [ax.get_ylim()[0], ax.get_ylim()[1]], 'r', linewidth = 3, label = 'pi')
ax.legend(bbox_to_anchor=(1, 1.01))
ax.set_xlabel('Angle between scanned LOS [mrad]', fontweight = 'bold')
ax.set_ylabel('Mounting Offset Resolution Error [mrad]', fontweight = 'bold')
ax.grid('on')
ax.set_xlim([-10, 3200])
    # ax.set_xscale('log')
    # ax.set_yscale('log')
ax.set_ylim([-0.1,10])
title = f'Impact of Moon Scan LOS colinearity on MOR Error {append_title}'
f.suptitle(title)
# Reformat to get angle from 0:90 for non-colinearity
if 0:
    angle_recalc = [np.deg2rad(180*1e3)-ii if ii > np.deg2rad(90*1e3) else ii for ii in non_colin_angle]



    f, ax = plt.subplots()
    ax.plot(angle_recalc, pe_mo)
    # ax.plot([180/57.3*1e3, 180/57.3*1e3], [ax.get_ylim()[0], ax.get_ylim()[1]], 'r--')
    ax.set_xlabel('Non-colinearity angle [mrad]', fontweight = 'bold')
    ax.set_ylabel('Mounting Offset Resolution Error [mrad]', fontweight = 'bold')
    # ax.set_xlim([10, 0])
    # ax.set_xscale('log')
    # ax.set_yscale('log')
    ax.grid('on')
    # ax.set_ylim([0,50])

