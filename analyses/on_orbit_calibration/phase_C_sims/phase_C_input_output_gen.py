#%% Feb 1, 2024 - generating inputs with expected errors
# according to the in-orbit phase C simulation plan
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
csv_output_path = r'orbital_simulations\leo_for_pmg\leo_leo_srpcheck'
fname_simparam = 'simulation_parameters.json'

# Use for full-1s time-step data:
fname_states = 'states_fine.dat'
# Use for coarse 60-s step data:
# fname_states = 'state_history.dat'

## MVP imports
import basic_tools.vector_operations as vec_calc
import basic_tools.time_conversion as t_conv
import basic_tools.in_out as io
import plotting_tools.basic_plotting as bplt
import plotting_tools.plotting_utilities as plt_util
import plotting_tools.combined_plots as cmbplt
import astronomy_tools.constants as const
import astronomy_tools.astro_targets as where_sun

import attitude_tools.attitude_resolution as att_res
import attitude_tools.conversions as conv
import attitude_tools.rotations as rot
import attitude_tools.attitude_simulation as att_sim

import basic_tools.in_out as savedat

import prediction_methods.error_generation as err_gen
import pointing_calculations.ae_calculation as ae_calc
import pointing_calculations.ae_profiles as ae_profile
import pointing_calculations.simulate_moon_scan as moon_scan
import tudat_tools.data_processing.data_processing_utilities as dputil


remove_error_components = 1
if remove_error_components:
    print('REMOVING ALL ERROR COMPONENTS')
data_raw, simulation_parameters = dputil.load_constellation_data(full_path = csv_output_path, 
                                                                 state_name = fname_states)
host_chosen = 'leo_polar1'
target_chosen = 'leo_polar2'
t_j2000 = data_raw[:,0]
t_gps_full = t_j2000 + t_conv.dt_j2000tt2gps()
r_host_full = data_raw[:,simulation_parameters['r_index'][host_chosen]]
v_host_full = data_raw[:,[ii+3 for ii in simulation_parameters['r_index'][host_chosen]]]
r_target_full = data_raw[:,simulation_parameters['r_index'][target_chosen]]
v_target_full = data_raw[:,[ii+3 for ii in simulation_parameters['r_index'][target_chosen]]]

t_data_req = 3* 3600 # 1 hour

ii_data = np.where((t_gps_full-t_gps_full[0] > t_data_req))[0][0]
r_host_t = r_host_full[:ii_data,:]
v_host = v_host_full[:ii_data,:]
t_gps = t_gps_full[:ii_data]
r_target_t = r_target_full[:ii_data,:]

### Error Components Affecting Commanded Az/El values
## Error Components 
mounting_offset_rpy = [90, 4, 2.5] # MOUNTING OFFSET random 3-axis wrotation
error_mounting_offset = 1e-3 # mrad, Mounting offset error after Phase B

## Mounting Offset
rot_bf2lct = conv.convert_ea2dcm(mounting_offset_rpy)
quat_mounting_offset_t = conv.convert_ea2quat(mounting_offset_rpy)
quat_mounting_offset_c = conv.convert_dcm2quat(conv.convert_eigenaxis2dcm([-1,2,3],error_mounting_offset) @ conv.convert_quat2dcm(quat_mounting_offset_t))
if remove_error_components:
    quat_mounting_offset_c = quat_mounting_offset_t
## Position and Attitude Errors

err_r_host_default = np.array([6, 7, 8]) # 12.2 m host GPS errors
err_r_target_default = np.array([80, 60, 10]) # 100 m target position error expected
err_att_host_default = np.array([3.2e-4, 3.2e-4, 3.2e-4]) # 0.55 mrad

err_r_host = err_gen.pos_err_gen([0,0,0], err_r_host_default, nrows = t_gps.shape[0])
err_r_target = err_gen.pos_err_gen([0,0,0], err_r_target_default, nrows = t_gps.shape[0])
err_att_host = err_gen.pos_err_gen([0,0,0], err_att_host_default, nrows = t_gps.shape[0])
### Error Components affecting tracked/logged Az/El
## Random error Components affecting AE Tracked/logged
RSS_components = {
    'gimbal_control' : 0.4, 
    'thermal' : 0.8
}
RSS_random_errors = np.linalg.norm(list(RSS_components.values()))*1e-3
err_pointing_random = err_gen.pos_err_gen([0], [RSS_random_errors/3], nrows = t_gps.shape[0])

if remove_error_components:
    RSS_random_errors = RSS_random_errors * 0
    err_pointing_random = err_pointing_random * 0
    err_r_host = err_r_host * 0
    err_r_target = err_r_target * 0
    err_att_host = err_att_host * 0
    err_r_host_default = err_r_host_default * 0
    err_r_target_default = err_r_target_default * 0
    err_att_host_default = err_att_host_default * 0
## PMG error components
# should sum to 500 urad
n_pmg_terms = 8
pm_err_tot = 0.5 * 1e-3
np.random.seed(1)
pmg_components_az = {
    'aoff' : pm_err_tot/n_pmg_terms * np.random.randn(),
    'aan' : pm_err_tot/n_pmg_terms * np.random.randn(),
    'aae' : pm_err_tot/n_pmg_terms * np.random.randn(),
    'npae' : pm_err_tot/n_pmg_terms * np.random.randn(),
    'bnp' : pm_err_tot/n_pmg_terms * np.random.randn(),
    'aes' : pm_err_tot/n_pmg_terms * np.random.randn(),
    'aec' : pm_err_tot/n_pmg_terms * np.random.randn(),
}
pmg_components_el = {
    'eoff' :pm_err_tot/n_pmg_terms * np.random.randn(),
    'ean' :  pm_err_tot/n_pmg_terms * np.random.randn(),
    'eae' :  pm_err_tot/n_pmg_terms * np.random.randn(),
    'eec' :  pm_err_tot/n_pmg_terms * np.random.randn(),
    'es2a' : pm_err_tot/n_pmg_terms * np.random.randn(),
    'ec2a' : pm_err_tot/n_pmg_terms * np.random.randn(),
    'es3a' : pm_err_tot/n_pmg_terms * np.random.randn(),
    'ec3a' : pm_err_tot/n_pmg_terms * np.random.randn(),
}

## Positions with errors
r_host_c = r_host_t + err_r_host
r_target_c = r_target_t + err_r_target


### True Az/El and Commanded Az/El calculation
if 0:
    # get body-frame attitude for specific pointing mode
    attitude_profile = 'earth_roll'
    roll_required = 0.2
    quat_eci2bf_t, quatdot, rot_eci2bf = att_sim.calc_quat_eci2bf(r_host_t, 
                                                                v_host, 
                                                                attitude_profile,
                                                                t_gps = t_gps.flatten(), 
                                                                roll_velocity=roll_required,
                                                                rotation_axis=1)
else:
    # get quaternion based on az/el profile
    importlib.reload(ae_profile)
    ang_rate = 0.51 # deg/s
    nr_el_steps = 5
    ae_desired, ii_req = ae_profile.get_ae_for_pmg(nr_el_steps=nr_el_steps, 
                                           ang_rate = ang_rate,
                                           t_vec = t_gps,
                                           az_lims=[-170, 170],
                                           el_lims = [85, -60]
                                           )
    quat_eci2bf_t, ea_eci2bf_command_all = att_res.calc_attitude_for_ae(
        r_host = r_host_c,
        t_gps = t_gps_full,
        r_target = r_target_c,
        quat_mo = quat_mounting_offset_c,
        ae_desired = ae_desired
    )

    t_gps = t_gps[:ii_req]
    r_host_t = r_host_t[:ii_req,:] 
    r_target_t = r_target_t[:ii_req,:]
    r_host_c = r_host_c[:ii_req,:] 
    r_target_c = r_target_c[:ii_req,:]
    # 
quat_eci2bf_c, errors_eci2bf = err_gen.add_errors_to_attitude(quat_true=quat_eci2bf_t,
                                                                  std_err_att=err_att_host_default
                                                                  )

ae_tracked_true = ae_calc.calc_ae_full(r_host_t, r_target_t, quat_eci2bf_t, quat_mounting_offset_t)
ae_commanded = ae_calc.calc_ae_full(r_host_c, 
                                    r_target_c, 
                                    quat_eci2bf_c , 
                                    quat_mounting_offset_c)

ae_tracked_withrandomerr = ae_calc.calc_ae_full(r_host_t, r_target_t, quat_eci2bf_t, quat_mounting_offset_t, centroid_error = err_pointing_random)
ae_with_pm, ae_pm_errors = err_gen.calculate_pmg_errors(ae_tracked_true, pmg_components_az, pmg_components_el)
ae_tracked_logged = np.copy(ae_tracked_withrandomerr)
ae_tracked_logged[:,:2] = ae_tracked_logged[:,:2] + ae_pm_errors
# ae_tracked_logged = ae_calc.calc_ae_full(r_host_ii_c, r_moon_ii_c, q_eci2bf_c, quat_mounting_offset_c)

# convert to [deg]  
ae_tracked_true[:,:2] = np.rad2deg(ae_tracked_true[:,:2])
ae_commanded[:,:2] = np.rad2deg(ae_commanded[:,:2])
ae_tracked_withrandomerr[:,:2] = np.rad2deg(ae_tracked_withrandomerr[:,:2])
ae_tracked_logged[:,:2] = np.rad2deg(ae_tracked_logged[:,:2])
#%% Store and save

## Available data - logged and commanded AE
fname_logcom = f'pmg_tracking_fixedae_noerr{remove_error_components}'
data_logcom = np.hstack((ae_commanded[:,:2], ae_tracked_logged[:,:2]))
colums_logcom = [
    't_gps_s',
    'az_command_deg',
           'el_command_deg',
           'az_trackedlog_deg',
           'el_trackedlog_deg',
           ]

fname_true = f'pmg_trackingtrue_fixedae_noerr{remove_error_components}'
data_true = ae_tracked_true[:,:2]
colums_true = [
    't_gps_s',
'az_true_deg',
'el_true_deg',
           ]

io.make_n_save(
    fname = fname_logcom,
    data = data_logcom,
    t_vec = t_gps,
    data_cols=colums_logcom,
    subfolder = 'pmg_sim_aefixed'
               )
## Unavailable data - true + PMG terms

io.make_n_save(
    fname = fname_true,
    data = data_true,
    t_vec = t_gps,
    data_cols=colums_true,
    subfolder = 'pmg_sim_aefixed'
               )

pmg_components_az.update(pmg_components_el)
io.save_dct2json(pmg_components_az,'pmg_coefficients',
                 subfolder = 'pmg_sim_aefixed')

#%% plot PMG terms
if 1:
    f, axs = plt.subplots(nrows = 2)
    ax = axs[0]
    vals = []
    xvals = []
    for ii, key in enumerate(pmg_components_az):
        xvals.append(key)
        vals.append(1e3*pmg_components_az[key])
    ax.stem(xvals, vals)
    ax.grid()
    ax.set_ylabel('PMG Az Comp. [mrad]')
    ax = axs[1]
    vals = []
    xvals = []
    for ii, key in enumerate(pmg_components_el):
        xvals.append(key)
        vals.append(1e3*pmg_components_el[key])
    ax.stem(xvals, vals)
    ax.set_ylabel('PMG El Comp. [mrad]')
    ax.grid()
#%%
if 1:
    importlib.reload(cmbplt)
    f, ax = cmbplt.plot_ae(t_gps - t_gps[0], ae_pm_errors*1e3, title = 'PM errors added', unit = 'mrad')
    # f, ax = cmbplt.plot_ae  (t_gps - t_gps[0], ae_tracked_true)
    f, ax = cmbplt.plot_ae(t_gps - t_gps[0], ae_commanded, title = 'Commanded Az/El')
    f, ax = cmbplt.plot_ae(t_gps - t_gps[0], 1e3*np.deg2rad(ae_commanded - ae_tracked_true), title = 'Error: True vs Commanded', unit = 'mrad')    
    # f, ax = cmbplt.plot_ae(t_gps - t_gps[0], ae_tracked_true - ae_tracked_withrandomerr, title = 'Error: True vs Logged', label = 'no PM',
    #                        axis_label_appends='Delta_', alpha = 0.5)
    f, ax = cmbplt.plot_ae(t_gps - t_gps[0], 1e3*np.deg2rad(ae_tracked_true - ae_tracked_logged), title = 'Error: True vs Tracked logged', unit = 'mrad')
    # f, ax = cmbplt.plot_ae(t_gps - t_gps[0], 1e3*np.deg2rad(ae_tracked_true - ae_tracked_logged), title = 'Error: True vs Tracked logged', unit = 'mrad', axlim = 'equal')


    
if 0:
    for ii, az in enumerate(ae_commanded[:-1,0]):
        if np.abs(ae_commanded[ii+1,0] - az)>180:

            print(f'''{ii} -> {ae_commanded[ii+1,0] - az}. AE : {ae_commanded[ii,:2]} -> {ae_commanded[ii+1,:2]}
                quat_com : {quat_eci2bf_c[ii]}. 
                EA MINUS 1 : {ea_eci2bf_command_all[ii-1,:3]}
                EA : {ea_eci2bf_command_all[ii,:3]}
                quat_com : {quat_eci2bf_c[ii+1]} 
                EA : {ea_eci2bf_command_all[ii+1,:3]}
                    
                    ''')
            