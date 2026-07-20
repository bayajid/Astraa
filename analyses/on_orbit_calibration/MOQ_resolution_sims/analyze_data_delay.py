#%% May 10, analyzing how long of a time delay between tracked and comm az/el data can be tolerated.
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
path_rel = r'analyses\on_orbit_calibration\MOQ_resolution_sims/'
fname_states = f'{path_rel}host_states_fine.csv'

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
import basic_tools.vector_operations as vec_op
import pointing_calculations.ae_calculation as ae_calc
import pointing_calculations.simulate_moon_scan as moon_scan
import tudat_tools.data_processing.data_processing_utilities as dputil
importlib.reload(bplt)
importlib.reload(moon_scan)
importlib.reload(att_res)

data_fine = pd.read_csv(fname_states).values
t_vec = data_fine[:,0] - data_fine[0,0]

t_desired = 25 # 10 seconds of data
ii_desired = np.where(t_vec < t_desired)[0]
data_sliced = data_fine[ii_desired,:]
astro = where_sun.body_fromsp(t_input = t_vec[0])
t_gps = data_sliced[:,0]
t_vec_desired = t_gps - t_gps[0]
r_moon = [astro.get_sun(body='moon', dt_second = dt_t) for dt_t in t_vec_desired]
r_moon = np.array(r_moon)
r_host = data_sliced[:,[1,2,3]]
v_host = data_sliced[:,[4,5,6]]

trolley_angles_used = [0.5]
downsampling_step = 100


# DEBUG/VERIFICATION PURPOSES
# title_append = 'no_mean_errors'
# title_append = 'double_mean_errors'
# title_append = 'only_cmd_errors'
# title_append = 'only_att_errors'
# title_append = 'only_att_errorsx2'
# title_append = 'full_magnitude'
# title_append = 'no_errors'
title_append = 'phase_A'

spacing = 'uniform'
# n_samples = 10000

nrows = r_host.shape[0]
mounting_offset_rpy = [5, 4, 2.5] # MOUNTING OFFSET random 3-axis wrotation
error_mounting_offset = 65e-3 # mrad, Mounting offset error

rot_bf2lct = conv.convert_ea2dcm(mounting_offset_rpy)
quat_mounting_offset = conv.convert_ea2quat(mounting_offset_rpy)
quat_mounting_offset_known = conv.convert_dcm2quat(conv.convert_eigenaxis2dcm([-1,2,3],error_mounting_offset) @ conv.convert_quat2dcm(quat_mounting_offset))
quat_mounting_offset_known = quat_mounting_offset_known / np.linalg.norm(quat_mounting_offset_known)

# TODO select attitude profile
importlib.reload(att_sim)
importlib.reload(conv)
importlib.reload(att_res)
storage_folder = 'ooc_phase_A_illum'
## COMPUTE ATTITUDE
attitude_profile = 'earth_roll'

# 
err_r_host_default = np.array([6, 7, 8]) # RANDOMIZE DIRECTION? ACTUALLY TODO what magnitude??
err_r_target_default = np.array([-60, -70, 80]) # RANDOMIZE DIRECTION? 100 m target position error expected
# https://space.leonardo.com/documents/16277711/19573187/Copia_di_A_STR_Autonomous_Star_Trackers_LQ_mm07786_.pdf?t=1538987562062 - 20 arcsec/axis
# Attitude error ~ 0.5 mrad
err_att_host_default = np.array([3.2e-4, 3.2e-4, 3.2e-4]) # rad # RANDOMIZE DIRECTION?

# mrad, RSS components
RSS_components = {
    'gimbal_control' : 0.4,
    'thermal' : 0.8,
    'other' : 0.1,
}
moon_ilum_fraction = .9
title_append = f'{title_append}_moon{moon_ilum_fraction}'
centroid_err = 0.25*17*(1-moon_ilum_fraction)

mean_components = {
    'pointing_model_residuals' : 0.5,
    'centroid_detection' : centroid_err
}
RSS_random_errors = np.linalg.norm(list(RSS_components.values()))*1e-3 # rad
sum_mean_errors = np.sum(list(mean_components.values())) * 1e-3
total_random_error_input = RSS_random_errors + sum_mean_errors
total_random_error_input = total_random_error_input # [rad]
importlib.reload(moon_scan)
importlib.reload(ae_calc)
# n_samples = 60000 

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
elif title_append == 'no_errors':
    sum_mean_errors = sum_mean_errors * 0
    RSS_random_errors = RSS_random_errors * 0
    err_r_host_default = err_r_host_default * 0
    err_r_target_default = err_r_target_default * 0
    err_att_host_default = err_att_host_default * 0
elif title_append == 'phase_A':
    err_r_host_default = err_r_host_default * 0
    err_r_target_default = err_r_target_default * 0
errors_chosen = {
        'err_r_host': err_r_host_default,
        'err_r_target': err_r_target_default,
        'err_att_host': err_att_host_default,
        'sum_mean_errors': sum_mean_errors,
        'rss_random_errors': RSS_random_errors,
    }
randomize_magnitude = 1
nr_inputs = r_host.shape[0]
dt_data = t_vec[1] - t_vec[0]
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
# SET SEED

# for ii_loop, att_angle in enumerate(angles_desired):
ea_eci2bf = np.zeros((nrows, 6))
roll_required = 0.0
try:
    quat_eci2bf, quatdot, rot_eci2bf = att_sim.calc_quat_eci2bf(r_host, v_host, attitude_profile,
                                                        t_gps = t_gps.flatten(), 
                                                        roll_velocity=roll_required,
                                                        rotation_axis=1)
except:
    quat_eci2bf, quatdot, rot_eci2bf = att_sim.calc_quat_eci2bf(r_host, v_host, attitude_profile,
                                                        t_gps = t_gps.flatten(), 
                                                        roll_velocity=roll_required*1.001,
                                                        rotation_axis=1)
for mm, dcm_ii in enumerate(rot_eci2bf):
    ea_eci2bf[mm,:3] = conv.convert_dcm2ea(dcm_ii)
# run-scan
# run-scan # TODO integrate QUEST into simulating moon scan code
ae_moon_commanded_all, ae_moon_true_all = moon_scan.simulate_ooc(
    ii_scans = ii_scans,
    function_option = 1,
    r_host = r_host, 
    r_target = r_moon, 
    ea_eci2bf_command_all = ea_eci2bf,
    quat_mounting_offset_t = quat_mounting_offset,
    quat_mounting_offset_c = quat_mounting_offset_known,
    manual_error_dict=errors_chosen,
    check_non_colin=1,
    randomize_error_direction = 1,
    centroid_dirction_randomizer = 1,
    randomize_magnitude = randomize_magnitude,
    just_calc_ae=1,
)
#%%

ae_c_full = ae_moon_commanded_all
ae_t_full = ae_moon_true_all
import basic_tools.vector_operations as vec_op
# get LOS
los_c_full = np.array([vec_op.convert_polar_to_cartesian(ae_ii) for ae_ii in ae_c_full])
los_t_full = np.array([vec_op.convert_polar_to_cartesian(ae_ii) for ae_ii in ae_t_full])

delta_c_angle = 1e3* np.array([vec_op.calc_dot_angle(los_c_full[0,:], los_ii) for los_ii in los_c_full[1:,:]])
delta_t_angle = 1e3* np.array([vec_op.calc_dot_angle(los_t_full[0,:], los_ii) for los_ii in los_t_full[1:,:]])
#%%
f, axs = plt.subplots(nrows = 2)
suptitles = ['Command','True']
xlims = [[0,5], [0,0.1]]
ylims = [[0, 5], [0, 0.1]]
for ii, d_angle in enumerate([delta_c_angle, delta_t_angle]):
    ax = axs[ii]
    
    ax.plot(t_vec_desired[1:], d_angle)
    ax.grid()
    ax.set_xlabel('t [s]')
    ax.set_ylabel('dtheta [mrad]')
    ax.set_title(suptitles[ii])
    ax.set_xlim(xlims[ii])
    ax.set_ylim(ylims[ii])
f.set_tight_layout('tight')
importlib.reload(bplt)
bplt.savefig(f, 'delta_angles', subfolder = 'phaseA_azel_update_rates', y_coord_tag=-2.5)

#%%
f, axs = plt.subplots(nrows = 2)
suptitles = ['Az','El']
# xlims = [[0,5], [0,0.1]]
# ylims = [[0, 5], [0, 0.1]]

for ii, ax in enumerate(axs):
    ax.plot(t_vec_desired, los_c_full[:,ii], label = 'comm')
    ax.plot(t_vec_desired, los_t_full[:,ii], label = 'track')
    ax.grid()
    ax.set_xlabel('t [s]')
    ax.set_ylabel(['Az [rad]', 'El [rad]'][ii])
    # ax.set_title(suptitles[ii])
    # ax.set_xlim(xlims[ii])
    # ax.set_ylim(ylims[ii])
    ax.legend()
f.set_tight_layout('tight')
bplt.savefig(f, 'AzElTotal', subfolder = 'phaseA_azel_update_rates', y_coord_tag = 0.3)