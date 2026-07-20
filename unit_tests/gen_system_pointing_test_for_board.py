#%% IMPORTS

## Prep test data for J2 proapgator, being implemented 
# November 2023

# Req columns: 
# t_GPS;s_DATA IN : Timestamped data, as inputs from sat bus [dt = 1s/10s/whatever]
# t_GPS_now : GPS time in the PRESENT; 5 ms steps
# t_GPS; s_true: true data, timestamped data as truth; 5 ms steps

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
import scipy as sp
sys.path.insert(1, os.getcwd()[:os.getcwd().index('astropynaric')+13])

# path jazz
path_cwd = os.getcwd()

csv_output_path = r'orbital_simulations\capella_leo_polar_2000km_medium_precision\capella_leo_polar_2000km'

fname_simparam = 'simulation_parameters.json'
fname_states = 'state_history.dat'

## MVP imports
import basic_tools.vector_operations as vec_calc
import astronomy_tools.constants as const
import plotting_tools.basic_plotting as bplt
import basic_tools.time_conversion as t_conv
import basic_tools.in_out as out
import prediction_methods.j2propagator as j2prop
import prediction_methods.interpolators as interp
import prediction_methods.error_generation as errgen
importlib.reload(out)
import attitude_tools.attitude_simulation as att_sim
import attitude_tools.conversions as att_conv
import basic_tools.in_out as io
import pointing_calculations.ae_calculation as ae_calc


## Loading satellite orbital data
importlib.reload(j2prop)
import tudat_tools.data_processing.data_processing_utilities as dputil
data_raw, simulation_parameters = dputil.load_constellation_data(full_path = csv_output_path)

host_chosen = simulation_parameters['sat_names'][0]
target_chosen = simulation_parameters['sat_names'][1]
output_folder = r'outputs/tables/orbital_quat_inputs' 
label = 'leo2leo_spin_2000km'
full_output_folder = fr'{output_folder}/{label}' 
# host_chosen = 'leo_target_polar'
# host_chosen = 'leo_host_incl'
# host_chosen = 'meo_target_incl'
print_cond = 0
add_errors = 0
t_j2000 = data_raw[:,0]
t_gps = t_j2000 + t_conv.dt_j2000tt2gps()
dt_raw = t_gps[1] - t_gps[0]

r_host = data_raw[:,simulation_parameters['r_index'][host_chosen]]
v_host = data_raw[:,[ii+3 for ii in simulation_parameters['r_index'][host_chosen]]]

r_target = data_raw[:,simulation_parameters['r_index'][target_chosen]]
v_target = data_raw[:,[ii+3 for ii in simulation_parameters['r_index'][target_chosen]]]

s_host = np.hstack((r_host, v_host))
s_target = np.hstack((r_target, v_target))
# generate full time-vector
#%%
q_all, q_dot_all, rot_eci2bf = att_sim.calc_quat_eci2bf(s_host[:,[0, 1,2]], s_host[:,[3,4,5]],
                                                         t_gps = t_gps, 
                                                         att_profile = 'earth_point', 
                                                         euler_rates = None, 
                                                         roll_velocity = 0.84,
                                                         add_ideal_jerk=0,
                                                         add_realistic_jerk=0,                                                         
                                                calc_qdot = 1)

# [s] TIME OFF-sets used to determine if datapoitns are in past/future/present
# ii_data_incoming_now = np.arange(1, len(t_gps_full_5ms), int(dt_timeupd_req/dt_truth_req))
# ii_data_timestamps = [ii - int(t_delay/dt_truth_req) for ii in ii_data_incoming_now]
# ii_data_timestamps = [ii + ii_truth_0 for ii in ii_data_timestamps]

if add_errors: # TODO
    err_r = errgen.pos_err_gen(error_r_3d_mean, error_r_3d, len(t_stamp_data))
    err_v = errgen.pos_err_gen(error_v_3d_mean, error_v_3d, len(t_stamp_data))
    # err_att = 
else:
    err_r = np.zeros((s_host.shape[0],3))
    err_v = np.zeros((s_host.shape[0],3))
    err_q = np.zeros((s_host.shape[0],4))

# Add errors TODO
s_host[:,:3] = s_host[:,:3] + err_r
s_host[:,3:] = s_host[:,3:] + err_v

q_eci2bf = np.hstack((q_all, q_dot_all))
q_mo = np.copy(q_eci2bf)
q_mo[:,0] = 1
q_mo[:,1:] = 0

aer_data = ae_calc.calc_ae_full(s_host, s_target, attitude_eci2bf = q_eci2bf,
                                                attitude_mountingoffset=q_mo[:,:4])
# aer_data[:,:2] = np.rad2deg(aer_data[:,:2])
if print_cond:
    for ii, tstamp in enumerate(t_stamp_data):
        print(f'At t={t_rec_data[ii] - t_gps_full_5ms[0]:.1f}, data coming with t stamp = {tstamp - t_gps_full_5ms[0]:.3f}; 3D err = {np.linalg.norm(err_r[ii,:]):.0f} m')

save = 1
plot = 1

# data_to_store = np.hstack((r_host_predicted, np.linalg.norm(dr, axis = 1).reshape(t_j2000.shape[0],1)))
# data_to_store[:,0] = t_gps

if save:
    output_success, df = io.save_azel(t_gps,
                                        s_host,
                                        s_target,
                                        q_eci2bf,
                                        q_mo,
                                        aer_data,
                                        fname = 'leo_2_leo_data',
                                        full_folder = full_output_folder)

    title_reftime = 'ref_time'
    output_times_dict = { 'day_used' : [1],
        'month_used' : [1],
        'h_start' : [1],
        'h_end' : [2],
        't_res' : [dt_raw]}
    df_date = pd.DataFrame.from_dict(output_times_dict)
    df_date.to_csv(f"{full_output_folder}//{title_reftime}.csv", index = False)
if plot:
    f, ax = plt.subplots()
    ax.plot(t_gps[:ii_end+ii_truth_0]-t_gps_full_5ms[0], s_host[:ii_end+ii_truth_0,0], label = 'raw_data', linewidth = 10, alpha = 0.2)

    ax.plot(t_gps_full_5ms-t_gps_full_5ms[0], states_full_5ms[:,0], label = '5 ms data for sim')

    ax.scatter(t_stamp_data-t_gps_full_5ms[0], s_data[:,0], label = 'data inputs', c = 'r', s = 10)
    ax.set_ylabel('x [m]')
    ax.set_xlabel('t since sim start [s]')
    ax.set_xlim([-30,50])
    ax.legend()
