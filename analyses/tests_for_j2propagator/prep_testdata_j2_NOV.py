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

test_case = 9 

if test_case >3:
    csv_output_path = r'orbital_simulations\single_sat\leo_hp_prop_1s_1d'
else:
    # csv_output_path = r'orbital_simulations\single_sat\leo_j2_prop'
    csv_output_path = r'orbital_simulations\single_sat\leo_j2_prop_1s_1d'

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
## LIST COMPREHENSION ft. if else
# [x+1 if x >= 45 else x+5 for x in l]

## Loading satellite orbital data
importlib.reload(j2prop)
import tudat_tools.data_processing.data_processing_utilities as dputil
data_raw, simulation_parameters = dputil.load_constellation_data(full_path = csv_output_path)

host_chosen = simulation_parameters['sat_names'][0]
# host_chosen = 'leo_target_polar'
# host_chosen = 'leo_host_incl'
# host_chosen = 'meo_target_incl'

t_j2000 = data_raw[:,0]
t_gps = t_j2000 + t_conv.dt_j2000tt2gps()
dt_raw = t_gps[1] - t_gps[0]

if test_case == 1:
    # present-time data
    test_name = '6 datapoints in 12s, present. No errors'
    ii_truth_0 = 1
    ii_data_incoming_now = [1, 2, 3, 5, 6, 12]
    ii_data_timestamps = [ii-1 for ii in ii_data_incoming_now]
    # setup test_cases
    dt_truth_req = 5e-3
    dt_truth_length = 20
    add_errors = False
if test_case == 2:
    # data coming with old/new timestamps
    # time WHEN teh data updates come
    test_name = '6 datapoints in 10s, future past. No errors'
    ii_truth_0 = 20

    ii_data_incoming_now = [1, 2, 3, 5, 6, 8]
    # [s] TIME OFF-sets used to determine if datapoitns are in past/future/present
    ii_data_timestamps = [-10, -9, 30, 1, 10, 3]
    # setup test_cases
    dt_truth_req = 5e-3
    dt_truth_length = 40
    add_errors = False
if test_case == 3:
    # data coming with old/new timestamps
    # time WHEN teh data updates come
    test_name = '4 datapoints, 1s past, 10s apart, 10m random errors'
    ii_truth_0 = 40

    ii_data_incoming_now = [10, 20, 30, 40]
    # # [s] TIME OFF-sets used to determine if datapoitns are in past/future/present
    ii_data_timestamps = [ii-1 for ii in ii_data_incoming_now]
    # # setup test_cases
    dt_truth_req = 5e-3
    dt_truth_length = 50

    add_errors = True
    error_r_3d = 10/3
    error_v_3d = 0.2/3
    error_r_3d_mean = 0
    error_v_3d_mean = 0
if test_case == 4:
    # data coming with old/new timestamps
    # time WHEN teh data updates come
    test_name = '2 datapoints, 1s past, 10s apart, 10m random errors, HP orbits'
    ii_truth_0 = 50

    ii_data_incoming_now = [0, 50]
    # # [s] TIME OFF-sets used to determine if datapoitns are in past/future/present
    ii_data_timestamps = [-1, 49]
    # # setup test_cases
    dt_truth_req = 5e-3
    dt_truth_length = 100

    add_errors = True
    error_r_3d = 10/3
    error_v_3d = 0.2/3
    error_r_3d_mean = 0
    error_v_3d_mean = 0
if test_case == 5:
    # data coming with old/new timestamps
    # time WHEN teh data updates come
    test_name = '1 datapoint, instan, 50s in future, 10m rand err'
    ii_truth_0 = 20

    ii_data_incoming_now = [0]
    # # [s] TIME OFF-sets used to determine if datapoitns are in past/future/present
    ii_data_timestamps = [50]
    # # setup test_cases
    dt_truth_req = 5e-3
    dt_truth_length = 100

    add_errors = True
    error_r_3d = 10
    error_v_3d = 0.2
    error_r_3d_mean = 0
    error_v_3d_mean = 0
if test_case == 6:
    # data coming with old/new timestamps
    # time WHEN teh data updates come
    test_name = '2 datapoint, 1s in past, 100s apart, 10m rand err'
    ii_truth_0 = 2000

    ii_data_incoming_now = [0, 100]
    # # [s] TIME OFF-sets used to determine if datapoitns are in past/future/present
    ii_data_timestamps = [-1, -1]
    # # setup test_cases
    dt_truth_req = 5e-3
    dt_truth_length = 100

    add_errors = True
    error_r_3d = 10
    error_v_3d = 0.2
    error_r_3d_mean = 0
    error_v_3d_mean = 0    
if test_case == 7:
    # data coming with old/new timestamps
    # time WHEN teh data updates come
    test_name = '15 datapoints, 0.5 s apart, 5x 0.5s old, 5x present, 5x 0.5s in future'
    ii_truth_0 = 2000

    ii_data_incoming_now = [0, 100]
    # # [s] TIME OFF-sets used to determine if datapoitns are in past/future/present
    ii_data_timestamps = [-1, -1]
    # # setup test_cases
    dt_truth_req = 5e-3
    dt_truth_length = 50

    add_errors = True
    error_r_3d = 10
    error_v_3d = 0.2
    error_r_3d_mean = 0
    error_v_3d_mean = 0    
if test_case == 8:
    # data coming with old/new timestamps
    # time WHEN teh data updates come
    test_name = '6 datapoints in 10s, future past. 10 m errors'
    ii_truth_0 = 20

    ii_data_incoming_now = [1, 2, 3, 5, 6, 8]
    # [s] TIME OFF-sets used to determine if datapoitns are in past/future/present
    ii_data_timestamps = [-10, -9, 30, 1, 10, 3]
    # setup test_cases
    dt_truth_req = 5e-3
    dt_truth_length = 50
    add_errors = True
    error_r_3d = 10
    error_v_3d = 0.2
    error_r_3d_mean = 0
    error_v_3d_mean = 0  
if test_case == 9:
    # data coming with present timestamps for interpolation
    # and spaced out ones for propagation

    test_name = '11 datapoints in 50s, 3x future 4 s in future, then 1s past 10m errors'
    ii_truth_0 = 50

    ii_data_incoming_now = [1, 6, 11, 16, 21, 26, 31, 36, 46, 51, 56]
    # [s] TIME OFF-sets used to determine if datapoitns are in past/future/present
    ii_data_timestamps = [6, 11, 16, 21, 26, 31, 36, 41, 46, 51, 56]
    ii_data_timestamps = [ii - 1 for ii in ii_data_timestamps]
    # setup test_cases
    dt_truth_req = 5e-3
    dt_truth_length = 100
    add_errors = True
    error_r_3d = 10
    error_v_3d = 0.2
    error_r_3d_mean = 0
    error_v_3d_mean = 0                   
# generate full time-vector
ii_end = ii_truth_0 + int(dt_truth_length / dt_raw)
t_gps_end = t_gps[ii_truth_0] + dt_truth_length
t_gps_full = t_gps[ii_truth_0:ii_end]
t_gps_full_5ms = np.arange(t_gps[ii_truth_0], t_gps_end, dt_truth_req)
r_host = data_raw[:,simulation_parameters['r_index'][host_chosen]]
v_host = data_raw[:,[ii+3 for ii in simulation_parameters['r_index'][host_chosen]]]
s_host = np.hstack((r_host, v_host))
print(f'Test case nr {test_case}\n:{test_name}. \nTruth length : {dt_truth_length} s')
ii_data_incoming_now = [ii + ii_truth_0 for ii in ii_data_incoming_now]
ii_data_timestamps = [ii + ii_truth_0 for ii in ii_data_timestamps]

states_full = s_host[ii_truth_0:ii_end, :]

state_interpolant = sp.interpolate.CubicSpline(t_gps_full.flatten(), states_full)
states_full_5ms = state_interpolant(t_gps_full_5ms)

casename = f'tests_j2_p{test_case}'

if test_case == 7:
    # get sub-second datapoints
    dt_rec_data = 0.5
    t_rec_data = np.arange(t_gps[ii_truth_0], t_gps[ii_truth_0+int(15*dt_rec_data)], dt_rec_data)
    t_stamp_data = np.copy(t_rec_data)
    t_stamp_data[:5] = t_stamp_data[:5] - 0.5
    t_stamp_data[-5:] = t_stamp_data[-5:] + 0.5
    
    s_data = state_interpolant(t_stamp_data)
else:
    # time vector when data is coming
    t_rec_data = t_gps[ii_data_incoming_now]

    # timestamp of the incoming data
    t_stamp_data = t_gps[ii_data_timestamps]
    s_data = s_host[ii_data_timestamps,:]

if add_errors:
    err_r = errgen.pos_err_gen(error_r_3d_mean, error_r_3d, len(t_stamp_data))
    err_v = errgen.pos_err_gen(error_v_3d_mean, error_v_3d, len(t_stamp_data))
else:
    err_r = np.zeros((s_data.shape[0],3))
    err_v = np.zeros((s_data.shape[0],3))

s_data[:,:3] = s_data[:,:3] + err_r
s_data[:,3:] = s_data[:,3:] + err_v



for ii, tstamp in enumerate(t_stamp_data):
    print(f'At t={t_rec_data[ii] - t_gps_full_5ms[0]}, data coming with t stamp = {tstamp - t_gps_full_5ms[0]}; 3D err = {np.linalg.norm(err_r[ii,:]):.0f} m')

save = 1
plot = 0
if 0:
    # PROPAGATE, make interpolants, compare interpolant matrices (debug SL)
    ii_test = 0
    s0 = s_data[0,:]
    t0 = t_stamp_data[0]
    # first long propagation
    t1 = t0 + 11
    s1 = j2prop.integrate_rk4(j2prop.calc_f_J2_separate, x_0=s0, t_vec_prop=[t0], t_step = 11, return_both=0)

    # second standard propagation
    t2 = t1 + 10
    s2 = j2prop.integrate_rk4(j2prop.calc_f_J2_separate, x_0=s1, t_vec_prop=[t1], t_step = 10, return_both=0)

    interp_class = interp.we_interpolating()
    interp_class.get_quad_interpolant([t1, t2], 
                                      r_both = np.vstack((s1[:3], s2[:3])), 
                                      v_both = np.vstack((s1[3:], s2[3:])))
    dr = interp_class.interpolate(np.array([t2+0.5])).flatten() - state_interpolant(t2 + 0.5)[:3]
    print(f'''Interp coeff : \n{interp_class.coeff} Error at dt=0.5s:\n
          {dr} m
          ''')
# data_to_store = np.hstack((r_host_predicted, np.linalg.norm(dr, axis = 1).reshape(t_j2000.shape[0],1)))
# data_to_store[:,0] = t_gps

if save:
    columns_datainc = ['t_now_gps','t_stamp_gps','x', 'y', 'z', 'vx', 'vy', 'vz']
    out.make_n_save(f'tstamp_incoming_{casename}', t_vec = t_rec_data, 
                    data = np.hstack((t_stamp_data.reshape((len(t_stamp_data),1)), s_data)), 
                    data_cols = columns_datainc, 
                    subfolder = 'j2prop_testdata')
    
    columns_truth = ['t_now_gps', 'x', 'y', 'z', 'vx', 'vy', 'vz']
    out.make_n_save(f'data_truth_{casename}', t_vec = t_gps_full_5ms, 
                    data = states_full_5ms, 
                    data_cols = columns_truth, 
                    subfolder = 'j2prop_testdata')
if plot:
    f, ax = plt.subplots()
    ax.plot(t_gps[:ii_end+ii_truth_0]-t_gps_full_5ms[0], s_host[:ii_end+ii_truth_0,0], label = 'raw_data', linewidth = 10, alpha = 0.2)

    ax.plot(t_gps_full_5ms-t_gps_full_5ms[0], states_full_5ms[:,0], label = '5 ms data for sim')

    ax.scatter(t_stamp_data-t_gps_full_5ms[0], s_data[:,0], label = 'data inputs', c = 'r', s = 10)
    ax.set_ylabel('x [m]')
    ax.set_xlabel('t since sim start [s]')
    ax.set_xlim([-30,50])
    ax.legend()
