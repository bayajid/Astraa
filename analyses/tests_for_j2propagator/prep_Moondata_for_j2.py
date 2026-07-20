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
import astronomy_tools.astro_targets as where_moon
import datetime as dt
# Create truth data to Moon
dt_raw = 1
truth_length = 1e3

t_now = dt.datetime(2023, 12, 12, 0, 0, 0)
t_gps_0 = t_conv.utc2gws(t_now, ls = 10)
moon_generator = where_moon.body_fromsp(t_gps_0, 'gps')
t_desired = np.arange(1,truth_length,dt_raw)

nrows = t_desired.shape[0]

r_host = np.zeros((nrows, 3))
v_host = np.zeros((nrows, 3))
t_gps = np.zeros(nrows)
for ii, dt_ii in enumerate(t_desired):
    r_moon_ii = moon_generator.get_sun(dt_ii, 'moon')
    r_host[ii,:] = r_moon_ii
    t_gps[ii] = t_gps_0 + dt_ii

test_case = 1
if test_case == 1:
    # data coming with present timestamps for interpolation
    # and spaced out ones for propagation

    test_name = '15 datapoints. 1Hz. 5x 1s in future. Then 5x present. Then 5x past'
    ii_truth_0 = 50

    ii_data_incoming_now = [ii for ii in range(1,16)]
    # [s] TIME OFF-sets used to determine if datapoitns are in past/future/present
    ii_data_timestamps = [2, 3, 4, 5, 6, 6, 7, 8, 9, 10, 10, 11, 12, 13, 14]
    # setup test_cases
    dt_truth_req = 5e-3
    dt_truth_length = 40
    add_errors = True
    error_r_3d = 0
    error_v_3d = 0
    error_r_3d_mean = 0
    error_v_3d_mean = 0                   
# generate full time-vector
ii_end = ii_truth_0 + int(dt_truth_length / dt_raw)
t_gps_end = t_gps[ii_truth_0] + dt_truth_length
t_gps_full = t_gps[ii_truth_0:ii_end]
t_gps_full_5ms = np.arange(t_gps[ii_truth_0], t_gps_end, dt_truth_req)
s_host = np.hstack((r_host, v_host))
print(f'Test case nr {test_case}\n:{test_name}. \nTruth length : {dt_truth_length} s')
ii_data_incoming_now = [ii + ii_truth_0 for ii in ii_data_incoming_now]
ii_data_timestamps = [ii + ii_truth_0 for ii in ii_data_timestamps]

states_full = s_host[ii_truth_0:ii_end, :]

state_interpolant = sp.interpolate.CubicSpline(t_gps_full.flatten(), states_full)
states_full_5ms = state_interpolant(t_gps_full_5ms)

casename = f'tests_j2_Moon{test_case}'

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

if save:
    columns_datainc = ['t_now_gps','t_stamp_gps','x', 'y', 'z', 'vx', 'vy', 'vz']
    out.make_n_save(f'tstamp_incoming_{casename}', t_vec = t_rec_data, 
                    data = np.hstack((t_stamp_data.reshape((len(t_stamp_data),1)), s_data)), 
                    data_cols = columns_datainc, 
                    main_folder = 'unit_tests',
                    subfolder = 'prop_data')
    
    columns_truth = ['t_now_gps', 'x', 'y', 'z', 'vx', 'vy', 'vz']
    out.make_n_save(f'data_truth_{casename}', t_vec = t_gps_full_5ms, 
                    data = states_full_5ms, 
                    data_cols = columns_truth, 
                    main_folder = 'unit_tests',
                    subfolder = 'prop_data')
if plot:
    f, ax = plt.subplots()
    ax.plot(t_gps[:ii_end+ii_truth_0]-t_gps_full_5ms[0], s_host[:ii_end+ii_truth_0,0], label = 'raw_data', linewidth = 10, alpha = 0.2)

    ax.plot(t_gps_full_5ms-t_gps_full_5ms[0], states_full_5ms[:,0], label = '5 ms data for sim')

    ax.scatter(t_stamp_data-t_gps_full_5ms[0], s_data[:,0], label = 'data inputs', c = 'r', s = 10)
    ax.set_ylabel('x [m]')
    ax.set_xlabel('t since sim start [s]')
    ax.set_xlim([-30,50])
    ax.legend()
