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
    csv_output_path = r'orbital_simulations/single_sat/leo_hp_prop_1s_1d'
else:
    # csv_output_path = r'orbital_simulations\single_sat\leo_j2_prop'
    csv_output_path = r'orbital_simulations/single_sat/leo_j2_prop_1s_1d'

fname_simparam = 'simulation_parameters.json'
fname_states = 'state_history.dat'

## MVP imports
import basic_tools.vector_operations as vec_calc
import astronomy_tools.constants as const
import attitude_tools.attitude_simulation as att_sim
import attitude_tools.conversions as att_conv
import attitude_tools.rotations as att_rot
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

plot_quaternions = 0

host_chosen = simulation_parameters['sat_names'][0]
# host_chosen = 'leo_target_polar'
# host_chosen = 'leo_host_incl'
# host_chosen = 'meo_target_incl'

t_j2000 = data_raw[:,0]
t_gps = t_j2000 + t_conv.dt_j2000tt2gps()
dt_raw = t_gps[1] - t_gps[0]

# data coming with present timestamps for interpolation
# and spaced out ones for propagation

test_name = '11 datapoints in 50s, 3x future 4 s in future, then 1s past 10m errors'
ii_truth_0 = 0
# setup test_cases
dt_truth_req = 0.025
dt_truth_length = 10
dt_timeupd_req = 0.25
t_delay = 0.05 # [s]
add_errors = True

## generate attitude
importlib.reload(att_sim)

r_host = data_raw[:,simulation_parameters['r_index'][host_chosen]]
v_host = data_raw[:,[ii+3 for ii in simulation_parameters['r_index'][host_chosen]]]
s_host = np.hstack((r_host, v_host))
print(f'Test case nr {test_case}\n:{test_name}. \nTruth length : {dt_truth_length} s')

# generate full time-vector
ii_end = ii_truth_0 + int(dt_truth_length / dt_raw)
t_gps_end = t_gps[ii_truth_0] + dt_truth_length
t_gps_full = t_gps[ii_truth_0:ii_end]
t_gps_full_fine = np.arange(t_gps[ii_truth_0], t_gps_end, dt_truth_req)

# [s] TIME OFF-sets used to determine if datapoitns are in past/future/present
ii_data_incoming_now = np.arange(0, len(t_gps_full_fine), int(dt_timeupd_req/dt_truth_req))
ii_data_timestamps = [ii + int(t_delay/dt_truth_req) for ii in ii_data_incoming_now]
# ii_data_timestamps = [ii + ii_truth_0 for ii in ii_data_timestamps]

states_full = s_host[ii_truth_0:ii_end, :]

state_interpolant = sp.interpolate.CubicSpline(t_gps_full.flatten(), states_full)
states_full_fine = state_interpolant(t_gps_full_fine)


q_all, q_dot_all, rot_eci2bf = att_sim.calc_quat_eci2bf(states_full_fine[:,[0, 1,2]], states_full_fine[:,[3,4,5]],
                                                         t_gps = t_gps_full_fine, 
                                                         att_profile = 'earth_point', 
                                                         euler_rates = None, 
                                                         roll_velocity = 0.84,
                                                         add_ideal_jerk=0,
                                                         add_realistic_jerk=0.16,                                                         
                                                calc_qdot = 1)

dq_array = np.zeros(q_all.shape)




## PROCESS QUATERNIONS TO NOT HAVE SIGN FLIPS (which dont affect rotation)
# but affect prediction qualities
flips = []
for ii, q_ii in enumerate(q_all[:-1]):
    dq = q_all[ii+1,:] - q_ii
    dq_array[ii,:] = np.abs(dq)
    if np.any(dq_array[ii,:] > 1):
        flips.append(ii+1)
for ii_flip in flips:
    q_all[ii_flip:,:] = -q_all[ii_flip:,:]
    q_dot_all[ii_flip:,:] = -q_dot_all[ii_flip:,:]
q_data_full = np.hstack((q_all, q_dot_all))

if plot_quaternions:
    f, axs = plt.subplots(nrows = 2)
    ax = axs[0]
    q_plotted = q_all
    for ii in range(4):
        ax.plot(q_plotted[:,ii])
    ax = axs[1]
    q_plotted = q_dot_all
    for ii in range(4):
        ax.plot(q_plotted[:,ii])



casename = f'tests_q_extrap'

# get sub-second datapoints
t_rec_data = np.round(t_gps_full_fine[ii_data_incoming_now],3)
# timestamp of the incoming data
t_stamp_data = np.round(t_gps_full_fine[ii_data_timestamps],3)
s_data = q_data_full[ii_data_timestamps,:]

add_errors = 1
if add_errors:
    # .5 mrad 3D
    err_ea = errgen.pos_err_gen(0, 3.0e-4, len(t_stamp_data)) # deg
    q_ea = np.array([att_conv.convert_ea2quat(err_ii, deg = 0) for err_ii in err_ea])


# ADD QUAT ERROR
for ii, q in enumerate(s_data[:,:4]):
    s_data[ii,:4] = att_rot.multiply_quat_hamiltonian(q_ea[ii,:],q).flatten()
# s_data[:,3:] = s_data[:,3:]



for ii, tstamp in enumerate(t_stamp_data):
    print(f'At t={t_rec_data[ii] - t_gps_full_fine[0]}, data coming with t stamp = {tstamp - t_gps_full_fine[0]};')

save = 1
plot = 1

# data_to_store = np.hstack((r_host_predicted, np.linalg.norm(dr, axis = 1).reshape(t_j2000.shape[0],1)))
# data_to_store[:,0] = t_gps

if plot:
    # plot quaternion difference from initial value
    f, ax = plt.subplots()
    q_err = np.zeros((s_data.shape[0], 4))
    for ii, q in enumerate(s_data[:-1,]):
        dq = s_data[ii+1,:4] - q[:4]
        q_err[ii,:] = dq
        theta_q1 = np.arccos(q[0])*2*1e6
        theta_q2 = np.arccos(s_data[ii+1,0])*2*1e6
        print(f'{ii} -> dtheta {theta_q2 - theta_q1}')
    for ii in range(4):
        ax.plot(t_rec_data-t_rec_data[0],q_err[:,ii], label = str(ii))

    f, ax = plt.subplots()
    for ii in range(1,2):
        ax.plot(t_rec_data-t_rec_data[0],s_data[:,ii], label = f'q{ii} data')
        ax.plot(t_gps_full_fine-t_gps_full_fine[0], q_data_full[:,ii], label = f'q{ii} true')
    ax.legend()
    ax.set_ylabel('Q scalar value')

    f,ax = plt.subplots()
    for ii in range(4):
        ax.plot(t_rec_data-t_rec_data[0],s_data[:,ii+4], label = f'q{ii} data')
    
    ax.legend()



if save:
    columns_datainc = ['t_now_gps','t_stamp_gps','q1', 'q2', 'q3', 'q4', 'qdot1', 'qdot2', 'qdot3', 'qdot4']
    out.make_n_save(f'tstamp_incoming_{casename}', t_vec = t_rec_data, 
                    data = np.hstack((t_stamp_data.reshape((len(t_stamp_data),1)), s_data)), 
                    data_cols = columns_datainc, 
                    main_folder = 'unit_tests',
                    subfolder = 'q_extrap')
    
    columns_truth = ['t_now_gps','q1', 'q2', 'q3', 'q4', 'qdot1', 'qdot2', 'qdot3', 'qdot4']
    out.make_n_save(f'data_truth_{casename}', t_vec = t_gps_full_fine, 
                    data = q_data_full, 
                    data_cols = columns_truth, 
                    main_folder = 'unit_tests',
                    subfolder = 'q_extrap')
