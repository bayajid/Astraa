#%% Functi ont ogenerate some crazy spacecraft attitude
# with angular velocities and rates. Several possible settings

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os 
import datetime
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import pandas as pd
import scipy.io
import importlib
import os, sys
sys.path.insert(1, os.getcwd()[:os.getcwd().index('astropynaric')+13])

import attitude_tools.rotations as rot
import attitude_tools.conversions as conv
import analyses.attitude_predictions.attitutde_plot_functions as attplt
import plotting_tools.basic_plotting as bplt


# path jazz
path_cwd = os.getcwd()
path_cust_data = fr'{path_cwd}/analyses/attitude_predictions/cust_data'
path_outputs = fr'{path_cwd}/analyses/attitude_predictions/outputs'

t_end_ownsim = 60 # [s] Simualte only for 60 seconds if using own inputs (sim_option != cust)

add_customer_noise = 1 # adding Euler angle noise (APE+AKE) from customer data
simulate_attitude = 1
save_outputs = 1
use_deg = 1 # conditional if deg are used

# sim_option = 'nojerk_noisy' # own attitude simulation + customer noise + jerk at t=40s
sim_option = 'jerk_noisy' # own attitude simulation + customer noise + jerk at t=40s
# sim_option = 'cust' # Only CUSTOMER angular rates used for simulation + noise. Dont limit t_end


plot_simulated_att = 1
plot_simulated_vsnoise = 1

plot_inputs = 1
plot_simstuff = 0 # simulated RPY
plot_single = 0
plot_diff = 0
plot_omegas = 0
plot_ea_numrates = 0 # dofferentiate input euler angles

# Load customer data - euler angle (noise) + estimated euler rates
fname = 'BusRateEst_10Hz.mat'
fname_atterr = 'BusAttEst_10Hz.mat'
full_att_path = f'{path_cust_data}/{fname_atterr}'
full_rate_path = f'{path_cust_data}/{fname}'
data_mat = scipy.io.loadmat(full_rate_path)
data_array = data_mat[fname[:-4]][0][0][0]
t_vec_loaded = data_array[:,0]
ea_rates_cust = data_array[:,1:] # Estimated euler rates from customer
data_mat_ea =  scipy.io.loadmat(full_att_path)
data_array = data_mat_ea[fname_atterr[:-4]][0][0][0]

ea_input = data_array[:,1:]
t_vec_loaded = data_array[:,0]

t_noise_profile_start = 530
t_0 = 0
t_step = t_vec_loaded[1] - t_vec_loaded[0]
t_vec_loaded = np.round(t_vec_loaded, int(1/t_step))

if sim_option == 'cust':    
    t_end = t_vec_loaded[-1]
    t_pred_start = t_noise_profile_start # used to shift noise profile to match prediction start time
else:
    t_pred_start = 40
    t_end = t_end_ownsim 
print(f'Simulation time : from {t_0} to {t_end} s')
t_vec_prop = np.round(np.arange(t_0, t_end+t_step, t_step), int(1/t_step))
ii_slicing = [ii for ii, t in enumerate(t_vec_loaded) if t in t_vec_prop]
t_vec_sliced = t_vec_loaded[ii_slicing]

## shift added APE+AKE to match at t_pred_start
# first index of time vector
dt_start = t_noise_profile_start - t_pred_start
ii_noise_start = [ii for ii, t in enumerate(t_vec_loaded) if t>=dt_start][0] 
ii_noise_sliced = [ii + ii_noise_start for ii in ii_slicing]

# Optional input plots
importlib.reload(attplt)
if plot_ea_numrates:
    f, ax = attplt.plot_ea_gradient(t_vec_sliced, ea_input)
if plot_inputs:
    ii_lim = 100
    f, ax = attplt.plot_ea_earates(t_vec_sliced, ea_input, ea_rates_cust, fname = 'EA and rates, customer', ii_lim = ii_lim, savefig = 0)

#%% Simulate attitude and measurements
nrows = len(t_vec_prop)
# Initial Euler angles
att_ea_0 = np.array([0, 5, 0]) # Roll; pitch; yaw angles [deg]
# Angular velocitis for own attitude simulation:
# rotate at max velocities and rates in X, Y, Z
# angular velocity reaches 0.84 deg/s at t=40
om_1 = 0.34
om_2 = 0.34
om_3 = 0.34

# Angular rates about X/Y/Z [deg/s^2]
om_dot_1 = 0.012
om_dot_2 = 0.012
om_dot_3 = 0.012

if sim_option == 'jerk_noisy':
    omega_dot_swapped_vec = np.array([-om_dot_1, -om_dot_2, -om_dot_3])
elif sim_option == 'nojerk_noisy':
    omega_dot_swapped_vec = np.array([om_dot_1, om_dot_2, om_dot_3])

if simulate_attitude:
    np.set_printoptions(precision = 3)
    print(f'-----SIMULATING estimated SC attitude-----')
    print(f'Simulation: {sim_option.upper()}')
    print(f'''                  INPUTS
    initial roll, pitch, yaw : {att_ea_0} [deg] 
    --------------------------------------------------
    Calculating attitude kinematics for {t_end} s, with {t_step} steps
    --------------------------------------------------
    ''')
    # create placeholders for arrays
    ea_all = np.zeros((nrows, 3))
    ea_dot_all = np.zeros((nrows, 3))
    om_all = np.zeros((nrows, 3))
    # initial values
    om_vec_initial = np.array([om_1, om_2, om_3])
    om_dot_initial = np.array([om_dot_1, om_dot_2, om_dot_3])
    
    if sim_option == 'cust':
        ea_dot_all = ea_rates_cust 
        # use euler rates to propagate euler angle
        for ii, t_ii in enumerate(t_vec_prop):
            if ii == 0:
                ## initialize
                ea_0 = att_ea_0
            # calc true attitude
            ea_ii = ea_0 + t_step * ea_dot_all[ii,:]
            # store
            ea_all[ii,:] = ea_0
            # update
            ea_0 = ea_ii
    else:
        # calulate angular velocity vector, compute euler rates and propagate euler angles
        for ii, t_ii in enumerate(t_vec_prop):
            if ii == 0:
                ## initialize
                om_0 = om_vec_initial                
                ea_0 = att_ea_0

            if t_ii >= 40 and 'jerk' in sim_option:
                omega_dot_vec_used = omega_dot_swapped_vec
                print(f't = {t_ii} We swapping accelerations to {omega_dot_vec_used} deg/s^2')
            else:
                omega_dot_vec_used = om_dot_initial
            # calc angular velocities
            om_ii = om_0 + t_step * omega_dot_vec_used # [deg/s]
            # calc euler angle rates
            ea_rate = conv.calc_ea_dot(ea_0, om_ii, use_deg)
            # calc true attitude
            ea_ii = ea_0 + t_step * ea_rate
            # store
            ea_all[ii,:] = ea_0
            ea_dot_all[ii,:] = ea_rate
            om_all[ii,:] = om_0

            # update
            ea_0 = ea_ii%360
            om_0 = om_ii         

    print(f'''                  DONE - debug purpose value checks
    First roll, pitch, yaw : {att_ea_0} [deg] 
    Final roll, pitch, yaw : {ea_ii} [deg] 
    --------------------------------------------------
    Calculated attitude kinematics for {t_end} s, with {t_step} steps
    --------------------------------------------------
    ''')        
    if save_outputs:
        print('Saving')
        ## Save
        data_true = np.hstack((t_vec_prop.reshape((nrows, 1)), ea_all, ea_dot_all))
        data_saved = [data_true]
        
        if add_customer_noise:
            print(f'Adding customer data noise! STD = {np.std(ea_input, axis = 0)} mdeg')
            ea_est = ea_all + ea_input[ii_noise_sliced,:]/1e3
            data_est = np.hstack((t_vec_prop.reshape((nrows, 1)), ea_est, ea_dot_all))
            data_saved.append(data_est)
        for ii, data_used in enumerate(data_saved):
            data_type = ['true', 'est'][ii]
            output_df = pd.DataFrame(
                columns = ['t_s', 'roll_deg', 'pitch_deg', 'yaw_deg', 'rollrate_degs', 'pitchdate_degs', 'yawrate_degs'],
                data = data_used)
            output_path = fr'{path_outputs}/{data_type}_attitude_{sim_option}.csv'
            output_df.to_csv(output_path, index=False)
            print(f'Saved {data_type} attitude to \n{output_path}')        
    else:
        print('Outputs not saved')
## PLOTS of outputs
if plot_simulated_att:
    f, ax = attplt.plot_sim_ea_noisysea(t_vec_prop, ea_all, ea_est, ii_lim = 6, fname = sim_option, savefig = 1)
    f, ax = attplt.plot_ea_earates(t_vec_prop, ea_est, om_all, fname = sim_option + ' simulated. !Showing omega!', EA_unit = 'deg')
    f, ax = attplt.plot_ea_earates(t_vec_prop, ea_est, ea_dot_all, fname = sim_option + ' simulated. !Showing EA_dot!', EA_unit = 'deg')
if plot_simulated_vsnoise:
    if sim_option == 'cust':
        ii_0 = 5310
        ii_f = 5410
    else:
        ii_0 = 400
        ii_f = 500

    f, ax = attplt.plot_ea(t_vec_prop, ea_all - ea_est, ii_0 = ii_0, ii_lim = ii_f, fname = 'EA Noise from pred start' + sim_option, savefig = 1)