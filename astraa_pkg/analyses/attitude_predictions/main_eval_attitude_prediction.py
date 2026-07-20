#%% 2023-05-15
# CLEANUP of attitude prediction

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import importlib
import os, sys
sys.path.insert(1, os.getcwd()[:os.getcwd().index('astropynaric')+13])

from basic_tools.vector_operations import calc_dot_angle
import attitude_tools.rotations as rot
import attitude_tools.conversions as conv
import analyses.attitude_predictions.attitude_prediction_utlities as attutil
import analyses.attitude_predictions.prediction_methods as attpred
import analyses.attitude_predictions.attitutde_plot_functions as attplt
importlib.reload(conv)
importlib.reload(rot)
importlib.reload(attutil)
importlib.reload(attpred)
importlib.reload(attplt)
np.set_printoptions(4)

path_cwd = os.getcwd()
path_outputs = fr'{path_cwd}/attitude_tools/attitude_predictions/outputs'
path_inputs = fr'{path_cwd}/attitude_tools/attitude_predictions/generated_attitude'

#%% SETTINGS and loaing true attitude data

total_pred_length = 10 # s

msg_update_rates = [10, 5, 1] # Hz
# msg_update_rates = [5]

# vector rotated with the true and predicted quaternions
ref_pt_vec = np.array([3000e3, 10e3, -50e3])

## Saving conditionals
# pointign error outputs from predictions
save_pe = 1
# true and est quaternions (not predicted)
save_quat = 0

## Plotting conditionals

plot_pe_due_to_eaerror = 0
# Run PE plotting code
make_pe_plots = 0
# Plotting true angles
plot_true_angles = 0
# Euler angle noise (est - true)
plot_noise = 0
# True euler angles and rates
plot_true_angle_rate = 0
# est - true quaternion
plot_true_vs_est_quat = 0
# Overlay of predicted quaternions, quaternion message and true quaternion
plot_q_msg_pred = 0
# quaternion prediction error (q_pred - q_true) and PE
plot_qprederr_pe = 1


### CHOSE INTERPOLATIONE/EXTRAPOALTION METHOD
interpolation_ind = 1
# attitude_ind = 2 # nojerk
# attitude_ind = 1 # jerk
attitude_ind = 0 # cust propagated

# Prediction Options
interpolation_settings = ['quadratic_interp',# 0
'nothing',
'linear_interp',
'landsat_slerp',
'ideal', # 4 - for customer only. Ideal vs noise attitude (no prediction)
'euler_prop']

### Settings for true attitude
attitude_scenario_settings = ['customer_may11', # 0 attitude simulated from given rates. Est and True setting = 'customer_may5' # attitude simulated from given rates.# euler angles and rates available (but rates are identical)
                    'jerk_noisy', # 1 - 084+swap + NOISE added
                    'nojerk_noisy',] # 2 - 084+swap + NOISE added

attitude_setting = attitude_scenario_settings[attitude_ind]
prediction_setting = interpolation_settings[interpolation_ind]

print(f'''----CHOSEN SETTINGS----
Attitude setting : {attitude_setting}
Attitude prediction : {prediction_setting}
''')

### LOADING DATA, COMPUTING QUATERNIONS true and estimated. No predictions yet
if attitude_setting == 'customer_may11':
    fname_est = 'est_attitude_cust.csv'
    fname_true  = 'true_attitude_cust.csv'
    t_pred_start = None
elif attitude_setting == 'jerk_noisy':
    fname_est = 'est_attitude_jerk_noisy.csv'
    fname_true  = 'true_attitude_jerk_noisy.csv'
    t_pred_start = 40
elif attitude_setting == 'nojerk_noisy':
    fname_est = 'est_attitude_nojerk_noisy.csv'
    fname_true  = 'true_attitude_nojerk_noisy.csv'
    t_pred_start = 40
# get path
folder_inputs = r'attitude_tools\attitude_predictions\outputs'
path_est = f'{folder_inputs}\{fname_est}'
path_true = f'{folder_inputs}\{fname_true}'

t_vec, ea_true, ea_dot_true, ea_est, ea_dot_est = attutil.load_attitude_angles(path_est, path_true)
# prediction start time for customer data
if type(t_pred_start) == type(None):
    # customer data, setup t_pred_start when ea_rate = max (around 530 seconds)
    t_pred_start = int(t_vec[np.where(np.abs(ea_dot_est[:,1]) == np.max(np.abs(ea_dot_est[:,1])))[0][0]])

# Check PE due to Attitude Knowledge error, no prediction yet 
if plot_pe_due_to_eaerror:
    pe_fromea = attutil.eval_atterr_pe(t_vec, ea_est, ea_true, ref_pt_vec)
    f, ax = attplt.plot_pe(t_vec, pe_fromea, t_pred_start, attitude_setting + ' EA est vs true')

# get quaternions and angular velocities
q_ham_true, q_dot_ham_true, om_all_true = attutil.get_quat(ea_true, ea_dot_true)
q_ham_est, q_dot_ham_est, om_all_est = attutil.get_quat(ea_est, ea_dot_est)

# Plot true and estimated quaternion difference
if plot_true_vs_est_quat:
    f, ax = attplt.plot_dq(t_vec, t_pred_start, q_ham_true, q_ham_est, attitude_setting)
if plot_true_angle_rate:
    f, ax = attplt.plot_ea(t_vec, t_pred_start, q_ham_true, q_ham_est, attitude_setting)
## saving true and estimated quaternions (optional)
if save_quat: 
    path_quat_est = fr'{path_outputs}/quat_est_{attitude_setting}.csv'
    quat_outputs_df = pd.DataFrame(data = np.hstack((t_vec.reshape((t_vec.shape[0],1)), q_ham_est,q_dot_ham_est)) , columns = ['t_s', 'q1', 'q2', 'q3', 'q4', 'qdot1', 'qdot2', 'qdot3', 'qdot4'])
    quat_outputs_df.to_csv(path_quat_est, index = False)

    path_quat_true = fr'{path_outputs}/quat_true_{attitude_setting}.csv'
    quat_outputs_df = pd.DataFrame(data = np.hstack((t_vec.reshape((t_vec.shape[0],1)), q_ham_true,q_dot_ham_true)) , columns = ['t_s', 'q1', 'q2', 'q3', 'q4', 'qdot1', 'qdot2', 'qdot3', 'qdot4'])
    quat_outputs_df.to_csv(path_quat_true, index = False)
    print(f'Saved Quat to\n{path_quat_est}\n{path_quat_true}')

if plot_true_angles:  # PLOT true angles and NOISE (optional)
    f, axs = attplt.plot_ea_earates(t_vec, ea_true, ea_dot_true, EA_unit = 'deg', fname = f'{attitude_setting} true EA/EA_dot')
    # add t pred start
    [ax.plot([t_pred_start, t_pred_start], list(ax.get_ylim()), c = 'r', label = 't_pred_start') for ax in axs]
    f, axs = attplt.plot_ea_earates(t_vec, ea_est, ea_dot_est, EA_unit = 'deg', fname = f'{attitude_setting} est EA/EA_dot')
    [ax.plot([t_pred_start, t_pred_start], list(ax.get_ylim()), c = 'r', label = 't_pred_start') for ax in axs]
if plot_noise: # PLOT Euler angle noise (optional)
    ii_0 = [ii for ii, t in enumerate(t_vec) if t > t_pred_start][0] - 20
    ii_lim = ii_0 + 40
    f, ax = attplt.plot_ea(t_vec, ea_est- ea_true, ii_0 = ii_0, ii_lim = ii_lim,
                            fname = f'{attitude_setting} - Euler Angle Noise')
    ax.plot([t_pred_start, t_pred_start], list(ax.get_ylim()), c = 'r', label = 't_pred_start')
#%% Predict and interpolate quaternions

nrows = len(t_vec)
t_step_prop = t_vec[1] - t_vec[0]
ndigits = int(1/t_step_prop)
for kk, update_rate in enumerate(msg_update_rates):
    # get attitude message
    q_message, q_dot_message = attutil.slice_att_msg(t_vec, t_pred_start, update_rate,
                                                q_ham_est, q_dot_ham_est)
    print(f'''Attitude message time-stamps:
    QUATERNION MESSAGe update at {update_rate} Hz -> {1/update_rate} s between messages

    total time interval of message : {len(q_message)}
    start : {q_message[0,0]} s
    end : {q_message[1,0]} s

    all eph time-stamps : {q_message[:,0]} s
    ''')
    print(f'Quaternion message \n[t, q1, q2, q3, q4]\n{q_message[0,:]}\n{q_message[1,:]}')
    # get prediction vector
    t_vec_prediction = np.arange(q_message[-1,0], q_message[-1,0]+total_pred_length+t_step_prop, t_step_prop)    
    t_vec_prediction = np.round(t_vec_prediction, ndigits)
    
    ### PREDICTION METHODS

    # Prediction method format - In : msg, t_vec_prediction. Out - q_predicted
    if prediction_setting == 'linear_interp':
        # TODO implement linear interpolation
        q_predicted = attpred.linear_extrap(q_message, q_dot_message, t_vec_prediction)
    elif prediction_setting == 'quadratic_interp':        
        q_predicted = attpred.quadratic_extrap(q_message, q_dot_message, t_vec_prediction)
    elif prediction_setting == 'nothing':
        q_predicted = attpred.constant_prediction(q_message, q_dot_message, t_vec_prediction)
    elif prediction_setting == 'ideal':
        # predicted quaternion is the Estimated hamiltonian quaternion
        q_predicted = attpred.ideal_prediction(q_ham_est, t_vec_prediction, t_vec)
    elif prediction_setting == 'landsat_slerp':    
        # TODO Bayajid's implemented Landsat attitude prediction
        q_predicted = attpred.interpolate_LANDSAT(q_message, q_dot_message, t_vec_prediction)
    elif prediction_setting == 'propagate_dumbway':
        q_predicted = attpred.euler_propagation(q_message, q_dot_message, t_vec_prediction)

    ### Get added PE due to prediction

    pe_pred, quat_true_sliced = attutil.eval_prediction_pe(t_vec, t_vec_prediction,
                                                           q_predicted, q_ham_true, ref_pt_vec)
    # Saving Pointing Errors
    if save_pe:
        nrows_pred = len(pe_pred)
        # prep output csv|
        columns = ['t_s', 'pe_urad']
        columns.extend([f'qtrue_{ii}' for ii in range(4)])    
        columns.extend([f'qpred_{ii}' for ii in range(4)])
        output_data = np.hstack((t_vec_prediction.reshape((nrows_pred,1)), pe_pred, quat_true_sliced[:nrows_pred,:], q_predicted[:,1:]))
        pe_df = pd.DataFrame(columns = columns, data = output_data)
        save_path = f'{path_outputs}\{attitude_setting}_{prediction_setting}_{update_rate}Hz.csv'
        pe_df.to_csv(save_path, index = False)
        print(f'PE saved to as {attitude_setting}_{prediction_setting}_{update_rate}Hz.csv')
    # Plot quaternion error and found PE
    if plot_qprederr_pe:
        f, axs = attplt.plot_dq_pe(t_vec_prediction,
                                   quat_true_sliced, q_predicted, pe_pred,
                                        attitude_setting,
                                        update_rate                               
                                        )
# run PE plotting code
if make_pe_plots:
    import attitude_tools.attitude_predictions.pe_att_extrap_xmsg_plot
    importlib.reload(attitude_tools.attitude_predictions.pe_att_extrap_xmsg_plot)
    plt.show()

# Plot true quaternion, predicted quaternion and attitude message

if plot_q_msg_pred:
    f, axs = attplt.plot_q_msg_pred(t_vec, q_ham_true,
                                    q_predicted,
                                    q_message,
                                    attitude_setting                                    
                                    )
    
print(f'\n      Finito')