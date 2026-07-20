#%% 2023-03-31 
# revisit attitude predictions due to potential errors in original 2023-Jan analysis
# issues involving 1) euler angle inputs into scipy.r
# 2) Lack of quaternion normalization
# 3) Quaternion rotation order of q V q_conj or q_conj V q
## https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.as_euler.html

import numpy as np
import matplotlib.pyplot as plt
import pathlib
import pandas as pd
import os
import importlib
import sys
import os, sys
sys.path.insert(1, os.getcwd()[:os.getcwd().index('astropynaric')+13])
from basic_tools.vector_operations import calc_dot_angle
import attitude_tools.rotations as rot
import attitude_tools.conversions as conv
import importlib
importlib.reload(conv)
np.set_printoptions(4)

path_cwd = os.getcwd()
path_outputs = fr'{path_cwd}/attitude_tools/outputs'
path_inputs = fr'{path_cwd}/attitude_tools/generated_attitude'

#%% SETTINGS and loaing true attitude data
importlib.reload(rot)
# saving conditionals
do_all_update_rates = 0 # try all Attitude message update rates or just 1 Hz
print_all_attitude_messages = 1 # if 0, will print only 1 attitude message (test purp.)
## DEBUG OPTIONS - keep all at 0 (off)
skip_predicting = 0 # use true quaternion for PE calculation (debug, test rotation)
swap_euler_angle_input = 0 # Keep 3-2-1 convention but input Yaw-Pitch-Roll to sp.r

# normalizign quaternion
make_predicted_quaternion_norm = 1
# How quaternion rate is calculated. All must be equivalent. 0 most straightforward

total_pred_length = 100 # s
save_pe = 1
# plotting
show_true_angles = 1
show_true_angle_rate = 0
show_quat_true = 1
show_quatrate_true = 1

try_not_predicting = 0 # evaluate PE if no prediction is made (lol)
### CHOSE INTERPOLATIONE/EXTRAPOALTION METHOD
interpolation_ind = 0
attitude_ind = 5

# Prediction Options
interpolation_settings = ['quadratic_interp_2pts_smarter',# 0
'linear_interp',  # 1
'landsat_slerp', # 2
'nothing', # 3
'est_vs_true', # 4 - for customer only. Ideal vs noise attitude (no prediction)
'propagate_cleanway',
'propagate_dumbway']
quat_rate_options = ['EA->Q+w->Qdot', 'ea;ea_dot->DCM, DCM_dot->Qdot']
### Settings for true attitude
attitude_scenario_settings = ['rotate_all_pred084', 
                     'rotate_azel_octhw467',
                    'rotate_all_pred084_swap', # switch acceleration sign at t=40 when w=0.82 deg/s
                    'customer_may11', # 3 attitude simulated from given rates. Est and True setting = 'customer_may5' # attitude simulated from given rates.# euler angles and rates available (but rates are identical)
                    'attitude_jerk_noisy', # 4/5 - 084+swap + NOISE added
                    'attitude_nojerk_noisy', # 4/5 - 084+swap + NOISE added
                    'real_stable_sat'] # real attitude tracking data in 1s steps for a stable satellite

attitude_setting = attitude_scenario_settings[attitude_ind]
prediction_setting = interpolation_settings[interpolation_ind]

print(f'''----CHOSEN SETTINGS----
1) Euler angle input swap from original : {bool(swap_euler_angle_input)}
3) Quaternion normalization : {bool(make_predicted_quaternion_norm)}
Attitude setting : {attitude_setting}
Attitude prediction : {prediction_setting}
''')

rotating_scenario_filter = 'rotate' # used to separate rotating simulated attitude
# from stable satellite case
### LOADING DATA
if rotating_scenario_filter in attitude_setting:
    fname = attitude_setting
    if 'delayed' in attitude_setting:
        fname = 'rotate_all_axes'
    att_true_df = pd.read_csv(fr'{path_inputs}/true_attitude_{fname}.csv')
    print(f'Loaded true attitude for setting : {attitude_setting}')
    att_all = att_true_df.iloc[:,[1,2,3]].values # deg
    om_all = att_true_df.iloc[:,[-3, -2, -1]].values # deg/s
    att_rates_all = att_true_df.iloc[:,[4,5,6]].values
    t_vec = np.round(att_true_df.iloc[:,0], 2)
    # placeholder for quaternions
    quat_ham = np.zeros((nrows, 5)) # t ; quat
    quat_dot_ham = np.zeros((nrows, 5)) # t ; quat_dot

## Load true Euler Angle data and convert to quaternions/quaternion rates\ [SIMULATED DATA]
if rotating_scenario_filter in attitude_setting: 
    for ii, t_ii in enumerate(t_vec):
        # intrinsic rotations used  
        om_ii = om_all[ii,:]
        om_ii_rad = np.deg2rad(om_ii)
        w_1, w_2, w_3 = om_ii_rad[0], om_ii_rad[1], om_ii_rad[2]
        if swap_euler_angle_input: 
            ea_input = np.flip(att_all[ii,:]) 
        else:
            ea_input = att_all[ii,:]
        
        q, q_dot = conv.calc_qdot(ea_input, om_ii, deg = 1)
        quat_ham[ii,0] = t_ii
        quat_ham[ii,1:] = q.flatten()
        quat_dot_ham[ii,0] = t_ii
        quat_dot_ham[ii,1:] = q_dot.flatten()
    quat_outputs_df = pd.DataFrame(data = np.hstack((quat_ham,quat_dot_ham[:,1:])) , columns = ['t_s', 'q1', 'q2', 'q3', 'q4', 'qdot1', 'qdot2', 'qdot3', 'qdot4'])
    quat_outputs_df.to_csv(fr'{path_outputs}/quat_true_{attitude_setting}.csv', index = False)
    
    if attitude_setting == "rotate_all_axes" or attitude_setting == "rotate_swap":
        t_pred_start = 40
    elif attitude_setting == "rotate_all_delayed":
        t_pred_start = 79
    elif 'rotate' in attitude_setting:
        t_pred_start = 40
    t_step_prop = 0.01 # t-step used in true attitude generation
elif attitude_setting == 'customer_may11' or 'noisy' in attitude_setting:
    # load, calc q_true -> truth
    # q_est -> pred -> compare to true
    att_true = pd.read_csv(fr'{path_inputs}/true_attitude_10Hz.csv')
    att_est = pd.read_csv(fr'{path_inputs}/est_attitude_10Hz.csv')
    
    ea_true = att_true.iloc[:,[1,2,3]].values
    ea_dot_true = att_true.iloc[:,[4,5,6]].values
    t_vec = att_true.iloc[:,0]
    ea_est = att_est.iloc[:,[1,2,3]].values
    ea_dot_est = att_est.iloc[:,[4,5,6]].values

    quat_ham = np.zeros((ea_est.shape[0], 5)) # est
    quat_dot_ham = np.zeros((ea_est.shape[0], 5)) # est
    quat_ham_true = np.zeros((ea_est.shape[0], 5)) # TRU
    quat_dot_ham_true = np.zeros((ea_est.shape[0], 5)) # TRU
    for ii, t_ii in enumerate(t_vec.values):
        # get true and estimated omega
        omega_true = conv.calc_omega(ea_true[ii,:], ea_dot_true[ii,:])
        omega_est = conv.calc_omega(ea_est[ii,:], ea_dot_est[ii,:])
        
        # est
        q, q_dot = conv.calc_qdot(ea_est[ii,:], omega_est, deg =1)
        quat_ham[ii,0] = t_ii
        quat_ham[ii,1:] = q.flatten()
        quat_dot_ham[ii,0] = t_ii
        quat_dot_ham[ii,1:] = q_dot.flatten()
        # true
        q, q_dot = conv.calc_qdot(ea_true[ii,:], omega_true, deg =1)
        quat_ham_true[ii,0] = t_ii
        quat_ham_true[ii,1:] = q.flatten()
        quat_dot_ham_true[ii,0] = t_ii
        quat_dot_ham_true[ii,1:] = q_dot.flatten()
    
    
    #   ORIGINAL CUSTOMER T_PRED_START
    # t_pred_start = int(t_vec.values[np.where(np.abs(ea_dot_est[:,1]) == np.max(np.abs(ea_dot_est[:,1])))[0][0]]+1)
    # SWAPPED TO 1 s earlier
    t_pred_start = int(t_vec.values[np.where(np.abs(ea_dot_est[:,1]) == np.max(np.abs(ea_dot_est[:,1])))[0][0]])
    if 'noisy' in attitude_setting:
        t_pred_start = 40

    t_pred_start = t_pred_start - 1
    t_step_prop = t_vec.values[1] - t_vec.values[0]
    quat_outputs_df = pd.DataFrame(data = np.hstack((quat_ham,quat_dot_ham[:,1:])) , columns = ['t_s', 'q1', 'q2', 'q3', 'q4', 'qdot1', 'qdot2', 'qdot3', 'qdot4'])
    quat_outputs_df.to_csv(fr'{path_outputs}/quat_est_{attitude_setting}.csv', index = False)
    
    quat_outputs_df = pd.DataFrame(data = np.hstack((quat_ham_true,quat_dot_ham_true[:,1:])) , columns = ['t_s', 'q1', 'q2', 'q3', 'q4', 'qdot1', 'qdot2', 'qdot3', 'qdot4'])
    quat_outputs_df.to_csv(fr'{path_outputs}/quat_true_{attitude_setting}.csv', index = False)

    # set up req variables
    att_all = ea_est

elif 'real' in attitude_setting:
    # reqs - t_vector starts from 0
    # data is at 1s rates. Can onloy get finer with interpolation. Just use 1s steps for testing purposes
    nrows_loaded = 1000
    dt_data = 1 # interval between given dataset
    fname = r'mystery_realsat_quaternions.csv'
    quat_df = pd.read_csv(fr'{path_outputs}/{fname}')
    quat_ham = quat_df.iloc[:nrows_loaded,[1,2,3,4]].values
    quat_dot_ham = quat_df.iloc[:nrows_loaded,[5,6,7,8]].values
    nrows = quat_dot_ham.shape[0]
    t_vec = np.round(np.arange(0, nrows, dt_data),1)
    ## append time-vector to quat, quat_dots
    quat_ham = np.hstack((t_vec.reshape((nrows, 1)), quat_ham))
    quat_dot_ham = np.hstack((t_vec.reshape((nrows, 1)), quat_dot_ham))

    t_pred_start = 100 # arbitrarily chosen start-time
    t_step_prop = dt_data

if rotating_scenario_filter in attitude_setting: 
    if show_true_angles:
        f, axs = plt.subplots(nrows = 3, ncols = 2, figsize = (12,8))
        f.suptitle(f'True Euler Angles, rates - {attitude_setting}')
        for ii, ax in enumerate(axs[:,0]): # angles
            ax.set_ylabel(['Roll', 'Pitch', 'Yaw'][ii] + ' [deg]', fontweight = 'bold')
            ax.plot(t_vec, att_all[:,ii], label = 'True')
            ax.grid()
            ax.legend()
            ax.plot([t_pred_start, t_pred_start], list(ax.get_ylim()), label = 'Prediction Start', c = 'r')
            ax.set_xlim([35, 45])
        ax.set_xlabel('t [s]', fontweight = 'bold')
        for ii, ax in enumerate(axs[:,1]): # rates
            ax.set_ylabel('Angular rate [deg/s]', fontweight = 'bold')
            ax.plot(t_vec, att_rates_all[:,ii], label = ['Roll', 'Pitch', 'Yaw'][ii] + ' True')
            ax.plot(t_vec, om_all[:,ii], label = 'w_'+['x', 'y', 'z'][ii] + ' True')
            ax.grid()
            ax.legend()
            ax.plot([t_pred_start, t_pred_start], list(ax.get_ylim()), label = 'Prediction Start', c = 'r')
            ax.set_xlim([35, 45])
        ax.set_xlabel('t [s]', fontweight = 'bold')

#%% Predict and interpolate quaternions
# prediction/message settings
nrows = len(t_vec)

if do_all_update_rates:
    quat_update_rates = [10, 4, 1] # Hz
else:
    quat_update_rates = [10, 5, 1]
if 'propagate' in prediction_setting:
    quat_update_rates = [0.1, 0.5, 1, 2] 
    if 'real' in attitude_setting:
        quat_update_rates = [1]
# Refrence pointing vector to be rotated with true/predicted quaternions
# for calculating the resulting pointing error

## Comment line below to use a single quaternion update rate
# quat_update_rates = [4, 2] # Hz

ref_pt_vec = np.array([100, 10, -10])
nr_eph = 3 # default nr eph points (DONT TOUCH! 2 point message cases handled later)
t0 = 0 ## REFERENCE START TIME. Can be used to start eph message later in the attitude time-series
if 0:
    print(f'''
    WE PREDICTING QUATERNIONS BOYS
    Method - {prediction_setting}
    Update rates tried : {quat_update_rates} Hz
    ''')
for kk, quat_update_rate in enumerate(quat_update_rates):
    t_eph_interval = 1 / quat_update_rate # interval between eph updates TODO might have some issues with floats
    eph_length = t_eph_interval*(nr_eph-1)

    t_eph_end = t_pred_start
    t_eph_start = t_eph_end - eph_length
    t_vec_eph = [t_eph_start, np.round(t_eph_start+t_eph_interval,3), t_eph_end]
    if 'real' in attitude_setting:
        t_vec_prediction = np.arange(t_eph_end, t_vec[-100]+t_step_prop, t_step_prop)
    else:
        t_vec_prediction = np.arange(t_eph_end, t_eph_end+total_pred_length+t_step_prop, t_step_prop)
    ### SLICING - ATTITUDE MESSAGE - PREDICTION VECTOR
    # Make slice to get quaternion ephemeris w/ time-stamps
    t_vec_msg = np.round(t_vec.values, int(1/t_step_prop))
    ii_message = np.array([kk for kk in range(nrows) if t_vec_msg[kk] in t_vec_eph])
    # quat_message = np.array([quat_row for quat_row in quat_ham_outputs if quat_row[0] in t_vec_eph])
    quat_message = quat_ham[ii_message,:]
    quat_dot_message = quat_dot_ham[ii_message, :]
    if attitude_setting == 'propagate_cleanway':
        om_message = om_all[ii_message,:] # [deg/s] 

    if kk == 0 or print_all_attitude_messages:
        print(f'''Attitude message time-stamps:
        QUATERNION MESSAGe update at {quat_update_rate} Hz -> {t_eph_interval} s between messages
        nr att points in message : {nr_eph}

        total time interval of message : {eph_length}
        start : {t_eph_start} s
        end : {t_eph_end} s

        all eph time-stamps : {t_vec_eph} s
        total prediction time vector: {t_vec_prediction} s
        ''')
        print(f'Quaternion message \n[t, q1, q2, q3, q4]\n{quat_message[0,:]}\n{quat_message[1,:]}\n{quat_message[2,:]}')
    t_vec_prediction = np.round(t_vec_prediction, 3)

    if prediction_setting == 'linear_interp':
        extrap_kind_used = 'linear'
        print(f'Implement manual linear polynomial fit')
        q0_interpolator = interpolate.interp1d(quat_message[1:,0], quat_message[1:,1],kind = extrap_kind_used, fill_value='extrapolate')
        q1_interpolator = interpolate.interp1d(quat_message[1:,0], quat_message[1:,2],kind = extrap_kind_used, fill_value='extrapolate')
        q2_interpolator = interpolate.interp1d(quat_message[1:,0], quat_message[1:,3],kind = extrap_kind_used, fill_value='extrapolate')
        q4_interpolator = interpolate.interp1d(quat_message[1:,0], quat_message[1:,4],kind = extrap_kind_used, fill_value='extrapolate')

        # predicted quaternion. 
        q0_prediced = q0_interpolator(t_vec_prediction).reshape(t_vec_prediction.shape[0], 1) # preDICED, lol
        q1_prediced = q1_interpolator(t_vec_prediction).reshape(t_vec_prediction.shape[0], 1)
        q2_prediced = q2_interpolator(t_vec_prediction).reshape(t_vec_prediction.shape[0], 1)
        q4_prediced = q4_interpolator(t_vec_prediction).reshape(t_vec_prediction.shape[0], 1)
        quat_predicted = np.hstack((t_vec_prediction.reshape(t_vec_prediction.shape[0], 1),
            q0_prediced, q1_prediced, q2_prediced, q4_prediced))
    elif prediction_setting == 'quadratic_interp_2pts_smarter':
        ii_used = [1,2] # last 2 messages received. 1st point extrap
        quat_message_2 = quat_message[ii_used,:] # use 2 message points
        quat_dot_message_2 = quat_dot_message[ii_used,:]
        # Get quaternion elements
        t_ind = 0 # column index for time-stamp. [t; q1; q2; q3; q4]
        q1_0 = quat_message_2[0, t_ind+1]
        q1_1 = quat_message_2[1, t_ind+1]
        qdot1_1 = quat_dot_message_2[1, t_ind+1]
        q2_0 = quat_message_2[0, t_ind+2]
        q2_1 = quat_message_2[1, t_ind+2]
        qdot2_1 = quat_dot_message_2[1, t_ind+2]
        q3_0 = quat_message_2[0, t_ind+3]
        q3_1 = quat_message_2[1, t_ind+3]
        qdot3_1 = quat_dot_message_2[1, t_ind+3]

        q4_0 = quat_message_2[0, t_ind+4]
        q4_1 = quat_message_2[1, t_ind+4]
        qdot4_1 = quat_dot_message_2[1, t_ind+4]

        if 0: # debug purposes, getting quat_rate numerically
            qdot1_1 = (quat_message_2[1,1] - quat_message_2[0,1])/t_eph_interval
            qdot2_1 = (quat_message_2[1,2] - quat_message_2[0,2])/t_eph_interval
            qdot3_1 = (quat_message_2[1,3] - quat_message_2[0,3])/t_eph_interval
            qdot4_1 = (quat_message_2[1,4] - quat_message_2[0,4])/t_eph_interval
        t0 = quat_message_2[0,0]
        t1 = quat_message_2[1,0]
        q_1_full = np.array([q1_0, q1_1, qdot1_1]).reshape([3,1])        
        q_2_full = np.array([q2_0, q2_1, qdot2_1]).reshape([3,1])
        q_3_full = np.array([q3_0, q3_1, qdot3_1]).reshape([3,1])
        q_4_full = np.array([q4_0, q4_1, qdot4_1]).reshape([3,1])
        # design meatrix : Q(t) = a + bt + ct^2
        # Q = A [a, b, c]^T
        A = np.array([[1, t0, t0**2],
                        [1, t1, t1**2],
                        [0, 1, 2*t1]] )
        A_inv = np.linalg.inv(A)
        # coefficients
        coeff_q1 = A_inv @ q_1_full
        coeff_q2 = A_inv @ q_2_full
        coeff_q3 = A_inv @ q_3_full
        coeff_q4 = A_inv @ q_4_full                    

        # predicted quaternions. 
        q1_prediced = coeff_q1[0] + coeff_q1[1] * t_vec_prediction + coeff_q1[2] * t_vec_prediction**2
        q1_prediced = q1_prediced.reshape(t_vec_prediction.shape[0], 1)

        q2_prediced = coeff_q2[0] + coeff_q2[1] * t_vec_prediction + coeff_q2[2] * t_vec_prediction**2
        q2_prediced = q2_prediced.reshape(t_vec_prediction.shape[0], 1)

        q3_prediced = coeff_q3[0] + coeff_q3[1] * t_vec_prediction + coeff_q3[2] * t_vec_prediction**2
        q3_prediced = q3_prediced.reshape(t_vec_prediction.shape[0], 1)

        q4_prediced = coeff_q4[0] + coeff_q4[1] * t_vec_prediction + coeff_q4[2] * t_vec_prediction**2
        q4_prediced = q4_prediced.reshape(t_vec_prediction.shape[0], 1)

        quat_predicted = np.hstack((t_vec_prediction.reshape(t_vec_prediction.shape[0], 1),
            q1_prediced, q2_prediced, q3_prediced, q4_prediced))
    elif prediction_setting == 'nothing':
        quat_message_2 = quat_message[[1,2],:] # use 2 message points
        quat_dot_message_2 = quat_dot_message[2,:]
        quat_predicted = np.zeros((t_vec_prediction.shape[0], 5))
        quat_predicted[:,0] = t_vec_prediction
        quat_predicted[:,1] = quat_message_2[1,1]
        quat_predicted[:,2] = quat_message_2[1,2]
        quat_predicted[:,3] = quat_message_2[1,3]
        quat_predicted[:,4] = quat_message_2[1,4]
    # Notfixed
    # elif prediction_setting == 'landsat_slerp':
    #     ## Quaternion prediction goes here
    #     quat_predicted = interpolate_LANDSAT(quat_message, t_vec_prediction)
    #     # TODO Bayajid's implemented Landsat attitude prediction
    # elif prediction_setting == 'slerp':
    #     # convert quat message into scipy's SLERP-friendly convention
    #     quat_predicted = extrapolate_quaternion(quat_message, t_vec_prediction)

    elif prediction_setting == 'propagate_dumbway':
        dt = quat_update_rate # [s], quaternion propagation step
        # propagation time-vector starts at t_last_msg [first t represents last msg!]
        t_vec_propagation = np.round(np.arange(quat_message[-1,0], quat_message[-1,0]+20, dt),2)
        ii_propagation = [ii for ii, t in enumerate(quat_ham) if t[0] in t_vec_propagation]
        # true quaternion at propagation time-steps
        quat_true_propagation = quat_ham[ii_propagation, :] # true quaternion values at propagaton steps
        # placeholder for propagated quaternion
        quat_propagated = np.zeros((t_vec_propagation.shape[0], 5))
        ## https://math.stackexchange.com/questions/1896379/how-to-use-the-quaternion-derivative
        # notfollowing quaternion maths
        for ii, t_ii in enumerate(t_vec_propagation):
            if ii == 0: # initialize
                q_0 = quat_message[-1,1:]
                q_ii = q_0
                q_dot0 = quat_dot_message[-1,1:] 
            else:
                ## get W from q_dot
                q_ii = q_0 + q_dot0 * dt

            quat_propagated[ii,0] = t_ii
            quat_propagated[ii,1:] = q_ii

            # update
            q_0 = q_ii
    # find where to cut off ephemeris message (probably, untested)
    if 'real' in attitude_setting:
        ii_eph_cutoff = [ii for ii, t in enumerate(t_vec) if t in t_vec_prediction][0]
    else:
        ii_eph_cutoff = [ii for ii, t in enumerate(t_vec.values) if t in t_vec_prediction][0]
    quat_est_pred = quat_ham[ii_eph_cutoff:,:]
    quat_true_pred = quat_ham_true[ii_eph_cutoff:,:] # Matches time-vector as quat_predicted
    ea_true_pred = ea_true[ii_eph_cutoff:,:] # Euler angles at prediction times
    ea_est_pred = ea_est[ii_eph_cutoff:,:]    
    ## calculate pointing error
    # get ref vector
    ref_pt_vec = ref_pt_vec / np.linalg.norm(ref_pt_vec)
    # rotate with true quaternion
    # cut off last 10 due to some rounding issue
    if 'propagate' not in prediction_setting:
        nrows_pred = t_vec_prediction.shape[0]
    else:        
        t_vec_prediction = t_vec_propagation
        nrows_pred = t_vec_propagation.shape[0]     
        quat_true = quat_true_propagation
        quat_predicted = quat_propagated
    los_true_all = np.zeros((nrows_pred, 3))
    los_pred_all = np.zeros((nrows_pred, 3))
    pe_all = np.zeros((nrows_pred, 1))
    for ii, t_ii in enumerate(t_vec_prediction):
        # los_pred = rot_predicted.as_matrix() @ ref_pt_vec
        #31-03-2023 New addition, testing quaternion multiplication (potential) issue
        if make_predicted_quaternion_norm:
            quat_predicted[ii,1:] = quat_predicted[ii,1:] / np.linalg.norm(quat_predicted[ii,1:])
        if prediction_setting == 'est_vs_true':
            q_input = quat_est_pred[ii, 1:5]
        else:
            q_input = quat_predicted[ii,1:]        
        q_true = quat_true_pred[ii,1:5]
        if skip_predicting:
            q_input = quat_true[ii,1:]
        
        los_pred = rot.rotate_with_quat(ref_pt_vec, q_input, h_q = 1, conj_switch = 0).flatten()        
        # dcm_pred = conv.convert_quat2dcm(q_input)
        # dcm_pred = conv.convert_ea2dcm(ea_est_pred[ii,:], deg = 1)
        # los_pred = dcm_pred @ ref_pt_vec
        
        # los_pred = rot.rotate_with_quat(ref_pt_vec, q_input, h_q = 1, conj_switch = 0).flatten()        
        
        # use true quaternion isntead
        # q_true = conv.convert_ea2quat(ea_true_pred[ii,:])
        # dcm_true = conv.convert_quat2dcm(q_true)
        
        dcm_true = conv.convert_ea2dcm(ea_true_pred[ii,:], deg = 1)
        los_true = dcm_true @ ref_pt_vec

        # los_true = rot.rotate_with_quat(ref_pt_vec, q_true)
        # los_true = rot.rotate_with_quat(ref_pt_vec, q_true)
        # rotate predicted vector only using quaternion
        # rotate with predicted quaternion
        # calc PE betwen the two vectors
        los_true_all[ii,:] = los_true
        los_pred_all[ii,:] = los_pred
        # store
        pe_all[ii] = calc_dot_angle(los_true, los_pred)
    
        # evaluate pointing error only at propagated time-steps
        # ideally would interpoalte betwen teh points too... but notime :p


        # pass
    # prep output csv
    columns = ['t_s', 'pe_urad']
    columns.extend([f'qtrue_{ii}' for ii in range(4)])    
    columns.extend([f'qpred_{ii}' for ii in range(4)])
    output_data = np.hstack((t_vec_prediction.reshape((nrows_pred,1)), pe_all*1e6, quat_true_pred[:nrows_pred,1:], quat_predicted[:,1:]))

    pe_df = pd.DataFrame(columns = columns, data = output_data)
    if save_pe:
        save_path = f'{path_outputs}\{attitude_setting}_{prediction_setting}_{t_eph_interval}s.csv'
        pe_df.to_csv(save_path, index = False)
        print(f'PE saved to as {attitude_setting}_{prediction_setting}_{t_eph_interval}s.csv')
if 0:
    import attitude_tools.attitude_predictions.pe_att_extrap_xmsg_plot
    importlib.reload(attitude_tools.attitude_predictions.pe_att_extrap_xmsg_plot)
    plt.show()

if 1: # 
    f, ax = plt.subplots()
    f.suptitle(f'{quat_update_rate} Hz Eph. PE setting {attitude_setting.upper()}. \n 1) EA swap : {swap_euler_angle_input}. 2) QuatNorm:{make_predicted_quaternion_norm}..')
    ax.plot(t_vec_prediction, pe_all*1e6, label = f'Prediction, {prediction_setting}')
    ax.legend()
    ax.grid()
    ax.set_ylabel('PE [urad]')
    ax.set_xlabel('Prediction time t [s]')
    # ax.set_ylim([0,10000])
    ax.set_xlim([t_vec_prediction[0],t_vec_prediction[0]+2])
    ax.set_yscale('log')

if show_quat_true and 0:
    f, axs = plt.subplots(4, figsize = (5,8))
    cdot_option = ['q;w', 'DCM;DCMdot']

    f.suptitle(f'Quaternions [Scalar-first] - {attitude_setting}; \nqdot from {cdot_option}')
    for ii, ax in enumerate(axs):
        ax.set_ylabel(['q1', 'q2', 'q3', 'q_0'][ii] + ' [-]', fontweight = 'bold')
        ax.plot(quat_ham[:,0], quat_ham[:,ii+1], label = 'True')
        ax.scatter(quat_message[:,0], quat_message[:,ii+1], label = 'MSG', c = 'y', s = 10)
        ax.plot(t_vec_prediction, quat_predicted[:,ii+1], label = 'Predicted')
        ax.grid()
        ax.set_ylim([np.min(quat_message[:,1+ii])/5, np.max(quat_message[:,1+ii])])
        ax.plot([t_pred_start, t_pred_start], list(ax.get_ylim()), label = 'Prediction Start', c = 'r')
        if ii == 3:
            ax.legend()
        ax.set_xlim([t_pred_start-5, t_pred_start+5])
        # ax.set_ylim(quat_message[-2,ii+1]*0.9,quat_message[-1,ii+1]*1.2 )
        # ax.set_ylim([[0.6, 0.7], [0.15, 0.25], [0.2, 0.3], [0.75, 0.6]][ii])
    ax.set_xlabel('t [s]', fontweight = 'bold')
if show_quatrate_true and 0:
    f, axs = plt.subplots(4, figsize = (5,8))
    
    cdot_option = ['q;w', 'DCM;DCMdot']
   
    f.suptitle(f'Quaternions [Scalar-last] - {attitude_setting}; \nqdot from {cdot_option}')
    for ii, ax in enumerate(axs):
        ax.set_ylabel(['qdot1', 'qdot2', 'qdot3', 'qdot_0'][ii] + ' [-]', fontweight = 'bold')
        ax.plot(quat_dot_ham[:,0], quat_dot_ham[:,ii+1], label = 'True')
        ax.scatter(quat_dot_message[:,0], quat_dot_message[:,ii+1], label = 'MSG', c = 'y', s = 10)
        # ax.plot(t_vec_prediction, quat_predicted[:,ii+1], label = 'Predicted')
        ax.grid()
        ax.plot([t_pred_start, t_pred_start], list(ax.get_ylim()), label = 'Prediction Start', c = 'r')
        if ii == 3:
            ax.legend()
        # ax.set_xlim([35, 45])
        # ax.set_ylim(quat_message[-2,ii+1]*0.9,quat_message[-1,ii+1]*1.2 )
        # ax.set_ylim([[0.6, 0.7], [0.15, 0.25], [0.2, 0.3], [0.75, 0.6]][ii])
    ax.set_xlabel('t [s]', fontweight = 'bold')
plt.show()