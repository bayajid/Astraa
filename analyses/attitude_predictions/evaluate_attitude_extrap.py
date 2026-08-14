#%% 2023-01-02 Simulate quaternion measurements at different rates
# and interpolate/extrapolate values to see resulting errors
# compared to the true attitude
# For conversion from Quat/DCM/EA, using scipy
## https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.as_euler.html
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import scipy.interpolate as interpolate
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp 
# import splines.quaternion
from tools_pe import calc_dot_angle
from tools_attitude import extrapolate_quaternion
import tools_attitude as rot
from scipy.linalg import expm

path_cwd = os.getcwd()
path_outputs = fr'{path_cwd}/outputs'

#%% settings and loaing true attitude data
# saving conditionals
do_all_update_rates = 1 # try all Attitude message update rates or just 1 Hz
print_all_attitude_messages = 1 # if 0, will print only 1 attitude message (test purp.)
save_pe = 1
# plotting
show_true_angles = 1
show_true_angle_rat1 = 1
show_quat_true = 1

try_not_predicting = 1 # evaluate PE if no prediction is made (lol)
# interpolation_method = 'quadratic_interp_basic'
# interpolation_method = 'quadratic_interp_2pts' # use 2 quaternions, predict 3rd one with qdot. Quadratic interpolation
interpolation_method = 'quadratic_interp_2pts_smarter'
# interpolation_method = 'linear_interp' # use 2 quaternions, linear interpolation
# interpolation_method = 'slerp'
# interpolation_method = 'propagate_cleanway' # use quaternion rate and propagate using quaternion composition rules
# interpolation_method = 'propagate_dumbway' # use quaternion rate and propagate using just euler

# Settings for true attitude
# setting = 'rotate_all_pred084'
setting = 'rotate_azel_octhw467'
# setting = 'rotate_all_pred084_swap' # switch acceleration sign at t=40 when w=0.82 deg/s
# setting = 'rotate_all_axes' # default - all 3 axes rotating like mad. Start at t=40s
# setting = 'rotate_swap' # default - all 3 axes rotating like mad. Switch acceleration sign to negative at t=40s
# setting = 'rotate_all_delayed' # all 3 axes rotating like mad. Start at t=80s
# setting = 'real_stable_sat' # real attitude tracking data in 1s steps for a stable satellite

rotating_scenario_filter = 'rotate' # used to separate rotating simulated attitude
# from stable satellite case

if rotating_scenario_filter in setting:
    fname = setting
    if 'delayed' in setting:
        fname = 'rotate_all_axes'
    att_true_df = pd.read_csv(fr'{path_outputs}/true_attitude_{fname}.csv')
    print(f'Loaded true attitude for setting : {setting}')
    att_all = att_true_df.iloc[:,[1,2,3]].values # deg
    om_all = att_true_df.iloc[:,[-3, -2, -1]].values # deg/s
    att_rates_all = att_true_df.iloc[:,[4,5,6]].values
    t_vec = np.round(att_true_df.iloc[:,0], 2)
    nrows = len(t_vec)
    # placeholder for quaternions
    quat_ham_outputs = np.zeros((nrows, 5)) # t ; quat
    quat_dot_ham_outputs = np.zeros((nrows, 5)) # t ; quat_dot

## Load true Euler Angle data and convert to quaternions/quaternion rates\ [SIMULATED DATA]
if rotating_scenario_filter in setting: 
    for ii, t_ii in enumerate(t_vec):
        # intrinsic rotations used    
        r_ii = R.from_euler('ZYX', att_all[ii,:], degrees = True)
        om_ii = om_all[ii,:]
        om_ii_rad = np.deg2rad(om_ii)
        w_1, w_2, w_3 = om_ii_rad[0], om_ii_rad[1], om_ii_rad[2]
        # HAMILTON convention - last element is the scalar
        r_ii_quat = r_ii.as_quat()
        q_dot = 1/2 * np.array([
            [0, w_3, -w_2, w_1],
            [-w_3, 0, w_1, w_2],
            [w_2, -w_1, 0, w_3],
            [-w_1, -w_2, -w_3, 0]]) @ r_ii_quat
        quat_ham_outputs[ii,0] = t_ii
        quat_ham_outputs[ii,1:] = r_ii_quat
        quat_dot_ham_outputs[ii,0] = t_ii
        quat_dot_ham_outputs[ii,1:] = q_dot
    quat_outputs_df = pd.DataFrame(data = np.hstack((quat_ham_outputs,quat_dot_ham_outputs[:,1:])) , columns = ['t_s', 'q1', 'q2', 'q3', 'q4', 'qdot1', 'qdot2', 'qdot3', 'qdot4'])
    quat_outputs_df.to_csv(fr'{path_outputs}/quat_true_{setting}.csv', index = False)

    if setting == "rotate_all_axes" or setting == "rotate_swap":
        t_pred_start = 40
    elif setting == "rotate_all_delayed":
        t_pred_start = 79
    elif 'rotate' in setting:
        t_pred_start = 40
    t_step_prop = 0.01 # t-step used in true attitude generation
elif 'real' in setting:
    # TODO load real satellite data
    # reqs - t_vector starts from 0
    # data is at 1s rates. Can onloy get finer with interpolation. Just use 1s steps for testing purposes
    nrows_loaded = 1000
    dt_data = 1 # interval between given dataset
    fname = r'mystery_realsat_quaternions.csv'
    quat_df = pd.read_csv(fr'{path_outputs}/{fname}')
    quat_ham_outputs = quat_df.iloc[:nrows_loaded,[1,2,3,4]].values
    quat_dot_ham_outputs = quat_df.iloc[:nrows_loaded,[5,6,7,8]].values
    nrows = quat_dot_ham_outputs.shape[0]
    t_vec = np.round(np.arange(0, nrows, dt_data),1)
    ## append time-vector to quat, quat_dots
    quat_ham_outputs = np.hstack((t_vec.reshape((nrows, 1)), quat_ham_outputs))
    quat_dot_ham_outputs = np.hstack((t_vec.reshape((nrows, 1)), quat_dot_ham_outputs))


    t_pred_start = 100 # arbitrarily chosen start-time
    t_step_prop = dt_data

if rotating_scenario_filter in setting: 
    if show_true_angles:
        f, axs = plt.subplots(3, figsize = (5,8))
        f.suptitle(f'True Euler Angles - {setting}')
        for ii, ax in enumerate(axs):
            ax.set_ylabel(['Roll', 'Pitch', 'Yaw'][ii] + ' [deg]', fontweight = 'bold')
            ax.plot(t_vec, att_all[:,ii], label = 'True')
            ax.grid()
            ax.legend()
            ax.plot([t_pred_start, t_pred_start], list(ax.get_ylim()), label = 'Prediction Start', c = 'r')
            ax.set_xlim([35, 45])
        ax.set_xlabel('t [s]', fontweight = 'bold')
    if show_true_angle_rat1:
        f, axs = plt.subplots(3, figsize = (5,8))
        f.suptitle(f'True Euler Angle Rates - {setting}')
        for ii, ax in enumerate(axs):
            ax.set_ylabel(['Roll', 'Pitch', 'Yaw'][ii] + ' rate [deg/s]', fontweight = 'bold')
            ax.plot(t_vec, att_rates_all[:,ii], label = 'True')
            ax.grid()
            ax.legend()
            ax.plot([t_pred_start, t_pred_start], list(ax.get_ylim()), label = 'Prediction Start', c = 'r')
            ax.set_xlim([35, 45])
        ax.set_xlabel('t [s]', fontweight = 'bold')

if show_quat_true:
    f, axs = plt.subplots(4, figsize = (5,8))
    f.suptitle(f'True Quaternions [Hamiltonian] - {setting}')
    for ii, ax in enumerate(axs):
        ax.set_ylabel(['q1', 'q2', 'q3', 'q4'][ii] + ' [-]', fontweight = 'bold')
        ax.plot(quat_ham_outputs[:,0], quat_ham_outputs[:,ii+1], label = 'True')
        ax.grid()
        ax.plot([t_pred_start, t_pred_start], list(ax.get_ylim()), label = 'Prediction Start', c = 'r')
        ax.legend()
        ax.set_xlim([35, 45])
        # ax.set_ylim([[0.6, 0.7], [0.15, 0.25], [0.2, 0.3], [0.75, 0.6]][ii])
    ax.set_xlabel('t [s]', fontweight = 'bold')
plt.show()


#%% Interpolate between quaternions
# settings
if do_all_update_rates:
    quat_update_rates = [5, 1, 0.5] # Hz
else:
    quat_update_rates = [1]
if 'propagate' in interpolation_method:
    quat_update_rates = [0.1, 0.5, 1, 2] 
    if 'real' in setting:
        quat_update_rates = [1]
# Refrence pointing vector to be rotated with true/predicted quaternions
# for calculating the resulting pointing error

## Comment line below to avoid using a single quaternion update rate
quat_update_rates = [4, 2] # Hz

ref_pt_vec = np.array([100, 10, -10])
nr_eph = 3 # default nr eph points (DONT TOUCH! 2 point message cases handled later)
t0 = 0 ## REFERENCE START TIME. Can be used to start eph message later in the attitude time-series

print(f'''
WE PREDICTING QUATERNIONS BOYS
Method - {interpolation_method}
Update rates tried : {quat_update_rates} Hz
''')
for kk, quat_update_rate in enumerate(quat_update_rates):
    t_eph_interval = 1 / quat_update_rate # interval between eph updates TODO might have some issues with floats
    eph_length = t_eph_interval*(nr_eph-1)

    # t_eph_end = [t for t in t_vec if t >= eph_length+t0][0]
    t_eph_end = t_pred_start
    t_eph_start = t_eph_end - eph_length
    t_vec_eph = [t_eph_start, np.round(t_eph_start+t_eph_interval,3), t_eph_end]
    # np.arange(t_eph_start, np.round(t_eph_end+t_eph_interval,3), t_eph_interval) # round interval end to avoid rounding errors
    # if t_vec_eph[-1] > t_eph_end: # time vector slicing keeps including an additional time-step
    #     t_vec_eph = t_vec_eph[:3]
    # t_vec_eph = np.arange(t_eph_start, np.round(t_eph_end,4), t_eph_interval) # round interval end to avoid rounding errors
    if 'real' in setting:
        t_vec_prediction = np.arange(t_eph_end, t_vec[-100]+t_step_prop, t_step_prop)
    else:
        t_vec_prediction = np.arange(t_eph_end, t_vec.values[-100]+t_step_prop, t_step_prop)
    ### SLICING - ATTITUDE MESSAGE - PREDICTION VECTOR
    # Make slice to get quaternion ephemeris w/ time-stamps
    ii_message = np.array([kk for kk in range(nrows) if t_vec[kk] in t_vec_eph])
    # quat_message = np.array([quat_row for quat_row in quat_ham_outputs if quat_row[0] in t_vec_eph])
    quat_message = quat_ham_outputs[ii_message,:]
    quat_dot_message = quat_dot_ham_outputs[ii_message, :]
    if setting == 'propagate_cleanway':
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
    if interpolation_method == 'quadratic_interp_basic':        
        # make interpolants of sliced quaternions
        extrap_kind_used = 'quadratic'
        q0_interpolator = interpolate.interp1d(quat_message[:,0], quat_message[:,1],kind = extrap_kind_used, fill_value='extrapolate')
        q1_interpolator = interpolate.interp1d(quat_message[:,0], quat_message[:,2],kind = extrap_kind_used, fill_value='extrapolate')
        q2_interpolator = interpolate.interp1d(quat_message[:,0], quat_message[:,3],kind = extrap_kind_used, fill_value='extrapolate')
        q4_interpolator = interpolate.interp1d(quat_message[:,0], quat_message[:,4],kind = extrap_kind_used, fill_value='extrapolate')

        # predicted quaternion. 
        q0_prediced = q0_interpolator(t_vec_prediction).reshape(t_vec_prediction.shape[0], 1) # preDICED, lol
        q1_prediced = q1_interpolator(t_vec_prediction).reshape(t_vec_prediction.shape[0], 1)
        q2_prediced = q2_interpolator(t_vec_prediction).reshape(t_vec_prediction.shape[0], 1)
        q4_prediced = q4_interpolator(t_vec_prediction).reshape(t_vec_prediction.shape[0], 1)
        quat_predicted = np.hstack((t_vec_prediction.reshape(t_vec_prediction.shape[0], 1),
            q0_prediced, q1_prediced, q2_prediced, q4_prediced))
    elif interpolation_method == 'linear_interp':
        extrap_kind_used = 'linear'
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
    elif interpolation_method == 'quadratic_interp_2pts':    
        extrap_kind_used = 'quadratic'    
        # ii_used = [0,1] # first 2 messages received. 3rd point extrap
        ii_used = [1,2] # last 2 messages received. 1st point extrap
        quat_message_2 = quat_message[ii_used,:] # use 2 message points
        quat_dot_message_2 = quat_dot_message[ii_used,:]
        if ii_used[-1] == 1:
            quat_pred = quat_message_2[-1,1:] + t_eph_interval * quat_dot_message_2[-1,1:]
            # add time-stamp
            quat_pred_full = np.hstack((quat_message[-1,0], quat_pred))
            # make 3-point message again
            quat_message_3 = np.vstack((quat_message_2, quat_pred_full))
        elif ii_used[-1] == 2: 
            quat_pred = quat_message_2[0,1:] - t_eph_interval * quat_dot_message_2[0,1:]        
            quat_pred_full = np.hstack((quat_message[0,0], quat_pred))
            # make 3-point message again
            quat_message_3 = np.vstack((quat_pred_full, quat_message_2))
        if print_all_attitude_messages:
            print(f'ACTUAL Quaternion message \n[t, q1, q2, q3, q4]\n{quat_message_3[0,:]}\n{quat_message_3[1,:]}\n{quat_message_3[2,:]}')
        if 0: # using scipy's interp1d.
            # re-do quadratic interpolation
            # make interpolants of sliced quaternions
            q0_interpolator = interpolate.interp1d(quat_message_3[:,0], quat_message_3[:,1],kind = extrap_kind_used, fill_value='extrapolate')
            q1_interpolator = interpolate.interp1d(quat_message_3[:,0], quat_message_3[:,2],kind = extrap_kind_used, fill_value='extrapolate')
            q2_interpolator = interpolate.interp1d(quat_message_3[:,0], quat_message_3[:,3],kind = extrap_kind_used, fill_value='extrapolate')
            q4_interpolator = interpolate.interp1d(quat_message_3[:,0], quat_message_3[:,4],kind = extrap_kind_used, fill_value='extrapolate')

            # predicted quaternion. 
            # Note - This is some next-level big brain indexing. 
            q0_prediced = q0_interpolator(t_vec_prediction).reshape(t_vec_prediction.shape[0], 1) # preDICED, lol
            q1_prediced = q1_interpolator(t_vec_prediction).reshape(t_vec_prediction.shape[0], 1)
            q2_prediced = q2_interpolator(t_vec_prediction).reshape(t_vec_prediction.shape[0], 1)
            q4_prediced = q4_interpolator(t_vec_prediction).reshape(t_vec_prediction.shape[0], 1)
            quat_predicted = np.hstack((t_vec_prediction.reshape(t_vec_prediction.shape[0], 1),
                q0_prediced, q1_prediced, q2_prediced, q4_prediced))
        elif 0:
            #USE Numpy polyfit (returns coefficients)
            # flip to change to same coefficient convention, starting from lowest order
            coeff_q1 = np.flip(np.polyfit(quat_message_3[:,0], quat_message_3[:,1], deg = 2))
            coeff_q2 = np.flip(np.polyfit(quat_message_3[:,0], quat_message_3[:,2], deg = 2))
            coeff_q3 = np.flip(np.polyfit(quat_message_3[:,0], quat_message_3[:,3], deg = 2))
            coeff_q4 = np.flip(np.polyfit(quat_message_3[:,0], quat_message_3[:,4], deg = 2))

            # predicted quaternion. 
            # Note - This is some next-level big brain indexing. 
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
        else:
            # using own interpolation
            t_ind = 0 # column index for time-stamp. [t; q1; q2; q3; q4]
            q1_0 = quat_message_3[0, t_ind+1]
            q1_1 = quat_message_3[1, t_ind+1]
            q1_2 = quat_message_3[2, t_ind+1]

            q2_0 = quat_message_3[0, t_ind+2]
            q2_1 = quat_message_3[1, t_ind+2]
            q2_2 = quat_message_3[2, t_ind+2]

            q3_0 = quat_message_3[0, t_ind+3]
            q3_1 = quat_message_3[1, t_ind+3]
            q3_2 = quat_message_3[2, t_ind+3]

            q4_0 = quat_message_3[0, t_ind+4]
            q4_1 = quat_message_3[1, t_ind+4]
            q4_2 = quat_message_3[2, t_ind+4]


            t0 = quat_message_3[0,0]
            t1 = quat_message_3[1,0]
            t2 = quat_message_3[2,0]
            q_1_full = np.array([q1_0, q1_1, q1_2]).reshape([3,1])        
            q_2_full = np.array([q2_0, q2_1, q2_2]).reshape([3,1])
            q_3_full = np.array([q3_0, q3_1, q3_2]).reshape([3,1])
            q_4_full = np.array([q4_0, q4_1, q4_2]).reshape([3,1])
            # design meatrix : Q(t) = a + bt + ct^2
            # Q = A [a, b, c]^T
            A = np.array([[1, t0, t0**2],
                            [1, t1, t1**2],
                            [1, t2, t2**2]] )
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
    elif interpolation_method == 'quadratic_interp_2pts_smarter':
        ii_used = [1,2] # last 2 messages received. 1st point extrap
        quat_message_2 = quat_message[ii_used,:] # use 2 message points
        quat_dot_message_2 = quat_dot_message[ii_used,:]
        np.set_printoptions(4)

        ## Choose elements, make polynomial coefficients
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
    elif interpolation_method == 'slerp':
        # convert quat message into scipy's SLERP-friendly convention
        quat_predicted = extrapolate_quaternion(quat_message, t_vec_prediction)
    elif interpolation_method == 'propagate_cleanway':
        dt = quat_update_rate # [s], quaternion propagation step
        # propagation time-vector starts at t_last_msg [first t represents last msg!]
        t_vec_propagation = np.round(np.arange(quat_message[-1,0], quat_message[-1,0]+10, dt),2)
        ii_propagation = [ii for ii, t in enumerate(quat_ham_outputs) if t[0] in t_vec_propagation]
        # true quaternion at propagation time-steps
        quat_true_propagation = quat_ham_outputs[ii_propagation, :] # true quaternion values at propagaton steps
        # placeholder for propagated quaternion
        quat_propagated = np.zeros((t_vec_propagation.shape[0], 5))
        ## https://math.stackexchange.com/questions/1896379/how-to-use-the-quaternion-derivative
        for ii, t_ii in enumerate(t_vec_propagation):
            if ii == 0: # initialize
                q_0 = quat_message[-1,1:]
                om_0 = np.deg2rad(om_message[-1,:]) # [rad/s]
                q_dot0 = quat_dot_message[-1,1:]
                q_ii = q_0
            else:
                ## get W from q_dot
                w_1 = om_0[0]
                w_2 = om_0[1]
                w_3 = om_0[2]

                W = np.array([
                [0, w_3, -w_2, w_1],
                [-w_3, 0, w_1, w_2],
                [w_2, -w_1, 0, w_3],
                [-w_1, -w_2, -w_3, 0]])

                dq = expm(1/2 * W * dt)

                # q_ii = rot.multiply_quat(q_0, dq)
                q_ii = dq @ q_0

            quat_propagated[ii,0] = t_ii
            quat_propagated[ii,1:] = q_ii

            # update
            q_0 = q_ii
    elif interpolation_method == 'propagate_dumbway':
        dt = quat_update_rate # [s], quaternion propagation step
        # propagation time-vector starts at t_last_msg [first t represents last msg!]
        t_vec_propagation = np.round(np.arange(quat_message[-1,0], quat_message[-1,0]+20, dt),2)
        ii_propagation = [ii for ii, t in enumerate(quat_ham_outputs) if t[0] in t_vec_propagation]
        # true quaternion at propagation time-steps
        quat_true_propagation = quat_ham_outputs[ii_propagation, :] # true quaternion values at propagaton steps
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
    if 'real' in setting:
        ii_eph_cutoff = [ii for ii, t in enumerate(t_vec) if t in t_vec_prediction][0]
    else:
        ii_eph_cutoff = [ii for ii, t in enumerate(t_vec.values) if t in t_vec_prediction][0]

    quat_true = quat_ham_outputs[ii_eph_cutoff:,:] # Matches time-vector as quat_predicted
    ## calculate pointing error
    # get ref vector
    ref_pt_vec = ref_pt_vec / np.linalg.norm(ref_pt_vec)
    # rotate with true quaternion
    # cut off last 10 due to some rounding issue
    if 'propagate' not in interpolation_method:
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
        rot_true = R.from_quat(quat_true[ii,1:])
        rot_predicted = R.from_quat(quat_predicted[ii,1:])
        # rotate with predicted quaternion
        los_true = rot_true.as_matrix() @ ref_pt_vec
        # los_pred = rot_predicted.as_matrix() @ ref_pt_vec
        # rotate predicted vector only using quaternion
        los_pred = rot.rotate_with_quat(ref_pt_vec, quat_predicted[ii,1:])
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
    output_data = np.hstack((t_vec_prediction.reshape((nrows_pred,1)), pe_all*1e6, quat_true[:nrows_pred,1:], quat_predicted[:,1:]))

    pe_df = pd.DataFrame(columns = columns, data = output_data)
    if save_pe:
        save_path = f'{path_outputs}\PE\{setting}_{interpolation_method}_{t_eph_interval}s.csv'
        pe_df.to_csv(save_path, index = False)
        print(f'PE saved to as {setting}_{interpolation_method}_{t_eph_interval}s.csv')
#%% analyze what happens if no predictions are made - constant attitude assumed
if try_not_predicting:
    interpolation_method = 'do_nothing'
    pe_all = np.zeros((nrows_pred, 1))
    quat_predicted[:,1:] = quat_message[-1,1:]
    rot_predicted = R.from_quat(quat_predicted[0,1:]) # KEPT CONSTANT
    for ii, t_ii in enumerate(t_vec_prediction):
        rot_true = R.from_quat(quat_true[ii,1:])
        los_true = rot_true.as_matrix() @ ref_pt_vec
        los_pred = rot_predicted.as_matrix() @ ref_pt_vec
        pe_all[ii] = calc_dot_angle(los_true, los_pred)
    columns = ['t_s', 'pe_urad']
    columns.extend([f'qtrue_{ii}' for ii in range(4)])    
    columns.extend([f'qpred_{ii}' for ii in range(4)])
    output_data = np.hstack((t_vec_prediction.reshape((nrows_pred,1)), pe_all*1e6, quat_true[:nrows_pred,1:], quat_predicted[:,1:]))

    pe_df = pd.DataFrame(columns = columns, data = output_data)
    if save_pe:
        save_path = f'{path_outputs}\PE\{setting}_{interpolation_method}.csv'
        pe_df.to_csv(save_path, index = False)
        print(f'PE saved to as {setting}_{interpolation_method}.csv')
#%% Plot final result
if 0: # TODO should add this to the loop, make separate PE lines for each update rate
    f, ax = plt.subplots()
    f.suptitle('PE due to quadratic quaternion extrapolation (NOT SLERP)')
    ax.plot(t_vec_prediction - t_vec_prediction[0], pe_all*1e6, label = f'{quat_update_rate} Hz Eph')
    ax.legend()
    ax.grid()
    ax.set_ylabel('PE [urad]')
    ax.set_xlabel('Prediction time t [s]')
    ax.set_ylim([0,1000])
    ##
    f2, ax2 = plt.subplots()
    for ii in range(4):
        ax2.plot(t_vec_prediction- t_vec_prediction[0], quat_true[:,ii+1] - quat_predicted[:,ii+1], label = f'q{ii}')
    ax2.legend()
    ax2.set_ylabel('Quaternion element error [-]')
    ax2.set_xlabel('Prediction time t [s]')
    ax2.grid()
    