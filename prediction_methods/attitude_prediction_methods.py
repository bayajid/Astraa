
import importlib
import os, sys
import numpy as np
sys.path.insert(1, os.getcwd()[:os.getcwd().index('astropynaric')+13])
# Store attitude prediction methods
import basic_tools.vector_operations as vec_op
def quadratic_extrap(q_message, q_dot_message, t_vec_prediction):
    # 2-point quadratic extrapolation    
    # Get quaternion elements
    t_ind = 0 # column index for time-stamp. [t; q1; q2; q3; q4]
    q1_0 = q_message[0, t_ind+1]
    q1_1 = q_message[1, t_ind+1]
    qdot1_1 = q_dot_message[1, t_ind+1]
    q2_0 = q_message[0, t_ind+2]
    q2_1 = q_message[1, t_ind+2]
    qdot2_1 = q_dot_message[1, t_ind+2]
    q3_0 = q_message[0, t_ind+3]
    q3_1 = q_message[1, t_ind+3]
    qdot3_1 = q_dot_message[1, t_ind+3]
    q4_0 = q_message[0, t_ind+4]
    q4_1 = q_message[1, t_ind+4]
    qdot4_1 = q_dot_message[1, t_ind+4]
    
    t0 = q_message[0,0]
    t1 = q_message[1,0]

    # set t_ref to start at 0
    dt_tref = t0
    t0 = t0 - dt_tref
    t1 = t1 - dt_tref
    t_vec_prediction = t_vec_prediction - dt_tref

    q_1_full = np.array([q1_0, q1_1, qdot1_1]).reshape([3,1])        
    q_2_full = np.array([q2_0, q2_1, qdot2_1]).reshape([3,1])
    q_3_full = np.array([q3_0, q3_1, qdot3_1]).reshape([3,1])
    q_4_full = np.array([q4_0, q4_1, qdot4_1]).reshape([3,1])

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
    # add back
    t_vec_prediction = t_vec_prediction + dt_tref
    # pack up
    q_predicted = np.hstack((t_vec_prediction.reshape(t_vec_prediction.shape[0], 1),
        q1_prediced, q2_prediced, q3_prediced, q4_prediced))
    return q_predicted
def quadratic_extrap_redone(q_both, qdot_both, t_both, t_vec_prediction):
    # 2-point quadratic extrapolation    
    # Get quaternion elements
    t_ind = 0 # column index for time-stamp. [t; q1; q2; q3; q4]
   
    t0 = t_both[0]
    t1 = t_both[1]

    # set t_ref to start at 0
    dt_tref = t0
    t0 = t0 - dt_tref
    t1 = t1 - dt_tref
    t_vec_prediction = t_vec_prediction - dt_tref

    q_1_full = np.array([q_both[0,1-1], q_both[1,1-1], qdot_both[1,1-1]]).reshape([3,1])        
    q_2_full = np.array([q_both[0,2-1], q_both[1,2-1], qdot_both[1,2-1]]).reshape([3,1])
    q_3_full = np.array([q_both[0,3-1], q_both[1,3-1], qdot_both[1,3-1]]).reshape([3,1])
    q_4_full = np.array([q_both[0,4-1], q_both[1,4-1], qdot_both[1,4-1]]).reshape([3,1])

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
    # add back
    t_vec_prediction = t_vec_prediction + dt_tref
    # pack up
    q_predicted = np.hstack((t_vec_prediction.reshape(t_vec_prediction.shape[0], 1),
        q1_prediced, q2_prediced, q3_prediced, q4_prediced))
    if q_predicted.shape[0] == 1:
        q_predicted = q_predicted.flatten()
    return q_predicted[1:]
def constant_prediction(q_message, q_dot_message, t_vec_prediction):
    # output constant Q VALUE, equal to alst q message row
    q_predicted = np.zeros((t_vec_prediction.shape[0], 5))
    q_predicted[:,0] = t_vec_prediction
    for ii, q in enumerate(q_predicted):
        q_predicted[ii,1:] = q_message[1,1:]
    return q_predicted

def ideal_prediction(q_ham_est, t_vec_prediction, t_vec):
    # output true quaternion, sliced to match t_vec_prediction
    q_predicted = np.zeros((t_vec_prediction.shape[0], 5))
    q_predicted[:,0] = t_vec_prediction
    
    ii_eph_cutoff = [ii for ii, t in enumerate(t_vec) if t in t_vec_prediction]
    q_predicted[:,1:] = q_ham_est[ii_eph_cutoff,:]
    return q_predicted

def euler_propagation(q_message, q_dot_message, t_vec_prediction):
    ## https://math.stackexchange.com/questions/1896379/how-to-use-the-quaternion-derivative
    # no tfollowing quaternion maths
    # Euler propgating quaternion. q_1 = q_0 + dt * q_dot
    # assume q_dot is constant
    dt = t_vec_prediction[1] - t_vec_prediction[0] # [s], quaternion propagation step
    q_propagated = np.zeros((t_vec_prediction.shape[0], 5))
    for ii, t_ii in enumerate(t_vec_prediction):
        if ii == 0: # initialize
            q_0 = q_message[-1,1:]
            q_ii = q_0
            q_dot0 = q_dot_message[-1,1:] 
        else:
            ## get W from q_dot
            q_ii = q_0 + q_dot0 * dt

        q_propagated[ii,0] = t_ii
        q_propagated[ii,1:] = q_ii

        # update
        q_0 = q_ii
    return q_propagated

def get_quad_pred_error(quat_est, quat_rate_est, quat_true, t_vec_loaded):
    """Combined script for quaternion prediction. Inputs : est quaternion array,
    true quaternion array, est quaternion rate, time vector

    Args:
        quat_est (_type_): _description_
        quat_rate_est (_type_): _description_
        quat_true (_type_): _description_
        t_vec_loaded (_type_): _description_

    Returns:
        _type_: pe_stored, q_pred_stored, t_vec_loaded
    """        
    pe_stored = np.zeros((t_vec_loaded.shape[0],1))
    q_pred_stored = np.zeros((t_vec_loaded.shape[0],4))
    
    # set up quaternion predictor
    
    for ii, t_ii in enumerate(t_vec_loaded):
        if ii>1:        
            ii_given = [ii-2,ii-1]
            ii_predicted = [ii]
            
            q_given = quat_est[ii_given,:]
            q_dot_given = quat_rate_est[ii_given,:]
            t_stamp_given = t_vec_loaded[ii_given]    
    
            t_pred = t_vec_loaded[ii_predicted]
            q_true_pred = quat_true[ii_predicted].flatten()
            q_pred = quadratic_extrap_redone(q_given, q_dot_given, t_stamp_given, t_pred).flatten()
            pe_rot = np.max(vec_op.get_pe_for_rot(q_pred, q_true_pred))
            
            pe_stored[ii] = pe_rot # urad
            q_pred_stored[ii,:] = q_pred 
            
    
    return pe_stored, q_pred_stored, t_vec_loaded