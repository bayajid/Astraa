#%%
import numpy as np
import importlib
import os, sys
sys.path.insert(1, os.getcwd()[:os.getcwd().index('astropynaric')+13])
import pandas as pd

from basic_tools.vector_operations import calc_dot_angle
import attitude_tools.conversions as conv
import attitude_tools.rotations as rot

def load_attitude_angles(path_est, path_true):
    # function to load true and noisy attitude data
    # and slice to euler angles/rates/omegas
    # add full path to estimated and to the true attitude (euler angle and rates)
    att_true = pd.read_csv(path_true)
    att_est = pd.read_csv(path_est)

    # unpack
    t_vec = att_true.iloc[:,0].values
    ea_true = att_true.iloc[:,[1,2,3]].values
    ea_dot_true = att_true.iloc[:,[4,5,6]].values
    ea_est = att_est.iloc[:,[1,2,3]].values
    ea_dot_est = att_est.iloc[:,[4,5,6]].values
    return t_vec, ea_true, ea_dot_true, ea_est, ea_dot_est

def get_quat(ea, ea_dot):
    # convert euler angles and  rates to hamiltonian quaternions
    # assume [deg, deg/s] inputs
    # 3-2-1 conv.
    quat_ham = np.zeros((ea.shape[0], 4)) # quaternion, estimated
    quat_dot_ham = np.zeros((ea.shape[0], 4)) # quaternion rate, estimated
    om_all = np.zeros((ea.shape[0], 3)) # rotational velocity vector, reproduced
    for ii, ea_ii in enumerate(ea):
        ea_dot_ii = ea_dot[ii,:]
        om_ii = conv.calc_omega(ea_ii, ea_dot_ii)
        # get q, qdot
        q, q_dot = conv.calc_qdot(ea_ii, om_ii, deg =1)
        # store
        quat_ham[ii,:] = q.flatten()
        quat_dot_ham[ii,:] = q_dot.flatten()
        om_all[ii,:] = om_ii

    return quat_ham, quat_dot_ham, om_all

def slice_att_msg(t_vec, t_pred_start, q_update_rate, 
                  quat_ham, quat_dot_ham, nr_eph = 2):
    # function wrapper to return quaternion message
    # for a given prediction time start, time vec, etc.
    # Only supports 2 value long ephemeris!
    t_eph_interval = 1 / q_update_rate # interval between eph updates TODO might have some issues with floats
    t_step_prop = t_vec[1] - t_vec[0]
    eph_length = t_eph_interval * int(nr_eph-1)
    t_eph_end = t_pred_start
    t_eph_start = t_eph_end - eph_length

    t_vec_msg = np.round(t_vec, int(1/t_step_prop))
    t_vec_eph = np.array([t_eph_start, t_eph_end]).reshape((2,1)) # assume 2 values
    # t_vec_eph = np.arange(t_eph_start, t_eph_end + t_eph_interval, t_eph_interval) # somehow include 40.2??
    
    ii_message = np.array([kk for kk, t in enumerate(t_vec) if t_vec_msg[kk] in t_vec_eph])
    
    quat_message = quat_ham[ii_message,:]
    quat_dot_message = quat_dot_ham[ii_message, :]
    quat_message = np.hstack((t_vec_eph, quat_message))
    quat_dot_message = np.hstack((t_vec_eph, quat_dot_message))

    return quat_message, quat_dot_message
def eval_atterr_pe(t_vec, ea_est, ea_true, ref_pt_vec):
    ## Evaluate PE introduced by quaternion prediction method
    ref_pt_vec = ref_pt_vec / np.linalg.norm(ref_pt_vec)
    
    nrows_pred = t_vec.shape[0]     
    
    # placeholders
    pe_all = np.zeros((nrows_pred, 1))
    
    for ii, t_ii in enumerate(t_vec):
        ea_true_ii = ea_true[ii,:]
        ea_est_ii = ea_est[ii,:]
        dcm_true = conv.convert_ea2dcm(ea_true_ii)
        dcm_est = conv.convert_ea2dcm(ea_est_ii)

        los_true = dcm_true @ ref_pt_vec
        los_pred = dcm_est @ ref_pt_vec
        
        # prediction pointing error via dot product rule
        pe_pred = calc_dot_angle(los_true, los_pred)*1e6 # [urad]
        pe_all[ii] = pe_pred
    return pe_all
def eval_prediction_pe(t_vec, t_vec_prediction, quat_predicted, q_ham_true, ref_pt_vec):
    ## Evaluate PE introduced by quaternion prediction method
    ii_maching = [ii for ii, t in enumerate(t_vec) if t in t_vec_prediction]
    ii_eph_cutoff = ii_maching[0]
    ii_eph_end = ii_maching[-1]
    ref_pt_vec = ref_pt_vec / np.linalg.norm(ref_pt_vec)
    quat_true_sliced = q_ham_true[ii_eph_cutoff:ii_eph_end+1,:]
    
    nrows_pred = t_vec_prediction.shape[0]     
    
    # placeholders
    pe_all = np.zeros((nrows_pred, 1))
    
    for ii, t_ii in enumerate(t_vec_prediction):

        q_true = quat_true_sliced[ii,:]
        q_pred = quat_predicted[ii,1:]

        los_true = rot.rotate_with_quat(ref_pt_vec, q_true)
        los_pred = rot.rotate_with_quat(ref_pt_vec, q_pred, h_q = 1, conj_switch = 0).flatten()        
        
        # prediction pointing error via dot product rule
        pe_pred = calc_dot_angle(los_true, los_pred)*1e6 # [urad]
        pe_all[ii] = pe_pred
    return pe_all, quat_true_sliced
def eval_pred_error(q_true, q_pred):
    # function to evaluate prediction errorf
    # rotating reference vectors with true/red
    # quatenrions and returning the PE for the vector
    # with the largest rotaiton difference
    ref_vecs = np.array([[1000, 0, 0],
                         [0, -1000, 0],
                         [0, 0, 1000],
                         [-1000, -2000, 1000]
                        ])
    pe_all = []
    for ii, ref_vec in enumerate(ref_vecs):
        los_true = rot.rotate_with_quat(ref_vec, q_true)
        los_pred = rot.rotate_with_quat(ref_vec, q_pred)
        
        # prediction pointing error via dot product rule
        pe_pred = calc_dot_angle(los_true, los_pred)*1e6 # [urad]
        pe_all.append(pe_pred)
    
    return pe_all, np.max(pe_all)

def vector_angular_error(q_true, q_pred, v_body=np.array([0, 0, 1])):
    """
    Calculates the angular difference between a body-frame vector v_body
    pointing in the estimated direction vs the reference direction.
    """
    # 1. Rotate the body vector into the reference frame for both quaternions
    v_est = rot.rotate_with_quat(v_body,q_pred)
    v_ref = rot.rotate_with_quat(v_body,q_true)
    
    # 2. Dot product to find the angle between the two 3D vectors
    # dot(v1, v2) = |v1||v2| cos(theta)
    dot_prod = np.dot(v_est, v_ref)
    dot_prod = np.clip(dot_prod, -1.0, 1.0) # Numerical safety
    
    angle_rad = np.arccos(dot_prod)
    
    return angle_rad * 1e6 # microradians

if __name__ == '__main__':
    import os, sys
    sys.path.insert(1, os.getcwd()[:os.getcwd().index('astropynaric')+13])
    fname_est = 'est_attitude_nojerk_noisy.csv'
    fname_true = 'true_attitude_nojerk_noisy.csv'
    folder_inputs = r'attitude_tools\attitude_predictions\outputs'
    path_est = f'{folder_inputs}\{fname_est}'
    path_true = f'{folder_inputs}\{fname_true}'
    outputs = load_attitude_angles(path_est, path_true)
    t_vec, ea_true, ea_dot_true, ea_est, ea_dot_est = outputs
    print(f'Done loading')
    quat_ham_true, quat_dot_ham_true, om_all_true = get_quat(ea_true, ea_dot_true)
    quat_ham_est, quat_dot_ham_est, om_all_est = get_quat(ea_est, ea_dot_est)
    print(f'Quat ready')
    t_pred_start = 40 # s
    q_update_rate = 5 # Hz
    quat_message, quat_dot_message = slice_att_msg(t_vec, t_pred_start, q_update_rate,
                                                   quat_ham_est, quat_dot_ham_est)
    print(f'''Attitude message time-stamps:
    QUATERNION MESSAGe update at {q_update_rate} Hz -> {1/q_update_rate} s between messages
    nr att points in message : {2}

    total time interval of message : {len(quat_message)}
    start : {quat_message[0,0]} s
    end : {quat_message[1,0]} s

    all eph time-stamps : {quat_message[:,0]} s
    ''')
    print(f'Quaternion message \n[t, q1, q2, q3, q4]\n{quat_message[0,:]}\n{quat_message[1,:]}')

    # total prediction time vector: {t_vec_prediction} s