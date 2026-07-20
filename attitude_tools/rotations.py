# Tools to perform vector rotations using 
# Euler angles/Quaternions/Direction Cosine Matrices
import numpy as np
from scipy.linalg import expm
import os
import sys
import pathlib
import numpy as np
import os, sys
sys.path.insert(1, os.getcwd()[:os.getcwd().index('astropynaric')+13])
import attitude_tools.conversions as conv

def rot_basic(angle, rot_ax, rad = 0):
    """Function to return the basic rotation matrices
    using euler angle inputs.
    Sequence : vec_rotated = ROT @ vec_initial
    
    Args:
        angle (float): angle to rotate around (default rad, can be deg w/ rad=0 input)
        rot_ax (int): rotation axis 1-> X, 2-> Y, 3-> Z
        rad (int, optional): Whether angle is 1-rad or 0-deg. Defaults to 0.

    Returns:
        array: Rotation matrix in form of numpy array
        
    """    
    # rot ax == 1-> Around X, 2->around Y, 3->around Z
    if not rad: # convert to radians if given in degrees
        angle_used = np.deg2rad(angle)
    else:
        angle_used = angle
    ct = np.cos(angle_used)
    st = np.sin(angle_used)

    if rot_ax == 1: # rotate around X
        ROT = np.array([
            [1, 0, 0],
            [0, ct, st],
            [0, -st, ct]
        ])
    elif rot_ax == 2: # rotate around Y
        ROT = np.array([
            [ct, 0, -st],
            [0, 1, 0],
            [st, 0, ct]
        ])
    elif rot_ax == 3: # rotate around Z
        ROT = np.array([
            [ct, st, 0],
            [-st, ct, 0],
            [0, 0, 1]
        ])
    return ROT

def multiply_quat(q_1, q_2):
    """Perform scalar-last quaternion multiplication

    Args:
        q_1 (4x1 vector): first quaternion
        q_2 (4x1 vector): second quaternion

    Returns:
        q_comp: 4x1 resulting composite quaternion
    """    
    q1 = q_2[0]
    q2 = q_2[1]
    q3 = q_2[2]
    q4 = q_2[3]
    Q2 = np.array([
        [q4, q3, -q2, q1],
        [-q3, q4, q1, q2],
        [q2, -q1, q4, q3],
        [-q1, -q2, -q3, q4]    
    ])
    q_comp = Q2 @ q_1
    return q_comp
def multiply_quat_ham_matrix(q_1, q_2):
    """Perform hamiltonian quaternion multiplication in matrix form
    
    Args:
        q_1 (4x1 vector): first quaternion (second applied rotation)
        q_2 (4x1 vector): second quaternion (first applied rotation)

    Returns:
        q_comp: 4x1 resulting composite quaternion
    """    
    q1 = q_1[0]
    q2 = q_1[1]
    q3 = q_1[2]
    q4 = q_1[3]
    Q2 = np.array([
        [q1, -q2, -q3, -q4],
        [q2, q1, q4, -q3],
        [q3, -q4, q1, q2],
        [q4, q3, -q2, q1]    
    ])
    q_comp = Q2 @ q_2
    return q_comp
def multiply_quat_hamiltonian(q_1, q_2):
    """Perform hamiltonian quaternion multiplication
    
    Args:
        q_1 (4x1 vector): first quaternion
        q_2 (4x1 vector): second quaternion

    Returns:
        q_comp: 4x1 resulting composite quaternion
    """    
    ## 27-04-2023 WHY IS IT IN THAT ORDER?? 
    # its the Hamiltonian product. The reverse can also be used
    # q1 x q2 = q2 . q1 (left - modified, right - hamiltonian)
    # for hamiltonian, its q . V . q conj. For the other, its q_conj x V x q
    # source - Crassidis2014 and Matlab quaternion multiplication
    p1 = q_2[0] # 
    p2 = q_2[1]
    p3 = q_2[2]
    p4 = q_2[3]

    q1 = q_1[0] # scalar
    q2 = q_1[1]
    q3 = q_1[2]
    q4 = q_1[3]

    q_comp = np.array([[p1*q1 - p2*q2 - p3*q3 - p4*q4],
                       [p1*q2 + p2*q1 + p3*q4 - p4*q3],
                       [p1*q3 - p2*q4 + p3*q1 + p4*q2],
                       [p1*q4 + p2*q3 - p3*q2 + p4*q1]
                       ])
    return q_comp
def rotate_with_quat_mat(vec, q, norm = 1):
    """Function to rotate a vector using a quaternion input using 
    vector-matrix composite operations
    designed to work both with hamiltonian (scalar-first) as well as
    scalar-last quaternion inputs

    Args:
        vec (3x1 array): vector to be rotated
        q (4x1 array): quaternion input
        conj_switch (bool, optional): 1: q_conj vec q. 0: q vec q_conj. Defaults to 1.
        h_q (bool, optional): Whether hamiltonian qutaernion is input. 1-Hamiltonian.. Defaults to 0.

    Returns:
        vec_out (3x1 array): vector rotated w/ quaternion
    """    
    
    if norm:
        q = q / np.linalg.norm(q)
    vec_augm = np.hstack(([0], vec))
    q_conj = np.copy(q)
    q_conj[1:] = -q_conj[1:]
    product_inner = multiply_quat_ham_matrix(vec_augm, q_conj)
    product = multiply_quat_ham_matrix(q, product_inner)[1:]
    return product
def rotate_with_quat_mat_swaperoo(vec, q, norm = 1):
    """Function to rotate a vector using a quaternion input using 
    vector-matrix composite operations
    designed to work with passive rotations

    Args:
        vec (3x1 array): vector to be rotated
        q (4x1 array): quaternion input
    Returns:
        vec_out (3x1 array): vector rotated w/ quaternion
    """    
    
    if norm:
        q = q / np.linalg.norm(q)
    vec_augm = np.hstack(([0], vec))
    q_conj = np.copy(q)
    q_conj[1:] = -q_conj[1:]
    product_inner = multiply_quat_ham_matrix(q_conj, vec_augm)
    product = multiply_quat_ham_matrix(product_inner, q)[1:]
    return product    
    
def rotate_with_quat(vec, q, conj_switch = 0, h_q = 1, norm = 1, reshuffle=1):    
    """Function to rotate a vector using a quaternion input
    designed to work both with hamiltonian (scalar-first) as well as
    scalar-last quaternion inputs

    Args:
        vec (3x1 array): vector to be rotated
        q (4x1 array): quaternion input
        conj_switch (bool, optional): 1: q_conj vec q. 0: q vec q_conj. Defaults to 1.
        h_q (bool, optional): Whether hamiltonian qutaernion is input. 1-Hamiltonian.. Defaults to 0.

    Returns:
        vec_out (3x1 array): vector rotated w/ quaternion
    """    
  

    if h_q: # hamiltonian product multiplication
        vec_q = np.hstack(([0], vec))
        if norm:
            q = q / np.linalg.norm(q)
        q_conj = np.copy(q)
        q_conj[1:] = -q_conj[1:]   
        if conj_switch: # switch order of q_conj and q in rotation
            q_1 = q_conj
            q_2 = q  
        else:
            q_1 = q
            q_2 = q_conj
        if not reshuffle:
            vec_q_rot0 = multiply_quat_hamiltonian(q_1, vec_q)
            vec_q_rotated = multiply_quat_hamiltonian(vec_q_rot0, q_2)
            # convert back to vector and return
        else:
            q_1 = q_conj
            q_2 = q  
            vec_q_rot0 = multiply_quat_hamiltonian(vec_q, q_1)
            vec_q_rotated = multiply_quat_hamiltonian(q_2, vec_q_rot0)
        vec_out = vec_q_rotated[1:] 
    elif not h_q: # scalar-last quaternion multiplication
        vec_q = np.hstack((vec, [0]))
        # get conjugate of rotation quaternion
        q_conj = np.copy(q)
        q_conj[:3] = -q_conj[:3]    
        if not conj_switch: # switch order of q_conj and q in rotation
            q_1 = q
            q_2 = q_conj
        else:
            q_1 = q_conj
            q_2 = q  
        # first product
        vec_q_rot0 = multiply_quat(q_1, vec_q)
        vec_q_rotated = multiply_quat(vec_q_rot0, q_2)
        # convert back to vector and return
        vec_out = vec_q_rotated[:3] 
    return vec_out.flatten()
def rotate_all_quat(los_eci, quat_all):
    """rotate all LOS with all quaternions

    Args:
        los_eci (Nx3): LOS in ECI
        quat_all (Nx4): hamiltonian quaternions from ECI to desired frame

    Returns:
        _type_: _description_
    """    # perform quaternion rotation for an array of vectors and qutaernions
    
    if quat_all.shape[0] == 1 or len(quat_all.shape) == 1: # extend quaternion vector to array
        quat_used = np.zeros((los_eci.shape[0], 4))
        quat_used[:,:] = quat_all
    else:
        quat_used = quat_all 

    los_rotated = np.zeros(los_eci.shape)
    for ii, los_ii in enumerate(los_eci):
        quat_ii = quat_used[ii,:]
        # rotate. q_ham Vec q_ham_conj
        los_rot_full = rotate_with_quat(los_ii, quat_ii)
        # store
        los_rotated[ii,:] = los_rot_full
    return los_rotated
def qmultiply(p, q):
    # function for quaternion multiplication as done in simulink
    out = [p(1) * q(1) - p(2) * q(2) - p(3) * q(3) - p(4) * q(4),
            p(1) * q(2) + p(2) * q(1) + p(3) * q(4) - p(4) * q(3),
            p(1) * q(3) - p(2) * q(4) + p(3) * q(1) + p(4) * q(2),
            p(1) * q(4) + p(2) * q(3) - p(3) * q(2) + p(4) * q(1)]
    return out
if __name__ == '__main__':
    if 0:
        los_given =  [170000,    -4360000,    -1181000]
        quat_given = [ 0.18,  0.54, -0.26, -0.78]
        los_rotated = rotate_with_quat(los_given, quat_given)
        print(f'LOS rotated : {los_rotated}')
    else:
        q2 = np.array([2,3,4,5])
        q1 = np.array([1,2,3,4])
        q1 = np.array([-0.9982615856102344, -7.2179467813316e-18, 0.058939008262276237, 0.0])
        los_given = np.array([-10, 100, 1000])
        print(f'Los given : {los_given}')        
        print(f'Classic full rotation with q1: {np.round(rotate_with_quat(los_given, q1),2)}')
        print(f'Swaperoo with q1 : {np.round(rotate_with_quat_mat_swaperoo(los_given, q1),2)}')
        print(f'NEW full rotation with q1: {np.round(rotate_with_quat_mat(los_given, q1),2)}')