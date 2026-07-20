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
import basic_tools.vector_operations as vec_op
import attitude_tools.conversions as conv
import pointing_calculations.ae_calculation as ae_calc

def compute_q_mountingoffset_quest_debug(az_el_commanded,
                                    az_el_measured,
                                    check_non_colin = 1,
                                    deg = 0,
                                    fct_option = 1,
                                    q_mountingoffset_precal = None):
    
    if not deg:
        az_el_commanded = np.rad2deg(az_el_commanded)
        az_el_measured = np.rad2deg(az_el_measured)
        
    
    los_meas = [ae_calc.xyz_from_aer((azel_meas[0], azel_meas[1], 1)) for azel_meas in az_el_measured]
    los_comm = [ae_calc.xyz_from_aer((azel_comm[0], azel_comm[1], 1)) for azel_comm in az_el_commanded]
    
    los_set_meas = np.array(los_meas)
    los_set_comm = np.array(los_comm)
    if fct_option == 2:
        function = svd        
    elif fct_option == 1:
        function = quest_debug
    q_from_precal_to_cal = function(los_set_comm, los_set_meas)

    q_from_bf_to_tf = q_from_precal_to_cal
    
    q_from_bf_to_tf = q_from_bf_to_tf.flatten()
    if check_non_colin:
        angle_between_los = vec_op.calc_dot_angle(los_set_meas[-1,:], los_set_meas[0,:])
        
            # angle_between_los = np.round(angle_between_los, 3)
    else:
        angle_between_los = None
    
    return q_from_bf_to_tf, angle_between_los


def quest_debug(body_vectors, ref_vectors, num_iterations = 10):
    """Quest algorithm to return a least squares estimate

    Args:
        body_vectors (Nx3): vectors in frame A
        ref_vectors (Nx3): vectors in frame B
    """    
    weights = np.ones(ref_vectors.shape[0])

    body_vectors_T = np.transpose(body_vectors)
    B = body_vectors_T @ ref_vectors
    q_out, qual = B_to_quat(B)
    axis = 3
    if 1:
        if qual < 0.1:
            B2 = B * np.array([1., -1., -1.])
            q_out, qual = B_to_quat(B2, num_iterations)
            axis = 0
            q_last = undo_quat(q_out, axis)
            q_last = q_last/np.linalg.norm(q_last)
        if qual < 0.1:
            B2 = B * np.array([-1., 1., -1.])
            q_out, qual = B_to_quat(B2, num_iterations)
            axis = 1
            q_last = undo_quat(q_out, axis)
            q_last = q_last/np.linalg.norm(q_last)
        if qual < 0.1:
            B2 = B * np.array([-1., -1., 1.])
            q_out, qual = B_to_quat(B2, num_iterations)
            axis = 2
            
    q_out = undo_quat(q_out, axis)
    q_last = q_out/np.linalg.norm(q_out)
        
    q_ham = np.copy(q_last)
    q_ham[0] = q_last[-1]
    q_ham[1:] = q_last[:-1]
    return q_ham
def compute_q_mountingoffset_quest(az_el_commanded,
                                    az_el_measured,
                                    check_non_colin = 1,
                                    deg = 0,
                                    q_mountingoffset_precal = None):
    
    if not deg:
        az_el_commanded = np.rad2deg(az_el_commanded)
        az_el_measured = np.rad2deg(az_el_measured)
        
    
    los_meas = [ae_calc.xyz_from_aer((azel_meas[0], azel_meas[1], 1)) for azel_meas in az_el_measured]
    los_comm = [ae_calc.xyz_from_aer((azel_comm[0], azel_comm[1], 1)) for azel_comm in az_el_commanded]
    
    los_set_meas = np.array(los_meas)
    los_set_comm = np.array(los_comm)
    
    q_from_precal_to_cal = quest(los_set_comm, los_set_meas)
    if q_mountingoffset_precal is not None:
        q_from_bf_to_tf = compute_q_product(q_from_precal_to_cal, 
                                            q_mountingoffset_precal)
    else:
        q_from_bf_to_tf = q_from_precal_to_cal
    q_from_bf_to_tf = q_from_bf_to_tf.flatten()
    if check_non_colin:
        angle_between_los = vec_op.calc_dot_angle(los_set_meas[-1,:], los_set_meas[0,:])
        
            # angle_between_los = np.round(angle_between_los, 3)
    else:
        angle_between_los = None
    
    return q_from_bf_to_tf, angle_between_los
def quest(body_vectors, ref_vectors, num_iterations = 10):
    """Quest algorithm to return a least squares estimate

    Args:
        body_vectors (Nx3): vectors in frame A
        ref_vectors (Nx3): vectors in frame B
    """    
    weights = np.ones(ref_vectors.shape[0])

    body_vectors_T = np.transpose(body_vectors)
    B = body_vectors_T @ ref_vectors
    q_out, qual = B_to_quat(B)
    axis = 3
    if 1:
        if qual < 0.1:
            B2 = B * np.array([1., -1., -1.])
            q_out, qual = B_to_quat(B2, num_iterations)
            axis = 0
            q_last = undo_quat(q_out, axis)
            q_last = q_last/np.linalg.norm(q_last)
        if qual < 0.1:
            B2 = B * np.array([-1., 1., -1.])
            q_out, qual = B_to_quat(B2, num_iterations)
            axis = 1
            q_last = undo_quat(q_out, axis)
            q_last = q_last/np.linalg.norm(q_last)
        if qual < 0.1:
            B2 = B * np.array([-1., -1., 1.])
            q_out, qual = B_to_quat(B2, num_iterations)
            axis = 2
            
    q_out = undo_quat(q_out, axis)
    q_last = q_out/np.linalg.norm(q_out)
        
    q_ham = np.copy(q_last)
    q_ham[0] = q_last[-1]
    q_ham[1:] = q_last[:-1]
    return q_ham
def svd(body_vectors,
        ref_vectors,
        stddevs = None
        ):
    """ Singular Value Decomposition method by [Markley1988].

        Args:
            body_vectors: (N, 3) array of measurement vectors, where N is the number of measurements.
            ref_vectors: (N, 3) array of reference vectors, where N is the number of measurements.
            stddevs: A list of size N containing the measurement standard deviations for each
                        measurement vector.

        Returns:
            Array of size (3, 3) containing the optimal attitude matrices.

        References:
            - [Shuster1981] Shuster, M.D. and Oh, S.D. "Three-Axis Attitude Determination from Vector
                            Observations," Journal of Guidance and Control, Vol.4, No.1, Jan.-Feb.
                            1981, pp. 70-77.
            - [Markley1988] Markley, F. Landis. "Attitude determination using vector observations and
                            the singular value decomposition." Journal of the Astronautical Sciences
                            36.3 (1988): 245-258.
    """

    ref_vectors = tf.cast(ref_vectors, tf.float32)
    body_vectors = tf.cast(body_vectors, tf.float32)
    if stddevs is None:
        stddevs = 1e-5 * np.ones(ref_vectors.shape[0])
    if body_vectors.shape != ref_vectors.shape:
        raise ValueError("body_vectors and ref_vectors are not the same size")

    # Equation 97 from [Shuster1981]
    sig_tot = 1. / tf.reduce_sum(1. / tf.convert_to_tensor(stddevs)**2)
    # Equation 96 from [Shuster1981]
    weights = sig_tot / tf.reshape(stddevs, [-1, 1])**2

    body_vectors = tf.transpose(body_vectors)

    # Vectorized form of equation 38 from [Shuster1981]
    B = tf.matmul(body_vectors, ref_vectors)
    # B = tf.matmul(body_vectors, ref_vectors * weights)

    s, u, v = tf.linalg.svd(B)

    s3 = tf.linalg.det(u) * tf.linalg.det(v)

    diag = tf.convert_to_tensor([1., 1., s3])

    optimal_matrix = tf.matmul(u * diag, v, transpose_b=True)


    q_out = conv.convert_dcm2quat(optimal_matrix.numpy().transpose())
    return q_out
def B_to_quat(B, num_iterations = 8):
    # Taken as quest_code from https://github.com/guilherme9820/AttitudeDetermination/blob/v3.0/attitude_determination/wahba_solutions.py
    S = B + B.transpose()
    z = np.array([B[1, 2] - B[2, 1], B[2, 0] - B[0, 2], B[0, 1] - B[1, 0]])
    Sz = np.array([z[0]*S[0, 0] + z[1]*S[0, 1] + z[2]*S[0, 2],
                    z[0]*S[1, 0] + z[1]*S[1, 1] + z[2]*S[1, 2],
                    z[0]*S[2, 0] + z[1]*S[2, 1] + z[2]*S[2, 2]])
    SSz = np.array(([Sz[0]*S[0, 0] + Sz[1]*S[0, 1] + Sz[2]*S[0, 2],
                    Sz[0]*S[1, 0] + Sz[1]*S[1, 1] + Sz[2]*S[1, 2],
                    Sz[0]*S[2, 0] + Sz[1]*S[2, 1] + Sz[2]*S[2, 2]]))
    
    # Characteristic Equation
    sigma = B[0, 0] + B[1, 1] + B[2, 2]  
    K1 = S - sigma * np.ones((3,3), dtype = float)
    K = np.array([[K1[0, 0], K1[0, 1], K1[0, 2], z[0]],
                    [K1[1, 0], K1[1, 1], K1[1, 2], z[1]],
                    [K1[2, 0], K1[2, 1], K1[2, 2], z[2]],
                    [z[0], z[1], z[2], sigma]])
    kappa = S[0, 0]*S[1, 1]+S[1, 1]*S[2, 2]+S[2, 2]*S[0, 0] \
        - S[0, 1]*S[1, 0]-S[1, 2]*S[2, 1]-S[2, 0]*S[0, 2]        
        
    kappa2 = np.sum(np.diag(K.transpose() * np.linalg.det(K)))
    # (eq. 71) from [Shuster1981]
    b = -2*sigma + kappa - (z[0]**2 + z[1]**2 + z[2]**2)
    c = -kappa2
    d = np.linalg.det(K)        

    lam = 1.
    for i in range(num_iterations):
        phi = lam**4 + b * lam**2 + c * lam + d
        phi_prime = 4 * lam**3 + 2 * b * lam + c

        lam -= (phi / phi_prime)

    # Equations 66 and 68 from [Shuster1981]
    alpha = lam**2 - sigma**2 + kappa
    beta = lam - sigma
    gamma = alpha*(lam + sigma) - np.linalg.det(S)
    X = alpha*z + beta*Sz + SSz

    # Optimal Quaternion (eq. 69) from [Shuster1981].
    # The codes will not work very well if implemented
    # following the papers directly, thus we must use the
    # quaternion inverse
    Q = np.array([-X[0], -X[1], -X[2], gamma])
    return Q, gamma
def undo_quat(quat, axis):
    if axis == 0:
        return np.array([quat[3], quat[2], -quat[1], -quat[0]])
    elif axis == 1:
        return np.array([-quat[2], quat[3], quat[0], -quat[1]])
    elif axis == 2:
        return np.array([quat[1], -quat[0], quat[3], -quat[2]])
    else:
        return quat

def triad(vecs_A, vecs_B):
    """TRIAD algorithm to return rotation from frame A to frame B

    Args:
        vecs_A (3x2): set of vectors in frame A
        vecs_B (3x2): set of vectors in frame B
        norms_given (int, optional): _description_. Defaults to 1.

    Returns:
        _type_: rotation matrix from frame A to frame B
    """    

    vec_A_1 = vecs_A[0,:]
    vec_A_2 = vecs_A[1,:]
    vec_B_1 = vecs_B[0,:]
    vec_B_2 = vecs_B[1,:]

    S_A = vec_A_1 / np.linalg.norm(vec_A_1)
    S_B = vec_B_1 / np.linalg.norm(vec_B_1)
    cross_A = np.cross(vec_A_1, vec_A_2)
    cross_B = np.cross(vec_B_1, vec_B_2)
    M_A = cross_A / np.linalg.norm(cross_A)
    M_B = cross_B / np.linalg.norm(cross_B)

    mat_A = np.hstack((S_A.reshape([3,1]), M_A.reshape([3,1]), np.cross(S_A, M_A).reshape([3,1])))
    mat_B = np.hstack((S_B.reshape([3,1]), M_B.reshape([3,1]), np.cross(S_B, M_B).reshape([3,1])))

    DCM_resolved = mat_B @ mat_A.transpose()
    return DCM_resolved

def get_mo_quat_fromscan(AE_exp, AE_found, deg = 0, check_non_colin = 0, fct_option = None):
    """integrated function to convert AE to Cartesian LOS
    and use TRIAD algorithm to resolve the attitude between the two sets of LOS

    Args:
        AE_exp (float): commanded Az El (what moon angle was given/expected, rad)
        AE_found (float): seen Az El (what comes out of the scan, rad)
        deg (bool, optional): if Az El are in deg. Defaults to 0.

    Returns:
        quat_mo: resolved mounting offset quaternion; angle_between_los [rad]
    """    

    ii_used = [0, 1] 

    if not deg:
        AE_found = np.rad2deg(AE_found)
        AE_exp = np.rad2deg(AE_exp)

    ae_meas_1 = AE_found[ii_used[0],:] # terminal frame
    ae_meas_2 = AE_found[ii_used[1],:]
    ae_exp_1 = AE_exp[ii_used[0],:] # body frame
    ae_exp_2 = AE_exp[ii_used[1],:]
    aer_meas_1 = np.hstack((ae_meas_1, 1))
    aer_meas_2 = np.hstack((ae_meas_2, 1))
    aer_exp_1 = np.hstack((ae_exp_1, 1))
    aer_exp_2 = np.hstack((ae_exp_2, 1))

    los_meas_1 = ae_calc.xyz_from_aer(aer_meas_1)
    los_meas_2 = ae_calc.xyz_from_aer(aer_meas_2)
    los_exp_1 = ae_calc.xyz_from_aer(aer_exp_1)
    los_exp_2 = ae_calc.xyz_from_aer(aer_exp_2)

    vec_meas = np.vstack((los_meas_1, los_meas_2))
    vec_exp = np.vstack((los_exp_1, los_exp_2))
            

    dcm_mo = triad(vec_exp, vec_meas)
    quat_mo = conv.convert_dcm2quat(dcm_mo)
    if check_non_colin:
        # compute non-colinearity of vector
        angle_between_los = vec_op.calc_dot_angle(los_meas_1, los_meas_2)
    else:
        angle_between_los = None

    return quat_mo, angle_between_los
def what_angle_between_ae(ae_set_1, ae_set_2, deg = 0, round = 0):
    """function to compute angle between
    two sets of az/el measurements. Inputs in [deg]

    Args:
        ae_set_1 (az/el): list of az/el or array
        ae_set_2 (az/el): list of az/el or array
        deg (int, optional): Whether to return [deg] or [rad]. Defaults to 0.

    Returns:
        angle_between_los (float): deg or rad between sets of meas
    """    
    if type(ae_set_1) == list:
        ae_set_1.append(1)
        ae_set_2.append(1)
        aer_set_1 = np.array(ae_set_1)
        aer_set_2 = np.array(ae_set_2)
    else:
        aer_set_1 = np.hstack((ae_set_1, 1))
        aer_set_2 = np.hstack((ae_set_2, 1))
    los_set_1 = ae_calc.xyz_from_aer(aer_set_1)
    los_set_2 = ae_calc.xyz_from_aer(aer_set_2)

    angle_between_los = vec_op.calc_dot_angle(los_set_1, los_set_2)
    if deg:
        angle_between_los = np.rad2deg(angle_between_los)
    if round:
        angle_between_los = np.round(angle_between_los, 3)
    return angle_between_los

def find_attitude_for_ae(pass_position_data, AE_chosen_all, ii_scans = [0,1]):
    """Function to compute attitude commands
    for a given host/target position and desired Az/El
    for now only supports 2 points

    Args:
        pass_position_data (_type_): _description_
        ii_scans (_type_): _description_
        AE_chosen_all (_type_): _description_
    """    
    r_host = pass_position_data.iloc[:,[1,2,3]].values
    t_gps = pass_position_data.iloc[:,[0]].values
    r_moon = pass_position_data.iloc[:,[4,5,6]].values
    illumination = pass_position_data.iloc[:,[7]].values
    
    dt_data = t_gps[1] - t_gps[0]
    
    nrows = 2

    ea_eci2bf_command_all = np.zeros((nrows, 6)) # EA; EA_rate [deg, deg/s]
    quat_eci2bf_command_all = np.zeros((nrows, 4)) # scalar-first
    for ii, ii_scan in enumerate(ii_scans):
        AE_req = AE_chosen_all[ii,:]
        r_host_ii = r_host[ii_scan,:]
        v_host_ii = [0,0,0] # placeholder, not needed here

        r_moon_ii = r_moon[ii_scan,:]
        los_eci = r_moon_ii - r_host_ii
        los_req_lct = ae_calc.xyz_from_aer([AE_req[0], AE_req[1], np.linalg.norm(los_eci)])
        
        # Compute commanded body-frame attitude
        # actual mointing offset is unknown
        rot_bf2lct_known = np.eye(3, dtype = float)

        los_req_bf = rot_bf2lct_known.transpose() @ los_req_lct
        los_req_bf_norm = los_req_bf / np.linalg.norm(los_req_bf)

        los_eci_norm = los_eci / np.linalg.norm(los_eci)
        rot_eci2bf_command = los_req_bf_norm.reshape((3,1)) @ los_eci_norm.reshape((1,3))
        # convert to quaternions and euler angles
        quat_eci2bf_command = conv.convert_dcm2quat(rot_eci2bf_command)
        ea_eci2bf_command = conv.convert_dcm2ea(rot_eci2bf_command) # Deg

        quat_eci2bf_command_all[ii,:] = quat_eci2bf_command
        ea_eci2bf_command_all[ii,:3] = ea_eci2bf_command
    
    ea_eci2bf_command_all[:-1,3:] = (ea_eci2bf_command_all[1:,:3] - ea_eci2bf_command_all[:-1,:3])/dt_data
    return quat_eci2bf_command_all, ea_eci2bf_command_all

def calc_attitude_for_ae(
                        r_host,
                        t_gps,
                        r_target,
                        quat_mo,
                        ae_desired):
    """Function to compute attitude commands
    for a given host/target position and desired Az/El

    Args:
        r_host
        r_target
        quat_mo
        ae_desired [deg]
        quat_mo_known
    Returns:
        quat_eci2bf
    """    

    
    dt_data = t_gps[1] - t_gps[0]
    
    nrows = ae_desired.shape[0]

    ea_eci2bf_command_all = np.zeros((nrows, 6)) # EA; EA_rate [deg, deg/s]
    quat_eci2bf_command_all = np.zeros((nrows, 4)) # scalar-first
    rot_bf2lct_known = conv.convert_quat2dcm(quat_mo)
    rot_lct2bf = rot_bf2lct_known.transpose()
    
    for ii, ae_ii in enumerate(ae_desired):
        
        r_host_ii = r_host[ii,:]
        v_host_ii = [0,0,0] # placeholder, not needed here

        r_target_ii = r_target[ii,:]
        los_eci = r_target_ii - r_host_ii
        los_req_lct = ae_calc.xyz_from_aer([ae_ii[0], ae_ii[1], np.linalg.norm(los_eci)])
        # Compute commanded body-frame attitude
        # actual mointing offset is unknown

        los_req_bf = rot_lct2bf @ los_req_lct
        los_req_bf_norm = los_req_bf / np.linalg.norm(los_req_bf)

        los_eci_norm = los_eci / np.linalg.norm(los_eci)
        rot_eci2bf_command = los_req_bf_norm.reshape((3,1)) @ los_eci_norm.reshape((1,3))
        # convert to quaternions and euler angles
        quat_eci2bf_command = conv.convert_dcm2quat(rot_eci2bf_command)
        ea_eci2bf_command = conv.convert_dcm2ea(rot_eci2bf_command) # Deg

        quat_eci2bf_command_all[ii,:] = quat_eci2bf_command
        ea_eci2bf_command_all[ii,:3] = ea_eci2bf_command
    
    return quat_eci2bf_command_all, ea_eci2bf_command_all
def calc_attitude_for_ae_single(
                        r_host,
                        r_target,
                        quat_mo,
                        ae_desired):
    """Function to compute attitude commands
    for a given host/target position and desired Az/El

    Args:
        r_host
        r_target
        quat_mo
        ae_desired [deg]
        quat_mo_known
    Returns:
        quat_eci2bf
    """    
    
    nrows = 1
    rot_bf2lct_known = conv.convert_quat2dcm(quat_mo)
    rot_lct2bf = rot_bf2lct_known.transpose()
    
    v_host_ii = [0,0,0] # placeholder, not needed here
    los_eci = r_target[:3] - r_host[:3]
    los_req_lct = ae_calc.xyz_from_aer([ae_desired[0], ae_desired[1], np.linalg.norm(los_eci)])
    # Compute commanded body-frame attitude
    # actual mointing offset is unknown

    los_req_bf = rot_lct2bf @ los_req_lct
    los_req_bf_norm = los_req_bf / np.linalg.norm(los_req_bf)

    los_eci_norm = los_eci / np.linalg.norm(los_eci)
    rot_eci2bf_command = los_req_bf_norm.reshape((3,1)) @ los_eci_norm.reshape((1,3))
    
    rot_axis = np.cross(los_eci_norm, los_req_bf_norm)
    rot_angle = np.arccos(np.dot(los_eci_norm, los_req_bf_norm))
    rot_eci2bf_cmd = conv.convert_eigenaxis2dcm(rot_axis, -rot_angle)
    quat_eci2bf_command = conv.convert_dcm2quat(rot_eci2bf_cmd)
    # convert to quaternions and euler angles
    # quat_eci2bf_command = conv.convert_dcm2quat(rot_eci2bf_command.transpose())
    # ea_eci2bf_command = conv.convert_dcm2ea(rot_eci2bf_command) # Deg

    
    return quat_eci2bf_command

    
if __name__ == '__main__':
    # generate attitude profile
    import matplotlib.pyplot as plt
    if 1:
        num_iterations = 10
        
        # QUEST IMPLEMENTATION
        body_vectors = np.array([[1,0,0], [1,0,0], [1,0,0],[1,0,0]])
        ref_vectors = np.array([[0,1,0], [0,1,0], [0,1,0], [0,1,0]])
        q_out = quest(body_vectors, ref_vectors)
    if 0: # old
        az_lim_0 = -175
        az_lim_1 = -az_lim_0
        e_0 = 85
        e_1 = -55
        nr_el_steps = 5
        el_values = np.linspace(e_0, e_1, nr_el_steps)
        el_values = np.append(el_values, el_values[-1])
        dt = 1
        t_vec = np.arange(0,5*3600,dt)
        ang_rate = 0.4 # deg/s

        ae_profile = np.zeros((t_vec.shape[0],2))

        rotate_az = 1
        rotate_el = 0

        ae_0 = [az_lim_0, e_0]
        az_lim_curr = az_lim_0

        el_ii = 1
        el_lim_curr = el_values[el_ii]
        ae_profile[0,:] = ae_0
        az_rotate_dir = np.sign(az_lim_curr * (-1))
        el_rotate_dir = -1

        a_0 = ae_0[0]
        e_0 = ae_0[1]
        for ii, t in enumerate(t_vec):
            a_1 = a_0
            e_1 = e_0
            if rotate_az:
                a_1 = a_1 + az_rotate_dir * ang_rate * dt

                if np.abs(a_1) >= np.abs(az_lim_curr):
                    rotate_az = 0
                    rotate_el = 1
                    az_lim_curr = az_lim_curr * (-1)
                    az_rotate_dir = - az_rotate_dir
                # go from az lim to az lim
            if rotate_el:
                e_1 = e_0 + el_rotate_dir * ang_rate * dt

                if e_1 <= el_lim_curr:
                    rotate_az = 1
                    rotate_el = 0
                    el_ii +=1
                    try:
                        el_lim_curr = el_values[el_ii]
                    except:
                        ae_profile = ae_profile[:ii]
                        break

            # store
            ae_profile[ii,:] = [a_1, e_1]
            a_0 = a_1
            e_0 = e_1
        if 1:
            f, ax = plt.subplots()
            ax.plot(ae_profile[:,0], ae_profile[:,1])
            ax.scatter(ae_profile[0,0], ae_profile[0,1], label = 'start', c = 'r')
            ax.scatter(ae_profile[-1,0], ae_profile[-1,1], label = f'end, t = {t/60:.0f} min', c = 'g')
            ax.legend()
            ax.set_xlabel('Az [deg]')
            ax.set_ylabel('El [deg]')
            ax.grid()