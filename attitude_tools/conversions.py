# Tools to convert attitude representations: Quat, Euler angles, 
# Direction cosines and their rates.
import os
import sys
import pathlib
import numpy as np
import os, sys
sys.path.insert(1, os.getcwd()[:os.getcwd().index('astropynaric')+13])
import attitude_tools.rotations as rot
## Kinematic conversions
def convert_dcm2quat(dcm, ham_q = 1, norm = 1):
    """Convert DCM to a quaternion

    Args:
        DCM: ROtation matrix, in format r_rotated = DCM @ r_initial
        ham_q (bool, optional): Output quaternion format 1-scalar first. 0-scalar last. Defaults to 1.

    Returns:
        q_out: quaternion
    """    
    C_11 = dcm[0,0]
    C_22 = dcm[1,1]
    C_33 = dcm[2,2]
    C_23 = dcm[1,2]
    C_32 = dcm[2,1]
    C_31 = dcm[2,0]
    C_13 = dcm[0,2]
    C_12 = dcm[0,1]
    C_21 = dcm[1,0]

    q_scalar = 1/2 * np.sqrt(C_11 + C_22 + C_33 + 1)
    if np.abs(q_scalar)<1e-7:
        print(f'DCM->Quaternion conversion failed with q_w = 0.\nHandle 180 deg rotations manually calcualting the quaternions\n:q = [0, [rot_axis]] eg [0, 1, 0, 0] for 180 deg roll or [0, 0, 1, 0] for 180 pitch'),
        print(f'Increasing scalar value from {q_scalar} to 0.001')
        q_scalar = 1e-4
        # sys.exit()
    q_vec = 1 / (4*q_scalar) * np.array([[C_23 - C_32], [C_31 - C_13], [C_12 - C_21]])
    if ham_q:
        q_out = np.vstack((q_scalar, q_vec))
    else:
        q_out = np.vstack((q_vec, q_scalar))
    if norm:
        q_out = q_out / np.linalg.norm(q_out)
    return q_out.flatten()
def convert_quat2ea(q, deg = 1):
    dcm = convert_quat2dcm(q)
    ea = convert_dcm2ea(dcm)
    if not deg: # to [rad] if desired
        ea = np.deg2rad(ea)
    return ea
def convert_quat2dcm(q):
    """Convert Q to DCM
    using https://www.mathworks.com/help/aerotbx/ug/quatrotate.html
    Args:
        q (_type_): scalar-first quat

    Returns:
        dcm: vec_out=DCM@vec_in
    """    

    q_w = q[0]
    q1 = q[1]
    q2 = q[2]
    q3 = q[3]
    C_11 = (1 - 2*q2**2 - 2*q3**2)
    C_22 = (1 - 2*q1**2 - 2*q3**2)
    C_33 = (1 - 2*q1**2 - 2*q2**2)
    C_12 = 2*(q1*q2 + q_w*q3)
    C_13 = 2*(q1*q3 - q_w*q2)
    C_21 = 2*(q1*q2 - q_w*q3)
    C_23 = 2*(q2*q3 + q_w*q1)
    C_31 = 2*(q1*q3 + q_w*q2)
    C_32 = 2*(q1*q3 - q_w*q2)
    dcm = np.zeros((3,3))
    dcm[0,0] = C_11
    dcm[1,1] = C_22
    dcm[2,2] = C_33
    dcm[1,2] = C_23
    dcm[2,1] = C_32
    dcm[2,0] = C_31
    dcm[0,2] = C_13
    dcm[0,1] = C_12
    dcm[1,0] = C_21
    return dcm
def convert_ea2quat(RPY, deg = 1, ham_q = 1):
    """Convert Euler angles to a quaternion

    Args:
        RPY (array 3x1): Roll, Pitch and Yaw angles for a 3-2-1 rotation sequence
        deg (bool, optional): Unit of RPY angles. 1-deg. 0-rad. Defaults to 1.
        ham_q (bool, optional): Output quaternion format 1-scalar first. 0-scalar last. Defaults to 1.

    Returns:
        q_out: quaternion
    """    
    dcm = convert_ea2dcm(RPY, deg)
    # Singularities
    if RPY[0] == 180 and RPY[1] != 180 and RPY[2] != 180:
        q_out = np.array([0, 1, 0, 0])
    elif RPY[1] == 180 and RPY[0] != 180 and RPY[2] != 180:
        q_out = np.array([0, 0, 1, 0])
    elif RPY[2] == 180 and RPY[0] != 180 and RPY[1] != 180:
        q_out = np.array([0, 0, 0, 1])
    else:
        q_out = convert_dcm2quat(dcm, ham_q)
    return q_out

def convert_ea2dcm(ea, deg = 1):
    """Convert Euler angles to Direction Cosine Matrix, 3-2-1 rotation sequence
    results in DCM to be used as vec_rotaed = DCM @ vec_initial

    Args:
        ea (3x1 array): Roll, Pitch, Yaw angles [deg by default]
        deg (bool, optional): deg - 1, rad - 0. Defaults to 1.

    Returns:
        DCM_complete: composite 3-2-1 rotation amtrix
    """    
    if deg:
        ea_rad = np.deg2rad(ea)
    else:
        ea_rad = ea
    ROT_3 = rot.rot_basic(ea_rad[2], rot_ax = 3, rad = 1)
    ROT_2 = rot.rot_basic(ea_rad[1], rot_ax = 2, rad = 1)
    ROT_1 = rot.rot_basic(ea_rad[0], rot_ax = 1, rad = 1)
    # Complete Direction Cosine matrix. x_B = DCM_complete @ x_A (where _A refers to being in ref frame A)
    # and DCM rotates from A to B
    DCM_complete = ROT_1 @ ROT_2 @ ROT_3 # CHANGED ORDER 31-03-2023. was 3@2@1@vec (1-2-3) before
    return DCM_complete
def convert_dcm2ea(DCM):
    """convert a DCM to Euler angles

    Args:
        DCM (rot matrix): 3x3

    Returns:
        EA: [deg] 3-2-1 roll pitch yaw
    """    
    # Convert DCM to euler angles (roll, pitch, yaw) [deg], 3-2-1
    # ea_1 - Roll, ea_2 - Pitch, ea_3 - Yaw
    ea_2 = -np.arcsin(DCM[0,2])
    ea_3 = np.arctan2(DCM[0,1] , DCM[0,0])
    ea_1 = np.arctan2(DCM[1,2] , DCM[2,2])
    ea_vec = np.array([ea_1, ea_2, ea_3]) # Roll-Pitch-Yaw
    ea_deg = np.rad2deg(ea_vec)
    return ea_deg

def convert_dcm2eigenaxis(dcm):
    """Convert DCM to axis-angle rep

    Args:
        dcm (3x3 array): rot matrix

    Returns:
        e, theta: axis, angle [rad]
    """    
    # convert DCM to angle-axis representation
    # return euler angle [rad] and eigenaxis 
    cos_theta = 1/2 * ( np.sum(np.diag(dcm)) - 1)
    theta = np.arccos(cos_theta)
    sin_theta = np.sin(theta)
    # eigenaxis
    e = 1 / (2*sin_theta) * np.array([[dcm[1,2]-dcm[2,1]],
    [dcm[2,0] - dcm[0,2]],
    [dcm[0,1] - dcm[1,0]],
    ]) 
    e = e / np.linalg.norm(e)
    return e.flatten(), theta

def convert_eigenaxis2dcm(e,theta):
    """convert eigenaxis to dcm

    Args:
        e (_type_): axis
        theta (float): angle [rad]

    Returns:
        _type_: _description_
    """    
    if type(e) != list:
        if len(e.shape)>1:
            e = e.flatten()
    # input euler eigenaxis e and angle theta [rad]
    # get resulting DCM
    e_norm = e / np.linalg.norm(e)
    e_1 = e_norm[0]
    e_2 = e_norm[1]
    e_3 = e_norm[2]
    e_norm = np.reshape(e_norm, (3,1))
    E = np.array([[0, -e_3, e_2],
    [e_3, 0, -e_1],
    [-e_2, e_1, 0]])
    DCM = np.eye(3)*np.cos(theta) + (1 - np.cos(theta))*e_norm@np.transpose(e_norm) - E*np.sin(theta)
    return DCM

## Kinematic Differential Equations
def calc_ea_dot(ea, om, deg = 1):
    """function to calcualte euler angle rates from
        the current euler angles and rotational velocity vector
        inputs in deg or rad

    Args:
        ea (ARRAY): Euler angles Roll Pitch Yaw
        om (array): rotational rate vector: omega_x, omega_y, omega_z 
        about the initial frame axes
        deg (bool, optional): If deg (1) or rad (0) are input. Defaults to 1.

    Returns:
        ea_dot: 3x1 array of euler angle rates [deg/s]
    """    
    # convert to rad, rad/s
    if deg: 
        ea_rad = np.deg2rad(ea)
        om_rad = np.deg2rad(om)
    else:
        ea_rad = ea
        om_rad = om

    ea_1 = ea_rad[0]
    ea_2 = ea_rad[1]
    ea_3 = ea_rad[2]
    
    # get sines and cosines
    c_ea1 = np.cos(ea_1)
    c_ea2 = np.cos(ea_2)
    c_ea3 = np.cos(ea_3)

    s_ea1 = np.sin(ea_1)    
    s_ea2 = np.sin(ea_2)
    s_ea3 = np.sin(ea_3)    

    ## from spacecraft attitude dynamics and control lectures
    ea_dot = 1 / c_ea2 * np.array([
        [c_ea2, s_ea1 * s_ea2, c_ea1 * s_ea2],
        [0, c_ea1 * c_ea2, -s_ea1 * c_ea2],
        [0, s_ea1, c_ea1]])
    ea_dot = ea_dot @ om_rad
    
    # convert to deg/s
    ea_dot_deg = np.rad2deg(ea_dot)

    return ea_dot_deg

def calc_omega(ea, ea_dot, deg = 1):
    """function to angular rates from
        the current euler angles and euler angle rates
        inputs in deg or rad

    Args:
        ea (ARRAY): Euler angles Roll Pitch Yaw
        ea_dot (array): rotational rate vector: rollrate, pithcrate, yawrate
        about the initial frame axes
        deg (bool, optional): If deg (1) or rad (0) are input. Defaults to 1.

    Returns:
        ea_dot: 3x1 array of euler angle rates [deg/s]
    """    
    # convert to rad, rad/s
    if deg: 
        ea_rad = np.deg2rad(ea)
        ea_dot_rad = np.deg2rad(ea_dot)
    else:
        ea_rad = ea
        ea_dot_rad = ea_dot

    ea_1 = ea_rad[0]
    ea_2 = ea_rad[1]
    ea_3 = ea_rad[2]

    # get sines and cosines
    c_ea1 = np.cos(ea_1)
    c_ea2 = np.cos(ea_2)
    c_ea3 = np.cos(ea_3)

    s_ea1 = np.sin(ea_1)    
    s_ea2 = np.sin(ea_2)
    s_ea3 = np.sin(ea_3)

    # product of ea_dot and 1/cos(pitch)
    ea_prod = 1 / c_ea2 * np.array([
        [c_ea2, s_ea1 * s_ea2, c_ea1 * s_ea2],
        [0, c_ea1 * c_ea2, -s_ea1 * c_ea2],
        [0, s_ea1, c_ea1]])
    om_vec = np.linalg.inv(ea_prod) @ ea_dot_rad 
    om_deg = np.rad2deg(om_vec)
    return om_deg
def omega2dcmdot(rpy, w, deg):
    """compute DCM_rate from EA and angular rates

    Args:
        rpy (euler angles): same unit as deg
        w (rotation rate): same unit as deg
        deg (bool): 1 for deg, 0 for rad

    Returns:
        dcm_dot: DCM rate
    """    # convert Euler angles and rotation vector to DCM rate
    dcm = convert_ea2dcm(rpy, deg)
    if deg:
        w_used = np.deg2rad(w)
    else:
        w_used = w
    w_1, w_2, w_3 = w_used[0], w_used[1], w_used[2]
    Om = np.array([[0, -w_3, w_2],
                   [w_3, 0, -w_1],
                    [-w_2, w_1, 0]]) # Omega_mat
    dcm_dot = - Om @ dcm
    return dcm_dot

def calc_qdot(rpy, w, deg = 1, q = None, dcm = None):
    """compute Q_dot and Q from EA and rotation rates w

    Args:
        rpy (ea): euler angles, in deg or rad
        w (omega): in deg or rad
        deg (bool): 1-deg input, 0 - rad
    Returns:
        _type_: q, q_dot
    """
    if q is None:
        if type(dcm) == type(None):
            dcm = convert_ea2dcm(rpy, deg = deg)    


        C_11 = dcm[0,0]
        C_22 = dcm[1,1]
        C_33 = dcm[2,2]
        C_23 = dcm[1,2]
        C_32 = dcm[2,1]
        C_31 = dcm[2,0]
        C_13 = dcm[0,2]
        C_12 = dcm[0,1]
        C_21 = dcm[1,0]

        C_11 = dcm[0,0]
        C_22 = dcm[1,1]
        C_33 = dcm[2,2]
        C_23 = dcm[1,2]
        C_32 = dcm[2,1]
        C_31 = dcm[2,0]
        C_13 = dcm[0,2]
        C_12 = dcm[0,1]
        C_21 = dcm[1,0]

        q_scalar = 1/2 * np.sqrt(C_11 + C_22 + C_33 + 1)
        q_vec = 1 / (4*q_scalar) * np.array([[C_23 - C_32], [C_31 - C_13], [C_12 - C_21]])
        
        # compute via q;w -> q_dot for comparison
        q = np.vstack((q_scalar, q_vec))    
    else:
        q_scalar = q[0]        
        q_vec = q[1:]
    q_1, q_2, q_3 = q_vec[0], q_vec[1], q_vec[2]
    w_0, w_1, w_2 = w[0], w[1], w[2]
    w_x, w_y, w_z = np.deg2rad(w_0), np.deg2rad(w_1), np.deg2rad(w_2) # rad/s
    q_c = q_scalar
    # Edited May 16 2024, changed signs on q1;q2;q3dot terms
    # qdot_ham = 1/2 * np.array([- w_x*q_1 - w_y*q_2 - w_z*q_3,
    #                             w_x*q_c - w_z*q_2 + w_y*q_3 ,
    #                             w_y*q_c + w_z*q_1 - w_x*q_3,
    #                             w_z*q_c - w_y*q_1 + w_x*q_2])
    # Original, that's similar to np.gradient:
    qdot_ham = 1/2 * np.array([- w_x*q_1 - w_y*q_2 - w_z*q_3,
                                w_x*q_c + w_z*q_2 - w_y*q_3 ,
                                w_y*q_c - w_z*q_1 + w_x*q_3,
                                w_z*q_c + w_y*q_1 - w_x*q_2])
    # scalar-last quaternion and rate
    q_dot = qdot_ham
    return q, q_dot

def convert_quat2dcm(q, h_q = 1):
    """Quaternion to DCM conversion

    Args:
        q (4x1p): quaternion vector
        h_q (bool, optional): 1 for Hamiltonian. 0 for scalar-last. Defaults to 1.

    Returns:
        array: 3X3 DCM for DCM@v=v_out
    """    # convert quaternion to DCM. Scalar-first by default (h_q = 1)
    if h_q:
        q_1, q_2, q_3, q_w = q[1], q[2], q[3], q[0]
    else:
        q_1, q_2, q_3, q_w = q[3], q[0], q[1], q[2]
    C_11 = 1 - 2 * (q_2**2 + q_3**2)
    C_22 = 1 - 2 * (q_1**2 + q_3**2)
    C_33 = 1 - 2 * (q_1**2 + q_2**2)
    C_12 = 2 * (q_1 * q_2 + q_3*q_w)
    C_13 = 2 * (q_1 * q_3 - q_2*q_w)
    C_21 = 2 * (q_2 * q_1 - q_3*q_w)
    C_23 = 2 * (q_2 * q_3 + q_1*q_w)
    C_31 = 2 * (q_3 * q_1 + q_2*q_w)
    C_32 = 2 * (q_3 * q_2 - q_1*q_w)
    DCM = np.array([[C_11, C_12, C_13], 
                    [C_21, C_22, C_23],
                    [C_31, C_32, C_33]])
    return DCM

def convert_daz_to_qmo(az_correction_manual):
    """  converting azimuth correction to Quaternion
    # by assuming it is only a single rotation around the surface-normal direction
    # (around up). Thus, neglecting any elevation offset

    Args:
        az_correction (float): az correction in DEG as output by rasterscan

    Returns:
        np.array: q_mo scalar-first
    """    # example of converting azimuth correction to Quaternion
    # by assuming it is only a single rotation around the surface-normal direction
    # (around up). Thus, neglecting any elevation offset
    q_mo = convert_ea2quat(RPY = [0, 0, az_correction_manual], deg = 1, ham_q = 1)
    return q_mo.flatten()

if __name__ == '__main__':
    for angle_desired in [1, 2, 3, 4, 5]:
        axis_desired = [-1, 2, 3]
        axis_desired = axis_desired / np.linalg.norm(axis_desired)
        angle_desired_rad = np.deg2rad(angle_desired)
        rot_desired = convert_eigenaxis2dcm(axis_desired, angle_desired_rad)
        quat_desired = convert_dcm2quat(rot_desired)
        print(f'{angle_desired} Quaternion desired: {quat_desired}')
        # print(f'In euler angles: {convert_quat2ea(quat_desired)}')