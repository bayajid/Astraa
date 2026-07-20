## Necessary tools and usage example for the Mounting Offset Quaternion Resolution
# Using 2 sets of Expected (Commanded) Azimuth/Elevation angles in the Terminal Frame
# and 2 sets of Measured (Detected) Azimuth/Elevation angles in the Terminal Frame
# in addition to the Mounting Offset Quaternion that was commanded during the 
# the current step of the calibration procedure.
# Date November 2, 2023
# Author: Kipras Paliusis

import numpy as np
def convert_dcm_to_q(dcm):
    """Convert DCM to a Hamiltonian quaternion

    Args:
        DCM (3x3 array): Rotation matrix, in format r_rotated = DCM @ r_initial
    Returns:
        q_out (4x, vector): quaternion
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

    q_vec = 1 / (4*q_scalar) * np.array([[C_23 - C_32], [C_31 - C_13], [C_12 - C_21]])
    q_out = np.vstack((q_scalar, q_vec))
    q_out = q_out.flatten()

    return q_out
def convert_polar_to_cartesian(pointing_angles):
    """Azimuth and Elevation to Cartesian coordinate conversion
    Non-zero Azimuth values required
    Length of vector assumed to be 1
    Args:
        pointing_angles (vector, 2x1): Azimuth [rad]; Elevation [rad]
    Outputs:
        vector_cartesian (vector 3x,): Pointing angles converted to cartesian components
    """    
    az = pointing_angles[0]
    el = pointing_angles[1]    
    r = 1
    
    z = np.sqrt(r**2 / (1 + 1 / np.tan(el)**2 )) * np.sign(el)

    # get quadrant for x
    if az>0:
        if az < np.pi/2:
            sign_x = 1
        else:
            sign_x = -1
    elif az<0:
        if az > -np.pi/2:
            sign_x = 1
        else:
            sign_x = -1    
    x = sign_x * np.sqrt((r **2 - z **2) / (1 + np.tan(az)**2))

    y = np.tan(az) * x

    # Store components into vector
    vector_cartesian = np.array([x, y, z])
    return vector_cartesian

def triad(vecs_A, vecs_B):
    """TRIAD algorithm to return rotation from frame A to frame B

    Args:
        vecs_A (array, 3x2): set of vectors in frame A
        vecs_B (array, 3x2): set of vectors in frame B        

    Returns:
        DCM_resolved (array, 3x3): rotation matrix from frame A to frame B
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

def compute_q_product(q_1, q_2):
    """Perform hamiltonian quaternion multiplication
    
    Args:
        q_1 (4x1 vector): first quaternion
        q_2 (4x1 vector): second quaternion

    Returns:
        q_comp (4x1 vector): resulting composite quaternion
    """        

    a1 = q_2[0]  
    b1 = q_2[1]
    c1 = q_2[2]
    d1 = q_2[3]

    a2 = q_1[0]
    b2 = q_1[1]
    c2 = q_1[2]
    d2 = q_1[3]

    q_comp = np.array([[a1*a2 - b1*b2 - c1*c2 - d1*d2],
                       [a1*b2 + b1*a2 + c1*d2 - d1*c2],
                       [a1*c2 - b1*d2 + c1*a2 + d1*b2],
                       [a1*d2 + b1*c2 - c1*b2 + d1*a2]
                       ])
    return q_comp
def compute_q_mountingoffset(az_el_expected,
                             az_el_current,
                             q_mountingoffset_precal):
    """Combined functions to resolve the mounting offset quaternion
    using inputs of commanded and measured azimuth and elevation angles 
    in the terminal frame. 
    Returned mounting offset quaternion is the composite rotation
    of the previously known mounting offset and the computed calibration
    correction.

    Args:
        az_el_current (array, 2x2): Commanded azimuth and elevation angles [rad]
        az_el_expected (array, 2x2): Measured azimuth and elevation angles [rad]
        q_mountingoffset_precal (vector, 4x): Currently known mounting offset quaternion

    Returns:
        q_from_bf_to_tf (vector, 4x): Calibrated mounting offset quaternion,
            describing the rotation from the spacecraft body to the terminal-frame.
    """    
    ## Convert Pointing Angles to normalized cartesian LOS
    los_current_1 = convert_polar_to_cartesian(az_el_current[0,:])
    los_current_2 = convert_polar_to_cartesian(az_el_current[1,:])

    los_expected_1 = convert_polar_to_cartesian(az_el_expected[0,:])
    los_expected_2 = convert_polar_to_cartesian(az_el_expected[1,:])

    los_set_current = np.vstack((los_current_1, los_current_2))
    los_set_expected = np.vstack((los_expected_1, los_expected_2))

    dcm_from_precal_to_cal = triad(los_set_expected, los_set_current)
    q_from_precal_to_cal = convert_dcm_to_q(dcm_from_precal_to_cal)

    ## Multiply resolved quaternion with initially known MO quaternion
    ## for complete composite rotation from S/C body-frame to Terminal Frame
    q_from_bf_to_tf = compute_q_product(q_from_precal_to_cal, 
                                        q_mountingoffset_precal)
    q_from_bf_to_tf = q_from_bf_to_tf.flatten()
    
    return q_from_bf_to_tf
    

if __name__ == '__main__':
    from scipy.spatial.transform import Rotation as R
    ### Example inputs
    # az_el_expected = np.array([[ 1.14629288,  0.02202081],
    #    [ 2.36716341, -0.02970502]]) # Az;El commanded 1 and Az;El commanded 2[rad]
    # az_el_current = np.array([[ 1.14856302,  0.02064609],
    #    [ 2.36948316, -0.02895276]]) # Az;El tracked 1 and Az;El tracked 2[rad]

    # Currently known pre-calibration Mounting Offset
    q_mountingoffset_precal = np.array([ 0.70730741,  0.70574091,  0.03980937, -0.00782328])
    
    # Unknown True Mounting Offset (Hamiltonian)
    q_mountingoffset_true = np.array([ 0.7070462 ,  0.70596952,  0.04008785, -0.00925574])

    # q_resolved = compute_q_mountingoffset(az_el_expected = az_el_expected,
    #                                       az_el_current = az_el_current,
    #                                       q_mountingoffset_precal = q_mountingoffset_precal)

        
    # print(f'''
    #     Pre-calibration Quaternion: {*np.round(q_mountingoffset_precal,5),}
    #     Post-calibration Quaternion:{*np.round(q_resolved,5),}
    # ''')


    # Define threshold for a "good" measurement (optional)
    threshold = 1e-3  # 1 mrad

    # print("="*60)
    # print('Using the MO Calibrated quaternion')
    # print("="*60)
    # r1 = R.from_quat(q_resolved)
    # r2 = R.from_quat(q_mountingoffset_true)

    # # Compute the relative rotation (error quaternion)
    # relative_rotation = r2.inv() * r1

    # # Get the axis and angle of the relative rotation
    # axis = relative_rotation.as_rotvec()  # rotation vector (axis * angle)
    # angle = np.linalg.norm(axis)  # the magnitude of the rotation vector gives the angle
    # print(f"Axis of rotation: {axis / angle if angle != 0 else axis}")
    # print(f"Rotation angle (in m-radians): {angle*1e3}")
    # # print(f"Rotation angle (in degrees): {np.degrees(angle)}")

    # if angle < threshold:
    #     print("The quaternion is very close to the true quaternion.")
    # else:
    #     print("The quaternion has a significant difference from the true quaternion.")


    # print("="*60)
    # print('Using the precalibrated quaternion')
    # print("="*60)
    # r1 = R.from_quat(q_mountingoffset_precal)
    # r2 = R.from_quat(q_mountingoffset_true)

    # # Compute the relative rotation (error quaternion)
    # relative_rotation = r2.inv() * r1

    # # Get the axis and angle of the relative rotation
    # axis = relative_rotation.as_rotvec()  # rotation vector (axis * angle)
    # angle = np.linalg.norm(axis)  # the magnitude of the rotation vector gives the angle
    # print(f"Axis of rotation: {axis / angle if angle != 0 else axis}")
    # print(f"Rotation angle (in m-radians): {angle*1e3}")

    # if angle < threshold:
    #     print("The quaternion is very close to the true quaternion.")
    # else:
    #     print("The quaternion has a significant difference from the true quaternion.")


    print("="*60)
    print("Measured LOS from RID-21")
    print("="*60)    
    # # Predicted LOS from RID-21 
    # # OH1_PRED_UNIT_VECTOR_X, OH1_PRED_UNIT_VECTOR_Y, OH1_PRED_UNIT_VECTOR_Z     
    los_set_pred =np.array([[0.41176849,  0.91102243,  0.02201903],
                            [-0.71450493,  0.69899969, -0.02970065]])
    # # OH1_LOS_UNIT_VECTOR_X,OH1_LOS_UNIT_VECTOR_Y,OH1_LOS_UNIT_VECTOR_Z 
    los_set_meas =np.array([[ 0.4097113 ,  0.9119816 ,  0.02064462],
                            [-0.71614032,  0.69735573, -0.02894872]]) 

    print(f"\nlos_set_meas_LOS:\n\n{los_set_meas}") 
    print(f"\nlos_set_pred_LOS:\n\n{los_set_pred}")    

    dcm_from_precal_to_cal = triad (los_set_pred, los_set_meas) 
    q_from_precal_to_cal = convert_dcm_to_q(dcm_from_precal_to_cal) 

    ## Multiply resolved quaternion with initially known MO quaternion 
    ## for complete composite rotation from S/C body-frame to Terminal Frame 
    q_from_bf_to_tf = compute_q_product(q_from_precal_to_cal, q_mountingoffset_precal) 
    q_from_bf_to_tf = q_from_bf_to_tf/np.linalg.norm(q_from_bf_to_tf) 
    q_from_bf_to_tf = q_from_bf_to_tf.flatten() 

    print(f"\nCorrected mounting offset quaternion: {q_from_bf_to_tf}")    
    print (f'''
        TRIAD Pre-calibration Quaternion Error: 
        {q_mountingoffset_true - q_mountingoffset_precal} 

        Post-calibration Error: 
        {q_mountingoffset_true - q_from_bf_to_tf} 

    ''') 

    print("="*60)
    print('Using the MO Calibrated quaternion')
    print("="*60)
    r1 = R.from_quat(q_from_bf_to_tf)
    r2 = R.from_quat(q_mountingoffset_true)

    # Compute the relative rotation (error quaternion)
    relative_rotation = r2.inv() * r1

    # Get the axis and angle of the relative rotation
    axis = relative_rotation.as_rotvec()  # rotation vector (axis * angle)
    angle = np.linalg.norm(axis)  # the magnitude of the rotation vector gives the angle
    print(f"Axis of rotation: {axis / angle if angle != 0 else axis}")
    print(f"Rotation angle (in m-radians): {angle*1e3}")
    # print(f"Rotation angle (in degrees): {np.degrees(angle)}")

    if angle < threshold:
        print("The quaternion is very close to the true quaternion.")
    else:
        print("The quaternion has a significant difference from the true quaternion.")


    print("="*60)
    print('Using the precalibrated quaternion')
    print("="*60)
    r1 = R.from_quat(q_mountingoffset_precal)
    r2 = R.from_quat(q_mountingoffset_true)

    # Compute the relative rotation (error quaternion)
    relative_rotation = r2.inv() * r1

    # Get the axis and angle of the relative rotation
    axis = relative_rotation.as_rotvec()  # rotation vector (axis * angle)
    angle = np.linalg.norm(axis)  # the magnitude of the rotation vector gives the angle
    print(f"Axis of rotation: {axis / angle if angle != 0 else axis}")
    print(f"Rotation angle (in m-radians): {angle*1e3}")

    if angle < threshold:
        print("The quaternion is very close to the true quaternion.")
    else:
        print("The quaternion has a significant difference from the true quaternion.")

