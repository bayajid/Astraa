import numpy as np
import os, sys
sys.path.insert(1, os.getcwd()[:os.getcwd().index('astropynaric')+13])

# sys.path.insert(1, os.getcwd()[:os.getcwd().index('astropynaric')+13])
import attitude_tools.conversions as conv
def get_pe_for_rot(rot_1, rot_2, rot = 'q', vector_samples = None, print_cond = 0):
    """    # function to compute angle
    # between the same vectors rotated with two rotation matrices

    Args:
        rot_1 (mat/vec): rotation one
        rorot_2t2 (mat/vec): rotation two
        rot (str, optional): 'q, ea or dcm', rotation format
            EA in deg
            quat - hamiltonian
    Returns:
        PE values in [urad] for vectors
    """    
    if rot == 'q':
        dcm_1 = conv.convert_quat2dcm(rot_1)
        dcm_2 = conv.convert_quat2dcm(rot_2)
    elif rot == 'ea':
        dcm_1 = conv.convert_ea2dcm(rot_1)
        dcm_2 = conv.convert_ea2dcm(rot_2)
    elif rot == 'dcm':
        dcm_1 = rot_1
        dcm_2 = rot_2

    if type(vector_samples) == type(None):
        vector_samples = np.array([[215, 15, 45],
                                    [-40, -215, -15],
                                    [15, -50, -215]])
        vector_samples = np.vstack((vector_samples, -vector_samples))
    
    nrows = vector_samples.shape[0]
    PE_storage = np.zeros((nrows,1))

    for ii, vec in enumerate(vector_samples):
        vec_rotated_1 = dcm_1 @ vec
        vec_rotated_2 = dcm_2 @ vec
        pe = calc_dot_angle(vec_rotated_1, vec_rotated_2)*1e6
        PE_storage[ii] = pe
    if print_cond:
        print(f'PE max = {np.max(pe)} urad. Rest : {*pe,} urad')
    return PE_storage

def norm_vector(r):
    """normalize a vec

    Args:
        r (_type_): VECTOR

    Returns:
        _type_: _description_
    """    # function to normalize a vector, making its length = 1
    norm_r = np.linalg.norm(r, ord = 2)
    r_normalized = r / norm_r
    return r_normalized

def get_tangential_comp(v1, v2):
    # function to compute the tangential projection of vector v1 onto vector v2
    # example use would be to input v1 = v_rel, v2 = LOS -> v_tang = v_rel, tangential to LOS
    v1_tang = np.zeros(v1.shape)
    for ii, v1_ii in enumerate(v1):
        v2_ii = v2[ii,:]
        # compute normal projection
        v1_norm_ii = np.dot(v1_ii, v2_ii/ np.linalg.norm(v2_ii)) * v2_ii / np.linalg.norm(v2_ii)
        # subtract vectors for tangential projection
        v1_tang_ii = v1_ii - v1_norm_ii
        v1_tang[ii,:] = v1_tang_ii
    return v1_tang
# def calc_dot_angle(vec_1, vec_2):
#     """function to compute angle between two vectors

#     Args:
#         vec_1 : first vector
#         vec_2 : second vector

#     Returns:
#         angle: angle [rad] between vectors
#     """    # function to calculate the angle between two vectors
#     # returns angle in rad
#     len_vec1 = np.linalg.norm(vec_1, ord = 2)
#     len_vec2 = np.linalg.norm(vec_2, ord = 2)
#     angle = np.arccos(np.dot(vec_1, vec_2)/(len_vec1 * len_vec2))
#     if angle != angle:
#         # return in case nan
#         angle = 0
#     return angle
def calc_dot_angle(vec_1, vec_2, setting = 'cartesian', deg = 0):
    """function to compute angle between two vectors

    Args:
        vec_1 : first vector
        vec_2 : second vector

    Returns:
        angle: angle [rad] between vectors
    """    # function to calculate the angle between two vectors
    # returns angle in rad
    if setting == 'polar':
        if not deg:
            vec_1 = convert_polar_to_cartesian(vec_1)
            vec_2 = convert_polar_to_cartesian(vec_2)
        else:
            vec_1 = convert_polar_to_cartesian(np.deg2rad(vec_1))
            vec_2 = convert_polar_to_cartesian(np.deg2rad(vec_2))
    len_vec1 = np.linalg.norm(vec_1, ord = 2)
    len_vec2 = np.linalg.norm(vec_2, ord = 2)
    dot = np.dot(vec_1, vec_2)/(len_vec1 * len_vec2)
    dot = np.clip(dot, -1.0, 1.0)  # Prevent invalid values for arccos
    angle = np.arccos(dot) # rad
    if angle != angle:
        # return in case nan
        angle = 0    
    return angle
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
    if az>=0:
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
