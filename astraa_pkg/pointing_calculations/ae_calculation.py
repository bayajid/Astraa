## Tools to compute azimuth/elevation in the Global Terminal Frame
# from states/attitude in Earth centerel inertial to the terminal-carried 
# global frame
import os
import sys
import pathlib
import numpy as np
import os, sys
sys.path.insert(1, os.getcwd()[:os.getcwd().index('astropynaric')+13])
import attitude_tools.terminal_rotations as lct_rot
import attitude_tools.conversions as conv
import attitude_tools.rotations as rot
import pointing_calculations.conversion_pointing as ae_conv
import link_processing_tools.visibility_checks as vis_check

def xyz_from_aer(aer, deg = 1):
    """AER to xyz
    NEEDS NONZERO Azimuth
    Args:
        aer (_type_): in deg, m
    """    
    if deg:
        az = np.deg2rad(aer[0])
        el = np.deg2rad(aer[1])
    else:
        az = aer[0]
        el = aer[1]
    r = aer[2]
    # get z
    z = np.sqrt(r**2 / (1 + 1 / np.tan(el)**2 ))
    z = z * np.sign(el)
    # get x 
    x = np.sqrt((r **2 - z **2) / (1 + np.tan(az)**2))
    az_deg = aer[0]
    # get quadrant
    if az_deg>=0:
        if az_deg <= 90:
            sign_x = 1
        else:
            sign_x = -1
    elif az_deg<0:
        if az_deg >= -90:
            sign_x = 1
        else:
            sign_x = -1
    x = sign_x * x
    y = np.tan(az) * x
    return np.array([x, y, z])
        
def calc_ae_full(states_host, states_target, attitude_eci2bf = None,                 
                    attitude_mountingoffset = None,
                     default_offset = 'none',
                     R_atm = None,
                     centroid_error = None,
                     centroid_dirction_randomizer = 1,
                     seed_chosen = 1,
                     output_interm_los = 0,
                     rotation_function = 1,
                     check_occultation = 0):    
    """Function to compute azimuth and elevation in global frame
    from ECI host/target positions and attitude in quaternions
    if no attitude is provided, attitude quaternions are genreated
    where the global frame's X will be aligned with the satellite's
    flight direction

    Args:
        states_host (array): host states r, v [m, m/s]
        states_target (array): target states r, v [m, m/s]
        attitude_eci2bf (_type_, optional): quaternions and rates (scalar-first) from
            ECI to body-frame
        default_ponting (str, optional): Attitude mode used is no attitude is input.
             Defaults to 'up_along' or 'down_along'.
        check_occultation (int, optional): [Astro-related] Bool to check if occultation occurs
        and slice out occulted LOS. Defaults to 0.
        centroid_error (float): [Scan-related] DCM Rotation for centroid detection error. Default to None
        centroid_dirction_randomizer (bool): [Scan-related] whether direction of Centroid Det Error shall be randomized
            around normal LOS plane
        output_interm_los (bool): whether cartesian LOS in ECI, BF and LCT shall be output
            1:3 AER. 4:6 LOS_ECI. 7:9 LOS_BF, 10:12 LOS_LCT
        rotation_function (bool, defaults to 1): which rotation function is used to apply
            quatenrion rotations. 2 Sets to OIT-times rotation, 1 uses updated/right option
    Returns:
        array: aer_lct - az, el [rad], slant-range [m]
    """    
    if rotation_function == 1:
        rotation_function = rot.rotate_with_quat_mat
        # rotation_function = rot.rotate_with_quat
    elif rotation_function == 2:
        rotation_function = rot.rotate_with_quat_mat_swaperoo
    if len(states_host.shape) == 1: # Reshape if singel row in input
        states_host = states_host.reshape((1, states_host.shape[0]))
        states_target = states_target.reshape((1, states_target.shape[0]))
        attitude_eci2bf = attitude_eci2bf.reshape((1, attitude_eci2bf.shape[0]))

    pos_host, vel_host = states_host[:,:3], states_host[:,3:]
    pos_target, vel_target = states_target[:,:3], states_target[:,3:]
    if check_occultation:
        ## Check for Earth occultation
        ii_vis = vis_check.check_occultation(states_host, states_target, R_atm=R_atm)
        pos_host = pos_host[ii_vis,:]
        vel_host = vel_host[ii_vis,:]
        pos_target = pos_target[ii_vis,:]
        vel_target = vel_target[ii_vis,:]
    nrows = np.shape(pos_host)[0]  

    # LOS in ECI
    los_eci = pos_target - pos_host
    slant_range = np.linalg.norm(los_eci, axis = 1)
    
    los_bf = np.zeros(los_eci.shape)
    los_lct = np.zeros(los_eci.shape)
    

    if len(slant_range) == 1: # Reshape if singel row in input
        slant_range = slant_range.reshape((1, slant_range.shape[0]))
    elif len(slant_range.shape) == 1: # gove 2nd dimension for many rows
        slant_range = slant_range.reshape((slant_range.shape[0], 1))
    if type(attitude_eci2bf) == type(None):
        # get rotation matrix from ECI to RSW (body-frame)
        rot_rsw_all = np.zeros((nrows, 3, 3))
        aer_quat_own = np.zeros((nrows, 3))
        q_eci2bf = np.zeros((nrows, 4))
        if default_offset == 'up_along':
            axes_rotlct = [1, 2]
            angles_rotlct = [90, 90]     
            # rotation from RSW to Global frame (constant)
            rot_rsw2lct = lct_rot.rotmat_rsw2lct(angles_rotlct, axes_rotlct)
        elif default_offset == 'down_along':
            axes_rotlct = [2,3,1]
            angles_rotlct = [180, 90, 90]
            # rotation from RSW to Global frame (constant)
            rot_rsw2lct = lct_rot.rotmat_rsw2lct(angles_rotlct, axes_rotlct)
        elif default_offset == 'none': # unity rotation
            rot_rsw2lct = np.eye(3).astype(float)

        for ii, (r_h, v_h) in enumerate(zip(pos_host, vel_host)):
            # rotation from ECI to RSW
            rot_rsw_all[ii] = lct_rot.calc_rotrsweci(r_h, v_h)                            
            rot_comb_ii = rot_rsw2lct @ rot_rsw_all[ii]
            q_ii = conv.convert_dcm2quat(rot_comb_ii)
            q_eci2bf[ii,:] = q_ii
    else:
        try:            
            q_eci2bf = attitude_eci2bf[:,:4]
        except:
            q_eci2bf = attitude_eci2bf[:4]
        if check_occultation: # if attitude was auto-generated
            # it already matches visibility sliced states
            q_eci2bf = q_eci2bf[ii_vis,:]
    for ii, los_ii in enumerate(los_eci):
        try:
            los_bf_ii = rotation_function(los_ii, q_eci2bf[ii,:])
        except:
            los_bf_ii = rotation_function(los_ii, q_eci2bf)
        los_bf[ii,:] = los_bf_ii
    # rotation_function(los_bf, q_bf2gf)
    ## Set up mounting offset
    if type(attitude_mountingoffset) == type(None):
        q_bf2gf = np.zeros((los_bf.shape[0],4))
        q_bf2gf[:,:4] = [1, 0, 0, 0] # unity rotation
    else:
        q_bf2gf = np.zeros((los_bf.shape[0],4))
        q_bf2gf[:,:4] = attitude_mountingoffset
    
    for ii, los_ii in enumerate(los_bf):
        los_lct_ii = rotation_function(los_ii, q_bf2gf[ii,:])
        los_lct[ii,:] = los_lct_ii
    # rotation_function(los_bf, q_bf2gf)
    # Add centrid detection rotation error [ONLY FOR SIMULATED MOON/SUN-SCANS]
    if type(centroid_error) != type(None):
        # Randomly select direction of rotation axis within normal LOS plane
        centroid_err_axis = np.cross([0,1,0], los_lct)
        np.random.seed(seed_chosen)
        # get rotation matrix for the rotation axis 
        if len(centroid_err_axis.shape) != 1:
            for ii, los_lct_ii in enumerate(los_lct):
                if centroid_dirction_randomizer:
                    angle_rotation = np.random.random()*2*np.pi
                    rot_axis_rotation = conv.convert_eigenaxis2dcm(los_lct_ii, angle_rotation)            
                else:
                    rot_axis_rotation = np.eye(3, dtype = float)
                dcm_centroid_err = conv.convert_eigenaxis2dcm(rot_axis_rotation @ centroid_err_axis[ii,:].reshape([3,1]), centroid_error) # [ii]
                los_lct_ii = dcm_centroid_err @ los_lct_ii.reshape([3,1])
                los_lct_ii = los_lct_ii.reshape([1,3])
                los_lct[ii,:] = los_lct_ii
        else:
            if centroid_dirction_randomizer:
                angle_rotation = np.random.random()*2*np.pi
                rot_axis_rotation = conv.convert_eigenaxis2dcm(los_lct, angle_rotation)            
            else:
                rot_axis_rotation = np.eye(3, dtype = float)

            # centroid_err_axis = centroid_err_axis / np.linalg.norm(centroid_err_axis)
            dcm_centroid_err = conv.convert_eigenaxis2dcm(rot_axis_rotation @ centroid_err_axis.reshape([3,1]), centroid_error)
            los_lct = dcm_centroid_err @ los_lct.reshape([3,1])
            los_lct = los_lct.reshape([1,3])

    ae_lct = ae_conv.conv_los2ae(los_lct, official_convention = 1, wrap = 1)
    aer_lct = np.hstack((ae_lct, slant_range))
    if output_interm_los:
        aer_lct = np.hstack((aer_lct, los_eci, los_bf, los_lct))
        if aer_lct.shape[0] == 1:
            aer_lct = aer_lct.flatten()
    return aer_lct