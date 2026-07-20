# Tools for composite rotations
# specific for pointing angle computation
# for host/target satellites
import os
import sys
import pathlib
import numpy as np
import os, sys
sys.path.insert(1, os.getcwd()[:os.getcwd().index('astropynaric')+13])
import basic_tools.vector_operations as vec
import attitude_tools.rotations as rot
import attitude_tools.conversions as conv
import pointing_calculations.conversion_pointing as ae_conv
def calc_quat_eci2lct(r_h, v_h, default_pointing = 'along_track'):
    """Function to get a quaternion from ECI to RSW -> LCT frame with default pointing in the along-track
        calculates ROT_ECI2RSW using the position/velocity vector of host
        and ROT_RSW2LCT assuming a default LCT pointing orientation
        convert the rotation matrices to a scalar-first quaternion

    Args:
        r_h (N x 3 array): host ECI post [m,m,m]
        v_h (N x 3 array): hot ECI vel [m/s, m/s, m/s]
        default_pointing (str, optional): LCT orientation, only supports default input. 
            Defaults to 'along_track'.
            'along_track_up' - X pointing along track. Z pointing towards zenith
            'along_track_down' - X pointing along track. Z pointing towards nadir
            crosstrack_NW
            crosstrack_NE
    Returns:
        q_all, rot_eci2lct: quaternion and DCM from ECI to LCT for LCT pointing in along-track direction
    """    
    nrows = r_h.shape[0]
    
    # placeholders
    rot_eci2rsw = np.zeros((nrows, 3, 3))
    rot_eci2lct = np.zeros((nrows, 3, 3)) # debug/test purposes
    q_all = np.zeros((nrows, 4))
    
    # get LCT orientation
    if default_pointing == 'along_track':
        axes_rotlct = [1, 2]
        angles_rotlct = [90, 90]
    elif default_pointing == 'along_track_up':
        # axes_rotlct = [3, 1]
        # axes_rotlct = [1,3]
        axes_rotlct = [3,1]
        angles_rotlct = [90, 90]
    elif default_pointing == 'behind_track_up':
        # axes_rotlct = [3, 1]
        # axes_rotlct = [1,3]
        axes_rotlct = [3,1, 3]
        angles_rotlct = [90, 90, 180]
    elif default_pointing == 'cross_track_up':
        # axes_rotlct = [3, 1]
        # axes_rotlct = [1,3]
        axes_rotlct = [3,1, 3]
        angles_rotlct = [90, 90, 90]
    elif default_pointing == 'along_track_down':
        axes_rotlct = [3, 1]
        angles_rotlct = [90, 270]
    elif default_pointing == 'behind_track_down':
        axes_rotlct = [3, 1, 3]
        angles_rotlct = [90, 270, 180]
    # TODO cross-track stuff
    else:    
        print(f'Pointing of {default_pointing} not yet implemented')
    rot_rsw2lct = rotmat_rsw2lct(angles_rotlct, axes_rotlct)
    # get combined rotation matrix and convert to qutaernions
    for ii, (r_h, v_h) in enumerate(zip(r_h, v_h)):
        rot_eci2rsw[ii] = calc_rotrsweci(r_h, v_h)    
        rot_comb_ii = rot_rsw2lct @ rot_eci2rsw[ii]
        rot_eci2lct[ii] = rot_comb_ii
        q_ii = conv.convert_dcm2quat(rot_comb_ii)
        q_all[ii,:] = q_ii
    return q_all, rot_eci2lct

def calc_rotrsweci(r_h, v_h, option = 'rsw'):
    """function to calculate and return the rotation matrix
     from ECI to RSW at the host's position and velocity

    Args:
        r_h (_type_): host sat pos [m] in ECI
        v_h (_type_): host sat vel [m/s] in ECI
        option (str, optional): orientation. Defaults to 'rsw' (R - RADIAL, S -along-track, W - cross-track)
            swr also possible for X along velocity.

    Returns:
        ROT_RSWtoECI: DCM from ECI to RSW or SWR
    """    
    R = vec.norm_vector(r_h) # radial direction unit vector (point at zenith)    
    rv_cross = np.cross(r_h, v_h)
    W = vec.norm_vector(rv_cross) # cross-track component, orthogonal to velocity and Radial
    S = np.cross(W, R)
    ROT_ECI2RSW = np.zeros((3,3))
    if option == 'rsw':
        ROT_ECI2RSW[0,:] = R
        ROT_ECI2RSW[1,:] = S
        ROT_ECI2RSW[2,:] = W
    elif option == 'swr': 
        ROT_ECI2RSW[0,:] = S
        ROT_ECI2RSW[1,:] = W
        ROT_ECI2RSW[2,:] = R
    return ROT_ECI2RSW

def eci2aersw(r_eci, v_eci, los_eci):
    """function to calculate the Azimuth and Elevation [rad]
        for a given LOS vector in the ECI frame
        first calculates the ECI to RSW rotation matrix
        then rotates the LOS and converts it to azimuth/elevation [rad]

    Args:
        r_eci (array Nx3): pos of host satellite in ECI
        v_eci (array Nx3): vel of host satellite in ECI
        los_eci (array Nx3): LOS vector from host to target in ECI

    Returns:
        ae_rsw : 0 Az/El [rad]; 
        los_rsw : 1 LOS in RSW [m]
        rot_rsw: 2 Rotation matrices from ECI to RSW 
    """    
    # make placeholders
    nrows = np.shape(r_eci)[0]
    los_rsw = np.zeros((nrows, 3))
    ae_rsw = np.zeros((nrows, 2))
    rot_rsw = np.zeros((nrows, 3, 3))
    ii = 0
    for ii, (r, v, los) in enumerate(zip(r_eci, v_eci, los_eci)):
        rot_rsweci_ii = calc_rotrsweci(r, v)
        los_rsw_ii = np.matmul(rot_rsweci_ii, los)
        ae_rsw_ii = ae_conv.calc_aersw(los_rsw_ii)
        # store
        los_rsw[ii,:] = los_rsw_ii
        ae_rsw[ii,:] = ae_rsw_ii
        rot_rsw[ii,:] = rot_rsweci_ii
    return ae_rsw, los_rsw, rot_rsw

def translatecog2lctrsw(los_rsw, bus_height, bus_width, bus_length):
    """ translate the COG of the Spacecraft bus to four LCT's
    mounted on each corner of the nadir-facing surface
    first calculates the vectors from SC COG to each LCT
    then translates the given los to each LCT
    
    calculate lines of sight from COG to LCT's 1-4. 
    Assuming NADIR facing surface
    LCT1 in the 
    | Nadir direction (-R) /FLIGHT DIRECTION (ALONG-track) (S)
      LCT2    LCT3        -CROSS-track direction (W)
      LCT1    LCT4 
    | radial direction (R)
    Args:
        los_rsw (array): LOS [x,y,z]
        bus_height (float): SC bus height
        bus_width (float): SC bus width
        bus_length (float): SC bus length

    Returns:
        output: 4x3 array, each row represents LOS of LCT placed at every corner
        of the width x length 
    """    
    
    los_cog2lct1 = np.array([-bus_height/2, -bus_length/2, -bus_width/2])
    los_cog2lct2 = np.array([-bus_height/2, bus_length/2, -bus_width/2])
    los_cog2lct3 = np.array([-bus_height/2, bus_length/2, bus_width/2])
    los_cog2lct4 = np.array([-bus_height/2, -bus_length/2, bus_width/2])
    # transform LOS to LCTs' positions in RSW frame 
    # (moves host position to LCT position, while target remains consistent)
    los_lct1 = - los_cog2lct1 + los_rsw
    los_lct2 = - los_cog2lct2 + los_rsw
    los_lct3 = - los_cog2lct3 + los_rsw
    los_lct4 = - los_cog2lct4 + los_rsw
    # output array
    output = np.zeros((4,3))
    output[0,:] = los_lct1
    output[1,:] = los_lct2
    output[2,:] = los_lct3
    output[3,:] = los_lct4
    return output
def rotmat_rsw2lct(rot_angles:list, rot_axes:list):
    """Function to calculate the static rotation matrices
    from RSW to LCT frames

    Args:
        rot_axes (list): list of consecutive rotation axes 
        rot_angles (list): list of consecutive rotation angles (IN DEGREES)

    Returns:
        array: complete rotation matrix to be used r_LCT = ROT @ r_RSW
    """    
    rot_0 = np.eye(3) 
    for angle, ax in zip(rot_angles, rot_axes):
        rot_i = rot.rot_basic(angle, ax)
        rot_0 = rot_i @ rot_0
    return rot_0

def rsw2lct(los_rsw, sc_hlw =[3.4, 1, 1,], axes_rotlct = None, angles_rotlct = None, 
            lct_spot = 'leo'):    
    if lct_spot == 'leo': # settings for LCT's mounted on LEO satellite - 4 on each corner
        if type(axes_rotlct) == type(None):
            axes_rotlct = [[1,2],
                    [2],
                    [1, 2],
                    [1, 2]]
            angles_rotlct = [[-90, 90],
                [90],
                [90, 90],
                [180, 90]]
            n_lct = 4
        else:
            n_lct = 1
    else:
        print(f'LCT mounting for {lct_spot} not yet implemented')
    rot_all = np.zeros((n_lct,3,3))
    # calculate static rotation matrices from RSW to LCT
    for ii, (axes, angles) in enumerate(zip(axes_rotlct, angles_rotlct)):
        rot_ii = rotmat_rsw2lct(list(angles), list(axes))
        rot_all[ii] = rot_ii
    
    ## translate los from BF origin to terminal origin
    nrows = np.shape(los_rsw)[0]
    los_outputs = np.zeros((n_lct, nrows, 3)) # Terminal nr, nr_rows, xyz [m,m,m]
    ae_outputs = np.zeros((n_lct, nrows, 2)) # Terminal nr, nr_rows, az, el [rad, rad]
    if n_lct == 4:
        for ii, los in enumerate(los_rsw):
            los_lctrcw_all = translatecog2lctrsw(los, sc_hlw[0], sc_hlw[1], sc_hlw[2])
            for jj, los_lctrsw in enumerate(los_lctrcw_all):
                # los_lct = np.matmul(los_lctrsw, rot_all[jj])
                los_lct = rot_all[jj] @ los_lctrsw
                los_outputs[jj, ii, :] = los_lct        
                # compute azimuth and elevation
                az_lct, el_lct = ae_conv.calc_ae(los_lct)
                ae_outputs[jj, ii, :] = [az_lct, el_lct]
    elif n_lct == 1:
        jj = 0
        for ii, los_lctrsw in enumerate(los_rsw):
            los_lct = rot_all[jj] @ los_lctrsw
            los_outputs[jj, ii, :] = los_lct        
            # compute azimuth and elevation
            az_lct, el_lct = ae_conv.calc_ae(los_lct)
            ae_outputs[jj, ii, :] = [az_lct, el_lct]
        
    return los_outputs, ae_outputs, rot_all