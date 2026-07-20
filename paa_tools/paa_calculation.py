## script with a collection
# of pointing calculation tools- 
## ECI input ->BF Pointing Angle and PAA calculation
# ECI -> BF rotations using quaternions
# XYZ -> AzEl conversion using Mynaric conventions
# Quaternion rotations
# AzEl and PAA dAz dEl plots
# figure saving
import pandas as pd
import sys
import pathlib
import json
import importlib
import os
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import importlib
parent_dir = pathlib.Path(__file__).parent.parent.resolve()
os.chdir(parent_dir)
print(f'\nCWD : {os.getcwd()}\n')
sys.path.insert(1, os.getcwd())
import attitude_tools.rotations as rot
import pointing_calculations.conversion_pointing as pt_conv
import basic_tools.vector_operations as vec
from astronomy_tools.constants import c
# %config InlineBackend.print_figure_kwargs={'facecolor' : "w"}
def compute_azel_paa(states_host, states_target, attitude_host, 
                     t_vec = None, 
                     mounting_offset = np.array([1,0,0,0]), official_convention = 1):
    """combined function to compute LOS in global-frame and convert to AzEl, and compute PAA's in Az, El

    Args:
        states_host (array): host states. [t; r_xyz; v_xyz], [m, m/s] ECI
        states_target (array): target states. [t; r_xyz; v_xyz], [m, m/s] ECI
        if input, then first column must already be x position
        attitude_host (array): host t, attitude quat, quat rate, scalar-first. From ECI to Terminal Frame
        t_vec (array): time vector. If not input, assume states have t as first column
        official_convention (bool): 0 - use unofficial az/el convention. 1 : use official TODO mention source of AE convention
    Returns:
        pt_angles: los_lct; Az; El, PAA_Az; PAA_El [s, rad] in Terminal Frame
    """    
    if type(t_vec) == None:
        t_vec = states_host[:,0]
        r_h = states_host[:,[1,2,3]]
        v_h = states_host[:,[4,5,6]]
        r_t = states_target[:,[1,2,3]]
        v_t = states_target[:,[4,5,6]]
        quat_all = attitude_host[:,1:5]
    else:
        r_h = states_host[:,[1-1,1,2]]
        v_h = states_host[:,[4-1,4,5]]
        r_t = states_target[:,[1-1,1,2]]
        v_t = states_target[:,[4-1,4,5]]
        quat_all = attitude_host[:,:4]
    # LOS in ECI
    los = r_t - r_h
    slant = np.linalg.norm(los, axis = 1)
    # tangential host velocity wrt LOS
    dt_paa = slant / c
    dt_paa = dt_paa.reshape((quat_all.shape[0], 1))

    v_rel = v_t - v_h
    v_rel_tangential = vec.get_tangential_comp(v_rel, los) 

    ### target pos at tx and rx
    # target offset by relative tangential velocity
    r_t_rx_reltang = r_t - v_rel_tangential * dt_paa
    r_t_tx_reltang = r_t + v_rel_tangential * dt_paa

    los_rx_rel = r_t_rx_reltang - r_h
    los_tx_rel = r_t_tx_reltang - r_h
    
    # rotate using quaternions with unit-tested method
    los_rx_bf_tan = rot.rotate_all_quat(los_rx_rel, quat_all=quat_all)
    los_tx_bf_tan = rot.rotate_all_quat(los_tx_rel, quat_all=quat_all)
    los_bf = rot.rotate_all_quat(los, quat_all=quat_all)
    
    # rotation from bf to lct (assuming quat is given from ECI to LCT, so nothing done here)
    los_rx_lct_tan = rot.rotate_all_quat(los_rx_bf_tan, quat_all = mounting_offset)
    los_tx_lct_tan = rot.rotate_all_quat(los_tx_bf_tan, quat_all = mounting_offset)
    los_lct =  rot.rotate_all_quat(los_bf, quat_all = mounting_offset)

    # convert to Azimuth, elevation
    ae_rx_lct_tan = pt_conv.conv_los2ae(los_rx_lct_tan, official_convention)
    ae_tx_lct_tan = pt_conv.conv_los2ae(los_tx_lct_tan, official_convention)
    ae_lct = pt_conv.conv_los2ae(los_lct, official_convention)
    ## Robust computation of 2-way Point ahead angles [urad]
    PAA_full_vrel = (ae_tx_lct_tan - ae_rx_lct_tan) # urad # LCT-frame

    pt_angles = np.hstack((los_lct, ae_lct, slant.reshape((slant.shape[0], 1)), PAA_full_vrel))

    return pt_angles

def compute_azel_modular(states_host, states_target, attitude_host, 
                     t_vec = None,
                     rotation_function = 1, 
                     mounting_offset = np.array([1,0,0,0]), official_convention = 1):
    """combined function to compute LOS in global-frame and convert to AzEl, and compute PAA's in Az, El

    Args:
        states_host (array): host states. [t; r_xyz; v_xyz], [m, m/s] ECI
        states_target (array): target states. [t; r_xyz; v_xyz], [m, m/s] ECI
        if input, then first column must already be x position
        attitude_host (array): host t, attitude quat, quat rate, scalar-first. From ECI to Terminal Frame
        t_vec (array): time vector. If not input, assume states have t as first column
        official_convention (bool): 0 - use unofficial az/el convention. 1 : use official TODO mention source of AE convention
        rotation_function: 1 - use original, 2 - use swapped order
    Returns:
        pt_angles: los_lct; Az; El, PAA_Az; PAA_El [s, rad] in Terminal Frame
    """    
    if type(t_vec) == None:
        t_vec = states_host[:,0]
        r_h = states_host[:,[1,2,3]]
        v_h = states_host[:,[4,5,6]]
        r_t = states_target[:,[1,2,3]]
        v_t = states_target[:,[4,5,6]]
        quat_all = attitude_host[:,1:5]
    else:
        r_h = states_host[:,[1-1,1,2]]
        v_h = states_host[:,[4-1,4,5]]
        r_t = states_target[:,[1-1,1,2]]
        v_t = states_target[:,[4-1,4,5]]
        quat_all = attitude_host[:,:4]
    # LOS in ECI
    los = r_t - r_h
    slant = np.linalg.norm(los, axis = 1)
    # tangential host velocity wrt LOS
    dt_paa = slant / c
    dt_paa = dt_paa.reshape((quat_all.shape[0], 1))

    v_rel = v_t - v_h
    v_rel_tangential = vec.get_tangential_comp(v_rel, los) 

    ### target pos at tx and rx
    # target offset by relative tangential velocity
    r_t_rx_reltang = r_t - v_rel_tangential * dt_paa
    r_t_tx_reltang = r_t + v_rel_tangential * dt_paa

    los_rx_rel = r_t_rx_reltang - r_h
    los_tx_rel = r_t_tx_reltang - r_h
    
    if rotation_function == 1:
        rotation_function = rot.rotate_with_quat_mat
        # rotation_function = rot.rotate_with_quat
    elif rotation_function == 2:
        rotation_function = rot.rotate_with_quat_mat_swaperoo
    
    # rotate using quaternions with unit-tested method
    los_rx_bf_tan = rotation_function(los_rx_rel.flatten(), quat_all.flatten())
    los_tx_bf_tan = rotation_function(los_tx_rel.flatten(), quat_all.flatten())
    los_bf = rotation_function(los.flatten(), quat_all.flatten())
    
    # rotation from bf to lct (assuming quat is given from ECI to LCT, so nothing done here)
    los_rx_lct_tan = rotation_function(los_rx_bf_tan, mounting_offset)
    los_tx_lct_tan = rotation_function(los_tx_bf_tan, mounting_offset)
    los_lct =  rotation_function(los_bf, mounting_offset)

    # convert to Azimuth, elevation
    ae_rx_lct_tan = pt_conv.conv_los2ae(los_rx_lct_tan, official_convention)
    ae_tx_lct_tan = pt_conv.conv_los2ae(los_tx_lct_tan, official_convention)
    ae_lct = pt_conv.conv_los2ae(los_lct, official_convention)
    ## Robust computation of 2-way Point ahead angles [urad]
    PAA_full_vrel = (ae_tx_lct_tan - ae_rx_lct_tan) # urad # LCT-frame

    pt_angles = np.hstack((los_lct, ae_lct.flatten(), slant, PAA_full_vrel.flatten()))

    return pt_angles

def calc_paa_analytical(states_host, states_target, t_vec = None):
    """combined function to compute AzEl and PAA's in Az, El for

    Args:
        states_host (array): host states. [t; r_xyz; v_xyz], [m, m/s] ECI
        states_target (array): target states. [t; r_xyz; v_xyz], [m, m/s] ECI
        if input, then first column must already be x position
        attitude_host (array): host t, attitude quat, quat rate, scalar-first. From ECI to Terminal Frame
        t_vec (array): time vector. If not input, assume states have t as first column
        official_convention (bool): 0 - use unofficial az/el convention. 1 : use official TODO mention source of AE convention
    Returns:
        paa_1dim, paa_dot_relmotion: 1 dimensional PAA [urad]
    """    
    if type(t_vec) == None:
        t_vec = states_host[:,0]
        r_h = states_host[:,[1,2,3]]
        v_h = states_host[:,[4,5,6]]
        r_t = states_target[:,[1,2,3]]
        v_t = states_target[:,[4,5,6]]
    else:
        r_h = states_host[:,[1-1,1,2]]
        v_h = states_host[:,[4-1,4,5]]
        r_t = states_target[:,[1-1,1,2]]
        v_t = states_target[:,[4-1,4,5]]
    los = r_t - r_h
    v_rel = v_t - v_h
    v_rel_tangential = vec.get_tangential_comp(v_rel, los) 
    v_rel_t_norm = np.linalg.norm(v_rel_tangential, axis = 1)
    paa_1dim =  2 * v_rel_t_norm / c * 1e6 # urad

    slant = np.linalg.norm(los, axis = 1)
    # tangential host velocity wrt LOS
    dt_paa = slant / c
    dt_paa = dt_paa.reshape((states_host.shape[0], 1))
    r_t_rx_reltang = r_t - v_rel_tangential * dt_paa
    r_t_tx_reltang = r_t + v_rel_tangential * dt_paa

    los_rx_rel = r_t_rx_reltang - r_h
    los_tx_rel = r_t_tx_reltang - r_h
    cospaa_cos_relmotion = [np.dot(los_rx_rel[ii,:], los_tx_rel[ii,:]) / (np.linalg.norm(los_rx_rel[ii,:]) * np.linalg.norm(los_tx_rel[ii,:])) for ii in range(len(los_tx_rel))]
    paa_dot_relmotion = np.arccos(cospaa_cos_relmotion)*1e6
    return paa_1dim, paa_dot_relmotion