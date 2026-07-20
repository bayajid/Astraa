import json
import csv
import numpy as np
import pandas as pd
import importlib
import os
import sys
import pathlib
# resolve path issues to load custom astropynaric subfolders
sys.path.insert(1, os.getcwd()[:os.getcwd().index('astropynaric')+13])
import tudat_tools.data_processing.data_loading as load
import attitude_tools.conversions as conv
import attitude_tools.rotations as rot
import pointing_calculations.conversion_pointing as ae_conv
import attitude_tools.terminal_rotations as lct_rot
# import basic_tools.constants as const
import paa_tools.paa_calculation as paa_calc
import link_processing_tools.visibility_checks as vis_check
importlib.reload(load)
importlib.reload(vis_check)
# import plotting_functions.modular_plotting as modplot
import matplotlib.pyplot as plt
                
def calc_link_params(t_vec, states_host, states_target, 
                     default_ponting = 'along_track',
                     check_occultation = 1, limit_nr_links = True):
    ## Calculate link parameters for input host and target states
    # 
    # slice position/velocity
    pos_host, vel_host = states_host[:,:3], states_host[:,3:]
    pos_target, vel_target = states_target[:,:3], states_target[:,3:]
    nrows = np.shape(pos_host)[0]  
    # get rotation matrix from ECI to RSW (body-frame)
    rot_rsw_all = np.zeros((nrows, 3, 3))
    for ii, (r_h, v_h) in enumerate(zip(pos_host, vel_host)):
        rot_rsw_all[ii] = rot.calc_rotrsweci(r_h, v_h)    
    # rotate
    los_eci_all = pos_target - pos_host    
    los_rsw_all = np.zeros((nrows, 3)) 
    for jj, los_eci in enumerate(los_eci_all):
        # los_rsw_all[jj,:] = np.matmul(rot_rsw_all[jj], los_eci)
        los_rsw_all[jj,:] = rot_rsw_all[jj] @ los_eci
        if 0:
            # Debugging purposes
            q_eci2bf = conv.convert_dcm2quat(rot_rsw_all[jj])
            los_rsw_ii = rot.rotate_with_quat(los_eci, q_eci2bf, h_q = 1)
            print(f'LOS_RSW diff {np.linalg.norm(los_rsw_ii - los_rsw_all[jj,:]) :.1e} m')
            pass

    
    # rotate LOS from body-frame to LCT frame using a constant rot. matrix
    if default_ponting == 'along_track':
        axes_rotlct = [[1, 2]]
        angles_rotlct = [[90, 90]]        
    los_lct_all, ae_lct_all, rot_bf2lct_all = lct_rot.rsw2lct(los_rsw_all, axes_rotlct = axes_rotlct,
                                          angles_rotlct=angles_rotlct)
    # remove unnecessary dimension
    los_lct_all = los_lct_all[0]
    ae_lct_all = ae_lct_all[0]
    rot_bf2lct_all = rot_bf2lct_all[0]

    # compute ECI to LCT frame quaternion from combined direction cosine matrices
    q_all = np.zeros((nrows, 4))
    ## Compute using quaternion + ae conversion
    aer_quat_own = np.zeros((nrows, 3))
    
    for ii, rot_eci2bf in enumerate(rot_rsw_all):
        rot_comb_ii = rot_bf2lct_all @ rot_eci2bf
        q_ii = conv.convert_dcm2quat(rot_comb_ii)
        q_all[ii,:] = q_ii
        # q_eci2rsw = conv.convert_dcm2quat(rot_eci2bf)
        los_eci_ii = los_eci_all[ii,:]
        if 0:
            # debug purposes
            los_lct_from_dcm = rot_comb_ii @ los_eci_ii
            los_lct_from_quat = rot.rotate_with_quat(los_eci_ii, q_all[ii,:], h_q = 1)
            ae_from_quat = rot.calc_aelct(los_lct_from_quat)
            aer_quat_own[ii,:2] = ae_from_quat
            # print(f'LOS_LCT diff Q v DCM {np.linalg.norm(los_lct_from_dcm - los_lct_from_quat) :.1e} m. Q v fct {np.linalg.norm(los_lct_all[ii,:] - los_lct_from_quat) :.1e} m.')
            print(f'AE diff Q v full AZ {ae_from_quat[0] - ae_lct_all[ii,0] :.1e} rad. El {ae_from_quat[1] - ae_lct_all[ii,1] :.1e} rad.')

    # compute slant range and store
    r_all = np.linalg.norm(los_lct_all, axis = 1)
    aer = np.hstack((ae_lct_all, r_all.reshape(nrows, 1))) # Az, El, slant range [rad, rad, m]
    # compute az, el rates [rad/s] slant rate [m/s]    
    aer_dot = np.zeros(aer.shape)
    if nrows>1:
        for ii in range(3):
            aer_dot[:,ii] = np.gradient(aer[:,ii], t_vec)
    
    # calculate PAA, full method [urad]
    paa_outputs = paa_calc.compute_azel_paa(states_host, states_target, q_all, t_vec, official_convention = 1)[:,-2:]*1e6
    # calcuilate PAA analytical [urad]
    paa_analytical = paa_calc.calc_paa_analytical(t_vec, states_host, states_target)
    if 0:
        print('Differences (debug purposes)')
        print(f'Az : [{np.min(np.abs(paa_outputs[:,1] - aer[:,0])):.1e} : {np.max(np.abs(paa_outputs[:,1] - aer[:,0])):.1e}] rad')
        print(f'El : [{np.min(np.abs(paa_outputs[:,2] - aer[:,1])):.1e} : {np.max(np.abs(paa_outputs[:,2] - aer[:,1])):.1e}] rad')
    
    # check for occultation conditions and slice outputs
    if check_occultation:
        ii_visible = vis_check.check_occultation(states_host, states_target, limit_nr_links = limit_nr_links)
        aer = aer[ii_visible,:]
        aer_dot = aer_dot[ii_visible,:] 
        paa_outputs = paa_outputs[ii_visible,:] 
        paa_analytical = [paa_analytical[0][ii_visible], paa_analytical[1][ii_visible]]
        t_vec = t_vec[ii_visible] 
    return t_vec, aer, aer_dot, paa_outputs, paa_analytical


def get_sat_ind(data_raw,
    sim_parameters,
    t_lim = None,
    sat_names = None
    ):
    """Function to compute indices of position/velocity
    for each satellite in the provided sim_parameters

    Args:
        data_raw (array): raw const states
        sim_parameters (dict): 
        t_lim (int/float, optional): final time point to cutoff states. Defaults to None.

    Returns:
        vector: t_vec - time vector
        array: data_used - cut or all states w/o t_vec
        dict: indices_dict - satellite pos indices in state mat
    """    
    sats_all = sim_parameters['sat_names']
    if type(sat_names) == type(None):
        sat_names = sats_all
    t_vec_all = data_raw[:,0]
    data_used = data_raw[:,1:]
    t_vec = t_vec_all
    # make t_vec start from 0
    t_vec0 = t_vec[0]
    t_vec = np.array([t - t_vec0 for t in t_vec])
    if t_lim != None:
        data_used = data_raw[t_vec<=t_lim, 1:]
        t_vec = t_vec[t_vec<=t_lim]
    # Get indices of each satellites position and velocity vectors
    indices_dict = {} 
    for ii, sat_name in enumerate(sats_all): # get all satellite state indices
        jj = ii # skip time vector
        indices_dict[sat_name] = {}        
        indices_dict[sat_name]['ind_pos'] = list(range(jj*6,jj*6+3))
        indices_dict[sat_name]['ind_vel'] = list(range(jj*6+3,jj*6+6))
    
    return t_vec, data_used, indices_dict
def load_constellation_data(chosen_constellation = None, 
                            simulation_folder = None,
                            nrows = None,
                            subfolder = 'initial_constellation',
                            state_name = 'state_history.dat',
                            full_path = None):
    """Function to load the raw states and simulation parameters of a simulated constellation

    Args:
        chosen_constellation (str): name of constellation folder
        simulation_folder (str): path where the simulation_outputs are stored
        subfolder (str, optional): folder between the simulation_folder and chosen_constellation. Defaults to 'initial_constellation'.
        full_path (str, optional): path to state_history and sim_parameters
    Returns:
        array, dict: data_raw - raw state outputs, sim_parameters - dict of simulation parameters (incl sat names)
    """          
    if type(full_path) == type(None):
        full_path = f'{simulation_folder}/{subfolder}/{chosen_constellation}'
    
    path_state_raw = f'{full_path}/{state_name}'
    path_parameters_raw = f'{full_path}/simulation_parameters.json'
    data_raw, sim_parameters = load.open_dat(base_path = full_path, nrows = nrows, 
                                             fname_states = state_name)
    # with open(path_parameters_raw, 'r') as j:
    #     sim_parameters = json.load(j)
    return data_raw, sim_parameters

def calculate_link_parameters(data_raw,
    sim_parameters,
    sat_host,
    t_lim = None,
    sat_names = None
    ):
    """Function to compute rel. pos. and Azimuth/Elevation between sat_host and all target sats
    additionally limits the raw_state dataset to t_lim (if specified)
    and provides indices for each satellites pos/vel in the raw states

    Args:
        data_raw (array): raw const states
        sat_host (str): host satellite name
        t_lim (int/float, optional): final time point to cutoff states. Defaults to None.

    Returns:
        array: data_used - cut or all states w/o t_vec
        vector: t_vec - time vector
        dict: indices_dict - satellite pos indices in state mat
        dict: los_dict - dictionary with LOS to all target sats
    """    
    sats_all = sim_parameters['sat_names']
    if type(sat_names) == type(None):
        sat_names = sats_all
    t_vec_all = data_raw[:,0]
    data_used = data_raw[:,1:]
    t_vec = t_vec_all
    # make t_vec start from 0
    t_vec0 = t_vec[0]
    t_vec = np.array([t - t_vec0 for t in t_vec])
    if t_lim != None:
        data_used = data_raw[t_vec<=t_lim, 1:]
        t_vec = t_vec[t_vec<=t_lim]
    # Get indices of each satellites position and velocity vectors
    indices_dict = {} 
    for ii, sat_name in enumerate(sats_all): # get all satellite state indices
        jj = ii # skip time vector
        indices_dict[sat_name] = {}        
        indices_dict[sat_name]['ind_pos'] = list(range(jj*6,jj*6+3))
        indices_dict[sat_name]['ind_vel'] = list(range(jj*6+3,jj*6+6))
    
    los_dict = {}
    pos_host = data_used[:,indices_dict[sat_host]['ind_pos']]
    vel_host = data_used[:,indices_dict[sat_host]['ind_vel']]
    los_dict[sat_host] = {}
    los_dict[sat_host]['pos_h'] = pos_host
    
    # create rotation matrices from ECI to RSW frame (only dependent on host pos and vel)
    nrows = np.shape(pos_host)[0]    
    rot_rsw = np.zeros((nrows, 3, 3))
    for ii, (r_h, v_h) in enumerate(zip(pos_host, vel_host)):
        rot_rsw[ii] = rot.calc_rotrsweci(r_h, v_h)
    
    for ii, sat_name in enumerate(sat_names): # loop over all possible target satellites
        if sat_name != sat_host: # found target
            los_dict[sat_host][sat_name] = {}
            ind_target = indices_dict[sat_name]['ind_pos']
            pos_target = data_used[:,ind_target]
            pos_rel = pos_target - pos_host
            # Get link distance
            slant_range = np.array([np.sqrt(los[0]**2 + los[1]**2 + los[2]**2) for los in pos_rel])
            ## get azimuth/elevation
            # RSW frame
            los_rsw = np.zeros((nrows, 3)) 
            for jj, los_eci in enumerate(pos_rel):
                rot_rsw_jj = rot_rsw[jj]
                los_rsw[jj,:] = np.matmul(rot_rsw_jj, los_eci)
            # LCT frames
            los_lct_all, ae_lct_all = rot.rsw2lct(los_rsw)
            # Store data
            los_dict[sat_host][sat_name]['rel_pos'] = pos_rel
            los_dict[sat_host][sat_name]['pos_t'] = pos_target
            los_dict[sat_host][sat_name]['slant_range'] = slant_range    
            los_dict[sat_host][sat_name]['los_rsw'] = los_rsw
            for jj in range(np.shape(los_lct_all)[0]):
                los_dict[sat_host][sat_name][f'lct{jj+1}'] = {}
                los_dict[sat_host][sat_name][f'lct{jj+1}']['ae'] = ae_lct_all[jj]
                los_dict[sat_host][sat_name][f'lct{jj+1}']['los'] = los_lct_all[jj]
    return data_used, t_vec, indices_dict, los_dict