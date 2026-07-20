# Code to produce switching of AER, dTA, etc from host to targets
# at boundary seam
# August 23

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os 
import datetime
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import importlib
import os, sys
import json
sys.path.insert(1, os.getcwd()[:os.getcwd().index('astropynaric')+13])

## MVP imports
import basic_tools.vector_operations as vec_calc
import astronomy_tools.constants as const
import astronomy_tools.element_conversion as el_conv
import astronomy_tools.astro_targets as where_sun
import astronomy_tools.analytical_tools as astro_calc
import plotting_tools.basic_plotting as bplt
import plotting_tools.combined_plots as combplt
import plotting_tools.modular_plotting as modplot
import attitude_tools.terminal_rotations as lct_rot
import pointing_calculations.ae_calculation as ae_calc
import basic_tools.time_conversion as t_conv
import tudat_tools.tudat_converter as tudatconv
from tudat_tools.data_processing.data_saving_utilities import dict2txt
import tudat_tools.data_processing.data_processing_utilities as dputil

def comp_switching_scheme_vars(
        dta,
        nr_sats_skipped,
        host_index, 
        data_raw, 
        simulation_parameters,
        aer,
        ind_sats,
        nr_tagets_labelled = 24,
        target_plane = 11,
        ):
    """_summary_

    Args:
        dta (_type_): _description_
        nr_sats_skipped (_type_): _description_
        host_index (_type_): _description_
        data_raw (_type_): _description_
        simulation_parameters (_type_): _description_
        aer (_type_): _description_
        ind_sats (_type_): _description_
        nr_tagets_labelled (int, optional): _description_. Defaults to 24.
        target_plane (int, optional): _description_. Defaults to 11.

    Returns:                
        arg_lat_host
        t_vec
        aer_storage_host
        dta_storage_host
        t_az_at_switch - time, azimuth to targ_now, azimuth to targ_Next
        target_connected
    """    


    t_vec = aer[:,0]
    dt = t_vec[1] - t_vec[0]

    sat_names = list(ind_sats.keys())

    targets = sat_names
    targeind_chosens_plotted = [7, 6, 5, 4, 3, 2, 1, 0, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8]

    target_labels = [f'S{ii+1}' for ii, ind in enumerate(targeind_chosens_plotted)]
    target_labels = [label if ii<nr_tagets_labelled else '' for ii, label in enumerate(target_labels)]
    
    if host_index == 2:
        targeind_chosens_plotted = targeind_chosens_plotted[1:]
        target_labels = target_labels[1:]
    target_name_base = f'sat_leo_t_{target_plane}'
    targets = [f'{target_name_base}_{ind}' for ind in targeind_chosens_plotted]

    sat_names = simulation_parameters['sat_names']
    host_ind_chosen = f'0_{host_index}'

    host_pack = [[ii, name] for ii, name in enumerate(sat_names) if host_ind_chosen in name][0]
    host_name = host_pack[1]
    host_state_ind = [1+ ii + host_pack[0]*6 for ii in range(6)]

    t_j2000 = data_raw[:,0]
    s_host = data_raw[:,host_state_ind]

    tud_converter = tudatconv.tudat_predictor()

    # importlib.reload(el_conv)
    # Compute true anomaly gaps
    kep_states_host, arg_lat_host = el_conv.get_kepler_param(tud_converter, s_host)
    arg_lat_host = np.rad2deg(arg_lat_host)%360
    delta_arg_lat_targets = []
    delta_arg_lat_dt_targets = [] # gradients
    arg_lat_targets = []
    arg_lat_targets_reverse = []
    for target_name in targets:
        target_ind = simulation_parameters['state_ind'][sat_names.index(target_name)]
        s_target = data_raw[:,[1+ ii + target_ind for ii in range(6)]]
        kep_states_target, arg_lat_target = el_conv.get_kepler_param(tud_converter, s_target, reverse = 1)
        arg_lat_target = np.rad2deg(arg_lat_target)
        delta_arg_lat_target = np.zeros(arg_lat_target.shape)
        d_arg_lat = np.abs(arg_lat_host-arg_lat_target)
        for ii, d_arg in enumerate(d_arg_lat):
            if d_arg > 180:        
                d_arg = np.abs(d_arg - 360)
                d_arg_lat[ii] = d_arg
        d_arg_lat_gradient = np.gradient(d_arg_lat, t_vec) # deg/s
        delta_arg_lat_target = d_arg_lat
        delta_arg_lat_dt_targets.append(d_arg_lat_gradient)
        delta_arg_lat_targets.append(delta_arg_lat_target)
        arg_lat_targets.append(arg_lat_target)
        arg_lat_targets_reverse.append(arg_lat_target)

    ## t LINES STUFF
    calc_tlines = 1
    shift_t_to_t0 = 0
    if calc_tlines:
        # Get T lines
        ii_targets_linked = list(range(host_index, len(targets), 1 + nr_sats_skipped))
        target_labels_plotted = np.array(target_labels)[ii_targets_linked]
        # to slice out time-vector and delta arg for np.interpolate interface
        delta_arg_lat_targets_increasing = []
        t_vecs_delta_arg = []
        # filter to only get growing dTA
        for ii, darglat in enumerate(delta_arg_lat_targets):
            ii_darglat_from0 = np.where(darglat < dta)[0][0]
            ii_darglat_fin = ii_darglat_from0+100
            delta_arg_lat_targets_increasing.append(darglat[ii_darglat_from0:ii_darglat_fin])
            t_vecs_delta_arg.append(t_vec[ii_darglat_from0:ii_darglat_fin])
            # ii_increasing = [jj for jj, dargdt in enumerate(delta_arg_lat_dt_targets[ii]) if dargdt>0]
            # delta_arg_lat_targets_increasing_ii = [darglat[jj] if jj in ii_increasing else 0 for jj, dlat in enumerate(darglat)]
            # delta_arg_lat_targets_increasing.append(delta_arg_lat_targets_increasing_ii)
        t_skips = [np.interp(dta, delta_arg_lat_targets_increasing[ii], t_vecs_delta_arg[ii]) for ii in ii_targets_linked]
        print(f'targets linked : {np.array(target_labels)[ii_targets_linked]}')
        # t_col0 = np.interp(dta, delta_arg_lat_targets[0], t_vec)
        # t_col3 = np.interp(dta, delta_arg_lat_targets[1 + nr_sats_skipped], t_vec)
        delta_arg_lat_targets_linked = np.array(delta_arg_lat_targets)[ii_targets_linked]
    # Compute switching scheme
    nrows = t_vec.shape[0]
    
    aer_storage_host = np.zeros((nrows, 3)) # AER
    dta_storage_host = np.zeros((nrows, 1)) # delta_TA
    t_az_at_switch = np.zeros((len(t_skips), 3))
    target_connected = []


    # TODO fix t_skips ending at 3160s

    for ii, t_ii in enumerate(t_vec):
        
        # Initialize
        if ii == 0:
            # TODO smarter way to find first target
            ii_used = 0
            ii_target = ii_targets_linked[ii_used]
            target_ii = targets[ii_target]
            t_skip_next = t_skips[ii_used]

            ind_targ = ind_sats[target_ii]

        if t_ii > t_skip_next:
            # get az at boundary for disconnected sat
            az_t_atskip = aer[ii, ind_targ][0]

            ii_used += 1            
            if ii_used >= len(t_skips):
                break
            ii_target = ii_targets_linked[ii_used]
            target_ii = targets[ii_target]
            
            t_skip_next = t_skips[ii_used]
            ind_targ = ind_sats[target_ii]
            # get az at boundary for next sat
            az_next_t_atskip = aer[ii, ind_targ][0]
            t_az_at_switch[ii_used-1,:] = [t_ii, az_t_atskip, az_next_t_atskip]

        
        aer_ii = aer[ii, ind_targ[:3]]

        # store                
        aer_storage_host[ii,:] = aer_ii 
        dta_storage_host[ii,:] = delta_arg_lat_targets_linked[ii_used][ii]
        target_connected.append(target_ii)
        
    return arg_lat_host[:ii], t_vec[:ii], aer_storage_host[:ii,:], dta_storage_host[:ii,:], t_az_at_switch, target_connected




if __name__ == '__main__':
    # path jazz
    path_cwd = os.getcwd()
    csv_output_path = r'orbital_simulations\terran_near_polar_split\NearPolar12x244.00h'
    fname_simparam = 'simulation_parameters.json'
    fname_states = 'state_history.dat'
    save_folder = r'outputs\tables\terran_const'
    folder_links = r'outputs\tables\terran_const'
    files_all = os.listdir(folder_links)
    
    target_plane = 11
    # mount_dir = 'X-backwards, Z-down'
    mount_dir = 'X-forwards, Z-up'
    switching = 'alt'
    # switching = 'seq'
    if switching == 'seq':        
        options = [20, 0, 0, 10]
    elif switching == 'alt':
        # TA trigger
        options = [35, 1, 0, 24]
    
    dta, nr_sats_skipped, host_index = options[0], options[1], options[2]

    host = f'_0_{host_index}'
    print(f'HOST {host}')
    
    r_lims = [500e3, 4e6]
    fnames = [f for f in files_all if host in f and f'_{target_plane}' in f]
    path_aer = fr'{folder_links}/{fnames[0]}'
    path_ind = fr'{folder_links}/{fnames[1]}'
    rows_used = 3.6e3
    data_raw, simulation_parameters = dputil.load_constellation_data(full_path = csv_output_path, nrows = rows_used)
    with open(path_ind, 'r') as j:
        ind_sats = json.load(j) 
    # load states
 

    host_label = f'N{host_index+1}'

    aer = np.loadtxt(path_aer, dtype = float, delimiter = ',')[:int(rows_used),:]
    arg_lat_host, aer, aer_storage_host, dta_storage_host, t_az_at_switch, target_connected = comp_switching_scheme_vars(
        dta,
        nr_sats_skipped,
        host_index, 
        data_raw, 
        simulation_parameters,
        aer,
        ind_sats,
        nr_tagets_labelled = 24,
        target_plane = 11,)
    t_plotted = t_vec/60
    # calc total az maneuver
    daz_sum =0
    for ii, t in enumerate(t_az_at_switch[:-1,0]):
        az_targ_i_end = t_az_at_switch[ii,1]
        az_targ_ii_start = t_az_at_switch[ii+1,2]

        az_targ_ii_end = t_az_at_switch[ii+1,1]
        daz_switch_targ = az_targ_ii_start - az_targ_i_end
        daz_follow_targ = az_targ_ii_end - az_targ_ii_start
        daz_sum += np.abs(daz_switch_targ) + np.abs(daz_follow_targ)
        print(f'{ii} switch to tar : {daz_switch_targ:.1f}, following : {daz_follow_targ:.1f} TOT = {daz_sum:.1f} deg')
    ## small checks - where azimuth is within a threshold
    az_min = 5
    ii_azlim_constrained = np.where(np.abs(aer_storage_host[:,0]) < az_min)[0]
    az_blocked = aer_storage_host[ii_azlim_constrained,0]
    t_constrained = t_plotted[ii_azlim_constrained]
    ## Make figs
    make_comb_az_plot = 1
    make_az_el_plot = 1
    make_comb_ta_r_plot = 0
    make_daz_plots = 1
    lim_x = 1

    add_azlim = 0
    # az_lims = [0, 90]
    az_lims = [-180, 180]
    # az_lims = [-90, 70]

    r_lims = [500, 5000]
    t_lim_0 = 20
    t_lim = t_lim_0 + 30

    if make_comb_az_plot:
        f, axs = plt.subplots(nrows = 2, figsize = (9,6))
    # ax.plot(t_plotted, arg_lat_host, label= 'S1')
        ax = axs[0]
        
        ax.plot(t_plotted, dta_storage_host, linestyle = '--')

        # ax.legend()
        ax.grid('on')
        
        if lim_x:
            ax.set_xlim([t_lim_0, t_lim])
            # ax.set_ylim([0, y_lim])
        ax.set_ylabel('TA separation [deg]', fontweight = 'bold')

        ax = axs[1]
        
        ax.plot(t_plotted, aer_storage_host[:,0])
        if add_azlim:
            ax.scatter(t_constrained, az_blocked, s = 6, c = 'r', label = f'|az| <{az_min:.0f} deg')
        if lim_x:
            ax.set_xlim([t_lim_0, t_lim])
            ax.set_ylim(az_lims)
        ax.legend(loc = 'lower right')
        ax.grid('on')
        ax.set_ylabel('Azimuth [deg]', fontweight = 'bold')
        ax.set_xlabel('t [min]', fontweight = 'bold')
        f.suptitle(f'{host_label} - links; Switching - {switching}; dTA = {dta} deg')

    if make_comb_ta_r_plot:
        f, axs = plt.subplots(nrows = 2, figsize = (9,6))
    # ax.plot(t_plotted, arg_lat_host, label= 'S1')
        ax = axs[0]
        
        ax.plot(t_plotted, dta_storage_host, linestyle = '--')

        # ax.legend()
        ax.grid('on')
        
        if lim_x:
            ax.set_xlim([t_lim_0, t_lim])
            # ax.set_ylim([0, y_lim])
        ax.set_ylabel('TA separation [deg]', fontweight = 'bold')

        ax = axs[1]
        
        ax.plot(t_plotted, aer_storage_host[:,2]/1e3)
        
        if lim_x:
            ax.set_xlim([t_lim_0, t_lim])
            ax.set_ylim(r_lims)
        ax.legend(loc = 'lower right')
        ax.grid('on')
        ax.set_ylabel('Slant-range [km]', fontweight = 'bold')
        ax.set_xlabel('t [min]', fontweight = 'bold')
        f.suptitle(f'{host_label} - links; Switching - {switching}; dTA = {dta} deg')
        
    if make_daz_plots:
        f, ax = plt.subplots()
        ax.plot(t_az_at_switch[:,1] - t_az_at_switch[:,2])
        ax.set_ylabel('delta Az at switch [deg]', fontweight = 'bold')
        ax.set_xlabel('Switch nr [-]', fontweight = 'bold')
        ax.grid()
        f.suptitle(f'{host_label} - links; Switching - {switching}; dTA = {dta} deg')
    if make_az_el_plot:
        f, axs = plt.subplots(nrows = 2, figsize = (9,6))
    # ax.plot(t_plotted, arg_lat_host, label= 'S1')
        ax = axs[0]
        
        ax.plot(t_plotted, aer_storage_host[:,0])

        # ax.legend()
        ax.grid('on')
        
        if lim_x:
            ax.set_xlim([t_lim_0, t_lim])
            # ax.set_ylim([0, y_lim])
        ax.set_ylabel('Azimuth [deg]', fontweight = 'bold')

        ax = axs[1]
        
        ax.plot(t_plotted, aer_storage_host[:,1])
        
        if lim_x:
            ax.set_xlim([t_lim_0, t_lim])            
        ax.legend(loc = 'lower right')
        ax.grid('on')
        ax.set_ylabel('Elevation [deg]', fontweight = 'bold')
        ax.set_xlabel('t [min]', fontweight = 'bold')
        f.suptitle(f'{host_label} - links; Switching - {switching}; dTA = {dta} deg. \nMounting : {mount_dir}')