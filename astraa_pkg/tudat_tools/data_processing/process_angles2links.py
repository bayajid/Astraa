#%%
import pathlib
import json
import csv
import os
import numpy as np
import pandas as pd
import sys
import importlib
import os, sys
sys.path.insert(1, os.getcwd()[:os.getcwd().index('astropynaric')+13])
import loading_functions.data_loading as load
import data_processing.rotations as rot
import data_processing.data_processing_utilities as dputil
import plotting_functions.modular_plotting as modplot
import matplotlib.pyplot as plt
import plotting_functions.plotting_basic as bplt
import plotting_functions.modular_plotting as modplot
import basic_tools.operations as basic
import basic_tools.constants as const
import matplotlib.backends.backend_pdf
importlib.reload(const)
importlib.reload(rot)
importlib.reload(load)
importlib.reload(dputil)
importlib.reload(bplt)
#%% base paths
pos_path = os.path.normpath(r"simulation_output\processed_outputs\2rel_position\\")
link_lct_path = os.path.normpath(r"simulation_output\processed_outputs\3azelslant\\")
output_path_link = os.path.normpath(r"simulation_output\processed_outputs\4links\\")
output_path_stats = os.path.normpath(r"simulation_output\processed_outputs\5link_statistics\\")
#%% Choose host, load data
sat_hosts = [pos[8:-4] for pos in os.listdir(pos_path) if 'los' in pos]
print(f'Available hosts: {*sat_hosts,}')
# req parameters
R_atm = 100e3
R_E = const.R_E
n_point_shown = 8*60*6
#%% Load processed data

# CUTS for shorter analysis TO BE MADE HERE
limit_cases = 1
print(f'Limiting Analysis: {bool(limit_cases)}')
if limit_cases:
    lct_analyzed = ['lct1', 'lct2', 'lct3', 'lct4'][:2]
    terminals_analyzed = ['mk2', 'mk3', 'general']
    sat_hosts_analyzed = sat_hosts[-1:] 
    ind_0 = 0
    ind_f = -1
else:
    lct_analyzed = ['lct1', 'lct2', 'lct3', 'lct4'][:2] # NOTE- unlimited cases dont do LCT3 or LCT4 (not necessary for planned analysis)
    terminals_analyzed = ['mk2', 'mk3', 'general']
    sat_hosts_analyzed = sat_hosts
for sat_host in sat_hosts_analyzed:
    if 'polar' in sat_host:
        cases_analyzed = const.cases_leo_p
        names_analyzed = const.names_leo_p
    elif 'incl' in sat_host:
        cases_analyzed = const.cases_leo_i
        names_analyzed = const.names_leo_i
    elif 'meo' in sat_host:
        cases_analyzed = const.cases_meoleo
        names_analyzed = const.names_meoleo
    if limit_cases:
        cases_analyzed = cases_analyzed[ind_0:ind_f]
        names_analyzed = names_analyzed[ind_0:ind_f]

    print(f'Analysis for {sat_host}')

    for terminal_type in terminals_analyzed:
        for lct_chosen in lct_analyzed:
            for nn, case_analyzed in enumerate(cases_analyzed):
                name_case = names_analyzed[nn]
                # output paths
                save_name_overview = f'{sat_host}_{name_case}_{lct_chosen}_linkoverview.csv'
                save_name_tseries = f'{sat_host}_{name_case}_{lct_chosen}_linkrates.csv'
                path_linkoverview = f'{output_path_link}\{terminal_type}\{save_name_overview}'
                path_linkseries = f'{output_path_link}\{terminal_type}\{save_name_tseries}'
                # data used
                data_path_aer = f'{link_lct_path}\\aer_{lct_chosen}_{sat_host}.csv'
                data_path_ecistates = f'{pos_path}\\pos_eci_{sat_host}.csv'
                # load states and Az El R 
                data_aer = pd.read_csv(data_path_aer)
                data_pos = pd.read_csv(data_path_ecistates)
                
                pos_h = dputil.get_cols(data_aer, setting = 'pos_h')
                vel_h = dputil.get_cols(data_pos, setting = 'vel_h')
                t_vec = data_aer.loc[:,'t'].values
                
                sats_target = [sat for sat in case_analyzed if sat != sat_host]

                link_dict = {}
                ii_all = range(len(t_vec))
                if terminal_type == 'mk2':
                    check_vis = 0
                    check_slant = 1
                    check_ae = 1
                    az_lims = [-175, 175]
                    el_lims = [-5, 25]
                    slant_limits = [1e3, 5000e3]
                    slant_range_limit = slant_limits[1] 
                elif terminal_type == 'mk3':
                    check_vis = 1
                    check_slant = 0
                    check_ae = 1
                    az_lims = [-175, 175]
                    el_lims = [-145, 145]
                elif terminal_type == 'general':
                    check_vis = 1
                    check_slant = 0
                    check_ae = 0
                sat_visibility_dict = {}
                n_linked_sats = 0
                for kk, sat_target in enumerate(sats_target):
                    # getting lists of indices for link/nolink
                    sat_visibility_dict[sat_target] = {}
                    aer_i = dputil.get_cols(data_aer, sat_target, 'aer') # az, el [deg] slant [m]
                    pos_t = dputil.get_cols(data_pos, sat_target, 'pos_t') # az, el [deg] slant [m]
                    vel_t = dputil.get_cols(data_pos, sat_target, 'vel_t') # az, el [deg] slant [m]
                    
                    if not ('leo' in sat_target and 'meo' in sat_host) and not ('meo' in sat_target and 'leo' in sat_host):
                        ## Not checking slant-limits of Mk2 if LEO-MEO
                        if check_slant: 
                            slant_range_i = aer_i[:,2]
                            ii_link = [ii for ii in ii_all 
                                                if slant_range_i[ii] <= slant_range_limit] # indices of links not exceeding slant range limit                            
                        else:
                            ii_link = ii_all
                    else:
                        ii_link = ii_all
                    # visibility checks. 2 cases, if MEO is target/LEO host and all others
                    if ('leo' in sat_target and 'meo' in sat_host) or ('meo' in sat_target and 'leo' in sat_host):
                        check_vis = 1
                    if check_vis:
                        if 'leo' in sat_target and 'meo' in sat_host:
                            ii_link = dputil.check_occultation_new(r_high = pos_h, r_low = pos_t, ii_all = ii_link, R_E = R_E, R_atm = R_atm)
                        elif ('meo' in sat_target and 'leo' in sat_host):
                            ii_link = dputil.check_occultation_new(r_high = pos_t, r_low = pos_h, ii_all = ii_link, R_E = R_E, R_atm = R_atm)
                        else:
                            ii_link = dputil.check_occultation_new(r_high = pos_t, r_low = pos_h, ii_all = ii_link, R_E = R_E, R_atm = R_atm)
                    # azimuth/elevation limit checks
                    if check_ae:        
                        if terminal_type == 'mk2':
                            ## standard check, elevation limited +/- 15deg
                            # check azimuth
                            ii_link = [ii for ii in ii_link 
                                                if aer_i[ii,0] >= az_lims[0] and aer_i[ii,0] <= az_lims[1]]
                            # check elevation
                            ii_link = [ii for ii in ii_link 
                                                if aer_i[ii,1] >= el_lims[0] and aer_i[ii,1] <= el_lims[1]]
                        elif terminal_type == 'mk3':
                            ## more complex check, elevation can handle angles above 90
                            ii_link_copied = ii_link
                            ii_link = []
                            for ii in ii_link_copied:
                                if np.abs(aer_i[ii,0]) > az_lims[1]: # ||az|| > 175 deg - outside of az range
                                    if np.abs(aer_i[ii,1]) < 180 - el_lims[1]: # if elevation below 35 (180 - 145), no link
                                        continue
                                ii_link.append(ii)
                    ii_no_link = list(sorted(set(ii_all) - set(ii_link))) 
                    t_link = t_vec[ii_link]
                    if len(ii_link) != 0: # link occurs 
                        n_linked_sats+=1
                        ## Split entire link range to separate links
                        ii_prev = ii_link[0] # previous index w.r.t. all array
                        ii_start = ii_link[0] # link start index
                        t_start = t_link[0] # link start time 
                        jj_start = 0 # index w.r.t. LINK array
                        n_link = 1 # total number of link tracker w/ current target
                        ## jj - indices in link_data. ii - indices in raw_data
                        # Overview outputs (1 value per link)
                        t_window_lst, type_lst, t_start_lst, t_end_lst, i_start_lst, i_end_lst, slant_range_min_lst, slant_range_max_lst= [],[],[],[],[],[],[],[]
                        link_lst, az_max, el_max, daz_max, del_max, ic_host, ic_target = [],[],[],[],[],[], []
                        ## Time-series outputs (all points per link)
                        az_h = []
                        el_h = []
                        r_h = []
                        # Gradients
                        dr_h = []
                        daz_h = []
                        del_h = []
                        t_h = []
                        for jj, ii_next in enumerate(ii_link):
                            d_ii = ii_next - ii_prev # difference in link array index                
                            if d_ii > 1 or (ii_start == ii_link[0] and ii_next+1 == len(ii_link)): # new link begins - doesnt trigger for cosntant links
                                if d_ii > 1:
                                    # get link end parameters if link end was found
                                    t_end = t_link[jj-1]
                                    link_type = 'window'
                                    t_window = t_end - t_start
                                    if ii_start == ii_prev: # in case detected link is only a single data point
                                        ii_prev +=1 # include the next state point

                                    row_start = ii_start if ii_start == 0 else ii_start # EXclude data point just before link occurs
                                    row_end = ii_prev                                
                                    # extract chosen LCT azimuth, elevation, LOS vector
                                else:
                                    row_start = ii_start
                                    t_start = t_link[jj_start]
                                    t_end = t_link[jj]
                                    if len(ii_no_link) == 0: # all found points were links
                                        link_type = 'permalink'
                                        t_window = np.inf
                                    else:
                                    # get link end parameters if link didn't end 
                                        link_type = 'window'
                                        t_window = t_end - t_start
                                        # n_point_shown = ii_start + 2
                                    row_end = n_point_shown
                                slant_range_next = aer_i[row_start:row_end+1,2]
                                ae_lct_chosen = aer_i[row_start:row_end+1,[0,1]]
                                t_vec_link = t_vec[row_start:row_end+1]
                                # Store time-series of link parameters
                                az = ae_lct_chosen[:,0]
                                el = ae_lct_chosen[:,1]
                                az_range = np.unwrap(az, period = 360)
                                el_range = np.unwrap(el, period = 360)
                                if az_range.shape[0]>1:
                                    daz_range = np.gradient(az_range, t_vec_link)
                                    del_range = np.gradient(el_range, t_vec_link)
                                    dr_range = np.gradient(slant_range_next, t_vec_link)
                                    t_h = np.hstack((t_h,t_vec_link))
                                    r_h = np.hstack((r_h, slant_range_next))
                                    az_h = np.hstack((az_h,az))
                                    el_h = np.hstack((el_h,el))
                                    # gradients                
                                    daz_h = np.hstack((daz_h,daz_range)) 
                                    del_h = np.hstack((del_h,del_range)) 
                                    dr_h = np.hstack((dr_h,dr_range))
                                    # Storing overview data
                                    type_lst.append(link_type)
                                    t_window_lst.append(t_window/60)
                                    t_start_lst.append(t_start/60)
                                    i_start_lst.append(row_start)
                                    i_end_lst.append(row_end)
                                    slant_range_min_lst.append(int(np.min(slant_range_next)/1e3))
                                    slant_range_max_lst.append(int(np.max(slant_range_next)/1e3))
                                    az_max.append(np.min(np.abs(az)))
                                    el_max.append(np.max(np.abs(el)))
                                    daz_max.append(np.min(np.abs(daz_h)))
                                    del_max.append(np.max(np.abs(del_h)))
                                    ic_host.append(np.concatenate([pos_h[row_start,:],vel_h[row_start,:]]))
                                    ic_target.append(np.concatenate([pos_t[row_start,:],vel_t[row_start,:]]))
                                    link_lst.append(n_link)
                                jj_start = jj
                                ## Update indices
                                n_link+=1
                                ii_start = ii_next
                                t_start = t_link[jj] # start of next link time
                            elif d_ii == 1: # link continues 
                                pass
                            ii_prev = ii_next
                        # output dataframes
                        sat_target_lst = [sat_target for tt in enumerate(t_h)]
                        sat_target_ovw = [sat_target for tt in enumerate(type_lst)]
                        dat_overview = {
                            'type' : type_lst,
                            'sat_target' : sat_target_ovw,
                            'link_nr' : link_lst,
                            't_window' : t_window_lst,
                            't_start' : t_start_lst,
                            'i_start' : i_start_lst,
                            'i_end' : i_end_lst,
                            'slant_range_min' : slant_range_min_lst,
                            'slant_range_max' : slant_range_max_lst,
                            'az_max' : az_max,
                            'el_max' : el_max,
                            'daz_max' : daz_max,
                            'del_max' : del_max,
                            'ic_host' : ic_host,
                            'ic_target' : ic_target,
                        }
                        dat_series = {
                        't' : t_h,
                        'sat_target' : sat_target_lst,
                        'r_h' : r_h,
                        'az_h' : az_h,
                        'el_h' : el_h,
                        'dr_h' : dr_h,
                        'daz_h' : daz_h,
                        'del_h' : del_h,
                        }
                        if n_linked_sats == 1:
                            overview_df = pd.DataFrame.from_dict(dat_overview)
                            tseries_df = pd.DataFrame.from_dict(dat_series)
                        else:
                            # stack up the dataframes                
                            overview_df = pd.concat([overview_df, pd.DataFrame.from_dict(dat_overview)])
                            tseries_df = pd.concat([tseries_df, pd.DataFrame.from_dict(dat_series)])
                try:
                    print(f'Saving {path_linkoverview}\n{path_linkseries}')
                    overview_df.to_csv(path_linkoverview, index = False)
                    tseries_df.to_csv(path_linkseries, index = False)
                except:
                    print(f'EMPTY: {path_linkoverview}\n{path_linkseries}')
print('All done :)')
print(f'Limiting Analysis Setting: {bool(limit_cases)}')
