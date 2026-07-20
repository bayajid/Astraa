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
import rotations as rot
import data_processing.data_processing_utilities as dputil
import plotting_functions.modular_plotting as modplot
import matplotlib.pyplot as plt
import plotting_functions.plotting_basic as bplt
import plotting_functions.modular_plotting as modplot
importlib.reload(rot)
importlib.reload(load)
importlib.reload(dputil)
importlib.reload(bplt)
#%% load overview table and raw constellation data
overview_tables_path = r'simulation_output\tables\link_parameters'
output_folder = 'simulation_output\\tables'
output_name = 'all_link_ic.csv'
save_overview_cut = 'links_selected.csv'
selected_links = f'{overview_tables_path}/{save_overview_cut}'

chosen_constellation = 'Leo_globalMeo_equator24.00h'
simulation_folder = dputil.get_sim_path(parent_dir)
data_raw, simulation_parameters = dputil.load_constellation_data(chosen_constellation, 
simulation_folder)
sat_names = simulation_parameters['sat_names']

link_table = pd.read_csv(selected_links)
link_table = link_table.sort_values('host')
#%% get IC and all other stuff
host_sats = link_table['host'].values
hosts_unique = list(set(sorted(host_sats)))
lct_chosen = 'lct1'
slant_limits = [10e3, 4000e3]
R_atm = 100e3
# load raw data
out_host = []
out_target = []
out_link_time = []
out_link_t0 = []
out_daz = []
out_del = []
out_ic_h = []
out_ic_t = []
for ii, sat_host in enumerate(hosts_unique):
    link_table_ii = link_table[link_table['host'] == sat_host]
    sats_target_ii = link_table_ii['target'].values
    links_ii =  link_table_ii['link'].values
    analyzed_sats = [sat for sat in sats_target_ii]
    analyzed_sats.append(sat_host)
    data_used, t_vec, indices_dict, los_dict = dputil.calculate_link_parameters(data_raw, simulation_parameters, sat_host, 
    t_lim = None, sat_names = analyzed_sats)
    link_dict_ii, sats_linked, sats_not_linked, sats_not_checked = dputil.process_los_dict(data_used, t_vec, sat_host, los_dict, indices_dict, slant_limits,
            sat_target_chosen=sats_target_ii, lct_chosen = lct_chosen, n_point_shown= data_used.shape[0], R_atm = R_atm)
    
    for jj, sat_target in enumerate(sats_target_ii):
        link_index = links_ii[jj] # number of selected link
        link_jj = link_dict_ii[sat_target][link_index]
        # get Az, El [deg] and their gradients
        ae_host = link_jj['ae_host']
        t_range = link_jj['t_vec']
        az = ae_host[:,0]
        el = ae_host[:,1]
        az_range = np.rad2deg(np.unwrap(az)) # deg
        el_range = np.rad2deg(np.unwrap(el)) # deg
        daz_range = np.gradient(az_range, t_range)
        del_range = np.gradient(el_range, t_range)

        out_host.append(sat_host)
        out_target.append(sat_target)
        out_link_time.append(link_jj['t_window'])
        out_link_t0.append(link_jj['t_start']) 
                
        out_daz.append(np.max(np.abs(daz_range)))
        out_del.append(np.max(np.abs(del_range)))
        out_ic_h.append(link_jj['ic_host'])
        out_ic_t.append(link_jj['ic_target'])

ic_dict = {
'host' : out_host,
'target' : out_target,
't_window' : out_link_time,
't_0' : out_link_t0,
'az_rate_max' : out_daz,
'el_rate_max' : out_del,
'ic_host' : out_ic_h,
'ic_target' : out_ic_t
}
ic_df = pd.DataFrame.from_dict(ic_dict)
ic_df.to_csv(f'{output_folder}\{output_name}', index = False)
# %%
