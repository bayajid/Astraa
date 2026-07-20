#%% Calculating AER for High precision LISL orbits
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
import matplotlib.backends.backend_pdf
importlib.reload(rot)
importlib.reload(load)
importlib.reload(dputil)
importlib.reload(bplt)

lct_chosen = 'lct2'
long_sim = 0 # setting to process 30 days of orbital data
#output parameters
n_digits = 3
save_cs_intermediate_los = 0
save_cs_intermediate_angles = 1
csv_output_path = r'C:\Users\KPaliusis\OneDrive - Mynaric AG\Documents\Github repositories\thesis_work\simulation_output\tables\link_parameters'
path_base = r'simulation_output\intersatellite_links\high_precision'
err_analyzed = 'tle'
pe_chosen_base = r'simulation_output\offset_states\tle_offsets'
save_base = r'simulation_output\offset_states\tle_offsets\processed\all_samples'
# simulation parameters
slant_limits_mk2 = [10e3, 5000e3] # slant range limits [m]
slant_limits_mk3 = [None, None]
R_atm = 200e3 # altitude above which links are established
dt = 60
t_lim = None # option to limit nr of loaded rows
simulation_folder = dputil.get_sim_path(parent_dir)
n_data_rows_loaded = int(t_lim/dt) if t_lim != None else None # number of data point to load
data_all = os.listdir(path_base)
for data_name in data_all:
    print(f'Data Chosen: {data_name}')
    path_hp_full = os.path.normpath(f'{path_base}\{data_name}\state_history.dat')
    data_hp = load.open_dat(path_hp_full)
    path_parameters_full = os.path.normpath(f'{path_base}\{data_name}\simulation_parameters.json')
    with open(path_parameters_full, 'r') as j:
        sim_parameters = json.load(j)
    output_linkcase = f'{pe_chosen_base}\{data_name}'    
    t_offsets = os.listdir(output_linkcase)

    data_used, t_vec, indices_dict, los_dict = dputil.calculate_link_parameters(data_hp, sim_parameters, 
    sat_host = sim_parameters['sat_names'][0], 
    t_lim = None)
    sat_host = sim_parameters['sat_names'][0]
    if save_cs_intermediate_angles:
        print(f'Saving Az, El, R')
        path_angles = f'{path_base}\{data_name}'
        pandas_dict = {}
        pandas_dict['t'] = t_vec
        pos_h = los_dict[sat_host]['pos_h']
        pandas_dict['posh_x'] = pos_h[:,0]
        pandas_dict['posh_y'] = pos_h[:,1]
        pandas_dict['posh_z'] = pos_h[:,2]

        angle_dict = pandas_dict.copy()
        for key in los_dict[sat_host].keys():
            if key != 'pos_h':                    
                lct_data = los_dict[sat_host][key][lct_chosen]
                ae = np.rad2deg(lct_data['ae'])
                slant = basic.calc_dmat(lct_data['los'])
                angle_dict[f'{key}_az'] = ae[:,0]
                angle_dict[f'{key}_el'] = ae[:,1]
                angle_dict[f'{key}_slant'] = slant
        lct_aer_df = pd.DataFrame.from_dict(angle_dict)
        output_path_aer = f'{path_angles}\\aer_{lct_chosen}_{sat_host}.csv'
        lct_aer_df.to_csv(output_path_aer, index=False)
        print(f'Saved AER_{lct_chosen} for {sat_host} in {output_path_aer}')
    