import pathlib
import json
import csv
import os
import numpy as np
import pandas as pd
import sys
import importlib
parent_dir = pathlib.Path(__file__).parent.parent.resolve()
#%%
os.chdir(parent_dir)
sys.path.insert(1, os.getcwd())
import loading_functions.data_loading as load
import rotations as rot
import data_processing_utilities as dputil
import plotting_functions.modular_plotting as modplot
import matplotlib.pyplot as plt
import plotting_functions.plotting_basic as bplt
import plotting_functions.modular_plotting as modplot
import basic_tools.operations as basic
import basic_tools.constants as const
import matplotlib.backends.backend_pdf
from matplotlib.offsetbox import AnchoredText
importlib.reload(const)
importlib.reload(rot)
importlib.reload(load)
importlib.reload(dputil)
importlib.reload(bplt)
## base paths
pos_path = os.path.normpath(r"simulation_output\processed_outputs\2rel_position\\")
link_lct_path = os.path.normpath(r"simulation_output\processed_outputs\3azelslant\\")
output_path_link = os.path.normpath(r"simulation_output\processed_outputs\4links\\")
output_path_stats = os.path.normpath(r"simulation_output\processed_outputs\5link_statistics\\")
output_path_tables = os.path.normpath(r"simulation_output\\tables\\")

output = 'link_nr'
csv_name = 'linkoverview'
lct_analyzed = 'lct2'
terminal_type = 'general'
data_folder = f'{output_path_link}/{terminal_type}'
data_all = os.listdir(data_folder)

ind_lim = None
sat_hosts = ['sat_meo_0_0', 'sat_meo_0_0', 'sat_leo_polar_4_4', 'sat_leo_polar_4_4', 'sat_leo_polar_4_4', 'sat_leo_incl_4_4'][0:ind_lim]
sat_targets = ['sat_meo_0_1', 'sat_leo_incl_2_4', 'sat_meo_0_1', 'sat_leo_polar_4_5', 'sat_leo_incl_11_8', 'sat_leo_incl_11_11'][0:ind_lim]
t_start = [0, 30, 30, 0, 60, 0]
cases_ind = [2, 0, -1, 0, 2, 1]
cols_taken = ['ic_host', 'ic_target', 't_window', 'link_nr', 't_start']
for ii, (sat_host, sat_target) in enumerate(zip(sat_hosts, sat_targets)):
    case_ind = cases_ind[ii]
    cases,names = dputil.get_cases(sat_host)
    name_analyzed = names[case_ind]
    data_available_host = [file for file in data_all if sat_host in file]
    data_available_lct = [file for file in data_available_host if lct_analyzed in file]
    data_filt = [file for file in data_available_lct if csv_name in file]
    data_name = [file for file in data_filt if name_analyzed in file.removeprefix(sat_host)][0]
    loaded_path = os.path.normpath(f'{data_folder}/{data_name}')
    # print(sat_host, sat_target, data_name)
    data_loaded = pd.read_csv((loaded_path))
    data_cut = data_loaded[data_loaded['sat_target'] == sat_target]
    data_cut = data_cut[data_cut['t_start'] >= t_start[ii]].iloc[0]
    ic_ht = data_cut[cols_taken].values
    
    dict_used = {
        'host' : sat_host,
        'target' : sat_target
    }
    for jj, key in enumerate(cols_taken):
        dict_used[key] = ic_ht[jj]
    if ii == 0:
        output_df = pd.DataFrame.from_dict([dict_used])
    else:
        output_df = pd.concat((output_df,pd.DataFrame.from_dict([dict_used])))
if 0:
    ic_path = os.path.normpath(f'{output_path_tables}/ic_used.csv')
    output_df.to_csv(ic_path, index = False)
    print(f'Saved IC to {ic_path}')
