#%% 
import pandas as pd
import sys
import pathlib
import json
import importlib
import os
import numpy as np
import matplotlib.pyplot as plt
## path
# path_orbdata = r'raw_inputs/orbit_parameters.json' # Old dataset
# 2023-01-18 replaced with data following official frame convention
folder = 'raw_inputs'
# path_orbdata = r'20230119_orbit_parameters_official_frame.json'
# path_orbdata = r'20230127_simple_test_case_elevation.json'
path_orbdata = r'20230127_simple_test_case_elevation.json'
# path_orbdata = r'20230119_orbit_parameters_official_frame_attitude_rotation_inverted.json' # inverted rotation from ECI to global
# path_orbdata = r'20230119_orbit_parameters_official_frame_attitude_debug_transposed.json' # transposed input
path_orbdata = f'{folder}/{path_orbdata}'

## load
with open(path_orbdata) as f:
    dat_full = json.load(f)
# place into dataframe
# df_entire_set = pd.DataFrame.from_dict(dat_full)
save2csv = 1
plot_paa = 1
#%% Process/plot
def conv_lst2array(json_full, key):
    # function to convert list of list data in json to array
    dat_lst = json_full[key]
    
    nrows = len(dat_lst)
    ncols = len(dat_lst[0])
    
    out_array = np.zeros((nrows, ncols))

    for ii, row in enumerate(dat_lst):
        out_array[ii,:] = row
    
    return out_array



pos_host = conv_lst2array(dat_full, 'pos1')*1e3 # [m]
vel_host = conv_lst2array(dat_full, 'vel1')*1e3 # [m/s]
att_host = conv_lst2array(dat_full, 'attitudes1') 
# att_host = conv_lst2array(dat_full, 'attitude1')
pos_target = conv_lst2array(dat_full, 'pos2')*1e3 # [m]
vel_target = conv_lst2array(dat_full, 'vel2')*1e3 # [m/s]
att_target = conv_lst2array(dat_full, 'attitudes2')
# att_target = conv_lst2array(dat_full, 'attitude2')
att_rate_target_raw = conv_lst2array(dat_full, 'vel_attitudes2')
paa1_array = conv_lst2array(dat_full, 'PAA1')
paa2_array = conv_lst2array(dat_full, 'PAA2')
slant = dat_full['distance']

t_vec_gps = dat_full['GPS time']
nrows = len(t_vec_gps)
t_gps = np.reshape(t_vec_gps, (nrows, 1)) # array form
# get quaternion rates

for ii, attrow in enumerate(att_host):
    if ii > 0:
        if np.dot(att_host[ii,:], att_host[ii-1,:]) < 0:
            att_host[ii,:] = -att_host[ii,:]
        if np.dot(att_target[ii,:], att_target[ii-1,:]) < 0:
            att_target[ii,:] = -att_target[ii,:]
att_host_rate = np.zeros(att_host.shape)
att_target_rate = np.zeros(att_host.shape)

for ii in range(4):
    att_host_rate[:,ii] = np.gradient(att_host[:,ii], t_vec_gps)
    att_target_rate[:,ii] = np.gradient(att_target[:,ii], t_vec_gps)

if plot_paa:
    f, axs = plt.subplots(nrows = 3, figsize = (10,12))
    paa_plotted = paa1_array

    paa_plotted_robust = pd.read_csv('paa_robust_full.csv')
    # paa_plotted = paa2_array
    # plot PAA
    marker_analytical = '>'
    marker_robust = 'o'
    ax = axs[0]
    colors = ['y', 'g']
    for ii in range(2):
        ax.plot(t_gps, paa_plotted[:,ii]*1e6, label = 'analytical ' + ['dAz', 'dEl'][ii], marker = marker_analytical, c = colors[ii], markevery = 500)
        ax.plot(t_gps, paa_plotted_robust.iloc[:,ii+1], label = 'robust ' + ['dAz', 'dEl'][ii], marker = marker_robust, c = colors[ii], markevery = 500)
    ax.set_ylabel('PAA 2-way [urad]')
    ax.set_xlabel('t [s]')
    ax.grid()
    ax.legend()
    # plot XYZ

    ax = axs[1]
    colors = ['y', 'g', 'orange']
    
    for ii in range(3):
        ax.plot(t_gps, pos_host[:,ii], label = 'host ' + 'xyz'[ii], marker = marker_analytical, c = colors[ii], markevery = 500)
        ax.plot(t_gps, pos_target[:,ii], label = 'target ' + 'xyz'[ii], marker = marker_robust, c = colors[ii], markevery = 500)
    ax.set_ylabel('Position ECI [m]')
    ax.set_xlabel('t [s]')
    ax.grid()
    ax.legend()    
    ax = axs[2]
    ax.plot(t_gps, slant)
    ax.set_ylabel('link distance [km]')
    ax.set_xlabel('t [s]')
    ax.grid()
#%% make outputs
host_pos_df = pd.DataFrame(np.hstack((t_gps, pos_host, vel_host)), columns = ['t_gps_s', 'x', 'y', 'z', 'vx','vy','vz'])
target_pos_df = pd.DataFrame(np.hstack((t_gps, pos_target, vel_target)), columns = ['t_gps_s', 'x', 'y', 'z', 'vx','vy','vz'])
host_attitude_df = pd.DataFrame(np.hstack((t_gps, att_host, att_host_rate)), columns = ['t_gps_s', 'q1', 'q2', 'q3', 'q4', 'qdot1', 'qodt2', 'qdot3' ,'qdot4'])
target_attitude_df = pd.DataFrame(np.hstack((t_gps, att_target, att_target_rate)), columns = ['t_gps_s', 'q1', 'q2', 'q3', 'q4', 'qdot1', 'qodt2', 'qdot3' ,'qdot4'])
hot_paa_analytical_df = pd.DataFrame(np.hstack((t_gps, paa1_array)), columns = ['t_gps_s', 'dAz', 'dEl'])
#% save to csv
if save2csv:
    fnames = ['host_states', 'target_states','host_attitude', 'target_attitude', 'host_paa']
    dfs = [host_pos_df, target_pos_df, host_attitude_df, target_attitude_df, hot_paa_analytical_df]
    for ii, fname in enumerate(fnames):
        df_saved = dfs[ii]
        df_saved.to_csv(f'csv_inputs\{fname}_ltb_raw.csv', index = 0)
        print(f'{fname} saved')
        