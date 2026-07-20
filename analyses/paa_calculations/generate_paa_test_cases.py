#%% Script to generate PAA test case inputs and expected outputs
import pandas as pd
import sys
import pathlib
import json
import importlib
import os
import numpy as np
import matplotlib.pyplot as plt
import paa_calculation_cleaned_up as paa_tool
from scipy.spatial.transform import Rotation as R
###
# Units/input descriptions: 
# Time - GPS seconds 
# Position/Velocity - m; m/s in ECI
# Attitude - Quaternions, scalar-first
# Az/El/PAA - urad
###
## Paths
folder_outputs = 'paa_testcases'

## constant attitude generation
attitude_constant = np.identity(3)
att_quat_scalarlast = R.from_matrix(attitude_constant).as_quat()
att_quat = np.copy(att_quat_scalarlast)
att_quat[1:] = att_quat_scalarlast[0:3]
att_quat[0] =  att_quat_scalarlast[3]

##
t_span = 300
dt = 1
t_vec_0 = 1338076818.0 # gps time
t_vec_f = t_vec_0 + t_span
t_vec = np.round(np.arange(t_vec_0, t_vec_f, dt), 1)
nrows = len(t_vec)

## Test cases - define initial conditions
case = 8

# initial conditions
if case == 1:
    r_host = [0,0,0]
    r_target = [3e6, 0, 0]
    v_host = [0,0,0]
    v_target = [0, 1e3, 0]
elif case == 2:
    r_host = [0,0,0]
    r_target = [3e6, 0, 0]
    v_host = [0,0,0]
    v_target = [0, 0, 1e3]
elif case == 3:
    r_host = [0,0,0]
    r_target = [3e6, 0, 0]
    v_host = [0,0,-1e3]
    v_target = [0, 0, 0]
elif case == 4:
    r_host = [0,0,0]
    r_target = [3e6, 0, 0]
    v_host = [0,0,0]
    v_target = [1e3, 0,  1e3]
elif case == 5:
    r_host = [0,0,0]
    r_target = [3e6, 0, 0]
    v_host = [0,0,0]
    v_target = [0, -1e3, -1e3]    
elif case == 6:
    r_host = [0,0,0]
    r_target = [3e6, 0, 0]
    v_host = [0,0,1e3]
    v_target = [0, 0, 1e3]
elif case == 7:
    r_host = [0,0,0]
    r_target = [3e6, 0, 0]
    v_host = [0,-1e3,0]
    v_target = [0, 1e3, 0]
elif case == 8:
    r_host = [0,0,0]
    r_target = [3e6, 0, 0]
    v_host = [0, -1e3, -1e3]
    v_target = [0, -1e3, -1e3]
### Create placeholders
state_host = np.zeros((nrows, 7))
state_target = np.zeros((nrows, 7))
attitude_host = np.zeros((nrows, 9))
pt_angles = np.zeros((nrows, 5)) # t, Az, El, PAA_Az, PAA_El


r_host = np.array(r_host)
r_target = np.array(r_target)
v_host = np.array(v_host)
v_target = np.array(v_target)
## generate inputs
for ii, t in enumerate(t_vec):
    # store
    state_host[ii,:] = np.hstack(([t, r_host, v_host]))
    state_target[ii,:] = np.hstack(([t, r_target, v_target]))
    attitude_host[ii,:5] = np.hstack(([t, att_quat]))
    # propagate
    r_host_i = r_host + v_host * dt
    r_taget_i = r_target + v_target * dt

    # update
    r_host = r_host_i
    r_target = r_taget_i
print(f'Inputs generated for Case{case} for {t_span} s with dt = {dt}')
# get outputs
make_plots = 1
pt_outputs = paa_tool.compute_azel_paa(state_host, state_target, attitude_host)
if make_plots:
    f1, f2 = paa_tool.make_pt_angle_plots(case, pt_outputs) 

dat_lst = [state_host, state_target, attitude_host, pt_outputs]
save_name_lst = ['host_states_raw', 'target_states_raw', 'host_attitude_raw', 'pt_angles']
cols_all = [
    ['t_gps_s', 'x_m', 'y_m', 'z_m', 'vx_ms', 'vy_ms', 'vz_ms'],
    ['t_gps_s', 'x_m', 'y_m', 'z_m', 'vx_ms', 'vy_ms', 'vz_ms'],
    ['t_gps_s', 'q1', 'q2', 'q3', 'q4', 'qdot1', 'qdot2', 'qdot3', 'qdot4'],
    ['t_gps_s', 'cpa_az_rad', 'cpa_el_rad', 'paa_az_rad', 'paa_el_rad']
]
for ii, savename in enumerate(save_name_lst):
    ## Save inputs/outputs
    output_case = f'Case{case}'
    output_dir = f'{folder_outputs}/{output_case}'
    full_save_path = f'{output_dir}/{save_name_lst[ii]}_case{case}.csv'
    try:
        os.mkdir(output_dir)
    except:
        pass
    df_out = pd.DataFrame(data = dat_lst[ii], columns= cols_all[ii])
    df_out.to_csv(full_save_path, index = False)
    print(f'Saved case{case} to {output_dir}')
    paa_tool.savefig(f1, f'azel_case{case}', output_dir)
    paa_tool.savefig(f2, f'paa_azel{case}', output_dir)