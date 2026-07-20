import pandas as pd
import sys
import pathlib
import json
import importlib
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
sys.path.insert(1, os.getcwd()[:os.getcwd().index('astropynaric')+13])

# path jazz
path_cwd = os.getcwd()
csv_output_path = r'orbital_simulations\tudat_raw_states'
fname_simparam = 'simulation_parameters.json'
fname_states = 'state_history.dat'

## MVP imports
import basic_tools.vector_operations as vec_calc
import astronomy_tools.constants as const
import astronomy_tools.astro_targets as where_sun
import plotting_tools.basic_plotting as bplt
import plotting_tools.combined_plots as cbmplt
import basic_tools.time_conversion as t_conv
import basic_tools.in_out as out
import attitude_tools.attitude_simulation as att_calc
import paa_tools.paa_calculation as paa_tool
import prediction_methods.error_generation as err_gen
## Loading satellite orbital data
import tudat_tools.data_processing.data_processing_utilities as dputil
data_raw, simulation_parameters = dputil.load_constellation_data(full_path = csv_output_path)

host_chosen = 'leo_host_incl'
# host_chosen = 'leo_host_polar'
target_chosen = 'leo_target_polar'
# host_chosen = 'leo_host_incl'
# host_chosen = 'meo_target_incl'

t_j2000 = data_raw[:,0]
t_gps = t_j2000 + t_conv.dt_j2000tt2gps()
t_gps = t_gps - t_gps[0]
r_host_true = data_raw[:,simulation_parameters['r_index'][host_chosen]]
v_host_true = data_raw[:,[ii+3 for ii in simulation_parameters['r_index'][host_chosen]]]
s_host_true = np.hstack((r_host_true, v_host_true))

r_target_true = data_raw[:,simulation_parameters['r_index'][target_chosen]]
v_target_true = data_raw[:,[ii+3 for ii in simulation_parameters['r_index'][target_chosen]]]
s_target_true = np.hstack((r_target_true, v_target_true))
attitude_host = att_calc.calc_quat_eci2lct(r_host_true, v_host_true)[0]

# compute_azel_paa

all_true = paa_tool.compute_azel_paa(s_host_true, s_target_true, attitude_host)
ae_true = np.rad2deg(all_true[:,[3,4]]) # [deg]
r_true = all_true[:,[5]]
paa_true = all_true[:,-2:]*1e6 # [urad, PAA_az, PAA_el]
cbmplt.plot_paa(t_gps, paa_true, (paa_true[:,0],paa_true[:,0]), fname = 'paa_true', save_figure=1)
cbmplt.plot_aer(t_gps, np.hstack((ae_true, r_true/1e3)), setting = 0)
# pee_comp = paa_tool.make_pt_angle_plots(case, pt_outputs) 

dr = 1000 # [m]
dv = 100 # [m/s]
ii = 0
f, ax = plt.subplots()
colors = 'rgcm'
dr_all = [1e3, 1e3, 1e3, 5e2]
dv_all = [5, 50, 100, 5]
for ii in range(4):    
    dr, dv = dr_all[ii], dv_all[ii]
    r_err_host = err_gen.pos_err_gen(0, dr, r_target_true.shape[0], ncols = 3, seed_used = 0)
    v_err_host = err_gen.pos_err_gen(0, dv, r_target_true.shape[0], ncols = 3, seed_used = 1)
    r_host_werr = r_host_true + r_err_host
    v_host_werr = v_host_true + v_err_host

    r_err_targ = err_gen.pos_err_gen(0, dr, r_target_true.shape[0], ncols = 3, seed_used = 2)
    v_err_targ = err_gen.pos_err_gen(0, dv, r_target_true.shape[0], ncols = 3, seed_used = 3)
    r_target_werr = r_target_true + r_err_targ
    v_target_werr = v_target_true + v_err_targ
    s_host_werr = np.hstack((r_host_werr, v_host_werr))
    s_target_werr =  np.hstack((r_target_werr, v_target_werr))

    paa_comp = paa_tool.compute_azel_paa(s_host_werr, s_target_werr, attitude_host)[:,-2:]*1e6 # [urad, PAA_az, PAA_el]

    paa_err = paa_true - paa_comp
    paa_err_az = paa_err[:,0]
    paa_err_el = paa_err[:,1]

    ax.plot(t_gps/60, paa_err_az, f'{colors[ii]}x', label = f'dr = {dr}m; dv = {dv}m/s')
    ax.plot(t_gps/60, paa_err_el, f'{colors[ii]}x')

ax.set_ylabel('PAA error [urad]')
# ax.set_title(f'PAA err [urad] w errors = dr = {dr} m, dv = {dv} m/s')
ax.legend()