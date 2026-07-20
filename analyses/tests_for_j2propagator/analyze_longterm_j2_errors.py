#%% IMPORTS
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os 
import datetime
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd
import importlib
import os, sys
sys.path.insert(1, os.getcwd()[:os.getcwd().index('astropynaric')+13])

# path jazz
path_cwd = os.getcwd()
csv_output_path = r'orbital_simulations/leo_for_prop_hil/leo_leo'#'outputs/tables/j2propagator_testdata'
fname_simparam = 'simulation_parameters.json'
fname_states = 'state_history.dat'

## MVP imports
import basic_tools.vector_operations as vec_calc
import astronomy_tools.constants as const
import astronomy_tools.astro_targets as where_sun
import plotting_tools.basic_plotting as bplt
import basic_tools.time_conversion as t_conv
import basic_tools.in_out as out
import prediction_methods.j2propagator as j2prop

## Loading satellite orbital data
import tudat_tools.data_processing.data_processing_utilities as dputil
import prediction_methods.error_generation as err_gen
data_raw, simulation_parameters = dputil.load_constellation_data(full_path = csv_output_path)
host_chosen = 'leo_polar_1'
t_j2000 = data_raw[:,0]
# first satellite indices:
ii_polar1 = [1,2,3,4,5,6]
ii_chosen = 0

s_truth = data_raw[:,ii_polar1]
x_ic_true = s_truth[ii_chosen,:]
t_ic = t_j2000[ii_chosen]
use_single_precision = True
err_pos = 12 # m, 3-sig
err_vel = 0.3 # m/s , 3-sig
seed_chosen = 1
nr_sims = 100
r_error_ic = err_gen.pos_err_gen(0, err_pos/3, ncols=3, nrows=nr_sims, seed_used=seed_chosen)
v_error_ic = err_gen.pos_err_gen(0, err_vel/3, ncols=3, nrows=nr_sims, seed_used=seed_chosen)
a = 1

t_0 = t_j2000[0]
t_end = t_0 + 7200
dt = 10 # s, j2 stepsize
t_vec_prop = np.arange(t_0, t_end+dt, dt)
interpolant = sp.interpolate.interp1d(t_j2000,s_truth, axis =0)
s_true_interpolated = interpolant(t_vec_prop)
r_errors_all = np.zeros((len(t_vec_prop), nr_sims+1))
r_errors_all[:,0] = t_vec_prop - t_vec_prop[0]
v_errors_all = np.zeros((len(t_vec_prop), nr_sims+1))
v_errors_all[:,0] = t_vec_prop - t_vec_prop[0]
# for ii, (r_err, v_err) in enumerate(zip(r_error_ic, v_error_ic)):
for ii, s_err in enumerate(zip(r_error_ic, v_error_ic)):
    x_prop_input = x_ic_true + np.concatenate(s_err)
    s_host_predicted = j2prop.propagate_orbit(x_prop_input, t_0, t_end, dt, sp=use_single_precision)
    s_errors = s_host_predicted[:,1:] - s_true_interpolated
    r_errors = np.linalg.norm(s_errors[:,:3], axis = 1)
    v_errors = np.linalg.norm(s_errors[:,3:], axis = 1)
    
    r_errors_all[:,ii+1] = r_errors
    v_errors_all[:,ii+1] = v_errors


r_errors = out.make_n_save(f'r_error_j2_sp{use_single_precision}', t_vec=r_errors_all[:,0], data=r_errors_all[:,1:], subfolder = 'j2prop_testdata')
v_errors = out.make_n_save(f'v_error_j2_sp{use_single_precision}', t_vec=v_errors_all[:,0], data=v_errors_all[:,1:], subfolder = 'j2prop_testdata')

f, axs = plt.subplots(ncols = 2)
for ii in range(nr_sims):
    axs[0].plot(r_errors_all[:,0], r_errors_all[:,ii+1])
    axs[1].plot(v_errors_all[:,0], v_errors_all[:,ii+1])
axs[0].grid('on')
axs[1].grid('on')
axs[0].set_ylabel('delta r [m]')
axs[1].set_ylabel('delta v [m/s]')
axs[0].set_xlabel('t prop [s]')
axs[1].set_xlabel('t prop [s]')
title = f'initial dr={err_pos:.0f}m; dv={err_vel}m/s'
axs[0].set_title(title)
axs[1].set_title(f'nr. samples : {nr_sims}')
bplt.savefig(f, name='j2_prop_errors_longterm', subfolder='j2prop_testdata', tag_option=1)
plt.show()
# %%
