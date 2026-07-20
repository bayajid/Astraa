## Host pos extrapolation errors using quadrat extrapolator
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
csv_output_path = r'orbital_simulations\tudat_raw_states'
fname_simparam = 'simulation_parameters.json'
fname_states = 'state_history.dat'

## MVP imports
import basic_tools.vector_operations as vec_calc
import astronomy_tools.constants as const
import basic_tools.vector_operations as vec_op
import astronomy_tools.astro_targets as where_sun
import prediction_methods.interpolators as interp
import plotting_tools.basic_plotting as bplt
import basic_tools.time_conversion as t_conv
import basic_tools.in_out as out
import prediction_methods.error_generation as errgen
## LIST COMPREHENSION ft. if else
# [x+1 if x >= 45 else x+5 for x in l]

## Loading satellite orbital data
import tudat_tools.data_processing.data_processing_utilities as dputil
data_raw, simulation_parameters = dputil.load_constellation_data(full_path = csv_output_path)
t_j2000 = data_raw[:,0]
t_0 = t_j2000[0]

states_host = data_raw[:,1:7]
#%% input errors; 3-sig
dr_3sig = 10 # m
dv_3sig = 0.06 # m/s

link_dist = 250e3 # m

t_vec_full = np.arange(t_0, t_0+3.6e3+1, 1)
make_input_err_plot = 1
dt_extrap = 1 # s, extrapolation time

#%%

pos_errors_quad_all = np.zeros((t_vec_full.shape[0],3))
errors_r = errgen.pos_err_gen(err_mean = 0, err_std = dr_3sig/3, nrows = len(t_vec_full), ncols = 3)
errors_v = errgen.pos_err_gen(err_mean = 0, err_std = dv_3sig/3, nrows = len(t_vec_full), ncols = 3)
errors_r[1:,:] = errors_r[0,:]
errors_v[1:,:] = errors_v[0,:]

if make_input_err_plot:
    f, axs = plt.subplots(nrows = 2)
    axs[0].plot(np.linalg.norm(errors_r, axis =1 ))
    axs[0].set_ylabel('Pos error [m]')
    axs[1].plot(np.linalg.norm(errors_v, axis =1 ))
    axs[1].set_ylabel('vel error [m/s]')
    axs[1].set_xlabel('t [s]')
    f.suptitle('Generated input errors')

pos_interpolator = interp.we_interpolating()


states_full_true = states_host
states_full_actual = np.copy(states_full_true)
states_full_actual[:,:3] = states_full_actual[:,:3] + errors_r
states_full_actual[:,3:] = states_full_actual[:,3:] + errors_v

# total errors
extrap_errors_pos = np.zeros(errors_r.shape)
extrap_errors_pointing = np.zeros((errors_r.shape[0],1))
# added errors
extrap_errors_pos_added = np.zeros((errors_r.shape[0],1))
extrap_errors_pointing_added = np.zeros((errors_r.shape[0],1))
for ii_0, s_0 in enumerate(states_full_true[:-3,:]):
    s_0_known = states_full_actual[ii_0,:]
    s_1_known = states_full_actual[ii_0+1,:]
    s_2_known = states_full_actual[ii_0+2,:]
    s_known = np.vstack((s_0_known, s_1_known))
    r_known = s_known[:,:3]
    v_known = s_known[:,3:]
    s_2_true = states_full_true[ii_0+2,:]
    t_0 = t_vec_full[ii_0] # first t
    t_1 = t_vec_full[ii_0+1] # next t
    t_2 = float(t_vec_full[ii_0+2]) # for error calc; 1 s away
    
    pos_interpolator.get_quad_interpolant([t_0, t_1], r_known, v_known)
    r_interp = pos_interpolator.interpolate(t_2)    
    
    dr = r_interp - s_2_true[:3]
    ds_initial = s_2_known - s_2_true
    dr_initial = np.linalg.norm(ds_initial[:3])
    dr_tot = np.linalg.norm(dr)
    extrap_errors_pos_added[ii_0,:] = np.abs((dr_tot - dr_initial))
    extrap_errors_pointing_added[ii_0] = np.abs((dr_tot - dr_initial))/link_dist*1e6 # urad
        
    
    extrap_errors_pos[ii_0,:] = dr
    extrap_errors_pointing[ii_0] = np.linalg.norm(dr)/link_dist*1e6 # urad

f, axs = plt.subplots(nrows = 2)
f.suptitle('Quadratic Extrapolation Results; Total Error')
axs[0].plot(np.linalg.norm(extrap_errors_pos[:-3], axis = 1))
axs[0].set_ylabel('3D Pos error [m]')
axs[1].plot(np.linalg.norm(extrap_errors_pointing[:-3], axis = 1))
axs[1].set_ylabel(f'PE [urad] at R = {link_dist/1e3:.0f} km')
axs[1].set_xlabel('t [s]')
plt.legend()


f, axs = plt.subplots(nrows = 2)
f.suptitle('Quadratic Extrapolation Results; Added Error')
axs[0].plot(extrap_errors_pos_added[:-3])
axs[0].set_ylabel('3D Pos error added [m]')
axs[1].plot(extrap_errors_pointing[:-3])
axs[1].set_ylabel(f'PE added [urad] at R = {link_dist/1e3:.0f} km')
axs[1].set_xlabel('t [s]')
plt.legend()